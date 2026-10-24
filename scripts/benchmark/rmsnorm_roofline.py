"""Liger RMSNorm (llama casting) at the training shapes: its default launch vs its own block kernels with other row
tiles / warps / grids, cold-L2 timings and parity.

Liger runs the one-row kernels whenever the hidden size is > 256 (forward: one row per program; backward: one program
per SM walking ~n_rows/n_SMs rows in sequence) and the block kernels (BLOCK_ROW rows per program) only below that.
This launches both kernel families directly with chosen settings and reports, per variant: time (L2 flushed before
each call), achieved bandwidth (bytes read + written), and parity vs the default launch (Y, dX: max |diff| in bf16
ulps and fraction of elements that differ; dW: relative error vs an fp64 reference beside the default's).

  tools/run_py.sh scripts/benchmark/rmsnorm_roofline.py [--rows 32768] [--hidden 1536]
"""

from __future__ import annotations

import argparse
import sys

import torch
import triton
from liger_kernel.ops.rms_norm import (
    _block_rms_norm_backward_kernel,
    _block_rms_norm_forward_kernel,
    _rms_norm_backward_kernel,
    _rms_norm_forward_kernel,
    rms_norm_backward,
    rms_norm_forward,
)
from liger_kernel.ops.utils import calculate_settings, torch_to_triton_dtype

DEVICE = "cuda"
LLAMA = 0  # liger _CASTING_MODE_LLAMA


def timed(fn, flush: torch.Tensor, iters: int = 9) -> float:
    fn()
    out = []
    for _ in range(iters):
        flush.fill_(1.0)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    out.sort()
    return out[len(out) // 2]


def ulps(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float]:
    ia = a.contiguous().view(torch.int16).int()
    ib = b.contiguous().view(torch.int16).int()
    d = (ia - ib).abs()
    return int(d.max().item()), (d > 0).float().mean().item()


def fwd_block(X, W, eps, block_row, warps):
    n_rows, n_cols = X.shape
    BLOCK_SIZE, _ = calculate_settings(n_cols)
    Y = torch.empty_like(X)
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)
    _block_rms_norm_forward_kernel[(triton.cdiv(n_rows, block_row),)](
        Y, Y.stride(0), X, X.stride(0), W, W.stride(0), RSTD, RSTD.stride(0), n_rows, n_cols, eps, 0.0, LLAMA,
        elementwise_affine=True, BLOCK_SIZE=BLOCK_SIZE, num_warps=warps, BLOCK_ROW=block_row)
    return Y, RSTD


def bwd_block(dY, X, W, RSTD, block_row, warps, grid_mult):
    """Block backward. grid = n_SMs * grid_mult programs, each striding over row tiles (grid_mult=0: one tile each)."""
    n_rows, n_cols = dY.shape
    BLOCK_SIZE, _ = calculate_settings(n_cols)
    n_tiles = triton.cdiv(n_rows, block_row)
    sms = torch.cuda.get_device_properties(X.device).multi_processor_count
    grid = n_tiles if grid_mult == 0 else min(n_tiles, sms * grid_mult)
    dX = torch.empty_like(dY)
    dW_part = torch.empty((grid, n_cols), dtype=torch.float32, device=X.device)
    _block_rms_norm_backward_kernel[(grid,)](
        dY, dY.stride(0), dX, dX.stride(0), X, X.stride(0), torch_to_triton_dtype[X.dtype], W, W.stride(0),
        RSTD, RSTD.stride(0), dW_part, dW_part.stride(0), n_rows, n_cols, 0.0, LLAMA,
        elementwise_affine=True, BLOCK_SIZE=BLOCK_SIZE, num_warps=warps, BLOCK_ROW=block_row)
    return dX, dW_part.sum(0).to(W.dtype)


def run(rows: int, D: int, iters: int) -> None:
    g = torch.Generator(device=DEVICE).manual_seed(rows + D)
    X = torch.randn(rows, D, device=DEVICE, generator=g).bfloat16()
    dY = torch.randn(rows, D, device=DEVICE, generator=g).bfloat16()
    W = (1 + 0.1 * torch.randn(D, device=DEVICE, generator=g)).bfloat16()
    eps = 1e-5
    flush = torch.empty(512 * 1024 * 1024 // 4, device=DEVICE)
    gb_fwd = rows * D * 2 * 2 / 1e9            # read X, write Y
    gb_bwd = rows * D * 2 * 3 / 1e9            # read dY, X; write dX
    # fp64 dW reference
    Xd = X.double()
    rstd = torch.rsqrt(Xd.pow(2).mean(-1, keepdim=True) + eps)
    dW_ref = (dY.double() * (Xd * rstd)).sum(0)

    # default Liger launch
    Y0, _X, RSTD0, BS, NW, cm = rms_norm_forward(X, W, eps, 0.0, "llama", None)
    dX0, dW0 = rms_norm_backward(dY.clone(), X, W, RSTD0, 0.0, cm, BS, NW, True, None)
    e0 = ((dW0.double() - dW_ref).norm() / dW_ref.norm()).item()
    t = timed(lambda: rms_norm_forward(X, W, eps, 0.0, "llama", None), flush, iters)
    print(f"== rows={rows} D={D}  (BLOCK_SIZE {BS}, default warps {NW})", flush=True)
    print(f"  fwd default            {t * 1e3:8.1f} us  {gb_fwd / t:5.2f} TB/s", flush=True)
    t = timed(lambda: rms_norm_backward(dY.clone(), X, W, RSTD0, 0.0, cm, BS, NW, True, None), flush, iters)
    tc = timed(lambda: dY.clone(), flush, iters)
    print(f"  bwd default            {(t - tc) * 1e3:8.1f} us  {gb_bwd / (t - tc):5.2f} TB/s   dW err {e0:.2e}"
          f"  (clone {tc * 1e3:.0f} us subtracted)", flush=True)

    for br in (1, 2, 4, 8, 16):
        for w in (2, 4, 8, 16):
            if br * BS // (32 * w) < 1 or br * BS // (32 * w) > 256:
                continue
            try:
                Y, RSTD = fwd_block(X, W, eps, br, w)
                mu, fr = ulps(Y, Y0)
                t = timed(lambda: fwd_block(X, W, eps, br, w), flush, iters)
                print(f"  fwd block BR={br:2d} w={w:2d}   {t * 1e3:8.1f} us  {gb_fwd / t:5.2f} TB/s   Y vs default: "
                      f"max {mu} ulp, {fr:.1e} differ", flush=True)
                for gm in (0, 4, 16):
                    dX, dW = bwd_block(dY, X, W, RSTD0, br, w, gm)
                    mu, fr = ulps(dX, dX0)
                    e = ((dW.double() - dW_ref).norm() / dW_ref.norm()).item()
                    t = timed(lambda: bwd_block(dY, X, W, RSTD0, br, w, gm), flush, iters)
                    print(f"  bwd block BR={br:2d} w={w:2d} grid={'tiles' if gm == 0 else f'{gm}xSM':6s} "
                          f"{t * 1e3:8.1f} us  {gb_bwd / t:5.2f} TB/s   dX vs default: max {mu} ulp, {fr:.1e} differ;"
                          f" dW err {e:.2e}", flush=True)
            except Exception as exc:  # an oversized tile can fail to compile; report and go on
                print(f"  BR={br} w={w}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32768)
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--qk_rows", type=int, default=32768 * 12)
    ap.add_argument("--head", type=int, default=128)
    ap.add_argument("--iters", type=int, default=9)
    a = ap.parse_args()
    run(a.rows, a.hidden, a.iters)
    run(a.qk_rows, a.head, a.iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
