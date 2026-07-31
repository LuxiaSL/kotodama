#!/usr/bin/env python3
"""Standalone Triton GEMV microbench for the 5 decode shapes.

Sweeps tile configs for a row-parallel GEMV (y = W @ x, W row-major (N, K)
bf16, fp32 accumulate, bf16 out) and times each against F.linear on identical
tensors. Timing runs inside a captured CUDA graph — both because launch gaps
would otherwise dominate 3-20 us kernels and because graph capture is exactly
how the kernel runs in production (cudagraph trees).

The integration bar is NOT F.linear eager: it is the CURRENT compiled build's
fused per-kernel times (results/profiles/engine_round2_baseline.json):
    qkv 19.5us (incl. fused norm) | o 7.1-31us | gate_up 15.6 | down 8.2 | lm_head 47.7

Usage (gpu-host):
    CUDA_VISIBLE_DEVICES=3 python scripts/benchmark/gemv_microbench.py --output /tmp/gemv_sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Optional

import torch
import triton
import triton.language as tl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# (N, K) per decode GEMV site; calls/step; current fused-kernel us/call from
# the round-2 baseline profile (the bar custom kernels must beat AFTER adding
# back the cost of any epilogue/prologue that un-fuses).
SHAPES: dict[str, dict[str, Any]] = {
    "qkv":     {"N": 5120,  "K": 3072, "calls": 28, "current_us": 19.5},
    "o":       {"N": 3072,  "K": 3072, "calls": 28, "current_us": 9.5},   # mix of 7.1/5.9/31
    "gate_up": {"N": 16384, "K": 3072, "calls": 28, "current_us": 15.6},
    "down":    {"N": 3072,  "K": 8192, "calls": 28, "current_us": 8.2},
    "lm_head": {"N": 49152, "K": 3072, "calls": 1,  "current_us": 47.7},
}

HBM_PEAK_GBS = 8000.0  # B200 HBM3e nominal


@triton.jit
def _gemv_kernel(
    w_ptr, x_ptr, y_ptr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y[n] = sum_k W[n, k] * x[k], fp32 accumulate. No masks: N % BLOCK_N == 0
    and K % BLOCK_K == 0 are asserted host-side (true for all decode shapes)."""
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    w_base = w_ptr + rn[:, None] * K
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        xv = tl.load(x_ptr + k0 + rk).to(tl.float32)
        wv = tl.load(w_base + k0 + rk[None, :]).to(tl.float32)
        acc += tl.sum(wv * xv[None, :], axis=1)
    tl.store(y_ptr + rn, acc.to(tl.bfloat16))


def gemv(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
         block_n: int, block_k: int, num_warps: int, num_stages: int) -> None:
    n, k = w.shape
    grid = (n // block_n,)
    _gemv_kernel[grid](
        w, x, y, K=k, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )


def _time_graphed(fn, n_inner: int = 32, n_replay: int = 20) -> float:
    """us per call, measured by replaying a captured graph of n_inner calls."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_inner):
            fn()
    g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_replay):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / (n_inner * n_replay)


def sweep_shape(name: str, spec: dict[str, Any], device: torch.device,
                quick: bool = False) -> dict[str, Any]:
    n, k = spec["N"], spec["K"]
    gen = torch.Generator(device="cpu").manual_seed(1234)
    w = (torch.randn((n, k), generator=gen) * 0.02).to(device=device, dtype=torch.bfloat16)
    x = (torch.randn((k,), generator=gen) * 0.5).to(device=device, dtype=torch.bfloat16)
    y = torch.empty((n,), device=device, dtype=torch.bfloat16)

    # References: bf16 F.linear (the eager dispatch) + fp64 ground truth.
    y_ref = torch.nn.functional.linear(x.unsqueeze(0), w).squeeze(0)
    y_true = (w.double() @ x.double())
    ref_err = float((y_ref.double() - y_true).abs().max())

    bytes_moved = (n * k + k + n) * 2  # bf16

    if quick:
        configs = [(8, 512, 4, 3), (16, 512, 4, 3), (8, 1024, 8, 3), (16, 1024, 8, 4)]
    else:
        configs = [
            (bn, bk, nw, ns)
            for bn in (4, 8, 16, 32)
            for bk in (256, 512, 1024)
            for nw in (4, 8)
            for ns in (2, 3, 4)
            if bn * bk * 2 * ns <= 200 * 1024  # rough smem ceiling incl. pipelining
        ]

    rows: list[dict[str, Any]] = []
    for bn, bk, nw, ns in configs:
        if n % bn or k % bk:
            continue
        try:
            gemv(w, x, y, bn, bk, nw, ns)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 — OOR smem etc.: skip config
            logger.debug("%s cfg (%d,%d,w%d,s%d) failed: %s", name, bn, bk, nw, ns, e)
            continue
        tri_err = float((y.double() - y_true).abs().max())
        # Same rounding class as cuBLAS: allow 4x its fp64 error + bf16 crumbs.
        if tri_err > ref_err * 4 + 1e-2:
            logger.warning("%s cfg (%d,%d,w%d,s%d): NUMERICS err %.4g vs ref %.4g — skipped",
                           name, bn, bk, nw, ns, tri_err, ref_err)
            continue
        us = _time_graphed(lambda: gemv(w, x, y, bn, bk, nw, ns))
        rows.append({
            "config": {"block_n": bn, "block_k": bk, "num_warps": nw, "num_stages": ns},
            "us": round(us, 2),
            "gbs": round(bytes_moved / us / 1e3, 0),
            "bw_pct": round(100 * bytes_moved / us / 1e3 / HBM_PEAK_GBS, 1),
            "max_err": tri_err,
        })

    ref_us = _time_graphed(lambda: torch.nn.functional.linear(x.unsqueeze(0), w))
    rows.sort(key=lambda r: r["us"])
    best = rows[0] if rows else None
    result = {
        "shape": {"N": n, "K": k},
        "bytes_mb": round(bytes_moved / 1e6, 1),
        "roofline_us": round(bytes_moved / HBM_PEAK_GBS / 1e3, 2),
        "f_linear_us": round(ref_us, 2),
        "current_fused_us": spec["current_us"],
        "ref_max_err_vs_fp64": ref_err,
        "best": best,
        "top5": rows[:5],
    }
    if best:
        logger.info(
            "%-8s best %5.2fus (%4.0f GB/s, %4.1f%% BW) cfg=%s | F.linear %5.2fus | current fused %5.2fus | roofline %5.2fus",
            name, best["us"], best["gbs"], best["bw_pct"],
            tuple(best["config"].values()), ref_us, spec["current_us"], result["roofline_us"],
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=list(SHAPES.keys()))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.cuda.init()
    logger.info("Device: %s", torch.cuda.get_device_name(device))

    results: dict[str, Any] = {}
    for name in args.shapes:
        if name not in SHAPES:
            logger.error("Unknown shape %s (have %s)", name, list(SHAPES))
            continue
        results[name] = sweep_shape(name, SHAPES[name], device, quick=args.quick)

    per_step_now = sum(SHAPES[s]["current_us"] * SHAPES[s]["calls"] for s in results)
    per_step_best = sum(
        (results[s]["best"]["us"] if results[s]["best"] else SHAPES[s]["current_us"])
        * SHAPES[s]["calls"]
        for s in results
    )
    summary = {
        "per_step_gemv_us_current": round(per_step_now, 1),
        "per_step_gemv_us_best": round(per_step_best, 1),
        "note": "best ignores un-fused prologue/epilogue costs — integration decides per shape",
    }
    results["_summary"] = summary
    logger.info("Per-step GEMV: current(fused) %.0fus -> best-case custom %.0fus",
                per_step_now, per_step_best)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Written: %s", args.output)


if __name__ == "__main__":
    main()
