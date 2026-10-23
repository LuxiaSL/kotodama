"""Cold-cache roofline for the single-query Phase-1 backward (the per-sublayer route): bytes moved / time, per kernel
and per autotune config, against torch's own elementwise kernels at the same shape.

Every timed call is preceded by an L2 flush (a write over a buffer larger than L2), because in training the
committed stack is cold when its route's backward runs — a backward timed right after its forward reads part of
the stack from L2 and looks faster than it is.

  tools/run_py.sh scripts/benchmark/attnres_bwd_roofline.py [--depths 6,8] [--mb 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model.flash_attn_res.kernels import phase_1 as k1  # noqa: E402
from kotodama.model.flash_attn_res.ops import phase_1 as p1ops  # noqa: E402

DEVICE = "cuda"
EPS = torch.finfo(torch.float32).eps


def timed(fn, flush: torch.Tensor, iters: int, cold: bool = True) -> float:
    """Median ms of fn() over iters calls, each after an L2 flush (cold) or back-to-back (hot)."""
    out = []
    fn()
    for _ in range(iters):
        if cold:
            flush.fill_(1.0)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    out.sort()
    return out[len(out) // 2]


def each_config(kernel, run, flush, iters) -> list[tuple[str, float]]:
    """Time `run` once per autotune config of `kernel` (one config at a time), cold."""
    saved = list(kernel.configs)
    res = []
    try:
        for cfg in saved:
            kernel.configs = [cfg]
            kernel.cache.clear()
            try:
                res.append((str(cfg), timed(run, flush, iters)))
            except Exception as exc:  # an over-large tile can fail to compile; report and go on
                res.append((str(cfg), float("nan")))
                print(f"    config {cfg} failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    finally:
        kernel.configs = saved
        kernel.cache.clear()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="6,8")
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--skip_configs", action="store_true", help="skip the per-autotune-config sweeps")
    a = ap.parse_args()
    D, BT = a.hidden, a.mb * a.seq
    flush = torch.empty(512 * 1024 * 1024 // 4, device=DEVICE, dtype=torch.float32)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    for S in (int(x) for x in a.depths.split(",")):
        src = torch.randn(S, a.mb, a.seq, D, device=DEVICE, generator=g).bfloat16()
        q = (torch.randn(1, D, device=DEVICE, generator=g) * 0.5).bfloat16()
        gout = torch.randn(1, a.mb, a.seq, D, device=DEVICE, generator=g).bfloat16()
        glse = torch.randn(1, a.mb, a.seq, device=DEVICE, generator=g) * 0.1
        prev = torch.randn_like(src)
        _, lse, irms, logits = p1ops._phase_1_batched_attention_forward_with_aux_triton_op(src, q, EPS, S, False)
        stack_gb = src.numel() * 2 / 1e9
        row_gb = BT * D * 2 / 1e9
        print(f"== S={S} mb={a.mb}: stack {stack_gb:.3f} GB", flush=True)

        def report(name: str, ms: float, gb: float) -> None:
            print(f"  {name:58s} {ms * 1e3:8.1f} us  {gb:6.3f} GB  {gb / ms:6.2f} TB/s", flush=True)

        # torch roofline at the stack shape
        out = torch.empty_like(src)
        report("torch add (stack + stack -> stack) cold", timed(lambda: torch.add(src, prev, out=out), flush, a.iters),
               3 * stack_gb)
        report("torch copy (stack -> stack) cold", timed(lambda: out.copy_(src), flush, a.iters), 2 * stack_gb)

        # v1 (default) backward: reads stack + grad_out, writes stack grad (bf16) + per-token fp32 query partials
        def v1():
            p1ops._batched_attention_backward_triton_op(src, q, lse, irms, logits, gout, glse, True, EPS, S)

        v1_gb = 2 * stack_gb + row_gb + BT * D * 4 * 2 / 1e9
        report("v1 bwd (+ query reduce) cold", timed(v1, flush, a.iters), v1_gb)
        report("v1 bwd (+ query reduce) hot (no flush)", timed(v1, flush, a.iters, cold=False), v1_gb)
        if not a.skip_configs:
            for cfg, ms in each_config(k1.phase_1_batched_attention_backward_kernel, v1, flush, a.iters):
                report(f"v1 {cfg[:52]}", ms, v1_gb)
        # v1 + the autograd add the chain replaces (unchained route = this pair)
        def v1_plus_add():
            v1()
            torch.add(src, prev, out=out)
        report("v1 bwd + stack add cold (= the unchained route)", timed(v1_plus_add, flush, a.iters),
               v1_gb + 3 * stack_gb)

        # chained single-query backward (prev read): reads stack + prev + grad_out, writes stack grad (the
        # --attnres_stack_grad_chain route; compare with "v1 bwd + stack add", the unchained route it replaces)
        chain_gb = 3 * stack_gb + row_gb

        def chained():
            p1ops._batched_attention_backward_chained_triton_op(src, q, lse, irms, logits, gout, glse, True, EPS, S,
                                                                prev)

        report("chained cold", timed(chained, flush, a.iters), chain_gb)
        report("chained hot (no flush)", timed(chained, flush, a.iters, cold=False), chain_gb)
        if not a.skip_configs:
            for cfg, ms in each_config(k1.phase_1_route_chained_backward_kernel, chained, flush, a.iters):
                report(f"chained {cfg[:50]}", ms, chain_gb)
        del src, prev, out, gout
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
