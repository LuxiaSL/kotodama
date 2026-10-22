"""AttnRes route kernels (Phase 1 + Phase 2): Triton-runtime launch (eager op) vs inductor-compiled launch (op inside torch.compile), in isolation.

Under whole-model compile inductor compiles user Triton kernels itself (triton_heuristics.user_autotune) instead of
the kernel's own runtime autotuner. This runs the same op at the same shape both ways under torch.profiler and
reports, per kernel name and path, the per-call device time and the launch metadata the profiler records (threads,
registers per thread, shared memory) — a binary or launch difference shows up there.

Run on a GPU host: tools/run_py.sh scripts/benchmark/attnres_p1_compiled.py [--depths 4,8] [--mb 4]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model.flash_attn_res.ops import phase_1 as p1ops  # noqa: E402
from kotodama.model.flash_attn_res.ops import phase_2 as p2ops  # noqa: E402

DEVICE = "cuda"
EPS = 1e-6


def kernel_stats(prof: profile, prefix: str) -> dict[str, list[tuple[float, str]]]:
    out: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.name.startswith(prefix):
            out[ev.name].append((ev.device_time, ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4,8")
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    D = a.hidden
    g = torch.Generator(device=DEVICE).manual_seed(0)
    for S in (int(x) for x in a.depths.split(",")):
        src0 = torch.randn(S, a.mb, a.seq, D, device=DEVICE, generator=g).bfloat16()
        q0 = (torch.randn(1, D, device=DEVICE, generator=g) * 0.5).bfloat16()
        gout = torch.randn(a.mb, a.seq, D, device=DEVICE, generator=g).bfloat16()

        def step(fn):
            src = src0.clone().requires_grad_(True)
            q = q0.clone().requires_grad_(True)
            out = fn(src, q)
            torch.autograd.grad(out, (src, q), grad_outputs=gout)

        partial0 = torch.randn(a.mb, a.seq, D, device=DEVICE, generator=g).bfloat16()

        def eager(src, q):
            # the per-sublayer route: Phase 1 over the stack, then the Phase-2 merge with the partial
            p1_out, p1_lse = p1ops.phase_1_batched_attention_triton_op(src, q, EPS, num_active=S)
            return p2ops.phase_2_online_softmax_merge_triton_op(partial0, q[0], p1_out[0], p1_lse[0], EPS)

        compiled = torch.compile(eager, fullgraph=True)
        with torch.no_grad():
            out_e, out_c = eager(src0, q0), compiled(src0, q0)
        print(f"S={S} mb={a.mb} route output eager vs compiled: bitwise {torch.equal(out_e, out_c)}, "
              f"max|Δ| {(out_e.float() - out_c.float()).abs().max().item():.2e}", flush=True)
        for name, fn in (("eager", eager), ("compiled", compiled)):
            for _ in range(3):
                step(fn)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(a.iters):
                    step(fn)
                torch.cuda.synchronize()
            for kname, durs in sorted(kernel_stats(prof, "phase_").items()):
                ds = sorted(d for d, _ in durs)
                print(f"S={S} mb={a.mb} {name:8s} {kname[:60]:60s} n={len(ds):3d} p50 {ds[len(ds) // 2]:8.1f} us "
                      f"min {ds[0]:8.1f}", flush=True)
            trace = f"/dev/shm/attnres_p1_{name}_S{S}.json"
            prof.export_chrome_trace(trace)
            print(f"  trace {trace}", flush=True)
        del src0, gout
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
