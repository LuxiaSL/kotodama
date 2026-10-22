"""Phase-1 autotune diagnosis: every config timed alone, and what the autotuner picks.

A training profile can show the batched Phase-1 kernels running far slower per call than a micro-bench at the
same shape, with a different launch config (threads, registers). This times each forward and backward (v1) config
on its own at the given stack depths, then reports the autotuner's pick (a) when its first call is the full
training shape and (b) when its first call is a tiny token count — the key does not include the token count, so
the first call's shape decides the config for every later call with the same key.

Run on a GPU host: tools/run_py.sh scripts/benchmark/attnres_p1_configs.py [--depths 4,8] [--mb 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model.flash_attn_res.kernels import configs as kcfg  # noqa: E402
from kotodama.model.flash_attn_res.kernels import phase_1 as p1k  # noqa: E402
from kotodama.model.flash_attn_res.ops import phase_1 as p1ops  # noqa: E402

DEVICE = "cuda"
EPS = 1e-6


def timed(fn, iters: int = 10) -> float:
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def cfg_str(c) -> str:
    return f"warps={c.num_warps} stages={c.num_stages} {c.kwargs or ''}".strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", default="4,8")
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=1536)
    a = ap.parse_args()
    D = a.hidden
    fwd_k, bwd_k = p1k.phase_1_batched_attention_forward_kernel, p1k.phase_1_batched_attention_backward_kernel
    g = torch.Generator(device=DEVICE).manual_seed(0)
    for S in (int(x) for x in a.depths.split(",")):
        src = torch.randn(S, a.mb, a.seq, D, device=DEVICE, generator=g).bfloat16()
        q = (torch.randn(1, D, device=DEVICE, generator=g) * 0.5).bfloat16()
        gout = torch.randn(1, a.mb, a.seq, D, device=DEVICE, generator=g).bfloat16()

        def fwd(s=src):
            return p1ops.phase_1_batched_attention_triton_op(s, q, EPS, num_active=S)

        srcg = src.clone().requires_grad_(True)
        qg = q.clone().requires_grad_(True)

        def fwd_bwd():
            out, _ = p1ops.phase_1_batched_attention_triton_op(srcg, qg, EPS, num_active=S)
            return torch.autograd.grad(out, (srcg, qg), grad_outputs=gout)

        print(f"== S={S} (B={a.mb} T={a.seq} D={D})", flush=True)
        for c in kcfg.forward_configs:
            kcfg.set_autotune_configs(fwd_k, [c])
            print(f"  fwd  {cfg_str(c):28s} {timed(fwd):7.3f} ms", flush=True)
        kcfg.set_autotune_configs(fwd_k, kcfg.forward_configs)
        t_fwd = timed(fwd)
        for c in kcfg.phase1_backward_configs:
            kcfg.set_autotune_configs(bwd_k, [c])
            print(f"  bwd  {cfg_str(c):28s} {timed(fwd_bwd) - t_fwd:7.3f} ms (incl. reduce + cast)", flush=True)
        kcfg.set_autotune_configs(bwd_k, kcfg.phase1_backward_configs)
        fwd_bwd()
        print(f"  autotune, first call at full shape: fwd {cfg_str(fwd_k.best_config)}; "
              f"bwd {cfg_str(bwd_k.best_config)}", flush=True)
        kcfg.set_autotune_configs(fwd_k, kcfg.forward_configs)
        kcfg.set_autotune_configs(bwd_k, kcfg.phase1_backward_configs)
        tiny_s = src[:, :1, :16].contiguous().requires_grad_(True)
        tiny_q = q.clone().requires_grad_(True)
        out, _ = p1ops.phase_1_batched_attention_triton_op(tiny_s, tiny_q, EPS, num_active=S)
        torch.autograd.grad(out, (tiny_s, tiny_q), grad_outputs=torch.randn_like(out))
        fwd_cfg, bwd_cfg = cfg_str(fwd_k.best_config), cfg_str(bwd_k.best_config)
        print(f"  autotune, first call at 16 tokens: fwd {fwd_cfg}; bwd {bwd_cfg}; then full shape: "
              f"fwd {timed(fwd):.3f} ms, bwd {timed(fwd_bwd) - timed(fwd):.3f} ms", flush=True)
        kcfg.set_autotune_configs(fwd_k, kcfg.forward_configs)
        kcfg.set_autotune_configs(bwd_k, kcfg.phase1_backward_configs)
        del src, gout, srcg
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
