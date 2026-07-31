"""Register/spill report for AttnRes routing kernels (7B throughput program).

Runs each backward kernel once at training shape (populating the autotune
cache), then introspects the compiled binaries for n_regs / n_spills /
shared-mem per config. This is the counter ncu would give us, without needing
admin perf-counter access — Triton records it at compile time.

Run on gpu-host: TRITON_PRINT_AUTOTUNING=1 tools/run_py.sh scripts/utils/kernel_spill_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.flash_attn_res.kernels import phase_1 as p1k  # noqa: E402
from src.model.flash_attn_res.kernels import phase_2 as p2k  # noqa: E402
from src.model.flash_attn_res.ops import phase_1 as p1ops  # noqa: E402
from src.model.flash_attn_res.ops import phase_2 as p2ops  # noqa: E402

DEVICE = "cuda"
D = 4096
EPS = 1e-5


def warm_phase2():
    b, t = 4, 4096
    g = torch.Generator(device=DEVICE).manual_seed(0)
    ps = torch.randn(b, t, D, device=DEVICE, generator=g).to(torch.bfloat16)
    qw = torch.randn(D, device=DEVICE, generator=g) * 0.1
    pn = torch.randn(b, t, D, device=DEVICE, generator=g).to(torch.bfloat16)
    lse = torch.randn(b, t, device=DEVICE, generator=g)
    gm = torch.randn(b, t, D, device=DEVICE, generator=g).to(torch.bfloat16)
    _m, logit, irms = p2ops._phase_2_online_softmax_merge_forward_with_aux_triton_op(
        ps, qw, pn, lse, EPS)
    p2ops._online_softmax_merge_backward_triton_op(
        ps, qw, pn, lse, logit, irms, gm, EPS)
    p2ops._online_softmax_merge_backward_v2_triton_op(
        ps, qw, pn, lse, logit, irms, gm, EPS)


def warm_phase1(s_n: int):
    b, t, nq = 4, 4096, 9
    g = torch.Generator(device=DEVICE).manual_seed(0)
    src = torch.randn(s_n, b, t, D, device=DEVICE, generator=g).to(torch.bfloat16)
    q = torch.randn(nq, D, device=DEVICE, generator=g) * 0.1
    go = torch.randn(nq, b, t, D, device=DEVICE, generator=g).to(torch.bfloat16)
    gl = torch.randn(nq, b, t, device=DEVICE, generator=g) * 0.01
    _o, lses, irms, logits = p1ops._phase_1_batched_attention_forward_with_aux_triton_op(
        src, q, EPS, s_n)
    p1ops._batched_attention_backward_triton_op(
        src, q, lses, irms, logits, go, gl, True, EPS, s_n)
    p1ops._batched_attention_backward_v2_triton_op(
        src, q, lses, irms, logits, go, gl, True, EPS, s_n)


def _find_compiled(obj, depth: int = 0, seen_ids=None):
    """Recursively walk dicts/tuples/lists on a JITFunction hunting for
    compiled-kernel objects (anything exposing n_regs). Triton renamed the
    cache attribute across 3.x versions; this survives all of them."""
    if seen_ids is None:
        seen_ids = set()
    if id(obj) in seen_ids or depth > 4:
        return
    seen_ids.add(id(obj))
    if hasattr(obj, "n_regs"):
        yield obj
        return
    values = ()
    if isinstance(obj, dict):
        values = obj.values()
    elif isinstance(obj, (list, tuple)):
        values = obj
    for v in values:
        yield from _find_compiled(v, depth + 1, seen_ids)


def report(kernel, name: str) -> None:
    print(f"\n=== {name} ===")
    jit_fn = getattr(kernel, "fn", kernel)
    found = []
    for attr in ("device_caches", "cache", "_device_caches"):
        holder = getattr(jit_fn, attr, None)
        if holder is not None:
            found.extend(_find_compiled(holder))
        if found:
            break
    if not found:
        found.extend(_find_compiled(jit_fn.__dict__))
    if not found:
        print("  (no compiled binaries located on JITFunction)")
        return
    for bin_ in found:
        n_regs = getattr(bin_, "n_regs", None)
        n_spills = getattr(bin_, "n_spills", None)
        meta = getattr(bin_, "metadata", None)
        nw = getattr(meta, "num_warps", "?") if meta else "?"
        ns = getattr(meta, "num_stages", "?") if meta else "?"
        shared = getattr(meta, "shared", None) if meta else None
        flag = "  <-- SPILLING" if (n_spills or 0) > 0 else ""
        print(f"  regs={n_regs:4} spills={n_spills:5} warps={nw} stages={ns} "
              f"shared={shared}{flag}")


def main() -> int:
    torch.cuda.init()
    print("warming phase-2 kernels (autotune)...")
    warm_phase2()
    print("warming phase-1 kernels S=3 and S=7 (autotune)...")
    warm_phase1(3)
    warm_phase1(7)

    report(p2k.phase_2_online_softmax_merge_backward_kernel, "phase-2 bwd v1")
    report(p2k.phase_2_online_softmax_merge_backward_v2_kernel, "phase-2 bwd v2 (chunked)")
    report(p1k.phase_1_batched_attention_backward_kernel, "phase-1 bwd v1")
    report(p1k.phase_1_batched_attention_backward_v2_kernel, "phase-1 bwd v2 (chunked draft)")
    report(p1k.phase_1_batched_attention_forward_kernel, "phase-1 fwd (for reference)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
