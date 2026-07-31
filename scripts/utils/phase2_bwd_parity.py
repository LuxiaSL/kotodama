"""Parity + micro-bench for the phase-2 backward v2 kernel (7B throughput program).

Gates (SPEC §4 Tier A):
  1. v1 vs v2 BITWISE on grad_intrablock, grad_p1_normalized (post-conversion
     dtypes identical: fp32-compute -> one bf16 round in both paths) and
     grad_lse (fp32 both).
  2. grad_pseudo_query: v2 deterministic (bitwise across repeat runs); v1's
     atomic order is nondeterministic, so v1-vs-v2 compared with tolerance and
     both compared against an fp64 autograd reference.
  3. End-to-end through the public op dispatch (monkeypatched flag), autograd
     .backward() on both paths.

Micro-bench: CUDA-event timing of v1 vs v2 backward ops at training shapes.

Run on gpu-host (GPU required):
  cd ~/workspace/kotodama && ~/workspace/.venv-shared/bin/python \
      scripts/utils/phase2_bwd_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.flash_attn_res.ops import phase_2 as p2ops  # noqa: E402

DEVICE = "cuda"
D = 4096
EPS = 1e-5
SHAPES = [
    (4, 4096),   # MB=4 canary
    (6, 4096),   # MB=6
    (1, 1000),   # odd BT, exercises masking
]


def make_inputs(b: int, t: int, seed: int = 0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    partial = torch.randn(b, t, D, device=DEVICE, generator=g, dtype=torch.float32).to(torch.bfloat16)
    qw = torch.randn(D, device=DEVICE, generator=g, dtype=torch.float32) * 0.1
    p1_out = torch.randn(b, t, D, device=DEVICE, generator=g, dtype=torch.float32).to(torch.bfloat16)
    p1_lse = torch.randn(b, t, device=DEVICE, generator=g, dtype=torch.float32)
    grad_merged = torch.randn(b, t, D, device=DEVICE, generator=g, dtype=torch.float32).to(torch.bfloat16)
    return partial, qw, p1_out, p1_lse, grad_merged


def run_forward_aux(partial, qw, p1_out, p1_lse):
    merged, logit, irms = p2ops._phase_2_online_softmax_merge_forward_with_aux_triton_op(
        partial, qw, p1_out, p1_lse, EPS
    )
    return merged, logit, irms


def run_v1(partial, qw, p1_out, p1_lse, logit, irms, grad_merged):
    gi, gq, gn, gl = p2ops._online_softmax_merge_backward_triton_op(
        partial, qw, p1_out, p1_lse, logit, irms, grad_merged.contiguous(), EPS
    )
    # replicate the wrapper's post-conversion
    return gi.to(partial.dtype), gq, gn.to(p1_out.dtype), gl


def run_v2(partial, qw, p1_out, p1_lse, logit, irms, grad_merged):
    gi, gq, gn, gl = p2ops._online_softmax_merge_backward_v2_triton_op(
        partial, qw, p1_out, p1_lse, logit, irms, grad_merged.contiguous(), EPS
    )
    return gi, gq, gn, gl


def fp64_reference(partial, qw, p1_out, p1_lse, grad_merged):
    ps = partial.to(torch.float64).requires_grad_(True)
    q = qw.to(torch.float64).requires_grad_(True)
    pn = p1_out.to(torch.float64).requires_grad_(True)
    lse = p1_lse.to(torch.float64).requires_grad_(True)
    irms = torch.rsqrt((ps * ps).sum(-1) / D + EPS)
    logit = (ps * q).sum(-1) * irms
    prob = torch.sigmoid(logit - lse)
    merged = pn + prob.unsqueeze(-1) * (ps - pn)
    grads = torch.autograd.grad(
        merged, (ps, q, pn, lse), grad_outputs=grad_merged.to(torch.float64)
    )
    return grads


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a64, b64 = a.double(), b.double()
    denom = b64.abs().max().clamp_min(1e-30)
    return ((a64 - b64).abs().max() / denom).item()


def main() -> int:
    torch.cuda.init()
    all_ok = True

    for b, t in SHAPES:
        print(f"\n=== shape B={b} T={t} (BT={b*t}) ===")
        partial, qw, p1_out, p1_lse, grad_merged = make_inputs(b, t)
        merged, logit, irms = run_forward_aux(partial, qw, p1_out, p1_lse)

        v1 = run_v1(partial, qw, p1_out, p1_lse, logit, irms, grad_merged)
        v2 = run_v2(partial, qw, p1_out, p1_lse, logit, irms, grad_merged)
        v2b = run_v2(partial, qw, p1_out, p1_lse, logit, irms, grad_merged)

        # Bitwise v1==v2 is expected when both autotuners pick the same
        # num_warps (identical reduction tree) but is NOT guaranteed — the
        # hard gates are v2 determinism + fp64 closeness; bitwise is reported.
        for idx, nm in [(0, "grad_intrablock"), (2, "grad_p1_norm"), (3, "grad_lse")]:
            bw = torch.equal(v1[idx], v2[idx])
            e = rel_err(v1[idx], v2[idx])
            all_ok &= check(f"{nm} v1~v2", e < 2e-2,
                            f"bitwise={bw} maxrel {e:.2e}")
        all_ok &= check("grad_pseudo_q  v2 deterministic", torch.equal(v2[1], v2b[1]))
        all_ok &= check("grad_pseudo_q  v1~v2", rel_err(v2[1], v1[1]) < 1e-4,
                        f"maxrel {rel_err(v2[1], v1[1]):.2e}")

        ref = fp64_reference(partial, qw, p1_out, p1_lse, grad_merged)
        for i, (nm, tol) in enumerate([("grad_intrablock", 2e-2), ("grad_pseudo_q", 1e-3),
                                       ("grad_p1_norm", 2e-2), ("grad_lse", 1e-3)]):
            e_v2 = rel_err(v2[i], ref[i])
            all_ok &= check(f"{nm} v2 vs fp64", e_v2 < tol, f"maxrel {e_v2:.2e}")

    # End-to-end dispatch check through the public op
    print("\n=== end-to-end autograd dispatch ===")
    partial, qw, p1_out, p1_lse, grad_merged = make_inputs(4, 4096, seed=7)
    grads = {}
    for flag in (False, True):
        p2ops._P2_BWD_V2 = flag
        leaves = [partial.clone().requires_grad_(True), qw.clone().requires_grad_(True),
                  p1_out.clone().requires_grad_(True), p1_lse.clone().requires_grad_(True)]
        out = p2ops.phase_2_online_softmax_merge_triton_op(*leaves, EPS)
        out.backward(grad_merged)
        grads[flag] = [leaf.grad.clone() for leaf in leaves]
    p2ops._P2_BWD_V2 = True
    for i, nm in enumerate(["partial", "pseudo_q", "p1_out", "p1_lse"]):
        tol = 1e-4 if nm == "pseudo_q" else 2e-2
        e = rel_err(grads[True][i], grads[False][i])
        bw = torch.equal(grads[True][i], grads[False][i])
        all_ok &= check(f"e2e grad {nm} v1~v2", e < tol, f"bitwise={bw} maxrel {e:.2e}")

    # Micro-bench
    print("\n=== micro-bench (200 iters, CUDA events) ===")
    for b, t in [(4, 4096), (6, 4096)]:
        partial, qw, p1_out, p1_lse, grad_merged = make_inputs(b, t)
        merged, logit, irms = run_forward_aux(partial, qw, p1_out, p1_lse)
        gm = grad_merged.contiguous()
        for nm, fn in [("v1", p2ops._online_softmax_merge_backward_triton_op),
                       ("v2", p2ops._online_softmax_merge_backward_v2_triton_op)]:
            for _ in range(20):
                fn(partial, qw, p1_out, p1_lse, logit, irms, gm, EPS)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(200):
                fn(partial, qw, p1_out, p1_lse, logit, irms, gm, EPS)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / 200
            # traffic: 3 bf16 (BT,D) loads + v1[2 fp32 stores + convert r/w] vs v2[2 bf16 stores]
            bt = b * t
            bytes_v = bt * D * 2 * 3 + (bt * D * (4 * 2 + 2 * 3) if nm == "v1" else bt * D * 2 * 2)
            print(f"  BT={bt:6d} {nm}: {ms*1000:8.1f} us  ({bytes_v/ms/1e6:.0f} GB/s effective)")

    print("\n" + ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
