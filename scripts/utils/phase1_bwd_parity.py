"""Parity + micro-bench for the phase-1 backward v2 kernel (7B throughput program).

Same gate structure as phase2_bwd_parity.py: v1~v2 tolerance (bitwise reported),
v2 determinism, fp64 autograd reference, end-to-end dispatch check, CUDA-event
micro-bench at training shapes (SRC up to 7 committed + padding, NQ=9 queries
per block at DD-boundaries).

Run on gpu-host: tools/run_py.sh scripts/utils/phase1_bwd_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model.flash_attn_res.ops import phase_1 as p1ops  # noqa: E402

DEVICE = "cuda"
D = 4096
EPS = 1e-5
NQ = 9
# (num_src_active, B, T) — production: up to 7 committed blocks, MB=4, T=4096
SHAPES = [
    (3, 4, 4096),
    (7, 4, 4096),
    (2, 1, 1000),  # odd BT + small src count, exercises masking
]


def make_inputs(s: int, b: int, t: int, seed: int = 0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    src = torch.randn(s, b, t, D, device=DEVICE, generator=g, dtype=torch.float32).to(torch.bfloat16)
    queries = torch.randn(NQ, D, device=DEVICE, generator=g, dtype=torch.float32) * 0.1
    grad_out = torch.randn(NQ, b, t, D, device=DEVICE, generator=g, dtype=torch.float32).to(torch.bfloat16)
    grad_lse = torch.randn(NQ, b, t, device=DEVICE, generator=g, dtype=torch.float32) * 0.01
    return src, queries, grad_out, grad_lse


def fp64_reference(src, queries, grad_out, grad_lse):
    s = src.to(torch.float64).requires_grad_(True)
    q = queries.to(torch.float64).requires_grad_(True)
    irms = torch.rsqrt((s * s).sum(-1) / D + EPS)          # (S,B,T)
    logits = torch.einsum("sbtd,nd->nsbt", s, q) * irms.unsqueeze(0)
    lse = torch.logsumexp(logits, dim=1)                    # (NQ,B,T)
    probs = torch.exp(logits - lse.unsqueeze(1))
    out = torch.einsum("nsbt,sbtd->nbtd", probs, s)
    grads = torch.autograd.grad(
        [out, lse], [s, q],
        grad_outputs=[grad_out.to(torch.float64), grad_lse.to(torch.float64)],
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

    for s_n, b, t in SHAPES:
        print(f"\n=== shape S={s_n} B={b} T={t} (BT={b*t}) ===")
        src, queries, grad_out, grad_lse = make_inputs(s_n, b, t)
        _out, lses, irms, logits = (
            p1ops._phase_1_batched_attention_forward_with_aux_triton_op(
                src, queries, EPS, s_n
            )
        )
        aux = (lses, irms, logits)

        v1 = p1ops._batched_attention_backward_triton_op(
            src, queries, lses, irms, logits, grad_out, grad_lse, True, EPS, s_n)
        v1 = (v1[0].to(src.dtype), v1[1])  # replicate wrapper conversion
        v2 = p1ops._batched_attention_backward_v2_triton_op(
            src, queries, lses, irms, logits, grad_out, grad_lse, True, EPS, s_n)
        v2b = p1ops._batched_attention_backward_v2_triton_op(
            src, queries, lses, irms, logits, grad_out, grad_lse, True, EPS, s_n)

        e = rel_err(v1[0], v2[0])
        all_ok &= check("grad_blocks v1~v2", e < 2e-2,
                        f"bitwise={torch.equal(v1[0], v2[0])} maxrel {e:.2e}")
        e = rel_err(v1[1], v2[1])
        all_ok &= check("grad_queries v1~v2", e < 1e-4, f"maxrel {e:.2e}")
        all_ok &= check("v2 deterministic (blocks)", torch.equal(v2[0], v2b[0]))
        all_ok &= check("v2 deterministic (queries)", torch.equal(v2[1], v2b[1]))

        ref = fp64_reference(src, queries, grad_out, grad_lse)
        e = rel_err(v2[0], ref[0])
        all_ok &= check("grad_blocks v2 vs fp64", e < 2e-2, f"maxrel {e:.2e}")
        e = rel_err(v2[1], ref[1])
        all_ok &= check("grad_queries v2 vs fp64", e < 1e-3, f"maxrel {e:.2e}")

        vt = p1ops._batched_attention_backward_torch(
            src, queries, lses, irms, logits, grad_out, grad_lse, True, EPS, s_n)
        vtb = p1ops._batched_attention_backward_torch(
            src, queries, lses, irms, logits, grad_out, grad_lse, True, EPS, s_n)
        e = rel_err(vt[0], v1[0])
        all_ok &= check("grad_blocks torch~v1", e < 2e-2, f"maxrel {e:.2e}")
        e = rel_err(vt[1], v1[1])
        # 5e-3: bf16 GEMM-output rounding on the dots einsum (Tier-C envelope;
        # the canary loss overlay is the binding gate for this impl)
        all_ok &= check("grad_queries torch~v1", e < 5e-3, f"maxrel {e:.2e}")
        all_ok &= check("torch deterministic (blocks)", torch.equal(vt[0], vtb[0]))
        all_ok &= check("torch deterministic (queries)", torch.equal(vt[1], vtb[1]))
        e = rel_err(vt[0], ref[0])
        all_ok &= check("grad_blocks torch vs fp64", e < 2e-2, f"maxrel {e:.2e}")
        e = rel_err(vt[1], ref[1])
        all_ok &= check("grad_queries torch vs fp64", e < 5e-3, f"maxrel {e:.2e}")

    # End-to-end dispatch check through autograd
    print("\n=== end-to-end autograd dispatch ===")
    src, queries, grad_out, grad_lse = make_inputs(7, 4, 4096, seed=7)
    grads = {}
    for impl in ("v1", "v2", "torch"):
        p1ops._P1_BWD_V2 = impl == "v2"
        p1ops._P1_BWD_TORCH = impl == "torch"
        s_leaf = src.clone().requires_grad_(True)
        q_leaf = queries.clone().requires_grad_(True)
        out, lse = p1ops.phase_1_batched_attention_triton_op(s_leaf, q_leaf, EPS, 7)
        torch.autograd.backward([out, lse], [grad_out, grad_lse])
        grads[impl] = (s_leaf.grad.clone(), q_leaf.grad.clone())
    p1ops._P1_BWD_V2 = False
    p1ops._P1_BWD_TORCH = False
    for other in ("v2", "torch"):
        for i, (nm, tol) in enumerate([("blocks", 2e-2),
                                       ("queries", 1e-4 if other == "v2" else 5e-3)]):
            e = rel_err(grads[other][i], grads["v1"][i])
            bw = torch.equal(grads[other][i], grads["v1"][i])
            all_ok &= check(f"e2e grad {nm} v1~{other}", e < tol,
                            f"bitwise={bw} maxrel {e:.2e}")

    # Micro-bench
    print("\n=== micro-bench (100 iters, CUDA events) ===")
    for s_n in (3, 7):
        src, queries, grad_out, grad_lse = make_inputs(s_n, 4, 4096)
        _out, lses, irms, logits = (
            p1ops._phase_1_batched_attention_forward_with_aux_triton_op(
                src, queries, EPS, s_n))
        for nm, fn in [("v1", p1ops._batched_attention_backward_triton_op),
                       ("v2", p1ops._batched_attention_backward_v2_triton_op),
                       ("torch", p1ops._batched_attention_backward_torch)]:
            for _ in range(10):
                fn(src, queries, lses, irms, logits, grad_out, grad_lse, True, EPS, s_n)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(100):
                fn(src, queries, lses, irms, logits, grad_out, grad_lse, True, EPS, s_n)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / 100
            print(f"  S={s_n} BT=16384 {nm}: {ms*1000:8.1f} us")

    print("\n" + ("ALL PASS" if all_ok else "FAILURES PRESENT"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
