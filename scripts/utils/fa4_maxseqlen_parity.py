"""FA4-varlen op: true per-batch max_seqlen vs the static upper bound (the sequence length).

Training can pass max_seqlen = sequence_length instead of the batch's longest document so the compiled graph does
not specialize on a per-batch int (each new value is a recompile, and ranks recompile at different steps). This
checks the bound is exact at the moonshot attention shape (12 q / 6 kv heads, head_dim 128, mb 8 x 4096):
forward out + lse bitwise equal; dq/dk/dv differences no larger than the kernel's own run-to-run floor (the
backward accumulates with atomics). Also times both.

Run on a GPU host: tools/run_py.sh scripts/utils/fa4_maxseqlen_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model.fa4_varlen import fa4_varlen_op  # noqa: E402

DEVICE = "cuda"
B, T, NQ, NKV, HD = 8, 4096, 12, 6, 128


def doc_lengths(g: torch.Generator, full_doc: bool) -> list[int]:
    """Per-row document lengths summing to T; optionally one row is a single full-length doc."""
    lens: list[int] = []
    for row in range(B):
        if full_doc and row == 0:
            lens.append(T)
            continue
        left = T
        while left > 0:
            n = min(left, int(torch.randint(16, 3000, (1,), generator=g)))
            lens.append(n)
            left -= n
    return lens


def run(q, k, v, go, cu, max_seqlen):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out, lse = fa4_varlen_op(q, k, v, cu, max_seqlen)
    out.backward(go)
    return out.detach(), lse.detach(), q.grad, k.grad, v.grad


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def bench(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def main() -> int:
    torch.cuda.init()
    ok = True
    g = torch.Generator().manual_seed(0)
    for case, full_doc in [("short docs", False), ("short docs (seed 2)", False), ("one full-length doc", True)]:
        lens = doc_lengths(g, full_doc)
        cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device=DEVICE)
        true_max = max(lens)
        gd = torch.Generator(device=DEVICE).manual_seed(len(lens))
        q = torch.randn(B * T, NQ, HD, device=DEVICE, generator=gd).bfloat16()
        k = torch.randn(B * T, NKV, HD, device=DEVICE, generator=gd).bfloat16()
        v = torch.randn(B * T, NKV, HD, device=DEVICE, generator=gd).bfloat16()
        go = torch.randn(B * T, NQ, HD, device=DEVICE, generator=gd).bfloat16()

        a1 = run(q, k, v, go, cu, true_max)
        a2 = run(q, k, v, go, cu, true_max)
        bnd = run(q, k, v, go, cu, T)
        print(f"{case}: {len(lens)} docs, true max_seqlen {true_max}")
        fwd_eq = torch.equal(a1[0], bnd[0]) and torch.equal(a1[1], bnd[1])
        print(f"  fwd out+lse bitwise equal (true vs bound): {fwd_eq}")
        ok &= fwd_eq
        for i, name in ((2, "dq"), (3, "dk"), (4, "dv")):
            floor, diff = max_abs(a1[i], a2[i]), max_abs(a1[i], bnd[i])
            within = diff <= floor
            print(f"  {name}: run-to-run floor {floor:.3e}  true-vs-bound {diff:.3e}  "
                  f"{'OK' if within else 'ABOVE FLOOR'}")
            ok &= within
        ms_true = bench(lambda: run(q, k, v, go, cu, true_max))
        ms_bnd = bench(lambda: run(q, k, v, go, cu, T))
        print(f"  fwd+bwd ms: true {ms_true:.3f}  bound {ms_bnd:.3f}  ({(ms_bnd / ms_true - 1) * 100:+.2f}%)")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
