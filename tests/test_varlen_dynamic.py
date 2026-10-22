"""--varlen_dynamic mechanism: a compiled call into kotodama::fa4_varlen compiles once across batches.

Training batches differ in doc count (cu_seqlens length) and longest doc (max_seqlen, an int). Fed as-is, dynamo
compiles static, then recompiles for the new doc count and again for the first new max_seqlen — under DDP each rank
hits that at a different step. With cu_seqlens' length marked dynamic and max_seqlen held at the sequence length
(what train.py does under --varlen_dynamic) the same sequence of batches compiles exactly one graph. A CPU
reference kernel (per-document causal SDPA) stands in for the CuTe kernel; the bound's exactness on the real
kernel is scripts/utils/fa4_maxseqlen_parity.py.
"""

from __future__ import annotations

from typing import Iterator

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

from kotodama.model import fa4_varlen as fa4

T, HQ, HKV, D = 32, 4, 2, 8


def _ref_varlen(q, k, v, cu_seqlens, max_seqlen):
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[1], q.shape[0], dtype=torch.float32)
    bounds = cu_seqlens.tolist()
    for s, e in zip(bounds[:-1], bounds[1:]):
        qh, kh, vh = (t[s:e].transpose(0, 1) for t in (q, k, v))
        o = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
        out[s:e] = o.transpose(0, 1)
        lse[:, s:e] = 0.0
    return out, lse


@pytest.fixture
def cpu_varlen_kernel(monkeypatch) -> Iterator[None]:
    monkeypatch.setattr(fa4, "_FA4_VARLEN_STUB_OK", True)
    op = fa4.fa4_varlen_op
    if "cpu" not in op._backend_fns:
        op.register_kernel("cpu", _ref_varlen)
    else:
        op.set_kernel_enabled("cpu", True)
    try:
        yield
    finally:
        op.set_kernel_enabled("cpu", False)


def _batches() -> list[tuple[torch.Tensor, int]]:
    """(cu_seqlens, true max_seqlen) for 2 rows of T tokens, ordered as training meets them: a new doc count
    while every batch still holds a full-length doc, then the first batch without one."""
    rows = [[[T], [10, 22]], [[3, 2, 27], [T]], [[16, 16], [30, 2]], [[8, 8, 8, 8], [T]]]
    out = []
    for r in rows:
        lens = [n for row in r for n in row]
        cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
        out.append((cu, max(lens)))
    return out


def _attn(q, k, v, cu_seqlens, max_seqlen):
    return fa4.fa4_varlen_op(q * 1.0, k, v, cu_seqlens, max_seqlen)[0].sum()


def _graphs_for(varlen_dynamic: bool) -> int:
    torch._dynamo.reset()
    counters.clear()
    fn = torch.compile(_attn, backend="eager")
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2 * T, h, D, generator=g) for h in (HQ, HKV, HKV))
    for cu, true_max in _batches():
        if varlen_dynamic:
            torch._dynamo.maybe_mark_dynamic(cu, 0)
            fn(q, k, v, cu, T)
        else:
            fn(q, k, v, cu, true_max)
    return counters["stats"]["unique_graphs"]


def test_as_fed_recompiles(cpu_varlen_kernel) -> None:
    assert _graphs_for(varlen_dynamic=False) >= 3


def test_varlen_dynamic_compiles_once(cpu_varlen_kernel) -> None:
    assert _graphs_for(varlen_dynamic=True) == 1


def test_bound_matches_true_max_on_reference(cpu_varlen_kernel) -> None:
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(2 * T, h, D, generator=g) for h in (HQ, HKV, HKV))
    for cu, true_max in _batches():
        a = fa4.fa4_varlen_op(q, k, v, cu, true_max)[0]
        b = fa4.fa4_varlen_op(q, k, v, cu, T)[0]
        assert torch.equal(a, b)
