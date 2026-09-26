"""FA4 (CuTeDSL SM100) varlen attention wrapped as a torch custom op.

C1 candidate (SPEC-7B): flash_attn.cute kernels are Blackwell-native
(tcgen05/TMA) but expose no torch.library registration — calling them raw
inside a compiled region is a hard incompatibility (dynamo can't trace CuTe
JIT). This wrapper registers forward+backward as opaque custom ops so
torch.compile schedules around them, same as the FA2 varlen path.

Kernels JIT-compile on first call per shape-class; run_train.sh sets
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 so compiled kernels persist.

Same math as FA2 varlen up to fp reassociation (Tier-C: gate with parity
smoke + canary loss overlay before any production use).
"""

from __future__ import annotations

import torch

_FA4_CUTE_AVAILABLE = False
try:
    from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

    _FA4_CUTE_AVAILABLE = True
except ImportError:
    _flash_attn_fwd = None  # type: ignore[assignment]
    _flash_attn_bwd = None  # type: ignore[assignment]


@torch.library.custom_op("kotodama::fa4_varlen", mutates_args=())
def fa4_varlen_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal varlen attention, (total_tokens, nheads, head_dim) packed layout.

    Returns (out, lse); lse is (nheads, total_tokens) fp32 — kept for the
    backward, discarded by the caller.
    """
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
        return_lse=True,
    )
    return out, lse


@fa4_varlen_op.register_fake
def _(q, k, v, cu_seqlens, max_seqlen):
    out = torch.empty_like(q)
    lse = q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("kotodama::fa4_varlen_bwd", mutates_args=())
def fa4_varlen_bwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward as its own opaque op — the registered autograd backward gets
    TRACED by AOTAutograd under compile; raw CuTe calls inside that trace are
    undefined behavior (NaN grads, canary 2026-07-09). Same reason the
    AttnRes phase kernels register backward as separate triton_ops.
    """
    dq, dk, dv = _flash_attn_bwd(
        q,
        k,
        v,
        out,
        grad_out.contiguous(),
        lse,
        causal=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
    )
    return dq, dk, dv


@fa4_varlen_bwd_op.register_fake
def _(q, k, v, out, grad_out, lse, cu_seqlens, max_seqlen):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    q, k, v, cu_seqlens, max_seqlen = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
    ctx.max_seqlen = max_seqlen


def _backward(ctx, grad_out, _grad_lse):
    q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
    dq, dk, dv = fa4_varlen_bwd_op(
        q, k, v, out, grad_out, lse, cu_seqlens, ctx.max_seqlen
    )
    return dq, dk, dv, None, None


fa4_varlen_op.register_autograd(_backward, setup_context=_setup_context)


def fa4_varlen_available() -> bool:
    return _FA4_CUTE_AVAILABLE
