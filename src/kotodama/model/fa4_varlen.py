"""FA4 (CuTeDSL SM100) attention wrapped as torch custom ops.

Two op pairs, one import guard:
  - kotodama::fa4_varlen / fa4_varlen_bwd — packed (total, H, D) + cu_seqlens
    (doc masking).
  - kotodama::fa4_dense / fa4_dense_bwd — full-causal (B, S, H, D), no
    cu_seqlens (the plain --attn_impl fa4 path).

flash_attn.cute kernels are Blackwell-native
(tcgen05/TMA) but expose no torch.library registration — calling them raw
inside a compiled region is a hard incompatibility (dynamo can't trace CuTe
JIT). This wrapper registers forward+backward as opaque custom ops so
torch.compile schedules around them, same as the FA2 varlen path.

Kernels JIT-compile on first call per shape-class; run_train.sh sets
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 so compiled kernels persist.

Same math as FA2 varlen up to fp reassociation (gate with
scripts/utils/fa4_varlen_parity.py + a canary loss overlay before any
production use).

Return-arity: the installed flash_attn.cute (upstream Dao-AILab/flash-attention
98024f9) returns
(out, lse) from _flash_attn_fwd; upstream head returns a longer tuple. Both
wrappers take res[0], res[1] (and bwd res[0..2]) so either works.

LSE layout (read from _flash_attn_fwd's `lse_shape` at 98024f9):
dense (B, H, S) fp32; varlen (H, total) fp32. Fakes mirror that, and every
fake allocates with new_empty(shape) — real outputs are always freshly
allocated contiguous tensors (the kernels contiguify q/k/v first), so an
empty_like of a strided input would lie about strides.
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

# The pinned FA4 (upstream 98024f9) calls SingleTileVarlenScheduler.producer_tail
# in the varlen forward but never defines it; our installs carry a no-op stub
# (upstream added the same stub in 29e40cf). A plain reinstall drops it.
_FA4_VARLEN_STUB_OK = True
if _FA4_CUTE_AVAILABLE:
    try:
        from flash_attn.cute import tile_scheduler as _ts

        _cls = getattr(_ts, "SingleTileVarlenScheduler", None)
        _FA4_VARLEN_STUB_OK = _cls is None or hasattr(_cls, "producer_tail")
    except ImportError:
        pass


def _require_varlen_stub() -> None:
    if not _FA4_VARLEN_STUB_OK:
        raise RuntimeError(
            "flash_attn.cute lacks SingleTileVarlenScheduler.producer_tail: this FA4 "
            "install is missing the varlen stub (pinned upstream 98024f9 + a no-op "
            "producer_tail stub). Re-apply the stub or use "
            "--attn_impl fa2 / the dense fa4 path (no doc masking)."
        )


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
    _require_varlen_stub()
    res = _flash_attn_fwd(
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
    return res[0], res[1]


@fa4_varlen_op.register_fake
def _(q, k, v, cu_seqlens, max_seqlen):
    out = q.new_empty((*q.shape[:-1], v.shape[-1]))
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
    undefined behavior (NaN grads). Same reason the
    AttnRes phase kernels register backward as separate triton_ops.
    """
    res = _flash_attn_bwd(
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
    return res[0], res[1], res[2]


def _fake_grads(q, k, v):
    return q.new_empty(q.shape), k.new_empty(k.shape), v.new_empty(v.shape)


@fa4_varlen_bwd_op.register_fake
def _(q, k, v, out, grad_out, lse, cu_seqlens, max_seqlen):
    return _fake_grads(q, k, v)


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


# ── dense (full-causal, no cu_seqlens) ────────────────────────────────────
# Same kernel + args as flash_attn.cute.flash_attn_func(q, k, v, causal=True)
# (default softmax_scale = 1/sqrt(head_dim), no window, no softcap, GQA via
# k/v head count; pack_gqa auto). return_lse=True is required: inside a
# custom op the inputs never require grad, so without it the kernel would
# skip the LSE the backward needs. In training (inputs requiring grad) the
# raw call allocated the LSE anyway, so this is the same kernel variant.


@torch.library.custom_op("kotodama::fa4_dense", mutates_args=())
def fa4_dense_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal attention, (B, S, nheads, head_dim) layout; k/v may have fewer
    heads (GQA). Returns (out, lse); out is (B, S, nheads, head_dim_v), lse is
    (B, nheads, S) fp32 — kept for the backward, discarded by the caller.
    """
    res = _flash_attn_fwd(q, k, v, causal=True, return_lse=True)
    return res[0], res[1]


@fa4_dense_op.register_fake
def _(q, k, v):
    bsz, seqlen, nheads, _ = q.shape
    out = q.new_empty((bsz, seqlen, nheads, v.shape[-1]))
    lse = q.new_empty((bsz, nheads, seqlen), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("kotodama::fa4_dense_bwd", mutates_args=())
def fa4_dense_bwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Opaque backward — see fa4_varlen_bwd_op for why it must be its own op."""
    res = _flash_attn_bwd(q, k, v, out, grad_out.contiguous(), lse, causal=True)
    return res[0], res[1], res[2]


@fa4_dense_bwd_op.register_fake
def _(q, k, v, out, grad_out, lse):
    return _fake_grads(q, k, v)


def _dense_setup_context(ctx, inputs, output):
    q, k, v = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)


def _dense_backward(ctx, grad_out, _grad_lse):
    # LSE gradient dropped: callers discard lse (same as flash_attn_func with
    # return_lse=False, which passes dlse=None).
    q, k, v, out, lse = ctx.saved_tensors
    return fa4_dense_bwd_op(q, k, v, out, grad_out, lse)


fa4_dense_op.register_autograd(_dense_backward, setup_context=_dense_setup_context)


def fa4_varlen_available() -> bool:
    return _FA4_CUTE_AVAILABLE
