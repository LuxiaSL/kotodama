"""
luxia-base model definition.

Standard Llama architecture with QK-norm and z-loss.
Designed for clean interpretability tooling compatibility.

Optional Liger kernel fusion (--use_liger):
  - LigerRMSNorm: fused RMSNorm for standard norms (NOT AttnRes norms)
  - LigerSiLUMulFunction: fused SwiGLU inner multiply
  - LigerCrossEntropyLoss: fused CE with z-loss (lse_square_scale)
  - LigerFusedLinearCrossEntropyLoss: fused lm_head + CE + z-loss (training-only)
  - liger_rotary_pos_emb: fused Triton RoPE for SDPA path

Optional attention backends (--attn_impl):
  - "sdpa": PyTorch F.scaled_dot_product_attention (default)
  - "fa2": Flash Attention 2 (flash_attn_func)
  - "fa4": Flash Attention 4 CuTeDSL SM100 kernels (lazy import, incompatible with torch.compile)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

logger = logging.getLogger(__name__)

# ── Optional Liger kernel imports ────────────────────────────────────────────
# Gracefully degrade if liger-kernel is not installed.
_LIGER_AVAILABLE = False
try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

    _LIGER_AVAILABLE = True
except ImportError:
    LigerRMSNorm = None  # type: ignore[assignment,misc]
    LigerSiLUMulFunction = None  # type: ignore[assignment,misc]
    LigerCrossEntropyLoss = None  # type: ignore[assignment,misc]
    LigerFusedLinearCrossEntropyLoss = None  # type: ignore[assignment,misc]
    liger_rotary_pos_emb = None  # type: ignore[assignment,misc]

# ── Optional fused Triton AttnRes kernels ───────────────────────────────────
_TRITON_ATTN_RES_AVAILABLE = False
_BATCH_PHASE1 = bool(os.environ.get("KOTODAMA_BATCH_P1"))  # opt-IN: batched Phase 1 regresses throughput (register spilling)
_P1_MAX_BATCH = 10
_TRITON_ATTN_RES_IMPORT_ERROR: Optional[str] = None
if not os.environ.get("KOTODAMA_NO_TRITON_ATTNRES"):
    try:
        import triton  # noqa: F401
        from .flash_attn_res import phase_1_batched_attention_triton_op  # noqa: F401
        from .flash_attn_res import phase_2_online_softmax_merge_triton_op  # noqa: F401
        _TRITON_ATTN_RES_AVAILABLE = True
    except Exception as _triton_exc:  # keep the reason inspectable, not silent
        _TRITON_ATTN_RES_IMPORT_ERROR = repr(_triton_exc)

# ── Optional Flash Attention 2 import ────────────────────────────────────────
_FA2_AVAILABLE = False
_FA2_VARLEN_AVAILABLE = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    _FA2_AVAILABLE = True
    _FA2_VARLEN_AVAILABLE = True
except ImportError:
    flash_attn_func = None  # type: ignore[assignment,misc]
    flash_attn_varlen_func = None  # type: ignore[assignment,misc]

# ── Optional FlexAttention (torch-native, Inductor-fused) ────────────────────
# C1 candidate (SPEC-7B): doc masking via BlockMask instead of FA2 varlen.
# Unlike FA2's opaque custom op, flex lowers through Inductor — the AC-budget
# partitioner can see and schedule around attention. Same math as SDPA up to
# fp reassociation. The BlockMask is built once per step in the train loop
# (outside compile) and threaded down as `block_mask`.
_FLEX_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import flex_attention

    _FLEX_AVAILABLE = True
except ImportError:
    flex_attention = None  # type: ignore[assignment,misc]

# ── Optional Flash Attention 4 (CuTeDSL SM100) ────────────────────────────
# Lazy import: FA4 patches cute.compile globally on import.
_FA4_AVAILABLE = False
_fa4_func = None


def _init_fa4() -> bool:
    """Initialize FA4 on first use. Heavy import — not done at module level."""
    global _FA4_AVAILABLE, _fa4_func
    if _FA4_AVAILABLE:
        return True
    try:
        from flash_attn.cute import flash_attn_func as _f

        _fa4_func = _f
        _FA4_AVAILABLE = True
        return True
    except ImportError:
        return False


_fa4_varlen_op = None


def _init_fa4_varlen() -> bool:
    """FA4 varlen custom op (compile-safe, doc masking). Lazy — see _init_fa4."""
    global _fa4_varlen_op
    if _fa4_varlen_op is not None:
        return True
    try:
        from .fa4_varlen import fa4_varlen_available, fa4_varlen_op

        if not fa4_varlen_available():
            return False
        _fa4_varlen_op = fa4_varlen_op
        return True
    except ImportError:
        return False


@dataclass
class LuxiaModelConfig:
    """Model configuration matching configs/model.yaml."""

    hidden_size: int = 3072
    num_layers: int = 28
    num_attention_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 8192
    vocab_size: int = 49152
    max_position_embeddings: int = 4096
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    qk_norm: bool = True
    tie_word_embeddings: bool = True
    z_loss_weight: float = 1e-5
    activation_checkpointing: bool = False
    # Liger fused kernels (RMSNorm, SwiGLU, CrossEntropy)
    use_liger: bool = False
    # Attention implementation: "auto" (FA2 if available, else SDPA), "fa2",
    # "fa4", "flex" (FlexAttention + BlockMask doc masking), "sdpa"
    attn_impl: str = "auto"
    # Block Attention Residuals (Moonshot, 2026)
    attn_res: bool = False
    attn_res_n_blocks: int = 7  # N=7 divides 28 layers cleanly into blocks of 4
    attn_res_boundaries: Optional[list[int]] = None  # explicit boundary layers (overrides n_blocks)
    # Freeze the first boundary layer's pre-attn routing params (never used:
    # routing is skipped when no blocks are committed). Lets DDP run with
    # find_unused_parameters=False. Disable only when resuming a checkpoint
    # saved with optimizer state from before 2026-07 (AdamW param-count mismatch).
    attn_res_freeze_unused: bool = True

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_kv_heads

    def param_count(self) -> int:
        """Estimate total parameter count."""
        embed = self.vocab_size * self.hidden_size
        # Per layer: attn (Q, K, V, O) + MLP (gate, up, down) + 2 norms
        q = self.hidden_size * self.num_attention_heads * self.head_dim
        k = self.hidden_size * self.num_kv_heads * self.head_dim
        v = self.hidden_size * self.num_kv_heads * self.head_dim
        o = self.num_attention_heads * self.head_dim * self.hidden_size
        attn = q + k + v + o
        # SwiGLU: gate + up project to intermediate, down projects back
        mlp = 3 * self.hidden_size * self.intermediate_size
        norms = 2 * self.hidden_size  # 2 RMSNorm per layer
        qk_norms = 2 * self.head_dim if self.qk_norm else 0  # Q-norm + K-norm
        per_layer = attn + mlp + norms + qk_norms
        total = embed + self.num_layers * per_layer
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.hidden_size
        total += self.hidden_size  # Final norm
        return total


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 500000.0,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE sin/cos frequencies (full head_dim, duplicated for rotation pairs)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim//2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # Full-dim: duplicate for rotation pairs (required for Liger RoPE)
    return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings (SDPA layout).

    x: (batch, n_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim) or (batch, seq_len, head_dim) — full head_dim
    """
    half = x.shape[-1] // 2
    x_rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        cos = cos.unsqueeze(1)  # (B, 1, S, D)
        sin = sin.unsqueeze(1)
    return x * cos + x_rotated * sin


def apply_rope_fa2(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings (FA2/FA4 layout).

    x: (batch, seq_len, n_heads, head_dim)
    cos, sin: (seq_len, head_dim) or (batch, seq_len, head_dim) — full head_dim
    """
    half = x.shape[-1] // 2
    x_rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:
        cos = cos.unsqueeze(2)  # (B, S, 1, D)
        sin = sin.unsqueeze(2)
    return x * cos + x_rotated * sin


def _select_norm_class(config: LuxiaModelConfig) -> type:
    """Select norm class: LigerRMSNorm when use_liger is enabled, else custom RMSNorm.

    NOT used for AttnRes norms — those always use custom RMSNorm because
    _route_static() accesses norm.eps directly, and LigerRMSNorm
    stores it as .variance_epsilon instead.
    """
    if config.use_liger and _LIGER_AVAILABLE:
        return LigerRMSNorm
    return RMSNorm


@torch.no_grad()
def _compute_routing_alphas(
    committed_stack: Optional[torch.Tensor],
    n_committed: int,
    partial: torch.Tensor,
    qw: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute routing softmax weights for one routing point. Returns (n_src,) avg alphas."""
    if n_committed == 0:
        return torch.ones(1, device=partial.device)
    all_src = torch.cat([committed_stack[:n_committed], partial.unsqueeze(0)], dim=0)
    rsqrt = torch.rsqrt(all_src.pow(2).mean(-1) + eps)
    logits = (all_src * qw).sum(-1) * rsqrt
    weights = F.softmax(logits, dim=0)
    return weights.mean(dim=(1, 2))


@torch.no_grad()
def _compute_attn_res_diagnostics(
    committed: list[torch.Tensor],
    partial: torch.Tensor,
    final_query: torch.Tensor,
    final_norm_weight: torch.Tensor,
    eps: float,
    intermediate_alphas: Optional[list[tuple[int, str, torch.Tensor]]] = None,
) -> dict[str, float]:
    """Compute per-block norms, final routing weights, and intermediate routing stats."""
    diag: dict[str, float] = {}
    for i, c in enumerate(committed):
        diag[f"attnres/block_norm/{i}"] = c.float().norm(dim=-1).mean().item()
    diag["attnres/partial_norm"] = partial.float().norm(dim=-1).mean().item()

    # Final aggregation alphas
    all_src = committed + [partial]
    src = torch.stack(all_src, dim=0)
    qw = final_query * final_norm_weight
    rsqrt = torch.rsqrt(src.pow(2).mean(-1) + eps)
    logits = (src * qw).sum(-1) * rsqrt
    weights = F.softmax(logits, dim=0)
    avg_w = weights.mean(dim=(1, 2))
    for i in range(len(all_src)):
        label = f"block_{i}" if i < len(committed) else "partial"
        diag[f"attnres/final_alpha/{label}"] = avg_w[i].item()

    # Intermediate routing: per-layer entropy + per-block average alpha
    if intermediate_alphas:
        n_blocks = len(committed)
        block_alpha_sums = [0.0] * (n_blocks + 1)  # +1 for partial
        block_alpha_counts = [0] * (n_blocks + 1)

        for layer_idx, sublayer, alphas in intermediate_alphas:
            n_src = alphas.shape[0]
            entropy = -(alphas * (alphas + 1e-10).log()).sum().item()
            max_entropy = math.log(n_src) if n_src > 1 else 1.0
            diag[f"attnres/routing_entropy/layer_{layer_idx}/{sublayer}"] = entropy / max_entropy

            for s in range(n_src):
                bucket = s if s < n_blocks else n_blocks  # last = partial
                block_alpha_sums[bucket] += alphas[s].item()
                block_alpha_counts[bucket] += 1

        for b in range(n_blocks + 1):
            if block_alpha_counts[b] > 0:
                label = f"block_{b}" if b < n_blocks else "partial"
                diag[f"attnres/avg_alpha/{label}"] = block_alpha_sums[b] / block_alpha_counts[b]

    return diag


def _resolve_attn_impl(config: LuxiaModelConfig) -> str:
    """Resolve attention implementation: "auto" picks FA2 if available, else SDPA."""
    impl = config.attn_impl
    if impl == "fa4":
        if not _init_fa4():
            raise ImportError(
                "attn_impl='fa4' requested but flash_attn.cute is not available. "
                "Requires flash-attn with CuTeDSL SM100 support."
            )
        _init_fa4_varlen()  # optional: enables the compile-safe varlen path
        return "fa4"
    if impl == "auto":
        # Don't auto-select FA4 (heavy import, JIT compile overhead)
        return "fa2" if _FA2_AVAILABLE else "sdpa"
    if impl == "fa2" and not _FA2_AVAILABLE:
        raise ImportError(
            "attn_impl='fa2' requested but flash-attn is not installed. "
            "Install with: pip install flash-attn"
        )
    if impl == "flex" and not _FLEX_AVAILABLE:
        raise ImportError(
            "attn_impl='flex' requested but torch.nn.attention.flex_attention "
            "is not available in this torch build."
        )
    return impl


class GQAttention(nn.Module):
    """Grouped-Query Attention with optional QK-norm.

    Supports three attention backends:
      - "sdpa": PyTorch's F.scaled_dot_product_attention (default fallback)
      - "fa2": Flash Attention 2 via flash_attn_func
      - "fa4": Flash Attention 4 CuTeDSL SM100 kernels (lazy import)

    FA2/FA4 use (B, S, nheads, D) layout; SDPA uses (B, nheads, S, D).
    When a custom mask is provided, always falls back to SDPA
    (FA2/FA4 don't accept arbitrary attention masks).
    """

    def __init__(self, config: LuxiaModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        self._attn_impl = _resolve_attn_impl(config)
        self._use_liger_rope = config.use_liger and _LIGER_AVAILABLE
        self.tp_group = None  # set by apply_tensor_parallelism

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # QK-norm: RMSNorm applied to Q and K after projection, before attention
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            NormClass = _select_norm_class(config)
            self.q_norm = NormClass(self.head_dim, eps=config.norm_eps)
            self.k_norm = NormClass(self.head_dim, eps=config.norm_eps)

    def _forward_fa2(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """FA2 path: (B, S, nheads, D) layout, no transposes."""
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos.ndim == 3:
            # Pre-gathered (B, S, D) tables from model.forward — no per-layer gather
            pos_cos, pos_sin = rope_cos, rope_sin
        elif position_ids is not None:
            pos_cos = rope_cos[position_ids]
            pos_sin = rope_sin[position_ids]
        else:
            pos_cos = rope_cos[:seq_len]
            pos_sin = rope_sin[:seq_len]
        q = apply_rope_fa2(q, pos_cos, pos_sin)
        k = apply_rope_fa2(k, pos_cos, pos_sin)

        # FA2's custom_op doesn't participate in autocast — ensure bf16
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        # FA2 handles GQA natively via shape: Q has more heads than K/V
        attn_output = flash_attn_func(q, k, v, causal=True)

        # Output is (B, S, nheads, D) — reshape directly, no transpose
        attn_output = attn_output.contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def _forward_fa2_varlen(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """FA2 variable-length path for document-boundary masking.

        Uses flash_attn_varlen_func to prevent attention across document
        boundaries within packed sequences. Each document is treated as an
        independent sequence for attention computation.

        Args:
            x: (B, S, D) input hidden states
            cu_seqlens: (total_docs+1,) int32 cumulative doc lengths (flat across batch)
            max_seqlen: maximum document length in this batch
            position_ids: (B, S) per-token positions (reset at doc boundaries)
        """
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE with document-local positions
        if rope_cos.ndim == 3:
            # Pre-gathered (B, S, D) tables from model.forward — no per-layer gather
            pos_cos, pos_sin = rope_cos, rope_sin
        else:
            pos_cos = rope_cos[position_ids]  # (B, S, D)
            pos_sin = rope_sin[position_ids]  # (B, S, D)
        q = apply_rope_fa2(q, pos_cos, pos_sin)
        k = apply_rope_fa2(k, pos_cos, pos_sin)

        # Flatten batch for varlen: (B, S, H, D) → (B*S, H, D)
        q = q.reshape(-1, self.num_heads, self.head_dim).bfloat16()
        k = k.reshape(-1, self.num_kv_heads, self.head_dim).bfloat16()
        v = v.reshape(-1, self.num_kv_heads, self.head_dim).bfloat16()

        attn_output = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )

        # Reshape back: (B*S, H, D) → (B, S, H*D)
        attn_output = attn_output.reshape(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def _forward_sdpa(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """SDPA path: (B, nheads, S, D) layout. Returns (output, (cached_k, cached_v))."""
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos.ndim == 3:
            # Pre-gathered (B, S, D) tables from model.forward — no per-layer gather
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
        elif position_ids is not None:
            pos_cos = rope_cos[position_ids]
            pos_sin = rope_sin[position_ids]
            q = apply_rope(q, pos_cos, pos_sin)
            k = apply_rope(k, pos_cos, pos_sin)
        elif self._use_liger_rope:
            pos_offset = past_kv[0].shape[2] if past_kv is not None else 0
            cos = rope_cos[pos_offset:pos_offset + seq_len].unsqueeze(0)
            sin = rope_sin[pos_offset:pos_offset + seq_len].unsqueeze(0)
            q, k = liger_rotary_pos_emb(q, k, cos, sin)
        else:
            pos_offset = past_kv[0].shape[2] if past_kv is not None else 0
            q = apply_rope(q, rope_cos[pos_offset:pos_offset + seq_len], rope_sin[pos_offset:pos_offset + seq_len])
            k = apply_rope(k, rope_cos[pos_offset:pos_offset + seq_len], rope_sin[pos_offset:pos_offset + seq_len])

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        new_kv = (k, v)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(past_kv is None and mask is None),
            enable_gqa=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output), new_kv

    def _forward_fa4(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """FA4 path: (B, S, nheads, D) layout, CuTeDSL SM100 kernels."""
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos.ndim == 3:
            # Pre-gathered (B, S, D) tables from model.forward — no per-layer gather
            pos_cos, pos_sin = rope_cos, rope_sin
        elif position_ids is not None:
            pos_cos = rope_cos[position_ids]
            pos_sin = rope_sin[position_ids]
        else:
            pos_cos = rope_cos[:seq_len]
            pos_sin = rope_sin[:seq_len]
        q = apply_rope_fa2(q, pos_cos, pos_sin)
        k = apply_rope_fa2(k, pos_cos, pos_sin)

        # FA4 doesn't participate in autocast — ensure bf16
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        # FA4 returns (out, lse) tuple — unpack
        attn_output, _lse = _fa4_func(q, k, v, causal=True)

        attn_output = attn_output.contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def _forward_fa4_varlen(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """FA4 varlen path via the kotodama::fa4_varlen custom op.

        Identical structure to _forward_fa2_varlen; the custom-op wrapper
        makes the CuTe kernels opaque to torch.compile instead of a hard
        incompatibility.
        """
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos.ndim == 3:
            pos_cos, pos_sin = rope_cos, rope_sin
        else:
            pos_cos = rope_cos[position_ids]
            pos_sin = rope_sin[position_ids]
        q = apply_rope_fa2(q, pos_cos, pos_sin)
        k = apply_rope_fa2(k, pos_cos, pos_sin)

        q = q.reshape(-1, self.num_heads, self.head_dim).bfloat16()
        k = k.reshape(-1, self.num_kv_heads, self.head_dim).bfloat16()
        v = v.reshape(-1, self.num_kv_heads, self.head_dim).bfloat16()

        attn_output, _lse = _fa4_varlen_op(q, k, v, cu_seqlens, max_seqlen)

        attn_output = attn_output.reshape(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def _forward_flex(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        block_mask,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """FlexAttention path: (B, nheads, S, D) layout, BlockMask doc masking.

        The BlockMask (causal-within-document) is built once per step in the
        train loop and shared by all layers. Inside the compiled model, flex
        lowers to a fused Inductor template — no opaque custom op.
        """
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos.ndim == 3:
            # Pre-gathered (B, S, D) tables from model.forward
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
        elif position_ids is not None:
            q = apply_rope(q, rope_cos[position_ids], rope_sin[position_ids])
            k = apply_rope(k, rope_cos[position_ids], rope_sin[position_ids])
        else:
            q = apply_rope(q, rope_cos[:seq_len], rope_sin[:seq_len])
            k = apply_rope(k, rope_cos[:seq_len], rope_sin[:seq_len])

        attn_output = flex_attention(
            q, k, v, block_mask=block_mask, enable_gqa=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        block_mask=None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.tp_group is not None:
            from src.training.tensor_parallel import copy_to_parallel_region, reduce_from_parallel_region
            x = copy_to_parallel_region(x, self.tp_group)

        new_kv: tuple[torch.Tensor, torch.Tensor] | None = None

        if (past_kv is not None or use_cache) and self._attn_impl != "sdpa":
            # KV cache only implemented for SDPA; fall through
            pass

        # Priority: KV cache > flex (BlockMask doc masking) > varlen (doc
        # masking) > FA4/FA2 full-causal > SDPA
        if past_kv is not None or use_cache:
            out, new_kv = self._forward_sdpa(x, rope_cos, rope_sin, mask, past_kv, position_ids)
        elif block_mask is not None and self._attn_impl == "flex" and x.is_cuda:
            out = self._forward_flex(x, rope_cos, rope_sin, block_mask, position_ids)
        elif cu_seqlens is not None and self._attn_impl == "fa4" and _fa4_varlen_op is not None and x.is_cuda:
            out = self._forward_fa4_varlen(x, rope_cos, rope_sin, cu_seqlens, max_seqlen, position_ids)
        elif cu_seqlens is not None and _FA2_VARLEN_AVAILABLE and self._attn_impl in ("fa2", "auto", "flex") and x.is_cuda:
            out = self._forward_fa2_varlen(x, rope_cos, rope_sin, cu_seqlens, max_seqlen, position_ids)
        elif cu_seqlens is not None:
            # SDPA fallback with block-causal mask built from cu_seqlens
            out = self._forward_sdpa(x, rope_cos, rope_sin, mask, position_ids=position_ids)
            out = out[0]
        elif self._attn_impl == "fa4" and mask is None and position_ids is None:
            out = self._forward_fa4(x, rope_cos, rope_sin)
        elif self._attn_impl == "fa2" and mask is None and position_ids is None:
            out = self._forward_fa2(x, rope_cos, rope_sin)
        else:
            out = self._forward_sdpa(x, rope_cos, rope_sin, mask, position_ids=position_ids)
            out = out[0]  # discard unused cache

        if self.tp_group is not None:
            out = reduce_from_parallel_region(out, self.tp_group)

        if use_cache:
            return out, new_kv
        return out


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: LuxiaModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self._use_liger = config.use_liger and _LIGER_AVAILABLE
        self.tp_group = None  # set by apply_tensor_parallelism

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_group is not None:
            from src.training.tensor_parallel import copy_to_parallel_region, reduce_from_parallel_region
            x = copy_to_parallel_region(x, self.tp_group)

        if self._use_liger:
            out = self.down_proj(
                LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x))
            )
        else:
            out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

        if self.tp_group is not None:
            out = reduce_from_parallel_region(out, self.tp_group)
        return out


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block, with optional Block AttnRes."""

    def __init__(self, config: LuxiaModelConfig) -> None:
        super().__init__()
        NormClass = _select_norm_class(config)
        self.attn_norm = NormClass(config.hidden_size, eps=config.norm_eps)
        self.attn = GQAttention(config)
        self.ffn_norm = NormClass(config.hidden_size, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(config)

        # Block AttnRes: per-layer pseudo-queries and key norms.
        # Always use custom RMSNorm — _route_static() accesses norm.eps
        # directly, and LigerRMSNorm stores it as .variance_epsilon.
        if config.attn_res:
            self.attn_res_query = nn.Parameter(torch.zeros(config.hidden_size))
            self.attn_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.mlp_res_query = nn.Parameter(torch.zeros(config.hidden_size))
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        block_mask=None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if use_cache:
            attn_out, new_kv = self.attn(self.attn_norm(x), rope_cos, rope_sin, mask, past_kv, use_cache=True, position_ids=position_ids)
            x = x + attn_out
        else:
            x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin, mask, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, block_mask=block_mask)
        x = x + self.ffn(self.ffn_norm(x))
        if use_cache:
            return x, new_kv
        return x


class LuxiaBaseModel(nn.Module):
    """
    luxia-base: Llama-family transformer with QK-norm and z-loss.

    Clean implementation designed for:
    - Anamnesis geometric monitoring (hook-friendly architecture)
    - FSDP2 compatibility (no custom autograd)
    - torch.compile compatibility (standard ops only)
    """

    def __init__(self, config: LuxiaModelConfig) -> None:
        super().__init__()
        self.config = config
        self._use_liger = config.use_liger and _LIGER_AVAILABLE

        if config.use_liger and not _LIGER_AVAILABLE:
            logger.warning(
                "use_liger=True but liger-kernel is not installed. "
                "Falling back to standard kernels."
            )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        NormClass = _select_norm_class(config)
        self.norm = NormClass(config.hidden_size, eps=config.norm_eps)

        # LM head — tied with embeddings
        if config.tie_word_embeddings:
            self.lm_head = None  # Use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Fused CE when Liger is available (used when logits already materialized, e.g. eval with labels)
        if self._use_liger:
            self.ce_loss = LigerCrossEntropyLoss(
                ignore_index=-100,
                lse_square_scale=config.z_loss_weight,
                return_z_loss=True,  # always returns CrossEntropyOutput; z_loss=0 when weight=0
            )
        else:
            self.ce_loss = None

        # Fused linear CE: skips logit materialization entirely (training-only)
        if self._use_liger:
            self.fused_linear_ce_loss = LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100,
                lse_square_scale=config.z_loss_weight,
                return_z_loss=True,  # always returns CrossEntropyOutput; z_loss=0 when weight=0
                accum_dtype=torch.float32,  # match autocast mixed-precision accumulation
            )
        else:
            self.fused_linear_ce_loss = None

        # Block AttnRes: final output aggregation query + norm
        # Always use custom RMSNorm for AttnRes norms (see _select_norm_class).
        if config.attn_res:
            self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
            self.final_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
            # Precompute block boundaries — explicit list or derived from n_blocks
            if config.attn_res_boundaries is not None:
                self._attn_res_boundary_set = frozenset(config.attn_res_boundaries)
            else:
                block_size = math.ceil(config.num_layers / config.attn_res_n_blocks)
                self._attn_res_boundary_set = frozenset(
                    range(0, config.num_layers, block_size)
                )
            # Max sources = number of committed blocks + 1 (partial)
            self._attn_res_max_sources = len(self._attn_res_boundary_set) + 1
            # Precompute validity masks for each routing call:
            # 2 per layer (pre-attention, pre-MLP) + 1 final = 2*num_layers + 1
            # Each mask is (max_sources,) bool — True for active slots.
            masks = torch.zeros(2 * config.num_layers + 1, self._attn_res_max_sources, dtype=torch.bool)
            n_committed = 0
            for i in range(config.num_layers):
                # Pre-attention: n_committed committed + 1 partial
                masks[2 * i, :n_committed + 1] = True
                # Boundary happens between pre-attention and pre-MLP
                if i in self._attn_res_boundary_set:
                    n_committed += 1
                # Pre-MLP: possibly one more committed
                masks[2 * i + 1, :n_committed + 1] = True
            # Final aggregation
            masks[2 * config.num_layers, :n_committed + 1] = True
            self.register_buffer("_attn_res_masks", masks, persistent=False)

            self._attn_res_active_counts = [int(masks[j].sum().item()) for j in range(masks.shape[0])]

            # Precompute block ranges for block-boundary activation checkpointing
            sorted_boundaries = sorted(self._attn_res_boundary_set)
            self._attn_res_block_ranges: list[tuple[int, int]] = []
            for idx, b in enumerate(sorted_boundaries):
                end = sorted_boundaries[idx + 1] if idx + 1 < len(sorted_boundaries) else config.num_layers
                self._attn_res_block_ranges.append((b, end))

            # The first boundary layer's pre-attn routing params are never used:
            # routing is skipped when n_committed == 0 (training path), and in
            # the cached path the single-source softmax is constant 1 regardless
            # of the query. Freezing them lets DDP run without
            # find_unused_parameters=True (per-step graph traversal).
            if config.attn_res_freeze_unused:
                first_boundary_layer = self.layers[sorted_boundaries[0]]
                first_boundary_layer.attn_res_query.requires_grad_(False)
                first_boundary_layer.attn_res_norm.weight.requires_grad_(False)

        # Precompute RoPE frequencies
        rope_cos, rope_sin = precompute_rope_frequencies(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Standard Llama initialization."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Scale down residual projections (o_proj, down_proj) by 1/sqrt(2*num_layers)
        residual_scale = 1.0 / math.sqrt(2 * self.config.num_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=std * residual_scale)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=std * residual_scale)

    def get_lm_head_weight(self) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head.weight
        return self.embed_tokens.weight

    @staticmethod
    def _route_static(
        buf: torch.Tensor,
        query: torch.Tensor,
        norm: nn.Module,
        active_mask: torch.Tensor,
        num_active: int = 0,
    ) -> torch.Tensor:
        """Compute Block Attention Residual routing with fixed-shape masked softmax.

        All tensor shapes are static (determined by max_sources at init), enabling
        torch.compile to trace a single graph without breaks.

        When Triton is available (CUDA), uses fused Phase 1 kernel from
        flash-attention-residuals. Falls back to PyTorch on CPU.

        Args:
            buf: (max_S, B, T, D) padded source buffer (inactive slots are zero)
            query: (D,) learned pseudo-query
            norm: RMSNorm for keys (per-layer, NOT shared)
            active_mask: (max_S,) bool — True for active slots
            num_active: number of active sources (avoids .sum() on mask at runtime)
        Returns:
            h: (B, T, D) attended mixture of active sources
        """
        qw = query * norm.weight  # (D,)
        eps = norm.eps

        if buf.is_cuda and _TRITON_ATTN_RES_AVAILABLE:
            from .flash_attn_res.ops.phase_1 import phase_1_batched_attention_triton_op
            out, _lse = phase_1_batched_attention_triton_op(
                buf, qw.unsqueeze(0), eps, num_active=num_active or int(active_mask.sum().item()),
            )
            return out[0]

        rsqrt = torch.rsqrt(buf.pow(2).mean(-1) + eps)  # (max_S, B, T)
        logits = (buf * qw).sum(-1) * rsqrt  # (max_S, B, T)
        logits = logits.masked_fill(~active_mask.view(-1, 1, 1), float("-inf"))
        weights = F.softmax(logits, dim=0)  # (max_S, B, T)
        return (weights.unsqueeze(-1) * buf).sum(0)  # (B, T, D)

    def _block_attn_res_from_list(
        self,
        sources: list[torch.Tensor],
        query: torch.Tensor,
        norm: nn.Module,
    ) -> torch.Tensor:
        """Compat shim: wraps _route_static for scripts that pass a variable-length list.

        Not used in the training forward pass (which uses _route_static directly).
        """
        max_s = self._attn_res_max_sources
        zero = torch.zeros_like(sources[0])
        padded = list(sources)
        while len(padded) < max_s:
            padded.append(zero)
        buf = torch.stack(padded, dim=0)
        active_mask = torch.zeros(max_s, dtype=torch.bool, device=buf.device)
        active_mask[:len(sources)] = True
        return self._route_static(buf, query, norm, active_mask)

    def _route_p12(
        self,
        committed_stack: Optional[torch.Tensor],
        n_committed: int,
        partial: torch.Tensor,
        query: torch.Tensor,
        norm: nn.Module,
    ) -> torch.Tensor:
        """Route using Phase 1 (inter-block attention) + Phase 2 (online merge).

        Eliminates _pad_and_stack entirely. Phase 1 attends over committed blocks,
        Phase 2 merges the result with the current partial via online softmax.
        Falls back to PyTorch when Triton is unavailable.
        """
        if n_committed == 0:
            return partial

        qw = query * norm.weight
        eps = norm.eps

        if committed_stack is not None and committed_stack.is_cuda and _TRITON_ATTN_RES_AVAILABLE:
            # NOTE: use the module-level op bindings (imported at line ~60).
            # A runtime relative re-import here resolves through sys.modules
            # and breaks when an embedding process (e.g. posttraining's
            # _import_from_pretraining) has purged/replaced the src package.
            p1_out, p1_lse = phase_1_batched_attention_triton_op(
                committed_stack, qw.unsqueeze(0), eps, num_active=n_committed)
            return phase_2_online_softmax_merge_triton_op(
                partial, qw, p1_out[0], p1_lse[0], eps)

        # PyTorch fallback
        rsqrt_c = torch.rsqrt(committed_stack.pow(2).mean(-1) + eps)
        logits_c = (committed_stack * qw).sum(-1) * rsqrt_c
        rsqrt_p = torch.rsqrt(partial.pow(2).mean(-1, keepdim=True) + eps)
        logit_p = ((partial * qw).sum(-1, keepdim=True) * rsqrt_p).squeeze(-1)
        all_logits = torch.cat([logits_c, logit_p.unsqueeze(0)], dim=0)
        weights = F.softmax(all_logits, dim=0)
        all_sources = torch.cat([committed_stack, partial.unsqueeze(0)], dim=0)
        return (weights.unsqueeze(-1) * all_sources).sum(0)

    def _collect_block_queries(
        self,
        block_idx: int,
        block_start: int,
        block_end: int,
    ) -> list[torch.Tensor]:
        """Collect all pre-weighted queries (query * norm.weight) for one block.

        Returns queries in routing order:
          [boundary_pre_mlp, inner_pre_attn, inner_pre_mlp, ..., next_boundary_pre_attn_or_final]
        """
        qws: list[torch.Tensor] = []
        lyr_b = self.layers[block_start]
        qws.append(lyr_b.mlp_res_query * lyr_b.mlp_res_norm.weight)
        for i in range(block_start + 1, block_end):
            lyr = self.layers[i]
            qws.append(lyr.attn_res_query * lyr.attn_res_norm.weight)
            qws.append(lyr.mlp_res_query * lyr.mlp_res_norm.weight)
        if block_idx + 1 < len(self._attn_res_block_ranges):
            nb = self._attn_res_block_ranges[block_idx + 1][0]
            qws.append(self.layers[nb].attn_res_query * self.layers[nb].attn_res_norm.weight)
        else:
            qws.append(self.final_res_query * self.final_res_norm.weight)
        return qws

    def _forward_attn_res(
        self,
        embed: torch.Tensor,
        mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        block_mask=None,
    ) -> torch.Tensor:
        """Forward pass with Block Attention Residuals using Phase 1 + Phase 2.

        When Triton is available, Phase 1 calls are batched per block — all
        queries sharing the same committed_stack are processed in one kernel
        launch (6 launches instead of 57).  Phase 2 (online sigmoid merge)
        runs per routing call as the partial evolves.

        With AC, attention and MLP are individually checkpointed.  Routing
        stays in the main graph (cheap, small outputs).
        """
        use_ac = self.config.activation_checkpointing and self.training
        rope_cos = self.rope_cos
        rope_sin = self.rope_sin
        if position_ids is not None:
            # Gather per-token RoPE tables ONCE per forward. Attention paths
            # detect the pre-gathered (B, S, D) shape and skip their own gather
            # — previously this ran per layer AND re-ran inside every AC
            # region's recompute during backward.
            rope_cos = rope_cos[position_ids]
            rope_sin = rope_sin[position_ids]
        block_ranges = self._attn_res_block_ranges
        batch_p1 = _TRITON_ATTN_RES_AVAILABLE and _BATCH_PHASE1 and embed.is_cuda

        committed: list[torch.Tensor] = []
        committed_stack: Optional[torch.Tensor] = None
        partial = embed
        eps = self.config.norm_eps
        capture_routing = not self.training
        intermediate_alphas: list[tuple[int, str, torch.Tensor]] = []

        prev_bnd_p1_out: Optional[torch.Tensor] = None
        prev_bnd_p1_lse: Optional[torch.Tensor] = None
        prev_bnd_qw: Optional[torch.Tensor] = None

        p1_outs: Optional[torch.Tensor] = None
        p1_lses: Optional[torch.Tensor] = None
        qws: list[torch.Tensor] = []
        qi = 0

        for block_idx, (block_start, block_end) in enumerate(block_ranges):
            n_committed = len(committed)

            # ── Boundary layer pre-attention routing ──────────────────
            # Uses the PREVIOUS block's committed state.
            if n_committed == 0:
                h_attn = partial
            elif batch_p1:
                h_attn = phase_2_online_softmax_merge_triton_op(
                    partial, prev_bnd_qw, prev_bnd_p1_out, prev_bnd_p1_lse, eps)
            else:
                h_attn = self._route_p12(
                    committed_stack, n_committed, partial,
                    self.layers[block_start].attn_res_query,
                    self.layers[block_start].attn_res_norm)
            if capture_routing and n_committed > 0:
                qw = self.layers[block_start].attn_res_query * self.layers[block_start].attn_res_norm.weight
                intermediate_alphas.append((block_start, "pre_attn",
                    _compute_routing_alphas(committed_stack, n_committed, partial, qw, eps)))

            # ── Commit at boundary ────────────────────────────────────
            committed.append(partial)
            partial = torch.zeros_like(embed)
            committed_stack = torch.stack(committed, dim=0)
            n_committed = len(committed)

            # ── Batch Phase 1 for all queries in this block ───────────
            if batch_p1:
                qws = self._collect_block_queries(block_idx, block_start, block_end)
                query_stack = torch.stack(qws, dim=0)
                nq = len(qws)
                if nq <= _P1_MAX_BATCH:
                    p1_outs, p1_lses = phase_1_batched_attention_triton_op(
                        committed_stack, query_stack, eps, num_active=n_committed)
                else:
                    p1_out_chunks: list[torch.Tensor] = []
                    p1_lse_chunks: list[torch.Tensor] = []
                    for s in range(0, nq, _P1_MAX_BATCH):
                        chunk_out, chunk_lse = phase_1_batched_attention_triton_op(
                            committed_stack, query_stack[s:s + _P1_MAX_BATCH],
                            eps, num_active=n_committed)
                        p1_out_chunks.append(chunk_out)
                        p1_lse_chunks.append(chunk_lse)
                    p1_outs = torch.cat(p1_out_chunks, dim=0)
                    p1_lses = torch.cat(p1_lse_chunks, dim=0)
                qi = 0

            # ── Process all layers in this block ──────────────────────
            for i in range(block_start, block_end):
                lyr = self.layers[i]

                # Pre-attention routing (boundary layer already routed above)
                if i != block_start:
                    if batch_p1:
                        h_attn = phase_2_online_softmax_merge_triton_op(
                            partial, qws[qi], p1_outs[qi], p1_lses[qi], eps)
                        qi += 1
                    else:
                        h_attn = self._route_p12(
                            committed_stack, n_committed, partial,
                            lyr.attn_res_query, lyr.attn_res_norm)
                    if capture_routing:
                        qw = lyr.attn_res_query * lyr.attn_res_norm.weight
                        intermediate_alphas.append((i, "pre_attn",
                            _compute_routing_alphas(committed_stack, n_committed, partial, qw, eps)))

                # Attention
                if use_ac:
                    def _attn_fn(h_in: torch.Tensor, p_in: torch.Tensor,
                                 _idx: int = i) -> torch.Tensor:
                        return p_in + self.layers[_idx].attn(
                            self.layers[_idx].attn_norm(h_in), rope_cos, rope_sin, mask,
                            position_ids=position_ids,
                            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                            block_mask=block_mask)
                    partial = torch_checkpoint(
                        _attn_fn, h_attn, partial,
                        use_reentrant=False, preserve_rng_state=False)
                else:
                    partial = partial + lyr.attn(
                        lyr.attn_norm(h_attn), rope_cos, rope_sin, mask,
                        position_ids=position_ids,
                        cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                        block_mask=block_mask)

                # Pre-MLP routing
                if batch_p1:
                    h_mlp = phase_2_online_softmax_merge_triton_op(
                        partial, qws[qi], p1_outs[qi], p1_lses[qi], eps)
                    qi += 1
                else:
                    h_mlp = self._route_p12(
                        committed_stack, n_committed, partial,
                        lyr.mlp_res_query, lyr.mlp_res_norm)
                if capture_routing:
                    qw = lyr.mlp_res_query * lyr.mlp_res_norm.weight
                    intermediate_alphas.append((i, "pre_mlp",
                        _compute_routing_alphas(committed_stack, n_committed, partial, qw, eps)))

                # MLP
                if use_ac:
                    def _mlp_fn(h_in: torch.Tensor, p_in: torch.Tensor,
                                _idx: int = i) -> torch.Tensor:
                        return p_in + self.layers[_idx].ffn(
                            self.layers[_idx].ffn_norm(h_in))
                    partial = torch_checkpoint(
                        _mlp_fn, h_mlp, partial,
                        use_reentrant=False, preserve_rng_state=False)
                else:
                    partial = partial + lyr.ffn(lyr.ffn_norm(h_mlp))

            # ── Carry Phase 1 output for next boundary's pre-attn ────
            if batch_p1:
                prev_bnd_p1_out = p1_outs[qi]
                prev_bnd_p1_lse = p1_lses[qi]
                prev_bnd_qw = qws[qi]

        # ── Final routing ─────────────────────────────────────────────
        if batch_p1:
            h_final = phase_2_online_softmax_merge_triton_op(
                partial, prev_bnd_qw, prev_bnd_p1_out, prev_bnd_p1_lse, eps)
        else:
            h_final = self._route_p12(
                committed_stack, len(committed), partial,
                self.final_res_query, self.final_res_norm)

        if not self.training:
            self._last_attn_res_diagnostics = _compute_attn_res_diagnostics(
                committed, partial, self.final_res_query,
                self.final_res_norm.weight, self.config.norm_eps,
                intermediate_alphas=intermediate_alphas,
            )

        return self.norm(h_final)

    def _forward_attn_res_cached(
        self,
        embed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """AttnRes forward with KV cache support for autoregressive decoding."""
        committed: list[torch.Tensor] = []
        partial = embed
        boundary_set = self._attn_res_boundary_set
        max_s = self._attn_res_max_sources
        masks = self._attn_res_masks
        active_counts = self._attn_res_active_counts
        rope_cos = self.rope_cos
        rope_sin = self.rope_sin
        zero = torch.zeros_like(embed)
        new_kv_list: list[tuple[torch.Tensor, torch.Tensor]] = []

        def _pad_and_stack(committed: list[torch.Tensor], partial: torch.Tensor) -> torch.Tensor:
            sources = committed + [partial]
            while len(sources) < max_s:
                sources.append(zero)
            return torch.stack(sources, dim=0)

        for i, layer in enumerate(self.layers):
            buf = _pad_and_stack(committed, partial)
            h = self._route_static(buf, layer.attn_res_query, layer.attn_res_norm, masks[2 * i], int(active_counts[2 * i]))

            if i in boundary_set:
                committed.append(partial.clone())
                partial = zero.clone()

            layer_past = past_kv[i] if past_kv is not None else None
            attn_out, layer_kv = layer.attn(layer.attn_norm(h), rope_cos, rope_sin, mask, layer_past, use_cache=True, position_ids=position_ids)
            new_kv_list.append(layer_kv)
            partial = partial + attn_out

            buf = _pad_and_stack(committed, partial)
            h = self._route_static(buf, layer.mlp_res_query, layer.mlp_res_norm, masks[2 * i + 1], int(active_counts[2 * i + 1]))

            mlp_out = layer.ffn(layer.ffn_norm(h))
            partial = partial + mlp_out

        buf = _pad_and_stack(committed, partial)
        x = self._route_static(buf, self.final_res_query, self.final_res_norm, masks[2 * self.config.num_layers], int(active_counts[2 * self.config.num_layers]))
        return self.norm(x), new_kv_list

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        past_kv: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        block_mask=None,
    ) -> dict[str, torch.Tensor]:
        x = self.embed_tokens(input_ids)
        new_kv_list: list[tuple[torch.Tensor, torch.Tensor]] = []

        # SDPA fallback: build block-causal mask when cu_seqlens is provided but
        # FA2 varlen is not available (e.g. CPU testing, SDPA-only builds)
        if cu_seqlens is not None and not _FA2_VARLEN_AVAILABLE:
            from src.data.dataset import build_block_causal_mask
            bsz, seq_len = input_ids.shape
            mask = build_block_causal_mask(cu_seqlens, bsz, seq_len, input_ids.device)
            cu_seqlens = None  # layers will use the mask instead
            max_seqlen = None

        if self.config.attn_res:
            if use_cache:
                x, new_kv_list = self._forward_attn_res_cached(x, mask, past_kv, position_ids)
            else:
                x = self._forward_attn_res(x, mask, position_ids, cu_seqlens, max_seqlen, block_mask)
        else:
            rope_cos = self.rope_cos
            rope_sin = self.rope_sin
            if position_ids is not None and not use_cache:
                # Gather per-token RoPE tables once (see _forward_attn_res note)
                rope_cos = rope_cos[position_ids]
                rope_sin = rope_sin[position_ids]
            for i, layer in enumerate(self.layers):
                layer_past = past_kv[i] if past_kv is not None else None
                if self.config.activation_checkpointing and self.training:
                    # Positional args match TransformerBlock.forward — earlier
                    # version dropped position_ids/cu_seqlens here, silently
                    # disabling doc masking under non-AttnRes AC.
                    x = torch_checkpoint(
                        layer, x, rope_cos, rope_sin, mask,
                        None, False, position_ids, cu_seqlens, max_seqlen,
                        block_mask,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                elif use_cache:
                    x, layer_kv = layer(x, rope_cos, rope_sin, mask, layer_past, use_cache=True, position_ids=position_ids)
                    new_kv_list.append(layer_kv)
                else:
                    x = layer(x, rope_cos, rope_sin, mask, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, block_mask=block_mask)
            x = self.norm(x)

        output: dict[str, Any] = {}
        if use_cache:
            output["past_kv"] = new_kv_list

        if labels is not None:
            if self.fused_linear_ce_loss is not None and self.training:
                # Fused path: skip logit materialization entirely
                # MUST use .contiguous() — slices from [:-1] / [1:] are non-contiguous
                shift_hidden = x[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                ce_out = self.fused_linear_ce_loss(
                    self.get_lm_head_weight(),                        # (V, D) — weight first!
                    shift_hidden.reshape(-1, shift_hidden.size(-1)),  # (B*(S-1), D)
                    shift_labels.reshape(-1),                         # (B*(S-1),)
                )
                output["loss"] = ce_out.loss
                output["z_loss"] = ce_out.z_loss
                # logits intentionally not populated — no consumers during training
            else:
                # Materialize logits for non-fused CE or evaluation with labels
                logits = F.linear(x, self.get_lm_head_weight())
                output["logits"] = logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                if self.ce_loss is not None:
                    ce_out = self.ce_loss(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    output["loss"] = ce_out.loss
                    output["z_loss"] = ce_out.z_loss
                else:
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )

                    if self.config.z_loss_weight > 0:
                        log_z = torch.logsumexp(shift_logits, dim=-1)
                        z_loss = self.config.z_loss_weight * (log_z ** 2).mean()
                        loss = loss + z_loss
                        output["z_loss"] = z_loss

                    output["loss"] = loss
        else:
            # Inference: materialize logits
            logits = F.linear(x, self.get_lm_head_weight())
            output["logits"] = logits

        return output

    def reinit_mlps(self) -> None:
        """Reinitialize all MLP weights. Used after NCA pre-pre-training."""
        std = 0.02
        residual_scale = 1.0 / math.sqrt(2 * self.config.num_layers)
        for layer in self.layers:
            nn.init.normal_(layer.ffn.gate_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.ffn.up_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=std * residual_scale)

    def reinit_embeddings(self, new_vocab_size: Optional[int] = None) -> None:
        """Reinitialize embedding layer. Used when switching from NCA to language vocab."""
        device = self.embed_tokens.weight.device
        if new_vocab_size is not None and new_vocab_size != self.config.vocab_size:
            self.config.vocab_size = new_vocab_size
            self.embed_tokens = nn.Embedding(new_vocab_size, self.config.hidden_size, device=device)
            if self.lm_head is not None:
                self.lm_head = nn.Linear(
                    self.config.hidden_size, new_vocab_size, bias=False, device=device
                )
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
