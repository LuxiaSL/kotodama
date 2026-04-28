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
from dataclasses import dataclass, field
from typing import Optional

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

# ── Optional Flash Attention 2 import ────────────────────────────────────────
_FA2_AVAILABLE = False
try:
    from flash_attn import flash_attn_func

    _FA2_AVAILABLE = True
except ImportError:
    flash_attn_func = None  # type: ignore[assignment,misc]

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
    # Attention implementation: "auto" (FA2 if available, else SDPA), "fa2", "fa4", "sdpa"
    attn_impl: str = "auto"
    # Block Attention Residuals (Moonshot, 2026)
    attn_res: bool = False
    attn_res_n_blocks: int = 7  # N=7 divides 28 layers cleanly into blocks of 4
    attn_res_boundaries: Optional[list[int]] = None  # explicit boundary layers (overrides n_blocks)

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
    cos, sin: (seq_len, head_dim) — full head_dim
    """
    half = x.shape[-1] // 2
    x_rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + x_rotated * sin


def apply_rope_fa2(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings (FA2/FA4 layout).

    x: (batch, seq_len, n_heads, head_dim)
    cos, sin: (seq_len, head_dim) — full head_dim
    """
    half = x.shape[-1] // 2
    x_rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)
    sin = sin.unsqueeze(0).unsqueeze(2)
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


def _resolve_attn_impl(config: LuxiaModelConfig) -> str:
    """Resolve attention implementation: "auto" picks FA2 if available, else SDPA."""
    impl = config.attn_impl
    if impl == "fa4":
        if not _init_fa4():
            raise ImportError(
                "attn_impl='fa4' requested but flash_attn.cute is not available. "
                "Requires flash-attn with CuTeDSL SM100 support."
            )
        return "fa4"
    if impl == "auto":
        # Don't auto-select FA4 (heavy import, JIT compile overhead)
        return "fa2" if _FA2_AVAILABLE else "sdpa"
    if impl == "fa2" and not _FA2_AVAILABLE:
        raise ImportError(
            "attn_impl='fa2' requested but flash-attn is not installed. "
            "Install with: pip install flash-attn"
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
    ) -> torch.Tensor:
        """FA2 path: (B, S, nheads, D) layout, no transposes."""
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rope_fa2(q, rope_cos[:seq_len], rope_sin[:seq_len])
        k = apply_rope_fa2(k, rope_cos[:seq_len], rope_sin[:seq_len])

        # FA2's custom_op doesn't participate in autocast — ensure bf16
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        # FA2 handles GQA natively via shape: Q has more heads than K/V
        attn_output = flash_attn_func(q, k, v, causal=True)

        # Output is (B, S, nheads, D) — reshape directly, no transpose
        attn_output = attn_output.contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def _forward_sdpa(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SDPA path: (B, nheads, S, D) layout."""
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self._use_liger_rope:
            # Liger fused RoPE — processes Q and K jointly in one Triton kernel
            # MUST unsqueeze(0) — kernel uses cos.shape[0] as batch count
            cos = rope_cos[:seq_len].unsqueeze(0)  # (1, S, D)
            sin = rope_sin[:seq_len].unsqueeze(0)
            q, k = liger_rotary_pos_emb(q, k, cos, sin)
        else:
            q = apply_rope(q, rope_cos[:seq_len], rope_sin[:seq_len])
            k = apply_rope(k, rope_cos[:seq_len], rope_sin[:seq_len])

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=mask is None,
            enable_gqa=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def _forward_fa4(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        """FA4 path: (B, S, nheads, D) layout, CuTeDSL SM100 kernels."""
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # FA4 uses same (B, S, nheads, D) layout as FA2
        q = apply_rope_fa2(q, rope_cos[:seq_len], rope_sin[:seq_len])
        k = apply_rope_fa2(k, rope_cos[:seq_len], rope_sin[:seq_len])

        # FA4 doesn't participate in autocast — ensure bf16
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        # FA4 returns (out, lse) tuple — unpack
        attn_output, _lse = _fa4_func(q, k, v, causal=True)

        attn_output = attn_output.contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.tp_group is not None:
            from src.training.tensor_parallel import copy_to_parallel_region, reduce_from_parallel_region
            x = copy_to_parallel_region(x, self.tp_group)

        # FA4/FA2 don't accept arbitrary masks — fall back to SDPA when mask is provided
        if self._attn_impl == "fa4" and mask is None:
            out = self._forward_fa4(x, rope_cos, rope_sin)
        elif self._attn_impl == "fa2" and mask is None:
            out = self._forward_fa2(x, rope_cos, rope_sin)
        else:
            out = self._forward_sdpa(x, rope_cos, rope_sin, mask)

        if self.tp_group is not None:
            out = reduce_from_parallel_region(out, self.tp_group)
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
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin, mask)
        x = x + self.ffn(self.ffn_norm(x))
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
    ) -> torch.Tensor:
        """Compute Block Attention Residual routing with fixed-shape masked softmax.

        All tensor shapes are static (determined by max_sources at init), enabling
        torch.compile to trace a single graph without breaks.

        Args:
            buf: (max_S, B, T, D) padded source buffer (inactive slots are zero)
            query: (D,) learned pseudo-query
            norm: RMSNorm for keys (per-layer, NOT shared)
            active_mask: (max_S,) bool — True for active slots
        Returns:
            h: (B, T, D) attended mixture of active sources
        """
        qw = query * norm.weight  # (D,)
        eps = norm.eps

        # Vectorized logit computation over all slots
        rsqrt = torch.rsqrt(buf.pow(2).mean(-1) + eps)  # (max_S, B, T)
        logits = (buf * qw).sum(-1) * rsqrt  # (max_S, B, T)

        # Mask inactive slots to -inf before softmax (they get weight 0)
        logits = logits.masked_fill(~active_mask.view(-1, 1, 1), float("-inf"))
        weights = F.softmax(logits, dim=0)  # (max_S, B, T)

        # Weighted sum over all slots (zero-weighted inactive slots contribute nothing)
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

    def _forward_attn_res(
        self,
        embed: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass with Block Attention Residuals (compile-friendly).

        Uses pad+stack to build a fixed-shape (max_S, B, T, D) buffer at each
        routing call.  All ops are out-of-place (no autograd version issues) and
        all tensor shapes are static (no torch.compile graph breaks).

        Activation checkpointing is supported: each layer's computation is wrapped
        in torch.utils.checkpoint when config.activation_checkpointing is set.
        """
        committed: list[torch.Tensor] = []  # each (B, T, D), own storage
        partial = embed
        boundary_set = self._attn_res_boundary_set
        max_s = self._attn_res_max_sources
        masks = self._attn_res_masks
        rope_cos = self.rope_cos
        rope_sin = self.rope_sin
        zero = torch.zeros_like(embed)  # reusable padding tensor

        def _pad_and_stack(committed: list[torch.Tensor], partial: torch.Tensor) -> torch.Tensor:
            """Pad sources to max_sources with zeros and stack (out-of-place)."""
            sources = committed + [partial]
            while len(sources) < max_s:
                sources.append(zero)
            return torch.stack(sources, dim=0)  # (max_S, B, T, D)

        for i, layer in enumerate(self.layers):
            # Pre-attention AttnRes: route over committed + partial
            buf = _pad_and_stack(committed, partial)
            h = self._route_static(buf, layer.attn_res_query, layer.attn_res_norm, masks[2 * i])

            # Block boundary: snapshot partial into committed, start fresh
            if i in boundary_set:
                committed.append(partial.clone())
                partial = zero.clone()

            # Attention sub-layer
            attn_out = layer.attn(layer.attn_norm(h), rope_cos, rope_sin, mask)
            partial = partial + attn_out

            # Pre-MLP AttnRes: route over committed + updated partial
            buf = _pad_and_stack(committed, partial)
            h = self._route_static(buf, layer.mlp_res_query, layer.mlp_res_norm, masks[2 * i + 1])

            # MLP sub-layer
            mlp_out = layer.ffn(layer.ffn_norm(h))
            partial = partial + mlp_out

        # Final aggregation
        buf = _pad_and_stack(committed, partial)
        x = self._route_static(buf, self.final_res_query, self.final_res_norm, masks[2 * self.config.num_layers])
        return self.norm(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        x = self.embed_tokens(input_ids)

        if self.config.attn_res:
            # AttnRes has its own layer iteration with routing interleaved.
            # Activation checkpointing is NOT applied here — the routing's shared
            # committed state makes per-layer checkpointing unsound, and memory
            # analysis shows 3B/6B/8B fit on 183GB B200s without it.
            if self.config.activation_checkpointing and self.training:
                logger.warning_once(
                    "activation_checkpointing has no effect with attn_res=True. "
                    "AttnRes routing shares state across layers, making per-layer "
                    "checkpointing unsound. Memory fits without it on B200."
                )
            x = self._forward_attn_res(x, mask)
        else:
            for layer in self.layers:
                if self.config.activation_checkpointing and self.training:
                    x = torch_checkpoint(
                        layer, x, self.rope_cos, self.rope_sin, mask,
                        use_reentrant=False,
                        preserve_rng_state=False,  # no dropout anywhere — skip RNG stash/replay
                    )
                else:
                    x = layer(x, self.rope_cos, self.rope_sin, mask)
            x = self.norm(x)

        output: dict[str, torch.Tensor] = {}

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
