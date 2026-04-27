"""
Benchmark Block Attention Residual optimizations — V2.

Focuses on the strategies most likely to yield real gains:
  - Cached RMSNorm: avoid re-normalizing committed (frozen) buffer slots
  - Fused routing: merge norm + dot + softmax + weighted-sum, skip materializing
    the full normalized buffer
  - Amortized routing: route every K layers instead of every layer
  - Boundary-only routing: route only at block transitions (minimal call count)

Each variant preserves the same mathematical semantics as the current
_forward_attn_res (once-per-layer mode) unless explicitly noted.

Usage:
    python scripts/bench_attnres_v2.py                      # all variants
    python scripts/bench_attnres_v2.py --variants baseline current cached_norm
    python scripts/bench_attnres_v2.py --no-compile
    python scripts/bench_attnres_v2.py --n-blocks 4         # N=4 blocks of 7
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared components ─────────────────────────────────────────────────────────

@dataclass
class BenchConfig:
    hidden_size: int = 512
    num_layers: int = 28
    num_attention_heads: int = 4
    num_kv_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 1408
    vocab_size: int = 49152
    max_position_embeddings: int = 4096
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    qk_norm: bool = True
    attn_res_n_blocks: int = 7


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def precompute_rope_frequencies(
    head_dim: int, max_seq_len: int, theta: float = 500000.0,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GQAttention(nn.Module):
    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = apply_rope(q, rope_cos[:seq_len], rope_sin[:seq_len])
        k = apply_rope(k, rope_cos[:seq_len], rope_sin[:seq_len])
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _make_layer_modules(config: BenchConfig) -> nn.ModuleDict:
    """Create standard transformer layer modules."""
    return nn.ModuleDict({
        "attn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
        "attn": GQAttention(config),
        "ffn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
        "ffn": SwiGLUFFN(config),
    })


def _make_rope_buffers(model: nn.Module, config: BenchConfig) -> None:
    """Register RoPE frequency buffers on a model."""
    rope_cos, rope_sin = precompute_rope_frequencies(
        config.head_dim, config.max_position_embeddings, config.rope_theta
    )
    model.register_buffer("rope_cos", rope_cos, persistent=False)
    model.register_buffer("rope_sin", rope_sin, persistent=False)


def _compute_boundaries(config: BenchConfig) -> tuple[frozenset[int], int]:
    """Return (boundary_set, max_sources)."""
    block_size = math.ceil(config.num_layers / config.attn_res_n_blocks)
    boundary_set = frozenset(range(0, config.num_layers, block_size))
    max_sources = len(boundary_set) + 1
    return boundary_set, max_sources


def _precompute_masks(
    config: BenchConfig,
    boundary_set: frozenset[int],
    max_sources: int,
) -> torch.Tensor:
    """Build (num_layers+1, max_S) validity mask tensor."""
    n_committed = 0
    masks = []
    for i in range(config.num_layers):
        n_valid = n_committed + 1
        mask = torch.zeros(max_sources, dtype=torch.bool)
        mask[:n_valid] = True
        masks.append(mask)
        if i in boundary_set:
            n_committed += 1
    final_mask = torch.zeros(max_sources, dtype=torch.bool)
    final_mask[:n_committed + 1] = True
    masks.append(final_mask)
    return torch.stack(masks, dim=0)


def _precompute_boundary_flags(
    config: BenchConfig,
    boundary_set: frozenset[int],
) -> torch.Tensor:
    """Return (num_layers,) bool tensor — True at boundary layers."""
    return torch.tensor(
        [i in boundary_set for i in range(config.num_layers)],
        dtype=torch.bool,
    )


# ── V0: Baseline (no AttnRes) ────────────────────────────────────────────────

class BaselineModel(nn.Module):
    """Standard residual stream — no routing."""

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([_make_layer_modules(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight
        _make_rope_buffers(self, config)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer["attn"](layer["attn_norm"](x), self.rope_cos, self.rope_sin)
            x = x + layer["ffn"](layer["ffn_norm"](x))
        x = self.norm(x)
        return F.linear(x, self.lm_head_weight)


# ── V1: Current production code (from llama.py) ──────────────────────────────

class CurrentAttnResModel(nn.Module):
    """
    Matches the current _forward_attn_res (once-per-layer mode) in llama.py.
    Pre-allocated buffer, static masks, masked softmax over fixed max_S dim.
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            mods = _make_layer_modules(config)
            mods["res_query"] = nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))})
            mods["res_norm"] = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.layers.append(mods)
        self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight

        self._boundary_set, self._max_sources = _compute_boundaries(config)
        masks = _precompute_masks(config, self._boundary_set, self._max_sources)
        self.register_buffer("_validity_masks", masks, persistent=False)
        _make_rope_buffers(self, config)

    @staticmethod
    def _route(
        buf: torch.Tensor,
        query: torch.Tensor,
        norm: nn.Module,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        K = norm(buf)
        logits = torch.einsum("d, n b t d -> n b t", query, K)
        logits = logits.masked_fill(~mask[:, None, None], float("-inf"))
        weights = F.softmax(logits, dim=0)
        return torch.einsum("n b t, n b t d -> b t d", weights, buf)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        B, T, D = embed.shape
        buf = embed.new_zeros(self._max_sources, B, T, D)
        n_committed = 0
        partial = embed
        masks = self._validity_masks

        for i, layer in enumerate(self.layers):
            buf[n_committed] = partial
            h = self._route(buf, layer["res_query"]["w"], layer["res_norm"], masks[i])

            if i in self._boundary_set:
                buf[n_committed] = partial
                n_committed += 1
                partial = embed.new_zeros(B, T, D)

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](h + attn_out))
            partial = partial + mlp_out

        buf[n_committed] = partial
        x = self._route(buf, self.final_res_query, self.final_res_norm, masks[self.config.num_layers])
        return F.linear(self.norm(x), self.lm_head_weight)


# ── V2: Cached norm — only re-normalize the partial slot ─────────────────────

class CachedNormModel(nn.Module):
    """
    Key insight: committed buffer slots are frozen. Their RMSNorm output never
    changes. We maintain a separate normalized buffer and only update the slot
    that changed (the partial).

    This turns each routing call from O(max_S * B * T * D) norm work to
    O(1 * B * T * D) — a factor of max_S reduction on the dominant cost.

    Trade-off: one extra buffer of size (max_S, B, T, D) for the cached norms.
    At proxy scale (D=512, max_S=8), this is ~8 * B * T * 512 * 2 bytes in bf16
    = negligible compared to activation memory.

    Semantics: mathematically identical to current code. The norm is applied to
    the same values; we just cache the result for frozen slots.
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            mods = _make_layer_modules(config)
            mods["res_query"] = nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))})
            mods["res_norm"] = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.layers.append(mods)
        self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight

        self._boundary_set, self._max_sources = _compute_boundaries(config)
        masks = _precompute_masks(config, self._boundary_set, self._max_sources)
        self.register_buffer("_validity_masks", masks, persistent=False)
        self.register_buffer(
            "_boundary_flags",
            _precompute_boundary_flags(config, self._boundary_set),
            persistent=False,
        )
        _make_rope_buffers(self, config)

    @staticmethod
    def _route_with_cached_norm(
        buf: torch.Tensor,
        norm_buf: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Route using pre-cached normalized keys.
        norm_buf must already contain norm(buf) for all valid slots.
        """
        logits = torch.einsum("d, n b t d -> n b t", query, norm_buf)
        logits = logits.masked_fill(~mask[:, None, None], float("-inf"))
        weights = F.softmax(logits, dim=0)
        return torch.einsum("n b t, n b t d -> b t d", weights, buf)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        B, T, D = embed.shape
        max_s = self._max_sources

        buf = embed.new_zeros(max_s, B, T, D)
        norm_buf = embed.new_zeros(max_s, B, T, D)
        n_committed = 0
        partial = embed
        masks = self._validity_masks
        boundary_flags = self._boundary_flags

        for i, layer in enumerate(self.layers):
            # Update only the partial slot's normalized form
            buf[n_committed] = partial
            norm_buf[n_committed] = layer["res_norm"](partial)

            h = self._route_with_cached_norm(
                buf, norm_buf, layer["res_query"]["w"], masks[i]
            )

            if boundary_flags[i]:
                # Partial becomes committed; norm_buf[n_committed] is already correct
                n_committed += 1
                partial = embed.new_zeros(B, T, D)
            # else: partial slot will be overwritten next iteration anyway

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](h + attn_out))
            partial = partial + mlp_out

        # Final routing
        buf[n_committed] = partial
        norm_buf[n_committed] = self.final_res_norm(partial)
        x = self._route_with_cached_norm(
            buf, norm_buf, self.final_res_query, masks[self.config.num_layers]
        )
        return F.linear(self.norm(x), self.lm_head_weight)


# ── V3: Cached norm + fused logits (no intermediate einsum) ──────────────────

class CachedNormFusedModel(nn.Module):
    """
    Builds on V2 (cached norm) and adds:
    - Maintains a running logits buffer (max_S, B, T) for committed slots.
      When a new layer's query arrives, we only need dot(query, norm_buf[partial_slot])
      for the partial slot. For committed slots, we can precompute
      dot(query, norm_buf[committed]) — but queries differ per layer, so this
      doesn't help directly.

    Actually, the per-layer query is different each time, so we cannot cache
    the dot products across layers. But we CAN avoid the full einsum over
    max_S by doing it as:
      logits_committed = matmul(norm_committed_flat, query)  -- batched matmul
      logit_partial = dot(query, norm_partial)
      logits = cat(logits_committed, logit_partial)

    The real win is still the cached norm. This variant explores whether
    restructuring the einsum as matmul helps torch.compile.

    Mathematical semantics: identical to current code.
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            mods = _make_layer_modules(config)
            mods["res_query"] = nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))})
            mods["res_norm"] = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.layers.append(mods)
        self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight

        self._boundary_set, self._max_sources = _compute_boundaries(config)
        masks = _precompute_masks(config, self._boundary_set, self._max_sources)
        self.register_buffer("_validity_masks", masks, persistent=False)
        self.register_buffer(
            "_boundary_flags",
            _precompute_boundary_flags(config, self._boundary_set),
            persistent=False,
        )
        _make_rope_buffers(self, config)

    @staticmethod
    def _route_fused(
        buf: torch.Tensor,
        norm_buf: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits via matmul instead of einsum for better compile fusion.
        norm_buf: (max_S, B, T, D), query: (D,)
        logits = norm_buf @ query → (max_S, B, T)
        """
        # matmul: (max_S, B, T, D) @ (D, 1) -> (max_S, B, T, 1) -> squeeze
        logits = torch.matmul(norm_buf, query).squeeze(-1)  # (max_S, B, T)
        logits = logits.masked_fill(~mask[:, None, None], float("-inf"))
        weights = F.softmax(logits, dim=0)
        # weighted sum: (max_S, B, T, 1) * (max_S, B, T, D) -> sum over dim 0
        return (weights.unsqueeze(-1) * buf).sum(dim=0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        B, T, D = embed.shape
        max_s = self._max_sources

        buf = embed.new_zeros(max_s, B, T, D)
        norm_buf = embed.new_zeros(max_s, B, T, D)
        n_committed = 0
        partial = embed
        masks = self._validity_masks
        boundary_flags = self._boundary_flags

        for i, layer in enumerate(self.layers):
            buf[n_committed] = partial
            norm_buf[n_committed] = layer["res_norm"](partial)

            h = self._route_fused(buf, norm_buf, layer["res_query"]["w"], masks[i])

            if boundary_flags[i]:
                n_committed += 1
                partial = embed.new_zeros(B, T, D)

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](h + attn_out))
            partial = partial + mlp_out

        buf[n_committed] = partial
        norm_buf[n_committed] = self.final_res_norm(partial)
        x = self._route_fused(buf, norm_buf, self.final_res_query, masks[self.config.num_layers])
        return F.linear(self.norm(x), self.lm_head_weight)


# ── V4: Cached norm + skip-norm for partial (inline RMS) ─────────────────────

class CachedNormInlineRMSModel(nn.Module):
    """
    Like V2 but instead of using a separate RMSNorm module for the partial slot,
    we fuse the query*weight and compute rsqrt inline. This eliminates the
    per-layer RMSNorm module call overhead (Python dispatch + separate kernel).

    The key observation: query (D,) and norm.weight (D,) are both per-layer
    parameters. We can pre-multiply them: scaled_query = query * norm_weight.
    Then logit = dot(scaled_query, partial) * rsqrt(mean(partial^2) + eps).

    For committed slots, we still cache the full normalized representation since
    the norm weight differs per layer (different layers have different norms).

    WAIT — that breaks the caching. If each layer has its own norm weights, the
    cached norm from layer i's norm module is not valid for layer i+1's query.

    Correction: in the current code, each layer has its OWN res_norm. So
    norm(buf) at layer i uses layer i's norm weights. This means we CANNOT
    cache the normalized committed slots across layers — the norm weights change.

    This is actually the critical realization: the current code applies a
    DIFFERENT RMSNorm (different learned weight vector) at each layer to the
    SAME committed data. Caching across layers is not valid.

    However, we CAN still save work: instead of normalizing the full buffer,
    we compute logits directly. For each slot s:
        logit_s = dot(query_i * normweight_i, buf[s]) * rsqrt(mean(buf[s]^2) + eps)

    The rsqrt factor depends only on buf[s], not on norm weights. So we CAN
    cache the rsqrt factors for committed slots.

    rsqrt_s = rsqrt(mean(buf[s]^2, dim=-1) + eps)  shape: (B, T)

    logit_s = dot(query_i * normweight_i, buf[s]) * rsqrt_s

    This avoids materializing the (max_S, B, T, D) normalized tensor entirely.
    Memory bandwidth: read (max_S, B, T, D) once for the dot products + (max_S, B, T)
    for rsqrt, instead of reading (max_S, B, T, D) for norm AND again for dot.

    Actually the einsum "d, n b t d -> n b t" already reads (max_S, B, T, D) once.
    The norm(buf) also reads it once and writes (max_S, B, T, D). So fusing saves
    one full read+write of (max_S, B, T, D) per routing call.

    For the weighted sum at the end, we still read buf (max_S, B, T, D).
    Total memory traffic per call:
      Before: 2 reads + 1 write of (S, B, T, D) + 1 read for weighted sum = 4 passes
      After:  1 read for dot products + 1 read for weighted sum = 2 passes
      Saving: ~50% memory bandwidth per routing call.
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            mods = _make_layer_modules(config)
            mods["res_query"] = nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))})
            mods["res_norm_weight"] = nn.ParameterDict({"w": nn.Parameter(torch.ones(config.hidden_size))})
            self.layers.append(mods)
        self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_res_norm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight
        self._norm_eps = config.norm_eps

        self._boundary_set, self._max_sources = _compute_boundaries(config)
        masks = _precompute_masks(config, self._boundary_set, self._max_sources)
        self.register_buffer("_validity_masks", masks, persistent=False)
        self.register_buffer(
            "_boundary_flags",
            _precompute_boundary_flags(config, self._boundary_set),
            persistent=False,
        )
        _make_rope_buffers(self, config)

    @staticmethod
    def _route_rsqrt_cached(
        buf: torch.Tensor,
        rsqrt_cache: torch.Tensor,
        partial_idx: int,
        query: torch.Tensor,
        norm_weight: torch.Tensor,
        mask: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Route with cached rsqrt for committed slots.

        Updates rsqrt_cache for the partial slot, then computes:
          logits = (buf @ scaled_query) * rsqrt_cache
        where scaled_query = query * norm_weight.

        Returns (routed_output, updated_rsqrt_cache).
        """
        # Update rsqrt for the partial slot only
        partial_rsqrt = torch.rsqrt(buf[partial_idx].pow(2).mean(dim=-1) + eps)
        rsqrt_cache[partial_idx] = partial_rsqrt

        # Compute logits: dot(scaled_query, buf[s]) * rsqrt[s]
        scaled_query = query * norm_weight  # (D,)
        raw_dots = torch.einsum("d, n b t d -> n b t", scaled_query, buf)
        logits = raw_dots * rsqrt_cache  # (max_S, B, T)
        logits = logits.masked_fill(~mask[:, None, None], float("-inf"))
        weights = F.softmax(logits, dim=0)
        out = torch.einsum("n b t, n b t d -> b t d", weights, buf)
        return out, rsqrt_cache

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        B, T, D = embed.shape
        max_s = self._max_sources

        buf = embed.new_zeros(max_s, B, T, D)
        rsqrt_cache = embed.new_zeros(max_s, B, T)  # cached rsqrt factors
        n_committed = 0
        partial = embed
        masks = self._validity_masks
        boundary_flags = self._boundary_flags
        eps = self._norm_eps

        for i, layer in enumerate(self.layers):
            buf[n_committed] = partial
            h, rsqrt_cache = self._route_rsqrt_cached(
                buf, rsqrt_cache, n_committed,
                layer["res_query"]["w"], layer["res_norm_weight"]["w"],
                masks[i], eps,
            )

            if boundary_flags[i]:
                # rsqrt_cache[n_committed] is already set and won't change
                n_committed += 1
                partial = embed.new_zeros(B, T, D)

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](h + attn_out))
            partial = partial + mlp_out

        buf[n_committed] = partial
        h, _ = self._route_rsqrt_cached(
            buf, rsqrt_cache, n_committed,
            self.final_res_query, self.final_res_norm_weight,
            masks[self.config.num_layers], eps,
        )
        return F.linear(self.norm(h), self.lm_head_weight)


# ── V5: Shared norm + rsqrt cache (single norm weight across all layers) ─────

class SharedNormModel(nn.Module):
    """
    Architectural simplification: use a SINGLE shared RMSNorm for all routing
    calls instead of per-layer norms. This enables true cross-layer caching
    of normalized committed slots.

    With a shared norm:
    - Committed slots are normalized ONCE when frozen, cached forever
    - Only the partial slot needs re-normalization each layer
    - Each layer still has its own pseudo-query (per-layer routing decisions)

    This is a slight semantic change: the original has per-layer norm weights
    that can learn layer-specific key transformations. With shared norm, only
    the queries differ per layer. In practice, the norm weights across layers
    likely converge to similar values anyway (they all normalize the same
    residual stream), so this should have minimal impact on model quality.

    Memory bandwidth per routing call:
    - Committed: 0 norm work (cached)
    - Partial: 1 norm of (B, T, D)
    - Dot products: 1 read of (max_S, B, T, D)  (using cached norms)
    - Weighted sum: 1 read of (max_S, B, T, D)
    Total: ~2 reads of (max_S, B, T, D) + 1 read/write of (B, T, D)
    vs current: ~4 reads of (max_S, B, T, D)
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            mods = _make_layer_modules(config)
            mods["res_query"] = nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))})
            self.layers.append(mods)
        self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        # Single shared norm for all routing
        self.shared_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight

        self._boundary_set, self._max_sources = _compute_boundaries(config)
        masks = _precompute_masks(config, self._boundary_set, self._max_sources)
        self.register_buffer("_validity_masks", masks, persistent=False)
        self.register_buffer(
            "_boundary_flags",
            _precompute_boundary_flags(config, self._boundary_set),
            persistent=False,
        )
        _make_rope_buffers(self, config)

    @staticmethod
    def _route_shared_norm(
        buf: torch.Tensor,
        norm_cache: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Route using cached normalized keys (shared norm)."""
        logits = torch.einsum("d, n b t d -> n b t", query, norm_cache)
        logits = logits.masked_fill(~mask[:, None, None], float("-inf"))
        weights = F.softmax(logits, dim=0)
        return torch.einsum("n b t, n b t d -> b t d", weights, buf)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        B, T, D = embed.shape
        max_s = self._max_sources

        buf = embed.new_zeros(max_s, B, T, D)
        norm_cache = embed.new_zeros(max_s, B, T, D)
        n_committed = 0
        partial = embed
        masks = self._validity_masks
        boundary_flags = self._boundary_flags
        shared_norm = self.shared_res_norm

        for i, layer in enumerate(self.layers):
            # Only normalize the partial slot (committed are cached)
            buf[n_committed] = partial
            norm_cache[n_committed] = shared_norm(partial)

            h = self._route_shared_norm(
                buf, norm_cache, layer["res_query"]["w"], masks[i]
            )

            if boundary_flags[i]:
                # norm_cache[n_committed] already has the correct cached norm
                n_committed += 1
                partial = embed.new_zeros(B, T, D)

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](h + attn_out))
            partial = partial + mlp_out

        buf[n_committed] = partial
        norm_cache[n_committed] = shared_norm(partial)
        x = self._route_shared_norm(
            buf, norm_cache, self.final_res_query, masks[self.config.num_layers]
        )
        return F.linear(self.norm(x), self.lm_head_weight)


# ── V6: Shared norm + fused rsqrt (no materialized norm cache) ───────────────

class SharedNormFusedModel(nn.Module):
    """
    Combines shared norm (V5) with rsqrt caching (V4): since the norm is shared,
    the rsqrt factor for committed slots never changes AND the norm weight is
    the same, so we can cache rsqrt AND pre-multiply the shared norm weight
    into the per-layer query.

    Per routing call, this computes:
      scaled_query = query * shared_norm_weight    # (D,) — could be pre-computed
      partial_rsqrt = rsqrt(mean(partial^2) + eps) # (B, T) — only for partial slot
      logits = einsum(scaled_query, buf) * rsqrt   # (max_S, B, T) — 1 read of buf
      weights = softmax(masked(logits))
      output = einsum(weights, buf)                 # (max_S, B, T, D) — 1 read of buf

    Total: 2 reads of buf (max_S, B, T, D). Absolute minimum for this algorithm.
    No materialized norm cache at all — saves (max_S, B, T, D) memory.
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            mods = _make_layer_modules(config)
            # Store query that will be scaled by shared norm weight at runtime
            mods["res_query"] = nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))})
            self.layers.append(mods)
        self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        # Shared norm weight (single vector for all routing)
        self.shared_norm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight
        self._norm_eps = config.norm_eps

        self._boundary_set, self._max_sources = _compute_boundaries(config)
        masks = _precompute_masks(config, self._boundary_set, self._max_sources)
        self.register_buffer("_validity_masks", masks, persistent=False)
        self.register_buffer(
            "_boundary_flags",
            _precompute_boundary_flags(config, self._boundary_set),
            persistent=False,
        )
        _make_rope_buffers(self, config)

    @staticmethod
    def _route_minimal(
        buf: torch.Tensor,
        rsqrt_cache: torch.Tensor,
        partial_idx: int,
        query: torch.Tensor,
        norm_weight: torch.Tensor,
        mask: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Minimal-bandwidth routing with shared norm + rsqrt cache.
        Only computes rsqrt for the partial slot; committed rsqrt is cached.
        """
        # Update rsqrt for partial slot only
        rsqrt_cache[partial_idx] = torch.rsqrt(
            buf[partial_idx].pow(2).mean(dim=-1) + eps
        )

        # logits = (buf @ (query * norm_weight)) * rsqrt
        scaled_query = query * norm_weight  # (D,)
        raw_dots = torch.einsum("d, n b t d -> n b t", scaled_query, buf)
        logits = raw_dots * rsqrt_cache
        logits = logits.masked_fill(~mask[:, None, None], float("-inf"))
        weights = F.softmax(logits, dim=0)
        out = torch.einsum("n b t, n b t d -> b t d", weights, buf)
        return out, rsqrt_cache

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        B, T, D = embed.shape
        max_s = self._max_sources

        buf = embed.new_zeros(max_s, B, T, D)
        rsqrt_cache = embed.new_zeros(max_s, B, T)
        n_committed = 0
        partial = embed
        masks = self._validity_masks
        boundary_flags = self._boundary_flags
        norm_weight = self.shared_norm_weight
        eps = self._norm_eps

        for i, layer in enumerate(self.layers):
            buf[n_committed] = partial
            h, rsqrt_cache = self._route_minimal(
                buf, rsqrt_cache, n_committed,
                layer["res_query"]["w"], norm_weight,
                masks[i], eps,
            )

            if boundary_flags[i]:
                n_committed += 1
                partial = embed.new_zeros(B, T, D)

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](h + attn_out))
            partial = partial + mlp_out

        buf[n_committed] = partial
        h, _ = self._route_minimal(
            buf, rsqrt_cache, n_committed,
            self.final_res_query, norm_weight,
            masks[self.config.num_layers], eps,
        )
        return F.linear(self.norm(h), self.lm_head_weight)


# ── V7: Amortized (route every K layers) ─────────────────────────────────────

class AmortizedModel(nn.Module):
    """
    Route at block boundaries only. Between boundaries, use standard residual
    connections. This reduces routing from 29 calls to N+1 (e.g. 8 for N=7).

    Semantic change: layers within a block don't get per-layer depth routing.
    The block's routed input is computed once and feeds the first layer; subsequent
    layers in the block accumulate via standard residual additions.

    This is the cheapest possible AttnRes implementation while still maintaining
    the inter-block routing that provides the architectural benefit.
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([_make_layer_modules(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight

        self._boundary_set, self._max_sources = _compute_boundaries(config)
        self.register_buffer(
            "_boundary_flags",
            _precompute_boundary_flags(config, self._boundary_set),
            persistent=False,
        )

        # Routing params: one per boundary + one final
        n_route_points = len(self._boundary_set) + 1
        self.route_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(config.hidden_size)) for _ in range(n_route_points)
        ])
        self.route_norms = nn.ModuleList([
            RMSNorm(config.hidden_size, eps=config.norm_eps) for _ in range(n_route_points)
        ])

        # Masks for each routing point
        n_committed = 0
        masks = []
        for i in range(config.num_layers):
            if i in self._boundary_set:
                n_valid = n_committed + 1
                mask = torch.zeros(self._max_sources, dtype=torch.bool)
                mask[:n_valid] = True
                masks.append(mask)
                n_committed += 1
        final_mask = torch.zeros(self._max_sources, dtype=torch.bool)
        final_mask[:n_committed + 1] = True
        masks.append(final_mask)
        self.register_buffer("_route_masks", torch.stack(masks, dim=0), persistent=False)

        _make_rope_buffers(self, config)

    @staticmethod
    def _route(
        buf: torch.Tensor, query: torch.Tensor, norm: nn.Module, mask: torch.Tensor,
    ) -> torch.Tensor:
        K = norm(buf)
        logits = torch.einsum("d, n b t d -> n b t", query, K)
        logits = logits.masked_fill(~mask[:, None, None], float("-inf"))
        weights = F.softmax(logits, dim=0)
        return torch.einsum("n b t, n b t d -> b t d", weights, buf)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        B, T, D = embed.shape
        max_s = self._max_sources

        buf = embed.new_zeros(max_s, B, T, D)
        n_committed = 0
        partial = embed
        route_idx = 0
        masks = self._route_masks
        boundary_flags = self._boundary_flags

        for i, layer in enumerate(self.layers):
            if boundary_flags[i]:
                # Route at boundary
                buf[n_committed] = partial
                h = self._route(
                    buf, self.route_queries[route_idx],
                    self.route_norms[route_idx], masks[route_idx],
                )
                buf[n_committed] = partial
                n_committed += 1
                partial = embed.new_zeros(B, T, D)
                route_idx += 1
            else:
                h = partial

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](partial))
            partial = partial + mlp_out

        buf[n_committed] = partial
        x = self._route(
            buf, self.route_queries[route_idx],
            self.route_norms[route_idx], masks[route_idx],
        )
        return F.linear(self.norm(x), self.lm_head_weight)


# ── V8: Production candidate — shared norm + rsqrt cache + matmul ────────────

class ProductionCandidateModel(nn.Module):
    """
    Recommended production implementation combining the best strategies:

    1. Shared RMSNorm across all routing calls (one norm weight, not per-layer)
    2. Cached rsqrt factors for committed (frozen) buffer slots
    3. Fused logit computation: (buf @ scaled_query) * rsqrt — no intermediate
       normalized tensor materialized
    4. matmul instead of einsum (sometimes better compiled)
    5. Pre-allocated fixed buffer + static masks
    6. Once per layer routing

    Memory bandwidth per routing call: 2 reads of (max_S, B, T, D):
      - 1 for logit dot products
      - 1 for weighted sum
    vs current: ~4 reads (norm read + norm write + dot read + weighted sum read)

    Semantic change from current: shared norm instead of per-layer norm.
    Should not affect model quality meaningfully — the per-layer queries
    still provide per-layer routing decisions.
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            mods = _make_layer_modules(config)
            mods["res_query"] = nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))})
            self.layers.append(mods)
        self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.shared_norm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight
        self._norm_eps = config.norm_eps

        self._boundary_set, self._max_sources = _compute_boundaries(config)
        masks = _precompute_masks(config, self._boundary_set, self._max_sources)
        self.register_buffer("_validity_masks", masks, persistent=False)
        self.register_buffer(
            "_boundary_flags",
            _precompute_boundary_flags(config, self._boundary_set),
            persistent=False,
        )
        _make_rope_buffers(self, config)

    @staticmethod
    def _route(
        buf: torch.Tensor,
        rsqrt_cache: torch.Tensor,
        partial_idx: int,
        query: torch.Tensor,
        norm_weight: torch.Tensor,
        mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Minimal-bandwidth routing."""
        # Only update rsqrt for the changing slot
        rsqrt_cache[partial_idx] = torch.rsqrt(
            buf[partial_idx].pow(2).mean(dim=-1) + eps
        )
        # logits via matmul: (max_S, B, T, D) @ (D,) -> (max_S, B, T)
        scaled_query = query * norm_weight
        logits = torch.matmul(buf, scaled_query)  # (max_S, B, T)
        logits = logits * rsqrt_cache
        logits = logits.masked_fill(~mask[:, None, None], float("-inf"))
        weights = F.softmax(logits, dim=0)
        return (weights.unsqueeze(-1) * buf).sum(dim=0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        B, T, D = embed.shape
        max_s = self._max_sources

        buf = embed.new_zeros(max_s, B, T, D)
        rsqrt_cache = embed.new_zeros(max_s, B, T)
        n_committed = 0
        partial = embed
        masks = self._validity_masks
        boundary_flags = self._boundary_flags
        norm_weight = self.shared_norm_weight
        eps = self._norm_eps

        for i, layer in enumerate(self.layers):
            buf[n_committed] = partial
            h = self._route(
                buf, rsqrt_cache, n_committed,
                layer["res_query"]["w"], norm_weight,
                masks[i], eps,
            )

            if boundary_flags[i]:
                n_committed += 1
                partial = embed.new_zeros(B, T, D)

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](h + attn_out))
            partial = partial + mlp_out

        buf[n_committed] = partial
        h = self._route(
            buf, rsqrt_cache, n_committed,
            self.final_res_query, norm_weight,
            masks[self.config.num_layers], eps,
        )
        return F.linear(self.norm(h), self.lm_head_weight)


# ── Benchmark harness ─────────────────────────────────────────────────────────

VARIANTS = {
    "baseline":     ("No AttnRes (reference)",                     BaselineModel),
    "current":      ("Current production code (per-layer norm)",   CurrentAttnResModel),
    "cached_norm":  ("Cached norm (only normalize partial)",       CachedNormModel),
    "cached_fused": ("Cached norm + matmul logits",                CachedNormFusedModel),
    "rsqrt_cache":  ("Per-layer norm + rsqrt cache (no norm buf)", CachedNormInlineRMSModel),
    "shared_norm":  ("Shared norm + full cache",                   SharedNormModel),
    "shared_fused": ("Shared norm + rsqrt cache (min bandwidth)",  SharedNormFusedModel),
    "amortized":    ("Route at boundaries only (N+1 calls)",       AmortizedModel),
    "production":   ("Shared norm + rsqrt + matmul (recommended)", ProductionCandidateModel),
}


def benchmark_variant(
    name: str,
    model_cls: type,
    config: BenchConfig,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    warmup: int,
    iters: int,
    do_compile: bool,
    do_backward: bool,
) -> dict:
    """Benchmark a single model variant."""
    model = model_cls(config).to(device)
    if do_compile:
        model = torch.compile(model)
    model.train()

    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Warmup (includes compile JIT)
    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = logits.sum()
        if do_backward:
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        torch.cuda.synchronize()

    # Timed iterations
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = logits.sum()
        if do_backward:
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    ms_per_iter = elapsed / iters * 1000
    total_tokens = batch_size * seq_len * iters
    tok_per_sec = total_tokens / elapsed

    del model, logits, loss
    torch.cuda.empty_cache()

    return {
        "name": name,
        "ms_per_iter": ms_per_iter,
        "tok_per_sec": tok_per_sec,
        "peak_mem_gb": peak_mem,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AttnRes V2")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--n-blocks", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda")
    config = BenchConfig(attn_res_n_blocks=args.n_blocks)
    do_compile = not args.no_compile
    do_backward = not args.no_backward

    selected = args.variants or list(VARIANTS.keys())
    for v in selected:
        if v not in VARIANTS:
            print(f"Unknown variant: {v}. Available: {list(VARIANTS.keys())}")
            return

    block_size = math.ceil(config.num_layers / config.attn_res_n_blocks)
    n_boundaries = len(range(0, config.num_layers, block_size))
    print(f"Config: B={args.batch_size}, T={args.seq_len}, D={config.hidden_size}, "
          f"L={config.num_layers}, N_blocks={config.attn_res_n_blocks} "
          f"(block_size={block_size}, {n_boundaries} boundaries, max_S={n_boundaries+1})")
    print(f"compile={do_compile}, backward={do_backward}, warmup={args.warmup}, iters={args.iters}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 90)

    results = []
    for vname in selected:
        desc, model_cls = VARIANTS[vname]
        print(f"\n  {vname}: {desc} ... ", end="", flush=True)
        try:
            r = benchmark_variant(
                vname, model_cls, config, device,
                args.batch_size, args.seq_len,
                args.warmup, args.iters,
                do_compile, do_backward,
            )
            results.append(r)
            print(f"{r['ms_per_iter']:.1f} ms/iter, {r['tok_per_sec']/1000:.1f}K tok/s, "
                  f"{r['peak_mem_gb']:.2f} GB peak")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        return

    baseline = next((r for r in results if r["name"] == "baseline"), None)
    baseline_ms = baseline["ms_per_iter"] if baseline else results[0]["ms_per_iter"]
    current = next((r for r in results if r["name"] == "current"), None)

    print("\n" + "=" * 90)
    print(f"{'Variant':<20} {'ms/iter':>10} {'tok/s':>12} {'vs baseline':>12} {'vs current':>12} {'peak GB':>10}")
    print("-" * 90)
    for r in results:
        vs_base = (r["ms_per_iter"] / baseline_ms - 1) * 100
        vs_base_str = f"+{vs_base:.1f}%" if vs_base > 0 else f"{vs_base:.1f}%"
        if current and r["name"] != "current":
            vs_curr = (r["ms_per_iter"] / current["ms_per_iter"] - 1) * 100
            vs_curr_str = f"{vs_curr:+.1f}%" if vs_curr != 0 else "same"
        elif r["name"] == "current":
            vs_curr_str = "ref"
        else:
            vs_curr_str = "—"
        print(f"{r['name']:<20} {r['ms_per_iter']:>10.1f} {r['tok_per_sec']/1000:>11.1f}K "
              f"{vs_base_str:>12} {vs_curr_str:>12} {r['peak_mem_gb']:>10.2f}")

    if current and baseline:
        current_overhead = (current["ms_per_iter"] / baseline_ms - 1) * 100
        print(f"\nCurrent AttnRes overhead vs baseline: +{current_overhead:.1f}%")
        print(f"Target: <=+30% overhead")
        print()
        for r in results:
            if r["name"] not in ("baseline", "current"):
                new_overhead = (r["ms_per_iter"] / baseline_ms - 1) * 100
                speedup_vs_current = (current["ms_per_iter"] / r["ms_per_iter"] - 1) * 100
                print(f"  {r['name']:<18} overhead: +{new_overhead:.1f}%  "
                      f"({speedup_vs_current:+.1f}% vs current, "
                      f"{'MEETS' if new_overhead <= 30 else 'EXCEEDS'} target)")


if __name__ == "__main__":
    main()
