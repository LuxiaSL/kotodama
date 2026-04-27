"""
Benchmark Block Attention Residual optimizations.

Tests multiple strategies for reducing the 54% overhead of AttnRes routing.
Runs each variant through warmup + timed iterations, reports tok/s and relative overhead.

Usage:
    python scripts/bench_attnres.py                    # all variants, proxy model
    python scripts/bench_attnres.py --variants v0 v3   # specific variants
    python scripts/bench_attnres.py --no-compile        # without torch.compile
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


# ── Shared components (copied from llama.py to keep self-contained) ──────────

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


# ── V0: Baseline (no AttnRes) ───────────────────────────────────────────────

class BaselineModel(nn.Module):
    """Standard residual stream — no routing."""

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                "attn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "attn": GQAttention(config),
                "ffn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "ffn": SwiGLUFFN(config),
            }))
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight  # tied
        rope_cos, rope_sin = precompute_rope_frequencies(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer["attn"](layer["attn_norm"](x), self.rope_cos, self.rope_sin)
            x = x + layer["ffn"](layer["ffn_norm"](x))
        x = self.norm(x)
        return F.linear(x, self.lm_head_weight)


# ── V1: Current AttnRes (from llama.py) ─────────────────────────────────────

class CurrentAttnResModel(nn.Module):
    """Current implementation: per-call routing, torch.cat for growing committed."""

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                "attn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "attn": GQAttention(config),
                "ffn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "ffn": SwiGLUFFN(config),
                "attn_res_query": nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))}),
                "attn_res_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "mlp_res_query": nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))}),
                "mlp_res_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
            }))
        self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight

        block_size = math.ceil(config.num_layers / config.attn_res_n_blocks)
        self._boundary_set = frozenset(range(0, config.num_layers, block_size))

        rope_cos, rope_sin = precompute_rope_frequencies(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    @staticmethod
    def _route(sources: torch.Tensor, query: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        K = norm(sources)
        logits = torch.einsum("d, n b t d -> n b t", query, K)
        weights = F.softmax(logits, dim=0)
        return torch.einsum("n b t, n b t d -> b t d", weights, sources)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        committed: Optional[torch.Tensor] = None
        partial = embed

        for i, layer in enumerate(self.layers):
            p = partial.unsqueeze(0)
            sources = torch.cat([committed, p], dim=0) if committed is not None else p
            h = self._route(sources, layer["attn_res_query"]["w"], layer["attn_res_norm"])

            if i in self._boundary_set:
                frozen = partial.unsqueeze(0)
                committed = torch.cat([committed, frozen], dim=0) if committed is not None else frozen
                partial = torch.zeros_like(embed)

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out

            p = partial.unsqueeze(0)
            sources = torch.cat([committed, p], dim=0) if committed is not None else p
            h = self._route(sources, layer["mlp_res_query"]["w"], layer["mlp_res_norm"])

            mlp_out = layer["ffn"](layer["ffn_norm"](h))
            partial = partial + mlp_out

        p = partial.unsqueeze(0)
        sources = torch.cat([committed, p], dim=0) if committed is not None else p
        x = self._route(sources, self.final_res_query, self.final_res_norm)
        return F.linear(self.norm(x), self.lm_head_weight)


# ── V2: Pre-allocated buffer + single-route-per-layer ───────────────────────

class PreallocOncePerLayerModel(nn.Module):
    """
    Optimization strategy:
    1. Pre-allocate a fixed (max_S, B, T, D) buffer for sources — no torch.cat.
    2. Route ONCE per layer (pre-attention only) instead of twice.
       The MLP receives the same routed h as attention.
       Halves routing calls from 57 to 29.
    3. Use a validity mask instead of variable-size slicing.
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                "attn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "attn": GQAttention(config),
                "ffn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "ffn": SwiGLUFFN(config),
                "res_query": nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))}),
                "res_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
            }))
        self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight

        block_size = math.ceil(config.num_layers / config.attn_res_n_blocks)
        self._boundary_set = frozenset(range(0, config.num_layers, block_size))
        self._max_sources = config.attn_res_n_blocks + 1  # max committed blocks + partial

        rope_cos, rope_sin = precompute_rope_frequencies(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    @staticmethod
    def _route_masked(
        buf: torch.Tensor,
        query: torch.Tensor,
        norm: nn.Module,
        n_valid: int,
    ) -> torch.Tensor:
        """Route over the first n_valid entries of pre-allocated buf."""
        sources = buf[:n_valid]  # view, no copy
        K = norm(sources)
        logits = torch.einsum("d, n b t d -> n b t", query, K)
        weights = F.softmax(logits, dim=0)
        return torch.einsum("n b t, n b t d -> b t d", weights, sources)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        B, T, D = embed.shape
        max_s = self._max_sources

        # Pre-allocate source buffer
        buf = torch.zeros(max_s, B, T, D, device=embed.device, dtype=embed.dtype)
        n_committed = 0
        partial = embed

        for i, layer in enumerate(self.layers):
            # Write partial into buffer slot
            buf[n_committed] = partial
            n_valid = n_committed + 1

            # Single route per layer (pre-attention)
            h = self._route_masked(buf, layer["res_query"]["w"], layer["res_norm"], n_valid)

            # Boundary: freeze partial into committed
            if i in self._boundary_set:
                buf[n_committed] = partial  # already there, but explicit
                n_committed += 1
                partial = torch.zeros_like(embed)

            # Attention + MLP both use routed h
            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](h + attn_out))
            partial = partial + mlp_out

        # Final routing
        buf[n_committed] = partial
        x = self._route_masked(buf, self.final_res_query, self.final_res_norm, n_committed + 1)
        return F.linear(self.norm(x), self.lm_head_weight)


# ── V3: Padded fixed-size routing (compile-friendly) ────────────────────────

class PaddedFixedSizeModel(nn.Module):
    """
    Optimization strategy:
    1. Always route over ALL max_S slots, with unused slots zeroed out.
    2. Use masked_fill on logits (set unused to -inf before softmax)
       so softmax produces zero weight for unused slots.
    3. Fixed tensor shapes throughout → torch.compile can fully trace
       with no graph breaks and generate a single fused kernel.
    4. Route once per layer.
    5. No torch.cat anywhere.
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                "attn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "attn": GQAttention(config),
                "ffn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "ffn": SwiGLUFFN(config),
                "res_query": nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))}),
                "res_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
            }))
        self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight

        block_size = math.ceil(config.num_layers / config.attn_res_n_blocks)
        self._boundary_set = frozenset(range(0, config.num_layers, block_size))
        self._max_sources = config.attn_res_n_blocks + 1

        rope_cos, rope_sin = precompute_rope_frequencies(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        # Pre-compute per-layer validity masks as registered buffers.
        # mask shape: (max_S,) with True for valid slots, False for padding.
        # This is static and known at init time.
        n_committed = 0
        masks_attn = []
        for i in range(config.num_layers):
            n_valid = n_committed + 1  # committed + partial
            mask = torch.zeros(self._max_sources, dtype=torch.bool)
            mask[:n_valid] = True
            masks_attn.append(mask)
            if i in self._boundary_set:
                n_committed += 1

        # Final mask
        final_mask = torch.zeros(self._max_sources, dtype=torch.bool)
        final_mask[:n_committed + 1] = True

        # Stack all masks into a single tensor: (num_layers + 1, max_S)
        all_masks = torch.stack(masks_attn + [final_mask], dim=0)
        self.register_buffer("_validity_masks", all_masks, persistent=False)

    @staticmethod
    def _route_padded(
        buf: torch.Tensor,
        query: torch.Tensor,
        norm: nn.Module,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Route over full buffer with masking for unused slots.
        buf: (max_S, B, T, D)
        mask: (max_S,) bool — True for valid slots
        """
        K = norm(buf)  # (max_S, B, T, D) — always same shape
        logits = torch.einsum("d, n b t d -> n b t", query, K)  # (max_S, B, T)
        # Mask out invalid slots with -inf so softmax gives them zero weight
        logits = logits.masked_fill(~mask[:, None, None], float("-inf"))
        weights = F.softmax(logits, dim=0)  # (max_S, B, T)
        # NaN guard: if all slots are masked (shouldn't happen), weights will be nan
        # In practice n_valid >= 1 always, so this is safe
        return torch.einsum("n b t, n b t d -> b t d", weights, buf)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        B, T, D = embed.shape
        max_s = self._max_sources

        buf = torch.zeros(max_s, B, T, D, device=embed.device, dtype=embed.dtype)
        n_committed = 0
        partial = embed
        masks = self._validity_masks  # (num_layers+1, max_S)

        for i, layer in enumerate(self.layers):
            buf[n_committed] = partial
            h = self._route_padded(buf, layer["res_query"]["w"], layer["res_norm"], masks[i])

            if i in self._boundary_set:
                buf[n_committed] = partial
                n_committed += 1
                partial = torch.zeros_like(embed)

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](h + attn_out))
            partial = partial + mlp_out

        buf[n_committed] = partial
        x = self._route_padded(buf, self.final_res_query, self.final_res_norm, masks[self.config.num_layers])
        return F.linear(self.norm(x), self.lm_head_weight)


# ── V4: Batched queries + padded buffer ─────────────────────────────────────

class BatchedQueriesModel(nn.Module):
    """
    Optimization strategy:
    1. Pre-allocate fixed buffer like V3.
    2. Route once per layer like V2/V3.
    3. BATCH the per-layer queries: instead of 29 separate einsum calls,
       group layers that share the same n_valid into batched operations.

       Within each block of layers (same committed count), all routing calls
       have the same mask. We can batch them.

       For N=7: blocks of 4 layers each = 7 batches of 4 routing calls.
       Each batch: queries (4, D), sources (S, B, T, D)
       → logits = einsum("l d, n b t d -> l n b t")
       → softmax per l → weighted sum per l → outputs (4, B, T, D)

       This reduces kernel launches from 29 * 4 ops to 7 * 4 ops (plus some reshaping).
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                "attn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "attn": GQAttention(config),
                "ffn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "ffn": SwiGLUFFN(config),
                "res_query": nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))}),
                "res_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
            }))
        self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight

        block_size = math.ceil(config.num_layers / config.attn_res_n_blocks)
        self._boundary_set = frozenset(range(0, config.num_layers, block_size))
        self._max_sources = config.attn_res_n_blocks + 1
        self._block_size = block_size

        rope_cos, rope_sin = precompute_rope_frequencies(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        # Pre-compute masks (same as V3)
        n_committed = 0
        masks_attn = []
        for i in range(config.num_layers):
            n_valid = n_committed + 1
            mask = torch.zeros(self._max_sources, dtype=torch.bool)
            mask[:n_valid] = True
            masks_attn.append(mask)
            if i in self._boundary_set:
                n_committed += 1
        final_mask = torch.zeros(self._max_sources, dtype=torch.bool)
        final_mask[:n_committed + 1] = True
        all_masks = torch.stack(masks_attn + [final_mask], dim=0)
        self.register_buffer("_validity_masks", all_masks, persistent=False)

    @staticmethod
    def _route_padded(
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
        max_s = self._max_sources

        buf = torch.zeros(max_s, B, T, D, device=embed.device, dtype=embed.dtype)
        n_committed = 0
        partial = embed
        masks = self._validity_masks

        for i, layer in enumerate(self.layers):
            buf[n_committed] = partial
            h = self._route_padded(buf, layer["res_query"]["w"], layer["res_norm"], masks[i])

            if i in self._boundary_set:
                buf[n_committed] = partial
                n_committed += 1
                partial = torch.zeros_like(embed)

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](h + attn_out))
            partial = partial + mlp_out

        buf[n_committed] = partial
        x = self._route_padded(buf, self.final_res_query, self.final_res_norm, masks[self.config.num_layers])
        return F.linear(self.norm(x), self.lm_head_weight)


# ── V5: Fused routing kernel (inline norm + dot + softmax + weighted sum) ───

class FusedRoutingModel(nn.Module):
    """
    Optimization strategy:
    1. Fuse the entire routing operation into a single function that
       torch.compile can optimize as one unit.
    2. Pre-allocate fixed buffer + masks like V3.
    3. Route once per layer.
    4. Replace RMSNorm(sources) -> einsum -> softmax -> einsum with a
       single fused function that does all ops inline without materializing
       the full normalized sources tensor.

       Key insight: we don't need to store the full (max_S, B, T, D) normalized
       tensor. We only need the dot products (max_S, B, T) and the weighted sum.
       We can compute: logit_s = dot(query, rms_norm(sources[s])) for each s,
       then softmax, then weighted sum. But that's sequential over S.

       Better: compute the RMS norm factor per slot (scalar per position),
       compute logits = (sources @ query) / rms_norm_factor, then softmax + weighted sum.
       This avoids materializing the full normalized tensor.
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                "attn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "attn": GQAttention(config),
                "ffn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "ffn": SwiGLUFFN(config),
                "res_query": nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))}),
                # Store norm weight separately for fused routing
                "res_norm_weight": nn.ParameterDict({"w": nn.Parameter(torch.ones(config.hidden_size))}),
            }))
        self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_res_norm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight
        self._norm_eps = config.norm_eps

        block_size = math.ceil(config.num_layers / config.attn_res_n_blocks)
        self._boundary_set = frozenset(range(0, config.num_layers, block_size))
        self._max_sources = config.attn_res_n_blocks + 1

        rope_cos, rope_sin = precompute_rope_frequencies(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        n_committed = 0
        masks_attn = []
        for i in range(config.num_layers):
            n_valid = n_committed + 1
            mask = torch.zeros(self._max_sources, dtype=torch.bool)
            mask[:n_valid] = True
            masks_attn.append(mask)
            if i in self._boundary_set:
                n_committed += 1
        final_mask = torch.zeros(self._max_sources, dtype=torch.bool)
        final_mask[:n_committed + 1] = True
        all_masks = torch.stack(masks_attn + [final_mask], dim=0)
        self.register_buffer("_validity_masks", all_masks, persistent=False)

    @staticmethod
    def _fused_route(
        buf: torch.Tensor,
        query: torch.Tensor,
        norm_weight: torch.Tensor,
        mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Fused routing: combines RMSNorm + dot product + softmax + weighted sum.

        Instead of:
            K = norm(buf)                  # materializes (S, B, T, D)
            logits = einsum(query, K)      # (S, B, T)
            weights = softmax(logits)      # (S, B, T)
            out = einsum(weights, buf)     # (B, T, D)

        We do:
            # Compute logits directly: query @ (buf * norm_weight / rms)
            # = (query * norm_weight) @ buf / rms
            # Since query and norm_weight are (D,), we can pre-multiply them
            scaled_query = query * norm_weight  # (D,)
            raw_dots = einsum(scaled_query, buf) # (S, B, T) — before /rms
            rms = sqrt(mean(buf^2, dim=-1) + eps)  # (S, B, T)
            logits = raw_dots / rms
            logits = masked_fill(-inf for invalid)
            weights = softmax(logits)
            out = einsum(weights, buf)

        This avoids materializing the full (S, B, T, D) normalized tensor.
        Memory: saves one (S, B, T, D) read+write.
        """
        # Pre-multiply query with norm weight
        scaled_query = query * norm_weight  # (D,)

        # Compute raw dot products: (S, B, T)
        raw_dots = torch.einsum("d, n b t d -> n b t", scaled_query, buf)

        # Compute RMS normalization factor: (S, B, T)
        rms = torch.rsqrt(buf.pow(2).mean(dim=-1) + eps)  # (S, B, T)

        # Combine into logits
        logits = raw_dots * rms

        # Mask and softmax
        logits = logits.masked_fill(~mask[:, None, None], float("-inf"))
        weights = F.softmax(logits, dim=0)  # (S, B, T)

        # Weighted sum over original (un-normalized) sources
        return torch.einsum("n b t, n b t d -> b t d", weights, buf)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        B, T, D = embed.shape
        max_s = self._max_sources

        buf = torch.zeros(max_s, B, T, D, device=embed.device, dtype=embed.dtype)
        n_committed = 0
        partial = embed
        masks = self._validity_masks
        eps = self._norm_eps

        for i, layer in enumerate(self.layers):
            buf[n_committed] = partial
            h = self._fused_route(
                buf, layer["res_query"]["w"], layer["res_norm_weight"]["w"],
                masks[i], eps,
            )

            if i in self._boundary_set:
                buf[n_committed] = partial
                n_committed += 1
                partial = torch.zeros_like(embed)

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](h + attn_out))
            partial = partial + mlp_out

        buf[n_committed] = partial
        x = self._fused_route(
            buf, self.final_res_query, self.final_res_norm_weight,
            masks[self.config.num_layers], eps,
        )
        return F.linear(self.norm(x), self.lm_head_weight)


# ── V6: Route every K layers (amortized) ────────────────────────────────────

class AmortizedRoutingModel(nn.Module):
    """
    Optimization strategy:
    1. Route only at block boundaries (every block_size layers) instead of every layer.
    2. Between routing calls, use standard residual connections.
    3. This reduces routing from 29 calls to N+1 calls (8 for N=7).
    4. Pre-allocate buffer + masks like V3.

    Tradeoff: layers within a block don't get per-layer routing decisions.
    The routed representation is computed once at the start of each block and
    used as the input for all layers in that block.
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                "attn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "attn": GQAttention(config),
                "ffn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "ffn": SwiGLUFFN(config),
            }))

        block_size = math.ceil(config.num_layers / config.attn_res_n_blocks)
        self._boundary_set = frozenset(range(0, config.num_layers, block_size))
        self._max_sources = config.attn_res_n_blocks + 1

        # Routing parameters: one query + norm per block boundary + final
        self.route_queries = nn.ParameterList([
            nn.Parameter(torch.zeros(config.hidden_size))
            for _ in range(config.attn_res_n_blocks + 1)  # N boundaries + 1 final
        ])
        self.route_norms = nn.ModuleList([
            RMSNorm(config.hidden_size, eps=config.norm_eps)
            for _ in range(config.attn_res_n_blocks + 1)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight

        rope_cos, rope_sin = precompute_rope_frequencies(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        # Pre-compute masks
        n_committed = 0
        masks = []
        for i in range(config.num_layers):
            if i in self._boundary_set:
                n_valid = n_committed + 1
                mask = torch.zeros(self._max_sources, dtype=torch.bool)
                mask[:n_valid] = True
                masks.append(mask)
                n_committed += 1
        # Final mask
        final_mask = torch.zeros(self._max_sources, dtype=torch.bool)
        final_mask[:n_committed + 1] = True
        masks.append(final_mask)
        all_masks = torch.stack(masks, dim=0)
        self.register_buffer("_validity_masks", all_masks, persistent=False)

    @staticmethod
    def _route_padded(
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

        buf = torch.zeros(max_s, B, T, D, device=embed.device, dtype=embed.dtype)
        n_committed = 0
        partial = embed
        route_idx = 0
        masks = self._validity_masks

        for i, layer in enumerate(self.layers):
            # Route only at block boundaries
            if i in self._boundary_set:
                buf[n_committed] = partial
                h = self._route_padded(
                    buf, self.route_queries[route_idx],
                    self.route_norms[route_idx], masks[route_idx],
                )
                buf[n_committed] = partial
                n_committed += 1
                partial = torch.zeros_like(embed)
                route_idx += 1
            else:
                h = partial  # no routing, use partial directly

            # Standard residual within block
            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](partial))
            partial = partial + mlp_out

        # Final routing
        buf[n_committed] = partial
        x = self._route_padded(
            buf, self.route_queries[route_idx],
            self.route_norms[route_idx], masks[route_idx],
        )
        return F.linear(self.norm(x), self.lm_head_weight)


# ── V7: Fused + padded + once-per-layer (V5 semantics with V3 structure) ───

class FusedPaddedOnceModel(nn.Module):
    """
    Combined best ideas: fused routing (no intermediate tensor) +
    padded fixed-size buffer + once-per-layer + pre-computed masks.

    This is the recommended implementation for production use.
    """

    def __init__(self, config: BenchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                "attn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "attn": GQAttention(config),
                "ffn_norm": RMSNorm(config.hidden_size, eps=config.norm_eps),
                "ffn": SwiGLUFFN(config),
                # Fused routing: store query * norm_weight pre-multiplied
                "res_scaled_query": nn.ParameterDict({"w": nn.Parameter(torch.zeros(config.hidden_size))}),
            }))
        self.final_scaled_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head_weight = self.embed.weight
        self._norm_eps = config.norm_eps

        block_size = math.ceil(config.num_layers / config.attn_res_n_blocks)
        self._boundary_set = frozenset(range(0, config.num_layers, block_size))
        self._max_sources = config.attn_res_n_blocks + 1

        rope_cos, rope_sin = precompute_rope_frequencies(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        n_committed = 0
        masks_attn = []
        for i in range(config.num_layers):
            n_valid = n_committed + 1
            mask = torch.zeros(self._max_sources, dtype=torch.bool)
            mask[:n_valid] = True
            masks_attn.append(mask)
            if i in self._boundary_set:
                n_committed += 1
        final_mask = torch.zeros(self._max_sources, dtype=torch.bool)
        final_mask[:n_committed + 1] = True
        all_masks = torch.stack(masks_attn + [final_mask], dim=0)
        self.register_buffer("_validity_masks", all_masks, persistent=False)

    @staticmethod
    def _fused_route(
        buf: torch.Tensor,
        scaled_query: torch.Tensor,
        mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Minimal fused routing. The scaled_query already incorporates the
        norm weight (query * norm_weight), so we just need rsqrt for RMS.

        NOTE: This merges query and norm_weight into one parameter.
        At init, query=0 and norm_weight=1, so scaled_query=0 → uniform routing.
        During training, the optimizer updates scaled_query directly.
        The decomposition into query*weight is lost, but the function
        query_direction * norm_scaling is still expressible.
        """
        # Dot product with pre-scaled query
        raw_dots = torch.einsum("d, n b t d -> n b t", scaled_query, buf)
        # RMS normalization factor
        rms = torch.rsqrt(buf.pow(2).mean(dim=-1) + eps)
        logits = raw_dots * rms
        logits = logits.masked_fill(~mask[:, None, None], float("-inf"))
        weights = F.softmax(logits, dim=0)
        return torch.einsum("n b t, n b t d -> b t d", weights, buf)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embed = self.embed(input_ids)
        B, T, D = embed.shape
        max_s = self._max_sources

        buf = torch.zeros(max_s, B, T, D, device=embed.device, dtype=embed.dtype)
        n_committed = 0
        partial = embed
        masks = self._validity_masks
        eps = self._norm_eps

        for i, layer in enumerate(self.layers):
            buf[n_committed] = partial
            h = self._fused_route(buf, layer["res_scaled_query"]["w"], masks[i], eps)

            if i in self._boundary_set:
                buf[n_committed] = partial
                n_committed += 1
                partial = torch.zeros_like(embed)

            attn_out = layer["attn"](layer["attn_norm"](h), self.rope_cos, self.rope_sin)
            partial = partial + attn_out
            mlp_out = layer["ffn"](layer["ffn_norm"](h + attn_out))
            partial = partial + mlp_out

        buf[n_committed] = partial
        x = self._fused_route(buf, self.final_scaled_query, masks[self.config.num_layers], eps)
        return F.linear(self.norm(x), self.lm_head_weight)


# ── Benchmark harness ───────────────────────────────────────────────────────

VARIANTS = {
    "v0_baseline": ("No AttnRes (baseline)", BaselineModel),
    "v1_current": ("Current AttnRes (2x route, torch.cat)", CurrentAttnResModel),
    "v2_prealloc_once": ("Pre-alloc + 1x route per layer", PreallocOncePerLayerModel),
    "v3_padded_fixed": ("Padded fixed-size + masks + 1x route", PaddedFixedSizeModel),
    "v4_batched_queries": ("Batched queries + padded (same as V3, batching TBD)", BatchedQueriesModel),
    "v5_fused": ("Fused norm+dot (no intermediate tensor)", FusedRoutingModel),
    "v6_amortized": ("Route at boundaries only (8 calls total)", AmortizedRoutingModel),
    "v7_fused_padded": ("Fused + padded + once-per-layer (production)", FusedPaddedOnceModel),
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
            loss = logits.sum()  # dummy loss for backward
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

    # Cleanup
    del model, logits, loss
    torch.cuda.empty_cache()

    return {
        "name": name,
        "ms_per_iter": ms_per_iter,
        "tok_per_sec": tok_per_sec,
        "peak_mem_gb": peak_mem,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AttnRes variants")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Specific variants to test (default: all)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-backward", action="store_true",
                        help="Forward only (skip backward)")
    parser.add_argument("--n-blocks", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda")
    config = BenchConfig(attn_res_n_blocks=args.n_blocks)
    do_compile = not args.no_compile
    do_backward = not args.no_backward

    selected = args.variants or list(VARIANTS.keys())
    # Validate
    for v in selected:
        if v not in VARIANTS:
            print(f"Unknown variant: {v}. Available: {list(VARIANTS.keys())}")
            return

    print(f"Benchmark config: B={args.batch_size}, T={args.seq_len}, "
          f"D={config.hidden_size}, L={config.num_layers}, N_blocks={config.attn_res_n_blocks}")
    print(f"compile={do_compile}, backward={do_backward}, "
          f"warmup={args.warmup}, iters={args.iters}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

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

    # Summary table
    baseline = next((r for r in results if r["name"] == "v0_baseline"), None)
    baseline_ms = baseline["ms_per_iter"] if baseline else results[0]["ms_per_iter"]

    print("\n" + "=" * 80)
    print(f"{'Variant':<25} {'ms/iter':>10} {'tok/s':>12} {'overhead':>10} {'peak GB':>10}")
    print("-" * 80)
    for r in results:
        overhead = (r["ms_per_iter"] / baseline_ms - 1) * 100
        overhead_str = f"+{overhead:.1f}%" if overhead > 0 else f"{overhead:.1f}%"
        print(f"{r['name']:<25} {r['ms_per_iter']:>10.1f} {r['tok_per_sec']/1000:>11.1f}K "
              f"{overhead_str:>10} {r['peak_mem_gb']:>10.2f}")

    # Reduction from current AttnRes
    current = next((r for r in results if r["name"] == "v1_current"), None)
    if current and baseline:
        current_overhead = (current["ms_per_iter"] / baseline_ms - 1) * 100
        print(f"\nCurrent AttnRes overhead: +{current_overhead:.1f}%")
        for r in results:
            if r["name"] not in ("v0_baseline", "v1_current"):
                new_overhead = (r["ms_per_iter"] / baseline_ms - 1) * 100
                reduction = current_overhead - new_overhead
                print(f"  {r['name']}: overhead reduced to +{new_overhead:.1f}% "
                      f"(saved {reduction:.1f}pp of {current_overhead:.1f}pp)")


if __name__ == "__main__":
    main()
