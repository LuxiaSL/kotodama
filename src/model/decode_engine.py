"""Static-shape single-stream decode engine for LuxiaBaseModel (AttnRes).

Serving-only fast path. Borrows weight tensors from a loaded LuxiaBaseModel —
llama.py remains the training/reference implementation and the parity oracle.

Design notes
------------
The decode step is written to be torch.compile + CUDA-graph friendly:

  * All persistent state (KV cache, attention bias, position, sampling
    presence) lives in registered nn.Module buffers → static addresses,
    so inductor's cudagraph trees accept the in-place mutations.
  * KV cache: (n_layers, B, n_kv_heads, max_seq_len, head_dim), written in
    place at position `pos` (a CUDA tensor — never a Python int).
  * Attention: SDPA over the FULL cache length with an additive mask that
    opens one position per step. Shape-static.
  * AttnRes routing is FUNCTIONAL: committed block outputs and the running
    partial are plain locals (mirroring llama._forward_attn_res_cached
    exactly, minus the padding/stacking — source counts per routing call
    site are compile-time constants, so every site unrolls into its own
    fused kernel). No buffer mutation → no functionalization blowups.
  * Sampling: on-GPU, matching serve.py semantics — presence-based
    repetition penalty over GENERATED tokens only, pure temperature
    (top_k/top_p intentionally unsupported: prod runs 0/0).
  * Steering (optional, site fixed at construction): a registered steer_vec
    buffer added to the running `partial` after attention at the site layer
    and at every later block entry (block-persistent write, the koto
    primitive — single-site writes die at DD-3B boundaries). Decode steps
    only; prefill/extend never inject. Zeros = off; the add is baked into
    an enabled engine's graphs so set_steer() never changes graph structure.
    docs/STEERING-SERVE.md has the API + the GPU gates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.model.gemv_kernels import decode_gemv, use_custom_gemv
from src.model.llama import LuxiaBaseModel, apply_rope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerSchedule:
    """Per-layer static schedule: is this layer an AttnRes block boundary?"""

    layer_idx: int
    is_boundary: bool


def compute_steer_points(steer_site: int, boundaries: set[int] | list[int], n_layers: int) -> tuple[int, ...]:
    """Layer indices that receive the block-persistent steering write.

    The steering-pilot architectural result (RESULTS-steering-pilot-2026-07-21
    §3): a single-site residual add is destroyed at the next DD-3B block
    boundary (commit-and-reset + routing recombination attenuate it ~100x, KL
    0.002 vs 0.78 nats persistent), so the koto write primitive re-asserts the
    vector at the first after-attn position of every block AFTER the site's.

    Mirrors posttraining/taste/steer_inject.forward_inject persist semantics.
    Sites there are SUBLAYER indices (after-attn of layer L = sublayer 2*L);
    persist points {2*L} + {2*b : boundary b, 2*b > 2*L} reduce in layer terms
    to: the site layer itself + every boundary layer strictly after it.
    """
    if not 0 <= steer_site < n_layers:
        raise ValueError(f"steer_site {steer_site} out of range [0, {n_layers})")
    return (steer_site, *sorted(b for b in set(boundaries) if steer_site < b < n_layers))


class SamplingParams:
    """serve.py sampling controls supported by the fast path.

    top_k == 0 and top_p in {0.0, 1.0} mean "disabled" (serve.py convention).
    Truncated sampling (top_k > 0 or 0 < top_p < 1) routes through a separate
    compiled sampler variant; the pure-temperature graph is untouched.
    """

    def __init__(
        self,
        temperature: float = 0.9,
        repetition_penalty: float = 1.2,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> None:
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        self.temperature = float(temperature)
        self.repetition_penalty = float(repetition_penalty)
        self.top_k = int(top_k)
        self.top_p = float(top_p)

    @property
    def truncated(self) -> bool:
        return self.top_k > 0 or 0.0 < self.top_p < 1.0


class DecodeEngine(nn.Module):
    """Single-stream (B=1) static-shape decode engine.

    Usage:
        engine = DecodeEngine(model, max_seq_len=4096)
        engine.compile_step()                          # optional
        logits = engine.prefill(input_ids)             # eager reference prefill
        tok = engine.sample_first(logits, params)
        while ...:
            tok = engine.step(tok, params)             # static decode step

    Contracts:
      * Weights are BOUND AT CONSTRUCTION: fused QKV/gate-up and rope tables
        are copies. After mutating model weights (checkpoint hot-reload),
        rebuild the engine — a stale engine emits silently wrong logits.
      * step()/sample_first() return the persistent cur_token buffer;
        step_logits() under compile returns CUDA-graph pool memory. Both are
        valid only until the next step — consume or copy immediately.
      * Do not call engine.to()/.half() — buffer lists hold raw tensor refs.
    """

    def __init__(
        self,
        model: LuxiaBaseModel,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        prefill_backends: Optional[list[SDPBackend]] = None,
        steer_site: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not model.config.attn_res:
            raise ValueError("DecodeEngine requires an attn_res model")
        if max_seq_len is not None and max_seq_len > model.config.max_position_embeddings:
            raise ValueError(
                f"max_seq_len {max_seq_len} exceeds rope table length "
                f"{model.config.max_position_embeddings}"
            )
        self.model = model
        self.config = model.config
        self.device = device or next(model.parameters()).device
        self.dtype = dtype
        self.max_seq_len = int(max_seq_len or model.config.max_position_embeddings)
        cfg = model.config

        self.n_layers = cfg.num_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.hidden = cfg.hidden_size
        self.eps = cfg.norm_eps

        # ── Static state (registered buffers → stable addresses for graphs) ──
        # Per-layer KV buffers (NOT one stacked tensor): in-place index_copy_
        # must target a graph input directly. Writing into a slice view of a
        # stacked buffer gets functionalized by inductor into full-buffer
        # materializations (~470 MB of copies per step, measured ~1.3 ms).
        kv_shape = (1, self.n_kv_heads, self.max_seq_len, self.head_dim)
        for li in range(self.n_layers):
            self.register_buffer(f"k_cache_{li}", torch.zeros(kv_shape, device=self.device, dtype=dtype), persistent=False)
            self.register_buffer(f"v_cache_{li}", torch.zeros(kv_shape, device=self.device, dtype=dtype), persistent=False)
        self.k_caches = [getattr(self, f"k_cache_{li}") for li in range(self.n_layers)]
        self.v_caches = [getattr(self, f"v_cache_{li}") for li in range(self.n_layers)]
        self.register_buffer(
            "attn_bias",
            torch.full((1, 1, 1, self.max_seq_len), float("-inf"), device=self.device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer("pos", torch.zeros((1,), device=self.device, dtype=torch.long), persistent=False)
        self.register_buffer(
            "presence", torch.zeros((cfg.vocab_size,), device=self.device, dtype=torch.bool), persistent=False
        )
        self.register_buffer("cur_token", torch.zeros((1, 1), device=self.device, dtype=torch.long), persistent=False)
        # Token ids whose KV is in the cache: token_history[:pos] <-> k/v[:, :, :pos].
        # Written in-graph by _forward_step (consumed token at its position) and
        # by prefill/extend (prompt ids), so the invariant cannot drift from the
        # cache contents — this is what prefix matching trusts.
        self.register_buffer(
            "token_history",
            torch.zeros((self.max_seq_len,), device=self.device, dtype=torch.long),
            persistent=False,
        )
        # Below this, an extend is not worth the bookkeeping — do a full prefill.
        self.min_cached_prefix = 16

        # RoPE tables (read-only).
        self.register_buffer("rope_cos_t", model.rope_cos.to(self.device), persistent=False)
        self.register_buffer("rope_sin_t", model.rope_sin.to(self.device), persistent=False)

        # ── Static schedule from the frozen boundary list ───────────────────
        boundary_set = model._attn_res_boundary_set
        self.schedule = [LayerSchedule(i, i in boundary_set) for i in range(self.n_layers)]

        # ── Steering (block-persistent residual write; fixed at construction) ──
        # steer_site is a LAYER index: the write lands after that layer's
        # `partial = partial + attn_out`, then re-asserts at each later block
        # entry (see compute_steer_points). DECODE STEPS ONLY — prefill/extend
        # paths never inject, which gives the pilot's generated-positions-only
        # semantics for free (re-prefilled prior turns are acceptably
        # unsteered, the documented pilot choice).
        #
        # COMPILE SAFETY: the buffer is created ONCE and only ever mutated
        # in-place (copy_/zero_) — torch.compile(max-autotune)/cudagraph trees
        # specialize on tensor identity, so reassignment would silently detach
        # the graphs from the live vector. A steering-enabled engine bakes the
        # add into its graphs UNCONDITIONALLY (zeros = off), so set_steer()
        # never changes graph structure; a steering-disabled engine
        # (steer_site=None) traces an empty point set -> literally zero new
        # ops, bit-identical to a pre-steering engine.
        self.steer_site = steer_site
        self._steer_active = False
        if steer_site is not None:
            self.steer_points = compute_steer_points(steer_site, boundary_set, self.n_layers)
            self._steer_point_set = frozenset(self.steer_points)
            self.register_buffer(
                "steer_vec", torch.zeros((self.hidden,), device=self.device, dtype=dtype), persistent=False
            )
        else:
            self.steer_points = ()
            self._steer_point_set = frozenset()
            self.steer_vec = None

        # ── Fused projection weights (decode-only) ──────────────────────────
        # QKV and gate/up merges: fewer, larger GEMVs. Output rows are
        # independent dot products, so results match the unfused projections.
        # Costs ~3.7 GB extra (originals stay live for the reference prefill).
        for li, layer in enumerate(model.layers):
            w_qkv = torch.cat(
                [layer.attn.q_proj.weight, layer.attn.k_proj.weight, layer.attn.v_proj.weight], dim=0
            )
            w_gate_up = torch.cat([layer.ffn.gate_proj.weight, layer.ffn.up_proj.weight], dim=0)
            self.register_buffer(f"w_qkv_{li}", w_qkv.contiguous(), persistent=False)
            self.register_buffer(f"w_gate_up_{li}", w_gate_up.contiguous(), persistent=False)
        self.w_qkvs = [getattr(self, f"w_qkv_{li}") for li in range(self.n_layers)]
        self.w_gate_ups = [getattr(self, f"w_gate_up_{li}") for li in range(self.n_layers)]
        # NOTE (measured 2026-06-11): pre-transposing o_proj/down_proj and
        # using x @ W_t regressed decode 394 → 348 tok/s — the NN orientation
        # dispatches an even slower path than nvjet's 27%-BW TNN config.
        # The o/down GEMV inefficiency needs a custom kernel, not a layout dodge.
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        # Sampling controls as 0-dim CUDA buffers: Python floats inside the
        # compiled step would become dynamo guards → recompile per value.
        self.register_buffer("inv_temp", torch.tensor(1.0 / 0.9, device=self.device, dtype=torch.float32), persistent=False)
        self.register_buffer("rep_penalty", torch.tensor(1.2, device=self.device, dtype=torch.float32), persistent=False)
        self._cur_temp = 0.9
        self._cur_rp = 1.2
        # Truncation dials for the truncated sampler variant, as buffers so one
        # graph serves every (k, p): disabled values are algebraic no-ops
        # (k = vocab -> the kth threshold is the min logit, masks nothing;
        # p = 2.0 -> cumprob - prob >= 2.0 is never true, masks nothing) —
        # exactly the branches serve.py skips. p MUST be 2.0, not 1.0: fp32
        # cumsum saturates to exactly 1.0 partway through the tail (measured:
        # ~11k of 49k tokens masked at p=1.0), and inside a k-only law the
        # renormalized survivor cumsum can saturate the same way.
        self.register_buffer(
            "top_k_eff", torch.tensor(cfg.vocab_size, device=self.device, dtype=torch.long), persistent=False
        )
        self.register_buffer("top_p_val", torch.tensor(2.0, device=self.device, dtype=torch.float32), persistent=False)
        self._cur_top_k = 0
        self._cur_top_p = 0.0

        self._compiled_forward = None
        self._compiled_step_sampled = None
        self._compiled_step_truncated = None
        # Block-forward buckets for compiled prefill/extend, descending. One
        # compiled callable — dynamo specializes a graph per static S. The
        # eager reference prefill measured FLAT ~23-26ms at T<=1024 (launch
        # bound); compiled blocks replace it and the 33-40ms MATH extend.
        self.block_buckets: tuple[int, ...] = (256, 64, 16)
        # Blocks attend the FULL max_seq_len cache under an additive mask
        # (MATH-class SDPA in-graph), so their cost scales with chunk COUNT,
        # not span — measured 2026-07-05: 1.4-1.6x faster than eager below
        # ~768 tokens, 0.3-0.7x SLOWER above (eager gets causal flash within
        # T). Spans above this route to the eager paths. (Future exact fix:
        # rectangular-flash-to-prefix + causal-flash-within-chunk merged by
        # logsumexp — kills both the mask and the full-cache read.)
        self.block_span_max: int = 768
        self._compiled_block = None
        # CPU mirror of self.pos for bounds checking without device syncs.
        self._pos_cpu = 0
        # Prefill SDPA backend pin (see prefill()); parameterized so tests can
        # run non-bf16 models on MATH. Prod default is the validated set.
        self._prefill_backends = (
            list(prefill_backends)
            if prefill_backends
            else [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        )
        # Multi-token EXTEND needs an explicit mask, which eliminates the fast
        # backends (measured on B200/torch2.x: FLASH rejects any attn_mask,
        # EFFICIENT rejects enable_gqa, CUDNN accepts but re-plans ~300ms per
        # novel (S, L) shape — and extend shapes are novel every turn). MATH
        # costs only ~2-5ms at our suffix sizes and is shape-oblivious.
        self._extend_backends = [SDPBackend.MATH]

        gemv_sites = {
            "qkv": self.w_qkvs[0],
            "o": model.layers[0].attn.o_proj.weight,
            "gate_up": self.w_gate_ups[0],
            "down": model.layers[0].ffn.down_proj.weight,
            "lm_head": model.get_lm_head_weight(),
        }
        custom_sites = [name for name, w in gemv_sites.items() if use_custom_gemv(w)]
        logger.info(
            "DecodeEngine: %d layers, max_seq_len=%d, %d boundaries, kv cache %.0f MB, "
            "custom GEMV: %s",
            self.n_layers, self.max_seq_len, len(boundary_set),
            sum(b.numel() for b in self.k_caches) * 2 * 2 / 1e6,
            custom_sites or "off",
        )
        if self.steer_site is not None:
            logger.info(
                "DecodeEngine steering: site layer %d (after-attn sublayer s%d), "
                "block-persistent write points %s (decode steps only, zeros = off)",
                self.steer_site, 2 * self.steer_site, list(self.steer_points),
            )

    # ── Linear dispatch: custom Triton GEMV for tuned shapes ────────────────

    def _linear(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """F.linear or the custom decode GEMV, decided per weight shape.

        The branch resolves at trace time (decode shapes are static), so the
        compiled graph bakes in one path per call site with zero replay cost.
        """
        if use_custom_gemv(w):
            return decode_gemv(x, w)
        return F.linear(x, w)

    # ── Routing (functional; matches llama._route_static math exactly) ──────

    def _route(self, sources: list[torch.Tensor], query: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
        """Attention-residual routing over n sources. n is static per call site."""
        qw = query * norm_weight  # (D,)
        stacked = torch.stack(sources, dim=0)                        # (n, 1, 1, D)
        rsqrt = torch.rsqrt(stacked.pow(2).mean(-1) + self.eps)      # (n, 1, 1)
        logits = (stacked * qw).sum(-1) * rsqrt                      # (n, 1, 1)
        weights = F.softmax(logits, dim=0)
        return (weights.unsqueeze(-1) * stacked).sum(0)              # (1, 1, D)

    # ── One attention sublayer with static cache ─────────────────────────────

    def _attn(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        attn = self.model.layers[layer_idx].attn

        qkv = self._linear(x, self.w_qkvs[layer_idx])  # (1, 1, q_dim + 2*kv_dim)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if attn.qk_norm:
            q = attn.q_norm(q)
            k = attn.k_norm(k)

        cos = self.rope_cos_t.index_select(0, self.pos)  # (1, head_dim)
        sin = self.rope_sin_t.index_select(0, self.pos)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k_cache = self.k_caches[layer_idx]
        v_cache = self.v_caches[layer_idx]
        k_cache.index_copy_(2, self.pos, k)
        v_cache.index_copy_(2, self.pos, v)

        out = F.scaled_dot_product_attention(
            q,
            k_cache,
            v_cache,
            attn_mask=self.attn_bias,
            enable_gqa=True,
        )
        out = out.transpose(1, 2).contiguous().view(1, 1, -1)
        return self._linear(out, attn.o_proj.weight)

    # ── Block forward: S tokens against the full cache (compiled prefill/extend) ──

    def _attn_block(
        self,
        layer_idx: int,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """_attn generalized to S queries writing K/V at `positions`."""
        attn = self.model.layers[layer_idx].attn
        S = x.shape[1]

        qkv = self._linear(x, self.w_qkvs[layer_idx])  # (1, S, q_dim + 2*kv_dim)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(1, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(1, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(1, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if attn.qk_norm:
            q = attn.q_norm(q)
            k = attn.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k_cache = self.k_caches[layer_idx]
        v_cache = self.v_caches[layer_idx]
        k_cache.index_copy_(2, positions, k)
        v_cache.index_copy_(2, positions, v)

        out = F.scaled_dot_product_attention(
            q,
            k_cache,
            v_cache,
            attn_mask=bias,
            enable_gqa=True,
        )
        out = out.transpose(1, 2).contiguous().view(1, S, -1)
        return self._linear(out, attn.o_proj.weight)

    def _forward_block(self, tokens: torch.Tensor) -> torch.Tensor:
        """S-token block forward. Compiled per static S (the block buckets).

        Mirrors _forward_step at S > 1: the causal-offset attention bias and
        rope positions derive from `pos` IN-GRAPH (query row i at absolute
        position pos+i attends keys <= pos+i), so the persistent attn_bias
        buffer is not consulted — callers restore it eagerly after chunking.
        Rows beyond pos+i (stale KV from longer past requests) stay masked.
        Returns last-position logits (1, 1, V); advances pos by S.

        DELIBERATELY NO STEERING WRITE: blocks forward PROMPT positions
        (prefill/extend), and the write is generated-positions-only. This is
        also what keeps re-prefilled prior turns unsteered (documented pilot
        choice) — see the steering block in __init__.
        """
        model = self.model
        S = tokens.shape[1]

        positions = self.pos + torch.arange(S, device=tokens.device)  # (S,)
        key_idx = torch.arange(self.max_seq_len, device=tokens.device)
        bias = torch.where(
            key_idx.view(1, 1, 1, -1) <= positions.view(1, 1, S, 1),
            torch.zeros((), device=tokens.device, dtype=self.dtype),
            torch.full((), float("-inf"), device=tokens.device, dtype=self.dtype),
        )
        cos = self.rope_cos_t.index_select(0, positions)  # (S, head_dim)
        sin = self.rope_sin_t.index_select(0, positions)

        self.token_history.index_copy_(0, positions, tokens.view(S))

        embed = model.embed_tokens(tokens)  # (1, S, D)
        committed: list[torch.Tensor] = []
        partial = embed

        for ls in self.schedule:
            layer = model.layers[ls.layer_idx]

            if committed:
                h_attn = self._route(
                    committed + [partial], layer.attn_res_query, layer.attn_res_norm.weight
                )
            else:
                h_attn = partial

            if ls.is_boundary:
                committed.append(partial)
                partial = torch.zeros_like(embed)

            attn_out = self._attn_block(
                ls.layer_idx, layer.attn_norm(h_attn), cos, sin, positions, bias
            )
            partial = partial + attn_out

            h_mlp = self._route(committed + [partial], layer.mlp_res_query, layer.mlp_res_norm.weight)
            gu = self._linear(layer.ffn_norm(h_mlp), self.w_gate_ups[ls.layer_idx])
            gate, up = gu.split(self.config.intermediate_size, dim=-1)
            partial = partial + self._linear(F.silu(gate) * up, layer.ffn.down_proj.weight)

        # Routing is per-position independent — only the last position's
        # logits are consumed, so slice before the final route + lm_head.
        last = [c[:, -1:] for c in committed] + [partial[:, -1:]]
        x = self._route(last, model.final_res_query, model.final_res_norm.weight)
        x = model.norm(x)
        logits = self._linear(x, model.get_lm_head_weight())  # (1, 1, V)

        self.pos.add_(S)
        return logits

    # ── The static decode step ────────────────────────────────────────────────

    def _forward_step(self, token: torch.Tensor) -> torch.Tensor:
        """One decode forward. Mirrors llama._forward_attn_res_cached with locals."""
        model = self.model

        # Open the current position in the attention bias BEFORE attending,
        # and record the consumed token at its slot (keeps token_history in
        # lockstep with the KV cache for prefix matching).
        self.attn_bias.index_fill_(3, self.pos, 0.0)
        self.token_history.index_copy_(0, self.pos, token.view(1))

        embed = model.embed_tokens(token)  # (1, 1, D)
        committed: list[torch.Tensor] = []
        partial = embed

        for ls in self.schedule:
            layer = model.layers[ls.layer_idx]

            # Pre-attention routing (committed + current partial).
            if committed:
                h_attn = self._route(
                    committed + [partial], layer.attn_res_query, layer.attn_res_norm.weight
                )
            else:
                h_attn = partial

            if ls.is_boundary:
                committed.append(partial)
                partial = torch.zeros_like(embed)

            attn_out = self._attn(ls.layer_idx, layer.attn_norm(h_attn))
            partial = partial + attn_out

            # Block-persistent steering write (decode/generated tokens only).
            # Membership is a trace-time constant: a steer-disabled engine
            # traces NO new ops here; an enabled one bakes the add in
            # unconditionally (steer_vec zeros = off), so toggling via
            # set_steer never changes the compiled graph.
            if ls.layer_idx in self._steer_point_set:
                partial = partial + self.steer_vec

            # Pre-MLP routing.
            h_mlp = self._route(committed + [partial], layer.mlp_res_query, layer.mlp_res_norm.weight)
            gu = self._linear(layer.ffn_norm(h_mlp), self.w_gate_ups[ls.layer_idx])
            gate, up = gu.split(self.config.intermediate_size, dim=-1)
            partial = partial + self._linear(F.silu(gate) * up, layer.ffn.down_proj.weight)

        x = self._route(committed + [partial], model.final_res_query, model.final_res_norm.weight)
        x = model.norm(x)
        logits = self._linear(x, model.get_lm_head_weight())  # (1, 1, V)

        self.pos.add_(1)
        return logits

    def _step_sampled(self, token: torch.Tensor) -> torch.Tensor:
        """Forward + Gumbel-max sampling, all in-graph.

        argmax(logits/T + Gumbel noise) is exact categorical sampling from
        softmax(logits/T) (Gumbel-max theorem) — identical distribution to
        serve.py's softmax+multinomial, fused by inductor, philox RNG is
        cudagraph-safe. Penalty math mirrors serve.py: fp32 cast first,
        >0 divide / else multiply, presence over generated ids only.
        rep_penalty == 1.0 passes through unchanged (identity by algebra),
        so no Python branch is needed.
        """
        logits = self._forward_step(token)[0, -1].float()  # (V,)

        rp = self.rep_penalty
        penalized = torch.where(logits > 0, logits / rp, logits * rp)
        logits = torch.where(self.presence, penalized, logits)

        scaled = logits * self.inv_temp
        u = torch.rand_like(scaled)
        gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
        next_token = (scaled + gumbel).argmax(keepdim=True)

        self.presence.scatter_(0, next_token, True)
        self.cur_token.copy_(next_token.view(1, 1))
        return self.cur_token

    def _step_sampled_truncated(self, token: torch.Tensor) -> torch.Tensor:
        """Forward + top-k/top-p + Gumbel-max sampling, all in-graph.

        Mirrors serve.py sample_next_token exactly: penalty -> temperature ->
        top-k (kth-value threshold — ties with the kth logit survive) ->
        top-p over the RENORMALIZED survivor distribution (exclusive cumsum
        < p keeps the first token that crosses) -> categorical draw.
        Disabled dials are algebraic no-ops (top_k_eff = vocab masks nothing;
        top_p_val = 2.0 never triggers), so one graph serves every law.
        Costs one full-vocab sort (~121 us measured on B200 @49152) over the
        pure-temperature step, which keeps its own unchanged graph.
        """
        logits = self._forward_step(token)[0, -1].float()  # (V,)

        rp = self.rep_penalty
        penalized = torch.where(logits > 0, logits / rp, logits * rp)
        logits = torch.where(self.presence, penalized, logits)
        scaled = logits * self.inv_temp

        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        kth = sorted_logits.index_select(0, (self.top_k_eff - 1).view(1))  # (1,)
        sorted_logits = torch.where(
            sorted_logits < kth, torch.full_like(sorted_logits, float("-inf")), sorted_logits
        )
        # -inf suffix -> softmax renormalizes over the top-k survivors, which
        # is what serve.py's second softmax sees.
        probs = sorted_logits.softmax(dim=-1)
        cutoff = probs.cumsum(dim=-1) - probs >= self.top_p_val
        sorted_logits = torch.where(
            cutoff, torch.full_like(sorted_logits, float("-inf")), sorted_logits
        )
        final = torch.full_like(scaled, float("-inf")).scatter(0, sorted_idx, sorted_logits)

        u = torch.rand_like(final)
        gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
        next_token = (final + gumbel).argmax(keepdim=True)

        self.presence.scatter_(0, next_token, True)
        self.cur_token.copy_(next_token.view(1, 1))
        return self.cur_token

    # ── Sampling (serve.py sample_next_token semantics, on-GPU) ─────────────

    def _sample(self, logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Sample from (1, 1, V) logits → (1, 1) token. Updates presence."""
        logits = logits[0, -1].float()  # (V,) — serve.py casts to float first

        rp = params.repetition_penalty
        if rp != 1.0:
            penalized = torch.where(logits > 0, logits / rp, logits * rp)
            logits = torch.where(self.presence, penalized, logits)

        if params.temperature == 0.0:
            token = logits.argmax(keepdim=True)
        else:
            logits = logits / params.temperature
            if params.top_k > 0:
                top_k = min(params.top_k, logits.size(-1))
                kth_val = logits.topk(top_k).values[-1]
                logits = logits.masked_fill(logits < kth_val, float("-inf"))
            if 0.0 < params.top_p < 1.0:
                sorted_logits, sorted_indices = logits.sort(descending=True)
                probs_sorted = sorted_logits.softmax(dim=-1)
                mask = probs_sorted.cumsum(dim=-1) - probs_sorted >= params.top_p
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)

        self.presence.scatter_(0, token, True)
        return token.view(1, 1)

    # ── Public API ───────────────────────────────────────────────────────────

    @torch.inference_mode()
    def reset(self) -> None:
        """Clear per-request state.

        KV contents are also zeroed (~1 GB of writes, ~0.25 ms): masked-out
        stale slots are weight-0 in SDPA, but 0 * inf = NaN, so one request
        that ever wrote a non-finite K/V row would otherwise poison every
        subsequent request until those slots are overwritten.
        """
        self.pos.zero_()
        self._pos_cpu = 0
        self.attn_bias.fill_(float("-inf"))
        self.presence.zero_()
        for li in range(self.n_layers):
            self.k_caches[li].zero_()
            self.v_caches[li].zero_()
        # NOTE: steer_vec is deliberately NOT cleared here — it is per-request
        # state owned by the caller (serve.py engages it before prefill and
        # clears it in a finally after every request).

    @torch.inference_mode()
    def set_steer(self, vec: Optional[torch.Tensor]) -> None:
        """Install (or clear) the steering vector for subsequent decode steps.

        None (or an all-zero vector) turns steering off. The vector is copied
        IN-PLACE into the persistent buffer (cast to engine dtype) — compiled
        graphs hold the buffer by identity, so this never recompiles or swaps
        a graph. The caller owns scaling: pass the FULL-SCALE composite
        (sum of alpha * site_median_norm * unit_vec), not a unit vector.

        Raises on a steering-disabled engine (except set_steer(None), which is
        a no-op there so callers can clear unconditionally in a finally).
        """
        if self.steer_site is None:
            if vec is None:
                return
            raise RuntimeError(
                "Engine was built without steering (steer_site=None); "
                "rebuild with steer_site=<layer> to enable set_steer()."
            )
        if vec is None:
            self.steer_vec.zero_()
            self._steer_active = False
            return
        if not isinstance(vec, torch.Tensor):
            raise TypeError(f"set_steer expects a torch.Tensor or None, got {type(vec).__name__}")
        if vec.numel() != self.hidden:
            raise ValueError(f"Steer vector has {vec.numel()} elements, expected hidden={self.hidden}")
        v = vec.detach().reshape(self.hidden).to(device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(v).all().item()):
            raise ValueError("Steer vector contains non-finite values")
        self.steer_vec.copy_(v.to(self.dtype))
        # CPU-side activity flag (one host sync, request granularity): gates
        # the prefix-cache suffix==1 shortcut, which must not run the
        # injecting compiled step on a prompt position while a vector is live.
        self._steer_active = bool((v != 0).any().item())

    def _chunk_plan(self, start: int, end: int) -> list[tuple[int, int]] | None:
        """Decompose [start, end) into block-bucket chunks.

        Greedy largest-bucket cover; a remainder smaller than the smallest
        bucket becomes a RIGHT-ALIGNED overlap chunk (re-forwarding a few
        already-cached tokens writes identical K/V — harmless, and it keeps
        every chunk a compiled static shape). Returns None when the span (or
        prompt head, for the overlap) can't be covered — caller falls back
        to the eager reference path.
        """
        b_min = self.block_buckets[-1]
        plan: list[tuple[int, int]] = []
        i = start
        while end - i >= b_min:
            b = next(B for B in self.block_buckets if B <= end - i)
            plan.append((i, b))
            i += b
        if i < end:
            if end - b_min < 0:
                return None  # prompt shorter than the smallest bucket
            plan.append((end - b_min, b_min))
        return plan

    @torch.inference_mode()
    def _run_chunks(self, input_ids: torch.Tensor, plan: list[tuple[int, int]]) -> torch.Tensor:
        """Execute a chunk plan via the compiled block forward.

        Each chunk pins pos eagerly (overlap chunks rewind it), replays the
        bucket graph, and leaves pos at chunk end. Restores the persistent
        attn_bias/pos state the decode step expects afterwards.
        """
        assert self._compiled_block is not None
        end = input_ids.shape[1]
        logits = None
        for s, b in plan:
            self.pos.fill_(s)
            logits = self._compiled_block(input_ids[:, s: s + b].contiguous())
        # Block graphs derive their bias in-graph; re-establish the decode
        # invariant on the persistent buffer: [0, end) open, rest closed.
        self.attn_bias.fill_(float("-inf"))
        self.attn_bias[..., :end] = 0.0
        self.pos.fill_(end)
        self._pos_cpu = end
        return logits[0, -1:].float()

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full prefill into the static cache; returns last logits (1, V) fp32.

        Routes through the compiled block forward when available (chunked,
        ~launch-free); the eager reference path (prefill_reference) covers
        un-compiled engines and prompts shorter than the smallest bucket.
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected (1, S) input_ids, got {tuple(input_ids.shape)}")
        prompt_len = input_ids.shape[1]
        self.reset()
        if prompt_len >= self.max_seq_len:
            raise ValueError(f"Prompt too long: {prompt_len} >= {self.max_seq_len}")
        plan = (
            self._chunk_plan(0, prompt_len)
            if self._compiled_block is not None and prompt_len <= self.block_span_max
            else None
        )
        if plan is None:
            return self._prefill_reference_inner(input_ids, prompt_len)
        return self._run_chunks(input_ids.to(self.device), plan)

    @torch.inference_mode()
    def prefill_reference(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Reference eager prefill (parity oracle; pre-block behavior)."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected (1, S) input_ids, got {tuple(input_ids.shape)}")
        prompt_len = input_ids.shape[1]

        # Reset BEFORE the length check: a caller that catches the error and
        # keeps stepping must not silently continue the previous request.
        self.reset()
        if prompt_len >= self.max_seq_len:
            raise ValueError(f"Prompt too long: {prompt_len} >= {self.max_seq_len}")
        return self._prefill_reference_inner(input_ids, prompt_len)

    def _prefill_reference_inner(self, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:

        # Pin prefill to flash (efficient as backstop): cuDNN's per-shape plan
        # selection costs ~300 ms on every novel prompt length, and prefill
        # shapes are arbitrary. This also makes prefill numerics independent
        # of process-global SDPA toggles (parity was measured on this kernel).
        with sdpa_kernel(self._prefill_backends):
            out = self.model(input_ids.to(self.device), use_cache=True)
        for li, (k, v) in enumerate(out["past_kv"]):
            self.k_caches[li][:, :, :prompt_len].copy_(k)
            self.v_caches[li][:, :, :prompt_len].copy_(v)
        self.attn_bias[..., :prompt_len] = 0.0
        self.token_history[:prompt_len].copy_(input_ids[0])
        self.pos.fill_(prompt_len)
        self._pos_cpu = prompt_len
        return out["logits"][0, -1:].float()

    # ── Prefix-cached prefill (extend-from-pos) ──────────────────────────────

    def _common_prefix_len(self, input_ids: torch.Tensor) -> int:
        """Longest common prefix between the cached sequence and a new prompt.

        Token-exact comparison on GPU; one host sync (turn granularity).
        """
        n = min(self._pos_cpu, input_ids.shape[1])
        if n <= 0:
            return 0
        ids = input_ids[0, :n].to(self.device)
        neq = (self.token_history[:n] != ids).nonzero()
        return n if neq.numel() == 0 else int(neq[0, 0].item())

    @torch.inference_mode()
    def prefill_cached(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Prefill reusing any cached common prefix from the previous request.

        Token-exact match against token_history; on a hit, only the suffix is
        forwarded (reference path, explicit causal-offset mask) and its KV is
        appended at the right positions. On a miss (or < min_cached_prefix
        tokens shared) this is exactly prefill().

        Per-request semantics preserved: presence resets here; capacity rules
        match prefill(). The stale-KV NaN insurance that reset() provides is
        replaced by an explicit finiteness gate on the appended suffix KV
        (non-finite -> warn + full re-prefill, which zeroes).

        Returns (last-position logits (1, V) float32, info dict with
        prefix_hit / common_prefix / suffix_len for observability).
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected (1, S) input_ids, got {tuple(input_ids.shape)}")
        prompt_len = input_ids.shape[1]
        if prompt_len >= self.max_seq_len:
            # Mirror prefill(): never leave a caller able to silently continue
            # the previous request after catching the error.
            self.reset()
            raise ValueError(f"Prompt too long: {prompt_len} >= {self.max_seq_len}")

        # Cap at plen-1: even a fully-cached prompt must re-forward its last
        # token to produce next-token logits (regenerate / undo-to-midpoint).
        common = min(self._common_prefix_len(input_ids), prompt_len - 1)
        if common < self.min_cached_prefix:
            logits = self.prefill(input_ids)
            return logits, {"prefix_hit": False, "common_prefix": 0, "suffix_len": prompt_len}

        suffix = input_ids[:, common:].to(self.device)
        suffix_len = prompt_len - common

        # Per-request state. KV rows beyond prompt_len go stale but stay
        # masked by the re-closed bias; they are finite by the same gate below.
        self.presence.zero_()
        self.attn_bias.fill_(float("-inf"))

        if suffix_len == 1 and not self._steer_active:
            # A 1-token extend IS a decode step (regenerate / undo-to-tip):
            # run it through the compiled step — no mask machinery, ~2.5ms.
            # Steering caveat: the compiled step injects steer_vec, but this
            # token is a PROMPT position — with a nonzero vector engaged we
            # skip this shortcut (falls through to the block/eager extend,
            # which never injects) to preserve generated-positions-only
            # semantics. Vector off -> unchanged fast path.
            # The step opens bias at its own position and records history.
            # .contiguous(): the slice has a nonzero storage offset, which the
            # compiled step has never seen as an input -> dynamo guard miss ->
            # ~10s recompile (measured; long enough to flap gateway health).
            self.attn_bias[..., :common] = 0.0
            self.pos.fill_(common)
            self._pos_cpu = common
            logits = self.step_logits(suffix.contiguous())
            return logits[0, -1:].float(), {
                "prefix_hit": True, "common_prefix": common, "suffix_len": 1,
            }

        # Compiled block extend (2026-07-05): chunked block forwards replace
        # the eager MATH-SDPA suffix path (measured 33-40ms launch floor ->
        # block replays). Overlap chunks may rewind into the cached region;
        # those tokens are present in input_ids, so re-forwarding writes
        # identical K/V. Falls through to the eager path when uncompiled.
        if self._compiled_block is not None and suffix_len <= self.block_span_max:
            plan = self._chunk_plan(common, prompt_len)
            if plan is not None:
                self.presence.zero_()
                lo = min(s for s, _ in plan)
                try:
                    logits = self._run_chunks(input_ids.to(self.device), plan)
                except Exception:
                    self.reset()  # same contract as the eager extend below
                    raise
                flags = []
                for li in range(self.n_layers):
                    flags.append(self.k_caches[li][:, :, lo:prompt_len].isfinite().all())
                    flags.append(self.v_caches[li][:, :, lo:prompt_len].isfinite().all())
                finite_ok = bool(torch.stack(flags).all().item())  # one host sync
                if not finite_ok:
                    logger.warning(
                        "prefill_cached(block): non-finite KV at common=%d suffix=%d — "
                        "falling back to full re-prefill", common, suffix_len,
                    )
                    logits = self.prefill(input_ids)
                    return logits, {"prefix_hit": False, "common_prefix": 0, "suffix_len": prompt_len}
                return logits, {
                    "prefix_hit": True, "common_prefix": common, "suffix_len": suffix_len,
                }

        past = [
            (self.k_caches[li][:, :, :common], self.v_caches[li][:, :, :common])
            for li in range(self.n_layers)
        ]
        # Causal-with-offset mask (True = attend): suffix query i (absolute
        # position common+i) sees keys j <= common+i. Required: the reference
        # SDPA path sets is_causal=False whenever past_kv is given, which is
        # only correct for single-token steps.
        mask = torch.ones(suffix_len, prompt_len, dtype=torch.bool, device=self.device)
        mask = mask.tril_(diagonal=common)

        try:
            with sdpa_kernel(self._extend_backends):
                out = self.model(suffix, use_cache=True, past_kv=past, mask=mask)
        except Exception:
            # Don't leave a half-extended state (bias closed, pos stale) for a
            # caller that catches and keeps going — same contract as prefill().
            self.reset()
            raise

        finite_flags = []
        for li, (k, v) in enumerate(out["past_kv"]):
            k_suf = k[:, :, common:]
            v_suf = v[:, :, common:]
            self.k_caches[li][:, :, common:prompt_len].copy_(k_suf)
            self.v_caches[li][:, :, common:prompt_len].copy_(v_suf)
            finite_flags.append(k_suf.isfinite().all())
            finite_flags.append(v_suf.isfinite().all())
        if not bool(torch.stack(finite_flags).all().item()):
            logger.warning(
                "prefill_cached: non-finite suffix KV at common=%d suffix=%d — "
                "falling back to full re-prefill", common, suffix_len,
            )
            logits = self.prefill(input_ids)
            return logits, {"prefix_hit": False, "common_prefix": 0, "suffix_len": prompt_len}

        self.attn_bias[..., :prompt_len] = 0.0
        self.token_history[:prompt_len].copy_(input_ids[0])
        self.pos.fill_(prompt_len)
        self._pos_cpu = prompt_len
        return out["logits"][0, -1:].float(), {
            "prefix_hit": True, "common_prefix": common, "suffix_len": suffix_len,
        }

    @torch.inference_mode()
    def sample_first(self, prefill_logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Sample the first token from prefill logits. Returns (1, 1) CUDA tensor."""
        token = self._sample(prefill_logits.view(1, 1, -1), params)
        self.cur_token.copy_(token)
        return self.cur_token

    def _check_capacity(self) -> None:
        if self._pos_cpu >= self.max_seq_len:
            raise RuntimeError(
                f"Decode position {self._pos_cpu} would exceed max_seq_len "
                f"{self.max_seq_len}; caller must stop at capacity (an OOB "
                "index_copy_ is a sticky device-side assert)."
            )

    def _sync_sampling_buffers(self, params: SamplingParams) -> None:
        if params.temperature != self._cur_temp:
            self.inv_temp.fill_(1.0 / params.temperature)
            self._cur_temp = params.temperature
        if params.repetition_penalty != self._cur_rp:
            self.rep_penalty.fill_(params.repetition_penalty)
            self._cur_rp = params.repetition_penalty
        if params.top_k != self._cur_top_k:
            # 0 (disabled) -> vocab_size: the kth threshold becomes the min
            # logit and masks nothing (serve.py skips the block entirely).
            k = params.top_k if params.top_k > 0 else self.config.vocab_size
            self.top_k_eff.fill_(min(k, self.config.vocab_size))
            self._cur_top_k = params.top_k
        if params.top_p != self._cur_top_p:
            # 0.0/1.0 (disabled) -> 2.0: unreachable even when fp32 cumsum
            # saturates to 1.0 (which it does — see the buffer comment).
            p = params.top_p if 0.0 < params.top_p < 1.0 else 2.0
            self.top_p_val.fill_(p)
            self._cur_top_p = params.top_p

    @torch.inference_mode()
    def step(self, token: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """One decode step: forward(token) → sample → next token (1, 1) on GPU.

        Returns the engine's persistent ``cur_token`` buffer — valid only
        until the next step. Consume immediately (``.item()`` / compare);
        never accumulate the returned tensor across steps.
        """
        self._check_capacity()
        # Fully-fused path (forward + sampler in one graph) for T > 0.
        # Truncated laws (top-k/top-p) use their own compiled variant; the
        # pure-temperature graph is byte-identical to before they existed.
        # (At temperature 0 serve.py argmaxes BEFORE truncation, so truncation
        # is a no-op there — the greedy path below is correct for all laws.)
        if params.temperature > 0.0:
            fused = (
                self._compiled_step_truncated if params.truncated else self._compiled_step_sampled
            )
            if fused is not None:
                self._sync_sampling_buffers(params)
                out = fused(token)
                self._pos_cpu += 1
                return out
        forward = self._compiled_forward or self._forward_step
        logits = forward(token)
        self._pos_cpu += 1
        next_token = self._sample(logits, params)
        self.cur_token.copy_(next_token)
        return self.cur_token

    @torch.inference_mode()
    def step_logits(self, token: torch.Tensor) -> torch.Tensor:
        """Forward only (parity testing): logits (1, 1, V) without sampling.

        Under a compiled step the result lives in CUDA-graph pool memory and
        is CLOBBERED by the next step call — copy (``.float()``/``.clone()``)
        before stepping again.
        """
        self._check_capacity()
        forward = self._compiled_forward or self._forward_step
        out = forward(token)
        self._pos_cpu += 1
        return out

    def compile_step(self, mode: str = "reduce-overhead") -> None:
        """torch.compile the decode-step variants and the block forward."""
        logger.info("Compiling decode step (mode=%s)...", mode)
        import torch._dynamo

        # cur_token is fed back as the next step's input; pinning its address
        # lets cudagraph trees record it directly instead of copying per replay.
        torch._dynamo.mark_static_address(self.cur_token)
        self._compiled_forward = torch.compile(self._forward_step, mode=mode, fullgraph=True)
        self._compiled_step_sampled = torch.compile(self._step_sampled, mode=mode, fullgraph=True)
        self._compiled_step_truncated = torch.compile(self._step_sampled_truncated, mode=mode, fullgraph=True)
        self._compiled_block = torch.compile(self._forward_block, mode=mode, fullgraph=True)

    @torch.inference_mode()
    def warm_blocks(self) -> None:
        """Trigger the per-bucket block-graph compiles (call once at startup).

        Each bucket size is a separate dynamo specialization; an unwarmed
        bucket would pay its compile on the first real request.
        """
        if self._compiled_block is None:
            return
        g = torch.Generator().manual_seed(3)
        for b in self.block_buckets:
            ids = torch.randint(4, self.config.vocab_size, (1, b + 1), generator=g).to(self.device)
            self.prefill(ids[:, :b])          # exact-bucket plan
            self.prefill(ids)                 # overlap-remainder plan
        self.reset()

    @torch.inference_mode()
    def generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
        """Raw-argmax generation helper for PARITY TESTING ONLY.

        Not a serving surrogate: applies no repetition penalty and no stop
        tokens (serve.py at temperature 0 still penalizes and stops).
        Prefill pinned to the reference path — parity harnesses gate the
        decode step from an identical cache; blocks have their own battery.
        """
        logits = self.prefill_reference(input_ids)
        token = logits.argmax(-1).view(1, 1)
        out_ids = [int(token.item())]
        budget = min(max_new_tokens - 1, self.max_seq_len - self._pos_cpu)
        for _ in range(budget):
            self.cur_token.copy_(token)
            logits = self.step_logits(self.cur_token)
            token = logits[0, -1].argmax().view(1, 1)
            out_ids.append(int(token.item()))
        return out_ids
