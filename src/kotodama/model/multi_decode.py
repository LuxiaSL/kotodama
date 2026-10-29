"""Batched row decode: B continuations decoded in lockstep, each row at its own position.

The single-stream DecodeEngine is kernel-count bound on deep looped models (every token replays every layer's
kernels), so a step over B rows costs about what a step over one row costs. A row is filled from an engine prefill
(prefix cache included): ``admit`` copies the engine's prompt state (KV, attention mask, position) into chosen rows
and samples each row's first token from the prompt's last-position logits. Rows filled from one prefill are n
candidates of one prompt; rows filled from several prefills decode different prompts of different lengths together.

Every row owns its KV cache slot, its attention bias (which positions it attends), its position (RoPE angle and KV
write slot), its presence set (repetition penalty over the row's own generated tokens), its sampling noise and, on an
engine built with lever tables (kotodama.model.levers), its lever slot; the weights, temperature, repetition penalty
and the engine's broadcast steering rows (``steer_buf``) are shared, so rows decoded together must share one sampling
law and one broadcast steering config, while their levers may differ row by row. A row nobody reads keeps stepping on whatever it holds; its position stops at
the last cache slot, so an idle row can never write out of bounds. Pure-temperature sampling only (the tsumugi law);
truncated laws and temperature 0 stay on the single-stream path.

Weights, routing and the layer schedule are borrowed from the engine; rebuild this decoder whenever the engine is
rebuilt.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kotodama.model.cascade_attn import attend_for, merge, pack_group_queries, unpack_group_outputs
from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model import attn_stats, signing
from kotodama.model.llama import apply_rope
from kotodama.model.regional import (REGION_MODES, StepGraph, build_regions, check_norms, layer_signatures,
                                     own_code, route_weights, run_regions, sign_write)

logger = logging.getLogger(__name__)

_U31 = (1 << 31) - 1


def seeded_uniform(seeds: torch.Tensor, steps: torch.Tensor, n: int) -> torch.Tensor:
    """Counter-based uniforms in (0, 1): row r's value for token j is a hash of (seeds[r], steps[r], j), so a row's
    noise depends on nothing but its own seed and step (not on its batch, its row index or the RNG state). Integer
    mixing on 31-bit values (every product stays below 2**62). Returns float32 (rows, n)."""
    j = torch.arange(n, device=seeds.device, dtype=torch.int64).view(1, -1)
    h = (seeds.view(-1, 1) * 0x5BD1E995 + steps.view(-1, 1) * 0x2C1B3C6D + j * 0x297A2D39) & _U31
    h = h ^ (h >> 15)
    h = (h * 0x7FEB352D) & _U31
    h = h ^ (h >> 13)
    h = (h * 0x46B6E8E5) & _U31
    h = h ^ (h >> 16)
    return (h.float() + 0.5) / float(1 << 31)


def candidate_seed(seed: int, candidate: int) -> int:
    """The row seed of candidate ``candidate`` of a request seeded ``seed`` (31-bit; distinct per candidate)."""
    x = (int(seed) * 0x9E3779B1 + int(candidate) * 0x85EBCA77 + 0x27D4EB2F) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 12
    return x & _U31


def _slot_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv: tuple[torch.Tensor, ...],
                 state: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """MultiSampleDecoder._attend on passed tensors: kv = (k_cache, v_cache), state = (attn_bias, pos, row_ids)."""
    B = q.shape[0]
    k_cache, v_cache = kv
    attn_bias, pos, row_ids = state
    k_cache[row_ids, :, pos] = k[:, :, 0]
    v_cache[row_ids, :, pos] = v[:, :, 0]
    out = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=attn_bias, enable_gqa=True)
    return out.transpose(1, 2).contiguous().view(B, 1, -1)


def _slot_probs(q: torch.Tensor, kv: tuple[torch.Tensor, ...], state: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """The attention probabilities of ``_slot_attend`` (after its KV write), fp32: q (B, heads, 1, head_dim) ->
    (B, heads, max_seq_len), absolute columns."""
    k_cache = kv[0]
    attn_bias = state[0]
    r = q.shape[1] // k_cache.shape[1]
    scores = torch.einsum("bhd,bhsd->bhs", q[:, :, 0].float(), k_cache.float().repeat_interleave(r, dim=1))
    scores = scores / math.sqrt(q.shape[-1]) + attn_bias[:, 0].float()
    return torch.softmax(scores, dim=-1)


def _cascade_probs(q: torch.Tensor, kv: tuple[torch.Tensor, ...], state: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """The attention probabilities of the cascade attention (after its tail write), fp32: q (B, heads, 1, head_dim)
    -> (B, heads, prompt slot width) in absolute columns (the group's prompt, then the row's tail from its prompt's
    end). The prompt part is scored per group over the group-major queries, as the attention reads it."""
    tail_k, _, prompt_k, _ = kv
    tail_pos, members, row_slot, prompt_lens, _ = state
    B, H, _, D = q.shape
    n_kv, T, W = tail_k.shape[2], tail_k.shape[1], prompt_k.shape[1]
    r = H // n_kv
    g, m = members.shape
    q_rows = q[:, :, 0].float()
    qg = pack_group_queries(q_rows, members, n_kv)  # (G, m * r, Hk, D)
    s1 = torch.einsum("gqkd,gskd->gkqs", qg, prompt_k.float()) / math.sqrt(D)  # (G, Hk, m * r, W)
    s1 = s1.masked_fill(torch.arange(W, device=q.device).view(1, 1, 1, W) >= prompt_lens.view(g, 1, 1, 1),
                        float("-inf"))
    s1 = s1.view(g, n_kv, m, r, W).permute(0, 2, 1, 3, 4).reshape(g * m, H, W)[row_slot]  # (B, H, W)
    s2 = torch.einsum("bhd,bthd->bht", q_rows, tail_k.float().repeat_interleave(r, dim=2)) / math.sqrt(D)
    s2 = s2.masked_fill(torch.arange(T, device=q.device).view(1, 1, T) > tail_pos.view(B, 1, 1), float("-inf"))
    lse = torch.logsumexp(torch.cat([s1, s2], dim=-1), dim=-1, keepdim=True)
    p = torch.exp(s1 - lse)
    start = prompt_lens[torch.div(row_slot, m, rounding_mode="floor")].long().view(B, 1, 1)
    cols = torch.clamp(start + torch.arange(T, device=q.device).view(1, 1, T), max=W - 1).expand(B, H, T)
    return p.scatter_add(-1, cols, torch.exp(s2 - lse))


def _cascade_attend_with(attend_fn: Callable) -> Callable:
    """CascadeDecoder._attend on passed tensors, over the part kernel ``attend_fn``: kv = (tail_k, tail_v, prompt_k,
    prompt_v), state = (tail_pos, members, row_slot, prompt_lens, row_ids). With ``lse`` it also returns each row's
    log-normaliser over its prompt and tail, (B, heads) fp32, the merge's two parts' added in log space."""

    def cascade_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv: tuple[torch.Tensor, ...],
                       state: tuple[torch.Tensor, ...], lse: bool = False):
        B, H, _, D = q.shape
        tail_k, tail_v, prompt_k, prompt_v = kv
        tail_pos, members, row_slot, prompt_lens, row_ids = state
        tail_k[row_ids, tail_pos] = k[:, :, 0]
        tail_v[row_ids, tail_pos] = v[:, :, 0]
        q_rows = q[:, :, 0]
        n_kv = tail_k.shape[2]
        o1, l1 = attend_fn(pack_group_queries(q_rows, members, n_kv), prompt_k, prompt_v, prompt_lens)
        o1, l1 = unpack_group_outputs(o1, l1, members.shape[1])
        o1, l1 = o1[row_slot], l1[row_slot]
        o2, l2 = attend_fn(q_rows.unsqueeze(1), tail_k, tail_v, (tail_pos + 1).to(torch.int32))
        out = merge(o1, l1, o2[:, 0], l2[:, :, 0]).reshape(B, 1, H * D)
        return (out, torch.logaddexp(l1, l2[:, :, 0])) if lse else out

    return cascade_attend


class MultiSampleDecoder(nn.Module):
    """B-row lockstep decoder over a DecodeEngine's weights, with per-row positions.

    Usage (the engine has just prefilled a prompt):
        dec = MultiSampleDecoder(engine, rows=8); dec.compile_step()   # once
        first = dec.admit([0, 1, 2], prefill_logits, params)            # rows 0-2 take this prompt
        ...                                                             # prefill another prompt, admit more rows
        while ...:
            toks = dec.step(params)                                     # (B, 1): next token per row
    ``start`` admits every row from one prefill (n candidates of one prompt). Returned token tensors are the
    persistent ``cur_tokens`` buffer: consume (``.tolist()``) before the next step.

    ``kv_from``: a decoder with at least as many rows whose KV storage this one views (its first ``rows`` rows), so
    several bucket widths cost the memory of the widest. Decoders sharing storage must not hold live rows at the same
    time; the server runs one at a time.
    """

    def __init__(self, engine: DecodeEngine, rows: int, kv_from: Optional["MultiSampleDecoder"] = None,
                 readout_k: int = 0, capture_layers: Sequence[int] = (), seeded: bool = False,
                 sign: Optional[signing.SignSpec] = None) -> None:
        super().__init__()
        if rows < 2:
            raise ValueError(f"rows must be >= 2 (one row is the engine's own path), got {rows}")
        if kv_from is not None and (kv_from.rows < rows or kv_from.engine is not engine):
            raise ValueError(f"kv_from must be a decoder over the same engine with >= {rows} rows")
        self.__dict__["engine"] = engine  # borrowed, not a submodule: its buffers stay the engine's
        self.rows = int(rows)
        self.max_seq_len = engine.max_seq_len
        self.n_layers = engine.n_layers
        dev = engine.device
        self._alloc_kv(kv_from)
        self.register_buffer("pos", torch.zeros((self.rows,), device=dev, dtype=torch.long), persistent=False)
        self.register_buffer("row_ids", torch.arange(self.rows, device=dev, dtype=torch.long), persistent=False)
        self.register_buffer("presence", torch.zeros((self.rows, engine.config.vocab_size), device=dev,
                                                     dtype=torch.bool), persistent=False)
        self.register_buffer("cur_tokens", torch.zeros((self.rows, 1), device=dev, dtype=torch.long),
                             persistent=False)
        # Each row's sampling law, set when the row is admitted (so rows of different laws decode in one batch):
        # 1 / temperature, repetition penalty, top-p (2.0 = off: the exclusive cumsum never reaches it) and log
        # min-p (-inf = off). The truncation pair is read only by the truncated step variant.
        self.register_buffer("row_inv_temp", torch.full((self.rows, 1), 1.0 / 0.9, device=dev, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("row_rp", torch.full((self.rows, 1), 1.2, device=dev, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("row_top_p", torch.full((self.rows, 1), 2.0, device=dev, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("row_log_min_p", torch.full((self.rows, 1), float("-inf"), device=dev,
                                                         dtype=torch.float32), persistent=False)
        # Each row's lever slot (0: the identity), read at every lever add of the step.
        if engine.levers is not None:
            self.register_buffer("row_slots", torch.zeros((self.rows,), device=dev, dtype=torch.long),
                                 persistent=False)
        # Per-step readouts (readout_k > 0, a trace-time constant): each row's chosen token's logprob and the top-k
        # tokens with their logprobs, all under the model's own next-token distribution (log-softmax of the raw
        # logits, before penalty and temperature), written in-graph at every step for the token that step chose.
        self.readout_k = int(readout_k)
        if self.readout_k > 0:
            self.register_buffer("step_logprob", torch.zeros((self.rows,), device=dev, dtype=torch.float32),
                                 persistent=False)
            self.register_buffer("step_top_ids", torch.zeros((self.rows, self.readout_k), device=dev,
                                                             dtype=torch.long), persistent=False)
            self.register_buffer("step_top_lps", torch.zeros((self.rows, self.readout_k), device=dev,
                                                             dtype=torch.float32), persistent=False)
        # MLP-input capture (capture_layers, a trace-time constant): the post-ffn_norm vector each step feeds the
        # MLP of these effective layers, kept for the step's input token (the last forced prompt token after
        # ``force``). Every neuron's activation there is silu(W_gate x) * (W_up x).
        self.capture_layers = tuple(int(x) for x in capture_layers)
        if any(not 0 <= x < self.n_layers for x in self.capture_layers):
            raise ValueError(f"capture_layers {self.capture_layers} outside [0, {self.n_layers})")
        if self.capture_layers:
            self.register_buffer("mlp_in", torch.zeros((self.rows, len(self.capture_layers), engine.hidden),
                                                       device=dev, dtype=engine.dtype), persistent=False)
        # Seeded rows (seeded, a trace-time constant): each row's sampling noise is seeded_uniform(row_seed, row_step),
        # its own seed and its own step count, so a seeded candidate draws the same noise in any batch, at any row.
        # A row admitted without a seed gets a fresh random one. Off: the noise is the process RNG (torch.rand_like).
        self.seeded = bool(seeded)
        if self.seeded:
            self.register_buffer("row_seed", torch.zeros((self.rows,), device=dev, dtype=torch.int64),
                                 persistent=False)
            self.register_buffer("row_step", torch.zeros((self.rows,), device=dev, dtype=torch.int64),
                                 persistent=False)
        # Signing (sign, a trace-time constant; kotodama.model.signing): each row's span surfaces at the sampled
        # layers, written by every step at slot pos - sign_start (the row's position at admission).
        self.sign = sign
        self._sign_start_host = [0] * self.rows
        if sign is not None:
            sign.check(self.n_layers)
            # the computor the rows describe; a lane that declares its weights refuses any other model
            self._sign_weights = signing.weights_digest(engine.model)
            if sign.weights is not None and sign.weights != self._sign_weights:
                raise ValueError(f"the signing lane declares weights {sign.weights[:12]}..., the engine's model is "
                                 f"{self._sign_weights[:12]}...")
            sigs = layer_signatures(engine.model._attn_res_boundary_set, self.n_layers)
            self._sign_route = signing.route_width(sigs)
            bufs = signing.allocate(sign, self.rows, hidden=engine.hidden, n_heads=engine.n_heads,
                                    n_kv_heads=engine.n_kv_heads, head_dim=engine.head_dim,
                                    intermediate=engine.config.intermediate_size, n_layers=self.n_layers,
                                    route=self._sign_route, device=dev, dtype=engine.dtype)
            for name, b in bufs.items():
                self.register_buffer(f"sign_{name}", b, persistent=False)
            self.sign_names = tuple(bufs)
            self.register_buffer("sign_start", torch.zeros((self.rows,), device=dev, dtype=torch.long),
                                 persistent=False)
            self.register_buffer("sign_slot", torch.zeros((self.rows,), device=dev, dtype=torch.long),
                                 persistent=False)
            # each layer's column indices, one tensor each: views of one buffer would differ in storage offset,
            # which the compiled regions guard on (a recompile per layer)
            self._sign_layer_ids = [torch.tensor(li, device=dev, dtype=torch.long) for li in range(self.n_layers)]
            self._sign_cols = [torch.tensor(j, device=dev, dtype=torch.long) for j in range(len(sign.layers))]
            # each layer's column in the partial buffer (depth: every layer; else the sampled ones), -1: none
            if sign.depth:
                self._sign_partial_cols = list(range(self.n_layers))
            else:
                self._sign_partial_cols = [sign.layers.index(li) if li in sign.layers else -1
                                           for li in range(self.n_layers)]
            logger.info("%s: signing %d layers%s over %d positions per row, %.0f MB", type(self).__name__,
                        len(sign.layers), " + the depth path" if sign.depth else "", sign.tail,
                        sum(b.numel() * b.element_size() for b in bufs.values()) / 1e6)
        self._row_truncated = [False] * self.rows  # host: which rows' laws truncate (picks the step variant)
        self._row_scope = ["generated"] * self.rows  # host: each row's penalty scope
        self._pos_host = [0] * self.rows  # host mirror of pos (no device syncs for capacity checks)
        self._compiled_step = None
        self._compiled_greedy = None
        self._compiled_truncated = None
        self._compiled_score = None
        self._greedy = False
        # each row's prompt ids (device), for penalty_scope "row": restart and rewind re-seed presence from them
        self._row_prompt: list[Optional[torch.Tensor]] = [None] * self.rows
        # score(): the token each row is scored on at the current step, and its logprob
        self.register_buffer("score_target", torch.zeros((self.rows, 1), device=dev, dtype=torch.long),
                             persistent=False)
        self.register_buffer("score_lp", torch.zeros((self.rows,), device=dev, dtype=torch.float32),
                             persistent=False)
        # Eager taps (kotodama.model.levers.capture): None outside a capture; a compiled step never runs inside one.
        self._taps = None
        logger.info("%s: %d rows, kv cache %.0f MB%s", type(self).__name__, self.rows, self.kv_bytes() / 1e6,
                    f" (shared with the {kv_from.rows}-row decoder)" if kv_from is not None else "")

    def _alloc_kv(self, kv_from: Optional["MultiSampleDecoder"]) -> None:
        """Every row's own full KV slot per layer, and the per-row attention mask over it."""
        eng = self.engine
        dev, dt = eng.device, eng.dtype
        kv_shape = (self.rows, eng.n_kv_heads, self.max_seq_len, eng.head_dim)
        for li in range(self.n_layers):
            if kv_from is None:
                k = torch.zeros(kv_shape, device=dev, dtype=dt)
                v = torch.zeros(kv_shape, device=dev, dtype=dt)
            else:  # leading-dim slices of a contiguous buffer: contiguous views at a fixed address
                k, v = kv_from.k_caches[li][:self.rows], kv_from.v_caches[li][:self.rows]
            self.register_buffer(f"k_cache_{li}", k, persistent=False)
            self.register_buffer(f"v_cache_{li}", v, persistent=False)
        self.k_caches = [getattr(self, f"k_cache_{li}") for li in range(self.n_layers)]
        self.v_caches = [getattr(self, f"v_cache_{li}") for li in range(self.n_layers)]
        self.register_buffer("attn_bias", torch.full((self.rows, 1, 1, self.max_seq_len), float("-inf"), device=dev,
                                                     dtype=dt), persistent=False)
        # the value _open_step writes, on the device: a Python scalar would be a host tensor, which a captured step
        # cannot copy from
        self.register_buffer("_bias_open", torch.zeros((), device=dev, dtype=dt), persistent=False)

    def _kv_buffers(self) -> list[torch.Tensor]:
        """The KV storage the compiled step writes in place (static addresses for the CUDA graph)."""
        return self.k_caches + self.v_caches

    def kv_bytes(self) -> int:
        """Bytes of KV storage this decoder addresses (views of a wider decoder's storage included)."""
        return sum(b.numel() * b.element_size() for b in self._kv_buffers())

    @staticmethod
    def supports(params: SamplingParams) -> bool:
        """The laws this decoder samples: temperature > 0 with any repetition penalty, top-p and min-p (top-k is the
        single-stream engine's)."""
        return params.temperature > 0.0 and params.top_k == 0

    # ── the batched decode step ─────────────────────────────────────────────

    def _attn(self, layer_idx: int, x: torch.Tensor, keep: Optional[list] = None, stats: int = 0) -> torch.Tensor:
        """``keep``: a list the pre-RoPE q, k and v are appended to (signing); ``stats``: write the layer's attention
        statistics (1: every layer's, 2: a sampled layer's too)."""
        eng = self.engine
        attn = eng.model.layers[layer_idx].attn
        B = x.shape[0]
        qkv = F.linear(x, eng.w_qkvs[layer_idx])  # (B, 1, q_dim + 2*kv_dim)
        q, k, v = qkv.split([eng.q_dim, eng.kv_dim, eng.kv_dim], dim=-1)
        q = q.view(B, 1, eng.n_heads, eng.head_dim).transpose(1, 2)
        k = k.view(B, 1, eng.n_kv_heads, eng.head_dim).transpose(1, 2)
        v = v.view(B, 1, eng.n_kv_heads, eng.head_dim).transpose(1, 2)
        if attn.qk_norm:
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        if keep is not None:
            keep.extend((q, k, v))
        cos = eng.rope_cos_t.index_select(0, self.pos).unsqueeze(1)  # (B, 1, head_dim): each row's own angle
        sin = eng.rope_sin_t.index_select(0, self.pos).unsqueeze(1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if stats and self._stats_fn is not None:  # the fused statistics read the attention's log-normalisers
            attend, kvs, state = self._regional_attend()
            out, lse = attend(q, k, v, kvs[layer_idx], state, lse=True)
            self._stats_fn(self._stats_args(layer_idx), q, kvs[layer_idx], state, stats == 2, lse)
        else:
            out = self._attend(layer_idx, q, k, v)  # (B, 1, n_heads * head_dim)
        if stats and self._stats_fn is None:
            _, kvs, state = self._regional_attend()
            signing.stats_write(self._stats_args(layer_idx), self._probs(q, kvs[layer_idx], state), stats == 2)
        return F.linear(out, attn.o_proj.weight)

    def _attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Write this step's k/v (B, n_kv_heads, 1, head_dim) at each row's position and attend each row's query
        (B, n_heads, 1, head_dim) over its own slot. Returns (B, 1, n_heads * head_dim)."""
        B = q.shape[0]
        k_cache, v_cache = self.k_caches[layer_idx], self.v_caches[layer_idx]
        # Row r writes its slot pos[r]: an index_put_ on the buffer itself, which inductor updates in place.
        k_cache[self.row_ids, :, self.pos] = k[:, :, 0]
        v_cache[self.row_ids, :, self.pos] = v[:, :, 0]
        out = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=self.attn_bias, enable_gqa=True)
        return out.transpose(1, 2).contiguous().view(B, 1, -1)

    def _open_step(self) -> None:
        """Each row opens its own position in the mask before attending."""
        self.attn_bias[self.row_ids, 0, 0, self.pos] = self._bias_open

    def _advance_step(self) -> None:
        # Live rows never reach the last slot (callers budget each row); an idle row parks there.
        self.pos.copy_(torch.clamp(self.pos + 1, max=self.max_seq_len - 1))

    def _forward_step(self, tokens: torch.Tensor) -> torch.Tensor:
        """One decode forward over all rows: tokens (B, 1) -> logits (B, V). Mirrors DecodeEngine._forward_step."""
        eng = self.engine
        model = eng.model
        self._open_step()
        self._sign_step()
        embed = model.embed_tokens(tokens)  # (B, 1, D)
        committed: list[torch.Tensor] = []
        if eng.levers is not None:
            # every layer's lever rows in one gather per step ([n_layers, B, hidden]): each layer adds its slice
            lever_attn = eng.lever_attn.index_select(1, self.row_slots)
            lever_mlp = eng.lever_mlp.index_select(1, self.row_slots)
            act_keep = eng.lever_act_keep.index_select(1, self.row_slots)
            act_add = eng.lever_act_add.index_select(1, self.row_slots)
        partial = embed
        if self.sign is not None:
            self.sign_embed[self.row_ids, self.sign_slot, 0] = embed[:, 0]
        for ls in eng.schedule:
            layer = model.layers[ls.layer_idx]
            routes = self.sign is not None
            signs = routes and ls.layer_idx in self.sign.layers
            w_attn = keep = None
            if committed:
                if routes:
                    h_attn, w_attn = route_weights(committed + [partial], layer.attn_res_query,
                                                   layer.attn_res_norm.weight, eng.eps)
                else:
                    h_attn = eng._route(committed + [partial], layer.attn_res_query, layer.attn_res_norm.weight)
            else:
                h_attn = partial
            if ls.is_boundary:
                committed.append(partial)
                partial = torch.zeros_like(embed)
            if signs:
                keep = []
            stats = 0
            if self.sign is not None and self.sign.attn:
                stats = 2 if signs else 1
            attn_out = self._attn(ls.layer_idx, layer.attn_norm(h_attn), keep, stats)
            if eng.levers is not None:
                attn_out = attn_out + lever_attn[ls.layer_idx].unsqueeze(1)
            partial = partial + attn_out
            if eng.steer_enabled:
                partial = partial + eng.steer_buf[ls.layer_idx]
            if routes:
                h_mlp, w_mlp = route_weights(committed + [partial], layer.mlp_res_query, layer.mlp_res_norm.weight,
                                             eng.eps)
            else:
                h_mlp = eng._route(committed + [partial], layer.mlp_res_query, layer.mlp_res_norm.weight)
            x_mlp = layer.ffn_norm(h_mlp)
            if ls.layer_idx in self.capture_layers:
                self.mlp_in[:, self.capture_layers.index(ls.layer_idx)] = x_mlp[:, 0]
            if self._taps is not None:
                self._taps.record(ls.layer_idx, partial, x_mlp)
            gu = F.linear(x_mlp, eng.w_gate_ups[ls.layer_idx])
            gate, up = gu.split(eng.config.intermediate_size, dim=-1)
            if routes:
                q_k_v = keep if signs else (None,) * 3
                sign_write(self._sign_args(ls.layer_idx), *q_k_v, gate if signs else None, w_attn, w_mlp)
            act = F.silu(gate) * up
            if eng.levers is not None:
                act = act * act_keep[ls.layer_idx].unsqueeze(1) + act_add[ls.layer_idx].unsqueeze(1)
            mlp_out = F.linear(act, layer.ffn.down_proj.weight)
            if eng.levers is not None:
                mlp_out = mlp_out + lever_mlp[ls.layer_idx].unsqueeze(1)
            partial = partial + mlp_out
            if self.sign is not None and self._sign_partial_cols[ls.layer_idx] >= 0:
                buf, cols, row_ids, slot = self._sign_partials()
                buf[row_ids, slot, cols[ls.layer_idx]] = partial[:, -1]
        x = self._final_route(committed + [partial])
        x = model.norm(x)
        logits = F.linear(x, model.get_lm_head_weight())  # (B, 1, V)
        self._advance_step()
        return logits[:, -1]

    # Scan length of each chunk of _exclusive_cumsum: a row's cumsum over the whole vocabulary is a scan long enough
    # for inductor to split it across blocks, and its split-scan codegen fails at batched decode shapes; two scans of
    # at most a few hundred elements compile as plain reductions.
    _SCAN_CHUNK = 256

    @classmethod
    def _exclusive_cumsum(cls, p: torch.Tensor) -> torch.Tensor:
        """Per row, the sum of the entries before each one ((B, V) -> (B, V)): a cumsum within chunks plus the
        running total of the chunks before."""
        B, V = p.shape
        C = cls._SCAN_CHUNK
        pad = (-V) % C
        x = F.pad(p, (0, pad)).view(B, -1, C)
        inner = x.cumsum(dim=-1)
        totals = inner[..., -1]
        before = totals.cumsum(dim=-1) - totals
        return (inner - x + before.unsqueeze(-1)).view(B, -1)[:, :V]

    @staticmethod
    def _truncate(scaled: torch.Tensor, top_p: torch.Tensor, log_min_p: torch.Tensor) -> torch.Tensor:
        """Top-p over the temperature-scaled logits (the smallest prefix of the sorted distribution whose exclusive
        cumulative probability stays below p; ties keep their sort order), then min-p over what survives (drop a
        token whose probability is below min_p times the top token's). Per row; off values are no-ops."""
        sorted_logits, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cut = MultiSampleDecoder._exclusive_cumsum(probs) >= top_p
        sorted_logits = torch.where(cut, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
        floor = sorted_logits[:, :1] + log_min_p  # log(min_p * p_max) relative to the shared normaliser
        sorted_logits = torch.where(sorted_logits < floor, torch.full_like(sorted_logits, float("-inf")),
                                    sorted_logits)
        return torch.full_like(scaled, float("-inf")).scatter(1, sorted_idx, sorted_logits)

    def _sample_rows(self, logits: torch.Tensor, greedy: bool = False, truncated: bool = False) -> torch.Tensor:
        """Per-row repetition penalty over the row's own presence set, then Gumbel-max at each row's temperature
        (``truncated``: after each row's top-p / min-p; ``greedy``: the argmax of the penalized logits)."""
        raw = logits.float()
        rp = self.row_rp
        penalized = torch.where(raw > 0, raw / rp, raw * rp)
        logits = torch.where(self.presence, penalized, raw)
        if greedy:
            nxt = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
        else:
            scaled = logits * self.row_inv_temp
            if truncated:
                scaled = self._truncate(scaled, self.row_top_p, self.row_log_min_p)
            if self.seeded:
                u = seeded_uniform(self.row_seed, self.row_step, scaled.shape[-1])
            else:
                u = torch.rand_like(scaled)
            gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
            nxt = (scaled + gumbel).argmax(dim=-1, keepdim=True)  # (B, 1)
        if self.seeded:
            self.row_step.add_(1)
        if self.readout_k > 0:
            lp = torch.log_softmax(raw, dim=-1)
            self.step_logprob.copy_(lp.gather(1, nxt).view(-1))
            top_lp, top_id = lp.topk(self.readout_k, dim=-1)
            self.step_top_ids.copy_(top_id)
            self.step_top_lps.copy_(top_lp)
        self.presence.scatter_(1, nxt, True)
        self.cur_tokens.copy_(nxt)
        return self.cur_tokens

    def _sign_logits(self, logits: torch.Tensor, chosen: torch.Tensor) -> None:
        """A signing decoder's logit statistics of the step's raw logits and its chosen token (sampled, or the scored
        one). Eager, after the step's sampler or scorer, so sampling and scoring compute them with the same kernels:
        inside the compiled samplers each would fuse them its own way."""
        if self.sign is not None:
            self.sign_logit_stats[self.row_ids, self.sign_slot, 0] = signing.logit_stats(logits.float(), chosen)

    def _step_sampled(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self._forward_step(tokens)
        out = self._sample_rows(logits)
        self._sign_logits(logits, self.cur_tokens)
        return out

    def _step_greedy(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self._forward_step(tokens)
        out = self._sample_rows(logits, greedy=True)
        self._sign_logits(logits, self.cur_tokens)
        return out

    def _step_truncated(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self._forward_step(tokens)
        out = self._sample_rows(logits, truncated=True)
        self._sign_logits(logits, self.cur_tokens)
        return out

    def _step_scored(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self._forward_step(tokens)
        out = self._score_logits(logits)
        self._sign_logits(logits, self.score_target)
        return out

    # ── the regional step (compile_step(regional=True)) ─────────────────────

    def _regional_attend(self) -> tuple[Callable, list[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
        """The attention a regional layer runs: (attend function, each layer's KV buffers, the per-step state)."""
        return _slot_attend, list(zip(self.k_caches, self.v_caches)), (self.attn_bias, self.pos, self.row_ids)

    def regional_signatures(self) -> list[tuple[int, bool, bool, bool]]:
        """Each effective layer's regional signature: (committed sources, block boundary, MLP input captured,
        signs)."""
        sigs = layer_signatures(self.engine.model._attn_res_boundary_set, self.n_layers)
        signed = self.sign.layers if self.sign is not None else ()
        attn = self.sign is not None and self.sign.attn
        level = (lambda li: 2 if li in signed else 1) if self.sign is not None else (lambda li: 0)
        return [(c, b, li in self.capture_layers, level(li), level(li) if attn else 0)
                for li, (c, b) in enumerate(sigs)]

    # The attention probabilities of this decoder's attention (signing statistics), on the regional attend's tensors.
    _probs = staticmethod(_slot_probs)
    # A function that writes a layer's attention statistics itself (stats_tuple, q, kv, state, sampled, lse), in place
    # of ``_probs`` + signing.stats_write, given the attention's log-normalisers (its regional attend's ``lse=True``);
    # None: the reference.
    _stats_fn: Optional[Callable] = None

    # ── signing (kotodama.model.signing) ────────────────────────────────────

    def _sign_step(self) -> None:
        """Each row's slot this step: its position since admission (the last slot once past the tail)."""
        if self.sign is not None:
            self.sign_slot.copy_(torch.clamp(self.pos - self.sign_start, 0, self.sign.tail - 1))

    def _sign_args(self, layer_idx: int) -> tuple:
        """A layer's sign tuple (regional.sign_write): its routing, and a sampled layer's surfaces."""
        head = (self.sign_route, self.row_ids, self.sign_slot, self._sign_layer_ids[layer_idx])
        col = self._sign_partial_cols[layer_idx]
        tail = (None, None) if col < 0 else (self.sign_depth if self.sign.depth else self.sign_hidden,
                                              self._sign_layer_ids[col])
        if layer_idx not in self.sign.layers:
            return head + (None,) * 5 + tail
        return head + tuple(getattr(self, f"sign_{n}") for n in signing.REGION_SURFACES) + (
            self._sign_cols[self.sign.layers.index(layer_idx)],) + tail

    def _stats_args(self, layer_idx: int) -> tuple:
        """A layer's stats tuple (kotodama.model.signing.stats_write)."""
        sampled = self.sign.layers.index(layer_idx) if layer_idx in self.sign.layers else None
        return tuple(getattr(self, f"sign_{n}") for n in signing.STATS) + (
            self.row_ids, self.sign_slot, self._sign_layer_ids[layer_idx],
            None if sampled is None else self._sign_cols[sampled], self.sign_start, self.pos)

    def _sign_partials(self) -> tuple:
        buf = self.sign_depth if self.sign.depth else self.sign_hidden
        return buf, self._sign_partial_cols, self.row_ids, self.sign_slot

    def signed(self, row: int, device: Optional[torch.device] = None) -> signing.SignedRow:
        """Row ``row``'s captured span since its admission (copies; default on the CPU)."""
        if self.sign is None:
            raise RuntimeError("this decoder was built without signing (sign=None)")
        eng = self.engine
        lever = int(self.row_slots[row]) if eng.levers is not None else 0
        bufs = {n: getattr(self, f"sign_{n}") for n in self.sign_names}
        return signing.export(self.sign, bufs, row, start=self._sign_start_host[row], pos=self._pos_host[row],
                              lever_slot=lever, steered=bool(eng.steer_enabled),
                              signatures=layer_signatures(eng.model._attn_res_boundary_set, self.n_layers),
                              n_heads=eng.n_heads, n_kv_heads=eng.n_kv_heads, head_dim=eng.head_dim,
                              weights=self._sign_weights, device=device)

    def _build_regions(self, **opts) -> None:
        """One compiled decode layer per distinct signature, and each layer's tensors in the order the layer takes."""
        eng = self.engine
        attend, kvs, self._regional_state = self._regional_attend()
        self._regional_layers, self.regional_variants = build_regions(
            eng, self.regional_signatures(), kvs, "decode_layer", attend=attend, steer=bool(eng.steer_enabled),
            capture_col=self.capture_layers.index, sign_args=self._sign_args, probs=self._probs,
            stats_args=self._stats_args, stats_fn=self._stats_fn, **opts)

    def _final_logits(self, committed: list[torch.Tensor], partial: torch.Tensor) -> torch.Tensor:
        """The step's tail after the last layer: final routing, norm and LM head, then every row's position advances.
        Returns logits (B, V)."""
        eng = self.engine
        model = eng.model
        x = self._final_route(committed + [partial])
        x = model.norm(x)
        logits = F.linear(x, model.get_lm_head_weight())
        self._advance_step()
        return logits[:, -1]

    def _final_route(self, sources: list[torch.Tensor]) -> torch.Tensor:
        """The final routing; a signing decoder writes its weights (the last position's) into sign_route_final."""
        eng = self.engine
        model = eng.model
        if self.sign is None:
            return eng._route(sources, model.final_res_query, model.final_res_norm.weight)
        x, w = route_weights(sources, model.final_res_query, model.final_res_norm.weight, eng.eps)
        buf = self.sign_route_final
        row = w[:, :, -1].transpose(0, 1).to(buf.dtype)
        if row.shape[1] < buf.shape[-1]:
            row = torch.cat([row, row.new_zeros((row.shape[0], buf.shape[-1] - row.shape[1]))], dim=-1)
        buf[self.row_ids, self.sign_slot, 0] = row
        return x

    def _regional_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``_forward_step`` with each layer run as its signature's compiled region: tokens (B, 1) -> logits (B, V)."""
        eng = self.engine
        self._open_step()
        self._sign_step()
        embed = eng.model.embed_tokens(tokens)
        if self.sign is not None:
            self.sign_embed[self.row_ids, self.sign_slot, 0] = embed[:, 0]
        levers = None
        if eng.levers is not None:
            levers = tuple(t.index_select(1, self.row_slots)
                           for t in (eng.lever_attn, eng.lever_mlp, eng.lever_act_keep, eng.lever_act_add))
        cos = eng.rope_cos_t.index_select(0, self.pos).unsqueeze(1)  # (B, 1, head_dim): each row's own angle
        sin = eng.rope_sin_t.index_select(0, self.pos).unsqueeze(1)
        committed, partial = run_regions(self._regional_layers, embed, cos, sin, self._regional_state, levers,
                                         eng.steer_buf if eng.steer_enabled else None,
                                         self.mlp_in if self.capture_layers else None)
        return self._regional_final(committed, partial)

    def _sample_plain(self, logits: torch.Tensor) -> torch.Tensor:
        return self._sample_rows(logits)

    def _sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        return self._sample_rows(logits, greedy=True)

    def _sample_truncated(self, logits: torch.Tensor) -> torch.Tensor:
        return self._sample_rows(logits, truncated=True)

    def _score_logits(self, logits: torch.Tensor) -> torch.Tensor:
        raw = logits.float()
        lp = torch.log_softmax(raw, dim=-1)
        self.score_lp.copy_(lp.gather(1, self.score_target).view(-1))
        return self.score_lp

    def _compile_regional(self, mode: str, backend, cuda_graph: Optional[bool]) -> None:
        check_norms(self.engine.model, self.engine.eps)
        on_cuda = self.engine.device.type == "cuda"
        cuda_graph = on_cuda if cuda_graph is None else bool(cuda_graph)
        if cuda_graph and not on_cuda:
            raise ValueError("whole-step CUDA graph capture needs a CUDA engine")
        opts = dict(fullgraph=True, dynamic=False, backend=backend)
        if backend == "inductor":
            opts["mode"] = REGION_MODES.get(mode, mode)
        self._build_regions(**opts)
        tag = f"{type(self).__name__}{self.rows}_{id(self):x}"
        self._regional_final = torch.compile(own_code(self._final_logits, f"final_{tag}"), **opts)
        pool = torch.cuda.graph_pool_handle() if cuda_graph else None

        def step_with(sampler: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
            compiled = torch.compile(own_code(sampler, f"{sampler.__name__.lstrip('_')}_{tag}"), **opts)
            chosen = self.score_target if sampler == self._score_logits else self.cur_tokens

            def run(tokens: torch.Tensor) -> torch.Tensor:
                logits = self._regional_forward(tokens)
                out = compiled(logits)
                self._sign_logits(logits, chosen)
                return out

            return StepGraph(run, self.cur_tokens, pool) if cuda_graph else run

        self._compiled_step = step_with(self._sample_plain)
        self._compiled_greedy = step_with(self._sample_greedy)
        self._compiled_truncated = step_with(self._sample_truncated)
        self._compiled_score = step_with(self._score_logits)
        logger.info("%s (%d rows): regional decode step, %d layer signatures over %d layers, regions %s, %s",
                    type(self).__name__, self.rows, self.regional_variants, self.n_layers,
                    opts.get("mode", backend), "whole-step CUDA graph" if cuda_graph else "no CUDA graph")

    # ── public API ──────────────────────────────────────────────────────────

    def _check_law(self, params: SamplingParams) -> None:
        if params.top_k > 0 or params.temperature < 0.0:
            raise ValueError("MultiSampleDecoder samples temperature >= 0 with repetition penalty, top-p and min-p "
                             "(no top-k)")

    def _sync_sampling(self, params: SamplingParams) -> None:
        """A step's law: greedy (temperature 0) for every row, else each row's own law, set when it was admitted."""
        self._check_law(params)
        self._greedy = params.temperature == 0.0

    def _set_row_law(self, rows: list[int], idx: torch.Tensor, params: SamplingParams) -> None:
        """``rows`` sample under ``params`` from here on (temperature, repetition penalty, top-p, min-p, scope)."""
        self._check_law(params)
        self._greedy = params.temperature == 0.0
        inv_t = 1.0 / params.temperature if params.temperature > 0.0 else 1.0
        top_p = params.top_p if 0.0 < params.top_p < 1.0 else 2.0
        log_min_p = math.log(params.min_p) if params.min_p > 0.0 else float("-inf")
        for buf, val in ((self.row_inv_temp, inv_t), (self.row_rp, params.repetition_penalty),
                         (self.row_top_p, top_p), (self.row_log_min_p, log_min_p)):
            buf.index_fill_(0, idx, val)
        trunc = params.truncated
        for r in rows:
            self._row_truncated[r] = trunc
            self._row_scope[r] = params.penalty_scope

    @torch.inference_mode()
    def admit(self, rows: Sequence[int], prefill_logits: torch.Tensor, params: SamplingParams,
              slots: Optional[Sequence[int] | int] = None, seeds: Optional[Sequence[int]] = None) -> list[int]:
        """Fill ``rows`` from the engine's prefilled prompt and sample each row's first token from the prompt's
        last-position logits (1, V). The prompt's KV positions are copied and the mask is the engine's (open on the
        prompt only), so a row attends to nothing it decoded before: its later positions are closed until its own
        decode steps rewrite and open them. Writes the first tokens into ``cur_tokens[rows]`` (the next step feeds
        them) and returns them. ``slots``: each row's lever slot (one int for every row; None: the identity)."""
        eng = self.engine
        rows = self._check_rows(rows)
        self._check_prompt_lever(len(rows), slots)
        self.set_slots(rows, slots)
        if eng._pos_cpu <= 0 or eng._pos_cpu >= self.max_seq_len:
            raise RuntimeError(f"engine position {eng._pos_cpu} is not a prefilled prompt")
        n = len(rows)
        p = eng._pos_cpu
        idx = torch.tensor(rows, device=self.pos.device, dtype=torch.long)
        self._set_row_law(rows, idx, params)
        contiguous = rows == list(range(rows[0], rows[0] + n))
        for li in range(self.n_layers):
            for dst, src in ((self.k_caches[li], eng.k_caches[li]), (self.v_caches[li], eng.v_caches[li])):
                if contiguous:
                    dst[rows[0]:rows[0] + n, :, :p].copy_(src[:, :, :p].expand(n, -1, -1, -1))
                else:  # one indexed write, the prompt broadcast over the rows (no materialized copies)
                    dst[:, :, :p].index_put_((idx,), src[:, :, :p].expand(n, -1, -1, -1))
        self.attn_bias.index_copy_(0, idx, eng.attn_bias.expand(n, -1, -1, -1).contiguous())
        self._set_prompt(rows, eng.token_history[:p])
        return self._seed(rows, idx, eng._pos_cpu, prefill_logits, seeds)

    @torch.inference_mode()
    def restart(self, rows: Sequence[int], prompt_len: int, prefill_logits: torch.Tensor,
                params: SamplingParams, seeds: Optional[Sequence[int]] = None) -> list[int]:
        """Start a fresh candidate in ``rows``, which already hold a ``prompt_len``-token prompt in their first
        cache slots (they were admitted from it and have only decoded since): the rows rewind to the prompt's end
        (positions past it are closed in the mask and rewritten as decoding proceeds) and sample a first token from
        that prompt's last-position logits (1, V). Costs no KV copy; the rows keep their lever slots. Returns the
        first tokens."""
        rows = self._check_rows(rows)
        if not 0 < prompt_len < self.max_seq_len:
            raise ValueError(f"prompt_len {prompt_len} outside (0, {self.max_seq_len})")
        idx = torch.tensor(rows, device=self.pos.device, dtype=torch.long)
        self._set_row_law(rows, idx, params)
        self.attn_bias[idx, :, :, prompt_len:] = float("-inf")
        return self._seed(rows, idx, prompt_len, prefill_logits, seeds)

    @torch.inference_mode()
    def set_slots(self, rows: Sequence[int], slots: Optional[Sequence[int] | int]) -> None:
        """Point ``rows`` at lever ``slots`` (one per row, or one int for all; None: the identity). Takes effect at
        the next step. On an engine without lever tables only None (or all zeros) is accepted."""
        rows = [int(r) for r in rows]
        if slots is None:
            vals = [0] * len(rows)
        elif isinstance(slots, int):
            vals = [int(slots)] * len(rows)
        else:
            vals = [int(x) for x in slots]
            if len(vals) != len(rows):
                raise ValueError(f"{len(vals)} slots for {len(rows)} rows")
        levers = self.engine.levers
        if levers is None:
            if any(vals):
                raise RuntimeError("the engine was built without lever tables (levers=0); rows can only read slot 0")
            return
        levers.check_slots(vals)
        idx = torch.tensor(rows, device=self.pos.device, dtype=torch.long)
        self.row_slots.index_copy_(0, idx, torch.tensor(vals, device=self.pos.device, dtype=torch.long))

    def _check_prompt_lever(self, n: int, slots: Optional[Sequence[int] | int]) -> None:
        """Rows admitted from a prefill must write their prompt the way it was written: a uniform lever's rows need
        the prompt prefilled under that lever (same content digest), every other row an unsteered prompt."""
        levers = self.engine.levers
        if levers is None or slots is None:
            want = [0] * n
        else:
            want = [int(slots)] * n if isinstance(slots, int) else [int(x) for x in slots]
        if levers is None:
            return
        kv = self.engine._kv_lever
        for s in set(want):
            plan = levers.plan(s) if s else None
            need = plan.digest() if plan is not None and plan.span == "uniform" else None
            if need != kv:
                raise ValueError(f"slot {s} writes its prompt {'under lever ' + need if need else 'unsteered'}, "
                                 f"the engine's prompt KV is {'under lever ' + kv if kv else 'unsteered'}")

    def _check_rows(self, rows: Sequence[int]) -> list[int]:
        rows = [int(r) for r in rows]
        if not rows or len(set(rows)) != len(rows) or not all(0 <= r < self.rows for r in rows):
            raise ValueError(f"rows must be distinct indices in [0, {self.rows}), got {rows}")
        return rows

    def _set_row_seeds(self, idx: torch.Tensor, seeds: Optional[Sequence[int]], n: int, step: int) -> None:
        if seeds is None:
            vals = torch.randint(0, 1 << 31, (n,), device=idx.device, dtype=torch.int64)
        else:
            if len(seeds) != n:
                raise ValueError(f"{len(seeds)} seeds for {n} rows")
            vals = torch.tensor([int(x) & _U31 for x in seeds], device=idx.device, dtype=torch.int64)
        self.row_seed.index_copy_(0, idx, vals)
        self.row_step.index_fill_(0, idx, step)

    def _set_prompt(self, rows: list[int], prompt: torch.Tensor) -> None:
        ids = prompt.reshape(-1).clone()
        for r in rows:
            self._row_prompt[r] = ids

    def _seed_presence(self, rows: list[int], idx: torch.Tensor) -> None:
        """Empty presence for ``rows``, or (a row whose penalty_scope is "row") the row's prompt ids."""
        self.presence.index_fill_(0, idx, False)
        for r in rows:
            if self._row_scope[r] != "row":
                continue
            ids = self._row_prompt[r]
            if ids is not None and ids.numel():
                self.presence[r].index_fill_(0, ids, True)

    def _seed(self, rows: list[int], idx: torch.Tensor, pos: int, prefill_logits: torch.Tensor,
              seeds: Optional[Sequence[int]] = None) -> list[int]:
        """Put ``rows`` at position ``pos`` with their presence seeded (empty, or the prompt under penalty_scope
        "row") and their first token sampled under that presence. ``seeds`` (seeded decoders): each row's seed (None:
        a fresh random seed per row); the first token is the row's step 0."""
        n = len(rows)
        self.pos.index_fill_(0, idx, pos)
        self._sign_from(rows, idx, pos)
        if self.seeded:
            self._set_row_seeds(idx, seeds, n, step=0)
        self._seed_presence(rows, idx)
        raw = prefill_logits.reshape(1, -1).float().expand(n, -1)
        logits = raw
        if any(self._row_scope[r] == "row" for r in rows):  # the penalty over the prompt applies to the first token
            rp = self.row_rp.index_select(0, idx)
            pres = self.presence.index_select(0, idx)
            logits = torch.where(pres, torch.where(raw > 0, raw / rp, raw * rp), raw)
        if self._greedy:
            first = logits.argmax(dim=-1, keepdim=True)  # (n, 1)
        else:
            scaled = logits * self.row_inv_temp.index_select(0, idx)
            if any(self._row_truncated[r] for r in rows):
                scaled = self._truncate(scaled, self.row_top_p.index_select(0, idx),
                                        self.row_log_min_p.index_select(0, idx))
            if self.seeded:
                u = seeded_uniform(self.row_seed.index_select(0, idx), self.row_step.index_select(0, idx),
                                   scaled.shape[-1])
            else:
                u = torch.rand_like(scaled)
            gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
            first = (scaled + gumbel).argmax(dim=-1, keepdim=True)  # (n, 1)
        if self.seeded:
            self.row_step.index_fill_(0, idx, 1)
        if self.readout_k > 0:
            lp = torch.log_softmax(raw, dim=-1)
            top_lp, top_id = lp.topk(self.readout_k, dim=-1)
            self.step_logprob.index_copy_(0, idx, lp.gather(1, first).view(-1))
            self.step_top_ids.index_copy_(0, idx, top_id)
            self.step_top_lps.index_copy_(0, idx, top_lp)
        self.presence.index_put_((idx, first.view(-1)), torch.ones(n, dtype=torch.bool, device=idx.device))
        self.cur_tokens.index_copy_(0, idx, first)
        for r in rows:
            self._pos_host[r] = pos
        return first.view(-1).tolist()

    @torch.inference_mode()
    def start(self, prefill_logits: torch.Tensor, params: SamplingParams,
              slots: Optional[Sequence[int] | int] = None, seeds: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Admit every row from the engine's prefilled prompt (n candidates of one prompt). Returns cur_tokens."""
        self.admit(range(self.rows), prefill_logits, params, slots=slots, seeds=seeds)
        return self.cur_tokens

    @torch.inference_mode()
    def step(self, params: SamplingParams) -> torch.Tensor:
        """Feed every row its last token, sample the next (temperature 0: the argmax). Returns cur_tokens (B, 1)."""
        self._sync_sampling(params)
        greedy = params.temperature == 0.0
        truncated = not greedy and any(self._row_truncated)
        if self._taps is not None:
            fn = self._step_greedy if greedy else (self._step_truncated if truncated else self._step_sampled)
        elif greedy:
            fn = self._compiled_greedy or self._step_greedy
        elif truncated:
            fn = self._compiled_truncated or self._step_truncated
        else:
            fn = self._compiled_step or self._step_sampled
        out = fn(self.cur_tokens)
        self._advance_host()
        return out

    @torch.inference_mode()
    def admit_empty(self, slots: Optional[Sequence[int] | int] = None, seeds: Optional[Sequence[int]] = None) -> None:
        """Start EVERY row from an empty cache at position 0, each reading its lever slot from its first position:
        the prompt is then fed through the decode step (``force``), so a uniform lever writes each row's prompt
        under its own lever, as the eager reference writes every position. Rows hold no shared prompt."""
        rows = list(range(self.rows))
        self.set_slots(rows, slots)
        self.release(rows)
        idx = torch.arange(self.rows, device=self.pos.device)
        self.pos.zero_()
        self.presence.zero_()
        self._pos_host = [0] * self.rows
        self._sign_from(rows, idx, 0)
        if self.seeded:
            self._set_row_seeds(idx, seeds, self.rows, step=0)
        self._admit_empty_storage(idx)

    def _sign_from(self, rows: list[int], idx: torch.Tensor, pos: int) -> None:
        """``rows``' spans start at ``pos``."""
        for r in rows:
            self._sign_start_host[r] = pos
        if self.sign is not None:
            self.sign_start.index_fill_(0, idx, pos)

    def _admit_empty_storage(self, idx: torch.Tensor) -> None:
        """Per-decoder storage for rows starting from an empty cache: every position closed in the mask (each step
        opens its own)."""
        self.attn_bias.fill_(float("-inf"))

    @torch.inference_mode()
    def force(self, prompt: torch.Tensor, params: SamplingParams) -> list[int]:
        """Feed every row the tokens of ``prompt`` ((B, S) or (1, S), one prompt for every row) through the decode
        step, after ``admit_empty``. The step at the last prompt token samples each row's first generated token
        under ``params`` (its readouts, when built with readout_k, are that token's); presence then holds only that
        token. Returns the first tokens."""
        if prompt.ndim != 2 or prompt.shape[0] not in (1, self.rows) or prompt.shape[1] < 1:
            raise ValueError(f"prompt must be (1, S) or ({self.rows}, S), got {tuple(prompt.shape)}")
        S = prompt.shape[1]
        if any(p + S >= self.max_seq_len for p in self._pos_host) or not all(self._room_for(S)):
            raise ValueError(f"a {S}-token prompt does not fit the rows' caches")
        prompt = prompt.to(self.pos.device).expand(self.rows, S)
        self._set_row_law(list(range(self.rows)), self.row_ids, params)
        for r in range(self.rows):
            self._row_prompt[r] = prompt[r].clone()
        for t in range(S):
            if t == S - 1:
                # the tokens sampled at earlier prompt positions were never generated
                self._seed_presence(list(range(self.rows)), self.row_ids)
                if self.seeded:
                    self.row_step.zero_()  # the first generated token is each row's step 0
            self.cur_tokens.copy_(prompt[:, t:t + 1])
            self.step(params)
        return self.cur_tokens.view(-1).tolist()

    def _room_for(self, n: int) -> list[bool]:
        return [True] * self.rows

    @torch.inference_mode()
    def score(self, tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-force ``tokens`` ((B, S) or (1, S)) through the decode step from where every row stands: step t
        feeds tokens[:, t] and scores tokens[:, t + 1] under the row's own lever (raw log-softmax). Returns the
        logprobs (B, S - 1); the rows end S - 1 positions further on (``rewind`` takes them back). A signing
        decoder signs the scored span as it signs a generated one (the scored token is each step's chosen token),
        so scoring a generated row's tokens from the same prompt replays its signature."""
        if tokens.ndim != 2 or tokens.shape[0] not in (1, self.rows) or tokens.shape[1] < 2:
            raise ValueError(f"tokens must be (1, S) or ({self.rows}, S) with S >= 2, got {tuple(tokens.shape)}")
        S = tokens.shape[1]
        if not all(self._room_for(S)) or any(p + S >= self.max_seq_len for p in self._pos_host):
            raise ValueError(f"{S} scored tokens do not fit the rows' caches")
        tokens = tokens.to(self.pos.device).expand(self.rows, S)
        fn = self._compiled_score if self._compiled_score is not None and self._taps is None else self._step_scored
        out = []
        for t in range(S - 1):
            self.cur_tokens.copy_(tokens[:, t:t + 1])
            self.score_target.copy_(tokens[:, t + 1:t + 2])
            out.append(fn(self.cur_tokens).clone())
            self._advance_host()
        return torch.stack(out, dim=1)

    @torch.inference_mode()
    def rewind(self, length: int, first: Sequence[int]) -> None:
        """Take every row back to the first ``length`` tokens of its sequence (later positions are closed and
        rewritten as decoding proceeds); ``first`` is the token each row feeds next, its presence set holds that
        token (and, under penalty_scope "row", the row's prompt ids)."""
        if not 0 < length < self.max_seq_len or len(first) != self.rows:
            raise ValueError(f"rewind needs 0 < length < {self.max_seq_len} and one first token per row")
        dev = self.pos.device
        self.pos.fill_(length)
        self._pos_host = [length] * self.rows
        self._rewind_storage(length)
        first_t = torch.tensor(list(first), device=dev, dtype=torch.long).view(-1, 1)
        self.cur_tokens.copy_(first_t)
        self._seed_presence(list(range(self.rows)), self.row_ids)
        self.presence.scatter_(1, first_t, True)
        if self.seeded:
            self.row_step.fill_(1)  # the first token was each row's step 0

    def _rewind_storage(self, length: int) -> None:
        self.attn_bias[:, :, :, length:] = float("-inf")

    @torch.inference_mode()
    def step_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward only (parity testing, captures): tokens (B, 1) -> logits (B, V), eager."""
        out = self._forward_step(tokens)
        self._advance_host()
        return out

    def _advance_host(self) -> None:
        self._pos_host = [p + 1 for p in self._pos_host]

    def compile_step(self, mode: str = "reduce-overhead", regional: bool = False, backend="inductor",
                     cuda_graph: Optional[bool] = None) -> None:
        """Compile the step variants (sampled, greedy, truncated, scored).

        Default: each variant is one graph of the whole step under ``mode`` (CUDA graphs come from the mode).
        ``regional``: each decode layer runs as the compiled graph of its signature (``regional_signatures``; one
        graph per distinct signature, reused by every layer that has it), the final routing, LM head and sampler as
        graphs of their own, all without CUDA graphs (``mode`` maps to its no-cudagraphs form); the depth loop runs in
        Python, and with ``cuda_graph`` (default: on a CUDA engine) each variant's whole step is captured as one CUDA
        graph on its third call and replayed after. ``backend`` (regional only): the torch.compile backend of the
        regions (``mode`` applies to inductor only)."""
        if regional:
            self._compile_regional(mode, backend, cuda_graph)
            return
        if backend != "inductor" or cuda_graph is not None:
            raise ValueError("backend and cuda_graph apply to the regional step only (regional=True)")
        if self.sign is not None:
            raise ValueError("a signing decoder compiles regionally (regional=True): the whole-step graph would fuse "
                             "its logit statistics differently in each step variant")
        import torch._dynamo

        torch._dynamo.mark_static_address(self.cur_tokens)
        for buf in self._kv_buffers():
            torch._dynamo.mark_static_address(buf)
        self._compiled_step = torch.compile(self._step_sampled, mode=mode, fullgraph=True,
                                            dynamic=self._compile_dynamic)
        self._compiled_greedy = torch.compile(self._step_greedy, mode=mode, fullgraph=True,
                                              dynamic=self._compile_dynamic)
        self._compiled_truncated = torch.compile(self._step_truncated, mode=mode, fullgraph=True,
                                                 dynamic=self._compile_dynamic)
        self._compiled_score = torch.compile(self._step_scored, mode=mode, fullgraph=True,
                                             dynamic=self._compile_dynamic)

    # torch.compile's ``dynamic``: None lets a second bucket width recompile the shared step function with symbolic
    # row counts (automatic dynamic shapes).
    _compile_dynamic: Optional[bool] = None

    def row_pos(self, row: int) -> int:
        """Row ``row``'s next write position (its prompt length plus the tokens it has been fed). A row at
        ``max_seq_len`` is full; the device position of such a row stays parked at the last slot."""
        return self._pos_host[row]

    def row_full(self, row: int) -> bool:
        """Whether row ``row`` has no cache slot left for another fed token."""
        return self._pos_host[row] >= self.max_seq_len

    def token_room(self, prompt_len: int) -> int:
        """Tokens a candidate of a ``prompt_len``-token prompt can generate before its row is full."""
        return self.max_seq_len - prompt_len

    def admit_capacity(self) -> int:
        """Rows ``admit`` can fill right now (every row: each row owns its whole slot)."""
        return self.rows

    def release(self, rows: Iterable[int]) -> None:
        """Mark ``rows`` as holding nothing anyone will restart from. Each row owns its slot here, so there is
        nothing to free; decoders whose rows share storage (``CascadeDecoder``) reclaim it."""


class CascadeDecoder(MultiSampleDecoder):
    """A ``MultiSampleDecoder`` whose rows share their prompt's KV: the rows admitted from one prefill form a group
    that holds ONE copy of the prompt's KV per layer, and each row stores only the KV it decodes itself (its tail).

    Decode attention is cascade attention (kotodama.model.cascade_attn): every group's rows attend to the group's
    prompt together, in one pass over its keys, each row attends to its own tail, and the two results are merged by
    log-sum-exp, which is attention over prompt + tail to within kernel rounding. The prompt's keys are read once per
    group instead of once per row, and only up to the prompt's length instead of over the whole slot, so a step costs
    less the more rows share a prompt; storage drops from ``rows`` full slots to ``groups`` prompt slots plus
    ``rows`` tails.

    Storage per layer: ``groups`` prompt slots (groups, max_seq_len, n_kv_heads, head_dim) and one tail per row
    (rows, tail_tokens, n_kv_heads, head_dim), position-major (flash-attn's KV-cache layout). A group holds up to
    ``group_rows`` rows; admitting more rows from one prefill fills several groups, each with its own copy. Admission
    takes free groups: ``admit_capacity`` is how many rows can be admitted now, and ``release`` hands back rows nobody
    will restart (a group is free once none of its rows is held). A row generates at most ``tail_tokens`` tokens
    (``token_room``), so callers route longer requests elsewhere.

    ``kv_from``: a wider ``CascadeDecoder`` (same tail length, at least as many groups) whose storage this one views.
    Defaults: ``groups`` = rows // 4 (at least 1), ``group_rows`` = min(rows, 32).
    """

    def __init__(self, engine: DecodeEngine, rows: int, groups: int = 0, group_rows: int = 0,
                 tail_tokens: int = 1024, kv_from: Optional["CascadeDecoder"] = None, readout_k: int = 0,
                 capture_layers: Sequence[int] = (), seeded: bool = False,
                 sign: Optional[signing.SignSpec] = None) -> None:
        groups = int(groups) or max(1, int(rows) // 4)
        group_rows = int(group_rows) or min(int(rows), 32)
        if not 1 <= groups <= rows or not 1 <= group_rows <= rows:
            raise ValueError(f"groups ({groups}) and group_rows ({group_rows}) must be in [1, rows={rows}]")
        if not 1 <= tail_tokens <= engine.max_seq_len:
            raise ValueError(f"tail_tokens {tail_tokens} outside [1, {engine.max_seq_len}]")
        if kv_from is not None and (not isinstance(kv_from, CascadeDecoder) or kv_from.groups < groups
                                    or kv_from.tail_tokens != tail_tokens):
            raise ValueError("kv_from must be a CascadeDecoder with the same tail_tokens and >= as many groups")
        # read by _alloc_kv, which the base constructor calls
        self.__dict__["_cascade_shape"] = (groups, group_rows, int(tail_tokens))
        super().__init__(engine, rows, kv_from=kv_from, readout_k=readout_k, capture_layers=capture_layers,
                         seeded=seeded, sign=sign)
        self.attend_backend, self._attend_fn = attend_for(engine.device, engine.dtype)
        # attention statistics by the fused kernel where it runs (kotodama.model.attn_stats), else the reference
        self._stats_fn = attn_stats.cascade_stats if attn_stats.available(engine.device, engine.dtype) else None
        self._row_group = [-1] * self.rows  # host: the group each row reads its prompt from (-1: none)
        self._group_members: list[set[int]] = [set() for _ in range(self.groups)]
        self._group_len = [0] * self.groups  # host mirror of prompt_lens
        self._tail_host = [0] * self.rows  # host mirror of tail_pos
        logger.info("CascadeDecoder: %d groups of up to %d rows, %d-token tails, %s attention", self.groups,
                    self.group_rows, self.tail_tokens, self.attend_backend)

    def _alloc_kv(self, kv_from: Optional[MultiSampleDecoder]) -> None:
        eng = self.engine
        self.groups, self.group_rows, self.tail_tokens = self.__dict__["_cascade_shape"]
        dev, dt = eng.device, eng.dtype
        prompt_shape = (self.groups, self.max_seq_len, eng.n_kv_heads, eng.head_dim)
        tail_shape = (self.rows, self.tail_tokens, eng.n_kv_heads, eng.head_dim)
        for li in range(self.n_layers):
            for name, shape, n in (("prompt_k", prompt_shape, self.groups), ("prompt_v", prompt_shape, self.groups),
                                   ("tail_k", tail_shape, self.rows), ("tail_v", tail_shape, self.rows)):
                if kv_from is None:
                    buf = torch.zeros(shape, device=dev, dtype=dt)
                else:  # leading-dim slices of a contiguous buffer: contiguous views at a fixed address
                    buf = getattr(kv_from, f"{name}_{li}")[:n]
                self.register_buffer(f"{name}_{li}", buf, persistent=False)
        self.prompt_ks = [getattr(self, f"prompt_k_{li}") for li in range(self.n_layers)]
        self.prompt_vs = [getattr(self, f"prompt_v_{li}") for li in range(self.n_layers)]
        self.tail_ks = [getattr(self, f"tail_k_{li}") for li in range(self.n_layers)]
        self.tail_vs = [getattr(self, f"tail_v_{li}") for li in range(self.n_layers)]
        # Keys a group's rows attend in its prompt slot (0 = an empty group: no weight in the merge).
        self.register_buffer("prompt_lens", torch.zeros((self.groups,), device=dev, dtype=torch.int32),
                             persistent=False)
        # members[g, i]: the row whose query is group g's i-th query; unused entries repeat a member (their output
        # is computed and never read). row_slot[r] = g * group_rows + i: where row r's shared-part output lands.
        self.register_buffer("members", torch.zeros((self.groups, self.group_rows), device=dev, dtype=torch.long),
                             persistent=False)
        self.register_buffer("row_slot", torch.zeros((self.rows,), device=dev, dtype=torch.long), persistent=False)
        # Each row's next tail slot (the tokens it has been fed since its prompt).
        self.register_buffer("tail_pos", torch.zeros((self.rows,), device=dev, dtype=torch.long), persistent=False)

    def _kv_buffers(self) -> list[torch.Tensor]:
        return self.prompt_ks + self.prompt_vs + self.tail_ks + self.tail_vs

    def _regional_attend(self) -> tuple[Callable, list[tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]:
        return (_cascade_attend_with(self._attend_fn), list(zip(self.tail_ks, self.tail_vs, self.prompt_ks,
                                                                 self.prompt_vs)),
                (self.tail_pos, self.members, self.row_slot, self.prompt_lens, self.row_ids))

    # Every bucket compiles its own static graph: the symbolic-row-count graph a later bucket otherwise gets is slower
    # (the 1B's 8-row step: 13.5 ms symbolic, 9.7 ms static).
    _compile_dynamic = False

    def _attend(self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, H, _, D = q.shape
        tail_k, tail_v = self.tail_ks[layer_idx], self.tail_vs[layer_idx]
        # Row r writes its tail slot tail_pos[r] in place (the prompt slots are only written by admit).
        tail_k[self.row_ids, self.tail_pos] = k[:, :, 0]
        tail_v[self.row_ids, self.tail_pos] = v[:, :, 0]
        q_rows = q[:, :, 0]  # (B, H, D)
        n_kv = tail_k.shape[2]
        o1, l1 = self._attend_fn(pack_group_queries(q_rows, self.members, n_kv), self.prompt_ks[layer_idx],
                                 self.prompt_vs[layer_idx], self.prompt_lens)
        o1, l1 = unpack_group_outputs(o1, l1, self.group_rows)
        o1, l1 = o1[self.row_slot], l1[self.row_slot]  # (B, H, D), (B, H)
        o2, l2 = self._attend_fn(q_rows.unsqueeze(1), tail_k, tail_v, (self.tail_pos + 1).to(torch.int32))
        return merge(o1, l1, o2[:, 0], l2[:, :, 0]).reshape(B, 1, H * D)

    _probs = staticmethod(_cascade_probs)

    def _open_step(self) -> None:
        """Nothing to open: the tail's length (tail_pos + 1) bounds what each row attends."""

    def _advance_step(self) -> None:
        super()._advance_step()
        # Live rows never fill their tail (callers budget each row by token_room); an idle row parks at the end.
        self.tail_pos.copy_(torch.clamp(self.tail_pos + 1, max=self.tail_tokens - 1))

    def _advance_host(self) -> None:
        super()._advance_host()
        self._tail_host = [t + 1 for t in self._tail_host]

    def row_full(self, row: int) -> bool:
        return super().row_full(row) or self._tail_host[row] >= self.tail_tokens

    def _admit_empty_storage(self, idx: torch.Tensor) -> None:
        """Every row joins a group whose prompt is empty (length 0: the shared pass gives it no weight), so the
        whole sequence, prompt included, lives in the row's own tail."""
        m = self.group_rows
        rows = idx.tolist()
        chunks = [rows[i:i + m] for i in range(0, len(rows), m)]
        if len(chunks) > self.groups:
            raise RuntimeError(f"{len(rows)} rows need {len(chunks)} groups, the decoder has {self.groups}")
        dev = self.pos.device
        slots = {}
        for g, chunk in enumerate(chunks):
            for i, r in enumerate(chunk):
                slots[r] = g * m + i
                self._row_group[r] = g
            self._group_members[g] = set(chunk)
            self._group_len[g] = 0
        self.prompt_lens.zero_()
        self.members.copy_(torch.tensor([c + [c[0]] * (m - len(c)) for c in chunks]
                                        + [[0] * m] * (self.groups - len(chunks)), device=dev, dtype=torch.long))
        self.row_slot.copy_(torch.tensor([slots[r] for r in rows], device=dev, dtype=torch.long))
        self.tail_pos.zero_()
        self._tail_host = [0] * self.rows

    def _room_for(self, n: int) -> list[bool]:
        return [t + n < self.tail_tokens for t in self._tail_host]

    def _rewind_storage(self, length: int) -> None:
        tails = []
        for r in range(self.rows):
            g = self._row_group[r]
            glen = self._group_len[g] if g >= 0 else 0
            if not 0 <= length - glen <= self.tail_tokens - 1:
                raise ValueError(f"row {r}: length {length} is not inside its group prompt + tail")
            tails.append(length - glen)
        self.tail_pos.copy_(torch.tensor(tails, device=self.pos.device, dtype=torch.long))
        self._tail_host = tails

    def token_room(self, prompt_len: int) -> int:
        return min(super().token_room(prompt_len), self.tail_tokens)

    def _free_groups(self) -> list[int]:
        return [g for g in range(self.groups) if not self._group_members[g]]

    def admit_capacity(self) -> int:
        return min(self.rows, len(self._free_groups()) * self.group_rows)

    def release(self, rows: Iterable[int]) -> None:
        emptied = []
        for r in rows:
            r = int(r)
            g = self._row_group[r]
            if g < 0:
                continue
            self._row_group[r] = -1
            self._group_members[g].discard(r)
            if not self._group_members[g]:
                self._group_len[g] = 0
                emptied.append(g)
        if emptied:  # the shared pass reads nothing of an empty group
            self.prompt_lens.index_fill_(0, torch.tensor(emptied, device=self.pos.device), 0)

    @torch.inference_mode()
    def admit(self, rows: Sequence[int], prefill_logits: torch.Tensor, params: SamplingParams,
              slots: Optional[Sequence[int] | int] = None, seeds: Optional[Sequence[int]] = None) -> list[int]:
        """Fill ``rows`` from the engine's prefilled prompt: the prompt's KV is copied once into a free group (one
        group per ``group_rows`` rows), each row's tail starts empty, and each row's first token is sampled from the
        prompt's last-position logits (1, V). Raises RuntimeError when too few groups are free (``admit_capacity``).
        ``slots`` as in ``MultiSampleDecoder.admit``. Returns the first tokens."""
        eng = self.engine
        rows = self._check_rows(rows)
        self._check_prompt_lever(len(rows), slots)
        self.set_slots(rows, slots)
        p = eng._pos_cpu
        if p <= 0 or p >= self.max_seq_len:
            raise RuntimeError(f"engine position {p} is not a prefilled prompt")
        self._sync_sampling(params)
        self.release(rows)  # whatever they held is overwritten
        m = self.group_rows
        chunks = [rows[i:i + m] for i in range(0, len(rows), m)]
        free = self._free_groups()
        if len(free) < len(chunks):
            raise RuntimeError(f"{len(rows)} rows need {len(chunks)} free groups, {len(free)} are free")
        used = free[:len(chunks)]
        slots: dict[int, int] = {}
        for g, chunk in zip(used, chunks):
            for li in range(self.n_layers):
                self.prompt_ks[li][g, :p].copy_(eng.k_caches[li][0, :, :p].transpose(0, 1))
                self.prompt_vs[li][g, :p].copy_(eng.v_caches[li][0, :, :p].transpose(0, 1))
            for i, r in enumerate(chunk):
                slots[r] = g * m + i
                self._row_group[r] = g
            self._group_members[g] = set(chunk)
            self._group_len[g] = p
        dev = self.pos.device
        gidx = torch.tensor(used, device=dev, dtype=torch.long)
        self.members.index_copy_(0, gidx, torch.tensor([c + [c[0]] * (m - len(c)) for c in chunks], device=dev,
                                                        dtype=torch.long))
        self.prompt_lens.index_fill_(0, gidx, p)
        idx = torch.tensor(rows, device=dev, dtype=torch.long)
        self.row_slot.index_copy_(0, idx, torch.tensor([slots[r] for r in rows], device=dev, dtype=torch.long))
        self.tail_pos.index_fill_(0, idx, 0)
        for r in rows:
            self._tail_host[r] = 0
        self._set_prompt(rows, eng.token_history[:p])
        self._set_row_law(rows, idx, params)
        return self._seed(rows, idx, p, prefill_logits, seeds)

    @torch.inference_mode()
    def restart(self, rows: Sequence[int], prompt_len: int, prefill_logits: torch.Tensor,
                params: SamplingParams, seeds: Optional[Sequence[int]] = None) -> list[int]:
        """Start a fresh candidate in ``rows``, which still belong to a group holding a ``prompt_len``-token prompt
        (they were admitted from it and not released): the tails empty and each row samples a first token from that
        prompt's last-position logits (1, V). Costs no KV copy. Returns the first tokens."""
        rows = self._check_rows(rows)
        for r in rows:
            g = self._row_group[r]
            if g < 0 or self._group_len[g] != prompt_len:
                raise ValueError(f"row {r} holds no {prompt_len}-token prompt")
        idx = torch.tensor(rows, device=self.pos.device, dtype=torch.long)
        self._set_row_law(rows, idx, params)
        self.tail_pos.index_fill_(0, idx, 0)
        for r in rows:
            self._tail_host[r] = 0
        return self._seed(rows, idx, prompt_len, prefill_logits, seeds)


def decode_rows(dec: MultiSampleDecoder, prefill_logits: torch.Tensor, params: SamplingParams, n: int,
                max_new_tokens: int, stop_ids: frozenset[int],
                stop_check: Optional[Callable[[list[int]], bool]] = None,
                slots: Optional[int] = None, seed: Optional[int] = None,
                stops: Optional[list[Optional[int]]] = None) -> list[list[int]]:
    """Generate up to ``max_new_tokens`` per row for the first ``n`` rows of one prompt (every row on lever slot
    ``slots``); a row ends at a stop token (not kept), when ``stop_check(row_ids)`` is true, or at capacity. ``seed``
    (a seeded decoder): candidate i draws with ``candidate_seed(seed, i)``. Returns each row's generated ids;
    ``stops`` (n entries), when given, receives the stop token that ended each row (None: it ended otherwise)."""
    if not 1 <= n <= dec.rows:
        raise ValueError(f"n={n} outside [1, {dec.rows}]")
    out: list[list[int]] = [[] for _ in range(n)]
    live = [True] * n
    seeds = None
    if seed is not None:
        if not dec.seeded:
            raise ValueError("a seed needs a seeded decoder (seeded=True)")
        seeds = [candidate_seed(seed, i) for i in range(dec.rows)]  # rows past n draw candidates nobody reads
    dec.start(prefill_logits, params, slots=slots, seeds=seeds)
    toks = dec.cur_tokens.view(-1).tolist()
    for _ in range(max_new_tokens):
        for r in range(n):
            if not live[r]:
                continue
            if toks[r] in stop_ids:
                live[r] = False
                if stops is not None:
                    stops[r] = toks[r]
                continue
            out[r].append(toks[r])
            if len(out[r]) >= max_new_tokens or (stop_check is not None and stop_check(out[r])):
                live[r] = False
        if not any(live) or dec.row_full(0):
            break
        toks = dec.step(params).view(-1).tolist()
    return out
