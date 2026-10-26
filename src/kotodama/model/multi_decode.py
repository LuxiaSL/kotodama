"""Batched multi-sample decode: n continuations of ONE prompt, decoded in lockstep.

The single-stream DecodeEngine is kernel-count bound on deep looped models (every token replays every layer's
kernels), so a step over B rows costs about what a step over one row costs. A request for n candidates therefore
prefills once on the engine (prefix cache included), copies that KV into B rows here, and decodes all rows at once.

Every row sits at the same position: the attention bias, ``pos`` and the rope tables stay shared; only the tokens,
the per-row KV after the prompt, the per-row presence set (repetition penalty over each row's own generated tokens)
and the sampling noise differ. Rows beyond the request's n are decoded and ignored, so one compiled graph serves
every n <= B. Steering reads the engine's ``steer_buf`` (same request, same writes for every row). Pure-temperature
sampling only (the tsumugi law); truncated laws and temperature 0 stay on the single-stream path.

Weights, routing and the layer schedule are borrowed from the engine; rebuild this decoder whenever the engine is
rebuilt.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.llama import apply_rope

logger = logging.getLogger(__name__)


class MultiSampleDecoder(nn.Module):
    """B-row lockstep decoder over a DecodeEngine's weights.

    Usage (the engine has just prefilled the prompt):
        dec = MultiSampleDecoder(engine, rows=8); dec.compile_step()   # once
        toks = dec.start(prefill_logits, params)                        # (B, 1): each row's first token
        while ...:
            toks = dec.step(params)                                     # (B, 1): next token per row
    Returned token tensors are the persistent ``cur_tokens`` buffer: consume (``.tolist()``) before the next step.
    """

    def __init__(self, engine: DecodeEngine, rows: int) -> None:
        super().__init__()
        if rows < 2:
            raise ValueError(f"rows must be >= 2 (one row is the engine's own path), got {rows}")
        self.__dict__["engine"] = engine  # borrowed, not a submodule: its buffers stay the engine's
        self.rows = int(rows)
        self.max_seq_len = engine.max_seq_len
        self.n_layers = engine.n_layers
        dev, dt = engine.device, engine.dtype
        kv_shape = (self.rows, engine.n_kv_heads, self.max_seq_len, engine.head_dim)
        for li in range(self.n_layers):
            self.register_buffer(f"k_cache_{li}", torch.zeros(kv_shape, device=dev, dtype=dt), persistent=False)
            self.register_buffer(f"v_cache_{li}", torch.zeros(kv_shape, device=dev, dtype=dt), persistent=False)
        self.k_caches = [getattr(self, f"k_cache_{li}") for li in range(self.n_layers)]
        self.v_caches = [getattr(self, f"v_cache_{li}") for li in range(self.n_layers)]
        self.register_buffer("attn_bias", torch.full((1, 1, 1, self.max_seq_len), float("-inf"), device=dev, dtype=dt),
                             persistent=False)
        self.register_buffer("pos", torch.zeros((1,), device=dev, dtype=torch.long), persistent=False)
        self.register_buffer("presence", torch.zeros((self.rows, engine.config.vocab_size), device=dev,
                                                     dtype=torch.bool), persistent=False)
        self.register_buffer("cur_tokens", torch.zeros((self.rows, 1), device=dev, dtype=torch.long),
                             persistent=False)
        self.register_buffer("inv_temp", torch.tensor(1.0 / 0.9, device=dev, dtype=torch.float32), persistent=False)
        self.register_buffer("rep_penalty", torch.tensor(1.2, device=dev, dtype=torch.float32), persistent=False)
        self._cur_temp, self._cur_rp = 0.9, 1.2
        self._pos_cpu = 0
        self._compiled_step = None
        logger.info("MultiSampleDecoder: %d rows, kv cache %.0f MB", self.rows,
                    sum(b.numel() for b in self.k_caches) * 2 * torch.finfo(dt).bits / 8 / 1e6)

    @staticmethod
    def supports(params: SamplingParams) -> bool:
        """The laws this decoder samples: pure temperature (> 0) with any repetition penalty."""
        return params.temperature > 0.0 and not params.truncated

    # ── the batched decode step ─────────────────────────────────────────────

    def _attn(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
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
        cos = eng.rope_cos_t.index_select(0, self.pos)
        sin = eng.rope_sin_t.index_select(0, self.pos)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k_cache, v_cache = self.k_caches[layer_idx], self.v_caches[layer_idx]
        k_cache.index_copy_(2, self.pos, k)
        v_cache.index_copy_(2, self.pos, v)
        out = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=self.attn_bias, enable_gqa=True)
        out = out.transpose(1, 2).contiguous().view(B, 1, -1)
        return F.linear(out, attn.o_proj.weight)

    def _forward_step(self, tokens: torch.Tensor) -> torch.Tensor:
        """One decode forward over all rows: tokens (B, 1) -> logits (B, V). Mirrors DecodeEngine._forward_step."""
        eng = self.engine
        model = eng.model
        self.attn_bias.index_fill_(3, self.pos, 0.0)
        embed = model.embed_tokens(tokens)  # (B, 1, D)
        committed: list[torch.Tensor] = []
        partial = embed
        for ls in eng.schedule:
            layer = model.layers[ls.layer_idx]
            if committed:
                h_attn = eng._route(committed + [partial], layer.attn_res_query, layer.attn_res_norm.weight)
            else:
                h_attn = partial
            if ls.is_boundary:
                committed.append(partial)
                partial = torch.zeros_like(embed)
            partial = partial + self._attn(ls.layer_idx, layer.attn_norm(h_attn))
            if eng.steer_enabled:
                partial = partial + eng.steer_buf[ls.layer_idx]
            h_mlp = eng._route(committed + [partial], layer.mlp_res_query, layer.mlp_res_norm.weight)
            gu = F.linear(layer.ffn_norm(h_mlp), eng.w_gate_ups[ls.layer_idx])
            gate, up = gu.split(eng.config.intermediate_size, dim=-1)
            partial = partial + F.linear(F.silu(gate) * up, layer.ffn.down_proj.weight)
        x = eng._route(committed + [partial], model.final_res_query, model.final_res_norm.weight)
        x = model.norm(x)
        logits = F.linear(x, model.get_lm_head_weight())  # (B, 1, V)
        self.pos.add_(1)
        return logits[:, -1]

    def _sample_rows(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-row repetition penalty over the row's own presence set, then Gumbel-max at 1/inv_temp."""
        logits = logits.float()
        rp = self.rep_penalty
        penalized = torch.where(logits > 0, logits / rp, logits * rp)
        logits = torch.where(self.presence, penalized, logits)
        scaled = logits * self.inv_temp
        u = torch.rand_like(scaled)
        gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
        nxt = (scaled + gumbel).argmax(dim=-1, keepdim=True)  # (B, 1)
        self.presence.scatter_(1, nxt, True)
        self.cur_tokens.copy_(nxt)
        return self.cur_tokens

    def _step_sampled(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._sample_rows(self._forward_step(tokens))

    # ── public API ──────────────────────────────────────────────────────────

    def _sync_sampling(self, params: SamplingParams) -> None:
        if not self.supports(params):
            raise ValueError("MultiSampleDecoder samples pure-temperature laws only (temperature > 0, no top-k/top-p)")
        if params.temperature != self._cur_temp:
            self.inv_temp.fill_(1.0 / params.temperature)
            self._cur_temp = params.temperature
        if params.repetition_penalty != self._cur_rp:
            self.rep_penalty.fill_(params.repetition_penalty)
            self._cur_rp = params.repetition_penalty

    @torch.inference_mode()
    def start(self, prefill_logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Take the engine's prefilled state into every row and sample each row's first token from the prompt's
        last-position logits (1, V). Returns cur_tokens (B, 1)."""
        eng = self.engine
        self._sync_sampling(params)
        for li in range(self.n_layers):
            self.k_caches[li].copy_(eng.k_caches[li].expand_as(self.k_caches[li]))
            self.v_caches[li].copy_(eng.v_caches[li].expand_as(self.v_caches[li]))
        self.attn_bias.copy_(eng.attn_bias)
        self.pos.copy_(eng.pos)
        self._pos_cpu = eng._pos_cpu
        self.presence.zero_()
        return self._sample_rows(prefill_logits.reshape(1, -1).expand(self.rows, -1))

    @torch.inference_mode()
    def step(self, params: SamplingParams) -> torch.Tensor:
        """Feed every row its last token, sample the next. Returns cur_tokens (B, 1)."""
        if self._pos_cpu >= self.max_seq_len:
            raise RuntimeError(f"decode position {self._pos_cpu} would exceed max_seq_len {self.max_seq_len}")
        self._sync_sampling(params)
        fn = self._compiled_step or self._step_sampled
        out = fn(self.cur_tokens)
        self._pos_cpu += 1
        return out

    @torch.inference_mode()
    def step_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward only (parity testing): tokens (B, 1) -> logits (B, V), eager."""
        out = self._forward_step(tokens)
        self._pos_cpu += 1
        return out

    def compile_step(self, mode: str = "reduce-overhead") -> None:
        import torch._dynamo

        torch._dynamo.mark_static_address(self.cur_tokens)
        self._compiled_step = torch.compile(self._step_sampled, mode=mode, fullgraph=True)

    @property
    def pos_cpu(self) -> int:
        return self._pos_cpu


def decode_rows(dec: MultiSampleDecoder, prefill_logits: torch.Tensor, params: SamplingParams, n: int,
                max_new_tokens: int, stop_ids: frozenset[int],
                stop_check: Optional[Callable[[list[int]], bool]] = None) -> list[list[int]]:
    """Generate up to ``max_new_tokens`` per row for the first ``n`` rows; a row ends at a stop token (not kept),
    when ``stop_check(row_ids)`` is true, or at capacity. Returns each row's generated ids."""
    if not 1 <= n <= dec.rows:
        raise ValueError(f"n={n} outside [1, {dec.rows}]")
    out: list[list[int]] = [[] for _ in range(n)]
    live = [True] * n
    toks = dec.start(prefill_logits, params).view(-1).tolist()
    for _ in range(max_new_tokens):
        for r in range(n):
            if not live[r]:
                continue
            if toks[r] in stop_ids:
                live[r] = False
                continue
            out[r].append(toks[r])
            if len(out[r]) >= max_new_tokens or (stop_check is not None and stop_check(out[r])):
                live[r] = False
        if not any(live) or dec.pos_cpu >= dec.max_seq_len:
            break
        toks = dec.step(params).view(-1).tolist()
    return out
