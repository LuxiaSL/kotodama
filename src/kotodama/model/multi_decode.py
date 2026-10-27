"""Batched row decode: B continuations decoded in lockstep, each row at its own position.

The single-stream DecodeEngine is kernel-count bound on deep looped models (every token replays every layer's
kernels), so a step over B rows costs about what a step over one row costs. A row is filled from an engine prefill
(prefix cache included): ``admit`` copies the engine's prompt state (KV, attention mask, position) into chosen rows
and samples each row's first token from the prompt's last-position logits. Rows filled from one prefill are n
candidates of one prompt; rows filled from several prefills decode different prompts of different lengths together.

Every row owns its KV cache slot, its attention bias (which positions it attends), its position (RoPE angle and KV
write slot), its presence set (repetition penalty over the row's own generated tokens) and its sampling noise; the
weights, temperature, repetition penalty and steering rows are shared, so rows decoded together must share one
sampling law and one steering config. A row nobody reads keeps stepping on whatever it holds; its position stops at
the last cache slot, so an idle row can never write out of bounds. Pure-temperature sampling only (the tsumugi law);
truncated laws and temperature 0 stay on the single-stream path.

Weights, routing and the layer schedule are borrowed from the engine; rebuild this decoder whenever the engine is
rebuilt.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.llama import apply_rope

logger = logging.getLogger(__name__)


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

    def __init__(self, engine: DecodeEngine, rows: int, kv_from: Optional["MultiSampleDecoder"] = None) -> None:
        super().__init__()
        if rows < 2:
            raise ValueError(f"rows must be >= 2 (one row is the engine's own path), got {rows}")
        if kv_from is not None and (kv_from.rows < rows or kv_from.engine is not engine):
            raise ValueError(f"kv_from must be a decoder over the same engine with >= {rows} rows")
        self.__dict__["engine"] = engine  # borrowed, not a submodule: its buffers stay the engine's
        self.rows = int(rows)
        self.max_seq_len = engine.max_seq_len
        self.n_layers = engine.n_layers
        dev, dt = engine.device, engine.dtype
        kv_shape = (self.rows, engine.n_kv_heads, self.max_seq_len, engine.head_dim)
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
        self.register_buffer("pos", torch.zeros((self.rows,), device=dev, dtype=torch.long), persistent=False)
        self.register_buffer("row_ids", torch.arange(self.rows, device=dev, dtype=torch.long), persistent=False)
        self.register_buffer("presence", torch.zeros((self.rows, engine.config.vocab_size), device=dev,
                                                     dtype=torch.bool), persistent=False)
        self.register_buffer("cur_tokens", torch.zeros((self.rows, 1), device=dev, dtype=torch.long),
                             persistent=False)
        self.register_buffer("inv_temp", torch.tensor(1.0 / 0.9, device=dev, dtype=torch.float32), persistent=False)
        self.register_buffer("rep_penalty", torch.tensor(1.2, device=dev, dtype=torch.float32), persistent=False)
        self._cur_temp, self._cur_rp = 0.9, 1.2
        self._pos_host = [0] * self.rows  # host mirror of pos (no device syncs for capacity checks)
        self._compiled_step = None
        logger.info("MultiSampleDecoder: %d rows, kv cache %.0f MB%s", self.rows,
                    sum(b.numel() for b in self.k_caches) * 2 * torch.finfo(dt).bits / 8 / 1e6,
                    f" (shared with the {kv_from.rows}-row decoder)" if kv_from is not None else "")

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
        cos = eng.rope_cos_t.index_select(0, self.pos).unsqueeze(1)  # (B, 1, head_dim): each row's own angle
        sin = eng.rope_sin_t.index_select(0, self.pos).unsqueeze(1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k_cache, v_cache = self.k_caches[layer_idx], self.v_caches[layer_idx]
        # Row r writes its slot pos[r]: an index_put_ on the buffer itself, which inductor updates in place.
        k_cache[self.row_ids, :, self.pos] = k[:, :, 0]
        v_cache[self.row_ids, :, self.pos] = v[:, :, 0]
        out = F.scaled_dot_product_attention(q, k_cache, v_cache, attn_mask=self.attn_bias, enable_gqa=True)
        out = out.transpose(1, 2).contiguous().view(B, 1, -1)
        return F.linear(out, attn.o_proj.weight)

    def _forward_step(self, tokens: torch.Tensor) -> torch.Tensor:
        """One decode forward over all rows: tokens (B, 1) -> logits (B, V). Mirrors DecodeEngine._forward_step."""
        eng = self.engine
        model = eng.model
        self.attn_bias[self.row_ids, 0, 0, self.pos] = 0.0  # each row opens its own position before attending
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
        # Live rows never reach the last slot (callers budget each row); an idle row parks there.
        self.pos.copy_(torch.clamp(self.pos + 1, max=self.max_seq_len - 1))
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
    def admit(self, rows: Sequence[int], prefill_logits: torch.Tensor, params: SamplingParams) -> list[int]:
        """Fill ``rows`` from the engine's prefilled prompt and sample each row's first token from the prompt's
        last-position logits (1, V). The whole KV slot is copied, so a row keeps nothing from what it decoded
        before. Writes the first tokens into ``cur_tokens[rows]`` (the next step feeds them) and returns them."""
        eng = self.engine
        rows = [int(r) for r in rows]
        if not rows or len(set(rows)) != len(rows) or not all(0 <= r < self.rows for r in rows):
            raise ValueError(f"rows must be distinct indices in [0, {self.rows}), got {rows}")
        if eng._pos_cpu <= 0 or eng._pos_cpu >= self.max_seq_len:
            raise RuntimeError(f"engine position {eng._pos_cpu} is not a prefilled prompt")
        self._sync_sampling(params)
        n = len(rows)
        idx = torch.tensor(rows, device=self.pos.device, dtype=torch.long)
        contiguous = rows == list(range(rows[0], rows[0] + n))
        for li in range(self.n_layers):
            for dst, src in ((self.k_caches[li], eng.k_caches[li]), (self.v_caches[li], eng.v_caches[li])):
                if contiguous:
                    dst[rows[0]:rows[0] + n].copy_(src.expand(n, -1, -1, -1))
                else:
                    dst.index_copy_(0, idx, src.expand(n, -1, -1, -1).contiguous())
        self.attn_bias.index_copy_(0, idx, eng.attn_bias.expand(n, -1, -1, -1).contiguous())
        self.pos.index_fill_(0, idx, eng._pos_cpu)
        # A fresh row's presence set is empty, so the penalty is the identity on its first token.
        scaled = prefill_logits.reshape(1, -1).float().expand(n, -1) * self.inv_temp
        u = torch.rand_like(scaled)
        gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
        first = (scaled + gumbel).argmax(dim=-1, keepdim=True)  # (n, 1)
        self.presence.index_fill_(0, idx, False)
        self.presence.index_put_((idx, first.view(-1)), torch.ones(n, dtype=torch.bool, device=idx.device))
        self.cur_tokens.index_copy_(0, idx, first)
        for r in rows:
            self._pos_host[r] = eng._pos_cpu
        return first.view(-1).tolist()

    @torch.inference_mode()
    def start(self, prefill_logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        """Admit every row from the engine's prefilled prompt (n candidates of one prompt). Returns cur_tokens."""
        self.admit(range(self.rows), prefill_logits, params)
        return self.cur_tokens

    @torch.inference_mode()
    def step(self, params: SamplingParams) -> torch.Tensor:
        """Feed every row its last token, sample the next. Returns cur_tokens (B, 1)."""
        self._sync_sampling(params)
        fn = self._compiled_step or self._step_sampled
        out = fn(self.cur_tokens)
        self._advance_host()
        return out

    @torch.inference_mode()
    def step_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward only (parity testing): tokens (B, 1) -> logits (B, V), eager."""
        out = self._forward_step(tokens)
        self._advance_host()
        return out

    def _advance_host(self) -> None:
        self._pos_host = [p + 1 for p in self._pos_host]

    def compile_step(self, mode: str = "reduce-overhead") -> None:
        import torch._dynamo

        torch._dynamo.mark_static_address(self.cur_tokens)
        for buf in self.k_caches + self.v_caches:
            torch._dynamo.mark_static_address(buf)
        self._compiled_step = torch.compile(self._step_sampled, mode=mode, fullgraph=True)

    def row_pos(self, row: int) -> int:
        """Row ``row``'s next write position (its prompt length plus the tokens it has been fed). A row at
        ``max_seq_len`` is full; the device position of such a row stays parked at the last slot."""
        return self._pos_host[row]


def decode_rows(dec: MultiSampleDecoder, prefill_logits: torch.Tensor, params: SamplingParams, n: int,
                max_new_tokens: int, stop_ids: frozenset[int],
                stop_check: Optional[Callable[[list[int]], bool]] = None) -> list[list[int]]:
    """Generate up to ``max_new_tokens`` per row for the first ``n`` rows of one prompt; a row ends at a stop token
    (not kept), when ``stop_check(row_ids)`` is true, or at capacity. Returns each row's generated ids."""
    if not 1 <= n <= dec.rows:
        raise ValueError(f"n={n} outside [1, {dec.rows}]")
    out: list[list[int]] = [[] for _ in range(n)]
    live = [True] * n
    dec.start(prefill_logits, params)
    toks = dec.cur_tokens.view(-1).tolist()
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
        if not any(live) or dec.row_pos(0) >= dec.max_seq_len:
            break
        toks = dec.step(params).view(-1).tolist()
    return out
