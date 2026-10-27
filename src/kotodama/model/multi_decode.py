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
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kotodama.model.cascade_attn import attend_for, merge, pack_group_queries, unpack_group_outputs
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
        dev = engine.device
        self._alloc_kv(kv_from)
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

    def _kv_buffers(self) -> list[torch.Tensor]:
        """The KV storage the compiled step writes in place (static addresses for the CUDA graph)."""
        return self.k_caches + self.v_caches

    def kv_bytes(self) -> int:
        """Bytes of KV storage this decoder addresses (views of a wider decoder's storage included)."""
        return sum(b.numel() * b.element_size() for b in self._kv_buffers())

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
        out = self._attend(layer_idx, q, k, v)  # (B, 1, n_heads * head_dim)
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
        self.attn_bias[self.row_ids, 0, 0, self.pos] = 0.0

    def _advance_step(self) -> None:
        # Live rows never reach the last slot (callers budget each row); an idle row parks there.
        self.pos.copy_(torch.clamp(self.pos + 1, max=self.max_seq_len - 1))

    def _forward_step(self, tokens: torch.Tensor) -> torch.Tensor:
        """One decode forward over all rows: tokens (B, 1) -> logits (B, V). Mirrors DecodeEngine._forward_step."""
        eng = self.engine
        model = eng.model
        self._open_step()
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
        self._advance_step()
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
        last-position logits (1, V). The prompt's KV positions are copied and the mask is the engine's (open on the
        prompt only), so a row attends to nothing it decoded before: its later positions are closed until its own
        decode steps rewrite and open them. Writes the first tokens into ``cur_tokens[rows]`` (the next step feeds
        them) and returns them."""
        eng = self.engine
        rows = self._check_rows(rows)
        if eng._pos_cpu <= 0 or eng._pos_cpu >= self.max_seq_len:
            raise RuntimeError(f"engine position {eng._pos_cpu} is not a prefilled prompt")
        self._sync_sampling(params)
        n = len(rows)
        p = eng._pos_cpu
        idx = torch.tensor(rows, device=self.pos.device, dtype=torch.long)
        contiguous = rows == list(range(rows[0], rows[0] + n))
        for li in range(self.n_layers):
            for dst, src in ((self.k_caches[li], eng.k_caches[li]), (self.v_caches[li], eng.v_caches[li])):
                if contiguous:
                    dst[rows[0]:rows[0] + n, :, :p].copy_(src[:, :, :p].expand(n, -1, -1, -1))
                else:  # one indexed write, the prompt broadcast over the rows (no materialized copies)
                    dst[:, :, :p].index_put_((idx,), src[:, :, :p].expand(n, -1, -1, -1))
        self.attn_bias.index_copy_(0, idx, eng.attn_bias.expand(n, -1, -1, -1).contiguous())
        return self._seed(rows, idx, eng._pos_cpu, prefill_logits)

    @torch.inference_mode()
    def restart(self, rows: Sequence[int], prompt_len: int, prefill_logits: torch.Tensor,
                params: SamplingParams) -> list[int]:
        """Start a fresh candidate in ``rows``, which already hold a ``prompt_len``-token prompt in their first
        cache slots (they were admitted from it and have only decoded since): the rows rewind to the prompt's end
        (positions past it are closed in the mask and rewritten as decoding proceeds) and sample a first token from
        that prompt's last-position logits (1, V). Costs no KV copy. Returns the first tokens."""
        rows = self._check_rows(rows)
        if not 0 < prompt_len < self.max_seq_len:
            raise ValueError(f"prompt_len {prompt_len} outside (0, {self.max_seq_len})")
        self._sync_sampling(params)
        idx = torch.tensor(rows, device=self.pos.device, dtype=torch.long)
        self.attn_bias[idx, :, :, prompt_len:] = float("-inf")
        return self._seed(rows, idx, prompt_len, prefill_logits)

    def _check_rows(self, rows: Sequence[int]) -> list[int]:
        rows = [int(r) for r in rows]
        if not rows or len(set(rows)) != len(rows) or not all(0 <= r < self.rows for r in rows):
            raise ValueError(f"rows must be distinct indices in [0, {self.rows}), got {rows}")
        return rows

    def _seed(self, rows: list[int], idx: torch.Tensor, pos: int, prefill_logits: torch.Tensor) -> list[int]:
        """Put ``rows`` at position ``pos`` with an empty presence set and their first token sampled."""
        n = len(rows)
        self.pos.index_fill_(0, idx, pos)
        # A fresh row's presence set is empty, so the penalty is the identity on its first token.
        scaled = prefill_logits.reshape(1, -1).float().expand(n, -1) * self.inv_temp
        u = torch.rand_like(scaled)
        gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
        first = (scaled + gumbel).argmax(dim=-1, keepdim=True)  # (n, 1)
        self.presence.index_fill_(0, idx, False)
        self.presence.index_put_((idx, first.view(-1)), torch.ones(n, dtype=torch.bool, device=idx.device))
        self.cur_tokens.index_copy_(0, idx, first)
        for r in rows:
            self._pos_host[r] = pos
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
        for buf in self._kv_buffers():
            torch._dynamo.mark_static_address(buf)
        self._compiled_step = torch.compile(self._step_sampled, mode=mode, fullgraph=True)

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
                 tail_tokens: int = 1024, kv_from: Optional["CascadeDecoder"] = None) -> None:
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
        super().__init__(engine, rows, kv_from=kv_from)
        self.attend_backend, self._attend_fn = attend_for(engine.device, engine.dtype)
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
    def admit(self, rows: Sequence[int], prefill_logits: torch.Tensor, params: SamplingParams) -> list[int]:
        """Fill ``rows`` from the engine's prefilled prompt: the prompt's KV is copied once into a free group (one
        group per ``group_rows`` rows), each row's tail starts empty, and each row's first token is sampled from the
        prompt's last-position logits (1, V). Raises RuntimeError when too few groups are free (``admit_capacity``).
        Returns the first tokens."""
        eng = self.engine
        rows = self._check_rows(rows)
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
        return self._seed(rows, idx, p, prefill_logits)

    @torch.inference_mode()
    def restart(self, rows: Sequence[int], prompt_len: int, prefill_logits: torch.Tensor,
                params: SamplingParams) -> list[int]:
        """Start a fresh candidate in ``rows``, which still belong to a group holding a ``prompt_len``-token prompt
        (they were admitted from it and not released): the tails empty and each row samples a first token from that
        prompt's last-position logits (1, V). Costs no KV copy. Returns the first tokens."""
        rows = self._check_rows(rows)
        for r in rows:
            g = self._row_group[r]
            if g < 0 or self._group_len[g] != prompt_len:
                raise ValueError(f"row {r} holds no {prompt_len}-token prompt")
        self._sync_sampling(params)
        idx = torch.tensor(rows, device=self.pos.device, dtype=torch.long)
        self.tail_pos.index_fill_(0, idx, 0)
        for r in rows:
            self._tail_host[r] = 0
        return self._seed(rows, idx, prompt_len, prefill_logits)


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
        if not any(live) or dec.row_full(0):
            break
        toks = dec.step(params).view(-1).tolist()
    return out
