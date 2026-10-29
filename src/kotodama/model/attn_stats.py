"""Attention statistics of the cascade decoder's step without materialising its probabilities, from the attention's
own log-normalisers (each row's lse over its prompt and tail, which the cascade attention returns): p = exp(s - lse)
in one pass over the keys, in two Triton launches that read keys the way the attention does and write the stats
buffers in place.

- Prompt part: program (g, j, u) takes group g's prompt column tiles j, j + splits, ... for the u-th block of
  ``sub`` member slots, and per key head scores those slots' stacked queries (each member row's query heads on that
  key head) against the tile in one dot, so a prompt key is read once per block of rows, not once per row. A block
  with no slot a row reads from (only the entries repeating a member, which fill a group past its rows) exits at
  once. It leaves partial sums per group slot and split, and writes the head-mean row's prompt columns.
- Row part: program b scores row b's own tail keys, adds its prompt slot's partials (in split order) and then its
  tail's, and writes the row's statistics, the head-mean row's tail columns and zeros past them.

Per row and head: entropy (as lse - sum p s), the mass on prompt columns (< the row's position at admission), the
recency mass (columns >= max(1, floor(0.8 (pos + 1)))), the sink (column 0); per row: the entropy of the head-mean
distribution and, at sampled layers, the head-mean row over the first ``width`` absolute columns. A query's scores
are its own dot products, and which tiles a program reads and the order its partials are added depend only on its
group's and row's lengths, so a row's statistics are the same whatever shares its batch. The arithmetic differs from
the reference (``multi_decode._cascade_probs`` + ``stats_write``) by rounding: scores are bf16 dot products
accumulated in fp32, the sums run in tile order, and the normaliser is the attention kernel's (its p sums to 1 within
that kernel's rounding).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON = True
except ImportError:  # CPU-only installs: the reference path computes the statistics
    _TRITON = False

# The launch geometry. Part of the arithmetic (the partial sums' grouping), so fixed rather than tuned per batch.
BLOCK = 32  # columns per tile
SPLITS = 16  # prompt-part programs per group and block of slots
SUB = 8  # member slots per prompt-part program (x the query heads per key head = the dot's rows, >= 16)
WARPS = 4
STAGES = 1  # the prompt part's tile-loop pipelining


def available(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the fused kernel runs for tensors on ``device`` in ``dtype``."""
    return _TRITON and device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)


if _TRITON:

    @triton.jit
    def _prompt_stats_kernel(
        q_ptr, pk_ptr, plen_ptr, mem_ptr, rslot_ptr, s0_ptr, pos_ptr, lse_ptr,
        ps_ptr, pm_ptr, rec_ptr, sink_ptr, ment_ptr,
        mrow_ptr, rid_ptr, slot_ptr, scol_ptr,
        stride_qb, stride_qh, stride_pg, stride_pw, stride_ph, stride_mr0, stride_mr1, stride_mr2,
        SCALE: tl.constexpr,  # a constexpr: a runtime float would arrive as fp64 inside a compiled graph
        M: tl.constexpr, MP: tl.constexpr, H: tl.constexpr, R: tl.constexpr, RP: tl.constexpr, HK: tl.constexpr,
        HKP: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr, WIDTH: tl.constexpr,
        SAMPLED: tl.constexpr, STAGES: tl.constexpr,
    ):
        """Program (g, j, u): partial sums per group slot (g * M + i) and head over group g's prompt tiles j, j + NS,
        ... for slots i in [u * MP, (u + 1) * MP). Query index x = (i - u * MP) * RP + c is slot i's query head
        hk * R + c on key head hk."""
        g = tl.program_id(0)
        j = tl.program_id(1)
        u = tl.program_id(2)
        slots = u * MP + tl.arange(0, MP)
        # the head-mean row is written only by the slot a row reads its prompt part from (row_slot), not by an
        # unused entry repeating it; a block with no such slot has nothing anyone reads
        rows = tl.load(mem_ptr + g * M + slots, mask=slots < M, other=0)
        own = (slots < M) & (tl.load(rslot_ptr + rows, mask=slots < M, other=-1) == g * M + slots)
        if tl.max(own.to(tl.int32), axis=0) == 0:
            return
        plen = tl.load(plen_ptr + g).to(tl.int32)
        xs = tl.arange(0, MP * RP)
        slot = u * MP + xs // RP
        c = xs % RP
        xmask = (slot < M) & (c < R)
        row = tl.load(mem_ptr + g * M + slot, mask=slot < M, other=0)
        s0 = tl.load(s0_ptr + row, mask=xmask, other=0).to(tl.int32)
        cut = tl.maximum(((tl.load(pos_ptr + row, mask=xmask, other=0) + 1).to(tl.float32) * 0.8).to(tl.int32), 1)
        ds = tl.arange(0, D)
        ns = tl.arange(0, BN)
        hks = tl.arange(0, HKP)
        ps = tl.zeros([HKP, MP * RP], dtype=tl.float32)
        pm = tl.zeros([HKP, MP * RP], dtype=tl.float32)
        rec = tl.zeros([HKP, MP * RP], dtype=tl.float32)
        sink = tl.zeros([HKP, MP * RP], dtype=tl.float32)
        ment = tl.zeros([MP], dtype=tl.float32)
        mrow_base = (tl.load(rid_ptr + rows, mask=own, other=0) * stride_mr0
                     + tl.load(slot_ptr + rows, mask=own, other=0) * stride_mr1 + tl.load(scol_ptr) * stride_mr2)
        # each tile reads every key head's stacked queries again (from cache): pipelining them would buffer
        # STAGES copies of all of them in shared memory
        for c0 in tl.range(j * BN, plen, NS * BN, num_stages=STAGES):
            col = c0 + ns
            valid = col < plen
            live = xmask[:, None] & valid[None, :]
            mc = tl.zeros([MP, BN], dtype=tl.float32)
            for hk in tl.static_range(HK):
                head = hk * R + c
                q = tl.load(q_ptr + row[:, None] * stride_qb + head[:, None] * stride_qh + ds[None, :],
                            mask=xmask[:, None], other=0.0)
                k = tl.load(pk_ptr + g * stride_pg + col[:, None] * stride_pw + hk * stride_ph + ds[None, :],
                            mask=valid[:, None], other=0.0)
                s = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
                lse = tl.load(lse_ptr + row * H + head, mask=xmask, other=0.0)
                p = tl.where(live, tl.exp(s - lse[:, None]), 0.0)
                at = (hks == hk)[:, None]
                ps = tl.where(at, ps + tl.sum(tl.where(live, p * s, 0.0), axis=1)[None, :], ps)
                pm = tl.where(at, pm + tl.sum(tl.where((col[None, :] < s0[:, None]), p, 0.0), axis=1)[None, :], pm)
                rec = tl.where(at, rec + tl.sum(tl.where((col[None, :] >= cut[:, None]), p, 0.0), axis=1)[None, :],
                               rec)
                sink = tl.where(at, sink + tl.sum(tl.where((col == 0)[None, :], p, 0.0), axis=1)[None, :], sink)
                mc += tl.sum(tl.reshape(p, [MP, RP, BN]), axis=1)
            mc = mc / H
            ment += tl.sum(tl.where(mc > 0, -mc * tl.log(tl.where(mc > 0, mc, 1.0)), 0.0), axis=1)
            if SAMPLED:
                tl.store(mrow_ptr + mrow_base[:, None] + col[None, :], mc,
                         mask=own[:, None] & (valid & (col < WIDTH))[None, :])
        heads = hks[:, None] * R + c[None, :]
        out = ((g * M + slot[None, :]) * NS + j) * H + heads
        omask = (hks < HK)[:, None] & xmask[None, :]
        tl.store(ps_ptr + out, ps, mask=omask)
        tl.store(pm_ptr + out, pm, mask=omask)
        tl.store(rec_ptr + out, rec, mask=omask)
        tl.store(sink_ptr + out, sink, mask=omask)
        tl.store(ment_ptr + (g * M + slots) * NS + j, ment, mask=slots < M)

    @triton.jit
    def _row_stats_kernel(
        q_ptr, tk_ptr, plen_ptr, rslot_ptr, tpos_ptr, s0_ptr, pos_ptr, lse_ptr,
        ps_ptr, pm_ptr, rec_ptr, sink_ptr, ment_ptr,
        hent_ptr, ment_out_ptr, hsc_ptr, mrow_ptr, rid_ptr, slot_ptr, lcol_ptr, scol_ptr,
        stride_qb, stride_qh, stride_tb, stride_tt, stride_th,
        stride_he0, stride_he1, stride_he2, stride_me0, stride_me1, stride_me2,
        stride_hs0, stride_hs1, stride_hs2, stride_mr0, stride_mr1, stride_mr2,
        SCALE: tl.constexpr, M: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, R: tl.constexpr,
        HK: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, NS: tl.constexpr, WIDTH: tl.constexpr,
        SAMPLED: tl.constexpr,
    ):
        """Program b: row b's sums over its own tail keys (absolute columns plen + t, t <= tail_pos), added after its
        prompt slot's partials, written into the stats buffers at (row id, slot, layer column)."""
        b = tl.program_id(0)
        rs = tl.load(rslot_ptr + b)
        plen = tl.load(plen_ptr + rs // M).to(tl.int32)
        n_tail = tl.load(tpos_ptr + b).to(tl.int32) + 1
        s0 = tl.load(s0_ptr + b).to(tl.int32)
        pos = tl.load(pos_ptr + b).to(tl.int32)
        cut = tl.maximum(((pos + 1).to(tl.float32) * 0.8).to(tl.int32), 1)
        rid = tl.load(rid_ptr + b)
        sl = tl.load(slot_ptr + b)
        hs = tl.arange(0, HP)
        hmask = hs < H
        head_kv = hs // R
        ds = tl.arange(0, D)
        ns = tl.arange(0, BN)
        q = tl.load(q_ptr + b * stride_qb + hs[:, None] * stride_qh + ds[None, :], mask=hmask[:, None], other=0.0)
        lse = tl.load(lse_ptr + b * H + hs, mask=hmask, other=0.0)
        mrow_base = rid * stride_mr0 + sl * stride_mr1 + tl.load(scol_ptr) * stride_mr2
        tps = tl.zeros([HP], dtype=tl.float32)
        tpm = tl.zeros([HP], dtype=tl.float32)
        trec = tl.zeros([HP], dtype=tl.float32)
        tsink = tl.zeros([HP], dtype=tl.float32)
        tment = tl.zeros([BN], dtype=tl.float32)
        for t0 in range(0, n_tail, BN):
            t = t0 + ns
            valid = t < n_tail
            col = plen + t
            s = tl.zeros([HP, BN], dtype=tl.float32)
            for hk in tl.static_range(HK):
                k = tl.load(tk_ptr + b * stride_tb + t[:, None] * stride_tt + hk * stride_th + ds[None, :],
                            mask=valid[:, None], other=0.0)
                s = tl.where((head_kv == hk)[:, None], tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE, s)
            live = valid[None, :] & hmask[:, None]
            p = tl.where(live, tl.exp(s - lse[:, None]), 0.0)
            tps += tl.sum(tl.where(live, p * s, 0.0), axis=1)
            tpm += tl.sum(tl.where((col < s0)[None, :], p, 0.0), axis=1)
            trec += tl.sum(tl.where((col >= cut)[None, :], p, 0.0), axis=1)
            tsink += tl.sum(tl.where((col == 0)[None, :], p, 0.0), axis=1)
            mc = tl.sum(p, axis=0) / H
            tment += tl.where(mc > 0, -mc * tl.log(tl.where(mc > 0, mc, 1.0)), 0.0)
            if SAMPLED:
                tl.store(mrow_ptr + mrow_base + col, mc, mask=valid & (col < WIDTH))
        if SAMPLED:  # the head-mean row is zero past the row's keys (a re-admitted row reuses its slots)
            for z0 in range(plen + n_tail, WIDTH, BN):
                zc = z0 + ns
                tl.store(mrow_ptr + mrow_base + zc, tl.zeros([BN], dtype=tl.float32), mask=zc < WIDTH)
        # the prompt slot's partials in split order, then the tail's
        part = rs * NS * H + hs
        ps = tl.load(ps_ptr + part, mask=hmask, other=0.0)
        pm = tl.load(pm_ptr + part, mask=hmask, other=0.0)
        rec = tl.load(rec_ptr + part, mask=hmask, other=0.0)
        sink = tl.load(sink_ptr + part, mask=hmask, other=0.0)
        ment = tl.load(ment_ptr + rs * NS)
        for jj in range(1, NS):
            ps += tl.load(ps_ptr + part + jj * H, mask=hmask, other=0.0)
            pm += tl.load(pm_ptr + part + jj * H, mask=hmask, other=0.0)
            rec += tl.load(rec_ptr + part + jj * H, mask=hmask, other=0.0)
            sink += tl.load(sink_ptr + part + jj * H, mask=hmask, other=0.0)
            ment += tl.load(ment_ptr + rs * NS + jj)
        lc = tl.load(lcol_ptr)
        tl.store(hent_ptr + rid * stride_he0 + sl * stride_he1 + lc * stride_he2 + hs, lse - (ps + tps), mask=hmask)
        tl.store(ment_out_ptr + rid * stride_me0 + sl * stride_me1 + lc * stride_me2, ment + tl.sum(tment, axis=0))
        if SAMPLED:
            base = rid * stride_hs0 + sl * stride_hs1 + tl.load(scol_ptr) * stride_hs2
            tl.store(hsc_ptr + base + hs, pm + tpm, mask=hmask)
            tl.store(hsc_ptr + base + H + hs, rec + trec, mask=hmask)
            tl.store(hsc_ptr + base + 2 * H + hs, sink + tsink, mask=hmask)


def cascade_stats(st: tuple, q: torch.Tensor, kv: tuple[torch.Tensor, ...], state: tuple[torch.Tensor, ...],
                  sampled: bool, lse: torch.Tensor, block: int = BLOCK, splits: int = SPLITS,
                  warps: int = WARPS, stages: int = STAGES, sub: int = SUB) -> None:
    """One layer's attention statistics into the stats buffers, as ``stats_write(st, _cascade_probs(q, kv, state),
    sampled)`` writes them: q (B, heads, 1, head_dim) after the attention's KV write, kv = (tail_k, tail_v,
    prompt_k, prompt_v), state = (tail_pos, members, row_slot, prompt_lens, row_ids), st the stats tuple, lse
    (B, heads) fp32 each row's log-normaliser over its prompt and tail (the cascade attention's ``lse=True``). The
    buffers' last dimension must be contiguous (signing.allocate's are)."""
    head_ent, mean_ent, head_sc, mean_rows, row_ids, slot, layer_col, sampled_col, start, pos = st
    tail_k, _, prompt_k, _ = kv
    tail_pos, members, row_slot, prompt_lens, _ = state
    B, H, _, D = q.shape
    hk = tail_k.shape[2]
    G, M = members.shape
    width = mean_rows.shape[-1]
    qr = q[:, :, 0].contiguous()
    lse = lse.float().contiguous()
    scol = sampled_col if sampled else layer_col  # read only at sampled layers
    f32 = dict(device=q.device, dtype=torch.float32)
    common = dict(SCALE=D ** -0.5, H=H, R=H // hk, HK=hk, D=D, BN=block, NS=splits, WIDTH=width,
                  SAMPLED=bool(sampled))
    pp = [torch.empty((G * M, splits, H), **f32) for _ in range(4)]
    pment = torch.empty((G * M, splits), **f32)
    _prompt_stats_kernel[(G, splits, triton.cdiv(M, sub))](
        qr, prompt_k, prompt_lens, members, row_slot, start, pos, lse, *pp, pment,
        mean_rows, row_ids, slot, scol,
        qr.stride(0), qr.stride(1), prompt_k.stride(0), prompt_k.stride(1), prompt_k.stride(2),
        mean_rows.stride(0), mean_rows.stride(1), mean_rows.stride(2),
        M=M, MP=sub, RP=triton.next_power_of_2(H // hk), HKP=triton.next_power_of_2(hk),
        STAGES=stages, num_warps=warps, **common)
    _row_stats_kernel[(B,)](
        qr, tail_k, prompt_lens, row_slot, tail_pos, start, pos, lse, *pp, pment,
        head_ent, mean_ent, head_sc, mean_rows, row_ids, slot, layer_col, scol,
        qr.stride(0), qr.stride(1), tail_k.stride(0), tail_k.stride(1), tail_k.stride(2),
        *head_ent.stride()[:3], *mean_ent.stride()[:3], *head_sc.stride()[:3], *mean_rows.stride()[:3],
        M=M, HP=triton.next_power_of_2(H), num_warps=4, **common)
