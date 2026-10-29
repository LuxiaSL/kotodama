"""Attention statistics of the cascade decoder's step without materialising its probabilities: one Triton program per
row reads the row's group prompt keys (up to the prompt's length, not the slot width) and its own tail keys twice,
once for each head's log-normaliser and once for the statistics kotodama.model.signing.stats_write takes from the
probabilities.

Per row and head: entropy (as lse - sum p s), the mass on prompt columns (< the row's position at admission), the
recency mass (columns >= max(1, floor(0.8 (pos + 1)))), the sink (column 0); per row: the entropy of the head-mean
distribution and, at sampled layers, the head-mean row over the first ``width`` absolute columns. Each row is
computed by its own program over its own keys, in a fixed order, so a row's statistics are the same whatever shares
its batch. The arithmetic differs from the reference (``multi_decode._cascade_probs`` + ``stats_write``) by rounding:
scores are bf16 dot products accumulated in fp32, and the sums run in tile order.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON = True
except ImportError:  # CPU-only installs: the reference path computes the statistics
    _TRITON = False


def available(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the fused kernel runs for tensors on ``device`` in ``dtype``."""
    return _TRITON and device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)


if _TRITON:

    @triton.jit
    def _cascade_stats_kernel(
        q_ptr, pk_ptr, tk_ptr, plen_ptr, rslot_ptr, tpos_ptr, s0_ptr, pos_ptr,
        ps_ptr, pm_ptr, rec_ptr, sink_ptr, ment_ptr, mrow_ptr,
        stride_qb, stride_qh, stride_pg, stride_pw, stride_ph, stride_tb, stride_tt, stride_th,
        scale,
        M_GROUP: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, R: tl.constexpr, HK: tl.constexpr,
        D: tl.constexpr, BN: tl.constexpr, WIDTH: tl.constexpr, SAMPLED: tl.constexpr,
    ):
        b = tl.program_id(0)
        g = tl.load(rslot_ptr + b) // M_GROUP
        plen = tl.load(plen_ptr + g).to(tl.int32)
        tpos = tl.load(tpos_ptr + b).to(tl.int32)
        s0 = tl.load(s0_ptr + b).to(tl.int32)
        pos = tl.load(pos_ptr + b).to(tl.int32)
        cut = tl.maximum(((pos + 1).to(tl.float32) * 0.8).to(tl.int32), 1)

        hs = tl.arange(0, HP)
        hmask = hs < H
        head_kv = hs // R
        ds = tl.arange(0, D)
        ns = tl.arange(0, BN)
        q = tl.load(q_ptr + b * stride_qb + hs[:, None] * stride_qh + ds[None, :], mask=hmask[:, None], other=0.0)

        # pass 1: each head's log-normaliser over the prompt columns, then the tail
        m_i = tl.zeros([HP], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([HP], dtype=tl.float32)
        n_tail = tpos + 1
        n_total = plen + n_tail
        for c0 in range(0, n_total, BN):
            col = c0 + ns  # absolute column for the prompt part, plen + t for the tail part
            in_prompt = col < plen
            valid = col < n_total
            t = col - plen
            s = tl.zeros([HP, BN], dtype=tl.float32) - float("inf")
            for hk in tl.static_range(HK):
                kp = tl.load(pk_ptr + g * stride_pg + col[:, None] * stride_pw + hk * stride_ph + ds[None, :],
                             mask=(in_prompt & valid)[:, None], other=0.0)
                kt = tl.load(tk_ptr + b * stride_tb + t[:, None] * stride_tt + hk * stride_th + ds[None, :],
                             mask=((~in_prompt) & valid)[:, None], other=0.0)
                k = tl.where(in_prompt[:, None], kp, kt)
                sk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
                s = tl.where((head_kv == hk)[:, None], sk, s)
            s = tl.where(valid[None, :] & hmask[:, None], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            l_i = l_i * tl.exp(m_i - m_safe) + tl.sum(tl.exp(s - m_safe[:, None]), axis=1)
            m_i = m_new
        lse = tl.where(hmask, m_i + tl.log(l_i), 0.0)

        # pass 2: the statistics from p = exp(s - lse)
        ps = tl.zeros([HP], dtype=tl.float32)
        pm = tl.zeros([HP], dtype=tl.float32)
        rec = tl.zeros([HP], dtype=tl.float32)
        sink = tl.zeros([HP], dtype=tl.float32)
        ment = tl.zeros([BN], dtype=tl.float32)
        for c0 in range(0, n_total, BN):
            col = c0 + ns
            in_prompt = col < plen
            valid = col < n_total
            t = col - plen
            s = tl.zeros([HP, BN], dtype=tl.float32)
            for hk in tl.static_range(HK):
                kp = tl.load(pk_ptr + g * stride_pg + col[:, None] * stride_pw + hk * stride_ph + ds[None, :],
                             mask=(in_prompt & valid)[:, None], other=0.0)
                kt = tl.load(tk_ptr + b * stride_tb + t[:, None] * stride_tt + hk * stride_th + ds[None, :],
                             mask=((~in_prompt) & valid)[:, None], other=0.0)
                k = tl.where(in_prompt[:, None], kp, kt)
                sk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
                s = tl.where((head_kv == hk)[:, None], sk, s)
            live = valid[None, :] & hmask[:, None]
            p = tl.where(live, tl.exp(s - lse[:, None]), 0.0)
            ps += tl.sum(tl.where(live, p * s, 0.0), axis=1)
            pm += tl.sum(tl.where((col < s0)[None, :], p, 0.0), axis=1)
            rec += tl.sum(tl.where((col >= cut)[None, :], p, 0.0), axis=1)
            sink += tl.sum(tl.where((col == 0)[None, :], p, 0.0), axis=1)
            mc = tl.sum(p, axis=0) / H
            ment += tl.where(mc > 0, -mc * tl.log(tl.where(mc > 0, mc, 1.0)), 0.0)
            if SAMPLED:
                tl.store(mrow_ptr + b * WIDTH + col, mc, mask=valid & (col < WIDTH))
        tl.store(ps_ptr + b * H + hs, lse - ps, mask=hmask)  # the head's entropy
        tl.store(pm_ptr + b * H + hs, pm, mask=hmask)
        tl.store(rec_ptr + b * H + hs, rec, mask=hmask)
        tl.store(sink_ptr + b * H + hs, sink, mask=hmask)
        tl.store(ment_ptr + b, tl.sum(ment, axis=0))


def cascade_stats(st: tuple, q: torch.Tensor, kv: tuple[torch.Tensor, ...], state: tuple[torch.Tensor, ...],
                  sampled: bool, block: int = 32) -> None:
    """One layer's attention statistics into the stats buffers, as ``stats_write(st, _cascade_probs(q, kv, state),
    sampled)`` writes them: q (B, heads, 1, head_dim) after the attention's KV write, kv = (tail_k, tail_v,
    prompt_k, prompt_v), state = (tail_pos, members, row_slot, prompt_lens, row_ids), st the stats tuple."""
    head_ent, mean_ent, head_sc, mean_rows, row_ids, slot, layer_col, sampled_col, start, pos = st
    tail_k, _, prompt_k, _ = kv
    tail_pos, members, row_slot, prompt_lens, _ = state
    B, H, _, D = q.shape
    hk = tail_k.shape[2]
    width = mean_rows.shape[-1]
    qr = q[:, :, 0].contiguous()
    f32 = dict(device=q.device, dtype=torch.float32)
    ent, pm, rec, sink = (torch.empty((B, H), **f32) for _ in range(4))
    ment = torch.empty((B,), **f32)
    mrow = torch.zeros((B, width), **f32) if sampled else ment
    _cascade_stats_kernel[(B,)](
        qr, prompt_k, tail_k, prompt_lens, row_slot, tail_pos, start, pos,
        ent, pm, rec, sink, ment, mrow,
        qr.stride(0), qr.stride(1), prompt_k.stride(0), prompt_k.stride(1), prompt_k.stride(2),
        tail_k.stride(0), tail_k.stride(1), tail_k.stride(2),
        D ** -0.5,
        M_GROUP=members.shape[1], H=H, HP=triton.next_power_of_2(H), R=H // hk, HK=hk, D=D, BN=block,
        WIDTH=width, SAMPLED=bool(sampled),
    )
    idx = (row_ids, slot, layer_col.expand(B))
    head_ent.index_put_(idx, ent)
    mean_ent.index_put_(idx, ment.unsqueeze(-1))
    if not sampled:
        return
    sidx = (row_ids, slot, sampled_col.expand(B))
    head_sc.index_put_(sidx, torch.cat([pm, rec, sink], dim=-1))
    mean_rows.index_put_(sidx, mrow)
