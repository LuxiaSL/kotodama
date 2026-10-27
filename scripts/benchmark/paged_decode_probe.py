#!/usr/bin/env python3
"""Feasibility probe for block-table (paged) KV and shared-prefix (cascade) attention in batched row decode (GPU).

``MultiSampleDecoder`` gives every row a full static KV slot (rows x kv_heads x max_seq_len x head_dim per layer) and
attends over the whole slot under a mask, so n candidates of one prompt hold n copies of the prompt's KV and every
row reads ``max_seq_len`` positions per step. This probe measures one attention call at the tsumugi decode shape
(12 query heads, 6 KV heads, head_dim 128, bf16) with ``rows`` rows in groups of ``n`` that share a prompt, each
group its own prompt, three ways:
  * ``dense``: the decoder's attention (masked SDPA over a dense per-row slot);
  * ``paged``: flash-attn's paged decode kernel (``flash_attn_with_kvcache`` with a ``block_table``; its page size is
    a multiple of 256): rows point at their group's shared prompt pages and own private pages, and attend over
    their real length only;
  * ``cascade``: the shared prompt pages attended once per group (the group's n query rows as n query positions of
    one sequence) plus each row's private pages, merged by log-sum-exp.
For each: relative error of the outputs against ``dense`` (another kernel family: a wrong table shows as O(1)
error, kernel rounding as ~1e-3), GPU time per call replayed from a captured CUDA graph (the decoder's step is a
captured graph, so launch overhead is not part of the cost), and the KV memory each layout needs. A custom-op
wrapper of the paged kernel is also run under ``torch.compile(fullgraph=True, mode="reduce-overhead")``, the way the
decoder's step is compiled.

    tools/run_py.sh scripts/benchmark/paged_decode_probe.py [--prompt 2600 --tail 200 --rows 8,32,64,128,256]
        [--out results/paged_decode_probe.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("paged_decode_probe")
H, KV, D = 12, 6, 128
PAGE = 256


def graph_ms(fn, n_iter: int = 200) -> float:
    """GPU milliseconds per call of ``fn`` replayed from a captured CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=2600, help="prompt tokens each group of n rows shares")
    ap.add_argument("--tail", type=int, default=200, help="tokens each row decoded past the prompt")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--rows", default="8,32,64,128,256")
    ap.add_argument("--n", type=int, default=8, help="rows per prompt (one request's candidates)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from flash_attn import flash_attn_with_kvcache

    dev, dt = torch.device("cuda"), torch.bfloat16
    S, P, T, n = a.max_seq_len, a.prompt, a.tail, a.n
    L = P + T
    pages_per_row = S // PAGE
    shared = P // PAGE  # whole prompt pages a group shares; the prompt's remainder and the tail are row-private
    n_private = pages_per_row - shared
    bytes_tok = 2 * KV * D * 2  # K and V, bf16, one layer
    res: dict = {"gpu": torch.cuda.get_device_name(), "prompt": P, "tail": T, "page": PAGE, "n": n,
                 "shared_pages": shared}
    gen = torch.Generator(device=dev).manual_seed(0)

    @torch.library.custom_op("kotodama_probe::paged_decode", mutates_args=())
    def paged_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlens: torch.Tensor,
                     table: torch.Tensor) -> torch.Tensor:
        return flash_attn_with_kvcache(q, k, v, cache_seqlens=seqlens, block_table=table)

    @paged_decode.register_fake
    def _(q, k, v, seqlens, table):
        return torch.empty_like(q)

    for B in [int(x) for x in a.rows.split(",") if x.strip()]:
        if B % n:
            continue
        G = B // n
        row: dict = {}
        # pool: G groups' shared prompt pages, then every row's private pages
        n_pages = G * shared + B * n_private
        kp = torch.randn((n_pages, PAGE, KV, D), device=dev, dtype=dt, generator=gen)
        vp = torch.randn((n_pages, PAGE, KV, D), device=dev, dtype=dt, generator=gen)
        table = torch.empty((B, pages_per_row), device=dev, dtype=torch.int32)
        rem = P - shared * PAGE
        for r in range(B):
            g = r // n
            table[r, :shared] = torch.arange(g * shared, (g + 1) * shared, device=dev, dtype=torch.int32)
            first = G * shared + r * n_private
            table[r, shared:] = torch.arange(first, first + n_private, device=dev, dtype=torch.int32)
            lead = G * shared + g * n * n_private  # the group's first row holds the prompt remainder
            if first != lead:  # every row of a group holds the same prompt remainder
                kp[first, :rem] = kp[lead, :rem]
                vp[first, :rem] = vp[lead, :rem]
        seqlens = torch.full((B,), L, device=dev, dtype=torch.int32)
        q = torch.randn((B, 1, H, D), device=dev, dtype=dt, generator=gen)

        # dense: the decoder's layout, gathered from the same pages
        kd = kp[table.long()].reshape(B, S, KV, D).transpose(1, 2).contiguous()
        vd = vp[table.long()].reshape(B, S, KV, D).transpose(1, 2).contiguous()
        bias = torch.full((B, 1, 1, S), float("-inf"), device=dev, dtype=dt)
        bias[..., :L] = 0
        qd = q.transpose(1, 2).contiguous()  # (B, H, 1, D)

        def dense() -> torch.Tensor:
            return F.scaled_dot_product_attention(qd, kd, vd, attn_mask=bias, enable_gqa=True).transpose(1, 2)

        out_d = dense()
        row["dense_ms"] = graph_ms(dense)

        def paged() -> torch.Tensor:
            return flash_attn_with_kvcache(q, kp, vp, cache_seqlens=seqlens, block_table=table)

        try:
            out_p = paged()
        except Exception as exc:  # the kernel may not be built for this GPU
            res[f"rows{B}"] = {"paged_error": repr(exc)}
            logger.warning("rows %d: paged kernel failed: %r", B, exc)
            continue
        row["paged_rel_err"] = rel(out_p, out_d)
        row["paged_ms"] = graph_ms(paged)

        # cascade: the group's rows as n query positions over the shared pages, each row over its private pages
        q_grp = q.view(G, n, H, D)
        trunk_table = table[::n, :shared].contiguous()
        trunk_len = torch.full((G,), shared * PAGE, device=dev, dtype=torch.int32)
        priv_table = table[:, shared:].contiguous()
        priv_len = torch.full((B,), L - shared * PAGE, device=dev, dtype=torch.int32)

        def cascade() -> torch.Tensor:
            o1, lse1 = flash_attn_with_kvcache(q_grp, kp, vp, cache_seqlens=trunk_len, block_table=trunk_table,
                                               return_softmax_lse=True)
            o2, lse2 = flash_attn_with_kvcache(q, kp, vp, cache_seqlens=priv_len, block_table=priv_table,
                                               return_softmax_lse=True)
            l1 = lse1.permute(0, 2, 1).reshape(B, H, 1)  # (G, H, n) -> per row
            l2 = lse2.view(B, H, 1)
            m = torch.maximum(l1, l2)
            w1, w2 = (l1 - m).exp(), (l2 - m).exp()
            o = (o1.reshape(B, H, D).float() * w1 + o2.reshape(B, H, D).float() * w2) / (w1 + w2)
            return o.to(dt).view(B, 1, H, D)

        try:
            row["cascade_rel_err"] = rel(cascade(), out_d)
            row["cascade_ms"] = graph_ms(cascade)
        except Exception as exc:
            row["cascade_error"] = repr(exc)[:400]

        try:
            fn = torch.compile(lambda q_, s_: paged_decode(q_, kp, vp, s_, table), fullgraph=True,
                               mode="reduce-overhead")
            for _ in range(3):
                got = fn(q, seqlens)
            row["compiled_paged_rel_err_vs_eager"] = rel(got, out_p)
        except Exception as exc:
            row["compile_error"] = repr(exc)[:400]

        row["dense_kv_mb_per_layer"] = B * S * bytes_tok / 1e6
        row["paged_kv_mb_per_layer"] = (G * shared + B * -(-(L - shared * PAGE) // PAGE)) * PAGE * bytes_tok / 1e6
        res[f"rows{B}"] = row
        logger.info("rows %d: %s", B, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        del kp, vp, kd, vd
        torch.cuda.empty_cache()
    logger.info("RESULT %s", json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
