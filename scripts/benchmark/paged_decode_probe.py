#!/usr/bin/env python3
"""Feasibility probe for block-table (paged) KV in batched row decode (GPU).

``MultiSampleDecoder`` gives every row a full static KV slot (rows x kv_heads x max_seq_len x head_dim per layer) and
attends over the whole slot under a mask, so n candidates of one prompt hold n copies of the prompt's KV and every
row reads ``max_seq_len`` positions per step. A paged layout lets rows point at shared prompt blocks and attend only
over their own length. This probe measures, for one attention call at the tsumugi decode shape (12 query heads,
6 KV heads, head_dim 128, bf16), whether flash-attn's paged decode kernel (``flash_attn_with_kvcache`` with a
``block_table``; its page size is a multiple of 256):
  * runs on this GPU and agrees with the decoder's attention (masked SDPA over a dense per-row slot holding the same
    KV), as relative error of the outputs (one kernel family vs another: a wrong block table shows as O(1) error,
    kernel rounding as ~1e-3);
  * runs inside ``torch.compile(fullgraph=True, mode="reduce-overhead")`` when wrapped as a custom op (the decoder's
    step is a captured CUDA graph);
  * costs less or more than the dense masked SDPA per call, at several row counts, and how much KV memory each
    layout needs when ``n`` rows share one prompt.

    tools/run_py.sh scripts/benchmark/paged_decode_probe.py [--prompt 2600 --tail 200 --rows 8,32,64,128,256]
        [--out results/paged_decode_probe.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("paged_decode_probe")
H, KV, D = 12, 6, 128
PAGE = 256


def timed(fn, n_iter: int = 50) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=2600, help="shared prompt tokens")
    ap.add_argument("--tail", type=int, default=200, help="tokens each row decoded past the prompt")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--rows", default="8,32,64,128,256")
    ap.add_argument("--n", type=int, default=8, help="rows per prompt, for the memory estimate")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from flash_attn import flash_attn_with_kvcache

    dev, dt = torch.device("cuda"), torch.bfloat16
    S, P, T = a.max_seq_len, a.prompt, a.tail
    L = P + T
    pages_per_row = S // PAGE
    shared = P // PAGE  # whole prompt pages every row of the prompt shares; the rest is row-private
    res: dict = {"gpu": torch.cuda.get_device_name(), "prompt": P, "tail": T, "page": PAGE,
                 "shared_pages": shared}
    g = torch.Generator(device=dev).manual_seed(0)

    @torch.library.custom_op("kotodama_probe::paged_decode", mutates_args=())
    def paged_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seqlens: torch.Tensor,
                     table: torch.Tensor) -> torch.Tensor:
        return flash_attn_with_kvcache(q, k, v, cache_seqlens=seqlens, block_table=table)

    @paged_decode.register_fake
    def _(q, k, v, seqlens, table):
        return torch.empty_like(q)

    for B in [int(x) for x in a.rows.split(",") if x.strip()]:
        row = {}
        # the paged pool: the shared prompt pages once, then each row's private pages (prompt remainder + tail)
        n_private = pages_per_row - shared
        n_pages = shared + B * n_private
        kp = torch.randn((n_pages, PAGE, KV, D), device=dev, dtype=dt, generator=g)
        vp = torch.randn((n_pages, PAGE, KV, D), device=dev, dtype=dt, generator=g)
        # rows of one prompt hold the same prompt KV in their private remainder page
        rem = P - shared * PAGE
        for r in range(1, B):
            base = shared + r * n_private
            kp[base, :rem] = kp[shared, :rem]
            vp[base, :rem] = vp[shared, :rem]
        table = torch.empty((B, pages_per_row), device=dev, dtype=torch.int32)
        table[:, :shared] = torch.arange(shared, device=dev, dtype=torch.int32)
        for r in range(B):
            table[r, shared:] = torch.arange(shared + r * n_private, shared + (r + 1) * n_private, device=dev,
                                             dtype=torch.int32)
        seqlens = torch.full((B,), L, device=dev, dtype=torch.int32)
        q = torch.randn((B, 1, H, D), device=dev, dtype=dt, generator=g)
        try:
            out_p = flash_attn_with_kvcache(q, kp, vp, cache_seqlens=seqlens, block_table=table)
        except Exception as exc:  # the kernel may not be built for this GPU
            res[f"rows{B}"] = {"paged_error": repr(exc)}
            logger.warning("rows %d: paged kernel failed: %r", B, exc)
            continue
        # the decoder's layout: a dense (B, KV, S, D) slot per row, gathered from the same pages, masked SDPA
        kd = kp[table.long()].reshape(B, S, KV, D).transpose(1, 2).contiguous()
        vd = vp[table.long()].reshape(B, S, KV, D).transpose(1, 2).contiguous()
        bias = torch.full((B, 1, 1, S), float("-inf"), device=dev, dtype=dt)
        bias[..., :L] = 0
        qd = q.transpose(1, 2)  # (B, H, 1, D)
        out_d = F.scaled_dot_product_attention(qd, kd, vd, attn_mask=bias, enable_gqa=True).transpose(1, 2)
        rel = float((out_p.float() - out_d.float()).norm() / out_d.float().norm())
        row["rel_err_vs_dense_sdpa"] = rel
        row["dense_sdpa_ms"] = timed(lambda: F.scaled_dot_product_attention(qd, kd, vd, attn_mask=bias,
                                                                            enable_gqa=True))
        row["paged_ms"] = timed(lambda: flash_attn_with_kvcache(q, kp, vp, cache_seqlens=seqlens,
                                                                block_table=table))
        # under compile + CUDA graphs, as the decoder's step runs
        try:
            fn = torch.compile(lambda q_, s_: paged_decode(q_, kp, vp, s_, table), fullgraph=True,
                               mode="reduce-overhead")
            for _ in range(3):
                got = fn(q, seqlens)
            row["compiled_rel_err_vs_eager"] = float((got.float() - out_p.float()).norm() / out_p.float().norm())
            row["compiled_paged_ms"] = timed(lambda: fn(q, seqlens))
        except Exception as exc:
            row["compile_error"] = repr(exc)[:400]
        bytes_tok = 2 * KV * D * 2  # K and V, bf16, one layer
        row["dense_kv_mb_per_layer"] = B * S * bytes_tok / 1e6
        prompts = max(B // a.n, 1)
        row["paged_kv_mb_per_layer_n_rows_per_prompt"] = (prompts * shared + B * n_private) * PAGE * bytes_tok / 1e6
        row["paged_kv_mb_per_layer_tight"] = (prompts * shared + B * -(-(L - shared * PAGE) // PAGE)) \
            * PAGE * bytes_tok / 1e6
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
