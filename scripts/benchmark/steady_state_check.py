#!/usr/bin/env python3
"""Steady-state decode rate: prefix-cache HIT vs MISS on the same replica.

Decode after an extend runs the identical compiled step over the same static
buffers, so rates should match exactly (modulo thermal noise) — this verifies
that empirically via /generate's decode_per_token_ms.
"""

from __future__ import annotations

import argparse

import httpx


def gen(client: httpx.Client, base: str, prompt: str, n: int) -> dict:
    r = client.post(f"{base}/generate", json={
        "prompt": prompt, "max_new_tokens": n, "temperature": 0.9,
    })
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:2399")
    ap.add_argument("--gen-tokens", type=int, default=300)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    client = httpx.Client(timeout=300)
    rows = []
    for rep in range(args.reps):
        # MISS: fresh prompt every rep (no shared prefix with anything cached)
        miss_prompt = f"Chronicle number {rep * 7919}: " + "the tide returns to ask its question. " * 30
        d_miss = gen(client, args.base, miss_prompt, args.gen_tokens)
        # HIT: extend the conversation the miss just cached
        hit_prompt = miss_prompt + d_miss["text"] + " And what came after was stranger still. "
        d_hit = gen(client, args.base, hit_prompt, args.gen_tokens)
        tm, th = d_miss["timing"], d_hit["timing"]
        rows.append((rep, tm, th, d_miss["prompt_tokens"], d_hit["prompt_tokens"]))
        print(
            f"rep {rep}: MISS plen={d_miss['prompt_tokens']:>4} hit={tm.get('prefix_cache_hit')} "
            f"prefill={tm['prefill_ms']:>6.1f}ms decode={tm.get('decode_per_token_ms', float('nan')):>5.2f}ms/tok | "
            f"HIT plen={d_hit['prompt_tokens']:>4} hit={th.get('prefix_cache_hit')} "
            f"common={th.get('prefix_common_tokens')} prefill={th['prefill_ms']:>6.1f}ms "
            f"decode={th.get('decode_per_token_ms', float('nan')):>5.2f}ms/tok"
        )
    miss_rates = [r[1].get("decode_per_token_ms") for r in rows]
    hit_rates = [r[2].get("decode_per_token_ms") for r in rows]
    print(f"\ndecode ms/tok — miss: {miss_rates}  hit: {hit_rates}")
    print(f"medians: miss={sorted(miss_rates)[len(miss_rates)//2]:.2f}  "
          f"hit={sorted(hit_rates)[len(hit_rates)//2]:.2f}")


if __name__ == "__main__":
    main()
