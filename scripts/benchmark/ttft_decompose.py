#!/usr/bin/env python3
"""Decompose per-turn prefill cost via /generate's timing dict.

Grows a prompt by appending each turn's completion plus new text, so each
request shares a long prefix with the previous one — then prints the
server-side timing fields (tokenize / prefill / prefix-cache hit info).
"""

from __future__ import annotations

import argparse

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:2399")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--gen-tokens", type=int, default=32)
    args = ap.parse_args()

    prompt = "The history of rivers is the history of patience. " * 40
    client = httpx.Client(timeout=120)
    for turn in range(args.turns):
        r = client.post(f"{args.base}/generate", json={
            "prompt": prompt, "max_new_tokens": args.gen_tokens, "temperature": 0.9,
        })
        r.raise_for_status()
        d = r.json()
        t = d["timing"]
        print(
            f"turn {turn}: prompt_tok={d['prompt_tokens']:>5} "
            f"tokenize={t.get('tokenize_ms', 0):>6.1f}ms "
            f"prefill={t.get('prefill_ms', 0):>7.1f}ms "
            f"hit={t.get('prefix_cache_hit')} "
            f"common={t.get('prefix_common_tokens')} "
            f"suffix={t.get('prefill_suffix_tokens')}"
        )
        prompt = prompt + d["text"] + " And the mountains answered with silence. " * 10


if __name__ == "__main__":
    main()
