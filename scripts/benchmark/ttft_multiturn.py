#!/usr/bin/env python3
"""Multi-turn TTFT probe against a live serve.py/gateway endpoint.

Simulates a growing conversation (each turn appends the assistant's actual
reply plus a new user paragraph) and measures per-turn time-to-first-chunk.
With --prefix-cache on the replica this should be near-flat in context depth
(suffix-only prefill); without it, TTFT grows with the full prompt length.

Usage:
    python scripts/benchmark/ttft_multiturn.py --base http://localhost:2399 --turns 8
"""

from __future__ import annotations

import argparse
import json
import time

import httpx

PARA = (
    "Consider the way rivers carve their valleys over geological time, "
    "patiently negotiating with stone until the landscape itself remembers. "
)


def turn(base: str, messages: list[dict], max_tokens: int) -> tuple[float, float, str]:
    """One streamed chat turn. Returns (ttft_s, total_s, reply_text)."""
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.9,
        "stream": True,
    }
    ttft = float("nan")
    chunks: list[str] = []
    t0 = time.perf_counter()
    with httpx.stream("POST", f"{base}/v1/chat/completions", json=body, timeout=180) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                if chunks == []:
                    ttft = time.perf_counter() - t0
                chunks.append(delta)
    return ttft, time.perf_counter() - t0, "".join(chunks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:2399")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--gen-tokens", type=int, default=96)
    ap.add_argument("--user-paras", type=int, default=2,
                    help="paragraphs (~30 tok each) added per user turn")
    args = ap.parse_args()

    messages: list[dict] = []
    print(f"{'turn':>4} | {'~ctx tok':>8} | {'ttft':>8} | {'total':>8} | reply head")
    for t in range(args.turns):
        messages.append({"role": "user", "content": PARA * args.user_paras + f"(turn {t})"})
        approx_ctx = sum(len(m["content"]) // 4 for m in messages)
        ttft, total, reply = turn(args.base, messages, args.gen_tokens)
        messages.append({"role": "assistant", "content": reply})
        head = reply[:48].replace("\n", " ")
        print(f"{t:>4} | {approx_ctx:>8} | {ttft * 1000:>6.0f}ms | {total * 1000:>6.0f}ms | {head}")


if __name__ == "__main__":
    main()
