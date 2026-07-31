#!/usr/bin/env python3
"""TTFT vs context length against a live gateway/replica (streaming first-chunk)."""

from __future__ import annotations

import argparse
import time

import httpx

WORD = "the river flows through ancient valleys and "  # ~8 tokens per repeat


def ttft_once(base: str, approx_tokens: int) -> float:
    content = WORD * max(1, approx_tokens // 8)
    body = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8,
        "temperature": 0.9,
        "stream": True,
    }
    t0 = time.perf_counter()
    with httpx.stream("POST", f"{base}/v1/chat/completions", json=body, timeout=120) as r:
        for line in r.iter_lines():
            if line.startswith("data:") and "content" in line:
                return time.perf_counter() - t0
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:2222")
    ap.add_argument("--lens", type=int, nargs="+", default=[128, 512, 1024, 2048, 3000])
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    header = f"{'~prompt tok':>11} | " + " | ".join(f"{'rep' + str(i):>8}" for i in range(args.reps))
    print(header)
    for n in args.lens:
        vals = [ttft_once(args.base, n) for _ in range(args.reps)]
        row = f"{n:>11} | " + " | ".join(f"{v * 1000:>6.0f}ms" for v in vals)
        print(row)


if __name__ == "__main__":
    main()
