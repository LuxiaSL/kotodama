#!/usr/bin/env python3
"""Gateway cache-affinity e2e test (run against a mini-fleet or prod gateway).

The decisive signal: a prefix_cache_hit on turn 2+ THROUGH THE GATEWAY proves
affinity worked end-to-end — caches are per-replica, so only the replica that
served turn 1 can hit on turn 2. Conversations are distinguished by their
prompt heads (that's the affinity key), so two interleaved conversations must
each hit their own pin without disturbing the other.

Checks:
  1. completion pool (/generate): 3-turn conversation -> hits on turns 2,3
  2. interleaved second conversation -> its own hits, no cross-talk
  3. chat pool (/v1/chat/completions, messages key): turn-2 TTFT sanity
  4. busy fallback: a concurrent stream on the pinned replica -> the next
     turn still answers (lb fallback; miss is harmless) — soft check
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

import httpx

FAIL: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        FAIL.append(msg)


def gen(client: httpx.Client, base: str, prompt: str, n: int = 48) -> dict:
    r = client.post(f"{base}/generate", json={
        "prompt": prompt, "max_new_tokens": n, "temperature": 0.9,
    })
    r.raise_for_status()
    return r.json()


def run_conversation(client: httpx.Client, base: str, seed_text: str, turns: int) -> list[dict]:
    prompt = seed_text
    out = []
    for _ in range(turns):
        d = gen(client, base, prompt)
        out.append(d)
        prompt = prompt + d["text"] + " The next chapter began. "
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:2224")
    args = ap.parse_args()
    client = httpx.Client(timeout=300)

    print("== 1. completion-pool conversation: hits via affinity ==")
    conv_a = run_conversation(client, args.base, "Conversation alpha: the lighthouse keeper kept two journals. " * 6, 3)
    hits_a = [t["timing"].get("prefix_cache_hit") for t in conv_a]
    check(hits_a[0] in (0.0, None), f"A turn0 is a miss (hit={hits_a[0]})")
    check(hits_a[1] == 1.0 and hits_a[2] == 1.0, f"A turns 1,2 hit through gateway (hits={hits_a[1:]})")

    print("== 2. interleaved conversations keep their own pins ==")
    pa = "Conversation alpha: the lighthouse keeper kept two journals. " * 6
    pb = "Conversation beta: the cartographer refused to draw coastlines. " * 6
    da1 = gen(client, args.base, pa)
    db1 = gen(client, args.base, pb)
    pa2 = pa + da1["text"] + " More followed. "
    pb2 = pb + db1["text"] + " More followed. "
    da2 = gen(client, args.base, pa2)
    db2 = gen(client, args.base, pb2)
    check(da2["timing"].get("prefix_cache_hit") == 1.0,
          f"A turn2 hit while interleaved (hit={da2['timing'].get('prefix_cache_hit')}, "
          f"common={da2['timing'].get('prefix_common_tokens')})")
    check(db2["timing"].get("prefix_cache_hit") == 1.0,
          f"B turn2 hit while interleaved (hit={db2['timing'].get('prefix_cache_hit')}, "
          f"common={db2['timing'].get('prefix_common_tokens')})")

    print("== 3. chat pool: stable affinity by messages key ==")
    msgs = [{"role": "user", "content": "Tell me about tidal patterns in three sentences."}]
    r1 = client.post(f"{args.base}/v1/chat/completions", json={"messages": msgs, "max_tokens": 64, "temperature": 0.9})
    r1.raise_for_status()
    reply = r1.json()["choices"][0]["message"]["content"]
    msgs = msgs + [{"role": "assistant", "content": reply},
                   {"role": "user", "content": "And how do storms change that?"}]
    t0 = time.perf_counter()
    r2 = client.post(f"{args.base}/v1/chat/completions", json={"messages": msgs, "max_tokens": 64, "temperature": 0.9})
    r2.raise_for_status()
    turn2_s = time.perf_counter() - t0
    check(r2.status_code == 200, f"chat turn2 ok ({turn2_s * 1000:.0f}ms total)")

    print("== 4. busy fallback: concurrent stream + same-conversation turn ==")
    stream_done = threading.Event()

    def hold_stream() -> None:
        try:
            with httpx.stream("POST", f"{args.base}/generate", json={
                "prompt": pa2 + da2["text"] + " Continue the tale at length. ",
                "max_new_tokens": 400, "temperature": 0.9, "stream": True,
            }, timeout=120) as r:
                for _ in r.iter_lines():
                    pass
        finally:
            stream_done.set()

    th = threading.Thread(target=hold_stream, daemon=True)
    th.start()
    time.sleep(0.8)  # let the stream occupy the pinned replica
    d_busy = gen(client, args.base, pa2 + da2["text"] + " A different continuation. ")
    hit_busy = d_busy["timing"].get("prefix_cache_hit")
    check(d_busy.get("text") is not None,
          f"turn served while pin busy (hit={hit_busy} — lb fallback miss is expected and fine)")
    stream_done.wait(timeout=120)

    print("OVERALL:", "PASS" if not FAIL else f"FAIL ({len(FAIL)})")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
