#!/usr/bin/env python3
"""Aoyama-style repeated-bigram statistics over uint16 token streams.

The induction-formation law (Aoyama/Wilcox/Schneider, ICML 2026) says
induction heads form when contexts contain FREQUENT, RELIABLE repeated
bigrams. Two in-context statistics over fixed windows:

  P(AB…A)      "opportunity": for a position j (token A = w[j], with a next
               token), the probability that A already occurred earlier in the
               window with a successor strictly before j.
  P(B|AB…A)    "reliability": given an opportunity, the probability that the
               successor repeats — w[j+1] equals the successor of the MOST
               RECENT earlier occurrence of A (what an induction head copies).
               ``any`` variant: any earlier successor of A matches w[j+1].

Convention: an earlier occurrence i counts only when i+1 < j (the earlier
successor B lies strictly before the later A), so degenerate "AA" overlaps
don't count as repeats.

Usage:
  python scripts/utils/bigram_stats.py --window 4096 --max-windows 400 \
      name1=path1.bin name2=path2.bin ...
  python scripts/utils/bigram_stats.py --selftest

SPEC-1B-FORMATION-LADDER §2 P0.2. CPU-only; streams are flat uint16 bins.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def window_stats(w: np.ndarray) -> dict[str, float]:
    """The two statistics for ONE window (1-D int array)."""
    last_occ: dict[int, int] = {}       # token -> most recent position
    last_succ: dict[int, int] = {}      # token -> successor at that position
    succ_seen: dict[int, set[int]] = {} # token -> all successors seen
    opportunities = 0
    reliable_recent = 0
    reliable_any = 0
    n_positions = 0
    L = len(w)
    for j in range(L - 1):              # w[j+1] must exist
        a = int(w[j])
        n_positions += 1
        i = last_occ.get(a)
        if i is not None and i + 1 < j:
            opportunities += 1
            nxt = int(w[j + 1])
            if last_succ[a] == nxt:
                reliable_recent += 1
            if nxt in succ_seen[a]:
                reliable_any += 1
        last_occ[a] = j
        last_succ[a] = int(w[j + 1])
        succ_seen.setdefault(a, set()).add(int(w[j + 1]))
    return {
        "p_opportunity": opportunities / max(n_positions, 1),
        "p_reliable_recent": reliable_recent / max(opportunities, 1),
        "p_reliable_any": reliable_any / max(opportunities, 1),
        "n_opportunities": opportunities,
        "n_positions": n_positions,
    }


def stream_stats(path: Path, window: int, max_windows: int,
                 seed: int) -> dict[str, float]:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    n_win = max(len(data) // window, 1)
    rng = np.random.default_rng(seed)
    idx = (rng.choice(n_win, size=min(max_windows, n_win), replace=False)
           if n_win > 1 else np.array([0]))
    rows = [window_stats(np.asarray(data[k * window: (k + 1) * window]))
            for k in sorted(idx.tolist())]
    def m(key):
        return float(np.mean([r[key] for r in rows]))
    def se(key):
        vals = [r[key] for r in rows]
        return float(np.std(vals) / max(len(vals), 1) ** 0.5)
    return {
        "path": str(path),
        "tokens": int(len(data)),
        "window": window,
        "n_windows": len(rows),
        "unique_tokens": int(len(np.unique(
            data[: min(len(data), 5_000_000)]))),
        "p_opportunity": m("p_opportunity"),
        "p_opportunity_se": se("p_opportunity"),
        "p_reliable_recent": m("p_reliable_recent"),
        "p_reliable_recent_se": se("p_reliable_recent"),
        "p_reliable_any": m("p_reliable_any"),
        "mean_opportunities_per_window": m("n_opportunities"),
    }


def selftest() -> None:
    # perfectly repetitive: every later position is an opportunity and every
    # successor repeats
    w = np.array([1, 2, 3] * 40)
    s = window_stats(w)
    assert s["p_reliable_recent"] == 1.0 and s["p_reliable_any"] == 1.0
    assert s["p_opportunity"] > 0.9, s
    # all-unique tokens: zero opportunities
    s = window_stats(np.arange(100))
    assert s["p_opportunity"] == 0.0
    # AB...AC: opportunity at the second A, but unreliable
    s = window_stats(np.array([1, 2, 9, 9, 8, 1, 3]))
    # positions: j=5 (token 1, prior occ at 0, succ 2, next is 3 -> miss);
    # j=3 (token 9, prior occ at 2 but i+1==j -> NOT an opportunity)
    assert s["n_opportunities"] == 1
    assert s["p_reliable_recent"] == 0.0 and s["p_reliable_any"] == 0.0
    # AA overlap excluded, later true repeat counts
    s = window_stats(np.array([5, 5, 5, 7, 5, 5]))
    # j=1: prior i=0, i+1==j excluded; j=2: prior i=1, i+1==j excluded... wait
    # j=2: i=1, i+1=2==j excluded; j=4: i=2, i+1=3 < 4 -> opportunity,
    # last_succ[5]=w[3]=7, next w[5]=5 -> miss; j hits only once
    assert s["n_opportunities"] == 1 and s["p_reliable_recent"] == 0.0
    # reliability via most-recent vs any: A B A C A B -> at second A (j=2):
    # opp? i=0,i+1=1<2 yes, succ B vs next C: miss. at third A (j=4): i=2,
    # i+1=3<4, most-recent succ = C, next = B -> recent miss, any hit (B seen)
    s = window_stats(np.array([1, 2, 1, 3, 1, 2]))
    assert s["n_opportunities"] == 2
    assert s["p_reliable_recent"] == 0.0 and s["p_reliable_any"] == 0.5
    print("bigram_stats selftest: OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("streams", nargs="*", help="name=path.bin")
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--max-windows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write JSON here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.streams:
        ap.error("give at least one name=path stream (or --selftest)")
    results = {}
    for spec in args.streams:
        name, _, path = spec.partition("=")
        r = stream_stats(Path(path), args.window, args.max_windows, args.seed)
        results[name] = r
        print(f"{name:>16}: P(AB…A) {r['p_opportunity']:.4f} "
              f"± {r['p_opportunity_se']:.4f} | "
              f"P(B|AB…A) recent {r['p_reliable_recent']:.4f} "
              f"± {r['p_reliable_recent_se']:.4f} | any "
              f"{r['p_reliable_any']:.4f} | uniq {r['unique_tokens']} | "
              f"windows {r['n_windows']}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
