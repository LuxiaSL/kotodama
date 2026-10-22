#!/usr/bin/env python3
"""Spike resilience read from heimdall telemetry: does rank recover after loss spikes?

The read: in a resilient run, a loss spike is followed by geo/rankme_last climbing
back to its pre-spike level; a brittle run stops recovering (the shipped 3B did from
~139K steps on). For each spike in train/loss (or train/grad_norm) — a point more than --k robust
sigmas above the rolling median of the previous --window points — report the last
RankMe probe before it, the lowest probe within --horizon steps after it, and the step
at which RankMe first returns within --tol of the pre-spike value (None = never).

Usage: python scripts/eval/resilience.py --job <id> [--signal train/loss] [--k 6] \
          [--window 50] [--horizon 2000] [--tol 0.01] [--out spikes.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hm_portrait as hp  # noqa: E402

Series = dict[int, float]


@dataclass(frozen=True)
class Spike:
    step: int
    value: float
    baseline: float
    sigmas: float
    rank_before: float | None
    rank_min_after: float | None
    recovered_at: int | None  # first probe step with rank >= (1 - tol) * rank_before


def find_spikes(sig: Series, k: float, window: int) -> list[tuple[int, float, float, float]]:
    steps = sorted(sig)
    out = []
    for i in range(window, len(steps)):
        prev = [sig[s] for s in steps[i - window:i]]
        med = statistics.median(prev)
        mad = statistics.median(abs(x - med) for x in prev) or 1e-12
        z = (sig[steps[i]] - med) / (1.4826 * mad)
        if z > k:
            out.append((steps[i], sig[steps[i]], med, z))
    merged: list[tuple[int, float, float, float]] = []  # one event per burst
    for sp in out:
        if merged and sp[0] - merged[-1][0] <= window:
            if sp[3] > merged[-1][3]:
                merged[-1] = sp
        else:
            merged.append(sp)
    return merged


def recovery(rank: Series, step: int, horizon: int, tol: float) -> tuple[float | None, float | None, int | None]:
    before = [s for s in sorted(rank) if s <= step]
    if not before:
        return None, None, None
    r0 = rank[before[-1]]
    after = [s for s in sorted(rank) if step < s <= step + horizon]
    if not after:
        return r0, None, None
    rmin = min(rank[s] for s in after)
    rec = next((s for s in after if rank[s] >= (1 - tol) * r0), None)
    return r0, rmin, rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--signal", default="train/loss")
    ap.add_argument("--rank-key", default="geo/rankme_last")
    ap.add_argument("--k", type=float, default=6.0)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--out")
    ap.add_argument("--url", default=os.environ.get(
        "KOTODAMA_SCHEDULER_URL", os.environ.get("HEIMDALL_URL", "http://127.0.0.1:7000")))
    a = ap.parse_args()
    d = hp.fetch(a.url.rstrip("/"), a.job, [a.signal, a.rank_key])
    sig, rank = d[a.signal], d[a.rank_key]
    if not sig or not rank:
        print(f"FAIL: missing series ({a.signal}: {len(sig)} pts, {a.rank_key}: {len(rank)} pts)")
        return 1
    spikes = []
    for step, val, med, z in find_spikes(sig, a.k, a.window):
        r0, rmin, rec = recovery(rank, step, a.horizon, a.tol)
        spikes.append(Spike(step, val, med, z, r0, rmin, rec))
    print(f"{a.job}: {len(spikes)} spikes in {a.signal} (k={a.k}, window={a.window}); "
          f"rank = {a.rank_key}, horizon {a.horizon}, tol {a.tol:.0%}")
    for s in spikes:
        rec = "NEVER" if s.recovered_at is None else f"step {s.recovered_at} (+{s.recovered_at - s.step})"
        print(f"  step {s.step:6d}: {a.signal}={s.value:.4f} vs {s.baseline:.4f} ({s.sigmas:.1f}σ)  "
              f"rank {s.rank_before} -> min {s.rank_min_after}  recovered: {rec}")
    if a.out:
        Path(a.out).write_text(json.dumps([asdict(s) for s in spikes], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
