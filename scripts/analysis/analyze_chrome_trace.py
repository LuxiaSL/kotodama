"""Aggregate a torch.profiler chrome trace: CUDA-kernel time by name + launch counts.

Replaces the in-run key_averages() summary when trace export succeeded but the
summary phase died (e.g. NCCL watchdog during rank-0 export — 2026-07-04).

Usage: python scripts/analysis/analyze_chrome_trace.py TRACE.json [--top 40]
Memory note: json.load of a 7GB trace needs ~50-70GB RAM — fine on gpu-host.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    print(f"loading {args.trace} ...", file=sys.stderr, flush=True)
    with open(args.trace) as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    print(f"{len(events)} events", file=sys.stderr, flush=True)

    kern_dur: dict[str, float] = defaultdict(float)
    kern_cnt: dict[str, int] = defaultdict(int)
    cat_dur: dict[str, float] = defaultdict(float)
    cat_cnt: dict[str, int] = defaultdict(int)

    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        dur = e.get("dur", 0.0)
        cat_dur[cat] += dur
        cat_cnt[cat] += 1
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kern_dur[e.get("name", "?")] += dur
            kern_cnt[e.get("name", "?")] += 1

    total_gpu = sum(v for k, v in cat_dur.items() if k in ("kernel", "gpu_memcpy", "gpu_memset"))
    total_launches = sum(v for k, v in cat_cnt.items() if k in ("kernel", "gpu_memcpy", "gpu_memset"))

    print("\n== category totals (us) ==")
    for c in sorted(cat_dur, key=cat_dur.get, reverse=True):
        print(f"{cat_dur[c]:>16,.0f}  {cat_cnt[c]:>10,}  {c}")

    print(f"\n== GPU total: {total_gpu/1e6:,.2f} s across {total_launches:,} launches ==")
    print(f"\n== top {args.top} GPU ops by total time ==")
    print(f"{'time(s)':>10} {'%':>6} {'count':>9} {'avg(us)':>9}  name")
    for name in sorted(kern_dur, key=kern_dur.get, reverse=True)[: args.top]:
        d, c = kern_dur[name], kern_cnt[name]
        print(f"{d/1e6:>10.2f} {100*d/max(total_gpu,1):>5.1f}% {c:>9,} {d/max(c,1):>9.1f}  {name[:120]}")


if __name__ == "__main__":
    main()
