#!/usr/bin/env python3
"""Plot the serving-optimization ledger: iteration # vs single-stream tok/s.

Usage:
    python plot_ledger.py [--ledger opt_ledger.jsonl] [--out opt_progress.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default=str(Path(__file__).parent / "opt_ledger.jsonl"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "opt_progress.png"))
    args = parser.parse_args()

    entries = [json.loads(line) for line in Path(args.ledger).read_text().splitlines() if line.strip()]
    # Entries with tok_s: null are non-throughput milestones (e.g. TTFT work) —
    # they stay in the ledger but not on the throughput plot.
    skipped = [e for e in entries if e.get("tok_s") is None]
    entries = [e for e in entries if e.get("tok_s") is not None]
    entries.sort(key=lambda e: e["iter"])

    iters = [e["iter"] for e in entries]
    tok_s = [e["tok_s"] for e in entries]
    baseline = tok_s[0]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(iters, tok_s, marker="o", linewidth=2, markersize=8, color="#2563eb", zorder=3)
    ax.axhline(baseline, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.annotate(f"baseline {baseline:.0f} tok/s", xy=(iters[-1], baseline),
                xytext=(-4, 6), textcoords="offset points", ha="right",
                color="#dc2626", fontsize=9)

    for e in entries:
        speedup = e["tok_s"] / baseline
        short = e["label"].split(":")[0]
        ax.annotate(
            f"{e['tok_s']:.0f} ({speedup:.1f}x)\n{short}",
            xy=(e["iter"], e["tok_s"]),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=8,
        )

    ax.set_xlabel("optimization iteration")
    ax.set_ylabel("single-stream decode tok/s (ctx 512, with sampler)")
    ax.set_title("kotodama 3B serving: single-stream throughput by iteration (B200, bf16)")
    ax.set_xticks(iters)
    ax.grid(alpha=0.25, zorder=0)
    ax.set_ylim(0, max(tok_s) * 1.18)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    for e in entries:
        print(f"  iter {e['iter']}: {e['tok_s']:>7.1f} tok/s ({e['tok_s']/baseline:.2f}x)  {e['label']}")
    for e in skipped:
        print(f"  iter {e['iter']}: (not plotted — no tok/s)  {e['label']}")


if __name__ == "__main__":
    main()
