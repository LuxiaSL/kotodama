#!/usr/bin/env python3
"""Assemble a markdown comparison table from run_lm_eval results.json files.

Usage::

    python -m scripts.eval.assemble_lmeval_table \
        --results analysis/lm_eval/3b-base-step22400.pt/results.json:base-44B \
                  analysis/lm_eval/3b-language-FINAL-step195311.pt/results.json:final-384B \
        -o analysis/lm_eval/final_comparison.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (task, metric, higher_is_better, display)
PREFERRED_METRICS: list[tuple[str, str, bool, str]] = [
    ("hellaswag", "acc_norm", True, "HellaSwag (acc_norm)"),
    ("piqa", "acc", True, "PIQA (acc)"),
    ("arc_easy", "acc", True, "ARC-Easy (acc)"),
    ("arc_challenge", "acc_norm", True, "ARC-Challenge (acc_norm)"),
    ("boolq", "acc", True, "BoolQ (acc)"),
    ("copa", "acc", True, "COPA (acc)"),
    ("sciq", "acc", True, "SciQ (acc)"),
    ("winogrande", "acc", True, "Winogrande (acc)"),
    ("lambada_openai", "acc", True, "LAMBADA (acc)"),
    ("lambada_openai", "perplexity", False, "LAMBADA (ppl)"),
    ("wikitext", "word_perplexity", False, "WikiText (word_ppl)"),
]


def load_results(path: Path) -> dict[str, dict[str, float]]:
    d = json.loads(path.read_text())
    out: dict[str, dict[str, float]] = {}
    for task, res in d.items():
        out[task] = {
            k.replace(",none", ""): v
            for k, v in res.items()
            if isinstance(v, (int, float))
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True,
                   help="path:label pairs")
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args()

    cols: list[tuple[str, dict[str, dict[str, float]]]] = []
    for spec in args.results:
        path_s, _, label = spec.rpartition(":")
        path = Path(path_s)
        if not path.exists():
            print(f"WARN: missing {path}")
            continue
        cols.append((label, load_results(path)))

    labels = [label for label, _ in cols]
    lines = [
        "| Task | " + " | ".join(labels) + " |",
        "|---|" + "|".join(["---:"] * len(labels)) + "|",
    ]
    for task, metric, hib, display in PREFERRED_METRICS:
        vals: list[float | None] = []
        for _, res in cols:
            vals.append(res.get(task, {}).get(metric))
        if all(v is None for v in vals):
            continue
        finite = [v for v in vals if v is not None]
        best = (max if hib else min)(finite) if finite else None
        cells = []
        for v in vals:
            if v is None:
                cells.append("—")
                continue
            pct = metric.startswith("acc")
            s = f"{v * 100:.1f}" if pct else f"{v:.2f}"
            cells.append(f"**{s}**" if v == best and len(finite) > 1 else s)
        lines.append(f"| {display} | " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    print(table)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
