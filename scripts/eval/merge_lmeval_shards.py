"""Merge per-shard lm-eval results.json files into one results.json.

Task-sharded runs (tools/launch_lmeval_sharded.sh) write
analysis/lm_eval_shards/<ckpt>/shard<N>/<ckpt>/results.json each covering a
disjoint task subset. This unions them into the standard
analysis/lm_eval/<ckpt>/results.json layout so assemble_lmeval_table.py and
check_lmeval_equivalence.py work unchanged.

Usage:
    python -m scripts.eval.merge_lmeval_shards \
        --shard-root analysis/lm_eval_shards/3b-language-FINAL-step195311.pt \
        --output analysis/lm_eval/3b-language-FINAL-step195311.pt/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True,
                        help="Directory containing shard*/ subdirectories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shard_files = sorted(args.shard_root.glob("shard*/**/results.json"))
    if not shard_files:
        print(f"ERROR: no shard*/**/results.json under {args.shard_root}",
              file=sys.stderr)
        return 1

    merged: dict[str, dict] = {}
    for f in shard_files:
        try:
            with open(f) as fh:
                shard = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            print(f"ERROR: cannot read {f}: {e}", file=sys.stderr)
            return 1
        for task, res in shard.items():
            if task in merged and merged[task] != res:
                print(f"ERROR: task '{task}' appears in multiple shards "
                      f"with different results", file=sys.stderr)
                return 1
            merged[task] = res
        print(f"  {f}: {sorted(shard.keys())}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(merged, fh, indent=2, default=str)
    print(f"Merged {len(shard_files)} shards, {len(merged)} tasks "
          f"-> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
