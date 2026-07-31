"""Equivalence gate: diff a new lm-eval results.json against a banked one.

Protocol (frozen with the eval-speed work, 2026-07-04): any harness change
(batching, sharding, kernels) must reproduce a banked row — same checkpoint,
same task — before its numbers enter a comparison. Point metrics (acc,
acc_norm, perplexity, ...) must match within --tol; stderr fields are
compared but only reported (they follow point metrics deterministically).

Usage:
    python -m scripts.eval.check_lmeval_equivalence \
        analysis/lm_eval/3b-language-FINAL-step195311.pt/results.json \
        analysis/lm_eval_batched/3b-language-FINAL-step195311.pt/results.json \
        --tol 0.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("banked", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--tol", type=float, default=0.0,
                        help="Max |delta| allowed on point metrics "
                             "(absolute, in metric units)")
    args = parser.parse_args()

    try:
        banked, new = load(args.banked), load(args.new)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    shared_tasks = sorted(set(banked) & set(new))
    if not shared_tasks:
        print("ERROR: no shared tasks between the two files", file=sys.stderr)
        return 2
    only_banked = sorted(set(banked) - set(new))
    if only_banked:
        print(f"note: tasks only in banked file (not checked): {only_banked}")

    failures: list[str] = []
    print(f"{'task':<18} {'metric':<28} {'banked':>12} {'new':>12} {'delta':>12}")
    for task in shared_tasks:
        for key, bval in sorted(banked[task].items()):
            if not isinstance(bval, (int, float)) or key == "alias":
                continue
            nval = new[task].get(key)
            if not isinstance(nval, (int, float)):
                failures.append(f"{task}/{key}: missing in new results")
                continue
            delta = nval - bval
            is_stderr = "stderr" in key
            flag = ""
            if not is_stderr and abs(delta) > args.tol:
                flag = "  <-- FAIL"
                failures.append(f"{task}/{key}: {bval} -> {nval} "
                                f"(|delta| {abs(delta):.6f} > tol {args.tol})")
            print(f"{task:<18} {key:<28} {bval:>12.6f} {nval:>12.6f} "
                  f"{delta:>+12.6f}{flag}")

    print()
    if failures:
        print(f"EQUIVALENCE FAIL ({len(failures)} metric(s) beyond "
              f"tol={args.tol}):")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"EQUIVALENCE PASS: all point metrics within tol={args.tol} "
          f"across {len(shared_tasks)} shared task(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
