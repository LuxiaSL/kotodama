"""Bindprobes CLI.

    python -m kotodama.eval.bindprobes gen --out bindprobes/items [--verify]   # banked pools, seed 2026
    python -m kotodama.eval.bindprobes selftest
    python -m kotodama.eval.bindprobes run --model kotodama-3b-base \\
        --items bindprobes/items/items.jsonl --out bindprobes/grids/kotodama-3b-base.jsonl \\
        [--endpoint URL] [--families attr,codevar,dialogue] [--conditions 0shot,3shot,3shot_collide]
    python -m kotodama.eval.bindprobes aggregate --grids bindprobes/grids
    python -m kotodama.eval.bindprobes verify-names --out bindprobes/verified_names.json [--tokenizers smollm2]

The endpoint defaults to ``$KOTODAMA_ENDPOINT``, else http://localhost:2222.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kotodama.eval.bindprobes import aggregate as agg
from kotodama.eval.bindprobes import grid, items, pools
from kotodama.eval.riders.aggregate import MultiTokenCandidateError
from kotodama.eval.riders.grid import LogprobsError


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _per_cell(s: str | None) -> dict[str, int]:
    """``attr=110,codevar=100`` or a bare ``N`` for every family."""
    if not s:
        return {}
    if "=" not in s:
        return {f: int(s) for f in items.FAMILY_NAMES}
    out = {}
    for part in _csv(s):
        k, v = part.split("=", 1)
        items.family(k)
        out[k] = int(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m kotodama.eval.bindprobes",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="emit items.jsonl (+ emit_meta.json)")
    g.add_argument("--out", required=True, help="output directory")
    g.add_argument("--seed", type=int, default=items.DEFAULT_SEED)
    g.add_argument("--families", default=",".join(items.FAMILY_NAMES))
    g.add_argument("--per-cell", default=None,
                   help="N for every family, or 'attr=110,codevar=100,dialogue=100' "
                        "(default: each family's ~1200-item default)")
    g.add_argument("--names", default=None,
                   help="verify-names JSON (default: the banked pools)")
    g.add_argument("--verify", action="store_true",
                   help="load the presets tokenizer: hard-fail on multi-token pool "
                        "strings or prompts >= the length bound")

    v = sub.add_parser("verify-names", help="re-verify the single-token name/object pools")
    v.add_argument("--out", required=True)
    v.add_argument("--tokenizers", default=None,
                   help="riders GRID_TOKENIZERS keys (default: all)")

    sub.add_parser("selftest", help="generator/simulator invariant checks")

    r = sub.add_parser("run", help="score items against /logprobs")
    r.add_argument("--model", required=True)
    r.add_argument("--items", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--endpoint", default=None,
                   help=f"server URL (default: $KOTODAMA_ENDPOINT or {grid.DEFAULT_ENDPOINT})")
    r.add_argument("--families", default=",".join(items.FAMILY_NAMES))
    r.add_argument("--conditions", default=",".join(items.CONDITIONS))
    r.add_argument("--batch", type=int, default=grid.BATCH)
    r.add_argument("--top-n", type=int, default=5)

    a = sub.add_parser("aggregate", help="summary + per-cell + error-type tables")
    a.add_argument("--grids", required=True, help="directory of bindprobe grid JSONLs")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "gen":
            fams = _csv(args.families)
            for f in fams:
                items.family(f)
            items.emit(Path(args.out), args.seed, fams, _per_cell(args.per_cell),
                       Path(args.names) if args.names else None, verify=args.verify)
        elif args.cmd == "verify-names":
            keys = _csv(args.tokenizers) if args.tokenizers else list(pools.GRID_TOKENIZERS)
            pools.verify_names(Path(args.out), keys)
        elif args.cmd == "selftest":
            return 1 if items.selftest() else 0
        elif args.cmd == "run":
            grid.run_grid_files(args.model, Path(args.items), Path(args.out),
                                endpoint=args.endpoint, families=_csv(args.families),
                                conditions=_csv(args.conditions), batch=args.batch,
                                top_n=args.top_n)
        elif args.cmd == "aggregate":
            agg.write_tables(Path(args.grids))
    except (ValueError, FileNotFoundError, RuntimeError, LogprobsError,
            MultiTokenCandidateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
