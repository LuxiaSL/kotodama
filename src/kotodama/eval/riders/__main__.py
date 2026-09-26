"""Rider-grid CLI.

    python -m kotodama.eval.riders gen --out riders/items           # banked pool, seed 1002
    python -m kotodama.eval.riders run --model kotodama-3b-base --kind koto \\
        --items riders/items/items.jsonl --shots riders/items/shots.json \\
        --out riders/grids/kotodama-3b-base.jsonl [--endpoint URL]
    python -m kotodama.eval.riders aggregate --grids riders/grids
    python -m kotodama.eval.riders smoothing --items riders/items/items.jsonl --grids riders/grids

The endpoint defaults to ``$KOTODAMA_ENDPOINT``, else http://localhost:2222.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kotodama.eval.riders import aggregate as agg
from kotodama.eval.riders import grid, items, smoothing


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m kotodama.eval.riders",
                                 description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="emit items.jsonl + shots.json")
    g.add_argument("--out", required=True, help="output directory")
    g.add_argument("--names", default=None,
                   help="verify-names JSON (default: the banked 2026-08-02 pool)")
    g.add_argument("--per-cell", type=int, default=200)
    g.add_argument("--d0-count", type=int, default=200)
    g.add_argument("--seed", type=int, default=items.DEFAULT_SEED)

    v = sub.add_parser("verify-names", help="re-verify the single-token name pool")
    v.add_argument("--out", required=True, help="verified_names.json path")

    sub.add_parser("selftest", help="generator invariant checks")

    r = sub.add_parser("run", help="score a grid against /logprobs")
    r.add_argument("--model", required=True)
    r.add_argument("--kind", choices=grid.KINDS, required=True)
    r.add_argument("--endpoint", default=None,
                   help="server URL (default: $KOTODAMA_ENDPOINT or "
                        f"{grid.DEFAULT_ENDPOINT})")
    r.add_argument("--items", required=True)
    r.add_argument("--shots", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--renderers", default=",".join(items.RENDERERS))
    r.add_argument("--tiers", default="d0,std", help="d0 = gate only; std = full grid")
    r.add_argument("--batch", type=int, default=grid.BATCH)
    r.add_argument("--top-n", type=int, default=5)

    a = sub.add_parser("aggregate", help="gate + four-register tables")
    a.add_argument("--grids", required=True, help="directory of grid JSONLs")

    s = sub.add_parser("smoothing", help="smoothing index (attractor pull)")
    s.add_argument("--items", required=True)
    s.add_argument("--grids", required=True)
    s.add_argument("--nshots", type=int, default=3)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "gen":
            items.emit(Path(args.names) if args.names else None, Path(args.out),
                       args.per_cell, args.d0_count, args.seed)
        elif args.cmd == "verify-names":
            items.verify_names(Path(args.out))
        elif args.cmd == "selftest":
            return 1 if items.selftest() else 0
        elif args.cmd == "run":
            grid.run_grid_files(args.model, args.kind, Path(args.items), Path(args.shots),
                                Path(args.out), endpoint=args.endpoint,
                                renderers=_csv(args.renderers), tiers=_csv(args.tiers),
                                batch=args.batch, top_n=args.top_n)
        elif args.cmd == "aggregate":
            agg.write_tables(Path(args.grids))
        elif args.cmd == "smoothing":
            smoothing.write_smoothing(Path(args.items), Path(args.grids), args.nshots)
    except (ValueError, FileNotFoundError, grid.LogprobsError,
            agg.MultiTokenCandidateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
