"""Smoothing index — attractor-pull vs update, read from banked rider grids.

On items whose QUERIED box changed content after declaration via swap/move (the
value is never re-named near the query), compare mass on the ORIGINALLY
DECLARED occupant (the momentum answer) vs the post-op GOLD:

    SI = P(declared) / (P(declared) + P(gold))

SI ≈ 0.5 = no preference; SI → 1 = the attractor rides over the update;
SI → 0 = clean rebinding. Per model × renderer (3-shot primary). Boxes whose
final content was put/overwritten (named in an op sentence — recency-copy
contaminates the read) are excluded.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class SIRow:
    model: str
    renderer: str
    si: float
    ci: list[float]
    n: int


def qualifying_items(items_path: Path) -> dict[str, str]:
    """item_id -> declared occupant, for std items whose queried box's final
    content arrived via swap/move and differs from the declared occupant."""
    out: dict[str, str] = {}
    for line in items_path.read_text().splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        if it["tier"] != "std" or it["m"] == 0:
            continue
        q = it["query_box"]
        declared = it["boxes0"].get(q)
        gold = it["gold"]
        if declared is None or declared == gold:
            continue
        # replay ops: did q's last touch name its object (put/overwrite) or not?
        named = False
        touched = False
        for op in it["ops"]:
            k = op["kind"]
            if k in ("put", "overwrite") and op["i"] == q:
                named, touched = True, True
            elif k == "swap" and q in (op["i"], op["j"]):
                named, touched = False, True
            elif k == "move" and op["j"] == q:
                named, touched = False, True
            elif k == "move" and op["i"] == q:
                named, touched = False, True  # emptied; later ops may refill
        if touched and not named:
            out[it["item_id"]] = declared
    return out


def smoothing_index(items_path: Path, grid_dir: Path, nshots: int = 3) -> list[SIRow]:
    """SI with bootstrap 95% CI (1000 resamples, seed 17) per (model, renderer)."""
    momentum = qualifying_items(items_path)
    print(f"{len(momentum)} qualifying items (swap/move-final, declared != gold)\n")

    rows_by: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for p in sorted(grid_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (r["tier"] != "std" or r["nshots"] != nshots
                    or r["item_id"] not in momentum):
                continue
            decl = momentum[r["item_id"]]
            lps = r["cand_lps"]
            if decl not in lps or r["gold"] not in lps:
                continue
            pd, pg = np.exp(lps[decl]), np.exp(lps[r["gold"]])
            rows_by[(r["model"], r["renderer"])].append(pd / (pd + pg))

    print(f"{'model':<22} {'renderer':<11} {'SI':>6} {'[95% CI]':>16} {'n':>5}")
    rng = np.random.default_rng(17)
    out_rows: list[SIRow] = []
    for (model, rend), vals in sorted(rows_by.items()):
        v = np.array(vals)
        boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"{model:<22} {rend:<11} {v.mean():>6.3f} "
              f"[{lo:>6.3f},{hi:>6.3f}] {len(v):>5}")
        out_rows.append(SIRow(model=model, renderer=rend, si=float(v.mean()),
                              ci=[float(lo), float(hi)], n=len(v)))
    return out_rows


def write_smoothing(items_path: Path, grid_dir: Path, nshots: int = 3) -> Path:
    """Compute SI and write ``smoothing_index.json`` into ``grid_dir``."""
    rows = smoothing_index(items_path, grid_dir, nshots)
    out = grid_dir / "smoothing_index.json"
    out.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    print(f"\n-> {out}")
    return out
