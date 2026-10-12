"""Bindprobe aggregation: per family × condition × cell, plus one summary line
per (model, family).

Metrics (wrong = gold not rank 1 among the item's candidates; ties count wrong):
  acc     gold rank-1 rate
  cc_acc  chance-corrected acc, (acc - chance) / (1 - chance), chance = mean 1/C
          — the riders' formula (``riders.aggregate.cc_point`` / ``cc_acc``)
  margin  mean logp(gold) - max logp(distractor)
  err_*   share of each primary error type among WRONG rows

Summary (the primary numbers): per family, 3shot acc (collision-free exemplars),
collision gap = acc(3shot) - acc(3shot_collide), shot gap = acc(3shot) -
acc(0shot); the json also carries the chance-corrected versions + a 95% bootstrap
CI on 3shot cc_acc.

Input: a directory of grid JSONLs (``grid.run_grid`` output, one per model);
validated by the riders loader (hard-fails on multi-token candidates).
Output: ``bindprobes_tables.md`` + ``bindprobes_summary.json`` next to the grids.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kotodama.eval.bindprobes.base import cell_key
from kotodama.eval.bindprobes.items import CONDITIONS, FAMILY_NAMES, family
from kotodama.eval.riders.aggregate import cc_acc, cc_point, check_rows, load

Row = Mapping[str, Any]
TABLES_MD = "bindprobes_tables.md"
SUMMARY_JSON = "bindprobes_summary.json"


def _acc(rows: Sequence[Row]) -> float:
    return float(np.mean([r["gold_rank_cand"] == 1 for r in rows])) if rows else float("nan")


def cell_metrics(rows: Sequence[Row], error_types: Sequence[str]) -> dict[str, float]:
    """acc, cc_acc, margin, n and ``err_<type>`` shares (``tie`` included) over ``rows``."""
    out: dict[str, float] = {"acc": _acc(rows), "cc_acc": cc_point(list(rows)),
                             "margin": float(np.mean([r["margin"] for r in rows])),
                             "n": float(len(rows))}
    wrong = [r for r in rows if r["gold_rank_cand"] != 1]
    for t in tuple(error_types) + ("tie",):
        out[f"err_{t}"] = (sum(r.get("error_type") == t for r in wrong) / len(wrong)
                           if wrong else 0.0)
    return out


def _check_shape(rows: Sequence[Row]) -> None:
    bad = [r for r in rows if "family" not in r or "condition" not in r]
    if bad:
        raise ValueError(f"{len(bad)} rows are not bindprobe rows (no family/condition); "
                         f"first: {bad[0].get('item_id')!r}")


def _fmt(m: Mapping[str, float]) -> str:
    return f"{m['acc']:.2f} cc{m['cc_acc']:+.2f} m{m['margin']:+.2f}"


def aggregate(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """(markdown, summary dict) for validated bindprobe grid rows."""
    check_rows(rows)
    _check_shape(rows)
    models = sorted({r["model"] for r in rows})
    fams = [f for f in FAMILY_NAMES if any(r["family"] == f for r in rows)]
    by: defaultdict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for r in rows:
        by[(r["model"], r["family"], r["condition"])].append(r)

    summary: dict[str, Any] = {"summary": {}, "cells": {}, "errors": {}}
    out = ["# Bindprobes tables", "",
           ("## Summary — 3shot acc (collision-free exemplars), collision gap "
            "(3shot − 3shot_collide), shot gap (3shot − 0shot)"), "",
           ("| model | family | 3shot acc | 3shot cc [95% CI] | 0shot acc | collide acc "
            "| collision gap | shot gap | n/cond |"),
           "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        for f in fams:
            a = {c: by.get((m, f, c), []) for c in CONDITIONS}
            accs = {c: _acc(v) for c, v in a.items()}
            ccs = {c: cc_point(list(v)) if v else float("nan") for c, v in a.items()}
            ci = list(cc_acc(list(a["3shot"]))) if a["3shot"] else [float("nan")] * 3
            entry = {
                "acc_3shot": accs["3shot"], "acc_0shot": accs["0shot"],
                "acc_3shot_collide": accs["3shot_collide"],
                "collision_gap": accs["3shot"] - accs["3shot_collide"],
                "shot_gap": accs["3shot"] - accs["0shot"],
                "cc_3shot": ci[0], "cc_3shot_ci": ci[1:],
                "cc_collision_gap": ccs["3shot"] - ccs["3shot_collide"],
                "cc_shot_gap": ccs["3shot"] - ccs["0shot"],
                "n": {c: len(v) for c, v in a.items()},
            }
            summary["summary"][f"{m}|{f}"] = entry
            n_cond = max(len(v) for v in a.values())
            out.append(f"| {m} | {f} | {accs['3shot']:.3f} | {ci[0]:+.2f} "
                       f"[{ci[1]:+.2f},{ci[2]:+.2f}] | {accs['0shot']:.3f} | "
                       f"{accs['3shot_collide']:.3f} | {entry['collision_gap']:+.3f} | "
                       f"{entry['shot_gap']:+.3f} | {n_cond} |")
    out.append("")

    for f in fams:
        fam = family(f)
        keys = [cell_key(c) for c in fam.CELLS]
        for m in models:
            out += [f"## {f} — {m}: per cell (acc, chance-corrected acc, mean margin)", "",
                    "| cell | " + " | ".join(CONDITIONS) + " |",
                    "|---|" + "---|" * len(CONDITIONS)]
            for ck in keys:
                cells = []
                for c in CONDITIONS:
                    sub = [r for r in by.get((m, f, c), []) if r["cell_key"] == ck]
                    if not sub:
                        cells.append("—")
                        continue
                    met = cell_metrics(sub, fam.ERROR_TYPES)
                    summary["cells"][f"{m}|{f}|{c}|{ck}"] = met
                    cells.append(_fmt(met))
                out.append(f"| {ck} | " + " | ".join(cells) + " |")
            out.append("")
            types = tuple(fam.ERROR_TYPES) + ("tie",)
            out += [f"### {f} — {m}: error-type shares among wrong rows (all cells)", "",
                    "| condition | wrong | " + " | ".join(types) + " |",
                    "|---|---|" + "---|" * len(types)]
            for c in CONDITIONS:
                sub = by.get((m, f, c), [])
                if not sub:
                    continue
                met = cell_metrics(sub, fam.ERROR_TYPES)
                summary["errors"][f"{m}|{f}|{c}"] = {t: met[f"err_{t}"] for t in types}
                n_wrong = sum(r["gold_rank_cand"] != 1 for r in sub)
                out.append(f"| {c} | {n_wrong} | "
                           + " | ".join(f"{met[f'err_{t}']:.2f}" for t in types) + " |")
            out.append("")
            if f == "attr":
                out += _attr_recency(by, m)
    return "\n".join(out), summary


def _attr_recency(by: Mapping[tuple[str, str, str], list[Row]], model: str) -> list[str]:
    """attr, qpos=last: acc(gold is last mention) − acc(gold is not)."""
    lines = [f"### attr — {model}: recency control (qpos=last)", "",
             "| condition | acc rc | acc non-rc | Δ |", "|---|---|---|---|"]
    for c in CONDITIONS:
        sub = [r for r in by.get((model, "attr", c), []) if r.get("qpos") == "last"]
        rc = [r for r in sub if r.get("recency_control")]
        nrc = [r for r in sub if not r.get("recency_control")]
        if rc and nrc:
            lines.append(f"| {c} | {_acc(rc):.2f} | {_acc(nrc):.2f} | "
                         f"{_acc(rc) - _acc(nrc):+.2f} |")
    return lines + [""]


def write_tables(grid_dir: Path) -> tuple[Path, Path]:
    md, summary = aggregate(load(grid_dir))
    md_path, js_path = grid_dir / TABLES_MD, grid_dir / SUMMARY_JSON
    md_path.write_text(md)
    js_path.write_text(json.dumps(summary, indent=2))
    print(md)
    print(f"\n-> {md_path}")
    return md_path, js_path
