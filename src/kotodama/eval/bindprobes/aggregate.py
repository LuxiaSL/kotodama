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

deixis: the generic tables above read its DEICTIC variant only (its paired
``third`` / ``twin`` rows would otherwise be pooled in); a dedicated section +
``summary["deixis"]`` carries the paired readouts (see :func:`deixis_readouts`):
cc acc per variant × condition × query kind, the DEIXIS PENALTY cc(third) −
cc(deictic), TWIN CONSISTENCY (joint accuracy of x and twin(x); joint − product
of marginals; chance² for reference), self − addressee asymmetry, collision gap
per variant, and error-type shares per variant. Validity gates V1–V5:
``bindprobes.gates``. Deixis rows must carry ``deixis_version`` = the current
generation (``deixis.VERSION``); other rows are refused (``deixis.check_version``), never pooled.
P = 2 deixis pairs have no twin, so twin consistency reads the P = 3 pairs.

Input: a directory of grid JSONLs (``grid.run_grid`` output, one per model);
validated by the riders loader (hard-fails on multi-token candidates).
Output: ``bindprobes_tables.md`` + ``bindprobes_summary.json`` next to the grids.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kotodama.eval.bindprobes import deixis as deixis_family
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
    all_rows = rows
    rows = [r for r in rows if not _paired_non_base(r)]
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
    deixis_rows = [r for r in all_rows if r["family"] == DEIXIS]
    if deixis_rows:
        readouts = deixis_readouts(deixis_rows)
        summary["deixis"] = readouts
        out += deixis_markdown(readouts)
    return "\n".join(out), summary


# ── deixis: paired readouts ────────────────────────────────────────────────────

DEIXIS = "deixis"
DEIXIS_VARIANTS = ("deictic", "third", "twin")
DEIXIS_KINDS = ("owner", "self", "addressee")
_ALL = "all"


def _paired_non_base(r: Row) -> bool:
    """A deixis third/twin row (kept out of the generic per-family tables)."""
    return r["family"] == DEIXIS and r.get("variant", "deictic") != "deictic"


def _correct(r: Row) -> bool:
    return bool(r["gold_rank_cand"] == 1)


def twin_consistency(x_rows: Sequence[Row], twin_rows: Sequence[Row]) -> dict[str, float]:
    """Pairs joined on ``pair_id``: P(x), P(twin), joint P(x ∧ twin), dependence =
    joint − P(x)·P(twin), and chance² = mean over pairs of (1/C_x)(1/C_twin)."""
    tw = {r["pair_id"]: r for r in twin_rows}
    pairs = [(r, tw[r["pair_id"]]) for r in x_rows if r["pair_id"] in tw]
    if not pairs:
        return {"n_pairs": 0.0, "p_x": float("nan"), "p_twin": float("nan"),
                "joint": float("nan"), "dependence": float("nan"), "chance2": float("nan")}
    px = float(np.mean([_correct(a) for a, _ in pairs]))
    pt = float(np.mean([_correct(b) for _, b in pairs]))
    joint = float(np.mean([_correct(a) and _correct(b) for a, b in pairs]))
    chance2 = float(np.mean([1.0 / a["n_candidates"] / b["n_candidates"] for a, b in pairs]))
    return {"n_pairs": float(len(pairs)), "p_x": px, "p_twin": pt, "joint": joint,
            "dependence": joint - px * pt, "chance2": chance2}


def deixis_readouts(rows: Sequence[Row]) -> dict[str, Any]:
    """Per model (keys = model names) — every number keyed [variant|cond|kind]:

      cc / acc / n     [variant][cond][kind]  (kind ∈ all, owner, self, addressee)
      penalty          [cond][kind] = cc(third) − cc(deictic)
      twin             [cond][kind] = :func:`twin_consistency` of deictic vs twin
      asymmetry        [variant][cond] = cc(self) − cc(addressee)
      collision_gap    [variant] = {acc, cc}: 3shot − 3shot_collide
      errors           [variant][cond] = primary error-type shares among wrong rows
    """
    deixis_family.check_version([r for r in rows if r["family"] == DEIXIS], "grid rows")
    by: defaultdict[tuple[str, str, str], list[Row]] = defaultdict(list)
    for r in rows:
        if r["family"] != DEIXIS:
            continue
        v = r.get("variant")
        if v not in DEIXIS_VARIANTS or r.get("qkind") not in DEIXIS_KINDS:
            raise ValueError(f"deixis row {r.get('item_id')!r} lacks variant/qkind controls")
        by[(r["model"], v, r["condition"])].append(r)
    err_types = family(DEIXIS).ERROR_TYPES
    out: dict[str, Any] = {}
    for m in sorted({k[0] for k in by}):
        cc: dict[str, Any] = {}
        acc: dict[str, Any] = {}
        n: dict[str, Any] = {}
        errors: dict[str, Any] = {}
        for v in DEIXIS_VARIANTS:
            cc[v], acc[v], n[v], errors[v] = {}, {}, {}, {}
            for c in CONDITIONS:
                sub = by.get((m, v, c), [])
                cc[v][c], acc[v][c], n[v][c] = {}, {}, {}
                for kind in (_ALL,) + DEIXIS_KINDS:
                    ks = sub if kind == _ALL else [r for r in sub if r["qkind"] == kind]
                    cc[v][c][kind] = cc_point(list(ks)) if ks else float("nan")
                    acc[v][c][kind] = _acc(ks)
                    n[v][c][kind] = len(ks)
                if sub:
                    met = cell_metrics(sub, err_types)
                    errors[v][c] = {t: met[f"err_{t}"] for t in tuple(err_types) + ("tie",)}
        penalty = {c: {k: cc["third"][c][k] - cc["deictic"][c][k] for k in (_ALL,) + DEIXIS_KINDS}
                   for c in CONDITIONS}
        twin: dict[str, Any] = {}
        for c in CONDITIONS:
            xs, ts = by.get((m, "deictic", c), []), by.get((m, "twin", c), [])
            twin[c] = {k: twin_consistency([r for r in xs if k == _ALL or r["qkind"] == k],
                                           [r for r in ts if k == _ALL or r["qkind"] == k])
                       for k in (_ALL,) + DEIXIS_KINDS}
        asym = {v: {c: cc[v][c]["self"] - cc[v][c]["addressee"] for c in CONDITIONS}
                for v in DEIXIS_VARIANTS}
        gap = {v: {"acc": acc[v]["3shot"][_ALL] - acc[v]["3shot_collide"][_ALL],
                   "cc": cc[v]["3shot"][_ALL] - cc[v]["3shot_collide"][_ALL]}
               for v in DEIXIS_VARIANTS}
        out[m] = {"cc": cc, "acc": acc, "n": n, "penalty": penalty, "twin": twin,
                  "asymmetry": asym, "collision_gap": gap, "errors": errors}
    return out


def _f(x: float) -> str:
    return "—" if math.isnan(x) else f"{x:+.3f}"


def deixis_markdown(readouts: Mapping[str, Any]) -> list[str]:
    kinds = (_ALL,) + DEIXIS_KINDS
    types = tuple(family(DEIXIS).ERROR_TYPES) + ("tie",)
    out: list[str] = []
    for m, d in readouts.items():
        out += [f"## deixis — {m}: chance-corrected acc by variant × condition × query kind", "",
                "| variant | condition | " + " | ".join(kinds) + " | n (all) |",
                "|---|---|" + "---|" * (len(kinds) + 1)]
        for v in DEIXIS_VARIANTS:
            for c in CONDITIONS:
                out.append(f"| {v} | {c} | " + " | ".join(_f(d["cc"][v][c][k]) for k in kinds)
                           + f" | {d['n'][v][c][_ALL]} |")
        out += ["", (f"### deixis — {m}: DEIXIS PENALTY = cc(third) − cc(deictic) "
                     "(primary: 3shot)"), "",
                "| condition | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
        for c in CONDITIONS:
            out.append(f"| {c} | " + " | ".join(_f(d["penalty"][c][k]) for k in kinds) + " |")
        out += ["", (f"### deixis — {m}: TWIN CONSISTENCY (x = deictic, twin = speakers "
                     "transposed)"), "",
                ("| condition | kind | P(x) | P(twin) | joint | joint − P(x)P(twin) | chance² "
                 "| pairs |"), "|---|---|---|---|---|---|---|---|"]
        for c in CONDITIONS:
            for k in kinds:
                t = d["twin"][c][k]
                out.append(f"| {c} | {k} | {_f(t['p_x'])} | {_f(t['p_twin'])} | "
                           f"{_f(t['joint'])} | {_f(t['dependence'])} | {_f(t['chance2'])} | "
                           f"{int(t['n_pairs'])} |")
        out += ["", (f"### deixis — {m}: self − addressee asymmetry (cc) and collision gap "
                     "(3shot − 3shot_collide)"), "",
                "| variant | asym 0shot | asym 3shot | asym collide | gap acc | gap cc |",
                "|---|---|---|---|---|---|"]
        for v in DEIXIS_VARIANTS:
            a, g = d["asymmetry"][v], d["collision_gap"][v]
            out.append(f"| {v} | {_f(a['0shot'])} | {_f(a['3shot'])} | "
                       f"{_f(a['3shot_collide'])} | {_f(g['acc'])} | {_f(g['cc'])} |")
        out += ["", f"### deixis — {m}: error-type shares among wrong rows", "",
                "| variant | condition | " + " | ".join(types) + " |",
                "|---|---|" + "---|" * len(types)]
        for v in DEIXIS_VARIANTS:
            for c in CONDITIONS:
                e = d["errors"][v].get(c)
                if e:
                    out.append(f"| {v} | {c} | " + " | ".join(f"{e[t]:.2f}" for t in types) + " |")
        out.append("")
    return out


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
