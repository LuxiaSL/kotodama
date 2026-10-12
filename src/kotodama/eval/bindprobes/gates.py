"""deixis validity gates V1–V5 (SPEC-bindprobes-deixis §5): the family is not a
ruler for any arm until every gate passes on the validation set.

  V1 construction    every item's gold reproduces from the simulator and every
                     pronoun resolves uniquely (``items.validate`` incl. the surface
                     addressee-rule check), pairs link (``deixis.check_pairs``), the
                     shortcut controls sit within ±5% per cell
                     (``deixis.audit_failures``), every grid row's gold matches its
                     item, and no row carries a multi-token candidate.
  V2 chance          at the ``chance`` checkpoint (0.25B), deictic 3shot cc acc within
                     ±.05 of 0 on every query kind.
  V3 learnability    at the ``learn`` checkpoint (A0* 30B), third 3shot cc ≥ .20.
  V4 control order   third ≥ deictic − .03 (3shot cc) on every ``ge2b`` checkpoint
                     (default: every model in the grids except the chance one).
  V5 twin sanity     at the chance checkpoint, 3shot twin-pair joint accuracy ≤
                     chance² + .03 (chance² = mean over pairs of (1/C_x)(1/C_twin)).

Model names are arguments (the grid's ``model`` field); nothing is hardcoded.
CLI: ``python -m kotodama.eval.bindprobes gates --grids DIR --items items.jsonl
--chance-model NAME --learn-model NAME [--ge2b A,B,...]`` — prints PASS/FAIL per
gate, writes ``bindprobes_gates.json`` next to the grids, exits 1 if any gate fails.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from kotodama.eval.bindprobes import deixis
from kotodama.eval.bindprobes.aggregate import DEIXIS, deixis_readouts, twin_consistency
from kotodama.eval.bindprobes.base import Item
from kotodama.eval.bindprobes.items import load_items, validate
from kotodama.eval.riders.aggregate import MultiTokenCandidateError, check_rows, load

GATES_JSON = "bindprobes_gates.json"
V2_TOL = 0.05
V3_MIN = 0.20
V4_SLACK = 0.03
V5_SLACK = 0.03
PRIMARY = "3shot"

Row = Mapping[str, Any]


@dataclass
class GateResult:
    gate: str
    passed: bool
    detail: str
    values: dict[str, Any] = field(default_factory=dict)


def v1_construction(items: Sequence[Item], rows: Sequence[Row]) -> GateResult:
    its = [it for it in items if it.family == DEIXIS]
    if not its:
        return GateResult("V1", False, "no deixis items")
    bad: list[str] = []
    for it in its:
        try:
            validate(it)
        except ValueError as exc:
            bad.append(str(exc))
    try:
        deixis.check_pairs(its)
    except ValueError as exc:
        bad.append(f"pairs: {exc}")
    aud = deixis.audit(its)
    balance = deixis.audit_failures(aud)
    gold = {it.item_id: it.gold for it in its}
    drows = [r for r in rows if r.get("family") == DEIXIS]
    mismatch = [r["item_id"] for r in drows if gold.get(r["item_id"], r["gold"]) != r["gold"]]
    unknown = sorted({r["item_id"] for r in drows if r["item_id"] not in gold})
    multi = [r["item_id"] for r in drows if r.get("multi_token_candidates")]
    ok = not (bad or balance or mismatch or unknown or multi)
    detail = (f"{len(its)} items: {len(bad)} invalid, {len(balance)} balance failures, "
              f"{len(mismatch)} gold mismatches, {len(unknown)} rows without an item, "
              f"{len(multi)} multi-token rows")
    return GateResult("V1", ok, detail, {"invalid": bad[:20], "balance": balance,
                                          "gold_mismatch": mismatch[:20],
                                          "unknown_rows": unknown[:20], "multi_token": multi[:20],
                                          "audit": aud})


def _missing(model: str, d: Mapping[str, Any], gate: str) -> Optional[GateResult]:
    if model not in d:
        return GateResult(gate, False, f"no deixis rows for model {model!r}")
    return None


def v2_chance(readouts: Mapping[str, Any], chance_model: str) -> GateResult:
    miss = _missing(chance_model, readouts, "V2")
    if miss:
        return miss
    cc = readouts[chance_model]["cc"]
    vals = {k: cc["deictic"][PRIMARY][k] for k in deixis.QKINDS}
    info = {f"{v}/{k}": cc[v][PRIMARY][k] for v in ("third", "twin") for k in deixis.QKINDS}
    ok = all(not math.isnan(v) and abs(v) <= V2_TOL for v in vals.values())
    detail = f"{chance_model} deictic {PRIMARY} cc " + ", ".join(
        f"{k} {v:+.3f}" for k, v in vals.items()) + f" (need |cc| ≤ {V2_TOL})"
    return GateResult("V2", ok, detail, {"deictic": vals, "info_other_variants": info})


def v3_learnability(readouts: Mapping[str, Any], learn_model: str) -> GateResult:
    miss = _missing(learn_model, readouts, "V3")
    if miss:
        return miss
    v = readouts[learn_model]["cc"]["third"][PRIMARY]["all"]
    ok = not math.isnan(v) and v >= V3_MIN
    return GateResult("V3", ok, f"{learn_model} third {PRIMARY} cc {v:+.3f} (need ≥ {V3_MIN})",
                      {"third_cc": v})


def v4_control_order(readouts: Mapping[str, Any], models: Sequence[str]) -> GateResult:
    if not models:
        return GateResult("V4", False, "no >= 2B checkpoints given/found")
    vals: dict[str, dict[str, float]] = {}
    fails: list[str] = []
    for m in models:
        if m not in readouts:
            fails.append(f"{m}: no rows")
            continue
        t = readouts[m]["cc"]["third"][PRIMARY]["all"]
        x = readouts[m]["cc"]["deictic"][PRIMARY]["all"]
        vals[m] = {"third": t, "deictic": x}
        if math.isnan(t) or math.isnan(x) or t < x - V4_SLACK:
            fails.append(f"{m}: third {t:+.3f} < deictic {x:+.3f} − {V4_SLACK}")
    detail = (f"{len(models)} checkpoints; " + ("; ".join(fails) if fails
                                                else f"third ≥ deictic − {V4_SLACK} on all"))
    return GateResult("V4", not fails, detail, vals)


def v5_twin(rows: Sequence[Row], chance_model: str) -> GateResult:
    sub = [r for r in rows if r.get("family") == DEIXIS and r["model"] == chance_model
           and r["condition"] == PRIMARY]
    t = twin_consistency([r for r in sub if r.get("variant") == "deictic"],
                         [r for r in sub if r.get("variant") == "twin"])
    if not t["n_pairs"]:
        return GateResult("V5", False, f"no {PRIMARY} twin pairs for {chance_model!r}")
    ok = t["joint"] <= t["chance2"] + V5_SLACK
    return GateResult("V5", ok, f"{chance_model} twin joint {t['joint']:.3f} vs chance² "
                                f"{t['chance2']:.3f} + {V5_SLACK} ({int(t['n_pairs'])} pairs)", t)


def gates(rows: Sequence[Row], items: Sequence[Item], chance_model: str, learn_model: str,
          ge2b_models: Optional[Sequence[str]] = None) -> list[GateResult]:
    """V1–V5 over grid rows (any families; deixis is read) + the emitted items."""
    rows = list(rows)
    out = [v1_construction(items, rows)]
    try:
        check_rows(rows)
    except MultiTokenCandidateError:
        pass   # V1 already reports the multi-token rows; the read gates still run
    drows = [r for r in rows if r.get("family") == DEIXIS]
    readouts = deixis_readouts(drows) if drows else {}
    if ge2b_models is None:
        ge2b_models = sorted(m for m in readouts if m != chance_model)
    out += [v2_chance(readouts, chance_model), v3_learnability(readouts, learn_model),
            v4_control_order(readouts, ge2b_models), v5_twin(drows, chance_model)]
    return out


def _load_rows_lenient(grid_dir: Path) -> list[dict[str, Any]]:
    """Every grid row; a multi-token hard-fail is reported by V1 instead of raised."""
    try:
        return load(grid_dir)
    except MultiTokenCandidateError:
        rows: list[dict[str, Any]] = []
        for p in sorted(grid_dir.glob("*.jsonl")):
            rows += [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
        return rows


def write_gates(grid_dir: Path, items_path: Path, chance_model: str, learn_model: str,
                ge2b_models: Optional[Sequence[str]] = None) -> tuple[bool, Path]:
    results = gates(_load_rows_lenient(grid_dir), load_items(items_path), chance_model,
                    learn_model, ge2b_models)
    for r in results:
        print(f"{r.gate} {'PASS' if r.passed else 'FAIL'}  {r.detail}")
    ok = all(r.passed for r in results)
    print(f"deixis gates: {'ALL PASS' if ok else 'FAIL'}")
    path = grid_dir / GATES_JSON
    path.write_text(json.dumps({"passed": ok, "chance_model": chance_model,
                                "learn_model": learn_model,
                                "gates": [asdict(r) for r in results]}, indent=2, default=str))
    print(f"-> {path}")
    return ok, path
