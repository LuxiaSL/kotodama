"""Rider-grid aggregation: difficulty-0 gate + the four-register margin tables.

Frozen bars (P0.6 prereg):
  gate 1: d0 rank-1 rate >= 95% per (model x renderer), read at 3-shot; failing
          renderer cells are marked HARNESS-BUG and their std reads are printed
          struck-through (still computed, never trusted).
  chance-corrected accuracy: (acc - chance) / (1 - chance), chance = mean 1/C.
  bootstrap 95% CIs over items (1000 resamples, percentile, seed 17).

Input: a directory of grid JSONLs (one per model, ``grid.run_grid`` output).
Output: stage1_tables.md + stage1_summary.json next to the grids.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

D0_BAR = 0.95
BOOT = 1000
RENDERER_ORDER = ["code", "narrative", "screenplay", "chat"]

Row = dict[str, Any]


class MultiTokenCandidateError(RuntimeError):
    """A grid row scored a multi-token candidate: name verification was violated."""


def check_rows(rows: list[Row]) -> None:
    """HARD FAIL if any row carries multi-token candidates — no read is valid."""
    bad = [r for r in rows if r.get("multi_token_candidates")]
    if bad:
        raise MultiTokenCandidateError(
            f"HARD FAIL: {len(bad)} rows carry multi-token candidates "
            f"(first: {bad[0]['model']}/{bad[0]['item_id']}) — name verification "
            f"was violated; no reads are valid.")


def load(grid_dir: Path) -> list[Row]:
    """Every row of every ``*.jsonl`` under ``grid_dir`` (validated)."""
    if not grid_dir.is_dir():
        raise FileNotFoundError(f"grid dir not found: {grid_dir}")
    rows: list[Row] = []
    for p in sorted(grid_dir.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    check_rows(rows)
    return rows


def cc_acc(rows: list[Row]) -> tuple[float, float, float]:
    """Chance-corrected accuracy with bootstrap 95% CI: (point, lo, hi)."""
    top1 = np.array([r["gold_rank_cand"] == 1 for r in rows], dtype=float)
    chance = np.array([1.0 / r["n_candidates"] for r in rows])

    def cc(idx: np.ndarray) -> float:
        a, c = top1[idx].mean(), chance[idx].mean()
        return (a - c) / (1 - c) if c < 1 else 0.0

    n = len(rows)
    point = cc(np.arange(n))
    rng = np.random.default_rng(17)
    boots = [cc(rng.integers(0, n, n)) for _ in range(BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def fmt_cell(rows: list[Row]) -> str:
    """``cc [lo,hi] r<median rank> m<mean margin>``, or an em dash if empty."""
    if not rows:
        return "—"
    point, lo, hi = cc_acc(rows)
    med_rank = int(np.median([r["gold_rank_cand"] for r in rows]))
    margin = float(np.mean([r["margin"] for r in rows]))
    return f"{point:+.2f} [{lo:+.2f},{hi:+.2f}] r{med_rank} m{margin:+.2f}"


def aggregate(rows: list[Row],
              renderers: list[str] = RENDERER_ORDER) -> tuple[str, dict[str, Any]]:
    """(markdown tables, summary dict) for validated grid rows."""
    check_rows(rows)
    models = sorted({r["model"] for r in rows})

    by: defaultdict[tuple[str, str, int, str], list[Row]] = defaultdict(list)
    for r in rows:
        by[(r["model"], r["renderer"], r["nshots"], r["tier"])].append(r)

    out: list[str] = ["# P0.6 Stage-1 tables", ""]
    summary: dict[str, Any] = {"d0_gate": {}, "cells": {}}

    # ── Gate 1: difficulty-0 ────────────────────────────────────────────────
    out += ["## Difficulty-0 gate (rank-1 rate; bar ≥95% at 3-shot)", "",
            "| model | " + " | ".join(renderers) + " |",
            "|---|" + "---|" * len(renderers)]
    gate_fail: set[tuple[str, str]] = set()
    for m in models:
        cells = []
        for rd in renderers:
            d0 = by.get((m, rd, 3, "d0"), [])
            d0z = by.get((m, rd, 0, "d0"), [])
            if not d0:
                cells.append("—")
                continue
            rate = np.mean([r["gold_rank_cand"] == 1 for r in d0])
            rate0 = (np.mean([r["gold_rank_cand"] == 1 for r in d0z])
                     if d0z else float("nan"))
            ok = rate >= D0_BAR
            if not ok:
                gate_fail.add((m, rd))
            mark = "" if ok else " **HARNESS-BUG**"
            cells.append(f"{rate:.1%} (0s {rate0:.1%}){mark}")
            summary["d0_gate"][f"{m}|{rd}"] = {"rate_3shot": float(rate),
                                               "rate_0shot": float(rate0),
                                               "pass": bool(ok)}
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out.append("")

    # ── Stage-1 margin table (std tier, pooled over N,M) ────────────────────
    for nshots in (3, 0):
        out += [f"## Four-register table — {nshots}-shot (std tier, pooled; "
                "chance-corrected acc [95% CI], median rank, mean margin)", "",
                "| model | " + " | ".join(renderers) + " |",
                "|---|" + "---|" * len(renderers)]
        for m in models:
            cells = []
            for rd in renderers:
                sub = by.get((m, rd, nshots, "std"), [])
                cell = fmt_cell(sub)
                if (m, rd) in gate_fail:
                    cell = f"~~{cell}~~"
                cells.append(cell)
                if sub:
                    p, lo, hi = cc_acc(sub)
                    summary["cells"][f"{m}|{rd}|{nshots}shot"] = {
                        "cc_acc": p, "ci": [lo, hi], "n": len(sub),
                        "gate_failed": (m, rd) in gate_fail}
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        out.append("")

    # ── N×M breakdown per model (3-shot, recency-control excluded) ──────────
    ns = sorted({r["n"] for r in rows if r["tier"] == "std"})
    ms_ = sorted({r["m"] for r in rows if r["tier"] == "std"})
    for m in models:
        out += [f"## {m} — N×M grid, 3-shot (chance-corrected acc)", "",
                "| renderer | " + " | ".join(f"N{n}" for n in ns) + " |",
                "|---|" + "---|" * len(ns)]
        for rd in renderers:
            cells = []
            for n in ns:
                parts = []
                for mm in ms_:
                    sub = [r for r in by.get((m, rd, 3, "std"), [])
                           if r["n"] == n and r["m"] == mm
                           and not r["recency_control"]]
                    if sub:
                        p, _, _ = cc_acc(sub)
                        parts.append(f"M{mm}:{p:+.2f}")
                cells.append(" ".join(parts) or "—")
            out.append(f"| {rd} | " + " | ".join(cells) + " |")
        out.append("")

    # ── Recency delta + shot gap (3-shot, std, m>0) ─────────────────────────
    out += ["## Recency-prior delta (acc_rc − acc_std, M>0, 3-shot) and "
            "shot gap (3s − 0s, std)", "",
            "| model | renderer | recency Δ | shot gap |", "|---|---|---|---|"]
    for m in models:
        for rd in renderers:
            s3 = by.get((m, rd, 3, "std"), [])
            s0 = by.get((m, rd, 0, "std"), [])
            std3 = [r for r in s3 if r["m"] > 0]
            rc = [r for r in std3 if r["recency_control"]]
            nrc = [r for r in std3 if not r["recency_control"]]
            if not (rc and nrc and s3 and s0):
                continue
            racc = np.mean([r["gold_rank_cand"] == 1 for r in rc])
            nacc = np.mean([r["gold_rank_cand"] == 1 for r in nrc])
            a3, _, _ = cc_acc(s3)
            a0, _, _ = cc_acc(s0)
            out.append(f"| {m} | {rd} | {racc - nacc:+.2f} | {a3 - a0:+.2f} |")
    out.append("")

    return "\n".join(out), summary


def write_tables(grid_dir: Path) -> tuple[Path, Path]:
    """Aggregate ``grid_dir`` and write stage1_tables.md + stage1_summary.json."""
    md, summary = aggregate(load(grid_dir))
    md_path = grid_dir / "stage1_tables.md"
    js_path = grid_dir / "stage1_summary.json"
    md_path.write_text(md)
    js_path.write_text(json.dumps(summary, indent=2))
    print(md)
    print(f"\n-> {md_path}")
    return md_path, js_path
