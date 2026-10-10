"""riders-lite: a small, balanced slice of the rider grid for in-loop scoring.

The training loop (``kotodama.training.inloop``) scores this slice on the live
weights at every eval point. Pieces:

* :func:`select_items` — N std-tier items, seeded, round-robin over the (n, m)
  difficulty cells so cells stay balanced; the SAME items in every register
  (paired register comparisons).
* :func:`reletter_shots` — the exemplar shots with their box letters remapped
  (F->A, G->B, H->C by default) to letters no std item uses, re-validated with
  the item simulator. The ``3shot_relet`` condition asks whether 3-shot gains
  come from the worked procedure or from copying the exemplars' box letters.
* :func:`build_jobs` — one ``/logprobs``-shaped request per (item, register,
  condition).
* :func:`summarize` — accuracy, chance-corrected accuracy (``aggregate``'s
  formula), mean margin and coarse error-type shares per register x condition,
  plus the shot gaps; keys are ``riders/{register}/{condition}/{metric}``.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from kotodama.eval.riders.aggregate import cc_point
from kotodama.eval.riders.errors import COARSE_TYPES
from kotodama.eval.riders.grid import build_payload_item
from kotodama.eval.riders.items import CELLS, N_SHOTS, Item, Op, _sim

REGISTERS: tuple[str, ...] = ("narrative", "screenplay", "chat", "code")
PROSE: tuple[str, ...] = ("narrative", "screenplay", "chat")
CONDITIONS: tuple[str, ...] = ("0shot", "3shot", "3shot_relet")
COND_NSHOTS: dict[str, int] = {"0shot": 0, "3shot": N_SHOTS, "3shot_relet": N_SHOTS}
RELETTER_MAP: dict[str, str] = {"F": "A", "G": "B", "H": "C"}
DEFAULT_PER_REGISTER = 200
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Job:
    """One scoring request + where its row goes."""
    id: str
    prompt: str
    candidates: list[str]
    item_index: int       # index into the selected item list
    renderer: str
    condition: str
    nshots: int


# ── Selection ──────────────────────────────────────────────────────────────────

def select_items(items: Sequence[Item], n: int = DEFAULT_PER_REGISTER,
                 seed: int = DEFAULT_SEED,
                 cells: Sequence[tuple[int, int]] = CELLS) -> list[Item]:
    """``n`` std-tier items, round-robin over ``cells`` (each cell shuffled with
    ``seed``); cells that run out drop out of the rotation. Deterministic."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = random.Random(seed)
    pools: list[list[Item]] = []
    for cell in cells:
        pool = [it for it in items if it.tier == "std" and (it.n, it.m) == tuple(cell)]
        rng.shuffle(pool)
        if pool:
            pools.append(pool)
    if not pools:
        raise ValueError("no std-tier items in any cell")
    out: list[Item] = []
    k = 0
    while len(out) < n and any(k < len(p) for p in pools):
        for p in pools:
            if k < len(p) and len(out) < n:
                out.append(p[k])
        k += 1
    return out


# ── Re-lettered shots ──────────────────────────────────────────────────────────

def _check_sim(item: Item) -> None:
    final = _sim(item.boxes0, [Op(**o) for o in item.ops])
    if final != item.final:
        raise ValueError(f"{item.item_id}: simulated final {final} != stored {item.final}")
    if final.get(item.query_box) != item.gold:
        raise ValueError(f"{item.item_id}: final[{item.query_box}]="
                         f"{final.get(item.query_box)!r} != gold {item.gold!r}")


def reletter_item(item: Item, mapping: Mapping[str, str] = RELETTER_MAP) -> Item:
    """``item`` with box letters remapped (boxes0, ops i/j, final, query_box)."""
    used = set(item.boxes0) | set(item.final) | {item.query_box}
    used |= {op[k] for op in item.ops for k in ("i", "j") if op.get(k) is not None}
    missing = used - set(mapping)
    if missing:
        raise ValueError(f"{item.item_id}: letters {sorted(missing)} not in the remap")
    ops = []
    for op in item.ops:
        o = dict(op)
        o["i"] = mapping[o["i"]]
        if o.get("j") is not None:
            o["j"] = mapping[o["j"]]
        ops.append(o)
    new = Item(**{**asdict(item),
                  "item_id": f"{item.item_id}_relet",
                  "boxes0": {mapping[b]: v for b, v in item.boxes0.items()},
                  "ops": ops,
                  "final": {mapping[b]: v for b, v in item.final.items()},
                  "query_box": mapping[item.query_box]})
    _check_sim(new)
    return new


def reletter_shots(shots: Sequence[Item], items: Sequence[Item] = (),
                   mapping: Mapping[str, str] = RELETTER_MAP) -> list[Item]:
    """Re-lettered copies of ``shots``, each validated (final[query_box] == gold).

    Raises if a target letter is one any of ``items`` uses (the point of the
    condition is letters the queried items never show)."""
    item_letters = {b for it in items for b in it.boxes0}
    clash = item_letters & set(mapping.values())
    if clash:
        raise ValueError(f"re-letter targets {sorted(clash)} are used by the items")
    for s in shots:
        _check_sim(s)
    return [reletter_item(s, mapping) for s in shots]


# ── Jobs ───────────────────────────────────────────────────────────────────────

def build_jobs(items: Sequence[Item], shots: Sequence[Item],
               relet_shots: Optional[Sequence[Item]],
               registers: Sequence[str] = REGISTERS,
               conditions: Sequence[str] = CONDITIONS) -> list[Job]:
    """Every (register, condition, item) request; koto prompts (chat = ChatML)."""
    jobs: list[Job] = []
    for cond in conditions:
        if cond not in COND_NSHOTS:
            raise ValueError(f"unknown condition {cond!r}; choose from {CONDITIONS}")
        shot_set = relet_shots if cond == "3shot_relet" else shots
        if shot_set is None:
            raise ValueError(f"condition {cond!r} needs shots")
        nshots = COND_NSHOTS[cond]
        for rd in registers:
            for k, it in enumerate(items):
                payload, _ = build_payload_item(it, rd, nshots, list(shot_set), "koto")
                jobs.append(Job(id=f"{it.item_id}|{rd}|{cond}", prompt=payload["prompt"],
                                candidates=payload["candidates"], item_index=k,
                                renderer=rd, condition=cond, nshots=nshots))
    return jobs


# ── Summary ────────────────────────────────────────────────────────────────────

def _cell_metrics(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    acc = float(np.mean([r["gold_rank_cand"] == 1 for r in rows]))
    out = {"acc": acc, "cc_acc": cc_point(list(rows)),
           "margin": float(np.mean([r["margin"] for r in rows])), "n": float(len(rows))}
    wrong = [r for r in rows if r["gold_rank_cand"] != 1]
    for t in COARSE_TYPES:
        out[f"err_{t}"] = (sum(r.get("error_type") == t for r in wrong) / len(wrong)
                           if wrong else 0.0)
    return out


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """``riders/{register}/{condition}/{acc,cc_acc,margin,n,err_*}`` per register
    plus the ``prose`` pool (narrative+screenplay+chat); ``riders/{register}/shot_gap``
    = cc(3shot) - cc(0shot) and ``relet_gap`` = cc(3shot_relet) - cc(3shot)."""
    by: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["renderer"], r["condition"])].append(r)
        if r["renderer"] in PROSE:
            by[("prose", r["condition"])].append(r)
    out: dict[str, float] = {}
    regs = sorted({rd for rd, _ in by})
    for rd in regs:
        cc: dict[str, float] = {}
        for cond in CONDITIONS:
            sub = by.get((rd, cond))
            if not sub:
                continue
            m = _cell_metrics(list(sub))
            cc[cond] = m["cc_acc"]
            for k, v in m.items():
                out[f"riders/{rd}/{cond}/{k}"] = v
        if "3shot" in cc and "0shot" in cc:
            out[f"riders/{rd}/shot_gap"] = cc["3shot"] - cc["0shot"]
        if "3shot_relet" in cc and "3shot" in cc:
            out[f"riders/{rd}/relet_gap"] = cc["3shot_relet"] - cc["3shot"]
    return out
