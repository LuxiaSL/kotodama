"""bindprobes-lite: a small balanced slice for (later) in-loop scoring.

NOT wired into training. Mirrors ``riders.lite``: :func:`select_items` (per
family, round-robin over cells, seeded), :func:`build_jobs` (one ``/logprobs``-
shaped request per item × condition — the same payload the HTTP runner sends),
:func:`summarize` (keys ``bindprobes/{family}/{condition}/{metric}`` plus
``collision_gap`` and ``shot_gap`` per family, in acc).

To score in-process: ``riders.score.score_logprob_items(logits_fn, tok, jobs, ...)``
then ``grid.score_row(items[job.item_index], result, job.condition, model)``.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from kotodama.eval.bindprobes.aggregate import cell_metrics
from kotodama.eval.bindprobes.base import Item, cell_key
from kotodama.eval.bindprobes.grid import build_payload_item
from kotodama.eval.bindprobes.items import CONDITIONS, COND_NSHOTS, FAMILY_NAMES, family

DEFAULT_PER_FAMILY = 200
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Job:
    id: str
    prompt: str
    candidates: list[str]
    item_index: int       # index into the selected item list
    family: str
    condition: str
    nshots: int


def select_items(items: Sequence[Item], per_family: int = DEFAULT_PER_FAMILY,
                 seed: int = DEFAULT_SEED,
                 families: Sequence[str] = FAMILY_NAMES) -> list[Item]:
    """``per_family`` items per family, round-robin over that family's cells (each
    cell shuffled with ``seed``); exhausted cells drop out. Deterministic."""
    if per_family < 1:
        raise ValueError(f"per_family must be >= 1, got {per_family}")
    out: list[Item] = []
    for f in families:
        rng = random.Random(f"{seed}:{f}")
        pools: list[list[Item]] = []
        for c in family(f).CELLS:
            pool = [it for it in items if it.family == f and it.cell_key == cell_key(c)]
            rng.shuffle(pool)
            if pool:
                pools.append(pool)
        picked: list[Item] = []
        k = 0
        while len(picked) < per_family and any(k < len(p) for p in pools):
            for p in pools:
                if k < len(p) and len(picked) < per_family:
                    picked.append(p[k])
            k += 1
        out += picked
    return out


def build_jobs(items: Sequence[Item], conditions: Sequence[str] = CONDITIONS) -> list[Job]:
    jobs: list[Job] = []
    for cond in conditions:
        if cond not in COND_NSHOTS:
            raise ValueError(f"unknown condition {cond!r}; choose from {CONDITIONS}")
        for k, it in enumerate(items):
            p = build_payload_item(it, cond)
            jobs.append(Job(id=p["id"], prompt=p["prompt"], candidates=p["candidates"],
                            item_index=k, family=it.family, condition=cond,
                            nshots=COND_NSHOTS[cond]))
    return jobs


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    by: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["family"], r["condition"])].append(r)
    out: dict[str, float] = {}
    for f in sorted({f for f, _ in by}):
        acc: dict[str, float] = {}
        for cond in CONDITIONS:
            sub = by.get((f, cond))
            if not sub:
                continue
            m = cell_metrics(sub, family(f).ERROR_TYPES)
            acc[cond] = m["acc"]
            for k, v in m.items():
                out[f"bindprobes/{f}/{cond}/{k}"] = v
        if "3shot" in acc and "3shot_collide" in acc:
            out[f"bindprobes/{f}/collision_gap"] = acc["3shot"] - acc["3shot_collide"]
        if "3shot" in acc and "0shot" in acc:
            out[f"bindprobes/{f}/shot_gap"] = acc["3shot"] - acc["0shot"]
    return out
