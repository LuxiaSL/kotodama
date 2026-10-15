"""bindprobes-lite: a small balanced slice for (later) in-loop scoring.

NOT wired into training. Mirrors ``riders.lite``: :func:`select_items` (per
family, round-robin over cells, seeded), :func:`build_jobs` (one ``/logprobs``-
shaped request per item × condition — the same payload the HTTP runner sends),
:func:`summarize` (keys ``bindprobes/{family}/{condition}/{metric}`` plus
``collision_gap`` and ``shot_gap`` per family, in acc).

deixis is selected by PAIR (10 base items per cell by default, each with its
deictic + third (+ twin at P=3) variants); ``summarize`` reports its deictic rows as
``bindprobes/deixis/...``, the variants as ``bindprobes/deixis.third/...`` and
``bindprobes/deixis.twin/...``, plus ``bindprobes/deixis/penalty`` (3shot
cc(third) − cc(deictic)), ``twin_joint`` and ``twin_dependence`` (3shot).

To score in-process: ``riders.score.score_logprob_items(logits_fn, tok, jobs, ...)``
then ``grid.score_row(items[job.item_index], result, job.condition, model)``.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from kotodama.eval.bindprobes.aggregate import DEIXIS, cell_metrics, deixis_readouts
from kotodama.eval.bindprobes.base import Item, cell_key
from kotodama.eval.bindprobes.grid import build_payload_item
from kotodama.eval.bindprobes.items import CONDITIONS, COND_NSHOTS, CORE_FAMILY_NAMES, family

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


def default_per_family(name: str) -> int:
    """deixis: ``LITE_PER_CELL`` pairs per cell; every other family: 200 items."""
    fam = family(name)
    per_cell = getattr(fam, "LITE_PER_CELL", None)
    return per_cell * len(fam.CELLS) if per_cell else DEFAULT_PER_FAMILY


def select_items(items: Sequence[Item], per_family: Optional[int] = None,
                 seed: int = DEFAULT_SEED,
                 families: Sequence[str] = CORE_FAMILY_NAMES) -> list[Item]:
    """``per_family`` items per family (None: :func:`default_per_family`),
    round-robin over that family's cells (each cell shuffled with ``seed``);
    exhausted cells drop out. A paired family (deixis) counts PAIRS and returns
    every variant of each selected pair. Deterministic."""
    if per_family is not None and per_family < 1:
        raise ValueError(f"per_family must be >= 1, got {per_family}")
    out: list[Item] = []
    for f in families:
        n_want = per_family if per_family is not None else default_per_family(f)
        rng = random.Random(f"{seed}:{f}")
        paired = f == DEIXIS
        by_pair: dict[str, list[Item]] = {}
        if paired:
            for it in items:
                if it.family == f:
                    by_pair.setdefault(str(it.meta["pair_id"]), []).append(it)
        pools: list[list[Item]] = []
        for c in family(f).CELLS:
            pool = [it for it in items if it.family == f and it.cell_key == cell_key(c)
                    and (not paired or it.meta.get("variant") == "deictic")]
            rng.shuffle(pool)
            if pool:
                pools.append(pool)
        picked: list[Item] = []
        k = 0
        while len(picked) < n_want and any(k < len(p) for p in pools):
            for p in pools:
                if k < len(p) and len(picked) < n_want:
                    picked.append(p[k])
            k += 1
        if paired:
            order = {v: i for i, v in enumerate(("deictic", "third", "twin"))}
            picked = [v for it in picked for v in sorted(
                by_pair[str(it.meta["pair_id"])], key=lambda x: order[x.meta["variant"]])]
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


def _summary_family(r: Mapping[str, Any]) -> str:
    """deixis third/twin rows report as ``deixis.third`` / ``deixis.twin``."""
    v = r.get("variant")
    return f"{DEIXIS}.{v}" if r["family"] == DEIXIS and v not in (None, "deictic") else r["family"]


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    by: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(_summary_family(r), r["condition"])].append(r)
    out: dict[str, float] = {}
    for f in sorted({f for f, _ in by}):
        acc: dict[str, float] = {}
        for cond in CONDITIONS:
            sub = by.get((f, cond))
            if not sub:
                continue
            m = cell_metrics(sub, family(f.split(".")[0]).ERROR_TYPES)
            acc[cond] = m["acc"]
            for k, v in m.items():
                out[f"bindprobes/{f}/{cond}/{k}"] = v
        if "3shot" in acc and "3shot_collide" in acc:
            out[f"bindprobes/{f}/collision_gap"] = acc["3shot"] - acc["3shot_collide"]
        if "3shot" in acc and "0shot" in acc:
            out[f"bindprobes/{f}/shot_gap"] = acc["3shot"] - acc["0shot"]
    drows = [r for r in rows if r["family"] == DEIXIS]
    if drows and len({r["model"] for r in drows}) == 1:
        d = next(iter(deixis_readouts(drows).values()))
        out[f"bindprobes/{DEIXIS}/penalty"] = d["penalty"]["3shot"]["all"]
        out[f"bindprobes/{DEIXIS}/twin_joint"] = d["twin"]["3shot"]["all"]["joint"]
        out[f"bindprobes/{DEIXIS}/twin_dependence"] = d["twin"]["3shot"]["all"]["dependence"]
    return out
