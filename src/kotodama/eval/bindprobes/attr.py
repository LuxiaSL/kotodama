"""``attr`` — static attribute binding (binding-ID style).

N named people are each bound to one object ("Lucy keeps the lamp."), with 1–2
UNBOUND objects mentioned in neutral sentences ("A spoon lies on the shelf.")
interleaved; nothing moves. One entity is queried at the end::

    Lucy keeps the lamp. A spoon lies on the shelf. Tom keeps the cup. So Lucy keeps the
    -> " lamp"

Cell = (N, qpos): N ∈ {2,4,6,8}; qpos = which declared entity is queried
(first / middle / last; N=2 has no middle -> 11 cells). Recency control (only
meaningful at qpos=last; balanced 50/50 there): ``recency_control`` = the gold
IS the last-mentioned object (no unbound sentence after its declaration).
At qpos first/middle the gold is never the last mention.

Candidates = the N bound objects + the unbound ones, in mention order. Every slot
draws from the count nouns of the pool (``pools.countable``), so the unbound
sentence "A/An o lies on the shelf." is grammatical and no slot is told apart by
the kind of noun in it.
Error types (priority order): ``wrong_binding`` (another entity's object; an
``adjacent`` tag is added when that entity was declared next to the queried one),
``distractor`` (an unbound object), ``other``.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional, TypedDict

from kotodama.eval.bindprobes.base import Cell, Item, rejection
from kotodama.eval.bindprobes.pools import Pools, countable, indefinite

FAMILY = "attr"
VERSION = 2                       # meta.attr_version; items' id/seed tag "attr-v2"
NS = (2, 4, 6, 8)
QPOS = ("first", "middle", "last")
CELLS: list[Cell] = [(n, q) for n in NS for q in QPOS if not (n == 2 and q == "middle")]
DEFAULT_PER_CELL = 110            # 11 cells -> 1210 items
SEPARATOR = "\n\n* * *\n\n"       # prose scene break between episodes
VERBS = ("keeps", "owns", "carries", "has")
ERROR_TYPES = ("wrong_binding", "distractor", "other")


class Spec(TypedDict):
    verb: str
    entities: list[list[str]]    # [[name, obj], ...] in declaration order
    distractors: list[str]       # unbound objects
    order: list[list[Any]]       # sentence order: ["bind", i] | ["free", j]
    query: str                   # queried name


def simulate(item: Item) -> str:
    """The queried entity's object (the binding table is the whole state)."""
    spec: Spec = item.spec  # type: ignore[assignment]
    table = {name: obj for name, obj in spec["entities"]}
    if len(table) != len(spec["entities"]):
        raise ValueError(f"{item.item_id}: duplicate entity names")
    if spec["query"] not in table:
        raise ValueError(f"{item.item_id}: query {spec['query']!r} is not an entity")
    return table[spec["query"]]


def _sentence(spec: Spec, kind: str, idx: int) -> str:
    if kind == "bind":
        name, obj = spec["entities"][idx]
        return f"{name} {spec['verb']} the {obj}."
    obj = spec["distractors"][idx]
    return f"{indefinite(obj).capitalize()} {obj} lies on the shelf."


def _mentions(spec: Spec) -> list[str]:
    return [spec["entities"][i][1] if k == "bind" else spec["distractors"][i]
            for k, i in spec["order"]]


def render(item: Item, with_answer: bool) -> str:
    spec: Spec = item.spec  # type: ignore[assignment]
    body = " ".join(_sentence(spec, k, i) for k, i in spec["order"])
    lead = f"{body} So {spec['query']} {spec['verb']} the"
    return f"{lead} {item.gold}." if with_answer else lead


def labels(item: Item) -> list[str]:
    return [name for name, _ in item.spec["entities"]]


def build(item_id: str, cell: Cell, spec: Spec) -> Item:
    """Item from a spec: gold from the simulator, candidates = mention order."""
    mentions = _mentions(spec)
    if sorted(i for k, i in spec["order"] if k == "bind") != list(range(len(spec["entities"]))):
        raise ValueError(f"{item_id}: every entity must be declared exactly once")
    names = [n for n, _ in spec["entities"]]
    qidx = names.index(spec["query"])
    it = Item(item_id=item_id, family=FAMILY, cell=list(cell), spec=dict(spec), gold="",
              candidates=list(dict.fromkeys(mentions)), mention_order=mentions,
              meta={"attr_version": VERSION, "qpos": cell[1], "query_index": qidx,
                    "recency_control": False,
                    "n_distractors": len(spec["distractors"])})
    it.gold = simulate(it)
    it.meta["recency_control"] = mentions[-1] == it.gold
    if len(it.candidates) != len(mentions):
        raise ValueError(f"{item_id}: an object is mentioned twice")
    return it


def gen_item(rng: random.Random, cell: Cell, pools: Pools, item_id: str, k: int,
             per_cell: int, template: Optional[Item] = None) -> Item:
    n, qpos = int(cell[0]), str(cell[1])
    if (n, qpos) not in CELLS:
        raise ValueError(f"attr: unknown cell {cell}")
    # recency control: balanced inside qpos=last cells (alternating by k); random for shots
    want_rc = qpos == "last" and (k % 2 == 0 if template is None else rng.random() < 0.5)

    def draw() -> Optional[Item]:
        names = rng.sample(pools.names, n)
        nd = rng.choice((1, 2))
        objs = rng.sample(countable(pools.objects), n + nd)
        bound, free = objs[:n], objs[n:]
        qidx = {"first": 0, "last": n - 1}.get(qpos)
        if qidx is None:
            qidx = rng.randint(1, n - 2)
        # distractor slot s in [0, n]: the sentence goes before bound sentence s (s=n: at the end)
        if qpos == "last":
            slots = [rng.randint(0, n - 1) for _ in free] if want_rc else \
                [n] + [rng.randint(0, n) for _ in free[1:]]
        else:
            slots = [rng.randint(0, n) for _ in free]
        order: list[list[Any]] = []
        for s in range(n + 1):
            order += [["free", j] for j, sj in enumerate(slots) if sj == s]
            if s < n:
                order.append(["bind", s])
        spec: Spec = {"verb": rng.choice(VERBS),
                      "entities": [[a, b] for a, b in zip(names, bound)],
                      "distractors": free, "order": order, "query": names[qidx]}
        it = build(item_id, cell, spec)
        if it.meta["recency_control"] != want_rc:
            return None
        return it

    return rejection(draw, f"attr item {item_id}")


def relabel(item: Item, mapping: Mapping[str, str], target: Optional[Item] = None) -> Item:
    spec: Spec = item.spec  # type: ignore[assignment]
    missing = set(labels(item)) - set(mapping)
    if missing:
        raise ValueError(f"{item.item_id}: names {sorted(missing)} not in the map")
    new: Spec = {**spec, "entities": [[mapping[n], o] for n, o in spec["entities"]],
                 "query": mapping[spec["query"]]}
    out = build(f"{item.item_id}_collide", tuple(item.cell), new)
    if out.gold != item.gold:
        raise ValueError(f"{item.item_id}: re-label changed the gold")
    return out


def error_types(item: Item, pick: str) -> list[str]:
    spec: Spec = item.spec  # type: ignore[assignment]
    names = [n for n, _ in spec["entities"]]
    q = names.index(spec["query"])
    for i, (_, obj) in enumerate(spec["entities"]):
        if obj == pick and i != q:
            return ["wrong_binding"] + (["adjacent"] if abs(i - q) == 1 else [])
    if pick in spec["distractors"]:
        return ["distractor"]
    return ["other"]
