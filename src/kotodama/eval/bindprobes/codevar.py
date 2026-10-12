"""``codevar`` — variable binding in (Python) code.

Three variables get string literals, then a straight-line program of copies,
reassignments and swaps-through-a-temp runs; the queried variable's value is
read off a ``print`` line whose output comment is the completion point::

    a = "cup"
    b = "lamp"
    c = "bell"
    d = a
    a = "card"
    print(d)  # output:
    -> " cup"

Python value semantics (``d = a`` copies the CURRENT value; rebinding ``a``
later does not change ``d``) — the simulator executes exactly that.

Cell = (depth, n_ops): ``depth`` ∈ {0,1,2,3} = the number of copy hops between
the literal and the queried variable's FINAL value (its provenance path;
a swap's temp counts as hops — depth is measured, not assumed, and items are
rejection-sampled to it); ``n_ops`` ∈ {1,2,4} = interfering ops besides the
alias chain (reassign ``x = "new"``, copy ``x = y``, swap ``t = x; x = y; y = t``).
At least one interfering op reads or writes a variable on the queried value's
provenance path (or the queried variable), and every item sets at least one
trap: a STALE value (the queried var or a path var held something else before
the value read) or a REFERENCE value (a path source rebound after it was read —
``d = a; a = "card"``). Items whose gold is the last literal written are
half-rejected (``gold_is_last_literal`` stays a recorded control; it is forced
in cell (0, 1), whose only op must rebind the queried variable).

Variable slots: 3 base + ``depth`` chain + 1 temp, a fixed count per cell, so
the ``3shot_collide`` map is a positional slot -> slot bijection.

Scored form: the object after ``# output:`` WITH its leading space (" cup") —
see ``pools`` for why the REPL/quote forms were rejected.

Candidates = every literal bound in the program (all distinct), in program order.
Error types (priority order): ``stale`` (a value the queried variable, or a
variable on its provenance path, held BEFORE the value that was read),
``alias`` (another variable's final value — the reference-semantics error),
``init`` (an initial literal), ``other``.
"""

from __future__ import annotations

import random
from itertools import pairwise
from typing import Mapping, Optional, TypedDict

from kotodama.eval.bindprobes.base import Cell, Item, rejection
from kotodama.eval.bindprobes.pools import Pools

FAMILY = "codevar"
DEPTHS = (0, 1, 2, 3)
N_OPS = (1, 2, 4)
CELLS: list[Cell] = [(d, m) for d in DEPTHS for m in N_OPS]
DEFAULT_PER_CELL = 100            # 12 cells -> 1200 items
SEPARATOR = "\n\n# ---\n\n"
N_BASE = 3
ERROR_TYPES = ("stale", "alias", "init", "other")
INTERFERE = ("reassign", "copy", "swap")


class Stmt(TypedDict, total=False):
    op: str        # "lit" | "copy"
    dst: str
    val: str       # lit
    src: str       # copy
    kind: str      # "decl" | "chain" | "reassign" | "copy" | "swap"
    group: int     # logical-op index (a swap = 3 statements, one group)


class Spec(TypedDict):
    vars: list[str]         # slots: base (3), chain (depth), temp (1)
    program: list[Stmt]
    query: str


class Trace(TypedDict):
    env: dict[str, str]
    history: dict[str, list[tuple[int, str]]]   # var -> [(stmt index, value)]
    path: dict[str, list[tuple[str, int]]]      # var -> provenance [(var, assign idx)], literal first


def execute(spec: Spec) -> Trace:
    """Run the program with Python value semantics; raise on a read of an unbound var."""
    env: dict[str, str] = {}
    history: dict[str, list[tuple[int, str]]] = {}
    path: dict[str, list[tuple[str, int]]] = {}
    for t, s in enumerate(spec["program"]):
        dst = s["dst"]
        if s["op"] == "lit":
            env[dst] = s["val"]
            path[dst] = [(dst, t)]
        elif s["op"] == "copy":
            src = s["src"]
            if src not in env:
                raise ValueError(f"statement {t}: read of unbound variable {src!r}")
            env[dst] = env[src]
            path[dst] = path[src] + [(dst, t)]
        else:
            raise ValueError(f"statement {t}: unknown op {s['op']!r}")
        history.setdefault(dst, []).append((t, env[dst]))
    return {"env": env, "history": history, "path": path}


def simulate(item: Item) -> str:
    spec: Spec = item.spec  # type: ignore[assignment]
    env = execute(spec)["env"]
    if spec["query"] not in env:
        raise ValueError(f"{item.item_id}: query {spec['query']!r} is never bound")
    return env[spec["query"]]


def depth_of(spec: Spec, var: str) -> int:
    return len(execute(spec)["path"][var]) - 1


def _line(s: Stmt) -> str:
    return f'{s["dst"]} = "{s["val"]}"' if s["op"] == "lit" else f'{s["dst"]} = {s["src"]}'


def render(item: Item, with_answer: bool) -> str:
    spec: Spec = item.spec  # type: ignore[assignment]
    lines = [_line(s) for s in spec["program"]] + [f"print({spec['query']})  # output:"]
    text = "\n".join(lines)
    return f"{text} {item.gold}" if with_answer else text


def labels(item: Item) -> list[str]:
    return list(item.spec["vars"])


def _literals(spec: Spec) -> list[str]:
    return [s["val"] for s in spec["program"] if s["op"] == "lit"]


def build(item_id: str, cell: Cell, spec: Spec) -> Item:
    lits = _literals(spec)
    if len(set(lits)) != len(lits):
        raise ValueError(f"{item_id}: literals must be distinct")
    tr = execute(spec)
    q = spec["query"]
    if q not in tr["env"]:
        raise ValueError(f"{item_id}: query {q!r} is never bound")
    it = Item(item_id=item_id, family=FAMILY, cell=list(cell), spec=dict(spec), gold="",
              candidates=list(lits), mention_order=list(lits),
              meta={"depth": len(tr["path"][q]) - 1,
                    "n_statements": len(spec["program"]),
                    "gold_is_last_literal": False})
    it.gold = simulate(it)
    it.meta["gold_is_last_literal"] = lits[-1] == it.gold
    it.meta["n_stale"] = len(stale_values(spec))
    it.meta["n_reference"] = len(reference_values(spec))
    return it


def _interferes_with_path(spec: Spec) -> bool:
    """Some interfering statement reads/writes the queried var or its provenance path."""
    tr = execute(spec)
    q = spec["query"]
    on_path = {v for v, _ in tr["path"][q]} | {q}
    for s in spec["program"]:
        if s["kind"] in INTERFERE and (s["dst"] in on_path or s.get("src") in on_path):
            return True
    return False


def gen_item(rng: random.Random, cell: Cell, pools: Pools, item_id: str, k: int,
             per_cell: int, template: Optional[Item] = None) -> Item:
    depth, n_ops = int(cell[0]), int(cell[1])
    if (depth, n_ops) not in CELLS:
        raise ValueError(f"codevar: unknown cell {cell}")
    n_slots = N_BASE + depth + 1

    def draw() -> Optional[Item]:
        vs = rng.sample(pools.variables, n_slots)
        base, chain, temp = vs[:N_BASE], vs[N_BASE:N_BASE + depth], vs[-1]
        lits = rng.sample(pools.objects, N_BASE + n_ops)
        decl_lits, spare = lits[:N_BASE], lits[N_BASE:]
        prog: list[Stmt] = [{"op": "lit", "dst": v, "val": o, "kind": "decl", "group": g}
                            for g, (v, o) in enumerate(zip(base, decl_lits))]
        plan = ["chain"] * depth + [rng.choice(INTERFERE) for _ in range(n_ops)]
        rng.shuffle(plan)
        defined = list(base)
        chain_src = rng.choice(base)
        ci = 0
        for g, kind in enumerate(plan, start=N_BASE):
            if kind == "chain":
                src = chain_src if ci == 0 else chain[ci - 1]
                prog.append({"op": "copy", "dst": chain[ci], "src": src, "kind": "chain",
                             "group": g})
                defined.append(chain[ci])
                ci += 1
            elif kind == "reassign":
                prog.append({"op": "lit", "dst": rng.choice(defined), "val": spare.pop(),
                             "kind": "reassign", "group": g})
            elif kind == "copy":
                if len(defined) < 2:
                    return None
                x, y = rng.sample(defined, 2)
                prog.append({"op": "copy", "dst": x, "src": y, "kind": "copy", "group": g})
            else:  # swap via temp
                if len(defined) < 2:
                    return None
                x, y = rng.sample(defined, 2)
                prog += [{"op": "copy", "dst": temp, "src": x, "kind": "swap", "group": g},
                         {"op": "copy", "dst": x, "src": y, "kind": "swap", "group": g},
                         {"op": "copy", "dst": y, "src": temp, "kind": "swap", "group": g}]
        # unused spare literals: the plan drew fewer reassigns than n_ops
        spec: Spec = {"vars": vs, "program": prog, "query": ""}
        tr = execute(spec)
        queryable = [v for v in defined if len(tr["path"][v]) - 1 == depth]
        if depth > 0 and chain[-1] in queryable:
            queryable = [chain[-1]] if rng.random() < 0.5 else queryable
        rng.shuffle(queryable)
        for q in queryable:
            spec["query"] = q
            if not _interferes_with_path(spec):
                continue
            if not (stale_values(spec) or reference_values(spec)):
                continue  # the ops touch the path but set no trap
            it = build(item_id, cell, spec)
            # soften the recency shortcut (gold = the last literal written): keep half
            if it.meta["gold_is_last_literal"] and rng.random() < 0.5:
                return None
            return it
        return None

    return rejection(draw, f"codevar item {item_id}", tries=5000)


def relabel(item: Item, mapping: Mapping[str, str], target: Optional[Item] = None) -> Item:
    spec: Spec = item.spec  # type: ignore[assignment]
    missing = set(spec["vars"]) - set(mapping)
    if missing:
        raise ValueError(f"{item.item_id}: variables {sorted(missing)} not in the map")
    prog: list[Stmt] = []
    for s in spec["program"]:
        n: Stmt = {**s, "dst": mapping[s["dst"]]}  # type: ignore[typeddict-item]
        if "src" in s:
            n["src"] = mapping[s["src"]]
        prog.append(n)
    new: Spec = {"vars": [mapping[v] for v in spec["vars"]], "program": prog,
                 "query": mapping[spec["query"]]}
    out = build(f"{item.item_id}_collide", tuple(item.cell), new)
    if out.gold != item.gold:
        raise ValueError(f"{item.item_id}: re-label changed the gold")
    return out


def stale_values(spec: Spec) -> set[str]:
    """Values the queried var — and every var on its provenance path — held BEFORE
    the value that flowed into the query (each var's history up to the read)."""
    tr = execute(spec)
    gold = tr["env"][spec["query"]]
    out: set[str] = set()
    # the queried var: everything it held before its final assignment
    q_hist = tr["history"][spec["query"]]
    out |= {v for _, v in q_hist[:-1]}
    # each hop (var, assign idx): what that var held before that assignment
    for var, t in tr["path"][spec["query"]]:
        out |= {v for t2, v in tr["history"][var] if t2 < t}
    out.discard(gold)
    return out


def reference_values(spec: Spec) -> set[str]:
    """Values a provenance SOURCE was rebound to AFTER it was read (``d = a`` then
    ``a = "card"``: ``card``) — what a reference-semantics reader would answer."""
    tr = execute(spec)
    gold = tr["env"][spec["query"]]
    path = tr["path"][spec["query"]]
    out: set[str] = set()
    for (src, _), (_, t_read) in pairwise(path):
        out |= {v for t2, v in tr["history"][src] if t2 > t_read}
    out.discard(gold)
    return out


def error_types(item: Item, pick: str) -> list[str]:
    spec: Spec = item.spec  # type: ignore[assignment]
    tr = execute(spec)
    q = spec["query"]
    t: list[str] = []
    if pick in stale_values(spec):
        t.append("stale")
    if any(v == pick for var, v in tr["env"].items() if var != q):
        t.append("alias")
    if pick in [s["val"] for s in spec["program"] if s["kind"] == "decl"]:
        t.append("init")
    return t or ["other"]

