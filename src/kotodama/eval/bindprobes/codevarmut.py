"""``codevarmut`` (battery v3) — alias-GRAPH binding: mutable lists in Python.

codevar is value semantics only (rebinding strings). Here names share list OBJECTS:
a mutation through one name is visible through every alias; rebinding a name or
copying with ``list(x)`` breaks the sharing::

    # python
    a = ["cup"]
    b = ["lamp"]
    c = ["bell"]
    d = a
    a.append("key")
    print(d[-1])  # output:
    -> " key"

Cell = (depth, trap): depth ∈ {1, 2} alias hops from a base list to the queried name;
trap (the op that decides the answer, placed after the hop it tests):
  mutate  append through ANOTHER name of the queried object  -> gold = the appended value
          (value-semantics misread answers the old last element)
  rebind  the hop's source is rebound to a new list (and may be appended to after)
          -> gold = the old object's last element (reference misread answers the new one)
  copy    one hop is ``y = list(x)``, then ``x`` is appended to -> gold = the copy's last
          element (alias misread answers the appended value)
Plus one interfering op on an unrelated list (append or rebind). Exactly 5 distinct
literals per item (3 base + 2), so chance = 1/5. ``gold_is_last_literal`` is scheduled
at 1/5 in mutate cells (cycled by item index); in rebind/copy cells it is structurally 0
(the gold is a declared literal), so a last-literal guesser is below chance pooled. A
pure value-semantics reader is right on rebind/copy and wrong on mutate; a pure
reference reader the reverse — only alias-graph tracking clears all three (report per
trap AND the trap-balanced mean). ``# python`` pins the language.

Error types: ``value_semantics`` (the answer under copy-on-alias), ``reference_semantics``
(the answer under alias-everything / rebind-in-place), ``stale`` (an earlier last
element of the queried object), ``other``.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional, Sequence, TypedDict

from kotodama.eval.bindprobes.base import Cell, Item, rejection

FAMILY = "codevarmut"
VERSION = 1
DEPTHS = (1, 2)
TRAPS = ("mutate", "rebind", "copy")
CELLS: list[Cell] = [(d, t) for d in DEPTHS for t in TRAPS]
DEFAULT_PER_CELL = 100            # 6 cells -> 600 items (5 | 100: exact last-literal schedule)
SEPARATOR = "\n\n# ---\n\n"
HEADER = "# python"
N_BASE = 3
N_LITERALS = 5
ERROR_TYPES = ("value_semantics", "reference_semantics", "stale", "other")
BALANCE_TOL = 0.05


class Stmt(TypedDict, total=False):
    op: str        # "lit" (x = ["v"]) | "alias" (x = y) | "copy" (x = list(y)) | "append" (x.append("v"))
    dst: str
    src: str
    val: str
    role: str      # "decl" | "hop" | "trap" | "noise"


class Spec(TypedDict):
    vars: list[str]
    program: list[Stmt]
    query: str


# ── Simulation ─────────────────────────────────────────────────────────────────

def execute(spec: Spec, mode: str = "python") -> tuple[dict[str, int], dict[int, list[str]]]:
    """Run the program. ``mode``: ``python`` (true semantics), ``value`` (every alias
    copies), ``reference`` (every copy aliases; rebinding an existing name replaces its
    object's contents in place). Returns (name -> object id, object id -> list)."""
    if mode not in ("python", "value", "reference"):
        raise ValueError(f"unknown mode {mode!r}")
    env: dict[str, int] = {}
    objs: dict[int, list[str]] = {}
    nxt = 0
    for t, s in enumerate(spec["program"]):
        op = s["op"]
        if op == "lit":
            if mode == "reference" and s["dst"] in env:
                objs[env[s["dst"]]][:] = [s["val"]]
            else:
                objs[nxt] = [s["val"]]
                env[s["dst"]] = nxt
                nxt += 1
        elif op in ("alias", "copy"):
            if s["src"] not in env:
                raise ValueError(f"statement {t}: read of unbound {s['src']!r}")
            share = (op == "alias") if mode == "python" else (mode == "reference")
            if share:
                env[s["dst"]] = env[s["src"]]
            else:
                objs[nxt] = list(objs[env[s["src"]]])
                env[s["dst"]] = nxt
                nxt += 1
        elif op == "append":
            if s["dst"] not in env:
                raise ValueError(f"statement {t}: append to unbound {s['dst']!r}")
            objs[env[s["dst"]]].append(s["val"])
        else:
            raise ValueError(f"statement {t}: unknown op {op!r}")
    return env, objs


def answer(spec: Spec, mode: str = "python") -> str:
    env, objs = execute(spec, mode)
    if spec["query"] not in env:
        raise ValueError(f"query {spec['query']!r} never bound")
    return objs[env[spec["query"]]][-1]


def simulate(item: Item) -> str:
    return answer(item.spec)  # type: ignore[arg-type]


def stale_values(spec: Spec) -> set[str]:
    """Earlier last elements of the queried object (its history before the final state)."""
    env_f, _ = execute(spec)
    q = spec["query"]
    out: set[str] = set()
    sub: Spec = {"vars": spec["vars"], "program": [], "query": q}
    for s in spec["program"]:
        sub["program"].append(s)
        env, objs = execute(sub)
        if q in env:
            out.add(objs[env[q]][-1])
    out.discard(answer(spec))
    return out


# ── Rendering ──────────────────────────────────────────────────────────────────

def _line(s: Stmt) -> str:
    op = s["op"]
    if op == "lit":
        return f'{s["dst"]} = ["{s["val"]}"]'
    if op == "alias":
        return f'{s["dst"]} = {s["src"]}'
    if op == "copy":
        return f'{s["dst"]} = list({s["src"]})'
    return f'{s["dst"]}.append("{s["val"]}")'


def render(item: Item, with_answer: bool) -> str:
    spec: Spec = item.spec  # type: ignore[assignment]
    lines = [HEADER] + [_line(s) for s in spec["program"]] + [f"print({spec['query']}[-1])  # output:"]
    text = "\n".join(lines)
    return f"{text} {item.gold}" if with_answer else text


def labels(item: Item) -> list[str]:
    return list(item.spec["vars"])


def _literals(spec: Spec) -> list[str]:
    return [s["val"] for s in spec["program"] if s["op"] in ("lit", "append")]


def build(item_id: str, cell: Cell, spec: Spec) -> Item:
    lits = _literals(spec)
    if len(set(lits)) != len(lits) or len(lits) != N_LITERALS:
        raise ValueError(f"{item_id}: need {N_LITERALS} distinct literals, got {lits}")
    it = Item(item_id=item_id, family=FAMILY, cell=list(cell), spec=dict(spec), gold="",
              candidates=list(lits), mention_order=list(lits),
              meta={"codevarmut_version": VERSION, "trap": cell[1], "depth": cell[0]})
    it.gold = simulate(it)
    it.meta.update({"gold_is_last_literal": lits[-1] == it.gold,
                    "value_answer": answer(spec, "value"),
                    "reference_answer": answer(spec, "reference")})
    return it


def _draw(rng: random.Random, cell: Cell, pools: Any, item_id: str,
          want_last: Optional[bool]) -> Optional[Item]:
    depth, trap = int(cell[0]), str(cell[1])
    vs = rng.sample(pools.variables, N_BASE + depth)
    base, chain = vs[:N_BASE], vs[N_BASE:]
    lits = rng.sample(pools.objects, N_LITERALS)
    prog: list[Stmt] = [{"op": "lit", "dst": v, "val": o, "role": "decl"}
                        for v, o in zip(base, lits[:N_BASE])]
    spare = lits[N_BASE:]
    src0 = rng.choice(base)
    copy_hop = rng.randrange(depth) if trap == "copy" else -1
    hops: list[Stmt] = []
    for i, dst in enumerate(chain):
        src = src0 if i == 0 else chain[i - 1]
        hops.append({"op": "copy" if i == copy_hop else "alias", "dst": dst, "src": src, "role": "hop"})
    q = chain[-1]
    # the trap, placed after the hop it tests
    if trap == "mutate":
        group = [src0] + chain[:-1]                 # other names of the queried object
        trap_ops: list[Stmt] = [{"op": "append", "dst": rng.choice(group), "val": spare[0], "role": "trap"}]
        after = depth                               # after every hop
    elif trap == "rebind":
        tgt_i = rng.randrange(depth)                # rebind the source of hop tgt_i
        tgt = src0 if tgt_i == 0 else chain[tgt_i - 1]
        trap_ops = [{"op": "lit", "dst": tgt, "val": spare[0], "role": "trap"}]
        after = tgt_i + 1
    else:
        tgt = src0 if copy_hop == 0 else chain[copy_hop - 1]
        trap_ops = [{"op": "append", "dst": tgt, "val": spare[0], "role": "trap"}]
        after = copy_hop + 1
    others = [b for b in base if b != src0]
    noise: Stmt = rng.choice([{"op": "append", "dst": rng.choice(others), "val": spare[1], "role": "noise"},
                              {"op": "lit", "dst": rng.choice(others), "val": spare[1], "role": "noise"}])
    body = hops[:after] + trap_ops + hops[after:]
    pos = rng.randint(0, len(body))
    body.insert(pos, noise)
    spec: Spec = {"vars": vs, "program": prog + body, "query": q}
    try:
        it = build(item_id, cell, spec)
    except ValueError:
        return None
    # the trap must bite: the misread under test differs from the truth
    misread = it.meta["value_answer"] if trap == "mutate" else it.meta["reference_answer"]
    if misread == it.gold:
        return None
    if want_last is not None and it.meta["gold_is_last_literal"] != want_last:
        return None
    return it


def gen_item(rng: random.Random, cell: Cell, pools: Any, item_id: str, k: int,
             per_cell: int, template: Optional[Item] = None) -> Item:
    if tuple(cell) not in CELLS:
        raise ValueError(f"codevarmut: unknown cell {cell}")
    cell = tuple(cell)
    # rebind/copy gold is always a declared literal (never the last one written), so the
    # last-literal control is scheduled (at 1/5) in mutate cells only
    want = (k % N_LITERALS == 0) if template is None and cell[1] == "mutate" else None
    return rejection(lambda: _draw(rng, cell, pools, item_id, want), f"codevarmut item {item_id}",
                     tries=20000)


def relabel(item: Item, mapping: Mapping[str, str], target: Optional[Item] = None) -> Item:
    spec: Spec = item.spec  # type: ignore[assignment]
    if set(spec["vars"]) - set(mapping):
        raise ValueError(f"{item.item_id}: variables missing from the map")
    prog = []
    for s in spec["program"]:
        n: Stmt = {**s, "dst": mapping[s["dst"]]}  # type: ignore[typeddict-item]
        if "src" in s:
            n["src"] = mapping[s["src"]]
        prog.append(n)
    out = build(f"{item.item_id}_collide", tuple(item.cell),
                {"vars": [mapping[v] for v in spec["vars"]], "program": prog,
                 "query": mapping[spec["query"]]})
    if out.gold != item.gold:
        raise ValueError(f"{item.item_id}: re-label changed the gold")
    return out


def validate_item(item: Item) -> None:
    if item.meta.get("codevarmut_version") != VERSION:
        raise ValueError(f"{item.item_id}: codevarmut_version mismatch")
    trap = item.cell[1]
    misread = item.meta["value_answer"] if trap == "mutate" else item.meta["reference_answer"]
    if misread == item.gold:
        raise ValueError(f"{item.item_id}: the {trap} trap does not bite")
    if not render(item, False).startswith(HEADER + "\n"):
        raise ValueError(f"{item.item_id}: missing language pin")


def audit(items: Sequence[Item]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for c in CELLS:
        sub = [it for it in items if it.family == FAMILY and tuple(it.cell) == c]
        if sub:
            out["/".join(map(str, c))] = {
                "n": len(sub), "chance": 1 / N_LITERALS if c[1] == "mutate" else 0.0,
                "gold_is_last_literal": sum(it.meta["gold_is_last_literal"] for it in sub) / len(sub)}
    return out


def audit_failures(aud: Mapping[str, Mapping[str, Any]], tol: float = BALANCE_TOL) -> list[str]:
    return [f"{ck}: last-literal {a['gold_is_last_literal']:.3f} vs {a['chance']:.3f}"
            for ck, a in aud.items()
            if abs(a["gold_is_last_literal"] - a["chance"]) > max(tol, 1.0 / a["n"]) + 1e-9]


def error_types(item: Item, pick: str) -> list[str]:
    spec: Spec = item.spec  # type: ignore[assignment]
    t: list[str] = []
    if pick == item.meta["value_answer"]:
        t.append("value_semantics")
    if pick == item.meta["reference_answer"]:
        t.append("reference_semantics")
    if pick in stale_values(spec):
        t.append("stale")
    return t or ["other"]
