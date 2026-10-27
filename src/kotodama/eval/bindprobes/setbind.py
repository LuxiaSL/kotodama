"""``setbind`` (battery v3) — many-to-one binding: several objects on one entity.

Four people hold objects; one is queried and holds a SET of k ∈ {1, 2, 3} objects at
query time; 2 unbound distractors; nothing else::

    Lucy has the lamp and the cup. A spoon lies on the shelf. Tom has the bell. Ada
    has the key and the coin. Ken has the pen. A fork lies on the shelf. So Lucy has the
    -> any of {" lamp", " cup"}

Cell = (k, layout, E):
  k       size of the queried person's FINAL set (1 / 2 / 3).
  layout  ``joint``: each person's objects in one sentence; ``split``: the queried
          person's objects declared in k separate sentences ("Lucy also has the cup.")
          interleaved with the others (k = 1: joint only).
  E       0 = static; 2 = two transfers in prose ("Lucy gives the cup to Tom."), at
          least one of which gains or loses the queried person an object; the query
          reads the final set.
Total objects in play C = 8 in every item (queried k + others + 2 distractors), so
chance is exact per cell: hit@1 = k/8, exact@k = 1/C(8, k).

``gold`` is a DESIGNATED member of the set (drawn at random at generation and kept in
``spec.designated``, so exemplar answers teach no primacy/recency regularity) so the
shared machinery (validation, rank read) works; the answer key is ``meta.gold_set`` and the
family's metrics come from the per-candidate logprobs (``set_metrics``):
  hit@1   the top candidate is in the set
  exact@k the top-k candidates are exactly the set
  margin  min logp(held) − max logp(non-held)   (> 0 ⇔ exact@k)

Shortcut controls (scheduled, audited):
  ``last_in_set``   the last-mentioned object before the query is in the set: the
                    "last mention" reader's hit@1. Scheduled at the hit@1 chance k/8
                    (cycled by item index), except in the static joint cells below.
  ``last_k_is_set`` the k most recently mentioned objects are exactly the set: the
                    "last k mentions" reader's exact@k. In a static joint item every
                    object is mentioned once and the queried person's objects share one
                    sentence, so the last mention is in the set exactly when that
                    sentence closes the scene, and then the last k mentions ARE the set:
                    the two controls are one event. Static joint cells with k ≥ 2 are
                    read by exact@k (chance 1/C(8, k)), so they schedule that event at a
                    budget matched to exact@k chance, ⌊n / C(8, k)⌋ items per n (the
                    queried sentence closes the scene on those items only). Their hit@1
                    "last mention" reader then scores below the hit@1 chance, and a
                    reader that never picks the last mention scores k/7 (1.14 × k/8).
                    Split layouts interleave the set and E = 2 ends on transfers, so
                    there the last k mentions are the set only when the transfers close
                    on exactly the set; those items are redrawn (rate 0).
  ``first_k_is_set`` the k first-mentioned objects are exactly the set (the queried
                    sentence opens the scene): the primacy reader's exact@k (at k = 1 its
                    hit@1). Scheduled in every joint cell at the same budget (k = 1: 1/8),
                    and at 0 in the split cells.
Error types (for a top pick outside the set): ``wrong_owner`` (another person's
object), ``stale`` (the queried person held it earlier, E = 2), ``distractor``,
``other``; a pick inside the set but not the designated gold is ``other_held``
(correct under set semantics).
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Mapping, Optional, Sequence, TypedDict

from kotodama.eval.bindprobes.base import Cell, Item, for_version, rejection, the_list, version_of
from kotodama.eval.bindprobes.pools import NON_COUNTABLE, Pools, countable, indefinite, letter_article

FAMILY = "setbind"
VERSION = 3
# the article of a distractor sentence, per generation
ARTICLE: dict[int, Callable[[str], str]] = {1: letter_article, 2: indefinite, 3: indefinite}
KS = (1, 2, 3)
LAYOUTS = ("joint", "split")
ES = (0, 2)
CELLS: list[Cell] = [(k, lay, e) for e in ES for k in KS for lay in LAYOUTS
                     if not (k == 1 and lay == "split")]
DEFAULT_PER_CELL = 64             # 10 cells -> 640 items (8 | 64: exact lm schedule)
SEPARATOR = "\n\n* * *\n\n"
N_PEOPLE = 4
N_DISTRACTORS = 2
C_TOTAL = 8                       # objects in play per item
ERROR_TYPES = ("wrong_owner", "stale", "distractor", "other", "other_held")
# Every object in play is a count noun (``pools.countable``): distractors take "A/An",
# and drawing bound objects from the same nouns keeps the noun kind from marking a slot.
BALANCE_TOL = 0.05


class Spec(TypedDict):
    people: list[str]               # scene order (labels)
    sentences: list[list[Any]]      # ["bind", person, [objs], also: bool] | ["free", obj] | ["give", g, obj, r]
    query: str
    designated: str                 # the set member rendered as an exemplar's answer


# ── Simulation ─────────────────────────────────────────────────────────────────

def run(spec: Spec) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(final holdings per person in acquisition order, every person's ever-held list).
    Raises on a precondition violation (double bind, give of an unheld object, self-give)."""
    people = spec["people"]
    if len(set(people)) != len(people):
        raise ValueError(f"duplicate people {people}")
    owner: dict[str, str] = {}
    held: dict[str, list[str]] = {p: [] for p in people}
    ever: dict[str, list[str]] = {p: [] for p in people}
    seen_give = False
    for k, s in enumerate(spec["sentences"]):
        kind = s[0]
        if kind == "bind":
            if seen_give:
                raise ValueError(f"sentence {k}: binding after a transfer")
            p, objs = s[1], s[2]
            if p not in people:
                raise ValueError(f"sentence {k}: {p!r} is not in the scene")
            for o in objs:
                if o in owner:
                    raise ValueError(f"sentence {k}: the {o} is bound twice")
                owner[o] = p
                held[p].append(o)
                ever[p].append(o)
        elif kind == "free":
            if s[1] in owner:
                raise ValueError(f"sentence {k}: distractor {s[1]!r} is also bound")
        elif kind == "give":
            g, o, r = s[1], s[2], s[3]
            seen_give = True
            if owner.get(o) != g:
                raise ValueError(f"sentence {k}: {g} does not hold the {o}")
            if g == r or r not in people:
                raise ValueError(f"sentence {k}: bad recipient {r!r}")
            owner[o] = r
            held[g].remove(o)
            held[r].append(o)
            ever[r].append(o)
        else:
            raise ValueError(f"sentence {k}: unknown kind {kind!r}")
    return held, ever


def gold_set(spec: Spec) -> list[str]:
    held, _ = run(spec)
    if spec["query"] not in held:
        raise ValueError(f"query {spec['query']!r} is not in the scene")
    return list(held[spec["query"]])


def _mentions(spec: Spec) -> list[str]:
    out: list[str] = []
    for s in spec["sentences"]:
        if s[0] == "bind":
            out += list(s[2])
        elif s[0] == "free":
            out.append(s[1])
        else:
            out.append(s[2])
    return out


def simulate(item: Item) -> str:
    """The designated gold (must be a member of the simulated final set)."""
    spec: Spec = item.spec  # type: ignore[assignment]
    gs = gold_set(spec)
    if spec["designated"] not in gs:
        raise ValueError(f"{item.item_id}: designated {spec['designated']!r} not in the set {gs}")
    return spec["designated"]


# ── Rendering ──────────────────────────────────────────────────────────────────

def _sentence(s: Sequence[Any], version: int = VERSION) -> str:
    if s[0] == "bind":
        also = " also" if s[3] else ""
        return f"{s[1]}{also} has {the_list(list(s[2]))}."
    if s[0] == "free":
        o = s[1]
        return f"{for_version(ARTICLE, version, FAMILY)(o).capitalize()} {o} lies on the shelf."
    return f"{s[1]} gives the {s[2]} to {s[3]}."


def _lead(spec: Spec) -> str:
    has_give = any(s[0] == "give" for s in spec["sentences"])
    return f"So now {spec['query']} has the" if has_give else f"So {spec['query']} has the"


def render(item: Item, with_answer: bool) -> str:
    spec: Spec = item.spec  # type: ignore[assignment]
    v = version_of(item)
    text = " ".join(_sentence(s, v) for s in spec["sentences"]) + " " + _lead(spec)
    return f"{text} {item.gold}." if with_answer else text


def labels(item: Item) -> list[str]:
    return list(item.spec["people"])


# ── Items ──────────────────────────────────────────────────────────────────────

def build(item_id: str, cell: Cell, spec: Spec, version: int = VERSION) -> Item:
    """Item from a spec; ``version`` is the generation it renders as (``relabel`` keeps the
    source item's)."""
    mentions = _mentions(spec)
    cands = list(dict.fromkeys(mentions))
    gs = gold_set(spec)
    it = Item(item_id=item_id, family=FAMILY, cell=list(cell), spec=dict(spec), gold="",
              candidates=cands, mention_order=[m for m in mentions if m in cands],
              meta={"setbind_version": version, "k": len(gs), "gold_set": gs,
                    "n_candidates": len(cands),
                    "last_in_set": bool(mentions) and mentions[-1] in gs,
                    "last_k_is_set": set(list(dict.fromkeys(reversed(mentions)))[:len(gs)])
                    == set(gs),
                    "first_k_is_set": set(list(dict.fromkeys(mentions))[:len(gs)]) == set(gs),
                    "n_transfers": sum(1 for s in spec["sentences"] if s[0] == "give"),
                    "layout": cell[1]})
    it.gold = simulate(it)
    return it


def exact_matched(cell: Sequence[Any]) -> bool:
    """A static joint cell with k ≥ 2: last_in_set and last_k_is_set are one event there,
    scheduled at the exact@k budget (module doc)."""
    return int(cell[0]) >= 2 and str(cell[1]) == "joint" and int(cell[2]) == 0


def schedule_lm(cell: Sequence[Any], k_index: int) -> bool:
    """last_in_set wanted for this test item: the first k of every C_TOTAL items (hit@1
    chance), or in an :func:`exact_matched` cell the last of every C(C_TOTAL, k) items
    (at most the exact@k chance)."""
    k = int(cell[0])
    if exact_matched(cell):
        m = math.comb(C_TOTAL, k)
        return k_index % m == m - 1
    return (k_index % C_TOTAL) < k


def schedule_fk(cell: Sequence[Any], k_index: int) -> Optional[bool]:
    """first_k_is_set wanted for this test item: in a joint cell on the middle item of
    every C(C_TOTAL, k), so the primacy reader matches the exact@k chance (at k = 1 the
    hit@1 chance 1/8); never in a split cell (its static items interleave the set, and
    its transfers could otherwise assemble the set at the front now and then)."""
    k = int(cell[0])
    if str(cell[1]) != "joint":
        return False
    m = math.comb(C_TOTAL, k)
    return k_index % m == m // 2


def _counts_others(rng: random.Random, n_left: int) -> Optional[list[int]]:
    """Split ``n_left`` objects over the 3 others, each 1–2 (None if impossible)."""
    for _ in range(50):
        c = [rng.choice((1, 2)) for _ in range(N_PEOPLE - 1)]
        if sum(c) == n_left:
            return c
    return None


def _draw(rng: random.Random, cell: Cell, pools: Pools, item_id: str,
          want_lm: Optional[bool], want_fk: Optional[bool] = None) -> Optional[Item]:
    k, layout, e = int(cell[0]), str(cell[1]), int(cell[2])
    people = rng.sample(pools.names, N_PEOPLE)
    q = rng.choice(people)
    objs = list(countable(pools.objects))
    if len(objs) < C_TOTAL:
        raise ValueError(f"setbind: object pool too small ({len(objs)})")
    pick = rng.sample(objs, C_TOTAL)
    free, bound = pick[:N_DISTRACTORS], pick[N_DISTRACTORS:]
    n_bound = len(bound)
    # initial queried size: E=0 -> k; E=2 -> k-1 / k / k+1 (the transfers fix it up)
    k0 = k if e == 0 else rng.choice([x for x in (k - 1, k, k + 1) if 1 <= x <= 3])
    others = [p for p in people if p != q]
    oc = _counts_others(rng, n_bound - k0)
    if oc is None:
        return None
    hold0: dict[str, list[str]] = {q: bound[:k0]}
    i = k0
    for p, c in zip(others, oc):
        hold0[p] = bound[i:i + c]
        i += c
    # declaration sentences (scene order); split layout: queried objects one per sentence, interleaved
    decl: list[list[Any]] = []
    if layout == "joint" or k0 == 1:
        for p in people:
            decl.append(["bind", p, list(hold0[p]), False])
    else:
        other_sents = [["bind", p, list(hold0[p]), False] for p in others]
        q_sents = [["bind", q, [o], j > 0] for j, o in enumerate(hold0[q])]
        # bins: before q1, between q_j and q_{j+1} (>= 1 other each), after the last
        if len(other_sents) < len(q_sents) - 1:
            return None
        rng.shuffle(other_sents)
        bins = [0] + [1] * (len(q_sents) - 1) + [0]
        for _ in range(len(other_sents) - (len(q_sents) - 1)):
            bins[rng.randrange(len(bins))] += 1
        it_others = iter(other_sents)
        decl = [next(it_others) for _ in range(bins[0])]
        for j, qs in enumerate(q_sents):
            decl.append(qs)
            decl += [next(it_others) for _ in range(bins[j + 1])]
    # distractors: random slots among the declarations (optionally one at the very end)
    sents: list[list[Any]] = list(decl)
    for o in free:
        sents.insert(rng.randint(0, len(sents)), ["free", o])
    if e == 2:
        held = {p: list(v) for p, v in hold0.items()}
        gives: list[list[Any]] = []
        for _ in range(e):
            givers = [p for p in people if held[p]]
            g = rng.choice(givers)
            o = rng.choice(held[g])
            r = rng.choice([p for p in people if p != g])
            held[g].remove(o)
            held[r].append(o)
            gives.append(["give", g, o, r])
        if len(held[q]) != k or not any(q in (gv[1], gv[3]) for gv in gives):
            return None
        sents += gives
    try:
        final = run({"people": people, "sentences": sents, "query": q, "designated": ""})[0][q]
    except ValueError:
        return None
    if not final:
        return None
    spec: Spec = {"people": people, "sentences": sents, "query": q, "designated": rng.choice(final)}
    try:
        it = build(item_id, cell, spec)
    except ValueError:
        return None
    if it.meta["k"] != k or len(it.candidates) != C_TOTAL:
        return None
    if want_lm is not None and it.meta["last_in_set"] != want_lm:
        return None
    if want_lm is not None and k >= 2 and not exact_matched(cell) and it.meta["last_k_is_set"]:
        return None                     # transfers that happen to close on exactly the set
    if want_fk is not None and it.meta["first_k_is_set"] != want_fk:
        return None
    return it


def gen_item(rng: random.Random, cell: Cell, pools: Pools, item_id: str, k: int,
             per_cell: int, template: Optional[Item] = None) -> Item:
    if tuple(cell) not in CELLS:
        raise ValueError(f"setbind: unknown cell {cell}")
    cell = tuple(cell)
    want: Optional[bool] = schedule_lm(cell, k) if template is None else None
    want_fk: Optional[bool] = schedule_fk(cell, k) if template is None else None
    return rejection(lambda: _draw(rng, cell, pools, item_id, want, want_fk),
                     f"setbind item {item_id}", tries=20000)


def relabel(item: Item, mapping: Mapping[str, str], target: Optional[Item] = None) -> Item:
    spec: Spec = item.spec  # type: ignore[assignment]
    missing = set(spec["people"]) - set(mapping)
    if missing:
        raise ValueError(f"{item.item_id}: people {sorted(missing)} not in the map")
    m = dict(mapping)

    def ren(s: list[Any]) -> list[Any]:
        if s[0] == "bind":
            return ["bind", m[s[1]], list(s[2]), s[3]]
        if s[0] == "give":
            return ["give", m[s[1]], s[2], m[s[3]]]
        return list(s)
    new: Spec = {"people": [m[p] for p in spec["people"]],
                 "sentences": [ren(s) for s in spec["sentences"]], "query": m[spec["query"]],
                 "designated": spec["designated"]}
    out = build(f"{item.item_id}_collide", tuple(item.cell), new, version=version_of(item))
    if set(out.meta["gold_set"]) != set(item.meta["gold_set"]):
        raise ValueError(f"{item.item_id}: re-label changed the gold set")
    return out


# ── Validation / audit ─────────────────────────────────────────────────────────

def validate_item(item: Item) -> None:
    spec: Spec = item.spec  # type: ignore[assignment]
    k, layout, e = item.cell
    if item.meta.get("setbind_version") != VERSION:
        raise ValueError(f"{item.item_id}: setbind_version {item.meta.get('setbind_version')!r}")
    gs = gold_set(spec)
    if len(gs) != k or sorted(gs) != sorted(item.meta["gold_set"]):
        raise ValueError(f"{item.item_id}: gold set {gs} does not match cell k={k} / meta")
    if len(item.candidates) != C_TOTAL:
        raise ValueError(f"{item.item_id}: {len(item.candidates)} candidates, want {C_TOTAL}")
    if not set(gs) <= set(item.candidates):
        raise ValueError(f"{item.item_id}: gold set not inside the candidates")
    if item.meta["n_transfers"] != e:
        raise ValueError(f"{item.item_id}: {item.meta['n_transfers']} transfers, cell E={e}")
    q_binds = [s for s in spec["sentences"] if s[0] == "bind" and s[1] == spec["query"]]
    if layout == "split" and e == 0 and len(q_binds) != k:
        raise ValueError(f"{item.item_id}: split layout needs {k} queried sentences")
    if any(o in NON_COUNTABLE for o in item.candidates):
        raise ValueError(f"{item.item_id}: a non-count noun in play")


def audit(items: Sequence[Item]) -> dict[str, dict[str, Any]]:
    """Per cell: n, the hit@1 and exact@k chances, both recency controls and the rate the
    schedule designs for ``last_in_set`` (the schedule's share over the cell's n items)."""
    out: dict[str, dict[str, Any]] = {}
    for c in CELLS:
        sub = [it for it in items if it.family == FAMILY and tuple(it.cell) == c]
        if not sub:
            continue
        n = len(sub)
        out["/".join(map(str, c))] = {
            "n": n, "chance_hit1": int(c[0]) / C_TOTAL,
            "chance_exactk": 1.0 / math.comb(C_TOTAL, int(c[0])),
            "last_in_set_designed": sum(schedule_lm(c, i) for i in range(n)) / n,
            "last_in_set": sum(it.meta["last_in_set"] for it in sub) / n,
            "last_k_is_set": sum(it.meta["last_k_is_set"] for it in sub) / n,
            "first_k_is_set": sum(it.meta["first_k_is_set"] for it in sub) / n}
    return out


def audit_failures(aud: Mapping[str, Mapping[str, Any]], tol: float = BALANCE_TOL) -> list[str]:
    """last_in_set within max(tol, 1/n) of its designed rate; the exact@k recency and
    primacy readers (last_k_is_set, first_k_is_set) at most 1.5 × the exact@k chance
    (k = 1: the hit@1 chance), or at most one item."""
    bad = []
    for ck, a in aud.items():
        band = max(tol, 1.0 / a["n"]) + 1e-9
        if abs(a["last_in_set"] - a["last_in_set_designed"]) > band:
            bad.append(f"{ck}: last_in_set {a['last_in_set']:.3f} vs designed "
                       f"{a['last_in_set_designed']:.3f}")
        for key in ("last_k_is_set", "first_k_is_set"):
            if a[key] > max(1.5 * a["chance_exactk"], 1.0 / a["n"]) + 1e-9:
                bad.append(f"{ck}: {key} {a[key]:.3f} vs exact@k chance {a['chance_exactk']:.3f}")
    return bad


# ── Errors + metrics ───────────────────────────────────────────────────────────

def error_types(item: Item, pick: str) -> list[str]:
    spec: Spec = item.spec  # type: ignore[assignment]
    gs = set(item.meta["gold_set"])
    if pick in gs:
        return ["other_held"]
    held, ever = run(spec)
    q = spec["query"]
    if pick in ever[q]:
        return ["stale"]
    if any(pick in v for p, v in held.items() if p != q):
        return ["wrong_owner"]
    if any(s[0] == "free" and s[1] == pick for s in spec["sentences"]):
        return ["distractor"]
    return ["other"]


def set_metrics(cand_lps: Mapping[str, float], gold: Sequence[str]) -> dict[str, float]:
    """hit@1 / exact@k / margin from per-candidate logprobs (keys as scored, with or
    without the leading space)."""
    lp = {c.strip(): float(v) for c, v in cand_lps.items()}
    gs = [g for g in gold]
    if not set(gs) <= set(lp):
        raise ValueError(f"gold set {gs} not in candidates {sorted(lp)}")
    order = sorted(lp, key=lambda c: -lp[c])
    k = len(gs)
    held = [lp[g] for g in gs]
    non = [v for c, v in lp.items() if c not in set(gs)]
    return {"hit1": float(order[0] in set(gs)),
            "exactk": float(set(order[:k]) == set(gs)),
            "margin": min(held) - max(non),
            "chance_hit1": k / len(lp), "chance_exactk": 1.0 / math.comb(len(lp), k)}
