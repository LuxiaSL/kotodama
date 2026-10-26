"""Forming vs updating a binding: one base item, three conditions that differ only in the lines after the bindings.

A rebinding chain (fluid F3, :mod:`kotodama.eval.binding.fluid`) confounds holding a binding with changing it: a
model can fail because the context is long, because many things move, or because the queried thing moved. This
battery separates the three with a paired design. A **base item** is K holders, each bound to one value, then U more
lines, then a query about one holder Q. Three conditions are derived from each base item and share its holders, its
initial values, its query, its candidate layout and its line count; only the U lines differ:

  A  static          U neutral lines with no holder and no value (templated filler). Gold = Q's initial value.
  B  update-other    U updates, all to holders other than Q. Gold = Q's initial value.
  C  update-target   U updates; exactly one updates Q, at a scheduled position among the U lines (the others are B's
                     lines unchanged, so B and C differ in that one line). Gold = Q's new value; Q's initial value is
                     the STALE candidate.

Templates are fluid F3's: prose ``The bag holds the shell.`` / update ``Then the bag is emptied and the camera is put
in it.`` / query ``In the end, the bag holds the``; code ``bag = shell`` / ``bag = camera`` / ``assert bag ==``. The
bindings come first, then the U lines, then the query.

Grid: K ∈ {2, 4, 8} × U ∈ {1, 2, 4} × register {prose, code} × 40 base items = 720 base items, 2,160 rows.

Candidates (5, chance 1/5), one fixed layout per base item: gold; Q's other value (C: the stale initial value; A / B:
the value C introduces, shown nowhere in A or B); and three other holders' CURRENT values in that condition (the
same three holders in all three conditions). Every candidate is drawn from fluid's ``VALUES``, a pool already checked
to be one token after a space, with distinct tokens, in every tokenizer of ``fluid.ZOO_TOKENIZERS``
(``python -m kotodama.eval.binding check-fluid-tokens``). No pool word is added here.

Choices left open, fixed here:
  - holders: fluid's ``HOLDERS`` (8), so K = 8 uses all of them; Q is a random one of the K.
  - Q's binding line sits at a position balanced exactly over the K binding lines per cell; the other holders' order
    is random. Q's update in C sits at a position balanced exactly over the U lines per cell.
  - B's U updates go to the other holders in shuffled rounds (no holder twice in a row while another is free), so a
    holder is updated more than once only when U > K − 1. An update empties the holder before the new value goes in
    and never names the old value, so every value in a prompt is mentioned exactly once.
  - A's neutral lines are the capacity battery's filler sentences (``kotodama.eval.binding.capacity``), minus any
    phrase containing a holder or value word, distinct within a prompt: prose ``The quiet river runs past the
    town.``; code, the same words as a comment (``# the quiet river runs past the town``).
  - K = 2 has one other holder, so "three other holders' current values" cannot be met: its two remaining
    candidates are values shown in none of the three prompts (role ``absent``), the same two in A, B and C.
  - candidate layout: per cell, the (Q initial, Q new) slot pair cycles through the 20 ordered pairs of distinct
    slots, each twice in 40 base items, so the gold position is exactly balanced in every condition; the remaining
    slots take the other holders in random order, shared by the three conditions.
  - gold identity: Q's initial and new values each cycle through a shuffled copy of ``VALUES`` per cell (never equal
    within a base item).
  - roles: ``gold``; ``stale`` (C) or ``q_unshown`` (A, B); ``other_current`` (a holder's value from its binding
    line) or ``other_updated`` (a holder's value from an update line); ``absent`` (K = 2).
  - bootstrap: resamples base items with replacement WITHIN each (K, U, register) cell (the grid is designed, not
    sampled), 10,000 resamples, seed 0; percentile 95% intervals. A resample whose acc(A) ≤ chance has no normalized
    cost and is left out of that statistic's interval (the count is reported).

Shortcut controls (recorded per row; :func:`audit`, :func:`audit_failures`):
  recency    gold = the last value mentioned in the prompt. A has no value after the bindings, so its last mention
             is the last binding line, which is Q's on 1/K of rows (Q's binding position is balanced over the K
             lines: recency carries no information about which holder is asked). In B the last mention is always
             another holder's new value: 0. In C Q's update is placed uniformly over the U lines, so the gold is the
             last mention on 1/U of rows (all of them at U = 1). The gate checks each condition's rate against
             these designed rates; B (0) and C (1/U) are NOT the chance rate 1/5, so :func:`summarize` also reports
             the update cost on the base items whose C gold is not the last mention (U ≥ 2, Q's update not last),
             where B and C are recency-matched (both 0).
  frequency  every candidate is mentioned at most once, and the gold exactly once, in every row (gated). Among the
             candidates present in a prompt, frequency carries no information.
  presence   the share a model gets by guessing uniformly among the candidates present in the prompt. A and B
             carry one absent candidate (Q's unshown value; three at K = 2), C none (two at K = 2), so a presence
             guesser scores 1/4 in A / B and 1/5 in C at K ≥ 4 (1/2 and 1/3 at K = 2). Reported, not gated.
  position / identity: exact balance (above), audited.
  pairing    :func:`pairing_failures` checks every base item's A / B / C rows against the invariants above.

Read-out (:func:`summarize`), per model, paired within base item:
  form              acc(A)
  interference cost acc(A) − acc(B)
  update cost       acc(B) − acc(C)
  normalized        cost / (acc(A) − chance)
  stale rate        C rows that pick the stale value, overall and as a share of C's errors, by K / U / register
  breakdowns        the same statistics per K, per U, per register and per cell
  intervals         paired bootstrap (above) for form, both costs, both normalized costs, normalized update −
                    normalized interference, and the recency-matched update cost
:func:`predictions` evaluates four comparisons of one release (and one other checkpoint) against the median of a
reference set of models; membership is passed in, not built in.
"""

from __future__ import annotations

import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

from kotodama.eval.binding import capacity as cap
from kotodama.eval.binding import fluid as fl

VERSION = "formupdate-v1"
BATTERY = "formupdate-v1"
READER = "formupdate"
K_GRID: tuple[int, ...] = (2, 4, 8)
U_GRID: tuple[int, ...] = (1, 2, 4)
REGISTERS: tuple[str, ...] = fl.F3_REGISTERS
BASE_ITEMS = 40
N_CANDIDATES = 5
CHANCE = 1 / N_CANDIDATES
OTHER_CANDIDATES = 3
BOOT = 10_000
BOOT_SEED = 0
BALANCE_TOL = fl.BALANCE_TOL
IDENTITY_MAX = fl.IDENTITY_MAX

Condition = Literal["A", "B", "C"]
CONDITIONS: tuple[Condition, ...] = ("A", "B", "C")
HOLDERS: tuple[str, ...] = fl.HOLDERS
VALUES: tuple[str, ...] = fl.VALUES

Row = dict[str, Any]


def _clean(phrases: Sequence[str]) -> tuple[str, ...]:
    banned = set(HOLDERS) | set(VALUES)
    return tuple(p for p in phrases if not set(p.split()) & banned)


# The capacity battery's filler vocabulary, minus phrases that name a holder or a value ("the edge of the map").
FILLER_ADJ: tuple[str, ...] = _clean(cap._ADJ)
FILLER_NOUN: tuple[str, ...] = _clean(cap._NOUN)
FILLER_VERB: tuple[str, ...] = _clean(cap._VERB)
FILLER_PLACE: tuple[str, ...] = _clean(cap._PLACE)

CELLS: tuple[tuple[int, int, str], ...] = tuple((k, u, reg) for k in K_GRID for u in U_GRID for reg in REGISTERS)


def cell_key(k: int, u: int, register: str) -> str:
    return f"K{k}|U{u}|{register}"


# ── generation ───────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _CellSchedule:
    slots: list[tuple[int, int]]          # (slot of Q's initial value, slot of Q's new value)
    q_bind: list[int]                     # Q's binding line, 0 … K − 1
    q_update: list[int]                   # Q's update line in C, 0 … U − 1
    initial: list[str]                    # Q's initial value
    new: list[str]                        # the value C gives Q


def _schedule(k: int, u: int, register: str, n: int) -> _CellSchedule:
    rng = random.Random(f"{VERSION}:{cell_key(k, u, register)}:schedule")
    pairs = [(i, j) for i in range(N_CANDIDATES) for j in range(N_CANDIDATES) if i != j]
    slots = fl._balanced(rng, n, pairs)
    q_bind = fl._balanced(rng, n, range(k))
    q_update = fl._balanced(rng, n, range(u))
    initial = fl._cycle(rng, VALUES, n)
    new = fl._cycle(rng, VALUES, n)
    for i in range(n):                    # Q's two values must differ: swap with the first compatible partner
        if new[i] == initial[i]:
            j = next(j for j in range(n) if j != i and new[j] != initial[i] and new[i] != initial[j])
            new[i], new[j] = new[j], new[i]
    return _CellSchedule(slots, q_bind, q_update, initial, new)


def _targets(rng: random.Random, others: Sequence[str], u: int) -> list[str]:
    """The holders B updates: shuffled rounds over ``others``, no holder twice in a row while another exists."""
    out: list[str] = []
    while len(out) < u:
        r = list(others)
        rng.shuffle(r)
        if out and len(r) > 1 and r[0] == out[-1]:
            r.append(r.pop(0))
        out.extend(r)
    return out[:u]


def _fillers(rng: random.Random, u: int) -> list[str]:
    out: list[str] = []
    while len(out) < u:
        s = f"{rng.choice(FILLER_ADJ)} {rng.choice(FILLER_NOUN)} {rng.choice(FILLER_VERB)} {rng.choice(FILLER_PLACE)}"
        if s not in out:
            out.append(s)
    return out


@dataclass(frozen=True)
class BaseItem:
    """One base item; :meth:`events` derives each condition's lines from it."""
    base_id: str
    k: int
    u: int
    register: str
    q: str                                      # the queried holder
    order: tuple[str, ...]                      # holders in binding order
    initial: dict[str, str]                     # holder → value on its binding line
    q_new: str                                  # the value C gives Q
    updates: tuple[tuple[str, str], ...]        # B's U update lines (holder, value)
    q_update_pos: int                           # the U line C turns into Q's update
    fillers: tuple[str, ...]                    # A's U neutral lines
    # candidate slots: ("q", "initial" | "new"), ("other", holder) or ("absent", value)
    layout: tuple[tuple[str, str], ...]

    def events(self, cond: Condition) -> list[tuple[str, Optional[str], Optional[str]]]:
        """(kind, holder, value) per line; kind ∈ bind / update / filler (a filler's holder is its text)."""
        out: list[tuple[str, Optional[str], Optional[str]]] = [("bind", h, self.initial[h]) for h in self.order]
        if cond == "A":
            return out + [("filler", f, None) for f in self.fillers]
        ups = list(self.updates)
        if cond == "C":
            ups[self.q_update_pos] = (self.q, self.q_new)
        elif cond != "B":
            raise ValueError(f"unknown condition {cond!r} (A | B | C)")
        return out + [("update", h, v) for h, v in ups]


def render(register: str, events: Sequence[tuple[str, Optional[str], Optional[str]]], q: str) -> str:
    """The prompt text: fluid F3's sentences (prose) or assignments (code), filler lines, then the query."""
    if register == "prose":
        parts = []
        for kind, h, v in events:
            if kind == "bind":
                parts.append(f"The {h} holds the {v}.")
            elif kind == "update":
                parts.append(f"Then the {h} is emptied and the {v} is put in it.")
            else:
                parts.append(f"The {h}.")
        return " ".join(parts) + f" In the end, the {q} holds the"
    if register == "code":
        lines = [f"{h} = {v}" if kind != "filler" else f"# the {h}" for kind, h, v in events]
        return "\n".join(lines) + f"\nassert {q} =="
    raise ValueError(f"unknown register {register!r} (prose | code)")


def _make_base(k: int, u: int, register: str, b: int, s: _CellSchedule) -> BaseItem:
    cell = cell_key(k, u, register)
    rng = random.Random(f"{VERSION}:{cell}:{b}")
    holders = rng.sample(HOLDERS, k)
    q, others = holders[0], holders[1:]
    order = list(others)
    rng.shuffle(order)
    order.insert(s.q_bind[b], q)
    v0, v1 = s.initial[b], s.new[b]
    n_absent = max(0, OTHER_CANDIDATES - len(others))
    draw = rng.sample([w for w in VALUES if w not in (v0, v1)], len(others) + u + n_absent)
    initial = {q: v0, **dict(zip(others, draw[:len(others)]))}
    writes = draw[len(others):len(others) + u]
    absent = draw[len(others) + u:]
    updates = tuple(zip(_targets(rng, others, u), writes))
    shown = rng.sample(others, OTHER_CANDIDATES) if len(others) >= OTHER_CANDIDATES else list(others)
    rest: list[tuple[str, str]] = [("other", h) for h in shown] + [("absent", w) for w in absent]
    rng.shuffle(rest)
    first, second = s.slots[b]
    layout: list[tuple[str, str]] = []
    for slot in range(N_CANDIDATES):
        layout.append(("q", "initial") if slot == first else ("q", "new") if slot == second else rest.pop(0))
    return BaseItem(f"{VERSION}|{cell}|b{b:02d}", k, u, register, q, tuple(order), initial, v1, updates,
                    s.q_update[b], tuple(_fillers(rng, u)), tuple(layout))


def _row(item: BaseItem, cond: Condition) -> Row:
    events = item.events(cond)
    current: dict[str, str] = {}
    updated: set[str] = set()
    for kind, h, v in events:
        if kind != "filler":
            assert h is not None and v is not None
            current[h] = v
            if kind == "update":
                updated.add(h)
    v0, v1 = item.initial[item.q], item.q_new
    gold = v1 if cond == "C" else v0
    cands: list[str] = []
    roles: dict[str, str] = {}
    for kind, x in item.layout:
        if kind == "q":
            w = v0 if x == "initial" else v1
            role = "gold" if w == gold else ("stale" if cond == "C" else "q_unshown")
        elif kind == "other":
            w = current[x]
            role = "other_updated" if x in updated else "other_current"
        else:
            w, role = x, "absent"
        cands.append(w)
        roles[w] = role
    if len(set(cands)) != N_CANDIDATES or current[item.q] != gold:
        raise AssertionError(f"{item.base_id} {cond}: candidates {cands} (gold {gold!r})")
    prompt = render(item.register, events, item.q)
    recent = next(v for kind, _, v in reversed(events) if kind != "filler")
    neutral = 1 / item.k if cond == "A" else 0.0 if cond == "B" else 1 / item.u
    row: Row = {
        "id": f"{item.base_id}|{cond}", "base_id": item.base_id, "condition": cond,
        "cell": cell_key(item.k, item.u, item.register), "K": item.k, "U": item.u, "register": item.register,
        "prompt": prompt, "candidates": [" " + w for w in cands], "gold": gold, "roles": roles,
        "gold_index": cands.index(gold), "holder": item.q, "q_initial": v0, "q_new": v1,
        "stale": v0 if cond == "C" else None, "q_bind_pos": item.order.index(item.q),
        "q_update_pos": item.q_update_pos, "n_lines": len(events),
    }
    row.update(fl._controls(prompt, cands, gold, recent))
    counts = [fl.count_word(prompt, w) for w in cands]
    present = sum(c > 0 for c in counts)
    row["neutral_recency"] = neutral
    row["max_count"] = max(counts)
    row["presence_follower"] = (1.0 / present) if row["gold_count"] > 0 else 0.0
    return row


def base_items(base: int = BASE_ITEMS) -> list[BaseItem]:
    out: list[BaseItem] = []
    for k, u, reg in CELLS:
        s = _schedule(k, u, reg, base)
        out.extend(_make_base(k, u, reg, b, s) for b in range(base))
    return out


def generate(base: int = BASE_ITEMS) -> list[Row]:
    """The battery's prompt rows, A / B / C of each base item in a row. Deterministic: no global RNG state."""
    return [_row(item, c) for item in base_items(base) for c in CONDITIONS]


# ── audits ───────────────────────────────────────────────────────────────────────────────────────────────────────

def audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per ``cell|condition`` and pooled per ``*|condition``: n, the recency rate against its designed rate (and
    chance), frequency (the follower's accuracy, the largest candidate mention count, the gold's mention counts),
    the presence follower, and the gold-position balance; under ``identity|condition``, the gold-identity balance."""
    by: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        by[f"{r['cell']}|{r['condition']}"].append(r)
        by[f"*|{r['condition']}"].append(r)
    out: dict[str, dict[str, Any]] = {}
    for key, sub in by.items():
        n = len(sub)
        pos = Counter(r["gold_index"] for r in sub)
        out[key] = {
            "n": n, "chance": CHANCE,
            "recency": sum(r["gold_is_recent"] for r in sub) / n,
            "recency_designed": sum(r["neutral_recency"] for r in sub) / n,
            "frequency": sum(r["freq_follower"] for r in sub) / n,
            "max_count": max(r["max_count"] for r in sub),
            "gold_count": sorted({r["gold_count"] for r in sub}),
            "presence": sum(r["presence_follower"] for r in sub) / n,
            "novelty": sum(r["novelty_follower"] for r in sub) / n,
            "position_max_dev": max(abs(pos.get(i, 0) / n - CHANCE) for i in range(N_CANDIDATES)),
        }
    for cond in CONDITIONS:
        cnt = Counter(r["gold"] for r in rows if r["condition"] == cond)
        if cnt:
            mean = sum(cnt.values()) / len(cnt)
            out[f"identity|{cond}"] = {"identities": len(cnt), "identity_max_ratio": max(cnt.values()) / mean,
                                       "identity_min": min(cnt.values())}
    return out


def audit_failures(aud: Mapping[str, Mapping[str, Any]], tol: float = BALANCE_TOL) -> list[str]:
    """Recency within max(tol, 1/n) of its designed rate; no candidate mentioned twice and the gold exactly once;
    gold positions within the same band of uniform; no gold identity above IDENTITY_MAX × the mean."""
    bad: list[str] = []
    for key, a in aud.items():
        if key.startswith("identity|"):
            if a["identity_max_ratio"] > IDENTITY_MAX:
                bad.append(f"{key}: gold identity max/mean {a['identity_max_ratio']:.2f} > {IDENTITY_MAX}")
            continue
        band = max(tol, 1 / a["n"]) + 1e-9
        if abs(a["recency"] - a["recency_designed"]) > band:
            bad.append(f"{key}: recency {a['recency']:.3f} vs designed {a['recency_designed']:.3f}")
        if a["max_count"] > 1 or a["gold_count"] != [1]:
            bad.append(f"{key}: a candidate mentioned {a['max_count']}× / gold mentioned {a['gold_count']}×")
        if a["position_max_dev"] > band:
            bad.append(f"{key}: gold position off uniform by {a['position_max_dev']:.3f}")
    return bad


def _code_lines(r: Mapping[str, Any]) -> list[str]:
    return r["prompt"].split("\n")[:-1]


def _prose_lines(r: Mapping[str, Any]) -> list[str]:
    body = r["prompt"].rsplit(" In the end, the ", 1)[0]
    return [s.strip() + "." for s in body.split(".") if s.strip()]


def lines_of(r: Mapping[str, Any]) -> list[str]:
    """The binding / update / filler lines of a row, in order (prose: one sentence per line)."""
    return _code_lines(r) if r["register"] == "code" else _prose_lines(r)


def pairing_failures(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Each base item has exactly one A, B and C row sharing holder, values, query, line count and the first K
    lines; A's U lines name no holder or value; B and C differ in exactly Q's update line; the candidate layout is
    shared (Q's two values in the same slots, the other slots naming the same holders' values or the same absent
    values)."""
    groups: defaultdict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    bad: list[str] = []
    for r in rows:
        if r["condition"] in groups[r["base_id"]]:
            bad.append(f"{r['base_id']}: two {r['condition']} rows")
        groups[r["base_id"]][r["condition"]] = r
    words = set(HOLDERS) | set(VALUES)
    for bid, g in groups.items():
        if set(g) != set(CONDITIONS):
            bad.append(f"{bid}: conditions {sorted(g)}")
            continue
        a, b, c = g["A"], g["B"], g["C"]
        k, u = a["K"], a["U"]
        la, lb, lc = lines_of(a), lines_of(b), lines_of(c)
        if not (len(la) == len(lb) == len(lc) == k + u == a["n_lines"]):
            bad.append(f"{bid}: line counts {len(la)}/{len(lb)}/{len(lc)} (K + U = {k + u})")
            continue
        if not la[:k] == lb[:k] == lc[:k]:
            bad.append(f"{bid}: binding lines differ")
        if any(set(x.rstrip(".").split()) & words for x in la[k:]):
            bad.append(f"{bid}: an A filler line names a holder or a value")
        diff = [i for i in range(k, k + u) if lb[i] != lc[i]]
        if diff != [k + c["q_update_pos"]]:
            bad.append(f"{bid}: B and C differ at lines {diff}, want [{k + c['q_update_pos']}]")
        for f in ("holder", "q_initial", "q_new", "q_bind_pos", "q_update_pos", "K", "U", "register", "cell"):
            if not a[f] == b[f] == c[f]:
                bad.append(f"{bid}: {f} differs across conditions")
        tail = {r["prompt"].rsplit("\n", 1)[-1] if r["register"] == "code" else r["prompt"].rsplit(". ", 1)[-1]
                for r in (a, b, c)}
        if len(tail) != 1:
            bad.append(f"{bid}: queries differ")
        ca, cb, cc = ([x.strip() for x in r["candidates"]] for r in (a, b, c))
        if not (ca.index(a["q_initial"]) == cb.index(b["q_initial"]) == cc.index(c["q_initial"])
                and ca.index(a["q_new"]) == cb.index(b["q_new"]) == cc.index(c["q_new"])):
            bad.append(f"{bid}: Q's values sit in different slots")
        if a["gold"] != a["q_initial"] or b["gold"] != b["q_initial"] or c["gold"] != c["q_new"]:
            bad.append(f"{bid}: gold is not A/B initial, C new")
        if fl.count_word(a["prompt"], a["q_new"]) or fl.count_word(b["prompt"], b["q_new"]):
            bad.append(f"{bid}: Q's new value shown in A or B")
        for i in range(N_CANDIDATES):
            ra, rb, rc = (r["roles"][x[i]] for r, x in ((a, ca), (b, cb), (c, cc)))
            if (ra == "absent" or rb == "absent" or rc == "absent") and not ca[i] == cb[i] == cc[i]:
                bad.append(f"{bid}: absent slot {i} differs")
    return bad


# ── read-out ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _pick(r: Mapping[str, Any]) -> str:
    lps = r["cand_lps"]
    return max(lps, key=lps.get)


@dataclass(frozen=True)
class _Item:
    cell: tuple[int, int, str]
    ok: tuple[bool, bool, bool]           # A, B, C correct
    stale: bool                           # C picks the stale value
    c_recent: bool                        # C's gold is the last mention


def _stats(items: Sequence[_Item]) -> dict[str, Any]:
    n = len(items)
    if not n:
        return {"n": 0}
    acc = {c: sum(it.ok[i] for it in items) / n for i, c in enumerate(CONDITIONS)}
    head = acc["A"] - CHANCE
    inter, upd = acc["A"] - acc["B"], acc["B"] - acc["C"]
    c_wrong = sum(not it.ok[2] for it in items)
    stale = sum(it.stale for it in items)
    return {"n": n, "acc": acc, "form": acc["A"], "interference": inter, "update": upd, "headroom": head,
            "norm_interference": inter / head if head > 0 else None,
            "norm_update": upd / head if head > 0 else None,
            "stale_rate": stale / n, "stale_of_errors": stale / c_wrong if c_wrong else None}


BOOT_STATS: tuple[str, ...] = ("form", "interference", "update", "norm_interference", "norm_update",
                               "norm_update_minus_interference", "update_nonrecent")
_BASE = 1 << 12                            # one digit per counter: holds any count up to the number of base items


def _counts_stats(a: int, b: int, c: int, n: int, sb: int, sc: int, sn: int) -> dict[str, Optional[float]]:
    acc_a, acc_b, acc_c = a / n, b / n, c / n
    head = acc_a - CHANCE
    ni = (acc_a - acc_b) / head if head > 0 else None
    nu = (acc_b - acc_c) / head if head > 0 else None
    return {"form": acc_a, "interference": acc_a - acc_b, "update": acc_b - acc_c, "norm_interference": ni,
            "norm_update": nu, "norm_update_minus_interference": (nu - ni) if nu is not None and ni is not None
            else None, "update_nonrecent": (sb - sc) / sn if sn else None}


def bootstrap(items: Sequence[_Item], n_boot: int = BOOT, seed: int = BOOT_SEED) -> dict[str, dict[str, Any]]:
    """Paired percentile intervals (95%) for :data:`BOOT_STATS`: base items resampled with replacement within each
    cell, every condition of a drawn item kept together."""
    if not items:
        raise ValueError("no base items")
    if len(items) >= _BASE:
        raise ValueError(f"{len(items)} base items overflow the counter digit {_BASE}")
    by: defaultdict[tuple[int, int, str], list[int]] = defaultdict(list)
    for it in items:
        sub = not it.c_recent
        # digits: A, B, C, recency-matched flag, B and C on recency-matched items
        code = (it.ok[0] + it.ok[1] * _BASE + it.ok[2] * _BASE ** 2 + sub * _BASE ** 3
                + (sub and it.ok[1]) * _BASE ** 4 + (sub and it.ok[2]) * _BASE ** 5)
        by[it.cell].append(code)
    cells = [by[k] for k in sorted(by)]
    n = len(items)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {s: [] for s in BOOT_STATS}
    for _ in range(n_boot):
        tot = 0
        for codes in cells:
            tot += sum(rng.choices(codes, k=len(codes)))
        d = [(tot // _BASE ** i) % _BASE for i in range(6)]
        for s, v in _counts_stats(d[0], d[1], d[2], n, d[4], d[5], d[3]).items():
            if v is not None:
                draws[s].append(v)
    out: dict[str, dict[str, Any]] = {}
    for s, xs in draws.items():
        xs.sort()
        out[s] = {"ci": [xs[int(.025 * len(xs))], xs[min(int(.975 * len(xs)), len(xs) - 1)]] if xs else None,
                  "defined": len(xs), "n_boot": n_boot}
    return out


@dataclass
class Summary:
    model: str
    n_items: int
    n_rows: int
    incomplete: int                                     # base items missing a condition (left out)
    multi_token_rows: int
    pooled: dict[str, Any]
    ci: dict[str, dict[str, Any]]
    point: dict[str, Optional[float]]                   # the pooled point estimate of every BOOT_STATS entry
    by_k: dict[int, dict[str, Any]] = field(default_factory=dict)
    by_u: dict[int, dict[str, Any]] = field(default_factory=dict)
    by_register: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_cell: dict[str, dict[str, Any]] = field(default_factory=dict)
    recency_split: dict[str, dict[str, Any]] = field(default_factory=dict)
    error_roles: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def form(self) -> float:
        return float(self.pooled["form"])

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "n_items": self.n_items, "n_rows": self.n_rows, "incomplete": self.incomplete,
                "multi_token_rows": self.multi_token_rows, "chance": CHANCE, "pooled": self.pooled,
                "point": self.point, "ci": self.ci, "by_k": self.by_k, "by_u": self.by_u,
                "by_register": self.by_register, "by_cell": self.by_cell, "recency_split": self.recency_split,
                "error_roles": self.error_roles}


def summarize(rows: Sequence[Mapping[str, Any]], n_boot: int = BOOT, seed: int = BOOT_SEED) -> Summary:
    """One model's read-out from scored rows (``score.prompt_row`` shape). Base items missing a condition are left
    out and counted; rows with a candidate count other than 5 are refused."""
    if not rows:
        raise ValueError("no rows")
    sizes = {len(r["candidates"]) for r in rows}
    if sizes != {N_CANDIDATES}:
        raise ValueError(f"candidate counts {sorted(sizes)}, want {N_CANDIDATES}")
    groups: defaultdict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r["condition"] in groups[r["base_id"]]:
            raise ValueError(f"{r['base_id']}: two {r['condition']} rows")
        groups[r["base_id"]][r["condition"]] = r
    items: list[_Item] = []
    roles: dict[str, Counter[str]] = {c: Counter() for c in CONDITIONS}
    incomplete = 0
    for g in groups.values():
        if set(g) != set(CONDITIONS):
            incomplete += 1
            continue
        picks = {c: _pick(g[c]) for c in CONDITIONS}
        for c in CONDITIONS:
            if picks[c] != g[c]["gold"]:
                roles[c][g[c]["roles"].get(picks[c], "?")] += 1
        rc = g["C"]
        items.append(_Item((int(rc["K"]), int(rc["U"]), str(rc["register"])),
                           tuple(picks[c] == g[c]["gold"] for c in CONDITIONS),  # type: ignore[arg-type]
                           picks["C"] == rc["stale"], bool(rc["gold_is_recent"])))
    if not items:
        raise ValueError("no base item has all three conditions")
    pooled = _stats(items)
    sub = [it for it in items if not it.c_recent]
    point = _counts_stats(sum(it.ok[0] for it in items), sum(it.ok[1] for it in items),
                          sum(it.ok[2] for it in items), len(items), sum(it.ok[1] for it in sub),
                          sum(it.ok[2] for it in sub), len(sub))
    s = Summary(model=str(rows[0].get("model", "?")), n_items=len(items), n_rows=len(rows), incomplete=incomplete,
                multi_token_rows=sum(bool(r.get("multi_token")) for r in rows), pooled=pooled,
                ci=bootstrap(items, n_boot, seed), point=point,
                error_roles={c: dict(v) for c, v in roles.items()})
    for k in K_GRID:
        s.by_k[k] = _stats([it for it in items if it.cell[0] == k])
    for u in U_GRID:
        s.by_u[u] = _stats([it for it in items if it.cell[1] == u])
    for reg in REGISTERS:
        s.by_register[reg] = _stats([it for it in items if it.cell[2] == reg])
    for k, u, reg in CELLS:
        s.by_cell[cell_key(k, u, reg)] = _stats([it for it in items if it.cell == (k, u, reg)])
    s.recency_split = {"c_gold_last": _stats([it for it in items if it.c_recent]), "c_gold_not_last": _stats(sub)}
    return s


# ── predictions ──────────────────────────────────────────────────────────────────────────────────────────────────

def _need(summaries: Mapping[str, Summary], name: str) -> Summary:
    if name not in summaries:
        raise KeyError(f"no form-update summary for {name!r} (have: {', '.join(sorted(summaries))})")
    return summaries[name]


def _cmp(x: Optional[float], ref: Optional[float], op: Literal[">=", ">", "<"]) -> Optional[bool]:
    if x is None or ref is None:
        return None
    return x >= ref if op == ">=" else x > ref if op == ">" else x < ref


def predictions(summaries: Mapping[str, Summary], reference: Sequence[str], release: str,
                anneal: str) -> dict[str, Any]:
    """Four comparisons on point estimates (intervals carried beside them), ``holds`` None where a normalized cost
    is undefined (acc(A) ≤ chance) for a model the comparison needs:

      P1  the release's form acc(A) ≥ the reference set's median form
      P2  the release's normalized update cost > the reference set's median normalized update cost
      P3  the release's (normalized update − normalized interference) > the reference set's median of the same
      P4  ``anneal``'s normalized update cost < the release's

    A reference model with an undefined normalized cost is left out of that median (listed under ``undefined``)."""
    if not reference:
        raise ValueError("empty reference set")
    if release in reference or anneal in reference:
        raise ValueError("the release and the anneal checkpoint are compared against the reference set, not in it")
    ref = {m: _need(summaries, m) for m in reference}
    rel, ann = _need(summaries, release), _need(summaries, anneal)

    def one(stat: str, op: Literal[">=", ">"]) -> dict[str, Any]:
        vals = {m: s.point[stat] for m, s in ref.items()}
        defined = {m: v for m, v in vals.items() if v is not None}
        med = statistics.median(defined.values()) if defined else None
        x = rel.point[stat]
        return {"stat": stat, "release": x, "release_ci": rel.ci[stat]["ci"], "reference": vals,
                "undefined": sorted(m for m, v in vals.items() if v is None), "median": med, "op": op,
                "holds": _cmp(x, med, op)}

    p4x, p4r = ann.point["norm_update"], rel.point["norm_update"]
    return {
        "reference": list(reference), "release": release, "anneal": anneal,
        "P1": one("form", ">="),
        "P2": one("norm_update", ">"),
        "P3": one("norm_update_minus_interference", ">"),
        "P4": {"stat": "norm_update", "anneal": p4x, "anneal_ci": ann.ci["norm_update"]["ci"], "release": p4r,
               "release_ci": rel.ci["norm_update"]["ci"], "op": "<", "holds": _cmp(p4x, p4r, "<")},
    }
