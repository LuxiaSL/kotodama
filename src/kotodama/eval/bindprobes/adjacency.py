"""``adjacency`` (battery v3) — STRUCTURAL "you" at P = 3: resolved by reply adjacency.

A 3-person chat; besides v2 declarations and vocative gives, objects move through
REQUEST -> GRANT pairs whose grant carries NO vocative::

    A chat between Ana, Ben and Cal.
    Ana: I have the cup.
    Cal: The lamp is mine.
    Ben: Can I have the cup?
    Ana: Sure, the cup is yours.
    Now the cup is with
    -> " Ben"

v2's unambiguity law forbids a vocative-free "you" at P = 3 (no derivable gold); here
the gold IS derivable — a grant answers the request in the line right before it, so
"yours"/"take the o" = the requester (``check_surface`` enforces the adjacency). This is
the structural channel the lineage never acquired (405B .22 shortcut-free on vocative-
free addressee items, records/t19-readout §4).

Cell = (E, n_grant): (1,1), (2,1), (2,2) — E transfers, n_grant of them grants (the
rest v2 vocative gives, rotated surfaces).
Paired variants (``meta.pair_id``):
  adj    grants without a vocative (the target)
  voc    identical events, each grant carrying the vocative (``Sure, Ben, the cup is yours.``)
  third  prose: ``Ben asks Ana for the cup. Ana gives the cup to Ben.``
Query = the narrator owner query (``Now the cup is with``; the lite query; 3 names).
Schedule (deixis v2 owner): gold = the last transfer's recipient / its giver / the
bystander, 1/3 each; E = 2: the queried object moved and is not in the last transfer.
``meta.critical`` = the queried object's last move was a grant (the adjacency read).
Readouts: structural penalty = acc(voc) − acc(adj) [critical subset primary];
deixis penalty = acc(third) − acc(voc).
Error types: ``granter`` (the grant's speaker — "yours" read as the speaker's),
``previous_owner``, ``other_participant``.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence, TypedDict

from kotodama.eval.bindprobes import deixis as dx
from kotodama.eval.bindprobes.base import Cell, Item, name_list, rejection

FAMILY = "adjacency"
VERSION = 1
CELLS: list[Cell] = [(1, 1), (2, 1), (2, 2)]
DEFAULT_PER_CELL = 150            # 450 base, 1350 with variants (V3-5: critical-subset paired SE ~.03)
SEPARATOR = dx.SEPARATOR
VARIANTS = ("adj", "voc", "third")
N_PEOPLE = 3
N_OBJECTS = (2, 3, 4)
ERROR_TYPES = ("granter", "previous_owner", "other_participant")
MAX_PROMPT_TOKENS = 1024
BALANCE_TOL = 0.05
REQUEST = "Can I have the {o}?"
# (no vocative, with vocative {v}). The grant NAMES the object, as every v2 give line
# does: otherwise "the speaker of the last line naming the object" = the requester =
# gold on every critical item, a shortcut that skips resolving "yours"/"take".
GRANT_FORMS: dict[str, tuple[str, str]] = {
    "yours": ("Sure, the {o} is yours.", "Sure, {v}, the {o} is yours."),
    "take": ("Sure, take the {o}.", "Sure, {v}, take the {o}."),
}
_OWNER_SCHED = ("recipient", "giver", "bystander")


class Spec(TypedDict):
    variant: str
    people: list[str]
    objects: list[str]
    events: list[dict[str, Any]]   # declare {speaker,obj,form} | give {speaker,voc,obj,form} | grant {requester,granter,obj,form}
    query: dict[str, Any]          # {"obj"}


# ── Simulation ─────────────────────────────────────────────────────────────────

def transfers(spec: Spec) -> list[dict[str, Any]]:
    """Every transfer as {kind, giver, recipient, obj}."""
    out = []
    for ev in spec["events"]:
        if ev["ev"] == "give":
            out.append({"kind": "give", "giver": ev["speaker"],
                        "recipient": dx.resolve_addressee(spec["people"], ev["speaker"], ev.get("voc")),
                        "obj": ev["obj"]})
        elif ev["ev"] == "grant":
            if ev["requester"] == ev["granter"]:
                raise ValueError("a grant to oneself")
            out.append({"kind": "grant", "giver": ev["granter"], "recipient": ev["requester"],
                        "obj": ev["obj"]})
    return out


def run(spec: Spec) -> tuple[dict[str, str], dict[str, list[str]]]:
    people = spec["people"]
    if len(people) != N_PEOPLE or len(set(people)) != N_PEOPLE:
        raise ValueError(f"adjacency: need {N_PEOPLE} distinct people")
    owner: dict[str, str] = {}
    hist: dict[str, list[str]] = {}
    tr = iter(transfers(spec))
    seen = False
    for k, ev in enumerate(spec["events"]):
        if ev["ev"] == "declare":
            if seen or ev["obj"] in owner or ev["obj"] not in spec["objects"]:
                raise ValueError(f"event {k}: bad declaration")
            owner[ev["obj"]] = ev["speaker"]
            hist[ev["obj"]] = [ev["speaker"]]
            continue
        seen = True
        t = next(tr)
        if owner.get(t["obj"]) != t["giver"]:
            raise ValueError(f"event {k}: {t['giver']} does not own the {t['obj']}")
        owner[t["obj"]] = t["recipient"]
        hist[t["obj"]].append(t["recipient"])
    if set(owner) != set(spec["objects"]):
        raise ValueError("undeclared objects")
    return owner, hist


def simulate(item: Item) -> str:
    owner, _ = run(item.spec)  # type: ignore[arg-type]
    return owner[item.spec["query"]["obj"]]


# ── Rendering ──────────────────────────────────────────────────────────────────

def _lines(spec: Spec) -> list[str]:
    out: list[str] = []
    for ev in spec["events"]:
        if ev["ev"] in ("declare", "give"):
            out.append(dx._deictic_line(ev)[0])
            continue
        r, g = ev["requester"], ev["granter"]
        bare, with_v = GRANT_FORMS[ev["form"]]
        out.append(f"{r}: " + REQUEST.format(o=ev["obj"]))
        out.append(f"{g}: " + (with_v.format(v=r, o=ev["obj"]) if spec["variant"] == "voc"
                               else bare.format(o=ev["obj"])))
    return out


def _third(spec: Spec) -> list[str]:
    out: list[str] = []
    expl: list[dict[str, Any]] = []
    for ev in spec["events"]:
        if ev["ev"] == "declare":
            expl.append({"ev": "declare", "owner": ev["speaker"], "obj": ev["obj"]})
        elif ev["ev"] == "give":
            rec = dx.resolve_addressee(spec["people"], ev["speaker"], ev.get("voc"))
            expl.append({"ev": "give", "giver": ev["speaker"], "recipient": rec, "obj": ev["obj"]})
        else:
            expl.append({"ev": "ask", "requester": ev["requester"], "granter": ev["granter"],
                         "obj": ev["obj"]})
            expl.append({"ev": "give", "giver": ev["granter"], "recipient": ev["requester"],
                         "obj": ev["obj"]})
    k = 0
    while k < len(expl):
        ev = expl[k]
        if ev["ev"] == "ask":
            out.append(f"{ev['requester']} asks {ev['granter']} for the {ev['obj']}.")
        else:
            out += [t for t, _ in dx._third_sentences([ev])] if ev["ev"] == "give" else []
            if ev["ev"] == "declare":
                objs = [ev["obj"]]
                while k + 1 < len(expl) and expl[k + 1]["ev"] == "declare" \
                        and expl[k + 1]["owner"] == ev["owner"]:
                    k += 1
                    objs.append(expl[k]["obj"])
                out += [t for t, _ in dx._third_sentences(
                    [{"ev": "declare", "owner": ev["owner"], "obj": o} for o in objs])]
        k += 1
    return out


def _rendered(spec: Spec) -> str:
    people = spec["people"]
    lead = f"Now the {spec['query']['obj']} is with"
    if spec["variant"] == "third":
        return " ".join([f"{name_list(people)} are sitting at a table."] + _third(spec) + [lead])
    return "\n".join([f"A chat between {name_list(people)}."] + _lines(spec) + [lead])


def render(item: Item, with_answer: bool) -> str:
    text = _rendered(item.spec)  # type: ignore[arg-type]
    return f"{text} {item.gold}." if with_answer else text


def labels(item: Item) -> list[str]:
    return list(item.spec["people"])


def check_surface(item: Item) -> None:
    """adj: every vocative-free addressed line is a grant whose preceding line is the
    matching request by another participant; every other addressed line has a valid
    vocative. voc: every addressed line has one. third: pronoun-free prose."""
    spec: Spec = item.spec  # type: ignore[assignment]
    text = _rendered(spec)
    if "(" in text or ")" in text:
        raise ValueError(f"{item.item_id}: parentheses")
    if spec["variant"] == "third":
        if "\n" in text or dx._PRONOUNS.search(text):
            raise ValueError(f"{item.item_id}: third is not pronoun-free prose")
        return
    body = text.split("\n")[1:-1]
    for i, ln in enumerate(body):
        m = re.match(r"^(\w+): (.*)$", ln)
        if not m or m.group(1) not in spec["people"]:
            raise ValueError(f"{item.item_id}: not a 'Name: utterance' line: {ln!r}")
        spk, utt = m.groups()
        rest = re.sub(r"^(?:Here|Sure), ", "", utt)
        v = re.match(r"^(\w+), ", rest)
        addressed = bool(dx._YOU.search(utt) or re.search(r"\btake\b", utt))
        if not addressed:
            continue
        if v:
            if v.group(1) not in spec["people"] or v.group(1) == spk:
                raise ValueError(f"{item.item_id}: bad vocative: {ln!r}")
            continue
        if spec["variant"] != "adj":
            raise ValueError(f"{item.item_id}: addressed line without a vocative in {spec['variant']}: {ln!r}")
        prev = re.match(r"^(\w+): Can I have the (\w+)\?$", body[i - 1]) if i else None
        if not prev or prev.group(1) == spk:
            raise ValueError(f"{item.item_id}: vocative-free grant not adjacent to a request: {ln!r}")


# ── Items ──────────────────────────────────────────────────────────────────────

def build(item_id: str, cell: Cell, spec: Spec, pair_id: Optional[str] = None) -> Item:
    people = list(spec["people"])
    text = _rendered(spec)
    pat = re.compile(r"\b(" + "|".join(map(re.escape, people)) + r")\b")
    it = Item(item_id=item_id, family=FAMILY, cell=list(cell), spec=dict(spec), gold="",
              candidates=people, mention_order=pat.findall(text),
              meta={"adjacency_version": VERSION, "variant": spec["variant"],
                    "pair_id": pair_id or item_id})
    it.gold = simulate(it)
    tr = transfers(spec)
    _, hist = run(spec)
    o = spec["query"]["obj"]
    last_move = [t for t in tr if t["obj"] == o]
    it.meta.update({
        "n_transfers": len(tr), "n_grant": sum(t["kind"] == "grant" for t in tr),
        "gold_role": ("recipient" if it.gold == tr[-1]["recipient"] else
                      "giver" if it.gold == tr[-1]["giver"] else "bystander"),
        "critical": bool(last_move) and last_move[-1]["kind"] == "grant",
        "queried_obj_moved": len(hist[o]) > 1,
        "queried_obj_in_last_transfer": tr[-1]["obj"] == o,
        "gold_is_last_mention": bool(it.mention_order) and it.mention_order[-1] == it.gold,
    })
    return it


def _draw(rng: random.Random, cell: Cell, pools: Any, item_id: str,
          template: Optional[Item], want: Optional[str]) -> Optional[Item]:
    e, n_grant = int(cell[0]), int(cell[1])
    people = rng.sample(pools.names, N_PEOPLE)
    objs0 = rng.sample(pools.objects, rng.choice(N_OBJECTS))
    first = {o: rng.choice(people) for o in objs0}
    objs = [o for x in people for o in objs0 if first[o] == x]
    owner = dict(first)
    events: list[dict[str, Any]] = [{"ev": "declare", "speaker": owner[o], "obj": o,
                                     "form": rng.choice(tuple(dx.DECLARE_FORMS))} for o in objs]
    kinds = ["grant"] * n_grant + ["give"] * (e - n_grant)
    rng.shuffle(kinds)
    start = rng.randrange(len(dx.GIVE_ORDER))
    for i, kind in enumerate(kinds):
        g = rng.choice([x for x in people if any(owner[o] == x for o in objs)])
        r = rng.choice([x for x in people if x != g])
        o = rng.choice([o for o in objs if owner[o] == g])
        if kind == "grant":
            events.append({"ev": "grant", "requester": r, "granter": g, "obj": o,
                           "form": rng.choice(tuple(GRANT_FORMS))})
        else:
            events.append({"ev": "give", "speaker": g, "voc": r, "obj": o,
                           "form": dx.GIVE_ORDER[(start + i) % len(dx.GIVE_ORDER)]})
        owner[o] = r
    spec: Spec = {"variant": "adj", "people": people, "objects": objs, "events": events,
                  "query": {}}
    tr = transfers(spec)
    last = tr[-1]
    pool = list(objs)
    if e >= 2:
        moved = {t["obj"] for t in tr}
        pool = [o for o in pool if o in moved and o != last["obj"]]
    role = want if want is not None else rng.choice(_OWNER_SCHED)
    target = {"recipient": last["recipient"], "giver": last["giver"]}.get(role)
    if target is None:
        target = next(x for x in people if x not in (last["recipient"], last["giver"]))
    pool = [o for o in pool if owner[o] == target]
    if not pool:
        return None
    # prefer the adjacency-critical object when the schedule allows it (keeps the critical n up)
    crit = [o for o in pool if [t for t in tr if t["obj"] == o] and
            [t for t in tr if t["obj"] == o][-1]["kind"] == "grant"]
    o = rng.choice(crit) if crit and rng.random() < 0.75 else rng.choice(pool)
    spec["query"] = {"obj": o}
    it = build(item_id, cell, spec)
    if template is not None and people.index(it.gold) == template.spec["people"].index(template.gold):
        return None
    return it


def gen_item(rng: random.Random, cell: Cell, pools: Any, item_id: str, k: int,
             per_cell: int, template: Optional[Item] = None) -> Item:
    if tuple(cell) not in CELLS:
        raise ValueError(f"adjacency: unknown cell {cell}")
    cell = tuple(cell)
    want = _OWNER_SCHED[k % len(_OWNER_SCHED)] if template is None else None
    return rejection(lambda: _draw(rng, cell, pools, item_id, template, want),
                     f"adjacency item {item_id}", tries=20000)


def expand(base: Item) -> list[Item]:
    pid = base.item_id
    shots = base.shot_items()
    out = []
    for v in VARIANTS:
        it = build(f"{pid}_{v}", tuple(base.cell), {**base.spec, "variant": v}, pair_id=pid)  # type: ignore[typeddict-item]
        it.meta["shots_disjoint"] = base.meta.get("shots_disjoint")
        it.shots = [build(f"{s.item_id}_{v}", tuple(s.cell), {**s.spec, "variant": v},  # type: ignore[typeddict-item]
                          pair_id=s.item_id).to_dict() for s in shots]
        it.collide_maps = [dict(m) for m in base.collide_maps]
        out.append(it)
    return out


_NAME_KEYS = ("speaker", "voc", "requester", "granter")


def relabel(item: Item, mapping: Mapping[str, str], target: Optional[Item] = None) -> Item:
    spec: Spec = item.spec  # type: ignore[assignment]
    if set(spec["people"]) - set(mapping):
        raise ValueError(f"{item.item_id}: people missing from the map")
    m = dict(mapping)
    new: Spec = {"variant": spec["variant"], "people": [m[p] for p in spec["people"]],
                 "objects": list(spec["objects"]),
                 "events": [{k: (m.get(v, v) if k in _NAME_KEYS and isinstance(v, str) else v)
                             for k, v in ev.items()} for ev in spec["events"]],
                 "query": dict(spec["query"])}
    if target is not None and sorted(new["people"]) == sorted(target.spec["people"]):
        new["people"] = list(target.spec["people"])
    out = build(f"{item.item_id}_collide", tuple(item.cell), new,
                pair_id=str(item.meta.get("pair_id", item.item_id)))
    if out.gold != m[item.gold]:
        raise ValueError(f"{item.item_id}: re-label changed the gold")
    return out


# ── Validation / audit ─────────────────────────────────────────────────────────

def validate_item(item: Item) -> None:
    e, n_grant = item.cell
    if item.meta.get("adjacency_version") != VERSION:
        raise ValueError(f"{item.item_id}: adjacency_version mismatch")
    if item.meta["n_transfers"] != e or item.meta["n_grant"] != n_grant:
        raise ValueError(f"{item.item_id}: spec does not match cell {item.cell}")
    if e >= 2 and (not item.meta["queried_obj_moved"] or item.meta["queried_obj_in_last_transfer"]):
        raise ValueError(f"{item.item_id}: E>=2 queried object must have moved, not last")
    check_surface(item)


def check_family(items: Sequence[Item]) -> None:
    by: defaultdict[str, dict[str, Item]] = defaultdict(dict)
    for it in items:
        if it.family == FAMILY:
            by[it.meta["pair_id"]][it.meta["variant"]] = it
    for pid, g in by.items():
        if set(g) != set(VARIANTS):
            raise ValueError(f"pair {pid}: variants {sorted(g)}")
        if len({x.gold for x in g.values()}) != 1 or len({x.meta["critical"] for x in g.values()}) != 1:
            raise ValueError(f"pair {pid}: gold/critical differ across variants")


def audit(items: Sequence[Item]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for c in CELLS:
        sub = [it for it in items if it.family == FAMILY and tuple(it.cell) == c
               and it.meta["variant"] == "adj"]
        if not sub:
            continue
        n = len(sub)
        out["/".join(map(str, c))] = {
            "n": n, "chance": 1 / 3,
            "gold_recipient": sum(it.meta["gold_role"] == "recipient" for it in sub) / n,
            "gold_giver": sum(it.meta["gold_role"] == "giver" for it in sub) / n,
            "gold_is_last_mention": sum(it.meta["gold_is_last_mention"] for it in sub) / n,
            "critical": sum(it.meta["critical"] for it in sub) / n}
    return out


def audit_failures(aud: Mapping[str, Mapping[str, Any]], tol: float = BALANCE_TOL) -> list[str]:
    bad = []
    for ck, a in aud.items():
        band = max(tol, 1.0 / a["n"]) + 1e-9
        for key in ("gold_recipient", "gold_giver", "gold_is_last_mention"):
            if abs(a[key] - a["chance"]) > band:
                bad.append(f"{ck}: {key} {a[key]:.3f} vs chance {a['chance']:.3f}")
    return bad


def error_types(item: Item, pick: str) -> list[str]:
    spec: Spec = item.spec  # type: ignore[assignment]
    o = spec["query"]["obj"]
    _, hist = run(spec)
    moves = [t for t in transfers(spec) if t["obj"] == o]
    if moves and moves[-1]["kind"] == "grant" and pick == moves[-1]["giver"]:
        return ["granter"]
    if pick in set(hist[o][:-1]) - {item.gold}:
        return ["previous_owner"]
    return ["other_participant"]
