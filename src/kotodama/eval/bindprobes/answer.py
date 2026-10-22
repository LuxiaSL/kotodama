"""``answer`` (battery v3) — PRODUCTION-side deixis: answer "who has it?" from a perspective.

A 4-person chat transcript whose body is deixis v2 (declarations + gives, every
you/imperative line opened by a vocative), ending in an in-register question/answer
pair; the answerer must name the holder AS THEY WOULD SAY IT::

    A chat between Ana, Ben, Cal and Dee.
    Ana: I have the cup.
    Cal: The lamp is mine.
    Ana: Here, Ben, take my cup.
    Dee: Who has the cup now?
    Ben: I think it's with
    -> " me"      (holder = the answerer)   / " you" (= the asker) / " Cal" (anyone else)

Cell = (E, channel): E ∈ {1, 2} gives; channel = how the answer's "you" is anchored:
  novoc  ``Ben: I think it's with``       — "you" = the asker by adjacency (structural)
  voc    ``Ben: Dee, I think it's with``  — a vocative names the asker
Paired variants of every base item (``meta.pair_id``):
  qa     the above; candidates [me, you, X, Y] (X, Y = the two others, scene order)
  narr   the same body + the narrator query ``Now the cup is with`` -> the holder's name
         (candidates = the 4 names) — the lite/form-a query, i.e. the seam test partner
  third  the same events as dialogue-family prose with names (no pronouns), then the
         same Q-A tail -> the same gold as qa
  null   CALIBRATION: the scene line + the same Q-A tail with the body HIDDEN (exemplars
         unchanged) — no information about the holder. Its per-candidate logprobs are the
         context prior of the tail ("it's with" -> " you"/" me" are frequent phrases);
         the qa/third READ is argmax(logp(item) − logp(null)) (contextual calibration).
         Its own "accuracy" is meaningless (gold = the qa gold, by construction).
Gold ROLE schedule (cycled, per cell): me ¼, you ¼, other ½ (each name ¼) — every
constant answer is at chance (1/4). Independently cycled: the holder is the last
body line's recipient ¼ / its speaker ¼ / neither ½ (names-level recency + last-
speaker controls at chance). E = 2: the queried object moved and is not in the last
body line (the v2 ownership-word rule).

Error types: ``perspective_flip`` (me<->you swapped), ``name_for_deixis`` (the holder
is the answerer/asker but a name was picked), ``deixis_for_name`` (holder is another
person but me/you was picked), ``previous_owner``, ``other``.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence, TypedDict

from kotodama.eval.bindprobes import deixis as dx
from kotodama.eval.bindprobes.base import Cell, Item, name_list, rejection

FAMILY = "answer"
VERSION = 2                       # v2 adds the null calibration variant
ES = (1, 2)
CHANNELS = ("novoc", "voc")
CELLS: list[Cell] = [(e, ch) for e in ES for ch in CHANNELS]
DEFAULT_PER_CELL = 160            # 16-cycle x 10 -> 640 base, 1920 with variants (role-contrast SE ~.039)
SEPARATOR = dx.SEPARATOR
VARIANTS = ("qa", "narr", "third", "null")
N_PEOPLE = 4
N_OBJECTS = (3, 4)
ME, YOU = "me", "you"
# cycled: each constant answer at chance; otherA/otherB = the holder is the first/second
# of the two other people in scene order (= candidate slot 3/4)
ROLES = ("me", "you", "otherA", "otherB")
RECENCY = ("recipient", "speaker", "neither", "neither")
ERROR_TYPES = ("perspective_flip", "name_for_deixis", "deixis_for_name", "previous_owner", "other")
MAX_PROMPT_TOKENS = 1024
BALANCE_TOL = 0.05
QUESTION = "Who has the {o} now?"
ANSWER_LEAD = "I think it's with"


class Spec(TypedDict):
    variant: str
    people: list[str]
    objects: list[str]
    events: list[dict[str, Any]]     # deixis v2 surface events (qa/narr) or explicit ones (third)
    tail: dict[str, Any]             # {"obj", "asker", "answerer", "channel"}


# ── Simulation ─────────────────────────────────────────────────────────────────

def explicit_events(spec: Spec) -> list[dict[str, Any]]:
    people = spec["people"]
    if len(people) != N_PEOPLE or len(set(people)) != N_PEOPLE:
        raise ValueError(f"answer: need {N_PEOPLE} distinct people, got {people}")
    if spec["variant"] == "third":
        return [dict(ev) for ev in spec["events"]]
    out: list[dict[str, Any]] = []
    for k, ev in enumerate(spec["events"]):
        if ev["ev"] == "declare":
            out.append({"ev": "declare", "owner": ev["speaker"], "obj": ev["obj"]})
        elif ev["ev"] == "give":
            rec = dx.resolve_addressee(people, ev["speaker"], ev.get("voc"))
            out.append({"ev": "give", "giver": ev["speaker"], "recipient": rec, "obj": ev["obj"]})
        else:
            raise ValueError(f"event {k}: unknown event {ev['ev']!r}")
    return out


def run(spec: Spec) -> tuple[dict[str, str], dict[str, list[str]]]:
    owner: dict[str, str] = {}
    hist: dict[str, list[str]] = {}
    seen_give = False
    for k, ev in enumerate(explicit_events(spec)):
        o = ev["obj"]
        if ev["ev"] == "declare":
            if seen_give or o in owner or o not in spec["objects"]:
                raise ValueError(f"event {k}: bad declaration of the {o}")
            owner[o] = ev["owner"]
            hist[o] = [ev["owner"]]
            continue
        seen_give = True
        if owner.get(o) != ev["giver"] or ev["giver"] == ev["recipient"]:
            raise ValueError(f"event {k}: {ev['giver']} cannot give the {o}")
        owner[o] = ev["recipient"]
        hist[o].append(ev["recipient"])
    if set(owner) != set(spec["objects"]):
        raise ValueError("undeclared objects")
    return owner, hist


def holder(spec: Spec) -> str:
    owner, _ = run(spec)
    o = spec["tail"]["obj"]
    if o not in owner:
        raise ValueError(f"queried object {o!r} not in play")
    return owner[o]


def role_of(spec: Spec) -> str:
    h, t = holder(spec), spec["tail"]
    if t["asker"] == t["answerer"] or not {t["asker"], t["answerer"]} <= set(spec["people"]):
        raise ValueError(f"bad asker/answerer {t}")
    return "me" if h == t["answerer"] else "you" if h == t["asker"] else "other"


def others(spec: Spec) -> list[str]:
    t = spec["tail"]
    return [p for p in spec["people"] if p not in (t["asker"], t["answerer"])]


def last_line(spec: Spec) -> dict[str, Any]:
    """The last BODY line as (speaker, addressee, objs) — before the tail."""
    ev = explicit_events(spec)[-1]
    if ev["ev"] == "declare":
        return {"speaker": ev["owner"], "addressee": None, "objs": [ev["obj"]]}
    return {"speaker": ev["giver"], "addressee": ev["recipient"], "objs": [ev["obj"]]}


def simulate(item: Item) -> str:
    spec: Spec = item.spec  # type: ignore[assignment]
    h = holder(spec)
    if spec["variant"] == "narr":
        return h
    r = role_of(spec)
    return ME if r == "me" else YOU if r == "you" else h


def candidates(spec: Spec) -> list[str]:
    return list(spec["people"]) if spec["variant"] == "narr" else [ME, YOU] + others(spec)


# ── Rendering ──────────────────────────────────────────────────────────────────

def _tail_lines(spec: Spec) -> list[str]:
    t = spec["tail"]
    q = f"{t['asker']}: " + QUESTION.format(o=t["obj"])
    lead = f"{t['answerer']}: " + (f"{t['asker']}, " if t["channel"] == "voc" else "") + ANSWER_LEAD
    return [q, lead]


def _rendered(spec: Spec) -> str:
    people = spec["people"]
    if spec["variant"] == "third":
        prose = [f"{name_list(people)} are sitting at a table."]
        prose += [t for t, _ in dx._third_sentences(spec["events"])]
        return " ".join(prose) + "\n" + "\n".join(_tail_lines(spec))
    lines = [f"A chat between {name_list(people)}."]
    lines += [dx._deictic_line(ev)[0] for ev in spec["events"]]
    if spec["variant"] == "null":
        return "\n".join([lines[0]] + _tail_lines(spec))
    if spec["variant"] == "narr":
        lines.append(f"Now the {spec['tail']['obj']} is with")
    else:
        lines += _tail_lines(spec)
    return "\n".join(lines)


def render(item: Item, with_answer: bool) -> str:
    text = _rendered(item.spec)  # type: ignore[arg-type]
    return f"{text} {item.gold}." if with_answer else text


def labels(item: Item) -> list[str]:
    return list(item.spec["people"])


def _mentions(text: str, cands: Sequence[str]) -> list[str]:
    pat = re.compile(r"\b(" + "|".join(map(re.escape, cands)) + r")\b")
    return pat.findall(text)


def check_surface(item: Item) -> None:
    """deixis v2 law on the body (vocative on every you/imperative line at P = 4); the
    tail's question carries no "you"; third: the prose paragraph is pronoun-free."""
    spec: Spec = item.spec  # type: ignore[assignment]
    text = _rendered(spec)
    if "(" in text or ")" in text:
        raise ValueError(f"{item.item_id}: parentheses")
    if spec["variant"] == "third":
        prose = text.split("\n")[0]
        if dx._PRONOUNS.search(prose):
            raise ValueError(f"{item.item_id}: pronoun in the third prose")
        return
    body = text.split("\n")[1:]
    body = body[:-1] if spec["variant"] == "narr" else body[:-2]
    if spec["variant"] == "null" and body:
        raise ValueError(f"{item.item_id}: the null variant must hide the body")
    for ln in body:
        m = re.match(r"^(\w+): (.*)$", ln)
        if not m or m.group(1) not in spec["people"]:
            raise ValueError(f"{item.item_id}: not a 'Name: utterance' line: {ln!r}")
        spk, utt = m.groups()
        rest = dx._HERE.sub("", utt)
        v = re.match(r"^(\w+), ", rest)
        if (dx._YOU.search(utt) or dx._IMPERATIVE.match(rest)) and \
                (not v or v.group(1) not in spec["people"] or v.group(1) == spk):
            raise ValueError(f"{item.item_id}: addressed line without a valid vocative: {ln!r}")
    if dx._YOU.search(QUESTION):
        raise ValueError("the question must not contain 'you'")


# ── Items ──────────────────────────────────────────────────────────────────────

def build(item_id: str, cell: Cell, spec: Spec, pair_id: Optional[str] = None) -> Item:
    if spec["variant"] not in VARIANTS:
        raise ValueError(f"{item_id}: unknown variant {spec['variant']!r}")
    cands = candidates(spec)
    it = Item(item_id=item_id, family=FAMILY, cell=list(cell), spec=dict(spec), gold="",
              candidates=cands, mention_order=_mentions(_rendered(spec), cands),
              meta={"answer_version": VERSION, "variant": spec["variant"],
                    "pair_id": pair_id or item_id, "channel": spec["tail"]["channel"]})
    it.gold = simulate(it)
    h = holder(spec)
    last = last_line(spec)
    _, hist = run(spec)
    it.meta.update({"role": role_of(spec), "holder": h,
                    "holder_is_last_recipient": h == last["addressee"],
                    "holder_is_last_speaker": h == last["speaker"],
                    "queried_obj_moved": len(hist[spec["tail"]["obj"]]) > 1,
                    "queried_obj_in_last_line": spec["tail"]["obj"] in last["objs"],
                    "n_transfers": sum(1 for ev in spec["events"] if ev["ev"] == "give")})
    return it


def _draw(rng: random.Random, cell: Cell, pools: Any, item_id: str,
          template: Optional[Item], want: Optional[tuple[str, str]]) -> Optional[Item]:
    e, channel = int(cell[0]), str(cell[1])
    people = rng.sample(pools.names, N_PEOPLE)
    objs0 = rng.sample(pools.objects, rng.choice(N_OBJECTS))
    first = {o: rng.choice(people) for o in objs0}
    objs = [o for x in people for o in objs0 if first[o] == x]
    owner = dict(first)
    events: list[dict[str, Any]] = [{"ev": "declare", "speaker": owner[o], "obj": o,
                                     "form": rng.choice(tuple(dx.DECLARE_FORMS))} for o in objs]
    start = rng.randrange(len(dx.GIVE_ORDER))
    moved: list[str] = []
    for i in range(e):
        s = rng.choice([x for x in people if any(owner[o] == x for o in objs)])
        a = rng.choice([x for x in people if x != s])
        o = rng.choice([o for o in objs if owner[o] == s])
        events.append({"ev": "give", "speaker": s, "voc": a, "obj": o,
                       "form": dx.GIVE_ORDER[(start + i) % len(dx.GIVE_ORDER)]})
        owner[o] = a
        moved.append(o)
    last_give = events[-1]
    pool = list(objs)
    if e >= 2:
        pool = [o for o in moved if o != last_give["obj"]]
    role, rec = want if want is not None else (rng.choice(ROLES), rng.choice(RECENCY))
    if role not in ROLES or rec not in RECENCY:
        raise ValueError(f"bad schedule slot {(role, rec)}")
    if rec == "recipient":
        pool = [o for o in pool if owner[o] == last_give["voc"]]
    elif rec == "speaker":
        pool = [o for o in pool if owner[o] == last_give["speaker"]]
    else:
        pool = [o for o in pool if owner[o] not in (last_give["voc"], last_give["speaker"])]
    if not pool:
        return None
    o = rng.choice(pool)
    h = owner[o]
    if role == "me":
        answerer, asker = h, rng.choice([x for x in people if x != h])
    elif role == "you":
        asker, answerer = h, rng.choice([x for x in people if x != h])
    else:
        rest = [x for x in people if x != h]
        asker, answerer = rng.sample(rest, 2)
        if role in ("otherA", "otherB"):
            oth = [x for x in people if x not in (asker, answerer)]
            if oth.index(h) != (0 if role == "otherA" else 1):
                return None
    spec: Spec = {"variant": "qa", "people": people, "objects": objs, "events": events,
                  "tail": {"obj": o, "asker": asker, "answerer": answerer, "channel": channel}}
    it = build(item_id, cell, spec)
    if template is not None:
        # the collide re-label is positional: an exemplar must not answer the test gold
        # (every variant: narr answers the holder's name, qa/third the role-mapped form)
        tg, tp = template.gold, template.spec["people"]
        if people.index(it.meta["holder"]) == tp.index(template.meta["holder"]):
            return None
        if it.gold in (ME, YOU) and it.gold == tg:
            return None
    return it


def gen_item(rng: random.Random, cell: Cell, pools: Any, item_id: str, k: int,
             per_cell: int, template: Optional[Item] = None) -> Item:
    if tuple(cell) not in CELLS:
        raise ValueError(f"answer: unknown cell {cell}")
    cell = tuple(cell)
    want = (ROLES[k % len(ROLES)], RECENCY[(k // len(ROLES)) % len(RECENCY)]) \
        if template is None else None
    return rejection(lambda: _draw(rng, cell, pools, item_id, template, want),
                     f"answer item {item_id}", tries=20000)


def _variant_spec(spec: Spec, variant: str) -> Spec:
    if variant == "qa":
        return spec
    if variant in ("narr", "null"):
        return {**spec, "variant": variant}
    return {**spec, "variant": "third", "events": explicit_events(spec)}


def expand(base: Item) -> list[Item]:
    pid = base.item_id
    shots = base.shot_items()
    out: list[Item] = []
    for v in VARIANTS:
        it = build(f"{pid}_{v}", tuple(base.cell), _variant_spec(base.spec, v), pair_id=pid)  # type: ignore[arg-type]
        it.meta["shots_disjoint"] = base.meta.get("shots_disjoint")
        # null calibrates the qa context exactly: the SAME (qa) exemplars, only the test body hidden
        sv = "qa" if v == "null" else v
        it.shots = [build(f"{s.item_id}_{sv}", tuple(s.cell), _variant_spec(s.spec, sv),  # type: ignore[arg-type]
                          pair_id=s.item_id).to_dict() for s in shots]
        it.collide_maps = [dict(m) for m in base.collide_maps]
        out.append(it)
    return out


_NAME_KEYS = ("speaker", "voc", "owner", "giver", "recipient")


def relabel(item: Item, mapping: Mapping[str, str], target: Optional[Item] = None) -> Item:
    spec: Spec = item.spec  # type: ignore[assignment]
    if set(spec["people"]) - set(mapping):
        raise ValueError(f"{item.item_id}: people missing from the map")
    m = dict(mapping)
    ren = lambda d: {k: (m.get(v, v) if k in _NAME_KEYS and isinstance(v, str) else v)
                     for k, v in d.items()}
    t = spec["tail"]
    new: Spec = {"variant": spec["variant"], "people": [m[p] for p in spec["people"]],
                 "objects": list(spec["objects"]), "events": [ren(ev) for ev in spec["events"]],
                 "tail": {**t, "asker": m[t["asker"]], "answerer": m[t["answerer"]]}}
    if target is not None and sorted(new["people"]) == sorted(target.spec["people"]):
        new["people"] = list(target.spec["people"])
    out = build(f"{item.item_id}_collide", tuple(item.cell), new,
                pair_id=str(item.meta.get("pair_id", item.item_id)))
    expect = item.gold if item.gold in (ME, YOU) else m.get(item.gold, item.gold)
    if out.gold != expect:
        raise ValueError(f"{item.item_id}: re-label changed the gold")
    return out


# ── Validation / audit ─────────────────────────────────────────────────────────

def validate_item(item: Item) -> None:
    spec: Spec = item.spec  # type: ignore[assignment]
    e, channel = item.cell
    if item.meta.get("answer_version") != VERSION:
        raise ValueError(f"{item.item_id}: answer_version mismatch")
    if item.meta["n_transfers"] != e or spec["tail"]["channel"] != channel:
        raise ValueError(f"{item.item_id}: spec does not match cell {item.cell}")
    if e >= 2 and (not item.meta["queried_obj_moved"] or item.meta["queried_obj_in_last_line"]):
        raise ValueError(f"{item.item_id}: E>=2 queried object must have moved, not in the last line")
    if len(item.candidates) != 4:
        raise ValueError(f"{item.item_id}: {len(item.candidates)} candidates")
    check_surface(item)


def check_family(items: Sequence[Item]) -> None:
    by: defaultdict[str, dict[str, Item]] = defaultdict(dict)
    for it in items:
        if it.family == FAMILY:
            by[it.meta["pair_id"]][it.meta["variant"]] = it
    for pid, g in by.items():
        if set(g) != set(VARIANTS):
            raise ValueError(f"pair {pid}: variants {sorted(g)}")
        q, n, t, z = g["qa"], g["narr"], g["third"], g["null"]
        if not (q.meta["holder"] == n.meta["holder"] == t.meta["holder"] == z.meta["holder"] == n.gold):
            raise ValueError(f"pair {pid}: holder differs across variants")
        if q.gold != t.gold or q.candidates != t.candidates:
            raise ValueError(f"pair {pid}: third changes the qa gold/candidates")
        if z.gold != q.gold or z.candidates != q.candidates or z.spec["tail"] != q.spec["tail"]:
            raise ValueError(f"pair {pid}: null is not the qa item with the body hidden")
        if any(re.search(rf"\b{re.escape(o)}\b", render(z, False))
               for o in z.spec["objects"] if o != z.spec["tail"]["obj"]):
            raise ValueError(f"pair {pid}: null leaks a body object")


def audit(items: Sequence[Item]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for c in CELLS:
        sub = [it for it in items if it.family == FAMILY and tuple(it.cell) == c
               and it.meta["variant"] == "qa"]
        if not sub:
            continue
        n = len(sub)
        out["/".join(map(str, c))] = {
            "n": n, "chance": 0.25,
            "role_me": sum(it.meta["role"] == "me" for it in sub) / n,
            "role_you": sum(it.meta["role"] == "you" for it in sub) / n,
            "holder_is_last_recipient": sum(it.meta["holder_is_last_recipient"] for it in sub) / n,
            "holder_is_last_speaker": sum(it.meta["holder_is_last_speaker"] for it in sub) / n,
            "gold_each_other_name": sum(it.meta["role"] == "other" and it.gold == it.candidates[2]
                                        for it in sub) / n}
    return out


def audit_failures(aud: Mapping[str, Mapping[str, Any]], tol: float = BALANCE_TOL) -> list[str]:
    bad = []
    for ck, a in aud.items():
        band = max(tol, 1.0 / a["n"]) + 1e-9
        for key in ("role_me", "role_you", "holder_is_last_recipient", "holder_is_last_speaker",
                    "gold_each_other_name"):
            if abs(a[key] - a["chance"]) > band:
                bad.append(f"{ck}: {key} {a[key]:.3f} vs {a['chance']:.3f}")
    return bad


def error_types(item: Item, pick: str) -> list[str]:
    spec: Spec = item.spec  # type: ignore[assignment]
    h = holder(spec)
    _, hist = run(spec)
    t = spec["tail"]
    if spec["variant"] == "narr":
        if pick in set(hist[t["obj"]][:-1]) - {h}:
            return ["previous_owner"]
        return ["other"]
    r = role_of(spec)
    if {item.gold, pick} == {ME, YOU}:
        return ["perspective_flip"]
    if r in ("me", "you") and pick not in (ME, YOU):
        return ["name_for_deixis"]
    if r == "other" and pick in (ME, YOU):
        return ["deixis_for_name"]
    prev = {t["answerer"]: ME, t["asker"]: YOU}
    if pick in {prev.get(p, p) for p in hist[t["obj"]][:-1]}:
        return ["previous_owner"]
    return ["other"]
