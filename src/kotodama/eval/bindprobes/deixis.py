"""``deixis`` (v3) — conversational referent binding through I / you / my / your.

A non-assistant chat transcript (``Name: utterance`` lines, no ChatML, no
User/Assistant) opened by a scene line; who holds what is stated ONLY through
speaker-relative reference::

    A chat between Ada, Ken and Rosa.
    Ada: I have the lamp.
    Ken: The cup is in my hand.
    Rosa: I have the bell.
    Ken: Here, Rosa, take my cup.
    Rosa: Ada, my bell is in your hand now.
    Now the cup is with
    -> " Rosa"

Design: declarations + GIVE only, ≥ 3 candidates, a dialogue-worded third-person
control. Every statement says who HAS or HOLDS an object (have / in my hand / take /
giving), never who owns it ("mine", "yours"), because the queries ask about holding
("is with", "holding") and the third twin narrates "has" / "gives … to". Items carry
``meta.deixis_version = VERSION``; ``items`` ids/seeds carry ``-v<VERSION>``; aggregate and
gates refuse deixis rows/items of any other version (:func:`check_version`).

Addressee rule (the unambiguity law): P = 2 -> "you" is the other speaker and no
utterance carries a vocative (two speakers need none, so it is always dropped);
P = 3 -> EVERY utterance containing you/your/yours or
an imperative opens with a vocative naming the addressee (``Here, Ben, take ...``
counts: the vocative follows the interjection). ``resolve_addressee`` is the one
place pronoun semantics live; :func:`explicit_events` raises on any unresolvable
or ambiguous "you", :func:`run` on every violated precondition.

Events (deictic surface -> effect; precondition):
  declare  ``I have the o.`` / ``The o is in my hand.``       owner(o)=speaker; o unowned
  give     rotated within an item (random start), P=3 / P=2 surfaces:
             giving  ``V, I'm giving you my o.`` / ``I'm giving you my o.``
             here    ``Here, V, take my o.``     / ``Here, take my o.``
             hand    ``V, my o is in your hand now.`` / ``My o is in your hand now.``
                                                              owner(o)=addressee; owner(o)=speaker
("owner" in this module is the holder.) Declarations open the scene, grouped by
declarer in scene order (each object declared once); then E ∈ {1, 2, 3} gives. No
give hands an object straight back: the give after "A gives o to B" is never "B
gives o to A" (:func:`gives_back`). Objects agree with "is" (``pools.singular``).

Queries (the completion ends right before the scored single token, closer ``.``):
  owner      ``Now the o is with``            -> a name; P = 3 ONLY (3 candidates).
             Same line in every variant. E ≥ 2: the queried object has moved and
             is not in the last line. E = 1: the recency slot (1/3) queries the
             given object; the other 2/3 query an object that stayed with its
             declarer (the giver's other object / the bystander's) — with one give
             and the recipient always named last, this is the only way to keep
             "gold = last-mentioned name" off 100% (``meta.queried_obj_moved``).
  self       ``S: Right now I'm holding the``          -> an object
  addressee  ``S: A, right now you're holding the``    -> an object (P = 2: no vocative)
  self/addressee: 3–4 objects in play, candidates = all objects in play; the
  queried person holds EXACTLY one object, and their holdings changed.

Paired variants — every base item ``x`` is emitted as linked variants
(``meta.pair_id`` = the base item id), all under every shot condition:
  deictic  ``x`` itself.
  third    the same events as dialogue-family prose with names, one paragraph, no
           pronouns, no parentheses: ``Ada, Ken and Rosa are sitting at a table. Ada
           has the lamp. Ken has the cup and the bell. Ken gives the cup to Rosa.
           Now Rosa is holding the`` (owner: ``Now the cup is with``). Same gold,
           candidates and object-mention order as ``x``.
  twin     P = 3 only. The same skeleton under a transposition sigma of two
           speakers (the third unchanged): every utterance spoken by a in ``x`` is
           spoken by b and vice versa, vocatives transposed with it (so I/you
           re-derive), the scene and query lines UNCHANGED, so the true answer
           flips. Owner: sigma moves the gold owner. self/addressee: sigma swaps
           the queried person with a partner who also holds exactly one object.
           Its ``3shot_collide`` maps are sigma∘(the base maps).
           P = 2 object cells have NO twin: with ≥ 3 objects and 2 people the
           partner holds ≥ 2 objects, so the transposed query has no single gold
           ("candidates ≥ 3" and "queried speaker holds exactly one" cannot both
           hold for a P = 2 twin). Their pairs are (deictic, third).

Shortcut controls (scheduled per cell on the deictic items; third mirrors them,
the twin only audits; every variant records them in ``meta``). They are scheduled
at the SHORTCUT-NEUTRAL rate (= chance 1/C) rather than 50%: with ≥ 3
candidates a 50% "gold = last mention" rate is itself a recency shortcut worth
cc ≈ .25–.33, and the outer bound is "≤ 50% ± 5%":
  gold_is_last_mention  gold is the last-mentioned candidate-type string.
                        owner: 1/3; object cells: 1/3 of the 3-object items and
                        1/4 of the 4-object items (n_objects is scheduled too).
  gold_is_last_speaker  owner: the gold owner speaks the last line; 1/3. (At
                        P = 3 the last name is the last give's recipient and the last
                        speaker its giver, so the owner schedule cycles
                        recipient / giver / bystander — the bystander is gold 1/3.)
  gold_in_last_line     the ownership-word rule: the gold object (owner: the
                        queried object) is NOT in the line before the query.
                        Enforced for owner E ≥ 2. Exempt: owner E = 1 (by design) and
                        every self/addressee cell — with give-only events every
                        line names exactly one object and the query line none, so
                        for object queries this IS gold_is_last_mention; enforcing
                        it (0%) would hand out an exclusion shortcut (chance
                        1/(C−1)), so the neutral recency schedule replaces it.
Audit: per cell, |share − chance| ≤ max(BALANCE_TOL, 1/n) (:func:`audit_failures`).

Error types (``error_types`` returns every label that applies, primary first):
  speaker_confusion  the wrong side of I/you. Owner query: the pick is the OTHER
                     party of the give that last moved the queried object. Object
                     queries: the pick is an object currently held by the query's
                     counterpart — addressee query: the query SPEAKER ("you" read
                     as "I"); self query: walking back from the last line, the
                     first line's speaker that is not the queried person, or the
                     addressee of a line the queried person spoke.
  previous_owner     owner query: an earlier owner of the queried object. Object
                     queries: an object the queried person owned earlier but not now.
  other_participant  owner query, none of the above.
  other_object       object query, none of the above.
"""

from __future__ import annotations

import random
import re
import string
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence, TypedDict

from kotodama.eval.bindprobes.base import Cell, Item, cell_key, name_list, rejection, the_list
from kotodama.eval.bindprobes.pools import PLURAL_FORM, Pools, singular

FAMILY = "deixis"
VERSION = 3                       # meta.deixis_version; items' id/seed tag "deixis-v3"
PS = (2, 3)
ES = (1, 2, 3)
QKINDS = ("owner", "self", "addressee")
OBJECT_KINDS = ("self", "addressee")
# owner only at P = 3 (3 name candidates): 3 + 6 + 6 = 15 cells
CELLS: list[Cell] = [(p, e, q) for p in PS for e in ES for q in QKINDS
                     if q != "owner" or p == 3]
DEFAULT_PER_CELL = 60             # 15 cells -> 900 base items
PILOT_PER_CELL = 10               # pilot battery: --per-cell deixis=10 (150 base)
LITE_PER_CELL = 10                # lite slice: 10 base items (+ variants) per cell
SEPARATOR = "\n\n* * *\n\n"
VARIANTS = ("deictic", "third", "twin")
ERROR_TYPES = ("speaker_confusion", "previous_owner", "other_participant", "other_object")
MAX_PROMPT_TOKENS = 1024          # prompts < 1024 tokens (checked with the tokenizer)
MIN_CANDIDATES = 3                # chance ≤ 1/3
N_OBJECTS_OWNER = (2, 3, 4)
N_OBJECTS_HELD = (3, 4)           # self/addressee: 3–4 objects in play
BALANCE_TOL = 0.05                # ± 5% per cell

TRANSFERS = ("give",)
GIVE_ORDER = ("giving", "here", "hand")
_YOU = re.compile(r"\byou(?:r|rs|'re)?\b", re.IGNORECASE)
_IMPERATIVE = re.compile(r"^(?:take|give)\b", re.IGNORECASE)
_HERE = re.compile(r"^Here, ")
_PRONOUNS = re.compile(r"\b(?:I|I'm|I'll|me|my|mine|you|your|yours|you're|he|him|his|she|"
                       r"her|hers|they|them|their|it)\b", re.IGNORECASE)


class Spec(TypedDict):
    variant: str                 # "deictic" | "third" | "twin"
    people: list[str]            # scene-line order (unchanged by the twin transposition)
    objects: list[str]           # objects in play, declaration order
    events: list[dict[str, Any]]  # surface events (deictic/twin) or explicit ones (third)
    query: dict[str, Any]
    sigma: Optional[list[str]]   # the twin transposition [a, b] (P = 3; None at P = 2)


class Line(TypedDict):
    speaker: str
    addressee: Optional[str]
    objs: list[str]


def has_twin(cell: Sequence[Any]) -> bool:
    """P = 3 cells carry a twin; P = 2 (object queries only) cannot (module doc)."""
    return int(cell[0]) == 3


def variants(cell: Sequence[Any]) -> tuple[str, ...]:
    return VARIANTS if has_twin(cell) else VARIANTS[:2]


# ── Resolution + simulation ────────────────────────────────────────────────────

def resolve_addressee(people: Sequence[str], speaker: str, voc: Optional[str]) -> str:
    """Who "you" is in an utterance by ``speaker`` opening with vocative ``voc``.

    P = 2: the other speaker (a vocative, if present, must name them). P = 3: the
    vocative is REQUIRED and names the addressee. Raises on anything else."""
    if speaker not in people:
        raise ValueError(f"speaker {speaker!r} is not in the chat {list(people)}")
    if len(people) == 2:
        other = people[1] if people[0] == speaker else people[0]
        if voc is not None and voc != other:
            raise ValueError(f"vocative {voc!r} is not the other speaker {other!r}")
        return other
    if voc is None:
        raise ValueError(f"'you' by {speaker} among {len(people)} people has no vocative "
                         f"(ambiguous)")
    if voc == speaker or voc not in people:
        raise ValueError(f"vocative {voc!r} by {speaker!r} is not another participant")
    return voc


def explicit_events(spec: Spec) -> list[dict[str, Any]]:
    """The events with every pronoun resolved: third-person form, explicit names.

    Third specs are already explicit (validated). Raises on any ambiguity or a
    malformed event; preconditions are checked by :func:`run`."""
    people = spec["people"]
    if len(set(people)) != len(people) or len(people) not in PS:
        raise ValueError(f"bad people {people}")
    if spec["variant"] == "third":
        for ev in spec["events"]:
            if ev["ev"] not in ("declare",) + TRANSFERS:
                raise ValueError(f"unknown event {ev['ev']!r}")
            for k in ("owner", "giver", "recipient"):
                if k in ev and ev[k] not in people:
                    raise ValueError(f"{ev['ev']}: {k}={ev[k]!r} is not in the chat")
        return [dict(ev) for ev in spec["events"]]
    out: list[dict[str, Any]] = []
    for k, ev in enumerate(spec["events"]):
        kind, s = ev["ev"], ev["speaker"]
        if kind == "declare":
            if s not in people:
                raise ValueError(f"event {k}: speaker {s!r} not in the chat")
            out.append({"ev": "declare", "owner": s, "obj": ev["obj"]})
        elif kind == "give":
            a = resolve_addressee(people, s, ev.get("voc"))
            # the surface form rides along (meta only: the third paraphrase is one form)
            out.append({"ev": "give", "giver": s, "recipient": a, "obj": ev["obj"],
                        "form": ev.get("form")})
        else:
            raise ValueError(f"event {k}: unknown event {kind!r} (declare/give only)")
    return out


def run(spec: Spec) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Resolve and execute the events. Returns (final owner per object, owner
    history per object incl. the declarer). Raises on any precondition violation."""
    evs = explicit_events(spec)
    objects = spec["objects"]
    if len(set(objects)) != len(objects) or not 2 <= len(objects) <= 4:
        raise ValueError(f"bad objects {objects}")
    owner: dict[str, str] = {}
    hist: dict[str, list[str]] = {}
    seen_transfer = False
    for k, ev in enumerate(evs):
        o = ev["obj"]
        if ev["ev"] == "declare":
            if seen_transfer:
                raise ValueError(f"event {k}: declaration after a transfer")
            if o not in objects or o in owner:
                raise ValueError(f"event {k}: the {o} is not an unowned object in play")
            owner[o] = ev["owner"]
            hist[o] = [ev["owner"]]
            continue
        seen_transfer = True
        frm, to = ev["giver"], ev["recipient"]
        if owner.get(o) != frm:
            raise ValueError(f"event {k}: {frm} does not own the {o} (owner {owner.get(o)})")
        if frm == to:
            raise ValueError(f"event {k}: {frm} gives to themself")
        owner[o] = to
        hist[o].append(to)
    if set(owner) != set(objects):
        raise ValueError(f"undeclared objects: {sorted(set(objects) - set(owner))}")
    return owner, hist


def holdings(spec: Spec) -> dict[str, list[str]]:
    """person -> objects held at the end (people order; objects in declaration order)."""
    owner, _ = run(spec)
    return {p: [o for o in spec["objects"] if owner[o] == p] for p in spec["people"]}


def declared(spec: Spec) -> dict[str, list[str]]:
    _, hist = run(spec)
    return {p: [o for o in spec["objects"] if hist[o][0] == p] for p in spec["people"]}


def query_person(spec: Spec) -> Optional[str]:
    """The person a self/addressee query asks about (None for owner queries)."""
    q = spec["query"]
    if q["kind"] == "owner":
        return None
    if spec["variant"] == "third":
        if q["person"] not in spec["people"]:
            raise ValueError(f"query person {q['person']!r} is not in the chat")
        return str(q["person"])
    if q["kind"] == "self":
        if q["speaker"] not in spec["people"]:
            raise ValueError(f"query speaker {q['speaker']!r} is not in the chat")
        return str(q["speaker"])
    return resolve_addressee(spec["people"], q["speaker"], q.get("voc"))


def simulate(item: Item) -> str:
    spec: Spec = item.spec  # type: ignore[assignment]
    q = spec["query"]
    owner, _ = run(spec)
    if q["kind"] == "owner":
        if len(spec["people"]) != 3:
            raise ValueError(f"{item.item_id}: owner queries are P = 3 only")
        if q["obj"] not in owner:
            raise ValueError(f"{item.item_id}: queried object {q['obj']!r} not in play")
        return owner[q["obj"]]
    if q["kind"] not in QKINDS:
        raise ValueError(f"{item.item_id}: unknown query kind {q['kind']!r}")
    person = query_person(spec)
    held = holdings(spec)[str(person)]
    if len(held) != 1:
        raise ValueError(f"{item.item_id}: {person} holds {held}, not exactly one object")
    return held[0]


def lines(spec: Spec) -> list[Line]:
    """Body lines as (speaker, addressee, objects) — the deictic conversation
    structure, also for the third variant (whose events encode who would speak)."""
    out: list[Line] = []
    for ev in explicit_events(spec):
        if ev["ev"] == "declare":
            out.append({"speaker": ev["owner"], "addressee": None, "objs": [ev["obj"]]})
        else:
            out.append({"speaker": ev["giver"], "addressee": ev["recipient"], "objs": [ev["obj"]]})
    return out


def last_line_objects(spec: Spec) -> list[str]:
    """Objects named in the line immediately preceding the query."""
    return lines(spec)[-1]["objs"]


def n_transfers(spec: Spec) -> int:
    return sum(1 for ev in spec["events"] if ev["ev"] in TRANSFERS)


# ── Rendering ──────────────────────────────────────────────────────────────────

_FMT = string.Formatter()
Segment = tuple[str, list[str]]   # (text, slot values in text order)


def _fill(tmpl: str, **kw: str) -> Segment:
    """Format ``tmpl`` and return (text, slot values in text order)."""
    slots = [kw[f] for _, f, _, _ in _FMT.parse(tmpl) if f]
    return tmpl.format(**kw), slots


# Holding words only (module doc): the queries ask who holds an object, so no form
# states ownership alone ("mine", "yours").
DECLARE_FORMS = {"have": "I have the {o}.", "hand": "The {o} is in my hand."}
# give surfaces: (P = 3 with the vocative {v}, P = 2 without)
GIVE_FORMS: dict[str, tuple[str, str]] = {
    "giving": ("{v}, I'm giving you my {o}.", "I'm giving you my {o}."),
    "here": ("Here, {v}, take my {o}.", "Here, take my {o}."),
    "hand": ("{v}, my {o} is in your hand now.", "My {o} is in your hand now."),
}


def gives_back(prev: Optional[Mapping[str, Any]], giver: str, recipient: str, obj: str) -> bool:
    """True when a give returns the previous give's object from its recipient to its
    giver (``prev`` = an explicit give event, or None): the ping-pong no generator
    writes."""
    return (prev is not None and prev.get("ev") == "give" and giver == prev["recipient"]
            and recipient == prev["giver"] and obj == prev["obj"])


def check_no_give_back(explicit: Sequence[Mapping[str, Any]], what: str) -> None:
    """Raise ``ValueError`` if any give in the explicit events hands its object
    straight back (:func:`gives_back` against the event before it)."""
    for k in range(1, len(explicit)):
        ev = explicit[k]
        if ev["ev"] == "give" and gives_back(explicit[k - 1], ev["giver"], ev["recipient"],
                                             ev["obj"]):
            raise ValueError(f"{what}: event {k} hands the {ev['obj']} straight back")


def _deictic_line(ev: Mapping[str, Any]) -> Segment:
    kind, s = ev["ev"], ev["speaker"]
    if kind == "declare":
        return _fill("{_s}: " + DECLARE_FORMS[ev["form"]], _s=s, o=ev["obj"])
    if kind == "give":
        with_voc, bare = GIVE_FORMS[ev["form"]]
        if ev.get("voc"):
            return _fill("{_s}: " + with_voc, _s=s, v=ev["voc"], o=ev["obj"])
        return _fill("{_s}: " + bare, _s=s, o=ev["obj"])
    raise ValueError(f"unknown event {kind!r}")


def _third_sentences(events: Sequence[Mapping[str, Any]]) -> list[Segment]:
    """dialogue-family prose: ``X has the a and the b.`` (consecutive declarations
    by one person merged, as dialogue's intro does) and ``G gives the o to R.``"""
    out: list[Segment] = []
    k = 0
    while k < len(events):
        ev = events[k]
        if ev["ev"] == "declare":
            objs = [ev["obj"]]
            while k + 1 < len(events) and events[k + 1]["ev"] == "declare" \
                    and events[k + 1]["owner"] == ev["owner"]:
                k += 1
                objs.append(events[k]["obj"])
            out.append((f"{ev['owner']} has {the_list(objs)}.", [ev["owner"]] + objs))
        elif ev["ev"] == "give":
            out.append(_fill("{g} gives the {o} to {r}.", g=ev["giver"], o=ev["obj"],
                             r=ev["recipient"]))
        else:
            raise ValueError(f"unknown event {ev['ev']!r}")
        k += 1
    return out


def _lead(spec: Spec) -> Segment:
    """The query lead-in (the closer after the answer is always ``.``)."""
    q = spec["query"]
    if q["kind"] == "owner":
        return _fill("Now the {o} is with", o=q["obj"])
    form = q.get("form", "a")
    if form not in QFORMS:
        raise ValueError(f"unknown query form {form!r}")
    if spec["variant"] == "third":
        if form == "b":
            return _fill("Now the thing in {p}'s hand is the", p=q["person"])
        return _fill("Now {p} is holding the", p=q["person"])
    if q["kind"] == "self":
        if form == "b":
            return _fill("{s}: The thing in my hand right now is the", s=q["speaker"])
        return _fill("{s}: Right now I'm holding the", s=q["speaker"])
    if form == "b":
        if q.get("voc"):
            return _fill("{s}: {v}, the thing in your hand right now is the", s=q["speaker"], v=q["voc"])
        return _fill("{s}: The thing in your hand right now is the", s=q["speaker"])
    if q.get("voc"):
        return _fill("{s}: {v}, right now you're holding the", s=q["speaker"], v=q["voc"])
    return _fill("{s}: Right now you're holding the", s=q["speaker"])


def _segments(spec: Spec) -> tuple[list[Segment], str]:
    """(scene + body + lead segments, the joiner): chat lines for deictic/twin, one
    prose paragraph for third."""
    people = spec["people"]
    names = {f"p{i}": p for i, p in enumerate(people)}
    slots = name_list([f"{{p{i}}}" for i in range(len(people))])
    if spec["variant"] == "third":
        segs = [_fill(slots + " are sitting at a table.", **names)]
        segs += _third_sentences(spec["events"])
        return segs + [_lead(spec)], " "
    segs = [_fill("A chat between " + slots + ".", **names)]
    segs += [_deictic_line(ev) for ev in spec["events"]]
    return segs + [_lead(spec)], "\n"


def _rendered(spec: Spec) -> tuple[str, list[str]]:
    """(text without the answer, slot values in text order)."""
    segs, joiner = _segments(spec)
    return joiner.join(t for t, _ in segs), [s for _, ss in segs for s in ss]


def render(item: Item, with_answer: bool) -> str:
    text, _ = _rendered(item.spec)  # type: ignore[arg-type]
    return f"{text} {item.gold}." if with_answer else text


def check_surface(item: Item) -> None:
    """Surface-level law checks on the rendered text (independent of the simulator).

    Every variant: no parentheses. deictic/twin: every line after the scene is
    ``Name: utterance`` (owner: the final narrator query line excepted); a P = 3
    utterance containing you/your/yours or an imperative opens (after an optional
    ``Here, ``) with a vocative naming another participant; P = 2 utterances carry
    none. third: one paragraph, no chat lines, no pronoun at all."""
    spec: Spec = item.spec  # type: ignore[assignment]
    people = spec["people"]
    text = render(item, False)
    if "(" in text or ")" in text:
        raise ValueError(f"{item.item_id}: parentheses in the text")
    if spec["variant"] == "third":
        if "\n" in text or ":" in text:
            raise ValueError(f"{item.item_id}: third variant is not one prose paragraph")
        if _PRONOUNS.search(text):
            raise ValueError(f"{item.item_id}: pronoun in the third variant: "
                             f"{_PRONOUNS.search(text).group(0)!r}")  # type: ignore[union-attr]
        return
    body = text.split("\n")[1:]
    if spec["query"]["kind"] == "owner":
        if not body or not body[-1].startswith("Now the "):
            raise ValueError(f"{item.item_id}: owner query is not the narrator line")
        body = body[:-1]
    for ln in body:
        m = re.match(r"^(\w+): (.*)$", ln)
        if not m or m.group(1) not in people:
            raise ValueError(f"{item.item_id}: not a 'Name: utterance' line: {ln!r}")
        spk, utt = m.groups()
        rest = _HERE.sub("", utt)
        v = re.match(r"^(\w+), ", rest)
        if v and (v.group(1) not in people or v.group(1) == spk):
            raise ValueError(f"{item.item_id}: vocative is not another participant: {ln!r}")
        if len(people) == 3 and not v and (_YOU.search(utt) or _IMPERATIVE.match(rest)):
            raise ValueError(f"{item.item_id}: P=3 addressed utterance without a vocative: "
                             f"{ln!r}")
        if len(people) == 2 and v:
            raise ValueError(f"{item.item_id}: P=2 utterance carries a vocative: {ln!r}")


# ── Items ──────────────────────────────────────────────────────────────────────

def labels(item: Item) -> list[str]:
    return list(item.spec["people"])


def objects(item: Item) -> set[str]:
    return set(item.spec["objects"])


def exempt_ownership_word(cell: Sequence[Any]) -> bool:
    """Owner E = 1 (by design) and every object cell (module doc: there the rule IS
    the recency control, scheduled at the neutral rate instead)."""
    return str(cell[2]) != "owner" or int(cell[1]) == 1


# Query phrasings for self/addressee (owner has one form). "a" = "…holding the" (the
# default); "b" = "the thing in my/your/X's hand … is the" — a paired re-phrasing of the
# SAME items (same events, same gold) to check that a result is not tied to one sentence.
QFORMS: tuple[str, ...] = ("a", "b")


def with_qform(item: Item, form: str) -> Item:
    """The same item (and its exemplars) re-rendered with query phrasing ``form``; the
    id gains ``-q<form>`` and the gold is re-derived from the simulator (must match)."""
    if form not in QFORMS:
        raise ValueError(f"unknown query form {form!r}")
    if item.meta.get("qkind") == "owner" or form == "a":
        return item
    def one(it: Item, suffix: str) -> Item:
        spec = dict(it.spec); spec["query"] = dict(spec["query"], form=form)
        new = build(it.item_id + suffix, tuple(it.cell), spec, pair_id=(it.meta.get("pair_id") or it.item_id) + suffix)
        if new.gold != it.gold:
            raise ValueError(f"{it.item_id}: gold changed under query form {form!r}")
        for k in ("audit", "slot"):
            if k in it.meta:
                new.meta[k] = it.meta[k]
        return new
    out = one(item, f"-q{form}")
    out.meta["qform"] = form
    out.shots = [one(Item.from_dict(sh), f"-q{form}").to_dict() for sh in item.shots]
    out.collide_maps = list(item.collide_maps)
    for k, v in item.meta.items():
        out.meta.setdefault(k, v)
    out.meta["pair_id"] = (item.meta.get("pair_id") or item.item_id) + f"-q{form}"
    return out


def build(item_id: str, cell: Cell, spec: Spec, pair_id: Optional[str] = None) -> Item:
    """Item from a spec: gold from the simulator, audit controls in ``meta``."""
    if spec["variant"] not in VARIANTS:
        raise ValueError(f"{item_id}: unknown variant {spec['variant']!r}")
    kind = spec["query"]["kind"]
    cands = list(spec["people"]) if kind == "owner" else list(spec["objects"])
    _, slots = _rendered(spec)
    mentions = [s for s in slots if s in cands]
    it = Item(item_id=item_id, family=FAMILY, cell=list(cell), spec=dict(spec), gold="",
              candidates=cands, mention_order=mentions,
              meta={"deixis_version": VERSION, "variant": spec["variant"],
                    "pair_id": pair_id or item_id, "qkind": kind,
                    "n_objects": len(spec["objects"]),
                    "events": [ev["ev"] for ev in spec["events"] if ev["ev"] in TRANSFERS],
                    "give_forms": [ev.get("form") for ev in spec["events"]
                                   if ev["ev"] in TRANSFERS]})
    it.gold = simulate(it)
    body = lines(spec)
    gold_obj = spec["query"]["obj"] if kind == "owner" else it.gold
    _, hist = run(spec)
    it.meta.update({
        "query_person": query_person(spec),
        "gold_is_last_mention": bool(mentions) and mentions[-1] == it.gold,
        # the deictic transcript's last speaker (third: the giver that sentence narrates)
        "gold_is_last_speaker": (body[-1]["speaker"] == it.gold) if kind == "owner" else None,
        "gold_in_last_line": gold_obj in last_line_objects(spec),
        "ownership_word_exempt": exempt_ownership_word(cell),
        "queried_obj_moved": (len(hist[gold_obj]) > 1) if kind == "owner" else None,
    })
    return it


# ── Generation ─────────────────────────────────────────────────────────────────

Want = dict[str, Any]   # owner: {"lm", "ls"}; object cells: {"n", "lm"}

# Shortcut-neutral schedules (module doc), cycled by the item index k.
_OWNER_SCHED: tuple[Want, ...] = ({"lm": True, "ls": False},    # gold = last recipient
                                  {"lm": False, "ls": True},    # gold = last giver
                                  {"lm": False, "ls": False})   # gold = the bystander
_OBJECT_SCHED: tuple[Want, ...] = tuple(
    {"n": n, "lm": j == 0} for n in N_OBJECTS_HELD for j in range(n))   # 1/n per n


def schedule(cell: Sequence[Any]) -> tuple[Want, ...]:
    return _OWNER_SCHED if str(cell[2]) == "owner" else _OBJECT_SCHED


def _draw(rng: random.Random, cell: Cell, pools: Pools, item_id: str,
          template: Optional[Item], want: Optional[Want]) -> Optional[Item]:
    """One random base (deictic) item for ``cell`` with its sigma, or None."""
    p, e, kind = int(cell[0]), int(cell[1]), str(cell[2])
    people = rng.sample(pools.names, p)
    if want is not None and "n" in want:
        n_obj = int(want["n"])
    else:
        n_obj = rng.choice(N_OBJECTS_OWNER if kind == "owner" else N_OBJECTS_HELD)
    objs0 = rng.sample(singular(pools.objects), n_obj)
    first = {o: rng.choice(people) for o in objs0}
    objs = [o for x in people for o in objs0 if first[o] == x]   # declarations grouped
    owner = dict(first)
    events: list[dict[str, Any]] = [{"ev": "declare", "speaker": owner[o], "obj": o,
                                     "form": rng.choice(tuple(DECLARE_FORMS))} for o in objs]
    start = rng.randrange(len(GIVE_ORDER))
    prev: Optional[dict[str, Any]] = None
    for i in range(e):
        s = rng.choice([x for x in people if any(owner[o] == x for o in objs)])
        a = rng.choice([x for x in people if x != s])
        own = [o for o in objs if owner[o] == s and not gives_back(prev, s, a, o)]
        if not own:
            return None
        o = rng.choice(own)
        events.append({"ev": "give", "speaker": s, "voc": a if p == 3 else None, "obj": o,
                       "form": GIVE_ORDER[(start + i) % len(GIVE_ORDER)]})
        owner[o] = a
        prev = {"ev": "give", "giver": s, "recipient": a, "obj": o}
    spec: Spec = {"variant": "deictic", "people": people, "objects": objs, "events": events,
                  "query": {}, "sigma": None}
    _, hist = run(spec)
    last = lines(spec)[-1]
    if kind == "owner":
        if e >= 2:
            pool = [o for o in objs if len(hist[o]) > 1 and o not in last["objs"]]
        elif want is not None and want["lm"]:
            pool = list(last["objs"])       # E = 1 recency slot: the given object
        else:
            pool = list(objs)
        if want is not None:
            pool = [o for o in pool if (owner[o] == last["addressee"]) == want["lm"]
                    and (owner[o] == last["speaker"]) == want["ls"]]
        if not pool:
            return None
        o = rng.choice(pool)
        gold = owner[o]
        spec["query"] = {"kind": "owner", "obj": o}
        spec["sigma"] = [gold, rng.choice([x for x in people if x != gold])]
    else:
        fin = holdings(spec)
        decl = declared(spec)
        one = [x for x in people if len(fin[x]) == 1]
        targets = [x for x in one if set(fin[x]) != set(decl[x])]
        if p == 3:   # the twin needs a partner who also holds exactly one object
            targets = [x for x in targets if any(y != x for y in one)]
        if want is not None:
            targets = [x for x in targets if (fin[x][0] in last["objs"]) == want["lm"]]
        if not targets:
            return None
        q = rng.choice(targets)
        if kind == "self":
            spec["query"] = {"kind": "self", "speaker": q}
        else:
            spec["query"] = {"kind": "addressee", "speaker": rng.choice([x for x in people
                                                                         if x != q]),
                             "voc": q if p == 3 else None}
        if p == 3:
            spec["sigma"] = [q, rng.choice([y for y in one if y != q])]
    it = build(item_id, cell, spec)
    # the collide re-label is positional: keep an owner exemplar's answer off the
    # test item's gold slot (so no collide exemplar answers the test gold)
    if template is not None and kind == "owner" and \
            people.index(it.gold) == template.spec["people"].index(template.gold):
        return None
    return it


def gen_item(rng: random.Random, cell: Cell, pools: Pools, item_id: str, k: int,
             per_cell: int, template: Optional[Item] = None) -> Item:
    """A BASE (deictic) item; :func:`expand` derives its third (+ twin) variants.
    Test items follow the cell's :func:`schedule` cyclically; exemplars draw their
    slot from it at random (same mix, so the shots teach no recency regularity)."""
    if tuple(cell) not in CELLS:
        raise ValueError(f"deixis: unknown cell {cell}")
    cell = tuple(cell)
    sched = schedule(cell)
    want: Want = sched[k % len(sched)] if template is None else rng.choice(sched)
    return rejection(lambda: _draw(rng, cell, pools, item_id, template, want),
                     f"deixis item {item_id}", tries=20000)


# ── Variants ───────────────────────────────────────────────────────────────────

def to_third(spec: Spec) -> Spec:
    """The same events and question narrated as dialogue-family prose with names."""
    if spec["variant"] == "third":
        return spec
    q = dict(spec["query"])
    if q["kind"] != "owner":
        q = {"kind": q["kind"], "person": query_person(spec),
             "asker": q["speaker"] if q["kind"] == "addressee" else None}
    return {"variant": "third", "people": list(spec["people"]), "objects": list(spec["objects"]),
            "events": explicit_events(spec), "query": q,
            "sigma": list(spec["sigma"]) if spec["sigma"] else None}


_NAME_KEYS = ("speaker", "voc", "owner", "giver", "recipient", "person", "asker")


def _rename(d: Mapping[str, Any], m: Mapping[str, str]) -> dict[str, Any]:
    return {k: (m.get(v, v) if k in _NAME_KEYS and isinstance(v, str) else v)
            for k, v in d.items()}


def to_twin(spec: Spec) -> Spec:
    """Transpose the two speakers of ``spec['sigma']`` in every EVENT (speakers and
    vocatives), keeping the scene line and the query line unchanged."""
    if spec["variant"] != "deictic" or not spec["sigma"]:
        raise ValueError("the twin is built from a deictic spec with a sigma (P = 3)")
    a, b = spec["sigma"]
    sig = {a: b, b: a}
    return {"variant": "twin", "people": list(spec["people"]), "objects": list(spec["objects"]),
            "events": [_rename(ev, sig) for ev in spec["events"]],
            "query": dict(spec["query"]), "sigma": [a, b]}


def expand(base: Item) -> list[Item]:
    """[deictic, third(, twin)] for one base item (with shots + collide maps).

    deictic + twin share the deictic exemplars; the third variant's exemplars are
    the same exemplars narrated in the third person. The twin's collide maps are
    sigma∘(base maps)."""
    pid = base.item_id
    spec: Spec = base.spec  # type: ignore[assignment]
    shots = base.shot_items()
    out: list[Item] = []
    for variant in variants(base.cell):
        if variant == "deictic":
            vspec, vshots, maps = spec, shots, base.collide_maps
        elif variant == "third":
            vspec = to_third(spec)
            vshots = [build(f"{s.item_id}_third", tuple(s.cell), to_third(s.spec),  # type: ignore[arg-type]
                            pair_id=s.item_id) for s in shots]
            maps = base.collide_maps
        else:
            vspec = to_twin(spec)
            vshots = shots
            a, b = spec["sigma"]  # type: ignore[misc]
            sig = {a: b, b: a}
            maps = [{k: sig.get(v, v) for k, v in m.items()} for m in base.collide_maps]
        it = build(f"{pid}_{variant}", tuple(base.cell), vspec, pair_id=pid)
        it.meta["shots_disjoint"] = base.meta.get("shots_disjoint")
        it.shots = [s.to_dict() for s in vshots]
        it.collide_maps = [dict(m) for m in maps]
        out.append(it)
    return out


def relabel(item: Item, mapping: Mapping[str, str], target: Optional[Item] = None) -> Item:
    """Rename people everywhere (scene, speakers, vocatives, query, sigma). A
    bijection keeps the addressee rule valid; the build re-simulates and raises
    otherwise."""
    spec: Spec = item.spec  # type: ignore[assignment]
    missing = set(spec["people"]) - set(mapping)
    if missing:
        raise ValueError(f"{item.item_id}: people {sorted(missing)} not in the map")
    if len({mapping[p] for p in spec["people"]}) != len(spec["people"]):
        raise ValueError(f"{item.item_id}: the re-label map is not injective")
    people = [mapping[p] for p in spec["people"]]
    if target is not None and sorted(people) == sorted(target.spec["people"]):
        # the collide case: exemplar scene lines list the names in the TEST item's
        # order (the twin's sigma-composed maps would otherwise reorder them)
        people = list(target.spec["people"])
    new: Spec = {"variant": spec["variant"], "people": people,
                 "objects": list(spec["objects"]),
                 "events": [_rename(ev, mapping) for ev in spec["events"]],
                 "query": _rename(spec["query"], mapping),
                 "sigma": [mapping[x] for x in spec["sigma"]] if spec["sigma"] else None}
    out = build(f"{item.item_id}_collide", tuple(item.cell), new,
                pair_id=str(item.meta.get("pair_id", item.item_id)))
    expect = mapping[item.gold] if spec["query"]["kind"] == "owner" else item.gold
    if out.gold != expect:
        raise ValueError(f"{item.item_id}: re-label changed the gold")
    return out


# ── Per-item / per-family validation ───────────────────────────────────────────

def approx_tokens(text: str) -> int:
    """A tokenizer-free, conservative token estimate: 1.25 x (words + punctuation +
    newlines). Every template word, name and object is a common single-token word."""
    return int(1.25 * len(re.findall(r"\w+|[^\w\s]|\n", text))) + 1


def check_version(records: Sequence[Mapping[str, Any]], what: str) -> None:
    """Refuse deixis records (item ``meta`` dicts or grid rows) not built by this
    generation: grids/items of another version must never be silently mixed with it."""
    bad = [r for r in records if r.get("deixis_version") != VERSION]
    if bad:
        raise ValueError(f"{len(bad)} deixis {what} lack deixis_version={VERSION} (first: "
                         f"{bad[0].get('item_id')!r}, version {bad[0].get('deixis_version')!r}); "
                         f"regenerate the items and re-run the grid with deixis v{VERSION}")


def validate_item(item: Item) -> None:
    """Deixis-specific invariants (called by ``items.validate``)."""
    spec: Spec = item.spec  # type: ignore[assignment]
    p, e, kind = item.cell
    if item.meta.get("deixis_version") != VERSION:
        raise ValueError(f"{item.item_id}: deixis_version {item.meta.get('deixis_version')!r} "
                         f"!= {VERSION}")
    if len(spec["people"]) != p or n_transfers(spec) != e or spec["query"]["kind"] != kind:
        raise ValueError(f"{item.item_id}: spec does not match cell {item.cell}")
    if item.meta.get("variant") != spec["variant"] or spec["variant"] not in variants(item.cell):
        raise ValueError(f"{item.item_id}: variant {spec['variant']!r} invalid for {item.cell}")
    if len(item.candidates) < MIN_CANDIDATES:
        raise ValueError(f"{item.item_id}: {len(item.candidates)} candidates (needs "
                         f">= {MIN_CANDIDATES})")
    if set(spec["objects"]) & PLURAL_FORM:
        raise ValueError(f"{item.item_id}: plural-form object read with 'is'")
    check_no_give_back(explicit_events(spec), item.item_id)
    if kind != "owner" and len(spec["objects"]) not in N_OBJECTS_HELD:
        raise ValueError(f"{item.item_id}: {len(spec['objects'])} objects in an object cell")
    check_surface(item)
    if not item.meta["ownership_word_exempt"] and item.meta["variant"] == "deictic" \
            and item.meta["gold_in_last_line"]:
        raise ValueError(f"{item.item_id}: ownership-word rule violated")


def final_state(spec: Spec) -> dict[str, str]:
    return run(spec)[0]


def check_pairs(items: Sequence[Item]) -> None:
    """Pair linking: every pair_id has exactly its cell's variants, same cell and
    candidates; third has the identical final state + gold; twin's final state is
    sigma(deictic's) and its gold differs (the flip)."""
    by: defaultdict[str, dict[str, Item]] = defaultdict(dict)
    for it in items:
        if it.family != FAMILY:
            continue
        v = it.meta["variant"]
        if v in by[it.meta["pair_id"]]:
            raise ValueError(f"pair {it.meta['pair_id']}: duplicate {v}")
        by[it.meta["pair_id"]][v] = it
    for pid, g in by.items():
        x = g.get("deictic")
        if x is None or set(g) != set(variants(x.cell)):
            raise ValueError(f"pair {pid}: variants {sorted(g)}")
        if any(v.cell != x.cell or v.candidates != x.candidates for v in g.values()):
            raise ValueError(f"pair {pid}: cell/candidates differ across variants")
        t = g["third"]
        fx = final_state(x.spec)  # type: ignore[arg-type]
        if final_state(t.spec) != fx or t.gold != x.gold:  # type: ignore[arg-type]
            raise ValueError(f"pair {pid}: third variant changes the state or gold")
        if t.mention_order != x.mention_order and x.cell[2] != "owner":
            raise ValueError(f"pair {pid}: third variant changes the object-mention order")
        if "twin" not in g:
            continue
        w = g["twin"]
        a, b = x.spec["sigma"]
        sig = {a: b, b: a}
        if final_state(w.spec) != {o: sig.get(h, h) for o, h in fx.items()}:  # type: ignore[arg-type]
            raise ValueError(f"pair {pid}: twin state is not sigma(deictic state)")
        if w.gold == x.gold:
            raise ValueError(f"pair {pid}: twin does not flip the answer")
        if w.spec["query"] != x.spec["query"]:
            raise ValueError(f"pair {pid}: twin query wording differs")


def audit(items: Sequence[Item]) -> dict[str, dict[str, dict[str, Any]]]:
    """Per cell × variant: n, chance (mean 1/C) and the share of each control."""
    by: defaultdict[tuple[str, str], list[Item]] = defaultdict(list)
    for it in items:
        if it.family == FAMILY:
            by[(it.cell_key, it.meta["variant"])].append(it)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for c in CELLS:
        ck = cell_key(c)
        for v in VARIANTS:
            sub = by.get((ck, v), [])
            if not sub:
                continue
            n = len(sub)
            ls = [it.meta["gold_is_last_speaker"] for it in sub
                  if it.meta["gold_is_last_speaker"] is not None]
            mv = [it.meta["queried_obj_moved"] for it in sub
                  if it.meta.get("queried_obj_moved") is not None]
            out.setdefault(ck, {})[v] = {
                "n": n,
                "chance": sum(1.0 / len(it.candidates) for it in sub) / n,
                "gold_is_last_mention": sum(it.meta["gold_is_last_mention"] for it in sub) / n,
                "gold_is_last_speaker": (sum(ls) / len(ls)) if ls else None,
                "gold_in_last_line": sum(it.meta["gold_in_last_line"] for it in sub) / n,
                "ownership_word_exempt": exempt_ownership_word(c),
                "queried_obj_moved": (sum(mv) / len(mv)) if mv else None,
            }
    return out


def audit_failures(aud: Mapping[str, Mapping[str, Mapping[str, Any]]],
                   tol: float = BALANCE_TOL) -> list[str]:
    """Shortcut controls on the deictic items: recency and (owner) last-speaker within
    max(tol, 1/n) of the cell's chance rate (the shortcut-neutral target; always
    inside the outer "≤ 50% ± 5%" bound); the ownership-word rule never violated outside
    the exempt cells."""
    bad: list[str] = []
    for ck, per in aud.items():
        a = per.get("deictic")
        if a is None:
            continue
        band = max(tol, 1.0 / a["n"]) + 1e-9
        if abs(a["gold_is_last_mention"] - a["chance"]) > band:
            bad.append(f"{ck}: recency {a['gold_is_last_mention']:.3f} vs chance "
                       f"{a['chance']:.3f}")
        if a["gold_is_last_speaker"] is not None and \
                abs(a["gold_is_last_speaker"] - a["chance"]) > band:
            bad.append(f"{ck}: last-speaker {a['gold_is_last_speaker']:.3f} vs chance "
                       f"{a['chance']:.3f}")
        if not a["ownership_word_exempt"] and a["gold_in_last_line"] > 0:
            bad.append(f"{ck}: ownership-word {a['gold_in_last_line']:.3f}")
    return bad


def check_family(items: Sequence[Item]) -> None:
    check_pairs(items)


# ── Error typing ───────────────────────────────────────────────────────────────

def _counterpart(spec: Spec) -> Optional[str]:
    """The query's "other side" (see module doc)."""
    q = spec["query"]
    person = query_person(spec)
    if q["kind"] == "addressee":
        return str(q["asker"] if spec["variant"] == "third" else q["speaker"])
    for ln in reversed(lines(spec)):
        if ln["speaker"] != person:
            return ln["speaker"]
        if ln["addressee"] is not None:
            return ln["addressee"]
    return None


def error_types(item: Item, pick: str) -> list[str]:
    spec: Spec = item.spec  # type: ignore[assignment]
    q = spec["query"]
    _, hist = run(spec)
    t: list[str] = []
    if q["kind"] == "owner":
        o = q["obj"]
        moves = [ev for ev in explicit_events(spec) if ev["ev"] == "give" and ev["obj"] == o]
        if moves and pick in (moves[-1]["giver"], moves[-1]["recipient"]) and pick != item.gold:
            t.append("speaker_confusion")
        if pick in set(hist[o][:-1]) - {item.gold}:
            t.append("previous_owner")
        return t or ["other_participant"]
    person = query_person(spec)
    fin = holdings(spec)
    cp = _counterpart(spec)
    if cp is not None and pick in fin[cp]:
        t.append("speaker_confusion")
    if person in hist.get(pick, []) and pick not in fin[str(person)]:
        t.append("previous_owner")
    return t or ["other_object"]
