"""``deixis`` — conversational referent binding through I / you / my / your.

A non-assistant chat transcript (``Name: utterance`` lines, no ChatML, no
User/Assistant) opened by a scene line; who owns what is stated ONLY through
speaker-relative reference::

    A chat between Ada, Ken and Rosa.
    Ada: I have the lamp.
    Ken: The cup is mine.
    Ken: Rosa, take my cup.
    Ada: Rosa, give me your cup.
    Rosa: Ada, here you go.
    (Now the cup belongs to
    -> " Ada"

Addressee rule (the unambiguity law): P = 2 -> "you" is the other speaker (no
vocatives); P = 3 -> EVERY utterance containing you/your/yours begins with a
vocative naming the addressee — so the request reply is ``Rosa: Ada, here you
go.`` at P = 3 (the spec's bare ``Here you go.`` contains "you"), and every give /
request / swap line carries a vocative. ``resolve`` is the one place pronoun
semantics live: it turns the SURFACE events into explicit (giver, recipient)
events, raising on any unresolvable or ambiguous "you" and on every violated
precondition.

Events (deictic surface -> effect; precondition):
  declare  ``I have the o.`` / ``The o is mine.``          owner(o)=speaker; o unowned
  give     ``[V, ]take my o.`` / ``[V, ]I'm giving you my o.``  owner(o)=addressee; owner(o)=speaker
  request  ``[V, ]give me your o.`` + replier ``[S, ]here you go.`` (one event, two lines)
                                                          owner(o)=requester; owner(o)=addressee,
                                                          and the replier IS that addressee
  swap     ``[V, ]let's swap: you take my o1 and I'll take your o2.``
                                                          exchange; o1=speaker's, o2=addressee's
Declarations open the scene (each object declared once, by its first owner);
then E ∈ {1, 2, 4} events from {give, request, swap}. 2–4 objects in play.

Queries (the completion ends right before the scored single token):
  owner      ``(Now the o belongs to``                  -> a name; candidates = all P names.
             The queried object has changed owner at least once.
  self       ``S: Right now I'm holding the``           -> an object; candidates = all objects
  addressee  ``S: A, right now you're holding the``     -> an object (P = 2 omits the vocative)
self/addressee query a person holding EXACTLY one object whose holdings changed.

Paired variants — every base item ``x`` is emitted three times, linked by
``meta.pair_id`` (= the base item id), all under every shot condition:
  deictic  ``x`` itself.
  third    the same events as narrator lines in the third person WITH NAMES and no
           pronouns at all (P=3 ``(Ada gives the cup to Ken.)``, P=2 ``(Ken gets the
           cup from Ada.)`` — the form mirrors which name the deictic line mentions
           last, so recency is identical across the pair; no possessive "her lamp"
           either, since names carry no gender here); same scene line, same
           question, same gold and candidates. The self/addressee question is narrated
           too (``(Right now Ken is holding the``) so the control carries no deictic
           step; the owner question is already a narrator line and is identical.
  twin     the same skeleton under a transposition sigma of two speakers (P = 3: the
           third is unchanged): every utterance spoken by a in ``x`` is spoken by b and
           vice versa, with vocatives transposed with it (so I/you re-derive), the scene
           line and the query line UNCHANGED. sigma is chosen so the true answer flips
           (owner: sigma moves the gold owner; self/addressee: sigma moves the queried
           person and a partner who also holds exactly one object). Its
           ``3shot_collide`` maps are sigma∘(the base maps), i.e. the twin's collide
           exemplars are the base's transposed too — which keeps "no collide exemplar
           answers the test gold" true for both members of the pair.

Shortcut controls (enforced per cell on the deictic items — the third variant
mirrors them exactly; the twin only AUDITS them, since its text keeps x's objects
while its gold moves; every variant records them in ``meta``):
  gold_is_last_mention   gold is the last-mentioned candidate-type string (names
                         for owner, objects for self/addressee); 50% per cell.
  gold_is_last_speaker   owner items: the gold owner speaks the last line; 50% per
                         owner cell (jointly scheduled with recency — see
                         ``_schedule``; a cell where some combinations are
                         infeasible uses the feasible diagonal).
  gold_in_last_line      the ownership-word rule: the gold object (owner items: the
                         queried object) is NOT named in the line immediately
                         preceding the query (a request's reply line — "Here you
                         go." / its narration — names no object). Exempt cells
                         (rule cannot hold): owner E=1 (spec) and P=2 E=1
                         self/addressee (both speakers must hold exactly one object,
                         so the only legal event is a swap of the two objects).
                         NB: "last EVENT" instead of "last line" would make the
                         recency control unbalanceable for object queries (the
                         last-mentioned object would then never be the gold — an
                         anti-recency shortcut that solves P=2 two-object items).

Error types (``error_types`` returns every label that applies, primary first):
  speaker_confusion  the wrong side of I/you. Owner query: the pick is the OTHER
                     party of the event that last moved the queried object (its
                     giver / the request's addressee / the swap partner). Object
                     queries: the pick is an object currently held by the query's
                     counterpart — addressee query: the query SPEAKER ("you" read
                     as "I"); self query: walking back from the last line, the
                     first line's speaker that is not the queried person, or the
                     addressee of a line the queried person spoke (P = 2: always the
                     other speaker).
  previous_owner     owner query: an earlier owner of the queried object. Object
                     queries: an object the queried person owned earlier but not now.
  other_participant  owner query, none of the above.
  other_object       object query, none of the above.
speaker_confusion is ranked first (the spec lists previous owner first): for owner
queries the other party of the last transfer is almost always also a previous
owner, so under the spec's order the deixis-specific label could never be
primary. The third variant is typed by the same rules on the same event
structure, so deictic - third differences per label are comparable.
"""

from __future__ import annotations

import random
import re
import string
from collections import defaultdict
from functools import cache
from typing import Any, Mapping, Optional, Sequence, TypedDict

from kotodama.eval.bindprobes.base import Cell, Item, cell_key, name_list, rejection
from kotodama.eval.bindprobes.pools import Pools

FAMILY = "deixis"
PS = (2, 3)
ES = (1, 2, 4)
QKINDS = ("owner", "self", "addressee")
CELLS: list[Cell] = [(p, e, q) for p in PS for e in ES for q in QKINDS]
DEFAULT_PER_CELL = 60             # 18 cells -> 1080 base items -> 3240 item texts
SEPARATOR = "\n\n* * *\n\n"
VARIANTS = ("deictic", "third", "twin")
ERROR_TYPES = ("speaker_confusion", "previous_owner", "other_participant", "other_object")
MAX_PROMPT_TOKENS = 1024          # spec: prompts < 1024 tokens (checked with the tokenizer)
N_OBJECTS = (2, 3, 4)
EVENT_WEIGHTS = (("give", 0.4), ("request", 0.3), ("swap", 0.3))
LITE_PER_CELL = 10                # lite slice: 10 base items (x 3 variants) per cell
BALANCE_TOL = 0.05                # spec: <= 50% ± 5% per cell

TRANSFERS = ("give", "request", "swap")
_YOU = re.compile(r"\byou(?:r|rs|'re)?\b", re.IGNORECASE)
_IMPERATIVE = re.compile(r"^(?:take|give|let's)\b", re.IGNORECASE)
_PRONOUNS = re.compile(r"\b(?:I|I'm|I'll|me|my|mine|you|your|yours|you're|he|him|his|she|"
                       r"her|hers|they|them|their|it)\b", re.IGNORECASE)


class Spec(TypedDict):
    variant: str                 # "deictic" | "third" | "twin"
    people: list[str]            # scene-line order (unchanged by the twin transposition)
    objects: list[str]           # objects in play, declaration order
    events: list[dict[str, Any]]  # surface events (deictic/twin) or explicit ones (third)
    query: dict[str, Any]
    sigma: Optional[list[str]]   # the twin transposition [a, b] (on every variant)


class Line(TypedDict):
    speaker: str
    addressee: Optional[str]
    objs: list[str]


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
            for k in ("owner", "giver", "recipient", "requester", "holder", "a", "b"):
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
            continue
        a = resolve_addressee(people, s, ev.get("voc"))
        if kind == "give":
            out.append({"ev": "give", "giver": s, "recipient": a, "obj": ev["obj"]})
        elif kind == "request":
            if ev["replier"] != a:
                raise ValueError(f"event {k}: {ev['replier']} replies to a request addressed "
                                 f"to {a}")
            back = resolve_addressee(people, ev["replier"], ev.get("reply_voc"))
            if back != s:
                raise ValueError(f"event {k}: the reply's 'you' is {back}, not the requester {s}")
            out.append({"ev": "request", "requester": s, "holder": a, "obj": ev["obj"]})
        elif kind == "swap":
            out.append({"ev": "swap", "a": s, "b": a, "a_obj": ev["give"], "b_obj": ev["take"]})
        else:
            raise ValueError(f"event {k}: unknown event {kind!r}")
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

    def move(o: str, frm: str, to: str, k: int) -> None:
        if owner.get(o) != frm:
            raise ValueError(f"event {k}: {frm} does not own the {o} (owner {owner.get(o)})")
        if frm == to:
            raise ValueError(f"event {k}: {frm} transfers to themself")
        owner[o] = to
        hist[o].append(to)

    for k, ev in enumerate(evs):
        kind = ev["ev"]
        if kind == "declare":
            o = ev["obj"]
            if seen_transfer:
                raise ValueError(f"event {k}: declaration after a transfer")
            if o not in objects or o in owner:
                raise ValueError(f"event {k}: the {o} is not an unowned object in play")
            owner[o] = ev["owner"]
            hist[o] = [ev["owner"]]
            continue
        seen_transfer = True
        if kind == "give":
            move(ev["obj"], ev["giver"], ev["recipient"], k)
        elif kind == "request":
            move(ev["obj"], ev["holder"], ev["requester"], k)
        elif kind == "swap":
            if ev["a_obj"] == ev["b_obj"]:
                raise ValueError(f"event {k}: swap of an object with itself")
            if owner.get(ev["b_obj"]) != ev["b"]:
                raise ValueError(f"event {k}: {ev['b']} does not own the {ev['b_obj']}")
            move(ev["a_obj"], ev["a"], ev["b"], k)
            owner[ev["b_obj"]] = ev["a"]
            hist[ev["b_obj"]].append(ev["a"])
        else:
            raise ValueError(f"event {k}: unknown event {kind!r}")
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
        kind = ev["ev"]
        if kind == "declare":
            out.append({"speaker": ev["owner"], "addressee": None, "objs": [ev["obj"]]})
        elif kind == "give":
            out.append({"speaker": ev["giver"], "addressee": ev["recipient"], "objs": [ev["obj"]]})
        elif kind == "request":
            out.append({"speaker": ev["requester"], "addressee": ev["holder"], "objs": [ev["obj"]]})
            out.append({"speaker": ev["holder"], "addressee": ev["requester"], "objs": []})
        else:
            out.append({"speaker": ev["a"], "addressee": ev["b"],
                        "objs": [ev["a_obj"], ev["b_obj"]]})
    return out


def last_line_objects(spec: Spec) -> list[str]:
    """Objects named in the line immediately preceding the query (a request's
    reply line names none)."""
    return lines(spec)[-1]["objs"]


def n_transfers(spec: Spec) -> int:
    return sum(1 for ev in spec["events"] if ev["ev"] in TRANSFERS)


# ── Rendering ──────────────────────────────────────────────────────────────────

_FMT = string.Formatter()


def _fill(tmpl: str, **kw: str) -> tuple[str, list[str]]:
    """Format ``tmpl`` and return (text, slot values in text order)."""
    slots = [kw[f] for _, f, _, _ in _FMT.parse(tmpl) if f]
    return tmpl.format(**kw), slots


def _utt(speaker: str, voc: Optional[str], body: str, **kw: str) -> tuple[str, list[str]]:
    """``Speaker: [Voc, ]body`` — the body is capitalized when there is no vocative."""
    if voc:
        return _fill("{_s}: {_v}, " + body, _s=speaker, _v=voc, **kw)
    return _fill("{_s}: " + body[0].upper() + body[1:], _s=speaker, **kw)


DECLARE_FORMS = {"have": "I have the {o}.", "mine": "the {o} is mine."}
GIVE_FORMS = {"take": "take my {o}.", "giving": "I'm giving you my {o}."}


def _deictic_lines(ev: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    kind, s = ev["ev"], ev["speaker"]
    if kind == "declare":
        return [_utt(s, None, DECLARE_FORMS[ev["form"]], o=ev["obj"])]
    if kind == "give":
        return [_utt(s, ev.get("voc"), GIVE_FORMS[ev["form"]], o=ev["obj"])]
    if kind == "request":
        return [_utt(s, ev.get("voc"), "give me your {o}.", o=ev["obj"]),
                _utt(ev["replier"], ev.get("reply_voc"), "here you go.")]
    if kind == "swap":
        return [_utt(s, ev.get("voc"), "let's swap: you take my {o1} and I'll take your {o2}.",
                     o1=ev["give"], o2=ev["take"])]
    raise ValueError(f"unknown event {kind!r}")


# Third-person line forms. ``form`` picks which party is named LAST in the line
# ("to": the receiving side / swap partner b; "from": the giving side / initiator
# a), so the narration can mirror the deictic line's last-mentioned name (P=2: the
# speaker tag; P=3: the addressee's vocative) — see ``to_third``. Object order
# always matches the deictic line, and the fulfil line names no object (as
# "Here you go." does not).
THIRD_FORMS: dict[str, dict[str, tuple[str, ...]]] = {
    "give": {"to": ("({g} gives the {o} to {r}.)",),
             "from": ("({r} gets the {o} from {g}.)",)},
    "request": {"to": ("({r} asks {h} for the {o}.)", "({h} gives {r} what {r2} asked for.)"),
                "from": ("({r} asks {h} for the {o}.)",
                         "({r} gets what {r2} asked for from {h}.)")},
    "swap": {"to": ("({a} and {b} swap: {a2} gives {b2} the {ao} and gets the {bo}.)",),
             "from": ("({a} and {b} swap: {b2} takes the {ao} and {a2} takes the {bo}.)",)},
}


def _third_lines(ev: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    kind = ev["ev"]
    if kind == "declare":
        return [_fill("({p} has the {o}.)", p=ev["owner"], o=ev["obj"])]
    if kind not in THIRD_FORMS:
        raise ValueError(f"unknown event {kind!r}")
    tmpls = THIRD_FORMS[kind][ev.get("form", "to")]
    if kind == "give":
        kw = {"g": ev["giver"], "r": ev["recipient"], "o": ev["obj"]}
    elif kind == "request":
        kw = {"r": ev["requester"], "r2": ev["requester"], "h": ev["holder"], "o": ev["obj"]}
    else:
        kw = {"a": ev["a"], "a2": ev["a"], "b": ev["b"], "b2": ev["b"], "ao": ev["a_obj"],
              "bo": ev["b_obj"]}
    return [_fill(t, **kw) for t in tmpls]


def _lead(spec: Spec) -> tuple[str, list[str], str]:
    """(query lead-in, its slots, the closer appended after the answer)."""
    q = spec["query"]
    if q["kind"] == "owner":
        t, s = _fill("(Now the {o} belongs to", o=q["obj"])
        return t, s, ".)"
    if spec["variant"] == "third":
        t, s = _fill("(Right now {p} is holding the", p=q["person"])
        return t, s, ".)"
    if q["kind"] == "self":
        t, s = _fill("{s}: Right now I'm holding the", s=q["speaker"])
        return t, s, "."
    t, s = _utt(q["speaker"], q.get("voc"), "right now you're holding the")
    return t, s, "."


def _rendered(spec: Spec) -> tuple[list[str], list[str]]:
    """(text lines incl. scene + lead, slot values in text order)."""
    scene, slots = _fill("A chat between " + name_list([f"{{p{i}}}" for i in
                                                        range(len(spec["people"]))]) + ".",
                         **{f"p{i}": p for i, p in enumerate(spec["people"])})
    out, mentions = [scene], list(slots)
    fn = _third_lines if spec["variant"] == "third" else _deictic_lines
    for ev in spec["events"]:
        for t, s in fn(ev):
            out.append(t)
            mentions += s
    t, s, _ = _lead(spec)
    return out + [t], mentions + s


def render(item: Item, with_answer: bool) -> str:
    spec: Spec = item.spec  # type: ignore[assignment]
    text_lines, _ = _rendered(spec)
    text = "\n".join(text_lines)
    return f"{text} {item.gold}{_lead(spec)[2]}" if with_answer else text


def check_surface(item: Item) -> None:
    """Surface-level law checks on the rendered text (independent of the simulator):
    deictic/twin — every P=3 utterance containing you/your/yours, or addressed by
    an imperative (take / give / let's), opens with a vocative naming another
    participant; P=2 utterances carry none; every line is ``Name: utterance``;
    third — no pronoun at all in the body or the question."""
    spec: Spec = item.spec  # type: ignore[assignment]
    people = spec["people"]
    body = render(item, False).split("\n")[1:]
    for ln in body:
        if spec["variant"] == "third" or ln.startswith("("):
            if not ln.startswith("("):
                raise ValueError(f"{item.item_id}: third-variant line is not narration: {ln!r}")
            if _PRONOUNS.search(ln):
                raise ValueError(f"{item.item_id}: pronoun in a narrated line: {ln!r}")
            continue
        m = re.match(r"^(\w+): (.*)$", ln)
        if not m or m.group(1) not in people:
            raise ValueError(f"{item.item_id}: not a 'Name: utterance' line: {ln!r}")
        spk, utt = m.groups()
        v = re.match(r"^(\w+), ", utt)
        if v and (v.group(1) not in people or v.group(1) == spk):
            raise ValueError(f"{item.item_id}: vocative is not another participant: {ln!r}")
        if len(people) == 3 and not v and (_YOU.search(utt) or _IMPERATIVE.match(utt)):
            raise ValueError(f"{item.item_id}: P=3 addressed utterance without a vocative: "
                             f"{ln!r}")
        if len(people) == 2 and re.match(r"^(\w+), ", utt):
            raise ValueError(f"{item.item_id}: P=2 utterance carries a vocative: {ln!r}")


# ── Items ──────────────────────────────────────────────────────────────────────

def labels(item: Item) -> list[str]:
    return list(item.spec["people"])


def objects(item: Item) -> set[str]:
    return set(item.spec["objects"])


def exempt_ownership_word(cell: Sequence[Any]) -> bool:
    p, e, kind = int(cell[0]), int(cell[1]), str(cell[2])
    return e == 1 and (kind == "owner" or p == 2)


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
              meta={"variant": spec["variant"], "pair_id": pair_id or item_id, "qkind": kind,
                    "n_objects": len(spec["objects"]),
                    "events": [ev["ev"] for ev in spec["events"] if ev["ev"] in TRANSFERS]})
    it.gold = simulate(it)
    body = lines(spec)
    gold_obj = spec["query"]["obj"] if kind == "owner" else it.gold
    it.meta.update({
        "query_person": query_person(spec),
        "gold_is_last_mention": bool(mentions) and mentions[-1] == it.gold,
        # the deictic transcript's last speaker (third: the speaker that line narrates)
        "gold_is_last_speaker": (body[-1]["speaker"] == it.gold) if kind == "owner" else None,
        "gold_in_last_line": gold_obj in last_line_objects(spec),
        "ownership_word_exempt": exempt_ownership_word(cell),
    })
    return it


# ── Generation ─────────────────────────────────────────────────────────────────

def _pick(rng: random.Random, feasible: set[str]) -> str:
    opts = [(f, w) for f, w in EVENT_WEIGHTS if f in feasible]
    x = rng.random() * sum(w for _, w in opts)
    for f, w in opts:
        x -= w
        if x <= 0:
            return f
    return opts[-1][0]


def _draw(rng: random.Random, cell: Cell, pools: Pools, item_id: str,
          template: Optional[Item]) -> Optional[Item]:
    """One random base (deictic) item for ``cell`` with its sigma, or None."""
    p, e, kind = int(cell[0]), int(cell[1]), str(cell[2])
    people = rng.sample(pools.names, p)
    n_obj = 2 if (p == 2 and kind != "owner") else rng.choice(N_OBJECTS)
    objs = rng.sample(pools.objects, n_obj)
    owner = {o: rng.choice(people) for o in objs}
    events: list[dict[str, Any]] = [{"ev": "declare", "speaker": owner[o], "obj": o,
                                     "form": rng.choice(tuple(DECLARE_FORMS))} for o in objs]

    def voc(x: str) -> Optional[str]:
        return x if p == 3 else None

    for _ in range(e):
        held = {x: [o for o in objs if owner[o] == x] for x in people}
        owners = [x for x in people if held[x]]
        feasible = {"give", "request"} | ({"swap"} if len(owners) >= 2 else set())
        kind_ev = _pick(rng, feasible)
        if kind_ev == "give":
            s = rng.choice(owners)
            a = rng.choice([x for x in people if x != s])
            o = rng.choice(held[s])
            events.append({"ev": "give", "speaker": s, "voc": voc(a), "obj": o,
                           "form": rng.choice(tuple(GIVE_FORMS))})
            owner[o] = a
        elif kind_ev == "request":
            a = rng.choice(owners)                      # the holder (addressee)
            s = rng.choice([x for x in people if x != a])
            o = rng.choice(held[a])
            events.append({"ev": "request", "speaker": s, "voc": voc(a), "obj": o,
                           "replier": a, "reply_voc": voc(s)})
            owner[o] = s
        else:
            s, a = rng.sample(owners, 2)
            o1, o2 = rng.choice(held[s]), rng.choice(held[a])
            events.append({"ev": "swap", "speaker": s, "voc": voc(a), "give": o1, "take": o2})
            owner[o1], owner[o2] = a, s
    spec: Spec = {"variant": "deictic", "people": people, "objects": objs, "events": events,
                  "query": {}, "sigma": None}
    _, hist = run(spec)
    fin = holdings(spec)
    decl = declared(spec)
    one = [x for x in people if len(fin[x]) == 1]
    if kind == "owner":
        moved = [o for o in objs if len(hist[o]) > 1]
        if not exempt_ownership_word(cell):
            moved = [o for o in moved if o not in last_line_objects(spec)]
        if not moved:
            return None
        o = rng.choice(moved)
        gold = owner[o]
        partner = rng.choice([x for x in people if x != gold])
        spec["query"] = {"kind": "owner", "obj": o}
        sigma = [gold, partner]
    else:
        targets = [x for x in one if set(fin[x]) != set(decl[x])]
        if not exempt_ownership_word(cell):
            targets = [x for x in targets if fin[x][0] not in last_line_objects(spec)]
        if not targets:
            return None
        q = rng.choice(targets)
        if kind == "self":
            partners = [x for x in one if x != q]
            if not partners:
                return None
            spec["query"] = {"kind": "self", "speaker": q}
            sigma = [q, rng.choice(partners)]
        else:
            s = rng.choice([x for x in people if x != q])
            partners = [x for x in one if x != q]
            if not partners:
                return None
            spec["query"] = {"kind": "addressee", "speaker": s, "voc": voc(q)}
            sigma = [q, rng.choice(partners)]
    spec["sigma"] = sigma
    it = build(item_id, cell, spec)
    # the collide re-label is positional: keep an owner exemplar's answer off the
    # test item's gold slot (so no collide exemplar answers the test gold)
    if template is not None and kind == "owner" and \
            people.index(it.gold) == template.spec["people"].index(template.gold):
        return None
    return it


# Recency × last-speaker schedules (owner cells), most to least balanced.
_FULL = ((True, True), (False, False), (True, False), (False, True))
_DIAG = ((True, True), (False, False))
_ANTI = ((True, False), (False, True))


@cache
def _schedule(cell: Cell) -> tuple[tuple[bool, Optional[bool]], ...]:
    """The per-k (gold_is_last_mention, gold_is_last_speaker) targets for a cell.

    Owner cells: the full 4-cycle if every combination occurs, else the feasible
    diagonal — both marginals stay 50% either way. (At P=2 the last-mentioned name
    IS the last speaker, so the two controls coincide; at P=3 the last name is the
    addressee's vocative, so they are disjoint.) Object cells: recency alternates.
    Feasibility is probed on a fixed-seed sample of draws from the banked item pool
    (deterministic; independent of the battery seed)."""
    from kotodama.eval.bindprobes.pools import load_pools
    pools = load_pools(None).item
    rng = random.Random(f"deixis-schedule:{cell_key(cell)}")
    seen: set[tuple[bool, Optional[bool]]] = set()
    for _ in range(4000):
        it = _draw(rng, cell, pools, "probe", None)
        if it is not None:
            ls = it.meta["gold_is_last_speaker"]
            seen.add((bool(it.meta["gold_is_last_mention"]), None if ls is None else bool(ls)))
        if len(seen) == 4:
            break
    scheds = (_FULL, _DIAG, _ANTI) if cell[2] == "owner" else (((True, None), (False, None)),)
    for sched in scheds:
        if set(sched) <= seen:
            return sched
    raise RuntimeError(f"deixis {cell}: no balanced recency/speaker schedule in {seen}")


def gen_item(rng: random.Random, cell: Cell, pools: Pools, item_id: str, k: int,
             per_cell: int, template: Optional[Item] = None) -> Item:
    """A BASE (deictic) item; :func:`expand` derives its third + twin variants."""
    if tuple(cell) not in CELLS:
        raise ValueError(f"deixis: unknown cell {cell}")
    cell = tuple(cell)
    want: Optional[tuple[bool, Optional[bool]]] = None
    if template is None:
        sched = _schedule(cell)
        want = sched[k % len(sched)]

    def draw() -> Optional[Item]:
        it = _draw(rng, cell, pools, item_id, template)
        if it is None or want is None:
            return it
        if it.meta["gold_is_last_mention"] != want[0]:
            return None
        if want[1] is not None and it.meta["gold_is_last_speaker"] != want[1]:
            return None
        return it

    return rejection(draw, f"deixis item {item_id}", tries=20000)


# ── Variants ───────────────────────────────────────────────────────────────────

def to_third(spec: Spec) -> Spec:
    """The same events and question narrated in the third person with names."""
    if spec["variant"] == "third":
        return spec
    q = dict(spec["query"])
    if q["kind"] != "owner":
        q = {"kind": q["kind"], "person": query_person(spec),
             "asker": q["speaker"] if q["kind"] == "addressee" else None}
    # mirror each deictic line's last-mentioned name: P=2 lines end on the speaker
    # tag's side (giver / fulfiller / initiator) -> "from"; P=3 lines end on the
    # addressee's vocative (recipient / requester / partner) -> "to"
    form = "from" if len(spec["people"]) == 2 else "to"
    events = [ev if ev["ev"] == "declare" else {**ev, "form": form}
              for ev in explicit_events(spec)]
    return {"variant": "third", "people": list(spec["people"]), "objects": list(spec["objects"]),
            "events": events, "query": q,
            "sigma": list(spec["sigma"]) if spec["sigma"] else None}


_NAME_KEYS = ("speaker", "voc", "replier", "reply_voc", "owner", "giver", "recipient",
              "requester", "holder", "a", "b", "person", "asker")


def _rename(d: Mapping[str, Any], m: Mapping[str, str]) -> dict[str, Any]:
    return {k: (m.get(v, v) if k in _NAME_KEYS and isinstance(v, str) else v)
            for k, v in d.items()}


def to_twin(spec: Spec) -> Spec:
    """Transpose the two speakers of ``spec['sigma']`` in every EVENT (speakers and
    vocatives), keeping the scene line and the query line unchanged."""
    if spec["variant"] != "deictic" or not spec["sigma"]:
        raise ValueError("the twin is built from a deictic spec with a sigma")
    a, b = spec["sigma"]
    sig = {a: b, b: a}
    return {"variant": "twin", "people": list(spec["people"]), "objects": list(spec["objects"]),
            "events": [_rename(ev, sig) for ev in spec["events"]],
            "query": dict(spec["query"]), "sigma": [a, b]}


def expand(base: Item) -> list[Item]:
    """[deictic, third, twin] for one base item (with shots + collide maps).

    deictic + twin share the deictic exemplars; the third variant's exemplars are
    the same exemplars narrated in the third person. The twin's collide maps are
    sigma∘(base maps)."""
    pid = base.item_id
    spec: Spec = base.spec  # type: ignore[assignment]
    shots = base.shot_items()
    out: list[Item] = []
    for variant in VARIANTS:
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
    newlines). Every template word, name and object is a common single-token word;
    measured SmolLM2 tokens / pieces on the default battery: 0.91-1.17."""
    return int(1.25 * len(re.findall(r"\w+|[^\w\s]|\n", text))) + 1


def validate_item(item: Item) -> None:
    """Deixis-specific invariants (called by ``items.validate``)."""
    spec: Spec = item.spec  # type: ignore[assignment]
    p, e, kind = item.cell
    if len(spec["people"]) != p or n_transfers(spec) != e or spec["query"]["kind"] != kind:
        raise ValueError(f"{item.item_id}: spec does not match cell {item.cell}")
    if item.meta.get("variant") != spec["variant"]:
        raise ValueError(f"{item.item_id}: meta variant != spec variant")
    check_surface(item)
    if not item.meta["ownership_word_exempt"] and item.meta["variant"] == "deictic" \
            and item.meta["gold_in_last_line"]:
        raise ValueError(f"{item.item_id}: ownership-word rule violated")


def final_state(spec: Spec) -> dict[str, str]:
    return run(spec)[0]


def check_pairs(items: Sequence[Item]) -> None:
    """Pair linking: every pair_id has exactly deictic/third/twin, same cell and
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
        if set(g) != set(VARIANTS):
            raise ValueError(f"pair {pid}: variants {sorted(g)}")
        x, t, w = g["deictic"], g["third"], g["twin"]
        if not (x.cell == t.cell == w.cell and x.candidates == t.candidates == w.candidates):
            raise ValueError(f"pair {pid}: cell/candidates differ across variants")
        fx = final_state(x.spec)  # type: ignore[arg-type]
        if final_state(t.spec) != fx or t.gold != x.gold:  # type: ignore[arg-type]
            raise ValueError(f"pair {pid}: third variant changes the state or gold")
        a, b = x.spec["sigma"]
        sig = {a: b, b: a}
        if final_state(w.spec) != {o: sig.get(h, h) for o, h in fx.items()}:  # type: ignore[arg-type]
            raise ValueError(f"pair {pid}: twin state is not sigma(deictic state)")
        if w.gold == x.gold:
            raise ValueError(f"pair {pid}: twin does not flip the answer")
        if w.spec["query"] != x.spec["query"]:
            raise ValueError(f"pair {pid}: twin query wording differs")


def audit(items: Sequence[Item]) -> dict[str, dict[str, dict[str, Any]]]:
    """Per cell × variant: n and the share of each shortcut control."""
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
            out.setdefault(ck, {})[v] = {
                "n": n,
                "gold_is_last_mention": sum(it.meta["gold_is_last_mention"] for it in sub) / n,
                "gold_is_last_speaker": (sum(ls) / len(ls)) if ls else None,
                "gold_in_last_line": sum(it.meta["gold_in_last_line"] for it in sub) / n,
                "ownership_word_exempt": exempt_ownership_word(c),
            }
    return out


def audit_failures(aud: Mapping[str, Mapping[str, Mapping[str, Any]]],
                   tol: float = BALANCE_TOL) -> list[str]:
    """Spec §2 controls on the deictic items: recency and (owner) last-speaker within
    50% ± tol; ownership-word rule never violated outside the exempt cells."""
    bad: list[str] = []
    for ck, per in aud.items():
        a = per.get("deictic")
        if a is None:
            continue
        if abs(a["gold_is_last_mention"] - 0.5) > tol + 1e-9:
            bad.append(f"{ck}: recency {a['gold_is_last_mention']:.3f}")
        if a["gold_is_last_speaker"] is not None and \
                abs(a["gold_is_last_speaker"] - 0.5) > tol + 1e-9:
            bad.append(f"{ck}: last-speaker {a['gold_is_last_speaker']:.3f}")
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
        last = [ev for ev in explicit_events(spec) if ev["ev"] in TRANSFERS
                and o in (ev.get("obj"), ev.get("a_obj"), ev.get("b_obj"))][-1]
        parties = {"give": ("giver", "recipient"), "request": ("holder", "requester"),
                   "swap": ("a", "b")}[last["ev"]]
        pair = {last[parties[0]], last[parties[1]]}
        if pick in pair and pick != item.gold:
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
