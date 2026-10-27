"""``dialogue`` — multi-party scene referents, in prose (NOT user/assistant).

3–4 named people sit at a table holding 3 objects; objects are handed around,
partly through pronouns that resolve only by tracking the scene::

    Lucy, Tom and Anna are sitting at a table. Lucy has the cup. Tom has the
    lamp and the bell. Lucy hands the cup to Tom. He passes it to Anna. Anna
    asks Tom for the lamp, so he hands it over. Now the cup is with
    -> " Anna"

Transfer forms (each counts as ONE transfer):
  give   ``G hands the o to R.`` / ``G hands the o1 and the o2 to R.``
  chain  ``<G|He|She> passes <it|them> to R.`` — ``it``/``them`` = the object(s)
         of the PREVIOUS transfer (number must agree); a subject pronoun resolves
         to the one person of that gender among the previous transfer's giver and
         recipient (only emitted when their genders differ, so it is unique)
  ask    ``R asks G for the o, so <he|she> gives it to <him|her>.`` — only for a
         mixed-gender pair, so both pronouns are unique. As the last sentence of a
         holder item (the rule below) it names the gold: ``…, so G gives it to
         <him|her>.`` when the gold is G, ``R asks G for the o, so <he|she> hands it
         over.`` (``surface: "over"``; the recipient slot holds R) when the gold is R
Language rules (generation and ``validate_item``):
  * a sentence-initial He/She is written only when the previous sentence names
    its referent after the last pronoun for them, or the scene holds no one else
    of that gender (:func:`pronoun_hazard`); after an ask ("…so he gives it to
    her.") with a second woman at the table the chain names the giver instead;
  * no object goes straight back: a transfer never returns an object of the
    previous transfer from its recipient to its giver (:func:`gives_back`);
  * objects agree with "it" and "is" (``pools.singular``: no "dice");
  * the last sentence never refers to a holder query's gold with a pronoun
    (:func:`final_pronoun_for`, :func:`name_final_gold`), so the query "…is with"
    never follows a "him"/"her" that names the answer. Each rewrite keeps the
    sentence's last-mentioned name, so the recency control is drawn as before:
    "…hands it over." leaves the giver (not the gold) last, as "…gives it to
    her." does, keeps the giver's pronoun and ends on no person pronoun.

Cell = (n_people ∈ {3,4}, transfers ∈ {1,2,4}, query kind ∈ {holder, object}):
  holder  ``Now the cup is with`` -> a name; candidates = every person. The
          queried object has moved at least once.
  object  ``Now Anna has the``   -> an object; candidates = every object. The
          queried person holds exactly one object, and their holdings changed.
At transfers >= 2 every item has at least one pronoun transfer (chain or ask).
Recency control, balanced 50/50 in every cell: ``gold_is_last_mention`` = the gold
is the last-mentioned name (holder) / object (object query) in the text.
Holder-query exemplars never answer with the test item's gold slot, so under
``3shot_collide`` (positional re-label) no exemplar answer equals the test gold —
as in the other families, whose exemplar answers are shot-pool objects.

The simulator resolves the SURFACE ops (pronouns included) — ``resolve`` is the
one place pronoun semantics live. Error types: ``previous`` (holder query: an
earlier holder of the object; object query: an object the person held earlier),
``other_person`` (anyone else / another person's current object), ``other``.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional, TypedDict

from kotodama.eval.bindprobes.base import Cell, Item, for_version, name_list, rejection, the_list, version_of
from kotodama.eval.bindprobes.pools import (FEMALE_CANDIDATES, MALE_CANDIDATES, PLURAL_FORM,
                                            Gender, Pools, singular)

FAMILY = "dialogue"
VERSION = 3                       # meta.dialogue_version; items' id/seed tag "dialogue-v3"
# The holder query's lead-in per generation (the object query has one form). v2's subject-
# complement frame ("the person with the o is") renders v2 items only: it is a harder question
# for every model whether or not the gold is the last name mentioned, so it does not control
# the "her" that "…is with" invites after "…gives it to her."; v3 keeps that pronoun out of
# the last sentence instead.
HOLDER_LEADS: dict[int, str] = {1: "Now the {o} is with", 2: "Now the person with the {o} is",
                                3: "Now the {o} is with"}
N_PEOPLE = (3, 4)
TRANSFERS = (1, 2, 4)
QKINDS = ("holder", "object")
CELLS: list[Cell] = [(n, t, q) for n in N_PEOPLE for t in TRANSFERS for q in QKINDS]
DEFAULT_PER_CELL = 100            # 12 cells -> 1200 items
SEPARATOR = "\n\n* * *\n\n"
N_OBJECTS = 3
ERROR_TYPES = ("previous", "other_person", "other")
SUBJ: dict[Gender, str] = {"m": "he", "f": "she"}
OBJ: dict[Gender, str] = {"m": "him", "f": "her"}
PRON_GENDER: dict[str, Gender] = {"he": "m", "him": "m", "she": "f", "her": "f"}
FORM_WEIGHTS = (("chain", 0.4), ("ask", 0.25), ("give1", 0.25), ("give2", 0.1))


class Spec(TypedDict):
    people: list[list[str]]        # [[name, gender], ...] in intro order
    holdings0: list[list[str]]     # [[obj, holder], ...] in intro mention order
    ops: list[dict[str, Any]]      # surface ops (see module doc)
    query: dict[str, str]          # {"kind": "holder", "obj": o} | {"kind": "object", "person": p}


class Resolved(TypedDict):
    giver: str
    recipient: str
    objs: list[str]


def resolve(spec: Spec) -> tuple[list[Resolved], dict[str, list[str]]]:
    """Resolve every surface op (pronouns included) and run the transfers.

    Returns (resolved ops, per-object holder history incl. the initial holder).
    Raises ``ValueError`` on any ambiguity, agreement or possession violation."""
    gender: dict[str, Gender] = {n: g for n, g in spec["people"]}  # type: ignore[misc]
    if len(gender) != len(spec["people"]):
        raise ValueError("duplicate people")
    holder: dict[str, str] = {}
    for obj, who in spec["holdings0"]:
        if who not in gender or obj in holder:
            raise ValueError(f"bad initial holding {obj!r} -> {who!r}")
        holder[obj] = who
    hist: dict[str, list[str]] = {o: [w] for o, w in holder.items()}
    out: list[Resolved] = []

    def unique(pron: str, among: list[str]) -> str:
        g = PRON_GENDER.get(pron)
        if g is None:
            raise ValueError(f"unknown pronoun {pron!r}")
        hits = list(dict.fromkeys(p for p in among if gender[p] == g))
        if len(hits) != 1:
            raise ValueError(f"pronoun {pron!r} is ambiguous/unresolvable among {among}")
        return hits[0]

    for k, op in enumerate(spec["ops"]):
        form = op["form"]
        if form == "give":
            giver, recip, objs = op["giver"], op["recipient"], list(op["objs"])
        elif form == "chain":
            if not out:
                raise ValueError(f"op {k}: '{op['pronoun']}' has no antecedent")
            prev = out[-1]
            objs = list(prev["objs"])
            want = {"it": 1, "them": 2}.get(op["pronoun"])
            if want != len(objs):
                raise ValueError(f"op {k}: '{op['pronoun']}' does not agree with {objs}")
            subj = op["subj"]
            giver = subj if subj in gender else unique(subj, [prev["giver"], prev["recipient"]])
            recip = op["recipient"]
        elif form == "ask":
            pair = [op["asker"], op["askee"]]
            # a slot holds a pronoun, or (the last sentence's gold) the name itself
            giver, recip = [x if x in pair else unique(x, pair) for x in (op["subj_pron"], op["obj_pron"])]
            if (giver, recip) != (op["askee"], op["asker"]):
                raise ValueError(f"op {k}: ask pronouns do not resolve askee -> asker")
            objs = [op["obj"]]
        else:
            raise ValueError(f"op {k}: unknown form {form!r}")
        if recip not in gender or giver not in gender or recip == giver:
            raise ValueError(f"op {k}: bad giver/recipient {giver!r} -> {recip!r}")
        for o in objs:
            if holder.get(o) != giver:
                raise ValueError(f"op {k}: {giver} does not hold the {o}")
            holder[o] = recip
            hist[o].append(recip)
        out.append({"giver": giver, "recipient": recip, "objs": objs})
    return out, hist


def gives_back(prev: Optional[Resolved], giver: str, recipient: str, objs: list[str]) -> bool:
    """True when a transfer hands an object of the previous transfer straight back
    (previous G -> R, this R -> G, a shared object): the ping-pong the generator
    never writes."""
    return (prev is not None and giver == prev["recipient"] and recipient == prev["giver"]
            and bool(set(objs) & set(prev["objs"])))


def _names_in(op: Mapping[str, Any]) -> list[str]:
    """The people an op's sentence names (pronouns excluded), in text order."""
    if op["form"] == "give":
        return [op["giver"], op["recipient"]]
    if op["form"] == "chain":
        return ([op["subj"]] if op["subj"] not in PRON_GENDER else []) + [op["recipient"]]
    return [op["asker"], op["askee"]] + _ask_named(op)


def _ask_named(op: Mapping[str, Any]) -> list[str]:
    """The names an ask writes in its pronoun slots, in text order (the "over" surface
    writes no recipient slot: its recipient is the asker, named first)."""
    slots = ("subj_pron",) if op.get("surface") == "over" else ("subj_pron", "obj_pron")
    return [op[k] for k in slots if op[k] not in PRON_GENDER]


def pronoun_hazard(prev_op: Mapping[str, Any], referent: str,
                   gender: Mapping[str, Gender]) -> Optional[str]:
    """Why a sentence-initial He/She for ``referent`` after ``prev_op``'s sentence
    would read as ambiguous, or None when it is clear. Unclear when the previous
    sentence names two people of the referent's gender, or when its last mention of
    the referent is itself a pronoun (an ask ends "…gives it to her.") and the scene
    holds another person of that gender."""
    g = gender[referent]
    named = set(_names_in(prev_op))
    if sum(1 for n in named if gender[n] == g) >= 2:
        return f"the previous sentence names two people of gender {g!r}"
    same = [n for n in gender if gender[n] == g]
    if prev_op["form"] == "ask" and prev_op["obj_pron"] in PRON_GENDER and len(same) > 1:
        return f"the referent {referent} was last a pronoun, with {len(same)} of gender {g!r} present"
    return None


def final_pronoun_for(spec: Spec) -> Optional[str]:
    """The pronoun the last transfer sentence uses for a holder query's gold, or None (an
    object query, or a last sentence that names the gold or does not mention them)."""
    if spec["query"].get("kind") != "holder" or not spec["ops"]:
        return None
    res, hist = resolve(spec)
    gold = hist[spec["query"]["obj"]][-1]
    op, last = spec["ops"][-1], res[-1]
    if op["form"] == "chain" and op["subj"] in PRON_GENDER and last["giver"] == gold:
        return str(op["subj"])
    if op["form"] == "ask":
        if op["subj_pron"] in PRON_GENDER and last["giver"] == gold:
            return str(op["subj_pron"])
        if op["obj_pron"] in PRON_GENDER and last["recipient"] == gold:
            return str(op["obj_pron"])
    return None


def name_final_gold(spec: Spec) -> Spec:
    """``spec`` with the last sentence naming a holder query's gold where it used a pronoun
    (:func:`final_pronoun_for`); the transfers and the gold are unchanged."""
    if final_pronoun_for(spec) is None:
        return spec
    res, hist = resolve(spec)
    gold = hist[spec["query"]["obj"]][-1]
    op = dict(spec["ops"][-1])
    if op["form"] == "chain":
        op["subj"] = gold
    elif res[-1]["giver"] == gold:
        op["subj_pron"] = gold
    else:
        op.update(surface="over", obj_pron=gold)
    return {**spec, "ops": spec["ops"][:-1] + [op]}  # type: ignore[typeddict-item]


def final_holdings(spec: Spec) -> dict[str, list[str]]:
    """person -> objects held at the end (people order; objects in intro order)."""
    _, hist = resolve(spec)
    out: dict[str, list[str]] = {n: [] for n, _ in spec["people"]}
    for obj, _ in spec["holdings0"]:
        out[hist[obj][-1]].append(obj)
    return out


def held_ever(spec: Spec, person: str) -> list[str]:
    _, hist = resolve(spec)
    return [o for o, _ in spec["holdings0"] if person in hist[o]]


def simulate(item: Item) -> str:
    spec: Spec = item.spec  # type: ignore[assignment]
    q = spec["query"]
    _, hist = resolve(spec)
    if q["kind"] == "holder":
        return hist[q["obj"]][-1]
    held = final_holdings(spec)[q["person"]]
    if len(held) != 1:
        raise ValueError(f"{item.item_id}: {q['person']} holds {held}, not exactly one")
    return held[0]


# ── Rendering ──────────────────────────────────────────────────────────────────

def _op_sentence(op: dict[str, Any]) -> str:
    form = op["form"]
    if form == "give":
        return f"{op['giver']} hands {the_list(op['objs'])} to {op['recipient']}."
    if form == "chain":
        subj = op["subj"]
        subj = subj.capitalize() if subj in PRON_GENDER else subj
        return f"{subj} passes {op['pronoun']} to {op['recipient']}."
    if form == "ask" and op.get("surface") == "over":
        if op["obj_pron"] != op["asker"]:
            raise ValueError(f"an 'over' ask holds the asker in its recipient slot, not {op['obj_pron']!r}")
        return (f"{op['asker']} asks {op['askee']} for the {op['obj']}, so "
                f"{op['subj_pron']} hands it over.")
    if form == "ask":
        return (f"{op['asker']} asks {op['askee']} for the {op['obj']}, so "
                f"{op['subj_pron']} gives it to {op['obj_pron']}.")
    raise ValueError(f"unknown form {form!r}")


def _intro(spec: Spec) -> list[str]:
    names = [n for n, _ in spec["people"]]
    out = [f"{name_list(names)} are sitting at a table."]
    for n in names:
        objs = [o for o, w in spec["holdings0"] if w == n]
        if objs:
            out.append(f"{n} has {the_list(objs)}.")
    return out


def _lead(spec: Spec, version: int) -> str:
    q = spec["query"]
    return (for_version(HOLDER_LEADS, version, FAMILY).format(o=q["obj"]) if q["kind"] == "holder"
            else f"Now {q['person']} has the")


def render(item: Item, with_answer: bool) -> str:
    spec: Spec = item.spec  # type: ignore[assignment]
    body = " ".join(_intro(spec) + [_op_sentence(o) for o in spec["ops"]])
    lead = f"{body} {_lead(spec, version_of(item))}"
    return f"{lead} {item.gold}." if with_answer else lead


def _mentions(spec: Spec) -> list[str]:
    """Candidate-type strings (names for holder queries, objects otherwise) in
    text order; pronouns are not mentions."""
    names = [n for n, _ in spec["people"]]
    if spec["query"]["kind"] == "holder":
        out = list(names)
        out += [n for n in names if any(w == n for _, w in spec["holdings0"])]
        for op in spec["ops"]:
            if op["form"] == "give":
                out += [op["giver"], op["recipient"]]
            elif op["form"] == "chain":
                out += ([op["subj"]] if op["subj"] in names else []) + [op["recipient"]]
            else:
                out += [op["asker"], op["askee"]] + _ask_named(op)
        return out
    out = [o for n in names for o, w in spec["holdings0"] if w == n]
    for op in spec["ops"]:
        if op["form"] == "give":
            out += list(op["objs"])
        elif op["form"] == "ask":
            out.append(op["obj"])
    return out


def labels(item: Item) -> list[str]:
    return [n for n, _ in item.spec["people"]]


def build(item_id: str, cell: Cell, spec: Spec, version: int = VERSION) -> Item:
    """Item from a spec; ``version`` is the generation it renders as (``relabel`` keeps the
    source item's)."""
    names = [n for n, _ in spec["people"]]
    objs_intro = [o for n in names for o, w in spec["holdings0"] if w == n]
    if [o for o, _ in spec["holdings0"]] != objs_intro:
        raise ValueError(f"{item_id}: holdings0 must be in intro (people) order")
    cands = names if spec["query"]["kind"] == "holder" else objs_intro
    mentions = _mentions(spec)
    ops = spec["ops"]
    n_pron = sum(1 for o in ops if o["form"] in ("chain", "ask"))
    it = Item(item_id=item_id, family=FAMILY, cell=list(cell), spec=dict(spec), gold="",
              candidates=list(cands), mention_order=mentions,
              meta={"dialogue_version": version, "qkind": spec["query"]["kind"],
                    "n_pronoun_ops": n_pron,
                    "forms": [o["form"] for o in ops], "gold_is_last_mention": False})
    it.gold = simulate(it)
    it.meta["gold_is_last_mention"] = bool(mentions) and mentions[-1] == it.gold
    return it


def validate_item(item: Item) -> None:
    """The language rules (module doc) on a generated item; raises ``ValueError``."""
    spec: Spec = item.spec  # type: ignore[assignment]
    if item.meta.get("dialogue_version") != VERSION:
        raise ValueError(f"{item.item_id}: dialogue_version {item.meta.get('dialogue_version')!r} "
                         f"!= {VERSION}")
    plural = sorted({o for o, _ in spec["holdings0"]} & PLURAL_FORM)
    if plural:
        raise ValueError(f"{item.item_id}: plural-form object(s) {plural} read with 'it'/'is'")
    gender: dict[str, Gender] = {n: g for n, g in spec["people"]}  # type: ignore[misc]
    res, _ = resolve(spec)
    ops = spec["ops"]
    for k in range(1, len(ops)):
        if gives_back(res[k - 1], res[k]["giver"], res[k]["recipient"], res[k]["objs"]):
            raise ValueError(f"{item.item_id}: op {k} hands an object straight back")
        if ops[k]["form"] == "chain" and ops[k]["subj"] in PRON_GENDER:
            why = pronoun_hazard(ops[k - 1], res[k]["giver"], gender)
            if why is not None:
                raise ValueError(f"{item.item_id}: op {k} opens with an unclear pronoun: {why}")
    pron = final_pronoun_for(spec)
    if pron is not None:
        raise ValueError(f"{item.item_id}: the last sentence calls the gold {pron!r}")


# ── Generation ─────────────────────────────────────────────────────────────────

def _pick_form(rng: random.Random, feasible: set[str]) -> str:
    forms = [(f, w) for f, w in FORM_WEIGHTS if f in feasible]
    total = sum(w for _, w in forms)
    x = rng.random() * total
    for f, w in forms:
        x -= w
        if x <= 0:
            return f
    return forms[-1][0]


def gen_item(rng: random.Random, cell: Cell, pools: Pools, item_id: str, k: int,
             per_cell: int, template: Optional[Item] = None) -> Item:
    n, n_tr, qkind = int(cell[0]), int(cell[1]), str(cell[2])
    if (n, n_tr, qkind) not in CELLS:
        raise ValueError(f"dialogue: unknown cell {cell}")
    # recency control: test items alternate gold == / != the last-mentioned
    # candidate-type string; exemplars are unconstrained
    want_last: Optional[bool] = (k % 2 == 0) if template is None else None

    def draw() -> Optional[Item]:
        if template is not None:
            genders: list[Gender] = [g for _, g in template.spec["people"]]  # type: ignore[misc]
        else:
            genders = [rng.choice(("f", "m")) for _ in range(n)]
            if len(set(genders)) < 2:
                return None
        fem = rng.sample(pools.female, genders.count("f"))
        mal = rng.sample(pools.male, genders.count("m"))
        names = [fem.pop() if g == "f" else mal.pop() for g in genders]
        gender = dict(zip(names, genders))
        objs = rng.sample(singular(pools.objects), N_OBJECTS)
        first = {o: rng.choice(names) for o in objs}
        holdings0 = [[o, w] for w in names for o in objs if first[o] == w]
        holder = dict(first)
        ops: list[dict[str, Any]] = []
        prev: Optional[Resolved] = None
        for _ in range(n_tr):
            held = {p: [o for o in objs if holder[o] == p] for p in names}
            feasible = {"give1"}
            if any(len(v) >= 2 for v in held.values()):
                feasible.add("give2")
            if prev is not None:
                feasible.add("chain")
            if any(held[p] and any(gender[r] != gender[p] for r in names) for p in names):
                feasible.add("ask")
            form = _pick_form(rng, feasible)
            if form in ("give1", "give2"):
                need = 1 if form == "give1" else 2
                g = rng.choice([p for p in names if len(held[p]) >= need])
                moved = rng.sample(held[g], need)
                r = rng.choice([p for p in names
                                if p != g and not gives_back(prev, g, p, moved)])
                ops.append({"form": "give", "giver": g, "recipient": r, "objs": moved})
            elif form == "chain":
                assert prev is not None
                g, moved = prev["recipient"], list(prev["objs"])
                r = rng.choice([p for p in names
                                if p != g and not gives_back(prev, g, p, moved)])
                subj = g
                if gender[prev["giver"]] != gender[g] \
                        and pronoun_hazard(ops[-1], g, gender) is None and rng.random() < 0.6:
                    subj = SUBJ[gender[g]]
                ops.append({"form": "chain", "subj": subj, "recipient": r,
                            "pronoun": "it" if len(moved) == 1 else "them"})
            else:
                g = rng.choice([p for p in names if held[p]
                                and any(gender[x] != gender[p] for x in names)])
                moved = [rng.choice(held[g])]
                askers = [p for p in names
                          if gender[p] != gender[g] and not gives_back(prev, g, p, moved)]
                if not askers:
                    return None
                r = rng.choice(askers)
                ops.append({"form": "ask", "asker": r, "askee": g, "obj": moved[0],
                            "subj_pron": SUBJ[gender[g]], "obj_pron": OBJ[gender[r]]})
            for o in moved:
                holder[o] = r
            prev = {"giver": g, "recipient": r, "objs": moved}
        if n_tr >= 2 and not any(o["form"] in ("chain", "ask") for o in ops):
            return None
        spec: Spec = {"people": [[p, gender[p]] for p in names], "holdings0": holdings0,
                      "ops": ops, "query": {}}
        _, hist = resolve(spec)
        if qkind == "holder":
            choices = [o for o in objs if len(hist[o]) > 1]
            if not choices:
                return None
            spec["query"] = {"kind": "holder", "obj": rng.choice(choices)}
            spec = name_final_gold(spec)
        else:
            fin = final_holdings(spec)
            choices = [p for p in names if len(fin[p]) == 1
                       and set(fin[p]) != {o for o, w in holdings0 if w == p}]
            if not choices:
                return None
            spec["query"] = {"kind": "object", "person": rng.choice(choices)}
        it = build(item_id, cell, spec)
        assert it.gold == (holder[spec["query"]["obj"]] if qkind == "holder"
                           else [o for o in objs if holder[o] == spec["query"]["person"]][0])
        if want_last is not None and it.meta["gold_is_last_mention"] != want_last:
            return None
        if template is not None and qkind == "holder":
            # the collide re-label is positional: keep the exemplar's answer off the
            # test item's gold slot, so a collide exemplar can only interfere
            t_names = labels(template)
            if names.index(it.gold) == t_names.index(template.gold):
                return None
        return it

    return rejection(draw, f"dialogue item {item_id}", tries=5000)


def relabel(item: Item, mapping: Mapping[str, str], target: Optional[Item] = None) -> Item:
    """Rename people. The map must preserve gender (pronouns stay valid): target
    genders come from ``target``'s people (the collide case: the test item), else
    from the curated name candidates; a gender-changing map raises."""
    spec: Spec = item.spec  # type: ignore[assignment]
    missing = set(labels(item)) - set(mapping)
    if missing:
        raise ValueError(f"{item.item_id}: people {sorted(missing)} not in the map")
    known: dict[str, str] = {n: "f" for n in FEMALE_CANDIDATES}
    known.update({n: "m" for n in MALE_CANDIDATES})
    if target is not None:
        known.update({n: g for n, g in target.spec["people"]})
    for n, g in spec["people"]:
        if known.get(mapping[n]) != g:
            raise ValueError(f"{item.item_id}: map {n}->{mapping[n]} does not keep gender {g!r}")

    def m(x: str) -> str:
        return mapping.get(x, x) if x not in PRON_GENDER else x

    ops = []
    for op in spec["ops"]:
        o = dict(op)
        for key in ("giver", "recipient", "subj", "asker", "askee", "subj_pron", "obj_pron"):
            if key in o:
                o[key] = m(o[key])
        ops.append(o)
    q = dict(spec["query"])
    if "person" in q:
        q["person"] = mapping[q["person"]]
    new: Spec = {"people": [[mapping[n], g] for n, g in spec["people"]],
                 "holdings0": [[o, mapping[w]] for o, w in spec["holdings0"]],
                 "ops": ops, "query": q}
    out = build(f"{item.item_id}_collide", tuple(item.cell), new, version=version_of(item))
    expect = mapping[item.gold] if q["kind"] == "holder" else item.gold
    if out.gold != expect:
        raise ValueError(f"{item.item_id}: re-label changed the gold")
    return out


def error_types(item: Item, pick: str) -> list[str]:
    spec: Spec = item.spec  # type: ignore[assignment]
    q = spec["query"]
    _, hist = resolve(spec)
    if q["kind"] == "holder":
        earlier = set(hist[q["obj"]][:-1]) - {item.gold}
        if pick in earlier:
            return ["previous"]
        return ["other_person"] if pick in labels(item) else ["other"]
    t: list[str] = []
    if pick in held_ever(spec, q["person"]):
        t.append("previous")
    if any(pick in objs for p, objs in final_holdings(spec).items() if p != q["person"]):
        t.append("other_person")
    return t or ["other"]
