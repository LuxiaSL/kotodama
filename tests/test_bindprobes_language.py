"""Bindprobes language layer: the article helper and count-noun pools, no mass noun after
"a"/"an", no plural-form object read as singular, no object handed straight back, no
unclear sentence-initial pronoun in ``dialogue``, a name-shaped holder slot, an answer
asker who is not a party to the queried object's last event, and holding words only in
the chat families."""

from __future__ import annotations

import re
from typing import Any

import pytest

from kotodama.eval.bindprobes import adjacency, answer, deixis, dialogue, items, pools
from kotodama.eval.bindprobes.base import Item

SPLIT = pools.load_pools(None)
PER_CELL = {"attr": 6, "setbind": 6, "dialogue": 8, "deixis": 3, "answer": 8, "adjacency": 6}
_MASS_AFTER_A = re.compile(r"\b[Aa]n? (" + "|".join(sorted(pools.NON_COUNTABLE)) + r")\b")


@pytest.fixture(scope="module")
def gen() -> dict[str, list[Item]]:
    return {f: items.generate_family(f, SPLIT, n, seed=21) for f, n in PER_CELL.items()}


def _with_shots(its: list[Item]) -> list[Item]:
    return [x for it in its for x in [it] + it.shot_items()]


# ── articles and count nouns ───────────────────────────────────────────────────

@pytest.mark.parametrize("noun,art", [("cup", "a"), ("apple", "an"), ("onion", "an"),
                                      ("orange", "an"), ("envelope", "an"), ("hour", "an"),
                                      ("umbrella", "an"), ("uniform", "a"), ("one", "a"),
                                      ("Lamp", "a")])
def test_indefinite_article(noun, art):
    assert pools.indefinite(noun) == art


@pytest.mark.parametrize("noun", sorted(pools.NON_COUNTABLE) + [""])
def test_indefinite_refuses_non_count_nouns(noun):
    with pytest.raises(ValueError):
        pools.indefinite(noun)


def test_countable_and_singular_pools():
    objs = SPLIT.shot.objects + SPLIT.item.objects
    assert pools.PLURAL_FORM <= pools.NON_COUNTABLE <= set(objs)
    assert not set(pools.countable(objs)) & pools.NON_COUNTABLE
    assert set(pools.singular(objs)) == set(objs) - pools.PLURAL_FORM
    # every count noun of the pool takes an article without raising
    assert all(pools.indefinite(o) in ("a", "an") for o in pools.countable(objs))


def test_no_mass_noun_after_an_article(gen):
    for f in ("attr", "setbind"):
        for it in gen[f]:
            assert not set(items._objects(it)) & pools.NON_COUNTABLE, it.item_id
            for cond in items.CONDITIONS:
                p = items.build_prompt(it, cond)
                assert not _MASS_AFTER_A.search(p), (it.item_id, cond)
                assert not re.search(r"\b[Aa] [aeiou]", p) and not re.search(r"\bAn [^aeiou]", p)
        assert any(" lies on the shelf" in items.build_prompt(it, "0shot") for it in gen[f])


def test_no_plural_form_object_in_singular_slots(gen):
    for f in ("dialogue", "deixis", "answer", "adjacency"):
        for it in _with_shots(gen[f]):
            assert not items._objects(it) & pools.PLURAL_FORM, it.item_id


# ── F16: no object handed straight back ────────────────────────────────────────

def _transfers(it: Item) -> list[tuple[str, str, frozenset[str]]]:
    s: Any = it.spec
    if it.family == "dialogue":
        res, _ = dialogue.resolve(s)
        return [(r["giver"], r["recipient"], frozenset(r["objs"])) for r in res]
    if it.family == "deixis":
        evs = deixis.explicit_events(s)
    elif it.family == "answer":
        evs = answer.explicit_events(s)
    else:
        evs = [{"ev": "give", **t} for t in adjacency.transfers(s)]
    return [(e["giver"], e["recipient"], frozenset([e["obj"]])) for e in evs if e["ev"] == "give"]


def test_no_ping_pong(gen):
    n_multi = 0
    for f in ("dialogue", "deixis", "answer", "adjacency"):
        for it in _with_shots(gen[f]):
            trs = _transfers(it)
            n_multi += len(trs) >= 2
            for a, b in zip(trs, trs[1:]):
                assert not (b[0] == a[1] and b[1] == a[0] and a[2] & b[2]), it.item_id
    assert n_multi > 100


PEOPLE = [["Lucy", "f"], ["Tom", "m"], ["Anna", "f"]]
HOLD = [["cup", "Lucy"], ["lamp", "Tom"], ["bell", "Tom"]]


def _dl(ops: list[dict[str, Any]], query: dict[str, str], cell: tuple = (3, 2, "holder")) -> Item:
    return dialogue.build("d0", cell, {"people": PEOPLE, "holdings0": HOLD, "ops": ops,
                                       "query": query})


def test_dialogue_validate_rejects_give_back_and_unclear_pronoun():
    back = _dl([{"form": "give", "giver": "Lucy", "recipient": "Tom", "objs": ["cup"]},
                {"form": "chain", "subj": "he", "pronoun": "it", "recipient": "Lucy"}],
               {"kind": "holder", "obj": "cup"})
    with pytest.raises(ValueError, match="straight back"):
        dialogue.validate_item(back)
    # after an ask the asker was last "her"; with Anna at the table "She" is unclear
    ask = {"form": "ask", "asker": "Lucy", "askee": "Tom", "obj": "lamp",
           "subj_pron": "he", "obj_pron": "her"}
    unclear = _dl([ask, {"form": "chain", "subj": "she", "pronoun": "it", "recipient": "Anna"}],
                  {"kind": "holder", "obj": "lamp"})
    assert unclear.gold == "Anna"
    with pytest.raises(ValueError, match="unclear pronoun"):
        dialogue.validate_item(unclear)
    named = _dl([ask, {"form": "chain", "subj": "Lucy", "pronoun": "it", "recipient": "Anna"}],
                {"kind": "holder", "obj": "lamp"})
    dialogue.validate_item(named)
    assert "Lucy passes it to Anna. Now the person with the lamp is" in dialogue.render(named, False)
    # the same pronoun is clear when no other woman is in the scene
    duo = dialogue.build("d1", (3, 2, "holder"), {
        "people": [["Lucy", "f"], ["Tom", "m"], ["Ben", "m"]],
        "holdings0": HOLD[:1] + [["lamp", "Tom"], ["bell", "Tom"]],
        "ops": [ask, {"form": "chain", "subj": "she", "pronoun": "it", "recipient": "Ben"}],
        "query": {"kind": "holder", "obj": "lamp"}})
    dialogue.validate_item(duo)


def test_pronoun_hazard_rules():
    g = {"Lucy": "f", "Tom": "m", "Anna": "f"}
    give = {"form": "give", "giver": "Lucy", "recipient": "Anna", "objs": ["cup"]}
    assert dialogue.pronoun_hazard(give, "Anna", g) is not None      # two women named
    give_mf = {"form": "give", "giver": "Tom", "recipient": "Anna", "objs": ["cup"]}
    assert dialogue.pronoun_hazard(give_mf, "Anna", g) is None
    ask = {"form": "ask", "asker": "Anna", "askee": "Tom", "obj": "cup",
           "subj_pron": "he", "obj_pron": "her"}
    assert dialogue.pronoun_hazard(ask, "Anna", g) is not None
    assert dialogue.pronoun_hazard(ask, "Anna", {"Anna": "f", "Tom": "m", "Ben": "m"}) is None


def test_dialogue_pronouns_clear_and_holder_slot_name_shaped(gen):
    n_pron = 0
    for it in _with_shots(gen["dialogue"]):
        s: Any = it.spec
        gender = {n: g for n, g in s["people"]}
        res, _ = dialogue.resolve(s)
        for k, op in enumerate(s["ops"]):
            if op["form"] == "chain" and op["subj"] in dialogue.PRON_GENDER:
                n_pron += 1
                assert dialogue.pronoun_hazard(s["ops"][k - 1], res[k]["giver"], gender) is None
        text = dialogue.render(it, False)
        assert not text.endswith(" is with")
        if s["query"]["kind"] == "holder":
            assert text.endswith(f"Now the person with the {s['query']['obj']} is")
    assert n_pron > 20


# ── F11: the answer asker is not a party ───────────────────────────────────────

def test_answer_asker_not_party(gen):
    qa = [it for it in gen["answer"] if it.meta["variant"] == "qa"]
    assert {it.meta["role"] for it in qa} == {"me", "you", "other"}
    for it in _with_shots(gen["answer"]):
        s: Any = it.spec
        if it.meta["role"] == "you":
            assert s["tail"]["asker"] == it.meta["holder"]       # what "you" names
            continue
        assert not answer.asker_is_party(s), it.item_id
        assert s["tail"]["asker"] not in answer.last_event_parties(s)
    # hand-break one: the asker becomes the giver of the queried object's last give
    bad = next(it for it in qa if it.meta["role"] == "other" and it.meta["queried_obj_moved"])
    giver = (answer.last_event_parties(bad.spec) - {bad.meta["holder"]}).pop()
    tail = {**bad.spec["tail"], "asker": giver}
    if tail["answerer"] == giver:
        tail["answerer"] = bad.spec["tail"]["asker"]
    broken = answer.build("b0", tuple(bad.cell), {**bad.spec, "tail": tail})  # type: ignore[typeddict-item]
    assert broken.meta["asker_is_party"] and broken.meta["role"] != "you"
    with pytest.raises(ValueError, match="asker"):
        answer.validate_item(broken)


# ── F10: holding words only ────────────────────────────────────────────────────

def test_chat_families_state_holding_not_ownership(gen):
    for f in ("deixis", "answer", "adjacency"):
        for it in gen[f]:
            for cond in ("0shot", "3shot"):
                assert not re.search(r"\b(mine|yours)\b", items.build_prompt(it, cond)), it.item_id
    # deictic and third twins keep the same events, gold and candidates
    by: dict[str, dict[str, Item]] = {}
    for it in gen["deixis"]:
        by.setdefault(it.meta["pair_id"], {})[it.meta["variant"]] = it
    for g in by.values():
        d, t = g["deictic"], g["third"]
        assert d.gold == t.gold and d.candidates == t.candidates
        assert deixis.final_state(d.spec) == deixis.final_state(t.spec)  # type: ignore[arg-type]


def test_deixis_check_no_give_back():
    evs = [{"ev": "declare", "owner": "Ada", "obj": "cup"},
           {"ev": "give", "giver": "Ada", "recipient": "Ken", "obj": "cup"},
           {"ev": "give", "giver": "Ken", "recipient": "Ada", "obj": "cup"}]
    with pytest.raises(ValueError, match="straight back"):
        deixis.check_no_give_back(evs, "x")
    deixis.check_no_give_back(evs[:2] + [{**evs[2], "recipient": "Rosa"}], "x")


def test_dialogue_renders_each_generation_with_its_own_holder_lead():
    it = _dl([{"form": "give", "giver": "Lucy", "recipient": "Tom", "objs": ["cup"]}],
             {"kind": "holder", "obj": "cup"}, (3, 1, "holder"))
    leads = {v: dialogue.render(dialogue.build("d0", tuple(it.cell), it.spec, version=v), False)  # type: ignore[arg-type]
             for v in (1, 2)}
    assert leads[1].endswith("Now the cup is with")
    assert leads[2].endswith("Now the person with the cup is")
    v1 = dialogue.build("d0", tuple(it.cell), it.spec, version=1)  # type: ignore[arg-type]
    del v1.meta["dialogue_version"]                 # the first generation carried no key
    assert dialogue.render(v1, False) == leads[1]
    assert dialogue.relabel(v1, {"Lucy": "Emma", "Tom": "Ben", "Anna": "Kate"}).meta["dialogue_version"] == 1
    with pytest.raises(ValueError, match="no renderer for version 9"):
        dialogue.render(dialogue.build("d0", tuple(it.cell), it.spec, version=9), False)  # type: ignore[arg-type]


def test_chat_wordings_per_generation():
    own = {"ev": "declare", "speaker": "Ada", "obj": "cup", "form": "mine"}
    assert deixis._deictic_line(own, "owning")[0] == "Ada: The cup is mine."
    with pytest.raises(ValueError, match="not in the 'holding' wording"):
        deixis._deictic_line(own, "holding")
    give = {"ev": "give", "speaker": "Ada", "voc": "Ken", "obj": "cup", "form": "yours"}
    assert deixis._deictic_line(give, "owning")[0] == "Ada: Ken, my cup is yours now."
    assert deixis.WORDING == {2: "owning", 3: "holding"} and answer.WORDING == {2: "owning", 3: "holding"}
    assert adjacency.WORDING[1][0] == "owning" and "yours" in adjacency.WORDING[1][1]
    assert pools.letter_article("hour") == "a" and pools.indefinite("hour") == "an"
