"""Bindprobes ``deixis`` v2: the simulator for give chains (incl. precondition
violations), pronoun/vocative resolution and the ambiguity raise, the three give
surfaces, candidate counts ≥ 3, owner only at P = 3, the single-held-object
invariant, the dialogue-worded third control, the twin flip (P = 3; no twin at
P = 2), shortcut-neutral balancing, determinism, prompt bound, pair linking,
collide re-label validity, error typing, the version guard, and the paired
readouts + validity/pilot gates on fake grids."""

from __future__ import annotations

import functools
import json
import re
from typing import Any, Optional

import pytest

from kotodama.eval.bindprobes import aggregate as agg
from kotodama.eval.bindprobes import deixis, gates, items, lite, pools
from kotodama.eval.bindprobes.__main__ import main as cli_main
from kotodama.eval.bindprobes.base import Item

SPLIT = pools.load_pools(None)
P3 = ["Ada", "Ken", "Rosa"]
P2 = ["Ada", "Ken"]


def _spec(people: list[str], objs: list[str], events: list[dict[str, Any]],
          query: dict[str, Any], sigma: Optional[list[str]] = None) -> deixis.Spec:
    return {"variant": "deictic", "people": people, "objects": objs, "events": events,
            "query": query, "sigma": sigma}


def _decl(s: str, o: str, form: str = "have") -> dict[str, Any]:
    return {"ev": "declare", "speaker": s, "obj": o, "form": form}


def _give(s: str, o: str, voc: Optional[str] = None, form: str = "giving") -> dict[str, Any]:
    return {"ev": "give", "speaker": s, "voc": voc, "obj": o, "form": form}


def _build(spec: deixis.Spec, cell: tuple[int, int, str]) -> Item:
    return deixis.build("x0", cell, spec)


# P=3 scene: Ken gives Rosa his cup; Rosa gives Ada her bell; Ada gives Ken the lamp.
OBJS3 = ["lamp", "cup", "bell"]
P3_EVENTS = [_decl("Ada", "lamp"), _decl("Ken", "cup", "mine"), _decl("Rosa", "bell"),
             _give("Ken", "cup", "Rosa", "here"), _give("Rosa", "bell", "Ada", "yours"),
             _give("Ada", "lamp", "Ken", "giving")]


# ── simulator: give chains + exact renders ─────────────────────────────────────

def test_give_chain_and_exact_render_p3():
    it = _build(_spec(P3, OBJS3, P3_EVENTS, {"kind": "self", "speaker": "Rosa"}), (3, 3, "self"))
    assert it.gold == "cup" and it.candidates == OBJS3
    assert deixis.holdings(it.spec) == {"Ada": ["bell"], "Ken": ["lamp"], "Rosa": ["cup"]}
    assert deixis.render(it, False) == (
        "A chat between Ada, Ken and Rosa.\n"
        "Ada: I have the lamp.\n"
        "Ken: The cup is mine.\n"
        "Rosa: I have the bell.\n"
        "Ken: Here, Rosa, take my cup.\n"
        "Rosa: Ada, my bell is yours now.\n"
        "Ada: Ken, I'm giving you my lamp.\n"
        "Rosa: Right now I'm holding the")
    assert deixis.render(it, True).endswith("Rosa: Right now I'm holding the cup.")
    third = deixis.build("t0", (3, 3, "self"), deixis.to_third(it.spec))
    assert deixis.render(third, False) == (
        "Ada, Ken and Rosa are sitting at a table. Ada has the lamp. Ken has the cup. "
        "Rosa has the bell. Ken gives the cup to Rosa. Rosa gives the bell to Ada. "
        "Ada gives the lamp to Ken. Now Rosa is holding the")
    assert deixis.render(third, True).endswith("holding the cup.") and third.gold == "cup"
    assert third.mention_order == it.mention_order
    own = _build(_spec(P3, OBJS3, P3_EVENTS, {"kind": "owner", "obj": "bell"}), (3, 3, "owner"))
    assert own.gold == "Ada" and own.candidates == P3
    assert deixis.render(own, False).endswith("Ken, I'm giving you my lamp.\nNow the bell is with")
    third_own = deixis.build("t1", (3, 3, "owner"), deixis.to_third(own.spec))
    assert deixis.render(third_own, False).endswith("to Ken. Now the bell is with")
    adr = _build(_spec(P3, OBJS3, P3_EVENTS, {"kind": "addressee", "speaker": "Ada", "voc": "Ken"}),
                 (3, 3, "addressee"))
    assert adr.gold == "lamp"
    assert deixis.render(adr, False).endswith("\nAda: Ken, right now you're holding the")
    assert deixis.render(deixis.build("t2", (3, 3, "addressee"), deixis.to_third(adr.spec)),
                         False).endswith("Now Ken is holding the")


def test_give_forms_p2_render():
    ev = [_decl("Ada", "lamp"), _decl("Ada", "cup", "mine"), _decl("Ken", "bell"),
          _give("Ada", "cup", form="here"), _give("Ken", "bell", form="yours")]
    it = _build(_spec(P2, ["lamp", "cup", "bell"], ev,
                      {"kind": "addressee", "speaker": "Ada", "voc": None}), (2, 2, "addressee"))
    assert it.gold == "cup"            # "you" = Ken at P=2, who now holds only the cup
    text = deixis.render(it, False)
    assert "Ada: The cup is mine.\nKen: I have the bell.\nAda: Here, take my cup.\n" \
           "Ken: My bell is yours now.\nAda: Right now you're holding the" in text
    ev2 = ev[:3] + [_give("Ada", "lamp", form="giving")]
    it2 = _build(_spec(P2, ["lamp", "cup", "bell"], ev2, {"kind": "self", "speaker": "Ada"}),
                 (2, 1, "self"))
    assert it2.gold == "cup" and "Ada: I'm giving you my lamp." in deixis.render(it2, False)
    third = deixis.to_third(it2.spec)
    assert deixis.final_state(third) == deixis.final_state(it2.spec)
    assert deixis._rendered(third)[0].startswith(
        "Ada and Ken are sitting at a table. Ada has the lamp and the cup. Ken has the bell. "
        "Ada gives the lamp to Ken.")


@pytest.mark.parametrize("events,why", [
    ([_decl("Ada", "lamp"), _decl("Ken", "lamp")], "declared twice"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _give("Ken", "lamp", "Rosa")], "give: not owner"),
    ([_decl("Ada", "lamp"), _give("Ada", "lamp", "Ken"), _decl("Ken", "cup")],
     "declaration after a transfer"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _give("Ada", "lamp", "Ada")], "give to self"),
    ([_decl("Ada", "lamp")], "an object never declared"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _give("Ada", "lamp")], "P=3 give, no vocative"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _give("Ada", "lamp", "Zed")],
     "vocative names nobody in the chat"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"),
      {"ev": "swap", "speaker": "Ada", "voc": "Ken", "give": "lamp", "take": "cup"}],
     "v1 swap is not a v2 event"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"),
      {"ev": "request", "speaker": "Ada", "voc": "Ken", "obj": "cup", "replier": "Ken",
       "reply_voc": "Ada"}], "v1 request is not a v2 event"),
])
def test_violations_raise(events, why):
    with pytest.raises(ValueError):
        _build(_spec(P3, ["lamp", "cup"], events, {"kind": "owner", "obj": "lamp"}), (3, 1, "owner"))


def test_owner_only_at_p3_and_candidate_counts():
    assert len(deixis.CELLS) == 15
    assert {c for c in deixis.CELLS if c[2] == "owner"} == {(3, e, "owner") for e in deixis.ES}
    assert deixis.ES == (1, 2, 3)
    ev = [_decl("Ada", "lamp"), _decl("Ken", "cup"), _give("Ada", "lamp")]
    with pytest.raises(ValueError):        # owner query at P=2 (2 candidates)
        _build(_spec(P2, ["lamp", "cup"], ev, {"kind": "owner", "obj": "lamp"}), (2, 1, "owner"))


def test_addressee_resolution_and_ambiguity():
    assert deixis.resolve_addressee(P2, "Ada", None) == "Ken"
    assert deixis.resolve_addressee(P2, "Ken", "Ada") == "Ada"
    assert deixis.resolve_addressee(P3, "Ada", "Rosa") == "Rosa"
    for args in ((P2, "Ada", "Rosa"), (P2, "Ada", "Ada"), (P3, "Ada", None), (P3, "Ada", "Ada"),
                 (P3, "Zed", "Ken")):
        with pytest.raises(ValueError):
            deixis.resolve_addressee(*args)
    with pytest.raises(ValueError):    # a P=3 addressee query without a vocative is ambiguous
        _build(_spec(P3, OBJS3, P3_EVENTS, {"kind": "addressee", "speaker": "Ada", "voc": None}),
               (3, 3, "addressee"))
    ev = P3_EVENTS[:4]                 # Rosa holds cup + bell: not exactly one object
    with pytest.raises(ValueError):
        _build(_spec(P3, OBJS3, ev, {"kind": "self", "speaker": "Rosa"}), (3, 1, "self"))
    # the surface checker catches hand-broken lines independent of the simulator
    it = _build(_spec(P3, OBJS3, P3_EVENTS, {"kind": "owner", "obj": "cup"}), (3, 3, "owner"))
    deixis.check_surface(it)
    broken = Item(**it.to_dict())
    orig = deixis.render
    try:
        for a, b in (("Here, Rosa, take", "Here, take"),        # imperative, no vocative
                     ("Rosa: Ada, my bell", "Rosa: My bell"),  # 'yours' without a vocative
                     ("Ken, I'm giving", "Ada, I'm giving"),   # vocative = the speaker
                     ("Now the cup is with", "(Now the cup is with")):   # parentheses
            deixis.render = functools.partial(_replaced, orig, a, b)
            with pytest.raises(ValueError):
                deixis.check_surface(broken)
    finally:
        deixis.render = orig


def _replaced(orig: Any, a: str, b: str, item: Item, with_answer: bool) -> str:
    return orig(item, with_answer).replace(a, b)


def test_error_types():
    own = _build(_spec(P3, OBJS3, P3_EVENTS, {"kind": "owner", "obj": "bell"}), (3, 3, "owner"))
    assert deixis.error_types(own, "Rosa") == ["speaker_confusion", "previous_owner"]
    assert deixis.error_types(own, "Ken") == ["other_participant"]
    # E=1, an object that never moved: no transfer party, only other_participant
    ev = [_decl("Ada", "lamp"), _decl("Ada", "cup"), _decl("Rosa", "bell"),
          _give("Ada", "cup", "Ken")]
    still = _build(_spec(P3, OBJS3, ev, {"kind": "owner", "obj": "lamp"}), (3, 1, "owner"))
    assert still.gold == "Ada" and not still.meta["queried_obj_moved"]
    assert deixis.error_types(still, "Ken") == ["other_participant"]
    slf = _build(_spec(P3, OBJS3, P3_EVENTS, {"kind": "self", "speaker": "Rosa"}), (3, 3, "self"))
    # counterpart = Ada (the last line's speaker, not Rosa); Ada holds the bell
    assert deixis.error_types(slf, "bell") == ["speaker_confusion", "previous_owner"]
    assert deixis.error_types(slf, "lamp") == ["other_object"]
    adr = _build(_spec(P3, OBJS3, P3_EVENTS, {"kind": "addressee", "speaker": "Ada", "voc": "Ken"}),
                 (3, 3, "addressee"))
    assert deixis.error_types(adr, "bell") == ["speaker_confusion"]   # Ada (the asker) holds it
    assert deixis.error_types(adr, "cup") == ["previous_owner"]
    third = deixis.build("t", (3, 3, "self"), deixis.to_third(slf.spec))
    for pick in ("lamp", "bell"):
        assert deixis.error_types(third, pick) == deixis.error_types(slf, pick)


# ── paired variants ────────────────────────────────────────────────────────────

def test_twin_flips_exactly_as_simulated_p3():
    x = _build(_spec(P3, OBJS3, P3_EVENTS, {"kind": "self", "speaker": "Rosa"}, ["Rosa", "Ken"]),
               (3, 3, "self"))
    tw = deixis.build("w", (3, 3, "self"), deixis.to_twin(x.spec), pair_id="x0")
    assert x.gold == "cup" and tw.gold == "lamp"
    assert deixis.final_state(tw.spec) == {"lamp": "Rosa", "cup": "Ken", "bell": "Ada"}
    xl, wl = deixis.render(x, False).split("\n"), deixis.render(tw, False).split("\n")
    assert xl[0] == wl[0] and xl[-1] == wl[-1]                      # scene + query unchanged
    assert wl[1:-1] == ["Ada: I have the lamp.", "Rosa: The cup is mine.", "Ken: I have the bell.",
                        "Rosa: Here, Ken, take my cup.", "Ken: Ada, my bell is yours now.",
                        "Ada: Rosa, I'm giving you my lamp."]


def test_no_twin_at_p2():
    assert deixis.variants((2, 1, "self")) == ("deictic", "third")
    assert deixis.variants((3, 1, "self")) == deixis.VARIANTS
    ev = [_decl("Ada", "lamp"), _decl("Ada", "cup"), _decl("Ken", "bell"), _give("Ada", "cup")]
    x = _build(_spec(P2, ["lamp", "cup", "bell"], ev, {"kind": "self", "speaker": "Ada"}),
               (2, 1, "self"))
    with pytest.raises(ValueError):
        deixis.to_twin(x.spec)


@pytest.fixture(scope="module")
def dx() -> list[Item]:
    return items.generate_family("deixis", SPLIT, 7, seed=3)


def _k(it: Item) -> int:
    return int(re.search(r"_(\d{3})_", it.item_id).group(1))   # type: ignore[union-attr]


def test_counts_candidates_and_single_held(dx):
    n_p3 = sum(1 for c in deixis.CELLS if c[0] == 3)
    assert len(dx) == 7 * (2 * len(deixis.CELLS) + n_p3)
    for it in dx:
        assert it.meta["deixis_version"] == deixis.VERSION
        assert it.item_id.startswith("deixis-v2_")
        assert len(it.candidates) >= 3
        if it.meta["qkind"] == "owner":
            assert it.cell[0] == 3 and it.candidates == it.spec["people"]
        else:
            assert it.candidates == it.spec["objects"] and len(it.candidates) in (3, 4)
            held = deixis.holdings(it.spec)[deixis.query_person(it.spec)]
            assert held == [it.gold]                                  # exactly one object
        assert set(it.meta["give_forms"]) <= set(deixis.GIVE_ORDER)
        if len(it.meta["give_forms"]) <= 3:                           # rotated: all distinct
            assert len(set(it.meta["give_forms"])) == len(it.meta["give_forms"])


def test_control_wording(dx):
    for it in dx:
        text = deixis.render(it, False)
        assert "(" not in text and ")" not in text
        if it.meta["variant"] != "third":
            continue
        assert "\n" not in text and ":" not in text
        assert text.startswith(f"{deixis.name_list(it.spec['people'])} are sitting at a table. ")
        assert not deixis._PRONOUNS.search(text)
        assert re.search(r"\. \w+ gives the \w+ to \w+\. ", text)
        q = it.spec["query"]
        lead = (f"Now the {q['obj']} is with" if q["kind"] == "owner"
                else f"Now {q['person']} is holding the")
        assert text.endswith(". " + lead)


def test_pairs_link_and_variants_agree(dx):
    deixis.check_pairs(dx)
    by: dict[str, dict[str, Item]] = {}
    for it in dx:
        by.setdefault(it.meta["pair_id"], {})[it.meta["variant"]] = it
    for pid, g in by.items():
        x, t = g["deictic"], g["third"]
        assert set(g) == set(deixis.variants(x.cell))
        fx = deixis.final_state(x.spec)
        assert deixis.final_state(t.spec) == fx and t.gold == x.gold
        if x.cell[2] == "owner":   # the query line is identical across variants
            assert deixis.render(x, False).split("\n")[-1] == deixis.render(t, False)[-len(
                deixis.render(x, False).split("\n")[-1]):]
        if "twin" in g:
            w = g["twin"]
            a, b = x.spec["sigma"]
            sig = {a: b, b: a}
            assert deixis.final_state(w.spec) == {o: sig.get(h, h) for o, h in fx.items()}
            assert w.gold != x.gold and w.spec["query"] == x.spec["query"]
            if x.cell[2] == "owner":
                assert w.gold == sig[x.gold]
        for v in g.values():   # every variant under every shot condition
            for c in items.CONDITIONS:
                assert items.build_prompt(v, c).endswith(deixis.render(v, False))
    with pytest.raises(ValueError):    # a P=3 pair missing its twin
        deixis.check_pairs([it for it in dx if it.meta["variant"] != "twin"])
    bad = [Item(**it.to_dict()) for it in dx]
    tw = next(it for it in bad if it.meta["variant"] == "twin")
    tw.spec = dict(tw.spec, events=next(it for it in dx if it.item_id == tw.meta["pair_id"]
                                       + "_deictic").spec["events"], variant="twin")
    with pytest.raises(ValueError):
        deixis.check_pairs(bad)


def test_balancing_follows_schedule_and_audit(dx):
    for it in dx:
        if it.meta["variant"] != "deictic":
            continue
        want = deixis.schedule(it.cell)[_k(it) % len(deixis.schedule(it.cell))]
        assert it.meta["gold_is_last_mention"] == want["lm"]
        if it.meta["qkind"] == "owner":
            assert it.meta["gold_is_last_speaker"] == want["ls"]
            if it.cell[1] >= 2:
                assert not it.meta["gold_in_last_line"] and it.meta["queried_obj_moved"]
            else:   # E=1: the recency slot queries the given object, the others did not move
                assert it.meta["queried_obj_moved"] == want["lm"]
        else:
            assert it.meta["n_objects"] == want["n"]
            # give-only: for object queries the last line's object IS the last mention
            assert it.meta["gold_in_last_line"] == it.meta["gold_is_last_mention"]
    aud = deixis.audit(dx)
    assert deixis.audit_failures(aud) == []
    for c in deixis.CELLS:
        a = aud["/".join(map(str, c))]
        assert a["third"]["gold_is_last_mention"] == a["deictic"]["gold_is_last_mention"]
        assert a["deictic"]["gold_is_last_mention"] <= 0.5 + deixis.BALANCE_TOL
    # the neutral rate: per object count, gold = last mention in exactly 1/n of a full cycle
    full = items.generate_family("deixis", SPLIT, 21, seed=4)
    for c in deixis.CELLS:
        sub = [it for it in full if it.cell == list(c) and it.meta["variant"] == "deictic"]
        if c[2] == "owner":
            assert sum(it.meta["gold_is_last_mention"] for it in sub) == 7
            assert sum(it.meta["gold_is_last_speaker"] for it in sub) == 7
        else:
            for n in (3, 4):
                sn = [it for it in sub if it.meta["n_objects"] == n]
                assert sum(it.meta["gold_is_last_mention"] for it in sn) * n == len(sn)
    ck = "3/2/owner"
    broken = {ck: {"deictic": {**aud[ck]["deictic"], "n": 60, "gold_is_last_mention": 0.5}}}
    assert deixis.audit_failures(broken) and "recency" in deixis.audit_failures(broken)[0]
    broken = {ck: {"deictic": {**aud[ck]["deictic"], "gold_in_last_line": 0.1}}}
    assert "ownership-word" in deixis.audit_failures(broken)[0]


def test_determinism():
    a = items.generate_family("deixis", SPLIT, 1, seed=5)
    b = items.generate_family("deixis", SPLIT, 1, seed=5)
    c = items.generate_family("deixis", SPLIT, 1, seed=6)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    assert [x.to_dict() for x in a] != [x.to_dict() for x in c]


def test_prompt_bound_approx(dx):
    for it in dx:
        for c in items.CONDITIONS:
            p = items.build_prompt(it, c)
            assert deixis.approx_tokens(p) < deixis.MAX_PROMPT_TOKENS
            assert len(p.encode()) < items.MAX_PROMPT_BYTES


def test_prompt_bound_real_tokenizer(dx):
    try:
        from transformers import AutoTokenizer

        from kotodama.presets import TOKENIZER_NAME
        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as exc:  # offline without a cache
        pytest.skip(f"tokenizer unavailable: {exc}")
    assert items.check_tokens(dx, tok) < deixis.MAX_PROMPT_TOKENS
    for it in dx[:30]:     # the tokenizer estimate stays an over-estimate
        p = items.build_prompt(it, "3shot")
        assert len(tok(p).input_ids) <= deixis.approx_tokens(p)


def test_collide_relabel_keeps_addressee_rule(dx):
    for it in dx:
        for c in items.collide_shots(it):
            deixis.check_surface(c)        # P=3 vocatives still name another participant
            assert c.spec["people"] == it.spec["people"]   # scene line in the item's order
            assert c.gold != it.gold
            assert c.meta["variant"] == ("third" if it.meta["variant"] == "third" else "deictic")
    x = next(it for it in dx if it.cell == [3, 2, "owner"] and it.meta["variant"] == "deictic")
    s = x.shot_items()[0]
    with pytest.raises(ValueError):    # not injective
        deixis.relabel(s, {p: x.spec["people"][0] for p in s.spec["people"]})
    with pytest.raises(ValueError):    # incomplete
        deixis.relabel(s, {s.spec["people"][0]: "Ada"})


def test_selftest_and_dispatch(dx):
    from kotodama.eval.bindprobes import errors
    for it in dx:
        items.validate(it, SPLIT)
        for c in it.candidates:
            if c != it.gold:
                assert errors.primary(it, c) in deixis.ERROR_TYPES
    assert errors.types_for("deixis") == deixis.ERROR_TYPES


def test_version_guard(dx):
    old = Item(**dx[0].to_dict())
    old.meta = {k: v for k, v in old.meta.items() if k != "deixis_version"}
    with pytest.raises(ValueError):
        items.validate(old)
    rows = _rows_for("m", dx[:6], True)
    v1_rows = [{k: v for k, v in r.items() if k != "deixis_version"} for r in rows]
    with pytest.raises(ValueError, match="deixis_version"):
        agg.aggregate(rows + v1_rows)
    with pytest.raises(ValueError, match="deixis_version"):
        agg.deixis_readouts([{**rows[0], "deixis_version": 1}])
    with pytest.raises(ValueError, match="deixis_version"):
        gates.gates(v1_rows, dx, "m", "m")
    with pytest.raises(ValueError, match="deixis_version"):
        gates.gates(rows, [old], "m", "m")
    agg.aggregate(rows)                # v2 rows are read


# ── readouts + gates on fake grids ─────────────────────────────────────────────

def _drow(model: str, cond: str, variant: str, kind: str, pair: str, correct: bool,
          n_cand: int = 3, item_id: Optional[str] = None, gold: str = "x") -> dict[str, Any]:
    return {"model": model, "item_id": item_id or f"{pair}_{variant}", "family": "deixis",
            "cell": [3, 2, kind], "cell_key": f"3/2/{kind}", "condition": cond,
            "nshots": 0 if cond == "0shot" else 3, "deixis_version": deixis.VERSION,
            "variant": variant, "pair_id": pair,
            "qkind": kind, "gold_is_last_mention": False, "gold_is_last_speaker": None,
            "gold_in_last_line": False, "gold": gold, "n_candidates": n_cand, "gold_lp": -1.0,
            "margin": 1.0 if correct else -1.0, "gold_rank_cand": 1 if correct else 2,
            "gold_rank_full": 1, "entropy": 1.0, "prompt_tokens": 9, "cand_lps": {}, "top": [],
            "multi_token_candidates": [], "pick": "x",
            "error_type": None if correct else "speaker_confusion", "error_types": []}


def _fake(model: str, x: list[int], tw: list[int], th: list[int]) -> list[dict[str, Any]]:
    """Per query kind and condition, 6 pairs (3 candidates) with the given patterns."""
    rows = []
    for cond in items.CONDITIONS:
        for kind in deixis.QKINDS:
            for k in range(6):
                pid = f"{model}-{kind}-{k}"
                rows += [_drow(model, cond, "deictic", kind, pid, bool(x[k])),
                         _drow(model, cond, "twin", kind, pid, bool(tw[k])),
                         _drow(model, cond, "third", kind, pid, bool(th[k]))]
    return rows


def test_readouts_known_numbers():
    rows = _fake("big", [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0])
    d = agg.deixis_readouts(rows)["big"]
    assert d["cc"]["deictic"]["3shot"]["all"] == pytest.approx(0.0)       # acc 1/3, C=3
    assert d["cc"]["third"]["3shot"]["owner"] == pytest.approx(0.5)       # acc 2/3
    assert d["penalty"]["3shot"]["self"] == pytest.approx(0.5)
    t = d["twin"]["3shot"]["all"]
    assert t["p_x"] == pytest.approx(1 / 3) and t["joint"] == pytest.approx(1 / 6)
    assert t["chance2"] == pytest.approx(1 / 9) and t["n_pairs"] == 18
    assert d["asymmetry"]["deictic"]["3shot"] == pytest.approx(0.0)
    assert d["errors"]["deictic"]["3shot"]["speaker_confusion"] == 1.0
    md, s = agg.aggregate(rows)
    assert s["summary"]["big|deixis"]["n"]["3shot"] == 18
    assert "DEIXIS PENALTY" in md and "TWIN CONSISTENCY" in md and s["deixis"]["big"]
    with pytest.raises(ValueError):
        agg.deixis_readouts([{**rows[0], "variant": None}])
    m = lite.summarize(rows)
    assert m["bindprobes/deixis/3shot/acc"] == pytest.approx(1 / 3)
    assert m["bindprobes/deixis.third/3shot/acc"] == pytest.approx(2 / 3)
    assert m["bindprobes/deixis/penalty"] == pytest.approx(0.5)
    # P=2 pairs without a twin: twin consistency reads only the pairs that have one
    no_twin = [r for r in rows if not (r["variant"] == "twin" and r["qkind"] == "self")]
    d2 = agg.deixis_readouts(no_twin)["big"]
    assert d2["twin"]["3shot"]["self"]["n_pairs"] == 0 and d2["twin"]["3shot"]["all"]["n_pairs"] == 12


def test_gates_v2_to_v5_on_fake_grid():
    rows = (_fake("small", [1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [1, 0, 0, 1, 0, 0])  # chance
            + _fake("big", [1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1])
            + _fake("leaky", [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]))
    res = {r.gate: r for r in gates.gates(rows, [], "small", "big", ["big"])}
    assert not res["V1"].passed                     # no items given
    assert res["V2"].passed and res["V2"].values["deictic"]["owner"] == pytest.approx(0.0)
    assert res["V3"].passed and res["V3"].values["third_cc"] == pytest.approx(1.0)
    assert res["V4"].passed
    assert res["V5"].passed and res["V5"].values["joint"] == 0.0
    res = {r.gate: r for r in gates.gates(rows, [], "big", "leaky")}   # ge2b = all but big
    assert not res["V2"].passed and not res["V3"].passed
    assert not res["V4"].passed and "leaky" in res["V4"].detail
    assert not res["V5"].passed
    res = {r.gate: r for r in gates.gates(rows, [], "nobody", "nobody", [])}
    assert not any(res[g].passed for g in ("V2", "V3", "V4", "V5"))


def test_pilot_gate_band():
    # 60 pairs per kind, 23 correct at C=3: acc .383, cc .075 -> inside .08, outside .05
    rows = []
    for cond in items.CONDITIONS:
        for kind in deixis.QKINDS:
            for k in range(60):
                pid = f"s-{kind}-{k}"
                rows += [_drow("small", cond, v, kind, pid, k < 23 if v != "twin" else k >= 37)
                         for v in deixis.VARIANTS]
                rows += [_drow("big", cond, v, kind, f"b-{kind}-{k}", v == "third" or k < 30)
                         for v in deixis.VARIANTS]
    full = {r.gate: r for r in gates.gates(rows, [], "small", "big")}
    pilot = {r.gate: r for r in gates.gates(rows, [], "small", "big", pilot=True)}
    assert pilot["V2"].values["deictic"]["self"] == pytest.approx(0.075)
    assert not full["V2"].passed and pilot["V2"].passed and pilot["V3"].passed
    assert not gates.verdict(list(pilot.values()), pilot=True)      # V1 fails: no items
    ok = [gates.GateResult("V1", True, "")] + [pilot[g] for g in ("V2", "V3")] + \
         [gates.GateResult("V4", False, ""), gates.GateResult("V5", False, "")]
    assert gates.verdict(ok, pilot=True) and not gates.verdict(ok)


def _rows_for(model: str, its: list[Item], correct: bool) -> list[dict[str, Any]]:
    return [{**_drow(model, c, it.meta["variant"], it.meta["qkind"], it.meta["pair_id"], correct,
                     len(it.candidates), it.item_id, it.gold), "cell": list(it.cell),
             "cell_key": it.cell_key} for it in its for c in items.CONDITIONS]


def test_gates_v1_and_cli(dx, tmp_path, capsys):
    rows = _rows_for("m", dx, True)
    v1 = gates.v1_construction(dx, rows)
    assert v1.passed, v1.detail
    assert not gates.v1_construction(dx, [{**rows[0], "gold": "nope"}]).passed
    assert not gates.v1_construction(dx, [{**rows[0], "multi_token_candidates": [" ab"]}]).passed
    tampered = [Item(**it.to_dict()) for it in dx]
    tampered[0].gold = next(c for c in tampered[0].candidates if c != tampered[0].gold)
    assert not gates.v1_construction(tampered, rows).passed
    gdir = tmp_path / "grids"
    gdir.mkdir()
    ipath = tmp_path / "items.jsonl"
    ipath.write_text("".join(json.dumps(it.to_dict()) + "\n" for it in dx))
    (gdir / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert cli_main(["gates", "--grids", str(gdir), "--items", str(ipath),
                     "--chance-model", "m", "--learn-model", "m"]) == 1    # V2/V5 fail (acc 1)
    out = json.loads((gdir / gates.GATES_JSON).read_text())
    assert [g["gate"] for g in out["gates"]] == ["V1", "V2", "V3", "V4", "V5"]
    assert out["gates"][0]["passed"] and out["gates"][2]["passed"] and not out["passed"]
    assert "V1 PASS" in capsys.readouterr().out
    assert cli_main(["gates", "--grids", str(gdir), "--items", str(ipath),
                     "--chance-model", "m", "--learn-model", "m", "--pilot"]) == 1
    printed = capsys.readouterr().out
    assert "PILOT GATE: FAIL" in printed and "(info)" in printed
    assert json.loads((gdir / gates.GATES_JSON).read_text())["pilot"] is True
    assert cli_main(["aggregate", "--grids", str(gdir)]) == 0
    # a v1 grid in the directory is refused, not pooled
    (gdir / "old.jsonl").write_text("".join(
        json.dumps({k: v for k, v in r.items() if k != "deixis_version"}) + "\n"
        for r in rows[:3]))
    assert cli_main(["aggregate", "--grids", str(gdir)]) == 2
    assert cli_main(["gates", "--grids", str(gdir), "--items", str(ipath),
                     "--chance-model", "m", "--learn-model", "m"]) == 2


def test_cli_gen_pilot_and_default(tmp_path, capsys):
    assert cli_main(["gen", "--out", str(tmp_path), "--families", "deixis",
                     "--per-cell", f"deixis={deixis.PILOT_PER_CELL}"]) == 0
    meta = json.loads((tmp_path / "emit_meta.json").read_text())
    n_p3 = sum(1 for c in deixis.CELLS if c[0] == 3)
    assert meta["n_items"]["deixis"] == 10 * (2 * 15 + n_p3) == 390
    assert meta["versions"]["deixis"] == deixis.VERSION
    assert meta["audit_failures"]["deixis"] == []
    loaded = items.load_items(tmp_path / "items.jsonl")
    assert sum(it.meta["variant"] == "deictic" for it in loaded) == 150
    sel = lite.select_items(loaded)
    assert len(sel) == len(loaded)                  # lite = 10 pairs/cell = the pilot size
    assert lite.default_per_family("deixis") == 150 and lite.default_per_family("attr") == 200
    assert cli_main(["gen", "--out", str(tmp_path), "--per-cell", "2"]) == 0
    meta = json.loads((tmp_path / "emit_meta.json").read_text())
    assert set(meta["n_items"]) == set(items.FAMILY_NAMES)
    assert meta["n_items"]["deixis"] == 2 * (2 * 15 + n_p3)
    assert meta["audit_failures"]["deixis"] == []


def test_query_form_b_same_items_same_gold() -> None:
    from kotodama.eval.bindprobes import deixis as dx
    from kotodama.eval.bindprobes.items import generate
    from kotodama.eval.bindprobes.pools import load_pools
    items = [it for it in generate(load_pools(None), 11, ["deixis"], {"deixis": 2}) if it.family == "deixis"]
    assert items
    changed = 0
    for it in items:
        b = dx.with_qform(it, "b")
        assert b.gold == it.gold
        if it.meta["qkind"] == "owner":
            assert b is it
            continue
        changed += 1
        assert b.item_id == it.item_id + "-qb" and b.meta["qform"] == "b"
        ta, tb = dx.render(it, False), dx.render(b, False)
        assert ta.rsplit("\n", 1)[0].rsplit(". ", 1)[0] == tb.rsplit("\n", 1)[0].rsplit(". ", 1)[0]  # same body
        assert "hand" in tb.splitlines()[-1] or "hand" in tb[-60:]
        assert "holding" not in tb[-60:]
        assert len(b.shots) == len(it.shots)
        for sa, sb in zip(it.shot_items(), b.shot_items()):
            assert sa.gold == sb.gold and "hand" in dx.render(sb, True)[-80:]
        dx.check_surface(b)
    assert changed > 0
