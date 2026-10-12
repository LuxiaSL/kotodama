"""Bindprobes ``deixis``: the simulator for every event (incl. precondition
violations), pronoun/vocative resolution and the ambiguity raise, the paired
variants (third = identical state, twin = sigma-flipped state), balancing,
determinism, prompt bound, pair linking, collide re-label validity, error typing,
and the paired readouts + validity gates on fake grids."""

from __future__ import annotations

import functools
import json
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


def _give(s: str, o: str, voc: Optional[str] = None, form: str = "take") -> dict[str, Any]:
    return {"ev": "give", "speaker": s, "voc": voc, "obj": o, "form": form}


def _req(s: str, o: str, replier: str, voc: Optional[str] = None,
         reply_voc: Optional[str] = None) -> dict[str, Any]:
    return {"ev": "request", "speaker": s, "voc": voc, "obj": o, "replier": replier,
            "reply_voc": reply_voc}


def _swap(s: str, give: str, take: str, voc: Optional[str] = None) -> dict[str, Any]:
    return {"ev": "swap", "speaker": s, "voc": voc, "give": give, "take": take}


def _build(spec: deixis.Spec, cell: tuple[int, int, str]) -> Item:
    return deixis.build("x0", cell, spec)


# P=3 scene used by several tests: Ken gives Rosa his cup; Ada requests Rosa's bell.
P3_EVENTS = [_decl("Ada", "lamp"), _decl("Ken", "cup", "mine"), _decl("Rosa", "bell"),
             _give("Ken", "cup", "Rosa"), _req("Ada", "bell", "Rosa", "Rosa", "Ada")]


# ── simulator: every event ────────────────────────────────────────────────────

def test_every_event_and_exact_render_p3():
    it = _build(_spec(P3, ["lamp", "cup", "bell"], P3_EVENTS, {"kind": "self", "speaker": "Rosa"}),
                (3, 2, "self"))
    assert it.gold == "cup" and it.candidates == ["lamp", "cup", "bell"]
    assert deixis.holdings(it.spec) == {"Ada": ["lamp", "bell"], "Ken": [], "Rosa": ["cup"]}
    assert deixis.render(it, False) == (
        "A chat between Ada, Ken and Rosa.\n"
        "Ada: I have the lamp.\n"
        "Ken: The cup is mine.\n"
        "Rosa: I have the bell.\n"
        "Ken: Rosa, take my cup.\n"
        "Ada: Rosa, give me your bell.\n"
        "Rosa: Ada, here you go.\n"
        "Rosa: Right now I'm holding the")
    assert deixis.render(it, True).endswith("Rosa: Right now I'm holding the cup.")
    third = deixis.build("t0", (3, 2, "self"), deixis.to_third(it.spec))
    assert deixis.render(third, False) == (
        "A chat between Ada, Ken and Rosa.\n"
        "(Ada has the lamp.)\n"
        "(Ken has the cup.)\n"
        "(Rosa has the bell.)\n"
        "(Ken gives the cup to Rosa.)\n"
        "(Ada asks Rosa for the bell.)\n"
        "(Rosa gives Ada what Ada asked for.)\n"
        "(Right now Rosa is holding the")
    assert deixis.render(third, True).endswith("holding the cup.)") and third.gold == "cup"
    # owner + addressee queries on the same scene
    own = _build(_spec(P3, ["lamp", "cup", "bell"], P3_EVENTS, {"kind": "owner", "obj": "bell"}),
                 (3, 2, "owner"))
    assert own.gold == "Ada" and own.candidates == P3
    assert deixis.render(own, False).endswith("\n(Now the bell belongs to")
    adr = _build(_spec(P3, ["lamp", "cup", "bell"], P3_EVENTS,
                       {"kind": "addressee", "speaker": "Ada", "voc": "Rosa"}), (3, 2, "addressee"))
    assert adr.gold == "cup"
    assert deixis.render(adr, False).endswith("\nAda: Rosa, right now you're holding the")


def test_swap_and_give_forms_p2():
    ev = [_decl("Ada", "lamp", "mine"), _decl("Ken", "cup"), _swap("Ken", "cup", "lamp")]
    it = _build(_spec(P2, ["lamp", "cup"], ev, {"kind": "self", "speaker": "Ada"}), (2, 1, "self"))
    assert it.gold == "cup" and deixis.holdings(it.spec) == {"Ada": ["cup"], "Ken": ["lamp"]}
    text = deixis.render(it, False)
    assert "Ken: Let's swap: you take my cup and I'll take your lamp." in text
    assert "Ada: The lamp is mine." in text
    ev2 = [_decl("Ada", "lamp"), _decl("Ada", "cup"), _give("Ada", "cup", form="giving")]
    it2 = _build(_spec(P2, ["lamp", "cup"], ev2,
                       {"kind": "addressee", "speaker": "Ada", "voc": None}), (2, 1, "addressee"))
    assert it2.gold == "cup"   # "you" = Ken at P=2, who now holds the cup
    assert "Ada: I'm giving you my cup.\nAda: Right now you're holding the" in deixis.render(it2, False)
    third = deixis.to_third(it2.spec)
    assert deixis.final_state(third) == deixis.final_state(it2.spec)
    assert "(Ken gets the cup from Ada.)" in "\n".join(deixis._rendered(third)[0])  # P=2 mirror


@pytest.mark.parametrize("events,why", [
    ([_decl("Ada", "lamp"), _decl("Ken", "lamp")], "declared twice"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _give("Ken", "lamp", "Rosa")], "give: not owner"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _req("Ada", "lamp", "Rosa", "Rosa", "Ada")],
     "request: addressee does not hold it"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _req("Ada", "cup", "Rosa", "Ken", "Ada")],
     "request: replier is not the addressee"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _req("Ada", "cup", "Ken", "Ken", "Rosa")],
     "request: reply 'you' is not the requester"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _swap("Ada", "lamp", "cup", "Rosa")],
     "swap: addressee lacks the object"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _swap("Ada", "lamp", "lamp", "Ken")],
     "swap: an object with itself"),
    ([_decl("Ada", "lamp"), _give("Ada", "lamp", "Ken"), _decl("Ken", "cup")],
     "declaration after a transfer"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _give("Ada", "lamp", "Ada")], "give to self"),
    ([_decl("Ada", "lamp")], "an object never declared"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _give("Ada", "lamp")], "P=3 give, no vocative"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _req("Ada", "cup", "Ken", "Ken", None)],
     "P=3 reply, no vocative"),
    ([_decl("Ada", "lamp"), _decl("Ken", "cup"), _give("Ada", "lamp", "Zed")],
     "vocative names nobody in the chat"),
])
def test_violations_raise(events, why):
    with pytest.raises(ValueError):
        _build(_spec(P3, ["lamp", "cup"], events, {"kind": "owner", "obj": "lamp"}), (3, 1, "owner"))


def test_addressee_resolution_and_ambiguity():
    assert deixis.resolve_addressee(P2, "Ada", None) == "Ken"
    assert deixis.resolve_addressee(P2, "Ken", "Ada") == "Ada"
    assert deixis.resolve_addressee(P3, "Ada", "Rosa") == "Rosa"
    for args in ((P2, "Ada", "Rosa"), (P2, "Ada", "Ada"), (P3, "Ada", None), (P3, "Ada", "Ada"),
                 (P3, "Zed", "Ken")):
        with pytest.raises(ValueError):
            deixis.resolve_addressee(*args)
    # a P=3 addressee query without a vocative is ambiguous
    with pytest.raises(ValueError):
        _build(_spec(P3, ["lamp", "cup", "bell"], P3_EVENTS,
                     {"kind": "addressee", "speaker": "Ada", "voc": None}), (3, 2, "addressee"))
    # self/addressee queries need exactly one held object
    with pytest.raises(ValueError):
        _build(_spec(P3, ["lamp", "cup", "bell"], P3_EVENTS, {"kind": "self", "speaker": "Ada"}),
               (3, 2, "self"))
    # the surface checker catches a hand-broken P=3 line independent of the simulator
    it = _build(_spec(P3, ["lamp", "cup", "bell"], P3_EVENTS, {"kind": "owner", "obj": "cup"}),
                (3, 2, "owner"))
    deixis.check_surface(it)
    broken = Item(**it.to_dict())
    orig = deixis.render
    try:
        for a, b in (("Ada, here you go", "Here you go"),    # 'you' without a vocative
                     ("Rosa, take", "Take"),                  # an imperative without one
                     ("Rosa, give me", "Ada, give me")):      # vocative = the speaker
            deixis.render = functools.partial(_replaced, orig, a, b)
            with pytest.raises(ValueError):
                deixis.check_surface(broken)
    finally:
        deixis.render = orig


def _replaced(orig: Any, a: str, b: str, item: Item, with_answer: bool) -> str:
    return orig(item, with_answer).replace(a, b)


def test_error_types():
    own = _build(_spec(P3, ["lamp", "cup", "bell"], P3_EVENTS, {"kind": "owner", "obj": "bell"}),
                 (3, 2, "owner"))
    assert deixis.error_types(own, "Rosa") == ["speaker_confusion", "previous_owner"]
    assert deixis.error_types(own, "Ken") == ["other_participant"]
    cup = _build(_spec(P3, ["lamp", "cup", "bell"], P3_EVENTS, {"kind": "owner", "obj": "cup"}),
                 (3, 2, "owner"))
    assert cup.gold == "Rosa" and deixis.error_types(cup, "Ken") == ["speaker_confusion",
                                                                     "previous_owner"]
    slf = _build(_spec(P3, ["lamp", "cup", "bell"], P3_EVENTS, {"kind": "self", "speaker": "Rosa"}),
                 (3, 2, "self"))
    # counterpart = Ada (Rosa's last line addressed her)
    assert deixis.error_types(slf, "lamp") == ["speaker_confusion"]
    assert deixis.error_types(slf, "bell") == ["speaker_confusion", "previous_owner"]
    adr = _build(_spec(P3, ["lamp", "cup", "bell"], P3_EVENTS,
                       {"kind": "addressee", "speaker": "Ken", "voc": "Rosa"}), (3, 2, "addressee"))
    assert deixis.error_types(adr, "bell") == ["previous_owner"]    # Ken (the asker) holds nothing
    assert deixis.error_types(adr, "lamp") == ["other_object"]
    third = deixis.build("t", (3, 2, "self"), deixis.to_third(slf.spec))
    for pick in ("lamp", "bell"):
        assert deixis.error_types(third, pick) == deixis.error_types(slf, pick)


# ── paired variants ────────────────────────────────────────────────────────────

def test_twin_flips_exactly_as_simulated_p2():
    ev = [_decl("Ada", "lamp", "mine"), _decl("Ken", "cup"), _swap("Ken", "cup", "lamp")]
    x = _build(_spec(P2, ["lamp", "cup"], ev, {"kind": "self", "speaker": "Ada"}, ["Ada", "Ken"]),
               (2, 1, "self"))
    tw = deixis.build("w", (2, 1, "self"), deixis.to_twin(x.spec), pair_id="x0")
    assert x.gold == "cup" and tw.gold == "lamp"
    assert deixis.final_state(tw.spec) == {"lamp": "Ada", "cup": "Ken"}
    xl, wl = deixis.render(x, False).split("\n"), deixis.render(tw, False).split("\n")
    assert xl[0] == wl[0] and xl[-1] == wl[-1]                      # scene + query unchanged
    assert wl[1:-1] == ["Ken: The lamp is mine.", "Ada: I have the cup.",
                        "Ada: Let's swap: you take my cup and I'll take your lamp."]


@pytest.fixture(scope="module")
def dx() -> list[Item]:
    return items.generate_family("deixis", SPLIT, 6, seed=3)


def test_pairs_link_and_variants_agree(dx):
    assert len(dx) == 6 * len(deixis.CELLS) * 3
    deixis.check_pairs(dx)
    by: dict[str, dict[str, Item]] = {}
    for it in dx:
        by.setdefault(it.meta["pair_id"], {})[it.meta["variant"]] = it
    for pid, g in by.items():
        x, t, w = g["deictic"], g["third"], g["twin"]
        assert {x.item_id, t.item_id, w.item_id} == {f"{pid}_{v}" for v in deixis.VARIANTS}
        a, b = x.spec["sigma"]
        sig = {a: b, b: a}
        fx = deixis.final_state(x.spec)
        assert deixis.final_state(t.spec) == fx and t.gold == x.gold
        assert deixis.final_state(w.spec) == {o: sig.get(h, h) for o, h in fx.items()}
        assert w.gold != x.gold and w.spec["query"] == x.spec["query"]
        if x.cell[2] == "owner":
            assert w.gold == sig[x.gold]
        assert not deixis._PRONOUNS.search("\n".join(deixis.render(t, False).split("\n")[1:]))
        for v in (x, t, w):    # every variant under every shot condition
            for c in items.CONDITIONS:
                assert items.build_prompt(v, c).endswith(deixis.render(v, False))
    # corruptions are caught
    with pytest.raises(ValueError):
        deixis.check_pairs([it for it in dx if it.meta["variant"] != "twin"])
    bad = [Item(**{**it.to_dict(), "gold": it.gold}) for it in dx]
    tw = next(it for it in bad if it.meta["variant"] == "twin")
    tw.spec = dict(tw.spec, events=next(it for it in dx if it.item_id == tw.meta["pair_id"]
                                       + "_deictic").spec["events"], variant="twin")
    with pytest.raises(ValueError):
        deixis.check_pairs(bad)


def test_balancing_and_audit(dx):
    aud = deixis.audit(dx)
    assert deixis.audit_failures(aud) == []
    for c in deixis.CELLS:
        a = aud["/".join(map(str, c))]["deictic"]
        assert a["gold_is_last_mention"] == 0.5
        if c[2] == "owner":
            assert a["gold_is_last_speaker"] == 0.5
        if not deixis.exempt_ownership_word(c):
            assert a["gold_in_last_line"] == 0.0
        # third mirrors the deictic recency (form choice), cell by cell
        assert aud["/".join(map(str, c))]["third"]["gold_is_last_mention"] == 0.5
    broken = {"2/2/owner": {"deictic": {**aud["2/2/owner"]["deictic"], "gold_is_last_mention": 0.7}}}
    assert deixis.audit_failures(broken) and "recency" in deixis.audit_failures(broken)[0]


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
        coll = items.collide_shots(it)
        for c in coll:
            deixis.check_surface(c)        # P=3 vocatives still name another participant
            assert set(c.spec["people"]) == set(it.spec["people"])
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


# ── readouts + gates on fake grids ─────────────────────────────────────────────

def _drow(model: str, cond: str, variant: str, kind: str, pair: str, correct: bool,
          n_cand: int = 2, item_id: Optional[str] = None, gold: str = "x") -> dict[str, Any]:
    return {"model": model, "item_id": item_id or f"{pair}_{variant}", "family": "deixis",
            "cell": [2, 2, kind], "cell_key": f"2/2/{kind}", "condition": cond,
            "nshots": 0 if cond == "0shot" else 3, "variant": variant, "pair_id": pair,
            "qkind": kind, "gold_is_last_mention": False, "gold_is_last_speaker": None,
            "gold_in_last_line": False, "gold": gold, "n_candidates": n_cand, "gold_lp": -1.0,
            "margin": 1.0 if correct else -1.0, "gold_rank_cand": 1 if correct else 2,
            "gold_rank_full": 1, "entropy": 1.0, "prompt_tokens": 9, "cand_lps": {}, "top": [],
            "multi_token_candidates": [], "pick": "x",
            "error_type": None if correct else "speaker_confusion", "error_types": []}


def _fake(model: str, x: list[int], tw: list[int], th: list[int]) -> list[dict[str, Any]]:
    """Per query kind and condition, 4 pairs with the given correctness patterns."""
    rows = []
    for cond in items.CONDITIONS:
        for kind in deixis.QKINDS:
            for k in range(4):
                pid = f"{model}-{kind}-{k}"
                rows += [_drow(model, cond, "deictic", kind, pid, bool(x[k])),
                         _drow(model, cond, "twin", kind, pid, bool(tw[k])),
                         _drow(model, cond, "third", kind, pid, bool(th[k]))]
    return rows


def test_readouts_known_numbers():
    rows = _fake("big", [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0])
    d = agg.deixis_readouts(rows)["big"]
    assert d["cc"]["deictic"]["3shot"]["all"] == pytest.approx(0.0)       # acc .5, C=2
    assert d["cc"]["third"]["3shot"]["owner"] == pytest.approx(0.5)       # acc .75
    assert d["penalty"]["3shot"]["self"] == pytest.approx(0.5)
    t = d["twin"]["3shot"]["all"]
    assert (t["p_x"], t["p_twin"], t["joint"], t["chance2"]) == (0.5, 0.5, 0.25, 0.25)
    assert t["dependence"] == pytest.approx(0.0) and t["n_pairs"] == 12
    assert d["asymmetry"]["deictic"]["3shot"] == pytest.approx(0.0)
    assert d["collision_gap"]["third"]["acc"] == pytest.approx(0.0)
    assert d["errors"]["deictic"]["3shot"]["speaker_confusion"] == 1.0
    # perfectly coupled pairs: dependence = joint − product > 0
    d2 = agg.deixis_readouts(_fake("c", [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]))["c"]
    assert d2["twin"]["3shot"]["all"]["dependence"] == pytest.approx(0.25)
    # generic tables read the DEICTIC rows only; the deixis section is appended
    md, s = agg.aggregate(rows)
    assert s["summary"]["big|deixis"]["n"]["3shot"] == 12
    assert s["summary"]["big|deixis"]["acc_3shot"] == 0.5
    assert "DEIXIS PENALTY" in md and "TWIN CONSISTENCY" in md and s["deixis"]["big"]
    with pytest.raises(ValueError):
        agg.deixis_readouts([{**rows[0], "variant": None}])
    m = lite.summarize(rows)
    assert m["bindprobes/deixis/3shot/acc"] == 0.5 and m["bindprobes/deixis.third/3shot/acc"] == 0.75
    assert m["bindprobes/deixis/penalty"] == pytest.approx(0.5)
    assert m["bindprobes/deixis/twin_joint"] == 0.25


def test_gates_v2_to_v5_on_fake_grid():
    rows = (_fake("small", [1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 1, 0])       # chance; anti-coupled
            + _fake("big", [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1])
            + _fake("leaky", [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0]))     # third << deictic
    res = {r.gate: r for r in gates.gates(rows, [], "small", "big", ["big"])}
    assert not res["V1"].passed                     # no items given
    assert res["V2"].passed and res["V2"].values["deictic"]["owner"] == pytest.approx(0.0)
    assert res["V3"].passed and res["V3"].values["third_cc"] == pytest.approx(1.0)
    assert res["V4"].passed
    assert res["V5"].passed and res["V5"].values["joint"] == 0.0
    res = {r.gate: r for r in gates.gates(rows, [], "big", "leaky")}   # ge2b = all but big
    assert not res["V2"].passed                     # big is far above chance
    assert not res["V3"].passed                     # leaky third cc = -0.5
    assert not res["V4"].passed and "leaky" in res["V4"].detail
    assert not res["V5"].passed                     # big's pairs are jointly right
    res = {r.gate: r for r in gates.gates(rows, [], "nobody", "nobody", [])}
    assert not any(res[g].passed for g in ("V2", "V3", "V4", "V5"))


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
    # CLI: emitted items + one grid per model -> PASS/FAIL table, json, exit code
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
    assert cli_main(["aggregate", "--grids", str(gdir)]) == 0


def test_cli_gen_includes_deixis_by_default(tmp_path, capsys):
    assert cli_main(["gen", "--out", str(tmp_path), "--per-cell", "2"]) == 0
    meta = json.loads((tmp_path / "emit_meta.json").read_text())
    assert set(meta["n_items"]) == set(items.FAMILY_NAMES) and "deixis" in meta["families"]
    assert meta["n_items"]["deixis"] == len(deixis.CELLS) * 2 * 3
    assert meta["audit_failures"]["deixis"] == []
    assert cli_main(["gen", "--out", str(tmp_path), "--families", "deixis",
                     "--per-cell", "deixis=2"]) == 0
    loaded = items.load_items(tmp_path / "items.jsonl")
    assert {it.family for it in loaded} == {"deixis"} and len(loaded) == 2 * 18 * 3
    sel = lite.select_items(loaded)
    assert len(sel) == len(loaded)                  # 10/cell default > 2/cell available
    assert lite.default_per_family("deixis") == 180 and lite.default_per_family("attr") == 200
