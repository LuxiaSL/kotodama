"""Battery v3 families: setbind, answer, adjacency, codevarmut.

Generation + the shared validator (simulator gold, candidates, mention order, exemplar
disjointness, collide re-labels, prompt bound), the per-family audits at the frozen
per-cell sizes, error typing of every wrong candidate, and family-specific semantics."""

from __future__ import annotations

import contextlib
import io
import math
import random

import pytest

from kotodama.eval.bindprobes import adjacency, answer, codevarmut, items, setbind
from kotodama.eval.bindprobes.base import Item
from kotodama.eval.bindprobes.pools import load_pools

V3 = {"setbind": setbind, "answer": answer, "adjacency": adjacency, "codevarmut": codevarmut}


@pytest.fixture(scope="module")
def split():
    return load_pools(None)


@pytest.fixture(scope="module")
def generated(split):
    return {f: items.generate_family(f, split, per_cell=12, seed=11) for f in V3}


def test_registered_but_not_in_core_default():
    assert set(V3) <= set(items.FAMILY_NAMES)
    assert set(items.V3_FAMILY_NAMES) == set(V3)
    assert items.CORE_FAMILY_NAMES == ("attr", "codevar", "dialogue", "deixis")


@pytest.mark.parametrize("fam", list(V3))
def test_generate_validate_and_type_errors(fam, generated, split):
    mod = V3[fam]
    its = generated[fam]
    assert its
    for it in its:
        items.validate(it, split)
        for c in it.candidates:
            if c != it.gold:
                types = mod.error_types(it, c)
                assert types and types[0] in mod.ERROR_TYPES, (it.item_id, c, types)


@pytest.mark.parametrize("fam", list(V3))
def test_audit_passes_at_frozen_size(fam, split):
    mod = V3[fam]
    its = items.generate_family(fam, split, per_cell=mod.DEFAULT_PER_CELL, seed=3)
    assert mod.audit_failures(mod.audit(its)) == []


@pytest.mark.parametrize("fam", list(V3))
def test_deterministic(fam, split):
    a = items.generate_family(fam, split, per_cell=4, seed=9)
    b = items.generate_family(fam, split, per_cell=4, seed=9)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]


# ── setbind ──────────────────────────────────────────────────────────────────

def test_setbind_set_semantics(generated):
    for it in generated["setbind"]:
        k = it.cell[0]
        assert len(it.meta["gold_set"]) == k and it.gold in it.meta["gold_set"]
        assert len(it.candidates) == setbind.C_TOTAL
        assert not set(it.candidates) & setbind.MASS_OBJECTS
        for c in it.meta["gold_set"]:
            if c != it.gold:
                assert setbind.error_types(it, c) == ["other_held"]


def test_setbind_metrics():
    lps = {" a": -1.0, " b": -1.5, " c": -0.5, " d": -3.0}
    m = setbind.set_metrics(lps, ["a", "b"])
    assert m["hit1"] == 0.0 and m["exactk"] == 0.0 and m["margin"] == pytest.approx(-1.0)
    m = setbind.set_metrics({" a": -0.1, " b": -0.2, " c": -2.0, " d": -3.0}, ["a", "b"])
    assert m["hit1"] == 1.0 and m["exactk"] == 1.0 and m["margin"] == pytest.approx(1.8)
    assert m["chance_exactk"] == pytest.approx(1 / math.comb(4, 2))
    with pytest.raises(ValueError):
        setbind.set_metrics(lps, ["zzz"])


def test_setbind_split_layout_interleaves(generated):
    for it in generated["setbind"]:
        if it.cell[1] == "split" and it.cell[2] == 0:
            sents = it.spec["sentences"]
            q_idx = [i for i, s in enumerate(sents) if s[0] == "bind" and s[1] == it.spec["query"]]
            assert len(q_idx) == it.cell[0]
            for a, b in zip(q_idx, q_idx[1:]):
                assert any(s[0] == "bind" and s[1] != it.spec["query"] for s in sents[a + 1:b])


def test_setbind_rejects_double_bind():
    spec = {"people": ["A", "B", "C", "D"], "sentences": [["bind", "A", ["x"], False],
                                                          ["bind", "B", ["x"], False]],
            "query": "A", "designated": "x"}
    with pytest.raises(ValueError):
        setbind.run(spec)


# ── answer ───────────────────────────────────────────────────────────────────

def test_answer_roles_and_variants(generated):
    by = {}
    for it in generated["answer"]:
        by.setdefault(it.meta["pair_id"], {})[it.meta["variant"]] = it
    for g in by.values():
        q, n = g["qa"], g["narr"]
        role = q.meta["role"]
        assert n.gold == q.meta["holder"]
        expect = {"me": "me", "you": "you"}.get(role, q.meta["holder"])
        assert q.gold == expect == g["third"].gold
        assert q.candidates[:2] == ["me", "you"] and len(q.candidates) == 4
        t = q.spec["tail"]
        text = answer.render(q, False)
        assert text.endswith(("I think it's with",)) and t["answerer"] + ": " in text.split("\n")[-1]
        if q.cell[1] == "voc":
            assert text.split("\n")[-1].startswith(f"{t['answerer']}: {t['asker']}, ")


def test_answer_perspective_flip_typed(generated):
    it = next(i for i in generated["answer"] if i.meta["variant"] == "qa" and i.gold == "me")
    assert answer.error_types(it, "you") == ["perspective_flip"]


# ── adjacency ────────────────────────────────────────────────────────────────

def test_adjacency_variants_share_gold_and_grants_name_object(generated):
    for it in generated["adjacency"]:
        text = adjacency.render(it, False)
        if it.meta["variant"] == "adj":
            for ln in text.split("\n"):
                if ": Sure, " in ln:
                    assert any(f"the {o}" in ln for o in it.spec["objects"])
                    after = ln.split(": Sure, ")[1]
                    assert not any(after.startswith(p + ", ") for p in it.spec["people"])


def test_adjacency_surface_law_rejects_unanchored_grant(generated):
    it = next(i for i in generated["adjacency"] if i.meta["variant"] == "adj" and i.meta["critical"])
    spec = dict(it.spec)
    evs = list(spec["events"])
    # move a grant away from its request: replace with a vocative-free give (illegal at P=3)
    k = next(j for j, ev in enumerate(evs) if ev["ev"] == "grant")
    g = evs[k]
    evs[k] = {"ev": "give", "speaker": g["granter"], "voc": None, "obj": g["obj"], "form": "giving"}
    spec["events"] = evs
    with pytest.raises(ValueError):
        adjacency.build("x", tuple(it.cell), spec)


def test_adjacency_granter_error(generated):
    it = next(i for i in generated["adjacency"] if i.meta["critical"] and i.meta["variant"] == "adj")
    granter = [t for t in adjacency.transfers(it.spec) if t["obj"] == it.spec["query"]["obj"]][-1]["giver"]
    assert adjacency.error_types(it, granter) == ["granter"]


# ── codevarmut ──────────────────────────────────────────────────────────────

def _exec_print(it: Item) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(codevarmut.render(it, False).replace("  # output:", ""), {})
    return buf.getvalue().strip()


def test_codevarmut_matches_real_python(generated):
    for it in generated["codevarmut"]:
        for x in [it] + it.shot_items() + items.collide_shots(it):
            assert _exec_print(x) == x.gold, x.item_id


def test_codevarmut_traps_bite(generated):
    for it in generated["codevarmut"]:
        trap = it.cell[1]
        mis = it.meta["value_answer"] if trap == "mutate" else it.meta["reference_answer"]
        assert mis != it.gold and mis in it.candidates
        assert codevarmut.render(it, False).startswith("# python\n")


def test_codevarmut_modes():
    spec = {"vars": ["a", "b"], "query": "b", "program": [
        {"op": "lit", "dst": "a", "val": "x"}, {"op": "alias", "dst": "b", "src": "a"},
        {"op": "append", "dst": "a", "val": "y"}]}
    assert codevarmut.answer(spec) == "y"
    assert codevarmut.answer(spec, "value") == "x"
    assert codevarmut.answer(spec, "reference") == "y"
    with pytest.raises(ValueError):
        codevarmut.answer(spec, "bogus")


def test_v3_prompts_token_bound_and_single_token_candidates(generated):
    tok = pytest.importorskip("transformers").AutoTokenizer
    try:
        from kotodama.presets import TOKENIZER_NAME
        t = tok.from_pretrained(TOKENIZER_NAME)
    except Exception as exc:  # offline CI without the tokenizer cache
        pytest.skip(f"tokenizer unavailable: {exc}")
    for fam, its in generated.items():
        assert items.check_tokens(its, t) < 1024


def test_answer_null_calibration_variant(generated):
    by = {}
    for it in generated["answer"]:
        by.setdefault(it.meta["pair_id"], {})[it.meta["variant"]] = it
    for g in by.values():
        z, q = g["null"], g["qa"]
        text = answer.render(z, False)
        assert text.count("\n") == 2                      # scene + question + answer lead only
        assert z.spec["tail"] == q.spec["tail"] and z.candidates == q.candidates
        assert [s["item_id"] for s in z.shots] == [s["item_id"] for s in q.shots]   # same qa exemplars
        assert items.build_prompt(z, "3shot").rsplit(answer.SEPARATOR, 1)[0] == \
            items.build_prompt(q, "3shot").rsplit(answer.SEPARATOR, 1)[0]
