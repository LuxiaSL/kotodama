"""Battery v3 families: setbind, answer, adjacency, codevarmut.

Generation + the shared validator (simulator gold, candidates, mention order, exemplar
disjointness, collide re-labels, prompt bound), the per-family audits at the frozen
per-cell sizes, error typing of every wrong candidate, and family-specific semantics."""

from __future__ import annotations

import contextlib
import io
import math
import random
import re

import pytest

from kotodama.eval.bindprobes import adjacency, answer, codevarmut, items, pools, setbind
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
        assert not set(it.candidates) & pools.NON_COUNTABLE
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


@pytest.fixture(scope="module")
def frozen(split):
    """setbind and adjacency at their frozen per-cell sizes, on the battery's seed."""
    return {f: items.generate_family(f, split, per_cell=V3[f].DEFAULT_PER_CELL, seed=2031)
            for f in ("setbind", "adjacency")}


def _mentions(text: str, words: list[str]) -> list[str]:
    pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(words, key=len, reverse=True))) + r")\b")
    return pat.findall(text)


def test_setbind_recency_and_primacy_readers_near_chance(frozen):
    """From the text alone: answering the last k (or the first k) objects mentioned is exactly the set no more
    often than 1.5 × the exact@k chance (or one item), and the last object mentioned is in the set no more often
    than 1.5 × the hit@1 chance."""
    by: dict[tuple, list[Item]] = {}
    for it in frozen["setbind"]:
        by.setdefault(tuple(it.cell), []).append(it)
    assert set(by) == set(setbind.CELLS)
    for cell, its in by.items():
        k, n = cell[0], len(its)
        last_k = first_k = hit_last = 0
        for it in its:
            body = setbind.render(it, False).rsplit(" So ", 1)[0]
            seq = _mentions(body, list(it.candidates))
            gold = set(it.meta["gold_set"])
            last_k += set(list(dict.fromkeys(reversed(seq)))[:k]) == gold
            first_k += set(list(dict.fromkeys(seq))[:k]) == gold
            hit_last += seq[-1] in gold
        exact = 1 / math.comb(setbind.C_TOTAL, k)
        assert last_k / n <= max(1.5 * exact, 1 / n) + 1e-9, (cell, last_k)
        assert first_k / n <= max(1.5 * exact, 1 / n) + 1e-9, (cell, first_k)
        assert hit_last / n <= 1.5 * k / setbind.C_TOTAL + 1e-9, (cell, hit_last)


def test_adjacency_has_no_recency_solved_cell_and_balanced_critical_roles(frozen):
    """No E = 1 cell; on the critical (scored) items of every variant and cell, the last-mentioned name, the
    penultimate speaker, the last transfer's recipient and the most-mentioned name each pick the gold at most
    1.5 × chance (1/3), read from the text."""
    assert all(c[0] >= 2 for c in adjacency.CELLS)
    by: dict[tuple, list[Item]] = {}
    for base in frozen["adjacency"]:
        for it in adjacency.expand(base):
            if it.meta["critical"]:
                by.setdefault((it.meta["variant"], tuple(it.cell)), []).append(it)
    assert {v for v, _ in by} == set(adjacency.VARIANTS) and {c for _, c in by} == set(adjacency.CELLS)
    for key, its in by.items():
        n = len(its)
        hits = {"last": 0.0, "penult_speaker": 0.0, "last_recipient": 0.0, "most_mentioned": 0.0}
        for it in its:
            text = adjacency.render(it, False)
            body = text.rsplit("Now the ", 1)[0]
            names = list(it.candidates)
            seq = _mentions(body.split("\n", 1)[-1] if "\n" in body else body.split(". ", 1)[1], names)
            assert not it.meta["queried_obj_in_last_transfer"]
            hits["last"] += seq[-1] == it.gold
            if it.meta["variant"] != "third":
                speakers = [ln.split(":")[0] for ln in body.strip().split("\n")[1:]]
                hits["penult_speaker"] += speakers[-2] == it.gold
            hits["last_recipient"] += adjacency.transfers(it.spec)[-1]["recipient"] == it.gold
            hits["most_mentioned"] += adjacency.most_frequent_credit(seq, it.gold)
        for h, x in hits.items():
            assert x / n <= 0.5 + 1e-9, (key, h, x / n)
        roles = [it.meta["gold_role"] for it in its]
        assert all(roles.count(r) == n // 3 for r in ("recipient", "giver", "bystander")), key


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
