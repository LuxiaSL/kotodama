"""Bindprobes: hand-built simulator cases for every op/pronoun form, generation
invariants (gold, candidates, determinism, exemplar pools, collide re-labels,
prompt bound), error typing, aggregate on a fake grid, and the scoring round
trip through ``riders.score`` (stub logits) — in-process and over a fake
``/logprobs`` server."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from kotodama.eval.bindprobes import aggregate as agg
from kotodama.eval.bindprobes import attr, codevar, dialogue, errors, grid, items, lite, pools
from kotodama.eval.bindprobes.__main__ import main as cli_main
from kotodama.eval.bindprobes.base import Item
from kotodama.eval.riders import grid as rgrid
from kotodama.eval.riders.aggregate import MultiTokenCandidateError
from kotodama.eval.riders.score import score_logprob_batch, score_logprob_items

SPLIT = pools.load_pools(None)


# ── attr ───────────────────────────────────────────────────────────────────────

def _attr(query: str = "Tom", order: list[list[Any]] | None = None) -> Item:
    spec: attr.Spec = {"verb": "keeps",
                       "entities": [["Lucy", "lamp"], ["Tom", "cup"], ["Anna", "bell"]],
                       "distractors": ["apple"],
                       "order": order or [["bind", 0], ["free", 0], ["bind", 1], ["bind", 2]],
                       "query": query}
    return attr.build("a0", (4, "middle"), spec)


def test_attr_simulate_render_and_controls():
    it = _attr()
    assert it.gold == "cup" and it.candidates == ["lamp", "apple", "cup", "bell"]
    assert attr.render(it, False) == ("Lucy keeps the lamp. An apple lies on the shelf. "
                                      "Tom keeps the cup. Anna keeps the bell. So Tom keeps the")
    assert attr.render(it, True).endswith("So Tom keeps the cup.")
    assert it.meta["recency_control"] is False and it.meta["query_index"] == 1
    last = _attr("Anna")
    assert last.gold == "bell" and last.meta["recency_control"] is True
    trailing = _attr("Anna", [["bind", 0], ["bind", 1], ["bind", 2], ["free", 0]])
    assert trailing.meta["recency_control"] is False
    with pytest.raises(ValueError):
        _attr("Nobody")
    with pytest.raises(ValueError):  # an entity declared twice / missing
        _attr(order=[["bind", 0], ["bind", 0], ["bind", 2]])


def test_attr_error_types_and_relabel():
    it = _attr()
    assert attr.error_types(it, "lamp") == ["wrong_binding", "adjacent"]  # Lucy, declared next to Tom
    assert attr.error_types(it, "bell") == ["wrong_binding", "adjacent"]
    assert attr.error_types(it, "apple") == ["distractor"]
    assert attr.error_types(it, "zebra") == ["other"]
    far = _attr("Lucy")
    assert far.gold == "lamp" and attr.error_types(far, "bell") == ["wrong_binding"]
    rl = attr.relabel(it, {"Lucy": "Ken", "Tom": "Roy", "Anna": "Ada"})
    assert rl.gold == "cup" and rl.spec["query"] == "Roy" and "Tom" not in attr.render(rl, False)
    with pytest.raises(ValueError):
        attr.relabel(it, {"Lucy": "Ken"})


# ── codevar ────────────────────────────────────────────────────────────────────

def _lit(dst: str, val: str, kind: str = "decl") -> codevar.Stmt:
    return {"op": "lit", "dst": dst, "val": val, "kind": kind, "group": 0}


def _cp(dst: str, src: str, kind: str = "chain") -> codevar.Stmt:
    return {"op": "copy", "dst": dst, "src": src, "kind": kind, "group": 0}


def _cv(program: list[codevar.Stmt], query: str, cell: tuple[int, int] = (1, 1)) -> Item:
    return codevar.build("c0", cell, {"vars": ["a", "b", "c", "d", "e", "t"],
                                      "program": program, "query": query})


DECL = [_lit("a", "cup"), _lit("b", "pen"), _lit("c", "key")]


def test_codevar_value_semantics_every_op():
    # copy then reassign the source: the copy keeps the OLD value
    it = _cv(DECL + [_cp("d", "a"), _lit("a", "card", "reassign")], "d")
    assert it.gold == "cup" and it.meta["depth"] == 1
    assert codevar.reference_values(it.spec) == {"card"}
    assert codevar.stale_values(it.spec) == set()
    # reassign the source BEFORE the copy: the copy sees the NEW value (old one is stale)
    it = _cv(DECL + [_lit("a", "card", "reassign"), _cp("d", "a")], "d")
    assert it.gold == "card" and codevar.stale_values(it.spec) == {"cup"}
    # swap via temp
    it = _cv(DECL + [_cp("t", "a", "swap"), _cp("a", "b", "swap"), _cp("b", "t", "swap")], "a")
    assert it.gold == "pen" and it.meta["depth"] == 1
    assert _cv(it.spec["program"], "b").gold == "cup"
    assert _cv(it.spec["program"], "b").meta["depth"] == 2   # cup went a -> t -> b
    # alias chain depth 3, with an interfering copy rebinding the chain head later
    prog = DECL + [_cp("d", "c"), _cp("e", "d"), _cp("a", "e"), _cp("c", "b", "copy")]
    it = _cv(prog, "a", (3, 1))
    assert it.gold == "key" and it.meta["depth"] == 3 and codevar.depth_of(it.spec, "a") == 3
    assert codevar.reference_values(it.spec) == {"pen"}   # c rebound after d = c read it
    assert it.candidates == ["cup", "pen", "key"] and it.mention_order == ["cup", "pen", "key"]
    # reading an unbound variable / querying one is an error, as are repeated literals
    with pytest.raises(ValueError):
        _cv(DECL + [_cp("d", "z")], "d")
    with pytest.raises(ValueError):
        _cv(DECL, "e")
    with pytest.raises(ValueError):
        _cv(DECL + [_lit("a", "pen", "reassign")], "a")


def test_codevar_render_and_error_types():
    prog = DECL + [_cp("d", "a"), _lit("a", "card", "reassign"), _lit("d", "map", "reassign"),
                   _lit("d", "ink", "reassign"), _cp("e", "b", "copy")]
    it = _cv(prog, "d", (0, 4))
    assert it.gold == "ink"
    text = codevar.render(it, False)
    assert text.splitlines()[0] == 'a = "cup"' and text.splitlines()[3] == "d = a"
    assert text.endswith("print(d)  # output:") and codevar.render(it, True).endswith("# output: ink")
    assert codevar.error_types(it, "map") == ["stale"]                 # d held map before ink
    assert codevar.error_types(it, "cup")[0] == "stale"                # d held cup (copied) first
    assert codevar.error_types(it, "card") == ["alias"]                # a's final value
    assert codevar.error_types(it, "pen") == ["alias", "init"]         # b's (and e's) final, init
    assert codevar.error_types(it, "key") == ["alias", "init"]
    it2 = _cv(DECL + [_lit("c", "map", "reassign"), _cp("d", "a")], "d")
    assert codevar.error_types(it2, "key") == ["init"]                 # overwritten initial literal
    rl = codevar.relabel(it, {v: v.upper() for v in it.spec["vars"]})
    assert rl.gold == "ink" and "print(D)" in codevar.render(rl, False)


# ── dialogue ───────────────────────────────────────────────────────────────────

PEOPLE = [["Lucy", "f"], ["Tom", "m"], ["Anna", "f"]]
HOLD = [["cup", "Lucy"], ["lamp", "Tom"], ["bell", "Tom"]]


def _dl(ops: list[dict[str, Any]], query: dict[str, str], cell=(3, 2, "holder")) -> Item:
    return dialogue.build("d0", cell, {"people": PEOPLE, "holdings0": HOLD, "ops": ops,
                                       "query": query})


def test_dialogue_pronoun_chains():
    ops = [{"form": "give", "giver": "Lucy", "recipient": "Tom", "objs": ["cup"]},
           {"form": "chain", "subj": "he", "pronoun": "it", "recipient": "Anna"}]
    it = _dl(ops, {"kind": "holder", "obj": "cup"})
    assert it.gold == "Anna"
    assert dialogue.render(it, False) == (
        "Lucy, Tom and Anna are sitting at a table. Lucy has the cup. Tom has the lamp and "
        "the bell. Lucy hands the cup to Tom. He passes it to Anna. Now the cup is with")
    assert it.mention_order == ["Lucy", "Tom", "Anna", "Lucy", "Tom", "Lucy", "Tom", "Anna"]
    # 'them' after a two-object give; then an ask with he/her
    ops2 = [{"form": "give", "giver": "Tom", "recipient": "Lucy", "objs": ["lamp", "bell"]},
            {"form": "chain", "subj": "Lucy", "pronoun": "them", "recipient": "Anna"},
            {"form": "ask", "asker": "Tom", "askee": "Anna", "obj": "bell",
             "subj_pron": "she", "obj_pron": "him"}]
    it2 = _dl(ops2, {"kind": "object", "person": "Anna"}, (3, 4, "object"))
    assert it2.gold == "lamp" and dialogue.final_holdings(it2.spec)["Tom"] == ["bell"]
    assert "Tom asks Anna for the bell, so she gives it to him." in dialogue.render(it2, False)
    assert dialogue.render(it2, False).endswith("Now Anna has the")
    resolved, hist = dialogue.resolve(it2.spec)
    assert [r["giver"] for r in resolved] == ["Tom", "Lucy", "Anna"]
    assert hist["bell"] == ["Tom", "Lucy", "Anna", "Tom"]


@pytest.mark.parametrize("ops", [
    # 'it' with no antecedent
    [{"form": "chain", "subj": "Tom", "pronoun": "it", "recipient": "Anna"}],
    # 'it' after a two-object give (number disagreement)
    [{"form": "give", "giver": "Tom", "recipient": "Lucy", "objs": ["lamp", "bell"]},
     {"form": "chain", "subj": "Lucy", "pronoun": "it", "recipient": "Anna"}],
    # 'she' ambiguous between two women (Lucy -> Anna)
    [{"form": "give", "giver": "Lucy", "recipient": "Anna", "objs": ["cup"]},
     {"form": "chain", "subj": "she", "pronoun": "it", "recipient": "Tom"}],
    # same-gender ask: pronouns cannot be resolved
    [{"form": "ask", "asker": "Lucy", "askee": "Anna", "obj": "cup", "subj_pron": "she",
      "obj_pron": "her"}],
    # giver does not hold the object
    [{"form": "give", "giver": "Anna", "recipient": "Tom", "objs": ["cup"]}],
    # pronoun resolves to the previous GIVER, who no longer holds the object
    [{"form": "give", "giver": "Tom", "recipient": "Lucy", "objs": ["lamp"]},
     {"form": "chain", "subj": "he", "pronoun": "it", "recipient": "Anna"}],
])
def test_dialogue_resolution_rejects_bad_scenes(ops):
    with pytest.raises(ValueError):
        _dl(ops, {"kind": "holder", "obj": "cup"})


def test_dialogue_error_types_and_relabel():
    ops = [{"form": "give", "giver": "Lucy", "recipient": "Tom", "objs": ["cup"]},
           {"form": "chain", "subj": "he", "pronoun": "it", "recipient": "Anna"}]
    it = _dl(ops, {"kind": "holder", "obj": "cup"})
    assert dialogue.error_types(it, "Tom") == ["previous"]
    assert dialogue.error_types(it, "Lucy") == ["previous"]
    obj_q = _dl(ops, {"kind": "object", "person": "Anna"}, (3, 2, "object"))
    assert obj_q.gold == "cup"
    with pytest.raises(ValueError):    # Lucy holds nothing after giving the cup away
        _dl([ops[0]], {"kind": "object", "person": "Lucy"}, (3, 1, "object"))
    tq = _dl(ops + [{"form": "give", "giver": "Tom", "recipient": "Lucy", "objs": ["bell"]}],
             {"kind": "object", "person": "Tom"}, (3, 4, "object"))
    assert tq.gold == "lamp"
    assert dialogue.error_types(tq, "bell") == ["previous", "other_person"]
    assert dialogue.error_types(tq, "cup") == ["previous", "other_person"]  # Tom held it mid-scene
    rl = dialogue.relabel(it, {"Lucy": "Ada", "Tom": "Ken", "Anna": "Rosa"})
    assert rl.gold == "Rosa" and "He passes it to Rosa." in dialogue.render(rl, False)
    with pytest.raises(ValueError):    # a gender-swapping map breaks 'he'
        dialogue.relabel(it, {"Lucy": "Ken", "Tom": "Ada", "Anna": "Rosa"})


# ── generation invariants ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def battery() -> list[Item]:
    return items.generate(SPLIT, seed=11, per_cell={f: 4 for f in items.FAMILY_NAMES})


def test_default_sizes_and_cells():
    sizes = {f: len(items.family(f).CELLS) * items.family(f).DEFAULT_PER_CELL
             for f in items.FAMILY_NAMES}
    assert sizes == {"attr": 1210, "codevar": 1200, "dialogue": 1200}
    assert (2, "middle") not in attr.CELLS and len(attr.CELLS) == 11


def test_generation_invariants(battery):
    assert {it.family for it in battery} == set(items.FAMILY_NAMES)
    for it in battery:
        items.validate(it, SPLIT)
        fam = items.family(it.family)
        assert fam.simulate(it) == it.gold and it.gold in it.candidates
        assert len(it.shots) == items.N_SHOTS == len(it.collide_maps)
        for s in it.shot_items():
            assert s.cell == it.cell and fam.simulate(s) == s.gold
    # balanced cells
    for f in items.FAMILY_NAMES:
        keys = [it.cell_key for it in battery if it.family == f]
        assert len(keys) == 4 * len(items.family(f).CELLS) and len(set(keys)) == len(items.family(f).CELLS)
    # controls balanced where they are generated balanced
    last = [it for it in battery if it.family == "attr" and it.meta["qpos"] == "last"]
    assert sum(it.meta["recency_control"] for it in last) * 2 == len(last)
    dl = [it for it in battery if it.family == "dialogue"]
    assert sum(it.meta["gold_is_last_mention"] for it in dl) * 2 == len(dl)
    cv = [it for it in battery if it.family == "codevar"]
    assert all(it.meta["depth"] == it.cell[0] for it in cv)
    assert all(it.meta["n_stale"] + it.meta["n_reference"] > 0 for it in cv)
    assert all(it.meta["n_pronoun_ops"] >= 1 for it in dl if it.cell[1] >= 2)


def test_determinism_by_seed():
    a = items.generate_family("codevar", SPLIT, 2, seed=5)
    b = items.generate_family("codevar", SPLIT, 2, seed=5)
    c = items.generate_family("codevar", SPLIT, 2, seed=6)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    assert [x.to_dict() for x in a] != [x.to_dict() for x in c]


def test_exemplar_pools_disjoint_and_collide_reuses_labels(battery):
    shot_strings = set(SPLIT.shot.names) | set(SPLIT.shot.objects) | set(SPLIT.shot.variables)
    item_strings = set(SPLIT.item.names) | set(SPLIT.item.objects) | set(SPLIT.item.variables)
    assert not shot_strings & item_strings
    for it in battery:
        fam = items.family(it.family)
        lab, objs = set(fam.labels(it)), items._objects(it)
        for s in items.condition_shots(it, "3shot"):
            assert not set(fam.labels(s)) & lab and not items._objects(s) & objs
        coll = items.condition_shots(it, "3shot_collide")
        for s, c in zip(it.shot_items(), coll):
            assert set(fam.labels(c)) <= lab and not items._objects(c) & objs
            assert items._objects(c) == items._objects(s)            # same objects, new labels
            assert c.gold != it.gold
        assert items.condition_shots(it, "0shot") == []
        p0, p3, pc = (items.build_prompt(it, c) for c in items.CONDITIONS)
        assert p3.endswith(p0) and pc.endswith(p0) and p3 != pc
        assert p3.count(fam.SEPARATOR) == pc.count(fam.SEPARATOR) == items.N_SHOTS
        for block in p3.split(fam.SEPARATOR)[:-1]:
            assert not any(n in block.split() for n in lab)          # no item label in 3shot shots
    # attr collide exemplars use EXACTLY the item's names (same N)
    a = next(it for it in battery if it.family == "attr")
    for c in items.collide_shots(a):
        assert sorted(attr.labels(c)) == sorted(attr.labels(a))


def test_prompt_length_bound(battery):
    for it in battery:
        for c in items.CONDITIONS:
            assert len(items.build_prompt(it, c).encode()) < items.MAX_PROMPT_BYTES < 4096
    big = [it for it in battery if it.cell in ([8, "last"], [3, 4], [4, 4, "holder"])]
    assert big


def test_validate_catches_corruption(battery):
    it = next(x for x in battery if x.family == "attr")
    with pytest.raises(ValueError):
        items.validate(Item(**{**it.to_dict(), "gold": it.candidates[-1]
                               if it.candidates[-1] != it.gold else it.candidates[0]}))
    with pytest.raises(ValueError):
        items.build_prompt(it, "5shot")


def test_selftest(capsys):
    assert items.selftest(per_cell=3) == 0


# ── error typing via the dispatcher ────────────────────────────────────────────

def test_error_dispatch(battery):
    for it in battery:
        fam = items.family(it.family)
        for c in it.candidates:
            if c == it.gold:
                with pytest.raises(ValueError):
                    errors.error_types(it, c)
            else:
                assert errors.primary(it, c) in fam.ERROR_TYPES
    assert errors.types_for("codevar") == ("stale", "alias", "init", "other")


# ── pools ──────────────────────────────────────────────────────────────────────

def test_pools_split_and_names_file(tmp_path):
    assert len(SPLIT.shot.names) == 16 and len(SPLIT.item.female) >= 8
    assert not set(SPLIT.shot.names) & set(SPLIT.item.names)
    assert SPLIT.item.gender[SPLIT.item.female[0]] == "f"
    p = tmp_path / "names.json"
    p.write_text(json.dumps({"verified": {"female": list(pools.BANKED_FEMALE),
                                          "male": list(pools.BANKED_MALE)}}))
    assert pools.load_pools(p).item == SPLIT.item
    p.write_text(json.dumps({"verified": {"female": ["Ada"], "male": ["Ken"]}}))
    with pytest.raises(ValueError):
        pools.load_pools(p)


class _FakeTok:
    """Splits ' Xyz' into one token unless the word is in ``multi``."""

    def __init__(self, multi: set[str]):
        self.multi = multi

    def __call__(self, s: str, add_special_tokens: bool = False):
        class R:
            input_ids = [1, 2] if s.strip() in self.multi else [1]
        return R()


def test_verify_names_with_fake_tokenizer(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pools, "load_tokenizer", lambda name: _FakeTok({"Anna", "lamp"}))
    rep = pools.verify_names(tmp_path / "v.json", ["smollm2"])
    assert "Anna" not in rep["verified"]["female"] and "lamp" not in rep["verified"]["objects"]
    assert rep["rejected"] == {"Anna": {"smollm2": 2}, "lamp": {"smollm2": 2}}
    assert pools.load_pools(tmp_path / "v.json").shot.female[0] == "Emma"
    with pytest.raises(ValueError):
        pools.check_split(SPLIT, _FakeTok({"Tom"}))
    with pytest.raises(ValueError):
        pools.verify_names(tmp_path / "v.json", ["nope"])


# ── scoring round trip (stub logits through riders.score) ──────────────────────

@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer

        from kotodama.presets import TOKENIZER_NAME
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as exc:  # offline without a cache
        pytest.skip(f"tokenizer unavailable: {exc}")


_STUB_ROW = torch.sin(torch.arange(49152, dtype=torch.float32) * 0.37) * 4.0


def _stub_logits(batch: torch.Tensor) -> torch.Tensor:
    """Deterministic (B, T, V) logits: one fixed distribution, broadcast (a view —
    no B x T x V allocation); candidate ranks then follow their token ids."""
    return _STUB_ROW.expand(batch.shape[0], batch.shape[1], _STUB_ROW.shape[0])


def test_pool_strings_single_token_and_real_token_bound(tok, battery):
    pools.check_split(SPLIT, tok)
    assert items.check_tokens(battery, tok) < 1024


def test_scorer_round_trip_in_process(tok, battery):
    sel = lite.select_items(battery, per_family=4)
    jobs = lite.build_jobs(sel)
    assert len(jobs) == len(sel) * len(items.CONDITIONS)
    res = score_logprob_items(_stub_logits, tok, jobs, 3, "cpu", max_seq_len=4096)
    rows = [grid.score_row(sel[j.item_index], r, j.condition, "stub") for j, r in zip(jobs, res)]
    for j, r, row in zip(jobs, res, rows):
        it = sel[j.item_index]
        by = {c["text"]: c["logprob"] for c in r["candidates"]}
        assert row["gold_lp"] == by[" " + it.gold]                       # unchanged values
        assert row["cand_lps"] == {k.strip(): v for k, v in by.items()}
        assert row["multi_token_candidates"] == []
        assert row["gold_rank_cand"] == 1 + sum(v >= row["gold_lp"] for k, v in by.items()
                                                if k != " " + it.gold)
        if row["gold_rank_cand"] == 1:
            assert row["error_type"] is None
        else:
            assert row["error_type"] in items.family(it.family).ERROR_TYPES + ("tie",)
    m = lite.summarize(rows)
    for f in items.FAMILY_NAMES:
        assert m[f"bindprobes/{f}/3shot/n"] == 4.0
        assert f"bindprobes/{f}/collision_gap" in m and f"bindprobes/{f}/shot_gap" in m


class _Resp:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._data


def test_run_grid_over_fake_server_matches_in_process(tok, battery, tmp_path, monkeypatch, capsys):
    """The HTTP runner (riders' client) on a server that runs riders.score == the
    in-process rows, field for field; resume re-scores nothing; aggregate reads it."""
    sel = lite.select_items(battery, per_family=2)
    bodies: list[dict[str, Any]] = []

    def post(url: str, json: dict[str, Any], timeout: float) -> _Resp:
        assert url.endswith("/logprobs")
        bodies.append(json)
        return _Resp({"results": score_logprob_batch(_stub_logits, tok, json["items"],
                                                     json["top_n"], "cpu"), "model": "stub"})

    monkeypatch.setattr(rgrid.requests, "post", post)
    monkeypatch.setattr(rgrid.requests, "get", lambda url, timeout: _Resp({"status": "ok"}))
    out = tmp_path / "grids" / "stub.jsonl"
    n = grid.run_grid("stub", sel, out, endpoint="http://bp.test", batch=5, top_n=3)
    assert n == len(sel) * 3 and all(len(b["items"]) <= 5 for b in bodies)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    jobs = lite.build_jobs(sel)
    ref = score_logprob_items(_stub_logits, tok, jobs, 3, "cpu")
    ref_rows = {(sel[j.item_index].item_id, j.condition):
                grid.score_row(sel[j.item_index], r, j.condition, "stub")
                for j, r in zip(jobs, ref)}
    assert len(rows) == len(ref_rows)
    for r in rows:
        assert r == json.loads(json.dumps(ref_rows[(r["item_id"], r["condition"])]))
    assert grid.run_grid("stub", sel, out, endpoint="http://bp.test") == 0
    md, summary = agg.aggregate(rows)
    assert set(summary["summary"]) == {f"stub|{f}" for f in items.FAMILY_NAMES}
    assert md.startswith("# Bindprobes tables")


# ── aggregate on a fake grid ───────────────────────────────────────────────────

def _row(fam: str, cond: str, rank: int, cell: str, etype: str | None = None, n_cand: int = 4,
         **kw: Any) -> dict[str, Any]:
    return {"model": "m", "item_id": f"{fam}-{cell}", "family": fam, "cell_key": cell,
            "condition": cond, "nshots": 0 if cond == "0shot" else 3, "gold": "x",
            "n_candidates": n_cand, "gold_lp": -1.0, "margin": 1.0 if rank == 1 else -1.0,
            "gold_rank_cand": rank, "gold_rank_full": rank, "entropy": 1.0,
            "prompt_tokens": 9, "cand_lps": {}, "top": [], "multi_token_candidates": [],
            "pick": "x", "error_type": None if rank == 1 else etype, "error_types": [], **kw}


def _fake_grid() -> list[dict[str, Any]]:
    rows = []
    for k in range(4):   # attr: 3shot 1.0, collide 0.5, 0shot 0.25
        rows.append(_row("attr", "3shot", 1, "4/last", qpos="last", recency_control=k < 2))
        rows.append(_row("attr", "3shot_collide", 1 if k < 2 else 2, "4/last", "wrong_binding",
                         qpos="last", recency_control=k < 2))
        rows.append(_row("attr", "0shot", 1 if k == 0 else 2, "2/first",
                         "distractor" if k == 1 else "wrong_binding"))
    for k in range(2):
        rows.append(_row("codevar", "3shot", 1 + k, "1/2", "stale"))
    return rows


def test_aggregate_fake_grid(tmp_path, capsys):
    md, s = agg.aggregate(_fake_grid())
    a = s["summary"]["m|attr"]
    assert a["acc_3shot"] == 1.0 and a["acc_3shot_collide"] == 0.5 and a["acc_0shot"] == 0.25
    assert a["collision_gap"] == 0.5 and a["shot_gap"] == 0.75
    assert a["cc_3shot"] == 1.0 and a["cc_collision_gap"] == pytest.approx(1.0 - (0.5 - .25) / .75)
    assert s["cells"]["m|attr|3shot_collide|4/last"]["err_wrong_binding"] == 1.0
    assert s["errors"]["m|attr|0shot"]["distractor"] == pytest.approx(1 / 3)
    c = s["summary"]["m|codevar"]
    assert c["acc_3shot"] == 0.5 and c["n"]["0shot"] == 0
    assert "| m | attr | 1.000 |" in md and "| 4/last | — | 1.00 cc+1.00 m+1.00 | 0.50" in md
    assert "### attr — m: recency control (qpos=last)" in md
    assert "| 3shot_collide | 1.00 | 0.00 | +1.00 |" in md
    (tmp_path / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _fake_grid()))
    md_path, js_path = agg.write_tables(tmp_path)
    assert md_path.name == agg.TABLES_MD and json.loads(js_path.read_text())["summary"]
    # hard-fails: multi-token candidates; rows that are not bindprobe rows
    bad = dict(_fake_grid()[0], multi_token_candidates=[" teabag"])
    (tmp_path / "bad.jsonl").write_text(json.dumps(bad) + "\n")
    with pytest.raises(MultiTokenCandidateError):
        agg.write_tables(tmp_path)
    assert cli_main(["aggregate", "--grids", str(tmp_path)]) == 2
    rider_row = {k: v for k, v in _fake_grid()[0].items() if k not in ("family", "condition")}
    with pytest.raises(ValueError):
        agg.aggregate([rider_row])


# ── lite + CLI ─────────────────────────────────────────────────────────────────

def test_lite_select_balanced(battery):
    sel = lite.select_items(battery, per_family=13, seed=1)
    for f in items.FAMILY_NAMES:
        sub = [it for it in sel if it.family == f]
        assert len(sub) == 13
        counts = {k: [it.cell_key for it in sub].count(k) for k in {it.cell_key for it in sub}}
        assert max(counts.values()) - min(counts.values()) <= 1
    assert [it.item_id for it in lite.select_items(battery, 13, seed=1)] == [it.item_id for it in sel]
    assert len(lite.select_items(battery, 10_000)) == len(battery)
    with pytest.raises(ValueError):
        lite.select_items(battery, 0)


def test_cli_gen_and_selftest(tmp_path, capsys):
    assert cli_main(["gen", "--out", str(tmp_path), "--per-cell", "1",
                     "--families", "attr,dialogue"]) == 0
    loaded = items.load_items(tmp_path / "items.jsonl")
    assert len(loaded) == len(attr.CELLS) + len(dialogue.CELLS)
    meta = json.loads((tmp_path / "emit_meta.json").read_text())
    assert meta["n_items"] == {"attr": 11, "dialogue": 12} and meta["seed"] == items.DEFAULT_SEED
    assert Item.from_dict(json.loads(json.dumps(loaded[0].to_dict()))) == loaded[0]
    assert cli_main(["gen", "--out", str(tmp_path), "--families", "nope"]) == 2
    assert cli_main(["gen", "--out", str(tmp_path), "--per-cell", "codevar=1,attr=2",
                     "--families", "codevar,attr"]) == 0
    assert json.loads((tmp_path / "emit_meta.json").read_text())["n_items"] == \
        {"codevar": 12, "attr": 22}
    assert cli_main(["run", "--model", "m", "--items", str(tmp_path / "items.jsonl"),
                     "--out", str(tmp_path / "g.jsonl"), "--conditions", "4shot"]) == 2
