"""The sampling-law bench (kotodama.posttrain.lawbench): the stage-A plan, cell rows, the law transform the damage
readout uses (against the server sampler's arithmetic), the meters, generation through a mocked endpoint (request
fields, shared seeds, the one-path dialogue, resume), fluency and damage sets, the summary, and the damage script's
teacher-forced scoring on a tiny model."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import pytest
import torch

from kotodama.posttrain import lawbench as lb
from kotodama.posttrain import selfplay as sp
from kotodama.serve.laws import law


def test_stage_a_plan() -> None:
    cells = lb.stage_a()
    ids = [c.id for c in cells]
    assert len(ids) == len(set(ids)) and ids[0] == "base"
    base = law("tsumugi")
    assert cells[0].temperature == base["temperature"] and cells[0].repetition_penalty == base["repetition_penalty"]
    t = [c for c in cells if c.ladder == "T"]
    assert sorted(c.temperature for c in t) == [0.6, 0.7, 0.8, 0.9, 1.0]
    assert all(c.repetition_penalty == base["repetition_penalty"] and not c.steer for c in t)
    assert sorted(c.repetition_penalty for c in cells if c.ladder == "rp") == [1.0, 1.1, 1.35]
    v7 = [c for c in cells if c.ladder == "V7"]
    assert len(v7) == 6 and {c.dose for c in v7} >= {0.05, -0.05} and all(c.steer[0][0] == "V7_s58" for c in v7)
    vrep = [c for c in cells if c.ladder == "Vrep"]
    assert {c.steer_rule for c in vrep} == {"K", "N", "S"} and all(c.dose < 0 for c in vrep)
    for lad in ("V7", "Vrep"):
        ctrls = [c for c in cells if c.ladder == f"rand:{lad}"]
        assert all(c.control and c.steer[0][0].startswith("Rband") for c in ctrls)
        # every vector cell has its controls at the same dose and rule
        for c in (x for x in cells if x.ladder == lad):
            twins = [r for r in ctrls if r.dose == c.dose and r.steer_rule == c.steer_rule]
            n = 3 if lad == "V7" else 2  # the V7 ladder carries a third control per dose
            assert len(twins) == n and len({r.steer[0][0] for r in twins}) == n
    small = lb.stage_a({"temperature": [0.6], "repetition_penalty": [], "vector_ladders": [], "random_per_dose": 0})
    assert [c.id for c in small] == ["base", "T0.6"]


def test_cell_row_roundtrip_and_validation() -> None:
    c = lb.Cell("x", "V7", 0.75, 1.2, min_p=0.05, steer=(("V7_s58", 0.1),), steer_rule="N")
    assert lb.Cell.from_row(json.loads(json.dumps(c.to_row()))) == c
    assert c.sampling()["min_p"] == 0.05 and c.vectors() == [{"name": "V7_s58", "alpha": 0.1}] and c.dose == 0.1
    for bad in (dict(temperature=0.0), dict(repetition_penalty=0.9), dict(steer_rule="Q"), dict(id="a b"),
                dict(steer=(("v", float("nan")),))):
        kw = {"id": "x", "ladder": "T", "temperature": 0.7, "repetition_penalty": 1.2, **bad}
        with pytest.raises(ValueError):
            lb.Cell(**kw)
    with pytest.raises(ValueError):
        lb.Cell.from_row({**c.to_row(), "surprise": 1})


def _reference_logprob(logits: torch.Tensor, target: int, T: float, rp: float, ids: list[int], top_p: float,
                       min_p: float) -> float:
    """The server sampler's arithmetic (kotodama.serve.server.sample_next_token), as a distribution."""
    x = logits.float().clone()
    if rp != 1.0 and ids:
        pid = torch.tensor(ids).unique()
        v = x[pid]
        x[pid] = torch.where(v > 0, v / rp, v * rp)
    x = x / T
    if 0 < top_p < 1:
        s, idx = x.sort(descending=True)
        p = s.softmax(-1)
        s[(p.cumsum(-1) - p) >= top_p] = float("-inf")
        x = s.scatter(0, idx, s)
    if min_p > 0:
        x = x.masked_fill(x < x.max() + math.log(min_p), float("-inf"))
    return float(x.log_softmax(-1)[target])


def test_law_logprobs_match_the_server_sampler() -> None:
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(5, 40, generator=g) * 3
    tgt = torch.tensor([1, 7, 0, 39, 12])
    laws = [(1.0, 1.0, [], 0.0, 0.0), (0.6, 1.2, [7, 3, 3], 0.0, 0.0), (1.3, 1.35, [0, 5], 0.0, 0.0),
            (0.8, 1.0, [], 0.9, 0.0), (0.9, 1.1, [12], 0.0, 0.1)]
    got = lb.law_logprobs(logits, tgt, [x[0] for x in laws], [x[1] for x in laws], [x[2] for x in laws],
                          [x[3] for x in laws], [x[4] for x in laws])
    for r, (T, rp, ids, tp, mp) in enumerate(laws):
        want = _reference_logprob(logits[r], int(tgt[r]), T, rp, ids, tp, mp)
        assert (math.isinf(want) and math.isinf(float(got[r]))) or float(got[r]) == pytest.approx(want, abs=1e-5)
    assert float(got[0]) == pytest.approx(float(logits[0].log_softmax(-1)[1]), abs=1e-6)  # identity law = plain
    # truncation removes a tail target
    worst = int(logits[3].argmin())
    out = lb.law_logprobs(logits[3:4], torch.tensor([worst]), [1.0], [1.0], [[]], [0.5], [0.0])
    assert math.isinf(float(out[0]))


def test_text_meters() -> None:
    loop = "we went down to the sea and " + "the boat is gone the boat is gone " * 20
    fresh = " ".join(f"word{i}" for i in range(120))
    assert lb.repeat_share(fresh) == 0.0 and lb.repeat_share(loop) > 0.7
    assert lb.text_loop_onset(fresh) is None and 0.0 < lb.text_loop_onset(loop) < 0.2
    assert lb.distinctness("a b c d", "a b c d") == 0.0 and lb.distinctness("a b c d", "e f g h") == 1.0
    assert lb.distinctness("a", "b c d") is None
    chk = {"min_chars": 20, "max_chars": 200, "min_sentences": 2, "forbid": r"\bCharacters\b"}
    assert lb.form_check("One sentence here. And another one.", chk)
    assert not lb.form_check("Only one sentence that is long enough.", chk)
    assert not lb.form_check("Characters: two. Then more.", chk)
    assert lb.form_check("x", None) is None


def test_dialogue_meters_and_uptake() -> None:
    turns = [{"speaker": "Ana", "text": "The lighthouse keeper left a brass compass for you."},
             {"speaker": "Ben", "text": "A compass? Where did he hide it?"},
             {"speaker": "Ana", "text": "Under the stairs, wrapped in oilcloth."},
             {"speaker": "Ben", "text": "Fine."},
             {"speaker": "Ana", "text": "Fine."},
             {"speaker": "Ben", "text": "Fine."}]
    assert lb.uptake(turns, 1) is True  # "compass" was introduced by Ana
    assert lb.uptake(turns, 3) is False
    assert lb.uptake(turns, 0) is None
    row = {"kind": "dialogue", "cast": ["Ana", "Ben"], "header": "", "finish": "eos",
           "turns": [{**t, "ended_by": "stop" if i < 5 else "eos", "vetoes": ["repeat_turn"] if i > 3 else [],
                      "siblings": [{"text": t["text"] + " Yes.", "ended_by": "eos"}]} for i, t in enumerate(turns)]}
    m = lb.dialogue_meters(row)
    assert m["turns"] == 6 and m["repeat_share"] == pytest.approx(2 / 6, abs=1e-3)
    assert m["eos_share"] == pytest.approx(7 / 12, abs=1e-3) and m["veto_share"] == pytest.approx(2 / 6, abs=1e-3)
    assert 0 < m["sibling_distinct"] < 1 and m["uptake"] is not None
    assert m["in_frame"] == 1.0
    review = {**row, "turns": row["turns"][:1] + [{**row["turns"][1], "text": (
        "I was afraid of it. The most interesting character is a young woman in the same room. In a scene that "
        "punctuates the novel's beginning, she laments a man in a white labcoat.")}]}
    assert lb.dialogue_meters(review)["in_frame"] == 0.5


def _mock_server(seen: list[dict]):
    httpx = pytest.importorskip("httpx")

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        k = len(seen)
        texts = [f" turn {k} draw {i} says something new {k * 7 + i}." for i in range(body["n"])]
        if body["prompt"].endswith(("Ana:", "Ben:")):  # a dialogue turn: it ends where the next speaker begins
            texts = [t + ("\nAna: interrupting" if k % 3 == 0 else "") + "\n\nAna:" for t in texts]
        return httpx.Response(200, json={"choices": [
            {"text": t, "finish_reason": "stop", "token_ids": {"completion": [1, 2]}} for t in texts]})

    return httpx.MockTransport(handler)


def test_generate_rows_requests_and_resume(tmp_path: Path) -> None:
    seen: list[dict] = []
    cells = [lb.Cell("base", "base", 0.75, 1.2),
             lb.Cell("v", "Vrep", 0.75, 1.2, min_p=0.05, penalty_scope="row", steer=(("Vrepperp_s58", -0.3),),
                     steer_rule="N")]
    comp = tmp_path / "c.jsonl"
    comp.write_text(json.dumps({"id": "p0", "prompt": "Once", "stop": ["\n"], "max_tokens": 50}) + "\n"
                    + json.dumps({"id": "p1", "prompt": "Twice"}) + "\n")
    dlg = tmp_path / "d.jsonl"
    dlg.write_text(json.dumps({"id": "s0", "cast": ["Ana", "Ben"], "header": "Ana and Ben at sea."}) + "\n")
    panels = [lb.load_panel("seed", comp), lb.load_panel("dialogue", dlg)]
    assert [p.kind for p in panels] == ["completion", "dialogue"]
    out = tmp_path / "run"
    cfg = lb.GenConfig(max_turns=4)
    stats = asyncio.run(lb.generate(cells, panels, "http://gw:1", out, cfg, 8, transport=_mock_server(seen),
                                    stamp={"engine": "abc", "regional": "true"}))
    assert stats["done"] == 6 and stats["failed"] == 0
    rows = lb.read_jsonl(out / "gen" / "seed.jsonl")
    assert all(r["prov"] == {"engine": "abc", "regional": "true"} for r in rows)
    assert {(r["cell"], r["item"]) for r in rows} == {(c, i) for c in ("base", "v") for i in ("p0", "p1")}
    assert all(len(r["draws"]) == 2 and r["draws"][0]["ended_by"] == "eos" for r in rows)
    first = [b for b in seen if b["prompt"] == "Once"]
    assert len(first) == 2 and first[0]["seed"] == first[1]["seed"] == lb.item_seed("p0")  # shared noise
    assert all(b["n"] == 2 and b["max_tokens"] == 50 and b["stop"] == ["\n"] for b in first)
    st = next(b for b in first if "vectors" in b)
    assert st["vectors"] == [{"name": "Vrepperp_s58", "alpha": -0.3}] and st["steer_rule"] == "N"
    assert st["min_p"] == 0.05 and st["penalty_scope"] == "row" and st["user"].startswith("lawbench|v|seed|p0")
    plain = next(b for b in first if "vectors" not in b)
    assert plain["frequency_penalty"] == pytest.approx(0.2) and plain["min_p"] == 0.0
    d = lb.read_jsonl(out / "gen" / "dialogue.jsonl")
    assert len(d) == 2 and all(len(r["turns"]) == 4 and r["finish"] == "max_turns" for r in d)
    t = d[0]["turns"]
    assert [x["speaker"] for x in t] == ["Ana", "Ben", "Ana", "Ben"]
    assert all(x["picked_draw"] == 0 and len(x["siblings"]) == 1 for x in t)
    assert all("Ana:" not in x["text"] for x in t)  # the stray label is cut
    # turn seeds differ by turn and are shared by the cells
    by_cell = {}
    for b in seen:
        if "Ana and Ben" in b["prompt"]:
            by_cell.setdefault("vectors" in b, []).append(b["seed"])
    assert by_cell[True] == by_cell[False] and len(set(by_cell[True])) == 4
    # resume: nothing left to do
    n = len(seen)
    again = asyncio.run(lb.generate(cells, panels, "http://gw:1", out, cfg, 8, transport=_mock_server(seen)))
    assert again["todo"] == 0 and len(seen) == n
    with pytest.raises(SystemExit):
        asyncio.run(lb.generate([lb.Cell("base", "base", 0.9, 1.2)], panels, "http://gw:1", out, cfg, 8,
                                transport=_mock_server(seen)))

    # fluency items: one per completion row, one per dialogue turn; then the per-row mean
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [len(w) for w in s.split()]

    items = {it["id"]: it for p in panels for it in p.items}
    fi = lb.fluency_items(rows + d, items, Tok())
    assert len(fi) == 4 + 8 and {x["id"].count("|") for x in fi} == {1, 2}
    assert lb.fluency_by_row({"a|x|0": [2.0, 1], "a|x|1": [4.0, 3], "b|y": [1.0, 2]}) == {"a|x": 1.5, "b|y": 0.5}
    # the damage set: base-cell draws and turns, prompts rendered up to the turn
    ds = lb.damage_set(out, {p.name: p for p in panels}, turns_per_dialogue=(1, 3))
    assert [x["id"] for x in ds] == ["seed|p0", "seed|p1", "dialogue|s0|1", "dialogue|s0|3"]
    assert ds[2]["prompt"].endswith("Ben:") and ds[2]["continuation"].startswith(" ")
    # summary + table
    (out / "fluency").mkdir()
    (out / "fluency" / "seed.json").write_text(json.dumps({"rows": {"base|p0": 2.0, "base|p1": 3.0}}))
    summ = lb.summarize(out, {p.name: p for p in panels}, {"cells": {"v": {"damage": 0.2, "panels": {"seed": 0.3}}}},
                        {"seed": {"strength": {"base": 0.1}}})
    assert summ["seed"]["base"]["nll"] == 2.5 and summ["seed"]["v"]["damage"] == 0.3
    assert summ["seed"]["base"]["bt"] == 0.1 and summ["dialogue"]["v"]["turns"] == 4.0
    table = lb.frontier_table(summ["seed"], ["damage", "nll"])
    assert "| v | Vrep | 0.300 |" in table


def test_completer_call_fields_and_seed_offset() -> None:
    httpx = pytest.importorskip("httpx")
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"choices": [{"text": " a", "finish_reason": "length",
                                                      "token_ids": {"completion": [1] * 5}}] * body["n"]})

    complete, close = sp.oai_completer(["http://a:1"], {**law("tsumugi"), "min_p": 0.1}, 8,
                                       transport=httpx.MockTransport(handler))

    async def go():
        try:
            return await complete("x", 10, [], seed=100, user="u", max_tokens=5)
        finally:
            await close()

    out = asyncio.run(go())
    assert [b["seed"] for b in seen] == [100, 108] and all(b["user"] == "u" and b["min_p"] == 0.1 for b in seen)
    assert all(b["max_tokens"] == 5 for b in seen) and out[0][2] is True  # 5 ids == this call's cap


_SPEC = importlib.util.spec_from_file_location(
    "law_damage", Path(__file__).resolve().parents[1] / "scripts" / "eval" / "law_damage.py")
law_damage = importlib.util.module_from_spec(_SPEC)
sys.modules["law_damage"] = law_damage
_SPEC.loader.exec_module(law_damage)

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)


@torch.inference_mode()
def test_damage_scoring_on_a_tiny_model() -> None:
    from torch.nn.attention import SDPBackend

    from kotodama.model.decode_engine import DecodeEngine
    from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
    from kotodama.model.multi_decode import MultiSampleDecoder

    torch.manual_seed(20261028)
    model = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    H = CFG["hidden_size"]
    unit = torch.nn.functional.normalize(torch.randn(H), dim=0)
    bank = {"Vx_s8": (4, unit)}
    meds = {4: 30.0}
    cells = [lb.Cell("plain", "x", 1.0, 1.0), lb.Cell("T", "T", 0.6, 1.0), lb.Cell("rp", "rp", 1.0, 1.3),
             lb.Cell("v", "V", 1.0, 1.0, steer=(("Vx_s8", 0.5),), steer_rule="S")]
    eng = DecodeEngine(model, max_seq_len=96, dtype=torch.float32, prefill_backends=[SDPBackend.MATH], levers=2)
    dec = MultiSampleDecoder(eng, 1 + len(cells))
    slots = []
    for c in cells:
        plan = law_damage.cell_plan(c, bank, meds, {}, eng)
        slots.append(0 if plan is None else eng.levers.bind(plan))
    assert slots[:3] == [0, 0, 0] and slots[3] != 0
    prompt = [5, 9, 13, 40, 2, 7]
    cont = [11, 11, 30, 11, 8]
    law_lp, plain_lp = law_damage.score_set(eng, dec, cells, slots, prompt, cont)
    assert law_lp.shape == (4, 5) and plain_lp.shape == (5,)
    full = model(torch.tensor([prompt + cont]))["logits"][0].float()
    ref = full[len(prompt) - 1:len(prompt) - 1 + len(cont)]
    want_plain = ref.log_softmax(-1)[torch.arange(5), torch.tensor(cont)]
    assert torch.allclose(plain_lp, want_plain, atol=1e-4)
    assert torch.allclose(law_lp[0], want_plain, atol=1e-4)  # the identity law is the plain model
    want_T = (ref / 0.6).log_softmax(-1)[torch.arange(5), torch.tensor(cont)]
    assert torch.allclose(law_lp[1], want_T, atol=1e-4)
    # the penalty counts the continuation so far: token 11 is penalized from its second occurrence on
    assert torch.allclose(law_lp[2][:1], want_plain[:1], atol=1e-4) and not torch.allclose(law_lp[2], want_plain)
    # the lever leaves the first (prefill-scored) token alone and moves the rest
    assert torch.allclose(law_lp[3][:1], want_plain[:1], atol=1e-4)
    assert (law_lp[3][1:] - want_plain[1:]).abs().max() > 1e-3
    with pytest.raises(SystemExit):
        law_damage.cell_plan(lb.Cell("n", "V", 1.0, 1.0, steer=(("Vx_s8", 0.5),), steer_rule="N"), bank, meds, {},
                             eng)
    # an act injection, uniform span: rows force the prompt under their own lever; the identity row stays plain and
    # the act row moves every scored token, the first included (its prompt was rewritten too)
    act = lb.Cell("act", "act", 1.0, 1.0, steer_span="uniform",
                  injection={"kind": "act", "sites": [4, 7], "neurons": [3, 10], "mu": [5.0, -5.0], "alpha": 1.0})
    eng2 = DecodeEngine(model, max_seq_len=96, dtype=torch.float32, prefill_backends=[SDPBackend.MATH], levers=1)
    slot = eng2.levers.bind(law_damage.cell_plan(act, bank, meds, {}, eng2))
    dec2 = MultiSampleDecoder(eng2, 2)
    act_lp, plain2 = law_damage.score_set(eng2, dec2, [act], [slot], prompt, cont)
    assert torch.allclose(plain2, want_plain, atol=1e-4)
    assert (act_lp[0] - want_plain).abs().min() > 1e-5
    with pytest.raises(ValueError):
        law_damage.score_set(eng2, dec2, [act, cells[0]], [slot, 0], prompt, cont)


def test_cell_injection_rows_and_requests(tmp_path: Path) -> None:
    inj = {"kind": "act", "sites": [6], "neurons": [780], "mu": [-0.85], "alpha": 0.5}
    c = lb.Cell("bb", "act", 0.75, 1.2, steer_span="uniform", injection=inj)
    assert lb.Cell.from_row(json.loads(json.dumps(c.to_row()))) == c
    with pytest.raises(ValueError):
        lb.Cell("x", "act", 0.75, 1.2, steer=(("v", 0.1),), injection=inj)
    seen: list[dict] = []
    comp = tmp_path / "c.jsonl"
    comp.write_text(json.dumps({"id": "p0", "prompt": "Once"}) + "\n")
    asyncio.run(lb.generate([c], [lb.load_panel("seed", comp)], "http://gw:1", tmp_path / "run", lb.GenConfig(), 4,
                            transport=_mock_server(seen)))
    assert seen[0]["injection"] == inj and seen[0]["steer_span"] == "uniform" and "vectors" not in seen[0]


def test_score_parallel_merges_chunks() -> None:
    items = [{"id": str(i)} for i in range(25)]
    out = lb.score_parallel(items, lambda part: {x["id"]: [1.0, 2] for x in part}, workers=3, chunk=4)
    assert sorted(out, key=int) == [str(i) for i in range(25)]
