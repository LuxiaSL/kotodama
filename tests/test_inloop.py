"""In-loop evals: schedule, shard/merge, riders-lite pieces (selection, re-lettered
shots, error typing, summary), server-vs-in-loop scoring parity, and the evaluator
end to end — single process, resume skip, and 2-process gloo == single process."""

from __future__ import annotations

import json
import math
import os
import socket
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kotodama.eval.riders import items as ri
from kotodama.eval.riders import lite
from kotodama.eval.riders.errors import coarse_error_type, error_types, passed_through, picked
from kotodama.eval.riders.score import (LogprobLimitError, plan_batches, score_logprob_batch,
                                        score_logprob_items)
from kotodama.training.inloop import (InloopConfig, InloopEvaluator, inloop_schedule,
                                      merge_shards, parse_schedule, parse_suites, shard)
from kotodama.training.muon import decay_start_step

VOCAB = 49152  # the presets tokenizer's size: rider candidates are real token ids


# ── schedule ───────────────────────────────────────────────────────────────────

def test_schedule_default_fractions():
    s = inloop_schedule(1000, decay_start_step(1000, 0.9))
    stable = [0] + [25 * k - 1 for k in range(1, 36)]          # 2.5%, 5%, ... 87.5%
    decay = [899 + 5 * k for k in range(20)]                   # 90%, 90.5%, ... 99.5%
    assert s == sorted(set(stable + decay + [999]))
    assert len(s) == 57


def test_schedule_threshold_rule_matches_save_schedule():
    # fraction f fires at the first step whose cumulative tokens reach f * total
    s = inloop_schedule(400, 400, "0:0.25")
    assert s == [0, 99, 199, 299, 399]


def test_schedule_stop_at_step_and_tiny_runs():
    s = inloop_schedule(1000, 900, final_step=499)
    assert s[-1] == 499 and all(x <= 499 for x in s) and 474 in s and 499 in s
    assert inloop_schedule(1, 0) == [0]
    assert inloop_schedule(3, 2, "0:0.5") == [0, 1, 2]
    assert inloop_schedule(0, 0) == []


def test_schedule_with_batch_ramp_tokens():
    # first 10 steps carry 1 token each, the rest 10: token fractions != step fractions
    per = [1] * 10 + [10] * 90
    cum, t = [], 0
    for x in per:
        t += x
        cum.append(t)
    s = inloop_schedule(100, 100, "0:0.5", cum_tokens=cum)
    total = cum[-1]
    half = next(i for i, c in enumerate(cum) if c >= 0.5 * total)
    assert s == [0, half, 99] and half != 49


def test_schedule_decay_phase_ordering_and_errors():
    # decay phase sorts by its resolved start even if written first
    assert inloop_schedule(100, 50, "decay:0.1,0:0.25") == inloop_schedule(100, 50, "0:0.25,decay:0.1")
    for bad in ("", "0", "x:0.1", "0:0", "0:1.5", "1.0:0.1"):
        with pytest.raises(ValueError):
            parse_schedule(bad)
    with pytest.raises(ValueError):
        inloop_schedule(10, 5, cum_tokens=[1, 2, 3])


def test_train_flags_map_to_config(monkeypatch):
    from kotodama.training import train as train_mod
    monkeypatch.setattr("sys.argv", ["train"])
    off = InloopConfig.from_args(train_mod.parse_args())
    assert off.suites == [] and off.schedule == "0:0.025,decay:0.005" and off.riders_relet
    monkeypatch.setattr("sys.argv", [
        "train", "--inloop_evals", "riders,induction", "--inloop_schedule", "0:0.1",
        "--inloop_riders_n", "50", "--no_inloop_riders_relet", "--inloop_max_batch_tokens",
        "4096", "--inloop_induction_len", "64", "--inloop_nca_gates_dir", "/x"])
    on = InloopConfig.from_args(train_mod.parse_args())
    assert on.suites == ["riders", "induction"] and on.schedule == "0:0.1"
    assert on.riders_n == 50 and not on.riders_relet and on.max_batch_tokens == 4096
    assert on.induction_len == 64 and on.nca_gates_dir == "/x"


def test_parse_suites():
    assert parse_suites(None) == [] and parse_suites("") == [] and parse_suites(" , ") == []
    assert parse_suites("riders-lite, induction,riders,uncheatable-mini,nca") == \
        ["riders", "induction", "uncheatable", "nca_gates"]
    assert parse_suites("on") == parse_suites("default") == ["riders"]  # riders-lite = default
    with pytest.raises(ValueError):
        parse_suites("riders,bogus")


# ── shard / merge ──────────────────────────────────────────────────────────────

def test_shard_merge_roundtrip():
    data = [f"x{i}" for i in range(11)]
    for w in (1, 2, 3, 8, 16):
        parts = [shard(data, r, w) for r in range(w)]
        assert sum(len(p) for p in parts) == len(data)
        assert merge_shards(parts, len(data)) == data
    parts = [shard(data, r, 2) for r in range(2)]
    with pytest.raises(ValueError):
        merge_shards(parts[:1], len(data))                  # missing
    with pytest.raises(ValueError):
        merge_shards(parts + [parts[0]], len(data))         # duplicate
    with pytest.raises(ValueError):
        shard(data, 2, 2)


# ── riders-lite pieces ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def battery() -> tuple[list[ri.Item], list[ri.Item]]:
    return ri.generate(list(ri.BANKED_VERIFIED_NAMES), 20, 5, ri.DEFAULT_SEED)


def test_select_items_balanced_and_deterministic(battery):
    its, _ = battery
    sel = lite.select_items(its, 40, seed=3)
    assert len(sel) == 40 and all(it.tier == "std" for it in sel)
    cells = [(it.n, it.m) for it in sel]
    counts = {c: cells.count(c) for c in ri.CELLS}
    assert set(counts.values()) <= {2, 3} and sum(counts.values()) == 40
    assert [it.item_id for it in lite.select_items(its, 40, seed=3)] == [it.item_id for it in sel]
    assert [it.item_id for it in lite.select_items(its, 40, seed=4)] != [it.item_id for it in sel]
    assert len({it.item_id for it in lite.select_items(its, 10_000)}) == 20 * len(ri.CELLS)


def test_reletter_shots(battery):
    its, shots = battery
    relet = lite.reletter_shots(shots, its)
    assert len(relet) == len(shots) == ri.N_SHOTS
    for s, r in zip(shots, relet):
        assert set(s.boxes0) == {"F", "G", "H"}
        assert set(r.boxes0) == {"A", "B", "C"} and set(r.final) == {"A", "B", "C"}
        assert r.query_box == lite.RELETTER_MAP[s.query_box] and r.gold == s.gold
        assert list(r.boxes0.values()) == list(s.boxes0.values())
        final = ri._sim(r.boxes0, [ri.Op(**o) for o in r.ops])
        assert final[r.query_box] == r.gold
        for o_s, o_r in zip(s.ops, r.ops):
            assert o_r["i"] == lite.RELETTER_MAP[o_s["i"]]
            assert o_r.get("j") == (lite.RELETTER_MAP[o_s["j"]] if o_s.get("j") else None)
        # the rendered shot shows only the new letters
        text = ri.render_narrative(r, True)
        assert "Box A" in text and "Box F" not in text and "Box G" not in text
    # a shot whose stored gold disagrees with the simulator is refused
    bad = ri.Item(**{**shots[0].__dict__, "gold": "nonsense"})
    with pytest.raises(ValueError):
        lite.reletter_shots([bad])
    # targets the items use are refused
    with pytest.raises(ValueError):
        lite.reletter_shots(shots, its, mapping={"F": "G", "G": "F", "H": "K"})


def test_build_jobs(battery):
    its, shots = battery
    sel = lite.select_items(its, 3)
    jobs = lite.build_jobs(sel, shots, lite.reletter_shots(shots, sel))
    assert len(jobs) == 3 * len(lite.REGISTERS) * len(lite.CONDITIONS)
    assert len({j.id for j in jobs}) == len(jobs)
    by = {(j.item_index, j.renderer, j.condition): j for j in jobs}
    p3, pr = by[(0, "narrative", "3shot")].prompt, by[(0, "narrative", "3shot_relet")].prompt
    assert p3 != pr and p3.split("###")[-1] == pr.split("###")[-1]  # same query, other shots
    assert by[(0, "chat", "0shot")].prompt.startswith("<|im_start|>")
    assert by[(0, "code", "0shot")].nshots == 0 and by[(0, "code", "3shot")].nshots == 3


def _item(**kw: Any) -> ri.Item:
    base = dict(item_id="t", n=3, m=1, tier="std", recency_control=False, gold_decl_pos=0)
    base.update(kw)
    return ri.Item(**base)


def test_error_typing():
    # F=bell G=cup H=key; swap(F,G) -> F=cup G=bell H=key; query F -> gold cup
    it = _item(boxes0={"F": "bell", "G": "cup", "H": "key"},
               ops=[{"kind": "swap", "i": "F", "j": "G", "obj": None}],
               final={"F": "cup", "G": "bell", "H": "key"}, query_box="F", gold="cup",
               candidates=["bell", "cup", "key"], mention_order=["bell", "cup", "key"])
    assert coarse_error_type(it, "bell") == "init_query"
    assert error_types(it, "bell")[0] == "init_query"
    assert coarse_error_type(it, "key") == "wrong_box"
    assert error_types(it, "key") == ["other_final", "other_init", "last_ment"]
    assert coarse_error_type(it, "lamp") == "other" and error_types(it, "lamp") == ["other"]
    # overwrite: G=cup -> G=pen; query G (gold pen); cup passed through G (stale, not a box's)
    it2 = _item(boxes0={"F": "bell", "G": "cup", "H": "key"},
                ops=[{"kind": "overwrite", "i": "G", "j": None, "obj": "pen"}],
                final={"F": "bell", "G": "pen", "H": "key"}, query_box="G", gold="pen",
                candidates=["bell", "cup", "key", "pen"],
                mention_order=["bell", "cup", "key", "pen"])
    assert passed_through(it2, "G") == {"cup"}
    assert coarse_error_type(it2, "cup") == "init_query"
    assert coarse_error_type(it2, "bell") == "wrong_box"
    assert error_types(it2, "bell")[:2] == ["box_F", "other_final"]
    # move G->H: H=cup, G empty; query H (gold cup); key was H's initial -> init_query
    it3 = _item(boxes0={"F": "bell", "G": "cup", "H": "key"},
                ops=[{"kind": "move", "i": "G", "j": "H", "obj": None}],
                final={"F": "bell", "G": None, "H": "cup"}, query_box="H", gold="cup",
                candidates=["bell", "cup", "key"], mention_order=["bell", "cup", "key"])
    assert coarse_error_type(it3, "key") == "init_query"
    assert picked({"cand_lps": {"bell": -2.0, "key": -0.5, "cup": -1.0}}) == "key"


def _rows(rd: str, cond: str, ranks: list[int], etype: str = "wrong_box") -> list[dict]:
    return [{"renderer": rd, "condition": cond, "gold_rank_cand": r, "n_candidates": 4,
             "margin": 1.0 if r == 1 else -1.0,
             **({} if r == 1 else {"error_type": etype})} for r in ranks]


def test_summarize():
    rows = (_rows("narrative", "0shot", [1, 2, 2, 2])
            + _rows("narrative", "3shot", [1, 1, 1, 2], "init_query")
            + _rows("code", "3shot_relet", [1, 1, 1, 1]))
    m = lite.summarize(rows)
    assert m["riders/narrative/0shot/acc"] == 0.25
    assert m["riders/narrative/0shot/cc_acc"] == pytest.approx(0.0)
    assert m["riders/narrative/3shot/cc_acc"] == pytest.approx((0.75 - 0.25) / 0.75)
    assert m["riders/narrative/shot_gap"] == pytest.approx(m["riders/narrative/3shot/cc_acc"])
    assert m["riders/narrative/0shot/err_wrong_box"] == 1.0
    assert m["riders/narrative/3shot/err_init_query"] == 1.0
    assert m["riders/narrative/0shot/margin"] == pytest.approx(-0.5)
    assert m["riders/prose/3shot/acc"] == 0.75                      # pooled prose
    assert m["riders/code/3shot_relet/err_other"] == 0.0             # no wrong rows
    assert "riders/code/relet_gap" not in m                          # no code 3shot


# ── scoring: shared function, server parity ────────────────────────────────────

@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer

        from kotodama.presets import TOKENIZER_NAME
        return AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as exc:  # offline without a cache
        pytest.skip(f"tokenizer unavailable: {exc}")


def _tiny(seed: int = 0) -> torch.nn.Module:
    from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
    torch.manual_seed(seed)
    return KotodamaModel(KotodamaModelConfig(
        hidden_size=64, num_layers=2, num_attention_heads=2, num_kv_heads=1, head_dim=32,
        intermediate_size=128, vocab_size=VOCAB, max_position_embeddings=1024,
        attn_impl="sdpa", use_liger=False, attn_res=True, attn_res_boundaries=[0, 1])).eval()


def _payload(battery, n: int = 3) -> list[dict[str, Any]]:
    its, shots = battery
    sel = lite.select_items(its, n)
    jobs = lite.build_jobs(sel, shots, lite.reletter_shots(shots, sel),
                           registers=("code", "chat"), conditions=("0shot", "3shot_relet"))
    return [{"id": j.id, "prompt": j.prompt, "candidates": j.candidates} for j in jobs]


def test_plan_batches():
    lens = [5, 1, 9, 3, 3, 7]
    b = plan_batches(lens, max_batch=2, max_total_tokens=100)
    assert sorted(i for x in b for i in x) == list(range(6)) and all(len(x) <= 2 for x in b)
    assert [lens[i] for x in b for i in x] == sorted(lens)
    assert plan_batches(lens, max_batch=64, max_total_tokens=10) == [[1, 3, 4], [0], [5], [2]]
    with pytest.raises(ValueError):
        plan_batches(lens, max_batch=0)


def test_score_limits(tok):
    model = _tiny()
    fn = lambda b: model(b)["logits"]  # noqa: E731
    item = {"id": "a", "prompt": "Box F contains the", "candidates": [" bell"]}
    with pytest.raises(LogprobLimitError, match="too long"):
        score_logprob_batch(fn, tok, [item], 0, "cpu", max_seq_len=3)
    with pytest.raises(LogprobLimitError, match="too large"):
        score_logprob_batch(fn, tok, [item, item], 0, "cpu", max_total_tokens=5)
    assert score_logprob_batch(fn, tok, [], 5, "cpu") == []


def test_server_and_inloop_scoring_identical(tok, battery, monkeypatch):
    """The /logprobs handler and the in-loop path run one function: same items,
    same model -> identical result dicts (ids, logprobs, ranks, entropy, top-n)."""
    pytest.importorskip("fastapi")
    try:
        from kotodama.serve import server
    except Exception as exc:  # pragma: no cover — server deps unavailable
        pytest.skip(f"server import unavailable: {exc}")
    model = _tiny()
    payload = _payload(battery)
    monkeypatch.setattr(server, "_model", model)
    monkeypatch.setattr(server, "_tokenizer", tok)
    monkeypatch.setattr(server, "_device", torch.device("cpu"))
    monkeypatch.setattr(server, "_max_seq_len", 1024)
    # the server scores whatever batch it is sent; send it the in-loop batch order
    lens = [len(tok.encode(p["prompt"])) for p in payload]
    order = [i for b in plan_batches(lens, 64, 10**9) for i in b]
    req = server.LogprobRequest(items=[payload[i] for i in order], top_n=5)
    srv = {r["id"]: r for r in server._score_logprobs(req)["results"]}
    with torch.no_grad():
        ours = score_logprob_items(lambda b: model(b)["logits"], tok, payload, 5, "cpu",
                                   max_seq_len=1024, max_total_tokens=10**9)
    assert [r["id"] for r in ours] == [p["id"] for p in payload]
    for r in ours:
        assert r == srv[r["id"]]
    assert all(c["n_tokens"] == 1 for r in ours for c in r["candidates"])
    # limits still surface as HTTP 400 from the server
    monkeypatch.setattr(server, "_max_seq_len", 4)
    with pytest.raises(server.HTTPException) as ei:
        server._score_logprobs(req)
    assert ei.value.status_code == 400


# ── evaluator end to end ───────────────────────────────────────────────────────

def _cfg(**kw: Any) -> InloopConfig:
    base = dict(suites=["riders", "induction"], riders_n=4, max_batch=16,
                max_batch_tokens=1024, induction_n_seqs=6, induction_len=16)
    base.update(kw)
    return InloopConfig(**base)


def _comparable(m: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in m.items() if not k.endswith("time_s")}


def test_evaluator_single_process(tok, tmp_path):
    model = _tiny().train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    logged: list[tuple[int, dict[str, float]]] = []
    ev = InloopEvaluator(_cfg(), out_dir=tmp_path, rank=0, world_size=1,
                         device=torch.device("cpu"), max_seq_len=1024, vocab_size=VOCAB,
                         log_fn=lambda step, **m: logged.append((step, m)), tokenizer=tok)
    pending = ev.plan(total_steps=200, decay_start=180, final_step=199)
    assert pending[0] == 0 and pending[-1] == 199 and ev.due(0)
    m = ev.run(0, model, tokens_consumed=123)
    assert m is not None and model.training                       # mode restored
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert len(logged) == 1 and logged[0][0] == 0                  # ONE batched hm_log call
    for rd in ("narrative", "screenplay", "chat", "code", "prose"):
        for cond in ("0shot", "3shot", "3shot_relet"):
            assert 0.0 <= m[f"inloop/riders/{rd}/{cond}/acc"] <= 1.0
            assert m[f"inloop/riders/{rd}/{cond}/n"] == (12 if rd == "prose" else 4)
        assert f"inloop/riders/{rd}/shot_gap" in m and f"inloop/riders/{rd}/relet_gap" in m
    assert math.isfinite(m["inloop/induction/induction_gain"])
    assert m["inloop/time_s"] > 0 and m["inloop/errors"] == 0.0
    # per-row file: meta line + 4 items x 4 registers x 3 conditions + 1 induction row
    path = tmp_path / "inloop" / "step_00000000.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert lines[0]["suite"] == "_meta" and lines[0]["tokens"] == 123
    riders = [x for x in lines if x["suite"] == "riders"]
    assert len(riders) == 48 and len([x for x in lines if x["suite"] == "induction"]) == 1
    wrong = [x for x in riders if x["gold_rank_cand"] != 1]
    assert all(x["error_type"] in ("init_query", "wrong_box", "other") for x in wrong)
    assert {x["condition"] for x in riders} == {"0shot", "3shot", "3shot_relet"}
    # resume: a written point is not re-run, nor are points before start_step
    ev2 = InloopEvaluator(_cfg(), out_dir=tmp_path, rank=0, world_size=1,
                          device=torch.device("cpu"), max_seq_len=1024, vocab_size=VOCAB,
                          tokenizer=tok)
    pend2 = ev2.plan(total_steps=200, decay_start=180, final_step=199, start_step=0)
    assert 0 not in pend2 and pend2 == pending[1:]
    assert ev2.plan(200, 180, 199, start_step=100) == [s for s in pending if s >= 100]


def test_evaluator_suite_failure_is_reported_not_raised(tok, tmp_path, monkeypatch):
    model = _tiny()
    ev = InloopEvaluator(_cfg(), out_dir=tmp_path, rank=0, world_size=1,
                         device=torch.device("cpu"), max_seq_len=1024, vocab_size=VOCAB,
                         tokenizer=tok)
    import kotodama.eval.induction as ind
    monkeypatch.setattr(ind, "token_losses", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    m = ev.run(5, model)
    assert m is not None and m["inloop/errors"] == 1.0
    assert "inloop/riders/code/0shot/acc" in m and "inloop/induction/induction_gain" not in m
    meta = json.loads((tmp_path / "inloop" / "step_00000005.jsonl").read_text().splitlines()[0])
    assert "boom" in meta["errors"]["induction"]


def test_evaluator_requires_paths(tok, tmp_path):
    kw = dict(out_dir=tmp_path, rank=0, world_size=1, device=torch.device("cpu"),
              max_seq_len=1024, vocab_size=VOCAB, tokenizer=tok)
    with pytest.raises(ValueError, match="uncheatable_path"):
        InloopEvaluator(_cfg(suites=["uncheatable"]), **kw)
    with pytest.raises(ValueError, match="nca_gates_dir"):
        InloopEvaluator(_cfg(suites=["nca_gates"]), **kw)
    with pytest.raises(ValueError, match="go together"):
        InloopEvaluator(_cfg(riders_items=str(tmp_path / "x.jsonl")), **kw)
    with pytest.raises(ValueError, match="language vocab"):          # NCA-phase vocab
        InloopEvaluator(_cfg(), **{**kw, "vocab_size": 1000})
    InloopEvaluator(_cfg(suites=["induction"]), **{**kw, "vocab_size": 1000})  # fine


def test_evaluator_uncheatable(tok, tmp_path):
    data = tmp_path / "unc"
    data.mkdir()
    (data / "wiki.jsonl").write_text("".join(json.dumps({"content": f"Doc {k}: the cat sat on the mat. " * (k + 2)}) + "\n" for k in range(3)))
    (data / "code.jsonl").write_text(json.dumps({"content": "def f(x):\n    return x + 1\n"}) + "\n")
    ev = InloopEvaluator(_cfg(suites=["uncheatable"], uncheatable_path=str(data),
                              uncheatable_docs=2),
                         out_dir=tmp_path, rank=0, world_size=1, device=torch.device("cpu"),
                         max_seq_len=1024, vocab_size=VOCAB, tokenizer=tok)
    assert len(ev.u_docs) == 3                                     # 2 wiki + 1 code
    m = ev.run(0, _tiny())
    assert m is not None
    for k in ("inloop/uncheatable/wiki/bpb", "inloop/uncheatable/code/bpb",
              "inloop/uncheatable/mean_bpb"):
        assert math.isfinite(m[k]) and m[k] > 0
    # random init on a 49K vocab: ~log2(V) bits/token, tokens ~ 3-5 bytes
    assert 1.0 < m["inloop/uncheatable/mean_bpb"] < 10.0


def test_evaluator_nca_gates(tmp_path):
    from kotodama.eval.nca_gates import resolve_token_map
    from kotodama.nca.dyadic import N_VALS
    ev_dir = tmp_path / "nca"
    ev_dir.mkdir()
    tm = resolve_token_map(ev_dir)                                 # no meta.json: legacy layout
    rows = [{"item_id": f"u{k}", "prefix": [1 + (k * 7 + j) % 50 for j in range(12)],
             "gold_val": k % N_VALS, "post_overwrite": k % 2 == 1} for k in range(9)]
    (ev_dir / "eval_ledger_unique.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (ev_dir / "eval_ledger_overwrite.jsonl").write_text(json.dumps(rows[0]) + "\n")
    model = _tiny()
    ev = InloopEvaluator(_cfg(suites=["nca_gates"], nca_gates_dir=str(ev_dir),
                              nca_gates_max_items=5),
                         out_dir=tmp_path, rank=0, world_size=1, device=torch.device("cpu"),
                         max_seq_len=1024, vocab_size=max(VOCAB, tm["VAL0"] + N_VALS))
    assert len(ev.n_items) == 6                                    # 5 capped + 1
    m = ev.run(0, model)
    assert m is not None and m["inloop/errors"] == 0.0
    for k in ("inloop/nca_gates/unique/all/acc", "inloop/nca_gates/overwrite/all/acc",
              "inloop/nca_gates/unique/post_overwrite/acc", "inloop/nca_gates/unique/mean_gold_lp"):
        assert k in m and math.isfinite(m[k])


# ── 2-process gloo == single process ───────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int, out_dir: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from transformers import AutoTokenizer

        from kotodama.presets import TOKENIZER_NAME
        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        ev = InloopEvaluator(_cfg(), out_dir=out_dir, rank=rank, world_size=world,
                             device=torch.device("cpu"), max_seq_len=1024,
                             vocab_size=VOCAB, tokenizer=tok)
        ev.plan(total_steps=10, decay_start=9, final_step=9)
        m = ev.run(0, _tiny())
        # a rank that saw SIGTERM makes EVERY rank skip the point (no deadlock)
        skipped = ev.run(9, _tiny(), stop_requested=(rank == 1))
        if rank == 0:
            Path(out_dir, "metrics.json").write_text(json.dumps(
                {"m": m, "skipped": skipped is None}))
        else:
            assert m is None
    finally:
        dist.destroy_process_group()


def test_two_process_gloo_matches_single(tok, tmp_path):
    single = InloopEvaluator(_cfg(), out_dir=tmp_path / "single", rank=0, world_size=1,
                             device=torch.device("cpu"), max_seq_len=1024,
                             vocab_size=VOCAB, tokenizer=tok).run(0, _tiny())
    assert single is not None
    out = tmp_path / "ddp"
    mp.spawn(_worker, args=(2, _free_port(), str(out)), nprocs=2, join=True)
    got = json.loads((out / "metrics.json").read_text())
    assert got["skipped"] is True
    a, b = _comparable(single), _comparable(got["m"])
    assert a.keys() == b.keys()
    for k in a:
        if k.endswith(("/acc", "/cc_acc", "/n")) or "/err_" in k or k.endswith("_gap"):
            assert a[k] == b[k], k                                 # discrete reads: exact
        else:
            assert a[k] == pytest.approx(b[k], rel=1e-4, abs=1e-5), k
    rows_s = (tmp_path / "single" / "inloop" / "step_00000000.jsonl").read_text().splitlines()[1:]
    rows_d = (out / "inloop" / "step_00000000.jsonl").read_text().splitlines()[1:]
    ids = lambda rows: [(r.get("item_id"), r.get("renderer"), r.get("condition"))  # noqa: E731
                        for r in map(json.loads, rows)]
    assert ids(rows_s) == ids(rows_d)                              # same global row order
    assert not (out / "inloop" / "step_00000009.jsonl").exists()
