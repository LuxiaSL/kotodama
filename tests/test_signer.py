"""Signing records by replay through the decode engine (kotodama.model.signer) on a tiny looped model, CPU fp32.

What is pinned here:
  * a generated row's record signs to the bytes its generation captured, every surface, whatever it is batched with:
    other prompts, shorter and longer spans (filler past a row's own span is trimmed), a decode lever, unused rows,
    and more records than one batch holds;
  * records the signer cannot reproduce or fully hold are skipped with the reason (other weights, steering or
    adapter, a lever without its plan, too few generated ids, a span past the sign tail), the rest still sign;
  * a batch the reducer refuses is retried row by row and only the refused row is skipped, with the reducer's error;
  * records and lever plans round-trip through their files, and malformed records are refused naming the line;
  * results are written as one feature matrix plus one JSON line per record;
  * the sign spool (kotodama.serve.sign): records appended by writers are signed one batch per drain, in order, each
    batch written before the spool's state moves past it; a line still being written waits; a drain that finds the
    spool locked by another does nothing; a signer built by RecordSigner.build signs as the hand-built one does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.levers import LeverPlan
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.model.multi_decode import CascadeDecoder
from kotodama.model.signer import (RecordSigner, SignRecord, load_plans, read_records, save_plans, trim,
                                   write_records, write_results)
from kotodama.model.signing import ReducedRow, SignSpec

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
ROWS = 4
LAYERS = (0, 3, 5, 8, 10)
STEPS = 6
SPEC = SignSpec(LAYERS, tail=12, depth=True, attn=True, width=40)
PARAMS = SamplingParams(temperature=0.8, repetition_penalty=1.2)


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    torch.manual_seed(20261029)
    model = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for layer in model.layers:
            for name, p in layer.named_parameters():
                if not name.startswith(("attn.", "ffn.")):
                    p.add_(0.3 * torch.randn(p.shape, generator=g))
    return model


def _ids(n: int, seed: int) -> list[int]:
    return torch.randint(4, CFG["vocab_size"], (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _setup(model: KotodamaModel):
    eng = DecodeEngine(model, max_seq_len=96, dtype=torch.float32, prefill_backends=[SDPBackend.MATH], levers=2)
    dec = CascadeDecoder(eng, ROWS, groups=4, group_rows=2, tail_tokens=32, seeded=True, sign=SPEC)
    dec.compile_step(regional=True, backend="aot_eager")
    return eng, dec


class _Keep:
    """A stand-in reducer that keeps the rows it is handed (features: the span length)."""

    reducer_identity = "keep-v1"

    def __init__(self, refuse_positions: int = -1) -> None:
        self.rows: dict[str, object] = {}
        self.calls: list[int] = []
        self.refuse = refuse_positions

    def reduce(self, rows, *, token_ids, engine):
        self.calls.append(len(rows))
        if any(r.positions == self.refuse for r in rows):
            raise ValueError(f"span of {self.refuse} positions refused")
        out = []
        for r, ids in zip(rows, token_ids):
            self.rows[tuple(ids)] = r
            # two feature layouts, as a reducer's below and above a step-count threshold
            digest = "short" if r.positions < 3 else "long"
            feats = [r.positions, len(ids)] + ([0.5] if digest == "long" else [])
            out.append(ReducedRow(features=np.array(feats, np.float32), names_digest=digest,
                                  lane_id=f"keep:{engine['weights'][:8]}",
                                  meta={"steps": r.positions, "feature_names": [f"{digest}{j}" for j in range(len(feats))]}))
        return out


def _generate(eng, dec, prompts: list[list[int]], slot: int) -> tuple[list[SignRecord], list]:
    """One prompt per row (rows 1 and 3 on lever ``slot``), STEPS steps; each row's record and signed span."""
    dec.release(range(ROWS))
    first = []
    for r, prompt in enumerate(prompts):
        s = slot if r % 2 else 0
        first += dec.admit([r], eng.prefill(torch.tensor([prompt]), slot=s), PARAMS, slots=[s], seeds=[10 + r])
    gen = [[t] for t in first]
    for _ in range(STEPS):
        for r, t in enumerate(dec.step(PARAMS).view(-1).tolist()):
            gen[r].append(t)
    lever = eng.levers.plan(slot).digest()
    recs = [SignRecord(request_id=f"r{r}", prompt_ids=tuple(prompts[r]), generated_ids=tuple(gen[r]),
                       weights=dec._sign_weights, lever=lever if r % 2 else None) for r in range(ROWS)]
    return recs, [dec.signed(r) for r in range(ROWS)]


def _same(a, b) -> None:
    assert (a.positions, a.kept, a.start) == (b.positions, b.kept, b.start)
    for name, x in a.surfaces.items():
        y = b.surfaces[name]
        if isinstance(x, list):
            assert len(x) == len(y) and all(torch.equal(u, v) for u, v in zip(x, y)), name
        else:
            assert torch.equal(x, y), name


@torch.inference_mode()
def test_records_sign_to_their_generation_bytes(model: KotodamaModel) -> None:
    eng, dec = _setup(model)
    rng = np.random.default_rng(0)
    plan = LeverPlan(attn={5: rng.standard_normal(CFG["hidden_size"]).astype(np.float32)}, span="decode")
    slot = eng.levers.bind(plan)
    prompts = [_ids(7, 1), _ids(11, 2), _ids(7, 1), _ids(5, 3)]
    recs, gen = _generate(eng, dec, prompts, slot)
    eng.levers.release(slot)
    # shorter spans: a generation's prefix is the generation of fewer tokens, its signature the prefix
    cut = {1: 4, 2: 3}
    for r, n in cut.items():
        recs[r] = SignRecord(request_id=recs[r].request_id, prompt_ids=recs[r].prompt_ids,
                             generated_ids=recs[r].generated_ids[:n], weights=recs[r].weights, lever=recs[r].lever)
        gen[r] = trim(gen[r], n - 1)
    keep = _Keep()
    signer = RecordSigner(dec, keep, compile_mode="aot_eager", plans={plan.digest(): plan})
    # two of them twice over: more records than one batch's rows, the duplicates beside other batchmates
    order = [3, 0, 1, 2, 1, 3]
    results = signer.sign([recs[i] for i in order])
    assert all(res.skipped is None for res in results)
    assert len(keep.calls) >= 2 and sum(keep.calls) == len(order)
    for i in set(order):
        _same(gen[i], keep.rows[recs[i].prompt_ids + recs[i].generated_ids])
    assert results[0].reduced.meta["steps"] == STEPS and results[0].reduced.lane_id.startswith("keep:")
    assert signer.identity["decoder"]["rows"] == ROWS and signer.identity["sign"]["tail"] == SPEC.tail
    rt = json.loads(json.dumps(signer.identity))
    again = signer.sign([SignRecord(request_id="x", prompt_ids=recs[0].prompt_ids,
                                    generated_ids=recs[0].generated_ids, weights=recs[0].weights, engine=rt)])
    assert again[0].same_engine and not results[0].same_engine


@torch.inference_mode()
def test_unsignable_records_are_skipped_with_the_reason(model: KotodamaModel) -> None:
    eng, dec = _setup(model)
    w = dec._sign_weights
    ok = SignRecord(request_id="ok", prompt_ids=tuple(_ids(6, 4)), generated_ids=tuple(_ids(5, 5)), weights=w)
    bad = {
        "weights": SignRecord(request_id="w", prompt_ids=(5, 6), generated_ids=(7, 8, 9), weights="0" * 64),
        "steering": SignRecord(request_id="s", prompt_ids=(5, 6), generated_ids=(7, 8), weights=w, steering="bank"),
        "adapter": SignRecord(request_id="a", prompt_ids=(5, 6), generated_ids=(7, 8), weights=w, adapter="lora"),
        "has no plan": SignRecord(request_id="l", prompt_ids=(5, 6), generated_ids=(7, 8), weights=w, lever="ab12"),
        "no fed position": SignRecord(request_id="n", prompt_ids=(5, 6), generated_ids=(7,), weights=w),
        "sign tail": SignRecord(request_id="t", prompt_ids=(5, 6), generated_ids=tuple(range(4, 4 + SPEC.tail + 2)),
                                weights=w),
        "width": SignRecord(request_id="c", prompt_ids=tuple(range(4, 40)), generated_ids=(7, 8, 9, 10, 11, 12),
                            weights=w),
    }
    results = RecordSigner(dec, _Keep(), compile_mode="aot_eager").sign([ok, *bad.values()])
    assert results[0].skipped is None and results[0].reduced is not None
    for res, why in zip(results[1:], bad):
        assert res.reduced is None and why in res.skipped, (why, res.skipped)


@torch.inference_mode()
def test_a_refused_batch_is_retried_row_by_row(model: KotodamaModel) -> None:
    eng, dec = _setup(model)
    w = dec._sign_weights
    recs = [SignRecord(request_id=f"r{n}", prompt_ids=tuple(_ids(6, n)), generated_ids=tuple(_ids(n + 1, 9)),
                       weights=w) for n in (2, 3, 4)]
    keep = _Keep(refuse_positions=3)
    results = RecordSigner(dec, keep, compile_mode="aot_eager").sign(recs)
    assert keep.calls == [3, 1, 1, 1]
    assert results[0].reduced is not None and results[2].reduced is not None
    assert results[1].reduced is None and "span of 3 positions refused" in results[1].skipped


def test_records_and_plans_round_trip(tmp_path) -> None:
    recs = [SignRecord(request_id="a", prompt_ids=(1, 2, 3), generated_ids=(4, 5), weights="f" * 64, lever="d",
                       dose=0.5, engine={"width": 128, "sign": {"layers": (0, 3)}}, meta={"seed": 7}),
            SignRecord(request_id="b", prompt_ids=(9,), generated_ids=(), weights="e" * 64)]
    path = tmp_path / "records.jsonl"
    assert write_records(path, recs) == 2
    back = list(read_records(path))
    assert [r.to_json() for r in back] == [json.loads(json.dumps(r.to_json())) for r in recs]
    with open(path, "a") as f:
        f.write(json.dumps({"format": "kotodama-sign-record/1", "request_id": "c", "prompt_ids": [1],
                            "generated_ids": [2], "weights": "e", "surprise": 1}) + "\n")
    with pytest.raises(ValueError, match=r"records.jsonl:3: unknown sign record fields \['surprise'\]"):
        list(read_records(path))
    with pytest.raises(ValueError, match="weights digest"):
        SignRecord(request_id="x", prompt_ids=(1,), generated_ids=(2,), weights="")
    rng = np.random.default_rng(1)
    plans = [LeverPlan(attn={2: rng.standard_normal(8).astype(np.float32)}, span="decode"),
             LeverPlan(mlp={1: rng.standard_normal(8).astype(np.float32)},
                       act={3: rng.standard_normal((2, 16)).astype(np.float32)})]
    digests = save_plans(tmp_path / "plans.npz", plans)
    loaded = load_plans(tmp_path / "plans.npz")
    assert sorted(loaded) == sorted(digests) and all(loaded[p.digest()].digest() == p.digest() for p in plans)


@torch.inference_mode()
def test_results_are_written_as_features_and_rows(model: KotodamaModel, tmp_path) -> None:
    eng, dec = _setup(model)
    w = dec._sign_weights
    recs = [SignRecord(request_id="a", prompt_ids=tuple(_ids(6, 1)), generated_ids=(5, 6, 7), weights=w,
                       meta={"seed": 1}),
            SignRecord(request_id="b", prompt_ids=(5,), generated_ids=(6,), weights=w)]
    recs.append(SignRecord(request_id="c", prompt_ids=tuple(_ids(4, 2)), generated_ids=(5, 6, 7, 8, 9), weights=w))
    signer = RecordSigner(dec, _Keep(), compile_mode="aot_eager", audit_every=1)
    summary = write_results(tmp_path, signer.sign(recs))
    assert summary == {"signed": 2, "skipped": 1, "layouts": 2, "audited": 2}
    assert signer.timing["rows"] == 2 and signer.timing["capture_s"] > 0 and signer.timing["reduce_s"] > 0
    rows = [json.loads(line) for line in open(tmp_path / "rows.jsonl")]
    assert [r["sign_id"] for r in rows] == ["a", "b", "c"] and [r["steps"] for r in rows] == [2, 0, 4]
    assert rows[0]["meta"] == {"seed": 1} and rows[0]["weights"] == w and "no fed position" in rows[1]["skipped"]
    assert rows[0]["reducer_meta"] == {"steps": 2}  # feature names are written once per layout, not per row
    short = np.load(tmp_path / rows[0]["file"])
    long = np.load(tmp_path / rows[2]["file"])
    assert short.tolist() == [[2.0, 9.0]] and long.tolist() == [[4.0, 9.0, 0.5]]
    assert json.loads((tmp_path / "names.json").read_text()) == {"short": ["short0", "short1"],
                                                                 "long": ["long0", "long1", "long2"]}
    raw = torch.load(tmp_path / "raw" / "c.pt")
    assert raw["positions"] == 4 and raw["surfaces"]["depth"].shape[0] == 4


@torch.inference_mode()
def test_the_spool_drains_one_batch_at_a_time(model: KotodamaModel, tmp_path) -> None:
    from kotodama.serve.sign import SignSpool

    eng, dec = _setup(model)
    w = dec._sign_weights
    rng = np.random.default_rng(3)
    plan = LeverPlan(mlp={4: rng.standard_normal(CFG["hidden_size"]).astype(np.float32)}, span="decode")
    recs = [SignRecord(request_id=f"q{i}", prompt_ids=tuple(_ids(5 + i, i)), generated_ids=tuple(_ids(3 + i, 50 + i)),
                       weights=w, lever=plan.digest() if i == 2 else None) for i in range(6)]
    spool = SignSpool(tmp_path / "spool", signed_dir=tmp_path / "bulk")
    spool.add("srv-a", recs[:4], plans=[plan])
    spool.add("srv-b", recs[4:])
    with open(tmp_path / "spool" / "records" / "srv-b.jsonl", "a") as f:
        f.write('{"format": "kotodama-sign-record/1", "request_id": "half')  # a writer mid-line
    assert spool.pending() == 6 and spool.has_pending()
    signer = RecordSigner(dec, _Keep(), compile_mode="aot_eager")
    first = spool.drain_once(signer)
    assert first["signed"] == ROWS and spool.pending() == 2
    second = spool.drain_once(signer)
    assert second["signed"] == 2 and spool.drain_once(signer) is None
    assert spool.pending() == 0 and spool.has_pending()  # only the half-written line is left
    batches = sorted(p for p in (tmp_path / "bulk").iterdir())
    index = [json.loads(line) for line in open(tmp_path / "spool" / "index.jsonl")]
    assert [Path(e["dir"]).name for e in index] == [b.name for b in batches] and index[0]["signed"] == ROWS
    ids = [json.loads(line)["sign_id"] for b in batches for line in open(b / "rows.jsonl")]
    assert ids == [r.request_id for r in recs]
    assert json.loads((batches[0] / "engine.json").read_text())["weights"] == w
    with spool._drain_lock() as mine:
        assert mine
        spool.add("srv-a", recs[:1])
        assert spool.drain_once(signer) is None  # another drain holds the spool
    built = RecordSigner.build(eng, _Keep(), layers=LAYERS, tail=SPEC.tail, width=SPEC.width, rows=ROWS, groups=4,
                               group_rows=2, tail_tokens=32, compile_mode="default", backend="aot_eager")
    lens = []
    prefill = eng.prefill
    eng.prefill = lambda ids, *a, **k: (lens.append(ids.shape[1]), prefill(ids, *a, **k))[1]
    try:
        built.warm()
    finally:
        eng.prefill = prefill
    # the warm-up prefill runs every block bucket the cache has room for (a long prompt's chunks launch kernels a
    # short one does not, and a recorded pin table must hold them)
    assert lens == [min(sum(eng.block_buckets), eng.max_seq_len - built.dec.tail_tokens)]
    assert built.identity["emulate_casts"] is False  # part of the lane: it changes the regions' arithmetic
    keep = built.reducer
    assert spool.drain_once(built)["signed"] == 1
    _same(keep.rows[recs[0].prompt_ids + recs[0].generated_ids],
          signer.reducer.rows[recs[0].prompt_ids + recs[0].generated_ids])


@torch.inference_mode()
def test_imported_rows_sign_like_served_ones(model: KotodamaModel, tmp_path) -> None:
    """Rows from a JSONL (ids, prompt_length, the server's injection field) become records whose lever is the plan the
    server makes of that injection: the worn rows sign to their generation's bytes under it, and an unworn import of
    the same tokens signs without it. An import is appended once per writer; a rule-N injection is refused."""
    from kotodama.serve.sign import SignSpool, import_rows
    from kotodama.serve.steering import injection_to_plan

    eng, dec = _setup(model)
    rng = np.random.default_rng(5)
    inj = {"sites": [5], "vectors": [rng.standard_normal(CFG["hidden_size"]).tolist()], "alpha": 0.7, "rule": "S",
           "kind": "attn"}
    plan = injection_to_plan(inj["sites"], inj["vectors"], inj["alpha"], inj["rule"], inj["kind"], "decode", None,
                             list(eng._boundaries), CFG["num_layers"], CFG["hidden_size"])
    slot = eng.levers.bind(plan)
    prompts = [_ids(7, 11), _ids(9, 12), _ids(6, 13), _ids(8, 14)]
    recs, gen = _generate(eng, dec, prompts, slot)
    eng.levers.release(slot)
    rows = [{"id": f"w{r}", "ids": list(rec.prompt_ids + rec.generated_ids), "prompt_length": len(rec.prompt_ids),
             "injection": inj if r % 2 else None, "meta": {"arm": "worn" if r % 2 else "unworn"}}
            for r, rec in enumerate(recs)]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    spool = SignSpool(tmp_path / "spool")
    assert import_rows(spool, path, "msteer", weights=dec._sign_weights, engine=eng) == len(rows)
    assert set(spool.plans()) == {plan.digest()}
    with pytest.raises(FileExistsError):
        import_rows(spool, path, "msteer", weights=dec._sign_weights, engine=eng)
    keep = _Keep()
    signer = RecordSigner(dec, keep, compile_mode="aot_eager")
    while spool.drain_once(signer) is not None:
        pass
    for r in range(ROWS):
        _same(gen[r], keep.rows[recs[r].prompt_ids + recs[r].generated_ids])
    signed = {json.loads(line)["sign_id"]: json.loads(line) for b in (tmp_path / "spool" / "signed").iterdir()
              for line in open(b / "rows.jsonl")}
    assert signed["w1"]["lever"] == plan.digest() and signed["w0"]["lever"] is None
    assert signed["w1"]["meta"]["arm"] == "worn" and signed["w0"]["meta"]["imported_by"] == "msteer"
    # the worn row's tokens imported unworn sign without the lever (another spool: _Keep keys rows by tokens)
    twin = tmp_path / "twin.jsonl"
    twin.write_text(json.dumps({**rows[1], "id": "u1", "injection": None}) + "\n")
    other = SignSpool(tmp_path / "spool-twin")
    import_rows(other, twin, "twin", weights=dec._sign_weights, engine=eng)
    unworn = _Keep()
    other.drain_once(RecordSigner(dec, unworn, compile_mode="aot_eager"))
    got = unworn.rows[recs[1].prompt_ids + recs[1].generated_ids]
    assert not torch.equal(got.surfaces["q"], gen[1].surfaces["q"])
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({**rows[1], "injection": {**inj, "rule": "N"}}) + "\n")
    with pytest.raises(ValueError, match="rule N"):
        import_rows(spool, bad, "bad", weights=dec._sign_weights, engine=eng)
    bad.write_text(json.dumps({**rows[0], "prompt_length": len(rows[0]["ids"])}) + "\n")
    with pytest.raises(ValueError, match="prompt_length"):
        import_rows(spool, bad, "bad2", weights=dec._sign_weights, engine=eng)
