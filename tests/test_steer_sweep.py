"""The neuron sweep's batch (scripts/sweep/steer_sweep.py) on a tiny looped model, CPU fp32, both spans: row 0 is
the damage reference, baseline rows reproduce it, neuron rows push the right column at the right layer, records
carry the explorer run-file keys, and the per-row arrays line up with the records."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine, SamplingParams
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.model.multi_decode import CascadeDecoder

_SPEC = importlib.util.spec_from_file_location(
    "steer_sweep", Path(__file__).resolve().parents[1] / "scripts" / "sweep" / "steer_sweep.py")
sweep = importlib.util.module_from_spec(_SPEC)
sys.modules["steer_sweep"] = sweep
_SPEC.loader.exec_module(sweep)

CFG = dict(hidden_size=64, num_layers=11, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, rope_theta=10000.0,
           qk_norm=True, tie_word_embeddings=True, attn_impl="sdpa", use_liger=False,
           attn_res=True, attn_res_boundaries=[0, 1, 3, 6, 9], loop_start=3, loop_len=3, loop_times=2)
ROWS = 8
RUN_FILE_KEYS = {"layer", "kind", "neuron", "ref_neuron", "prompt", "dose", "add", "rel_push", "text", "finish", "n_gen",
             "token_ids", "logprob", "capture_id"}


class _Tok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    torch.manual_seed(20261028)
    m = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    with torch.no_grad():
        for layer in m.layers:
            layer.ffn.down_proj.weight.mul_(4.0)  # neuron pushes visible at the tiny scale
    return m


def _args(mode: str):
    class A:
        pass
    a = A()
    a.mode, a.max_tokens, a.damage_tokens = mode, 6, 4
    return a


@pytest.mark.parametrize("mode", ["decode", "uniform"])
@torch.inference_mode()
def test_batch_records_and_damage(model: KotodamaModel, mode: str) -> None:
    sweep.TOKENIZER[0] = _Tok()
    eng = DecodeEngine(model, max_seq_len=96, dtype=torch.float32, prefill_backends=[SDPBackend.MATH],
                       levers=ROWS)
    cap = [7, 8] if mode == "uniform" else [8]
    dec = CascadeDecoder(eng, ROWS, groups=2, group_rows=4, tail_tokens=40, readout_k=3, capture_layers=cap)
    L = 7  # the loop's second pass over module 4
    units = sweep.Units(model, L, list(range(4)), random_frac=0.5, seed=0, p50=3.0)
    assert units.module_layer == 4 and len(units.units) == 6
    ids = torch.randint(4, CFG["vocab_size"], (1, 9), generator=torch.Generator().manual_seed(1))
    cont = torch.randint(4, CFG["vocab_size"], (1, 4), generator=torch.Generator().manual_seed(2))
    batch = [{"kind": "baseline", "neuron": -1, "prompt": "p", "dose": 0.0}]
    batch += [{"kind": k, "neuron": n, "prompt": "p", "dose": d} for k, n in units.units[:3] for d in (-0.7, 2.0)]
    out = sweep.run_batch(_args(mode), eng, dec, units, batch, ids, ids.shape[1], cont,
                          SamplingParams(temperature=0.0, repetition_penalty=1.0), frozenset(), [7, 8], [8], cap)
    recs = out["records"]
    assert len(recs) == len(batch) and all(RUN_FILE_KEYS <= set(r) for r in recs)
    base = recs[0]
    assert base["kind"] == "baseline" and base["damage"] == 0.0 and base["ref_neuron"] is None
    steered = [r for r in recs if r["kind"] != "baseline"]
    assert all(r["rel_push"] == abs(r["dose"]) and r["layer"] == L and r["span"] == mode for r in steered)
    assert any(r["token_ids"] != base["token_ids"] for r in steered)
    big = [r for r in steered if r["dose"] == 2.0]
    assert all(r["damage"] > 0 for r in big)  # a large push costs the unsteered continuation logprob
    for r in steered:  # the push has norm |dose| * p50 whatever the unit
        _, row = units.add_and_row(r["kind"], r["neuron"], r["dose"])
        assert np.isclose(np.linalg.norm(row), abs(r["dose"]) * 3.0, rtol=1e-4)
    arr = out["arrays"]
    assert arr["top_ids"].shape == (len(batch), 6, 3)
    assert (arr["top_ids"][0, :, 0] == np.array(base["token_ids"])).all()  # greedy: the chosen token is the top-1
    if mode == "uniform":
        assert arr["mlp_in"].shape == (len(batch), 2, CFG["hidden_size"])
    else:
        assert arr["mlp_in_first"].shape == (len(batch), 1, CFG["hidden_size"])
    assert eng.levers.free == ROWS - (len(batch) - 1)  # every steered job took one slot, the baseline none


@torch.inference_mode()
def test_shards_merge_into_layer_runs(model: KotodamaModel, tmp_path: Path) -> None:
    """Two shards' parts merge into one run per layer and span: lines in shard order, row-aligned sidecars."""
    _SPEC_M = importlib.util.spec_from_file_location(
        "sweep_merge", Path(__file__).resolve().parents[1] / "scripts" / "sweep" / "merge.py")
    merge = importlib.util.module_from_spec(_SPEC_M)
    _SPEC_M.loader.exec_module(merge)
    import json

    sweep.TOKENIZER[0] = _Tok()
    eng = DecodeEngine(model, max_seq_len=96, dtype=torch.float32, prefill_backends=[SDPBackend.MATH],
                       levers=ROWS)
    dec = CascadeDecoder(eng, ROWS, groups=2, group_rows=4, tail_tokens=40, readout_k=3, capture_layers=[8])
    units = sweep.Units(model, 7, list(range(4)), random_frac=0.5, seed=0, p50=3.0)
    ids = torch.randint(4, CFG["vocab_size"], (1, 9), generator=torch.Generator().manual_seed(3))
    cont = torch.randint(4, CFG["vocab_size"], (1, 4), generator=torch.Generator().manual_seed(4))
    shard_dir = tmp_path / "shards" / "decode"
    shard_dir.mkdir(parents=True)
    meta = {"checkpoint": "x", "span": "decode", "doses": [-0.7, 2.0], "prompt_ids": {"p": ids[0].tolist()},
            "dose_norms_sha256": "s", "max_tokens": 6, "seed": 0, "capture_offsets": [0, 1],
            "first_token_offsets": [1], "n_layers": CFG["num_layers"], "design_name": "main_rel"}
    jobs = [[{"kind": "baseline", "neuron": -1, "prompt": "p", "dose": 0.0},
             {"kind": "neuron", "neuron": 0, "prompt": "p", "dose": 2.0}],
            [{"kind": "neuron", "neuron": 1, "prompt": "p", "dose": -0.7},
             {"kind": "random", "neuron": 0, "prompt": "p", "dose": 2.0}]]
    for shard, batch in enumerate(jobs):
        (shard_dir / f"meta.s{shard}.json").write_text(json.dumps(meta))
        out = sweep.run_batch(_args("decode"), eng, dec, units, batch, ids, ids.shape[1], cont,
                              SamplingParams(temperature=0.0, repetition_penalty=1.0), frozenset(), [7, 8], [8], [8])
        with open(shard_dir / f"L7.s{shard}.jsonl", "a") as f:
            sweep.write_part(f, shard_dir, 7, shard, 0, out)
    np.savez(shard_dir / "L7.prompt_caps.npz", mlp_in=np.zeros((1, 2, CFG["hidden_size"]), np.float16),
             prompts=np.array(["p"]), layers=np.array([7, 8]))
    assert merge.merge_layer(shard_dir, tmp_path, "decode", 7, meta) == 4
    lines = [json.loads(x) for x in (tmp_path / "runs/decode/tsumugi-decode-L7.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in lines] == ["baseline", "neuron", "neuron", "random"]
    assert [r["row"] for r in lines] == [0, 1, 2, 3] and "part" not in lines[0]
    run_meta = json.loads((tmp_path / "runs/decode/tsumugi-decode-L7.meta.json").read_text())
    assert run_meta["layer"] == 7 and run_meta["module_layer"] == 4 and run_meta["design_name"] == "main_rel"
    topk = np.load(tmp_path / "topk/decode/tsumugi-decode-L7.topk.npz")
    assert topk["top_ids"].shape == (4, 6, 3)
    for r in lines:  # greedy: each row's sidecar top-1 is its own chosen token
        n = r["n_gen"]
        assert topk["top_ids"][r["row"], :n, 0].tolist() == r["token_ids"]
    first = np.load(tmp_path / "captures/decode/tsumugi-decode-L7.mlp_in_first.npy")
    assert first.shape == (4, 1, CFG["hidden_size"])
    cmeta = json.loads((tmp_path / "captures/decode/tsumugi-decode-L7.mlp_in.json").read_text())
    assert cmeta["per_prompt"] is True and cmeta["first_token"]["layers"] == [8]


@pytest.mark.parametrize("mode", ["decode", "uniform"])
@torch.inference_mode()
def test_batch_runs_in_bf16(model: KotodamaModel, mode: str) -> None:
    """The serving dtype: bf16 captures and readouts reach the numpy arrays."""
    import copy

    sweep.TOKENIZER[0] = _Tok()
    m = copy.deepcopy(model).bfloat16()
    eng = DecodeEngine(m, max_seq_len=96, dtype=torch.bfloat16, prefill_backends=[SDPBackend.MATH], levers=ROWS)
    cap = [7, 8] if mode == "uniform" else [8]
    dec = CascadeDecoder(eng, ROWS, groups=2, group_rows=4, tail_tokens=40, readout_k=3, capture_layers=cap)
    units = sweep.Units(m, 7, list(range(4)), random_frac=0.5, seed=0, p50=3.0)
    ids = torch.randint(4, CFG["vocab_size"], (1, 9), generator=torch.Generator().manual_seed(5))
    cont = torch.randint(4, CFG["vocab_size"], (1, 4), generator=torch.Generator().manual_seed(6))
    batch = [{"kind": "neuron", "neuron": n, "prompt": "p", "dose": 2.0} for n in range(3)]
    out = sweep.run_batch(_args(mode), eng, dec, units, batch, ids, ids.shape[1], cont,
                          SamplingParams(temperature=0.0, repetition_penalty=1.0), frozenset(), [7, 8], [8], cap)
    key = "mlp_in" if mode == "uniform" else "mlp_in_first"
    assert out["arrays"][key].dtype == np.float16 and np.isfinite(out["arrays"][key]).all()
