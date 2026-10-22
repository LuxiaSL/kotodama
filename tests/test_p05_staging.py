"""Staging guards: the p05-armd-v3 run config + the gate-sweep runner.

The config tests pin the facts the run depends on (DD-v1 from the
preset table, the v3 EOS contract, the 6-rank batch geometry that train.py
would otherwise floor silently, the checkpoint cadence). The sweep tests run
scripts/eval/nca_gates_sweep.py end to end on CPU over a tiny real model
saved in train.py's checkpoint format (one plain, one .zst) + a real v3 eval set.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from kotodama import presets
from kotodama.data.dataset import TokenizedDataset
from kotodama.nca import gen_dyadic_data
from kotodama.training import train as train_mod

REPO = Path(__file__).resolve().parents[1]
CONFIG = "configs/p05-armd-v3.yaml"
P05_RANKS = 6


def _sweep_module():
    spec = importlib.util.spec_from_file_location(
        "nca_gates_sweep", REPO / "scripts" / "eval" / "nca_gates_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def p05_args(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(sys, "argv", ["train", "--config", CONFIG])
    return train_mod.parse_args()


# ── config ────────────────────────────────────────────────────────────────────

def test_p05_config_shape_and_boundaries(p05_args):
    assert p05_args.model_size == "proxy"
    assert p05_args.attn_res is True
    got = tuple(int(x) for x in p05_args.attn_res_boundaries.split(","))
    assert got == presets.DDV1 == presets.DEFAULT_BOUNDARIES["proxy"]
    assert p05_args.seed == 42
    assert p05_args.total_tokens == 750_000_000


def test_p05_eos_contract(p05_args):
    # v3 writes EOS=0 between episodes; the dataset splits docs on its
    # default eos_id (train.py never passes one) -> they must agree.
    assert p05_args.doc_masking is True
    default_eos = inspect.signature(TokenizedDataset).parameters["eos_id"].default
    assert gen_dyadic_data.EOS_ID == default_eos == 0
    assert p05_args.data_path.endswith("nca_v3.bin")  # the --finalize output


def test_p05_batch_geometry_exact_on_six_ranks(p05_args):
    per_micro = p05_args.micro_batch_size * p05_args.sequence_length * P05_RANKS
    # train.py: grad_accum = max(1, global // per_micro) — must not floor
    assert p05_args.global_batch_tokens % per_micro == 0
    grad_accum = max(1, p05_args.global_batch_tokens // per_micro)
    assert grad_accum * per_micro == p05_args.global_batch_tokens


def test_p05_checkpoint_cadence(p05_args):
    tokens_per_step = p05_args.global_batch_tokens
    total_steps = p05_args.total_tokens // tokens_per_step
    periodic = [s for s in range(total_steps)
                if s > 0 and s % p05_args.save_every == 0]
    assert (total_steps - 1) not in periodic  # final save never collides
    n_points = len(periodic) + 1
    assert n_points >= 15 and n_points <= p05_args.keep_checkpoints
    assert abs(p05_args.save_every * tokens_per_step - 50e6) < 2e6


# ── sweep ─────────────────────────────────────────────────────────────────────

def test_list_checkpoints_prefers_plain(tmp_path):
    sweep = _sweep_module()
    for name in ("step_00000010.pt", "step_00000010.pt.zst",
                 "step_00000020.pt.zst", "step_00000005.pt", "other.pt",
                 "step_00000030.tmp"):
        (tmp_path / name).write_bytes(b"")
    got = [(s, p.name) for s, p in sweep.list_checkpoints(tmp_path)]
    assert got == [(5, "step_00000005.pt"), (10, "step_00000010.pt"),
                   (20, "step_00000020.pt.zst")]


def test_parse_boundaries():
    sweep = _sweep_module()
    assert sweep.parse_boundaries(None, "proxy") == list(presets.DDV1)
    assert sweep.parse_boundaries("dd3b", "proxy") == list(presets.DD3B)
    assert sweep.parse_boundaries("off", "proxy") is None
    assert sweep.parse_boundaries("0,4", "proxy") == [0, 4]
    with pytest.raises(SystemExit):
        sweep.parse_boundaries(None, "not-a-preset")


def _stub_result(step, tokens, acc_u, acc_o, init=False):
    def led(acc):
        return {"aggregate": {"all": {"acc": acc, "med_rank": 5,
                                      "mean_gold_lp": -2.0, "n": 400},
                              "chance_acc": 0.1}}
    return {"step": step, "tokens": tokens,
            "checkpoint": "init(seed=42)" if init else f"step_{step}.pt",
            "ledger": {"eval_ledger_unique.jsonl": led(acc_u),
                       "eval_ledger_overwrite.jsonl": led(acc_o)},
            "smoothing": {"aggregate": {"si_mean": 0.6, "si_se": 0.01, "n": 200}}}


def test_summary_table_verdict_excludes_init():
    sweep = _sweep_module()
    rows = sweep.summarize([
        _stub_result(340, 50_135_040, 0.07, 0.06),
        _stub_result(0, 0, 0.30, 0.30, init=True),   # init must not count toward the above-chance verdict
        _stub_result(680, 100_270_080, 0.09, 0.12),
    ])
    assert [r["step"] for r in rows] == [0, 340, 680]
    assert rows[0]["is_init"] and rows[1]["ledger_unique_acc"] == 0.07
    table = sweep.format_table(rows, 0.1)
    assert "ledger_unique_acc: max 0.090 @ step 680" in table
    assert "(L-P1: NOT above)" in table
    assert "ledger_overwrite_acc: max 0.120 @ step 680" in table
    assert "(L-P1: ABOVE)" in table


def test_sweep_end_to_end_cpu(tmp_path, monkeypatch):
    """Tiny real model -> train.py-format checkpoints (plain + .zst) ->
    real v3 eval set -> sweep main() -> per-step JSONs + summary."""
    from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig
    from kotodama.nca.dyadic import (DyadicConfig, N_VALS, emit_ledger_items,
                                     token_map)
    from kotodama.nca.generator import NCAConfig, NCARule, sample_rule_config
    from kotodama.training.checkpoint import CheckpointManager
    from kotodama.training.muon import HybridScheduler, build_hybrid_optimizer

    vocab = token_map(NCAConfig(reserve_eos=True))["VAL0"] + N_VALS + 8
    shape = dict(hidden_size=64, num_layers=4, num_attention_heads=2,
                 num_kv_heads=1, head_dim=32, intermediate_size=128,
                 vocab_size=vocab, max_position_embeddings=4096)
    (tmp_path / "model.yaml").write_text(json.dumps({"tiny": shape}))

    # eval set exactly as --finalize lays it out
    ev = tmp_path / "data"
    ev.mkdir()
    cfg = DyadicConfig(nca=NCAConfig(grid_size=8, n_groups=2, num_steps=24,
                                     burn_in=2, filter_enabled=False,
                                     reserve_eos=True),
                       commits_per_traj=(4, 6), queries_per_traj=(3, 5),
                       p_tag_reuse=0.0, min_query_lag_steps=3,
                       turn_len_min=2, turn_len_max=4)

    def mk(seed):
        torch.manual_seed(seed)
        rc = sample_rule_config(cfg.nca, rng=random.Random(seed))
        return NCARule(d_state=10, n_groups=2, kernel_size=rc["kernel_size"],
                       hidden_dim=rc["hidden_dim"],
                       num_hidden_layers=rc["num_hidden_layers"])
    items = emit_ledger_items(cfg, [(mk(1), mk(2))], 4, seed0=10)
    (ev / "eval_ledger_unique.jsonl").write_text(
        "\n".join(json.dumps(x) for x in items))
    (ev / "meta.json").write_text(json.dumps({"reserve_eos": True}))

    # two checkpoints in train.py's format
    boundaries = [0, 2]
    torch.manual_seed(0)
    model = LuxiaBaseModel(LuxiaModelConfig(
        **shape, qk_norm=True, attn_res=True, attn_res_boundaries=boundaries,
        attn_impl="sdpa"))
    muon, adamw = build_hybrid_optimizer(model)
    sched = HybridScheduler(muon, adamw, warmup_steps=1, total_steps=10)
    ck = tmp_path / "ckpt"
    mgr = CheckpointManager(ck, keep_last_n=10)
    mgr.save(step=3, model=model, muon_opt=muon, adamw_opt=adamw,
             scheduler=sched, tokens_consumed=12_288)
    mgr.save(step=7, model=model, muon_opt=muon, adamw_opt=adamw,
             scheduler=sched, tokens_consumed=24_576)
    has_zstd = shutil.which("zstd") is not None
    if has_zstd:
        subprocess.run(["zstd", "-q", "--rm", str(ck / "step_00000007.pt")],
                       check=True)

    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setattr(sys, "argv", [
        "sweep", "--ckpt-dir", str(ck), "--eval-dir", str(ev),
        "--config-path", str(tmp_path / "model.yaml"), "--config-section", "tiny",
        "--boundaries", "0,2", "--init-seed", "42", "--device", "cpu"])
    sweep = _sweep_module()
    assert sweep.main() == 0

    out = ck / "gates"
    per_step = {json.loads(p.read_text())["step"]: json.loads(p.read_text())
                for p in out.glob("gates_step*.json")}
    assert sorted(per_step) == [0, 3, 7]
    assert per_step[0]["tokens"] == 0
    assert per_step[3]["tokens"] == 12_288 and per_step[7]["tokens"] == 24_576
    for r in per_step.values():
        agg = r["ledger"]["eval_ledger_unique.jsonl"]["aggregate"]["all"]
        assert agg["n"] == 4 and 0.0 <= agg["acc"] <= 1.0
    summary = json.loads((out / "summary.json").read_text())
    assert [r["step"] for r in summary["rows"]] == [0, 3, 7]
    assert summary["chance_acc"] == pytest.approx(1 / N_VALS)
    if has_zstd:  # the decompressed copy is cleaned out of TMPDIR
        assert not list((tmp_path / "tmp" / "kotodama_checkpoints").glob("*.pt"))
