"""Tests for kotodama.eval.nca_gates (the P0.6 in-domain gate scorer).

Stub models pin the scoring math exactly (rank/acc accounting, SI direction);
one integration test runs a real tiny LuxiaBaseModel over items emitted by
the real v3 emitters, CPU-only.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from kotodama.eval.nca_gates import (aggregate_ledger, aggregate_smoothing,
                                     resolve_token_map, score_ledger,
                                     score_smoothing)
from kotodama.nca.dyadic import (DyadicConfig, N_VALS, emit_ledger_items,
                                 emit_smoothing_pairs, token_map)
from kotodama.nca.generator import NCAConfig, NCARule, sample_rule_config

V3_TM = token_map(NCAConfig(reserve_eos=True))
VOCAB = V3_TM["VAL0"] + N_VALS + 8


class PeakModel:
    """Stub with the LuxiaBaseModel contract: every position's logits put a
    +10 peak on ``peak_id``, flat elsewhere."""

    def __init__(self, peak_id: int):
        self.peak_id = peak_id

    def __call__(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        B, S = ids.shape
        logits = torch.zeros(B, S, VOCAB)
        logits[..., self.peak_id] = 10.0
        return {"logits": logits}


def _item(gold: int, **labels) -> dict:
    return {"item_id": f"t{gold}", "prefix": [1, 2, 3],
            "gold_val": gold, "candidates": list(range(N_VALS)), **labels}


def test_score_ledger_rank_and_acc():
    val0 = V3_TM["VAL0"]
    items = [_item(3, committer_side=0, lag_steps=2, is_requery=False,
                   post_overwrite=False),
             _item(4, committer_side=1, lag_steps=12, is_requery=True,
                   post_overwrite=True)]
    right = score_ledger(PeakModel(val0 + 3), items, V3_TM)
    assert right[0]["acc"] == 1 and right[0]["rank"] == 1
    assert right[1]["acc"] == 0 and right[1]["rank"] == 2  # one cand above
    wrong = score_ledger(PeakModel(val0 + 9), items, V3_TM)
    assert [r["acc"] for r in wrong] == [0, 0]
    # peak OUTSIDE the VAL alphabet: candidates tie, gold rank 1 by <=-tie rule
    off = score_ledger(PeakModel(0), items, V3_TM)
    assert all(r["rank"] == 1 for r in off)


def test_aggregate_ledger_splits():
    val0 = V3_TM["VAL0"]
    items = [_item(1, committer_side=0, lag_steps=2, is_requery=False,
                   post_overwrite=False),
             _item(1, committer_side=1, lag_steps=20, is_requery=True,
                   post_overwrite=True),
             _item(2, committer_side=1, lag_steps=5, is_requery=False,
                   post_overwrite=False)]
    agg = aggregate_ledger(score_ledger(PeakModel(val0 + 1), items, V3_TM))
    assert agg["all"]["n"] == 3 and agg["all"]["acc"] == pytest.approx(2 / 3)
    assert agg["self_committed"]["n"] == 1
    assert agg["other_committed"]["n"] == 2
    assert agg["lag_le8"]["n"] == 2 and agg["lag_gt8"]["n"] == 1
    assert agg["first_query"]["n"] == 2 and agg["re_query"]["n"] == 1
    assert agg["unique_write"]["n"] == 2 and agg["post_overwrite"]["n"] == 1
    assert agg["chance_acc"] == pytest.approx(1 / N_VALS)


def test_score_smoothing_direction():
    # model prefers token 7 everywhere -> the frame made of 7s wins
    pair = {"item_id": "s0", "prefix": [1, 2, 3],
            "true_frame": [7, 7, 7, 7],
            "counterfactual_frame": [8, 8, 8, 8]}
    rows = score_smoothing(PeakModel(7), [pair])
    assert rows[0]["lp_true"] > rows[0]["lp_counterfactual"]
    assert rows[0]["si"] < 0.5  # low SI = responds to the event
    rows = score_smoothing(PeakModel(8), [pair])
    assert rows[0]["si"] > 0.5  # high SI = rides over it
    agg = aggregate_smoothing(rows)
    assert agg["n"] == 1 and 0.5 < agg["si_mean"] <= 1.0


def test_resolve_token_map(tmp_path):
    # no meta.json -> legacy v1 layout
    assert resolve_token_map(tmp_path)["VAL0"] == 10023
    (tmp_path / "meta.json").write_text('{"reserve_eos": true}')
    assert resolve_token_map(tmp_path)["VAL0"] == 10024
    assert resolve_token_map(tmp_path, "false")["VAL0"] == 10023


def test_integration_real_model_real_items():
    from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig
    torch.manual_seed(0)
    model = LuxiaBaseModel(LuxiaModelConfig(
        hidden_size=64, num_layers=2, num_attention_heads=2, num_kv_heads=1,
        head_dim=32, intermediate_size=128, vocab_size=VOCAB,
        max_position_embeddings=4096, attn_res=False,
        attn_impl="sdpa")).eval()  # "auto" picks flash-attn: no CPU kernel

    cfg = DyadicConfig(
        nca=NCAConfig(grid_size=8, n_groups=2, num_steps=24, burn_in=2,
                      filter_enabled=False, reserve_eos=True),
        commits_per_traj=(4, 6), queries_per_traj=(3, 5),
        p_tag_reuse=0.5, min_query_lag_steps=3, turn_len_min=2,
        turn_len_max=4)

    def mk(seed):
        torch.manual_seed(seed)
        rc = sample_rule_config(cfg.nca, rng=random.Random(seed))
        return NCARule(d_state=10, n_groups=2,
                       kernel_size=rc["kernel_size"],
                       hidden_dim=rc["hidden_dim"],
                       num_hidden_layers=rc["num_hidden_layers"])

    pairs = [(mk(1), mk(2))]
    led = emit_ledger_items(cfg, pairs, 4, seed0=10)
    smo = emit_smoothing_pairs(cfg, pairs, 2, seed0=40)

    rows = score_ledger(model, led, V3_TM)
    agg = aggregate_ledger(rows)
    assert agg["all"]["n"] == 4
    assert 0.0 <= agg["all"]["acc"] <= 1.0
    assert all(1 <= r["rank"] <= N_VALS for r in rows)
    assert all(np.isfinite(r["gold_lp"]) for r in rows)
    srows = score_smoothing(model, smo)
    assert len(srows) == 2 and all(0.0 <= r["si"] <= 1.0 for r in srows)


def test_model_overrides_cpu_and_rope():
    from kotodama.eval.nca_gates import model_overrides
    assert model_overrides("cpu") == {"attn_impl": "sdpa", "use_liger": False}
    assert model_overrides("cuda") == {}
    assert model_overrides("cuda:0", rope_fp32=True) == {"rope_fp32": True}
    assert model_overrides("cpu", rope_fp32=True)["rope_fp32"] is True
