"""Handoff-factorial flags: --reset_routing_layers (spec parse + reset to fresh init,
others untouched) and --stop_at_step (loop end without touching the schedule)."""

from __future__ import annotations

import pytest
import torch

from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig
from kotodama.training.train import _loop_end_step, parse_routing_spec

N_LAYERS = 6


def _model(seed: int) -> LuxiaBaseModel:
    torch.manual_seed(seed)
    cfg = LuxiaModelConfig(
        hidden_size=64, num_layers=N_LAYERS, num_attention_heads=4, num_kv_heads=2, head_dim=16,
        intermediate_size=128, vocab_size=199, max_position_embeddings=96, attn_impl="sdpa",
        attn_res=True, attn_res_boundaries=[0, 1, 3], use_liger=False,
    )
    return LuxiaBaseModel(cfg)


def test_parse_routing_spec():
    assert parse_routing_spec("24-27,final", 28) == ([24, 25, 26, 27], True)
    assert parse_routing_spec(" 3, 1-2 ,3", 6) == ([1, 2, 3], False)
    assert parse_routing_spec("final", 6) == ([], True)
    for bad in ("", ",", "5-3", "x", "1-", "6", "0-6", "-1"):
        with pytest.raises(ValueError):
            parse_routing_spec(bad, 6)


def test_reset_routing_restores_fresh_init_and_leaves_rest():
    fresh = _model(0).state_dict()
    m = _model(1)
    with torch.no_grad():  # stand-in for an NCA-trained routing imprint
        for p in m.parameters():
            p.add_(torch.randn_like(p))
    before = {k: v.clone() for k, v in m.state_dict().items()}

    names = m.reset_routing(*parse_routing_spec("4-5,final", N_LAYERS))
    after = m.state_dict()

    expect = {f"layers.{i}.{s}_res_{t}" for i in (4, 5) for s in ("attn", "mlp")
              for t in ("query", "norm.weight")} | {"final_res_query", "final_res_norm.weight"}
    assert set(names) == expect
    for k in after:
        if k in expect:
            assert torch.equal(after[k], fresh[k]), k  # zeros / ones, as at init
        else:
            assert torch.equal(after[k], before[k]), k  # untouched (incl. layers 0-3 routing)


def test_reset_routing_rejects_bad_input():
    m = _model(0)
    with pytest.raises(ValueError):
        m.reset_routing([N_LAYERS])
    no_res = LuxiaBaseModel(LuxiaModelConfig(
        hidden_size=64, num_layers=2, num_attention_heads=4, num_kv_heads=2, head_dim=16,
        intermediate_size=128, vocab_size=199, max_position_embeddings=96, attn_impl="sdpa",
        attn_res=False, use_liger=False,
    ))
    with pytest.raises(ValueError):
        no_res.reset_routing([0])


def test_loop_end_step():
    assert _loop_end_step(14305, None) == 14305
    assert _loop_end_step(14305, 600) == 600
    assert _loop_end_step(100, 600) == 100  # never past the schedule
    with pytest.raises(ValueError):
        _loop_end_step(100, 0)
