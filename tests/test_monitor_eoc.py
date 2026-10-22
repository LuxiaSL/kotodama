"""EoC ruler: converged re-randomized power iteration, scale-free
relative gain, and the MLP sub-layer probed at its own (pre-MLP) routed input."""
from __future__ import annotations

import math

import pytest
import torch

from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig
from kotodama.monitoring.geometric import GeometricMonitor, MonitorConfig


def _tiny(attn_res: bool = True) -> LuxiaBaseModel:
    torch.manual_seed(0)
    kw = dict(hidden_size=64, num_layers=2, num_attention_heads=2, num_kv_heads=1, head_dim=32,
              intermediate_size=128, vocab_size=256, max_position_embeddings=128, attn_impl="sdpa")
    if attn_res:
        kw.update(attn_res=True, attn_res_boundaries=[0, 1])
    return LuxiaBaseModel(LuxiaModelConfig(**kw))


def _mon(model: LuxiaBaseModel) -> GeometricMonitor:
    return GeometricMonitor(model, MonitorConfig(tier1_sample_layers=[0, 1], tier2_twonn_layers=[0]))


def test_rerandomized_power_iteration_converges():
    """A re-randomized call must converge like a cold one (a short 5-iteration restart
    under-reads σ every 10th call: a sawtooth in every EoC series)."""
    mon = _mon(_tiny())
    scale = torch.tensor([2.0, 1.5] + [0.5] * 62)
    fn = lambda h: h * scale  # symmetric J: dominant |eigenvalue| = 2.0
    h = torch.randn(1, 8, 64)
    mon._warm_vectors["k"] = torch.randn(1, 8, 64)  # a warm vector exists ...
    mon._eoc_call_count = 10                          # ... but this call re-randomizes
    sigma, _ = mon._power_iterate(h, fn, "k", torch.device("cpu"), n_iters=5)
    assert sigma == pytest.approx(2.0, rel=5e-3)


def test_relative_gain_is_scale_free():
    """Norm-then-linear (layer 0 on the raw embedding): raw σ ∝ 1/‖h‖, the relative gain
    σ·‖h‖/‖f(h)‖ does not move when the input is rescaled."""
    mon = _mon(_tiny())
    torch.manual_seed(1)
    W = torch.randn(64, 64) / 8
    W = (W + W.T) / 2
    fn = lambda h: (h / h.pow(2).mean(-1, keepdim=True).sqrt()) @ W
    h = torch.randn(1, 8, 64) * 0.03
    s_small, g_small = mon._power_iterate(h, fn, "a", torch.device("cpu"), n_iters=30)
    s_big, g_big = mon._power_iterate(h * 100, fn, "b", torch.device("cpu"), n_iters=30)
    assert s_small / s_big == pytest.approx(100, rel=0.05)
    assert g_small == pytest.approx(g_big, rel=0.05)


def test_mlp_probed_at_its_own_routed_input():
    model = _tiny(attn_res=True).eval()
    mon = _mon(model)
    ids = torch.randint(1, 256, (2, 16))
    hs, _ = mon._probe_forward(ids)
    assert set(mon._mlp_inputs) == {0, 1}
    # layer 0: pre-attn input is the embedding; pre-MLP input includes attn's output
    assert not torch.allclose(mon._mlp_inputs[0], hs[0])
    assert not model.layers[0].ffn_norm._forward_pre_hooks  # hooks removed
    m = mon._eoc_metrics(hs, torch.device("cpu"), n_iters=5)
    for i in (0, 1):
        for sub in ("attn", "mlp"):
            assert math.isfinite(m[f"eoc/jacobian_gain/layer_{i}/{sub}"])
            assert m[f"eoc/jacobian_sigma/layer_{i}/{sub}"] > 0
    assert {"eoc/gain_max", "eoc/gain_mean", "eoc/sigma_max", "eoc/layer0_sigma"} <= set(m)
