"""The geometric monitor's dead-neuron ruler must be able to see a dead neuron.

`dead_mlp_units` hooks the SwiGLU output feeding down_proj: the residual stream is never
dead, so a ruler measured there reads exactly 0.0 for a whole run.
"""
from __future__ import annotations

import pytest
import torch

from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.monitoring.geometric import GeometricMonitor, MonitorConfig

INTER = 128


@pytest.mark.parametrize("attn_res", [False, True])
def test_dead_mlp_units_sees_killed_neurons(attn_res: bool):
    torch.manual_seed(0)
    kw = dict(hidden_size=64, num_layers=2, num_attention_heads=2, num_kv_heads=1, head_dim=32,
              intermediate_size=INTER, vocab_size=256, max_position_embeddings=128, attn_impl="sdpa")
    if attn_res:
        kw.update(attn_res=True, attn_res_boundaries=[0, 1])
    model = KotodamaModel(KotodamaModelConfig(**kw))
    killed = [3, 17, 90]
    with torch.no_grad():
        for j in killed:
            model.layers[1].ffn.gate_proj.weight[j].zero_()
            model.layers[1].ffn.up_proj.weight[j].zero_()
    mon = GeometricMonitor(model, MonitorConfig(tier1_sample_layers=[0, 1], tier2_twonn_layers=[0]))
    ids = torch.randint(1, 256, (2, 32))
    mon._probe_forward(ids)
    m1 = mon._mlp_unit_metrics(1, "geo/layer_1")
    m0 = mon._mlp_unit_metrics(0, "geo/layer_0")
    assert m1["geo/layer_1/dead_mlp_units"] == pytest.approx(len(killed) / INTER)
    assert m0["geo/layer_0/dead_mlp_units"] == 0.0
    assert m1["geo/layer_1/weak_mlp_units"] >= len(killed) / INTER
    # hooks are removed after the probe
    assert not model.layers[1].ffn.down_proj._forward_pre_hooks
