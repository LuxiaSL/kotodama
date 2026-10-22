"""Training telemetry additions: per-layer weight scale / grad norms / AttnRes router
health (dead-router detection), and the hyperparameter post's secret filter."""

from __future__ import annotations

import torch

from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.training import train as T


def _model() -> KotodamaModel:
    torch.manual_seed(3)
    cfg = KotodamaModelConfig(
        hidden_size=64, num_layers=4, num_attention_heads=4, num_kv_heads=2, head_dim=16,
        intermediate_size=128, vocab_size=99, max_position_embeddings=32, attn_impl="sdpa",
        attn_res=True, attn_res_boundaries=[0, 1, 3], use_liger=False,
    )
    m = KotodamaModel(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "res_query" in n:
                p.normal_(0, 0.3)
    return m


def _backward(m: KotodamaModel) -> None:
    m.train()
    ids = torch.randint(3, 99, (2, 16))
    out = m(input_ids=ids, labels=ids)
    out["loss"].backward()


def test_layer_health_keys_and_values():
    m = _model()
    _backward(m)
    h = T._layer_health_metrics(m)
    # weight scale, every projection of every layer
    for i in range(4):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            w = dict(m.named_parameters())[f"layers.{i}.{'attn' if proj[0] in 'qkvo' else 'ffn'}.{proj}.weight"]
            assert abs(h[f"weight_rms/layer_{i}/{proj}"] - w.detach().pow(2).mean().sqrt().item()) < 1e-6
        assert h[f"grad/layer_{i}/attn"] > 0 and h[f"grad/layer_{i}/mlp"] > 0
        assert h[f"router/layer_{i}/mlp/query_norm"] > 0
    assert "weight_rms/embed" in h
    # router grads live everywhere a query routes; layer 0's pre-attn never routes
    assert h["router/layer_2/mlp/grad_norm"] > 0 and h["router/final/grad_norm"] > 0
    assert h["router/dead_sites"] == float(
        sum(1 for k, v in h.items() if k.endswith("/grad_norm") and k.startswith("router/") and v == 0))


def test_dead_router_is_counted():
    m = _model()
    _backward(m)
    base = T._layer_health_metrics(m)["router/dead_sites"]
    m.layers[2].mlp_res_query.grad.zero_()   # the dead-router signature: exactly-zero gradient
    h = T._layer_health_metrics(m)
    assert h["router/layer_2/mlp/grad_norm"] == 0.0
    assert h["router/dead_sites"] == base + 1


def test_no_grad_means_no_grad_keys():
    m = _model()
    h = T._layer_health_metrics(m)
    assert not any(k.startswith("grad/") or k.endswith("/grad_norm") for k in h)
    assert "router/dead_sites" not in h and "weight_rms/layer_0/q_proj" in h


def test_hm_config_drops_secretish_and_nonscalar(monkeypatch):
    sent = {}

    class _C:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json): sent.update(json["config"])

    import httpx
    monkeypatch.setattr(httpx, "Client", _C)
    monkeypatch.setattr(T, "_hm_job_id", "job")
    T.hm_config({"muon_lr": 0.02, "total_tokens": 30, "anchor_tokens": "10B,24B",
                 "wandb_api_key": "x", "hf_token": "x", "token": "x", "auth_header": "x",
                 "seq": [1, 2], "none": None, "flag": True})
    assert sent == {"muon_lr": 0.02, "total_tokens": 30, "anchor_tokens": "10B,24B", "flag": True}


def test_wrapped_model_names_still_resolve():
    """The training model can carry a wrapper prefix (compile / DDP); keys must not depend on it."""
    m = _model()
    wrapper = torch.nn.Module()
    wrapper._orig_mod = m
    h = T._layer_health_metrics(wrapper)
    assert "weight_rms/embed" in h and "weight_rms/layer_3/down_proj" in h
    assert "router/final/query_norm" in h
