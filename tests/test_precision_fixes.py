"""Precision fixes from the 2026-09-28 bf16-class audit.

1. Newton-Schulz must be scale-invariant: tiny momentum (~1e-7) was cast to fp16 before
   normalization -> subnormals / flush-to-zero -> a distorted orthogonalized update.
2. The sharpness probe must leave the weights bit-identical (it used to +e/-2e/+e in
   place on bf16 weights, leaving a residual on rank 0 only -> DDP rank divergence).
"""
from __future__ import annotations

import torch

from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig
from kotodama.monitoring.geometric import GeometricMonitor, MonitorConfig
from kotodama.training.muon import _batched_newton_schulz, newton_schulz_orthogonalize


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


def test_newton_schulz_is_scale_invariant_down_to_tiny_momentum():
    torch.manual_seed(0)
    M = torch.randn(64, 160)
    ref = newton_schulz_orthogonalize(M)
    for scale in (1e-5, 1e-7, 1e-8):
        assert _cos(newton_schulz_orthogonalize(M * scale), ref) > 0.99999, scale
    B = torch.randn(3, 64, 160)
    refb = _batched_newton_schulz(B)
    assert _cos(_batched_newton_schulz(B * 1e-8), refb) > 0.99999


def test_sharpness_probe_restores_weights_exactly():
    torch.manual_seed(0)
    model = LuxiaBaseModel(LuxiaModelConfig(
        hidden_size=64, num_layers=2, num_attention_heads=2, num_kv_heads=1, head_dim=32,
        intermediate_size=128, vocab_size=256, max_position_embeddings=128, attn_impl="sdpa",
    )).to(torch.bfloat16)
    ids = torch.randint(1, 256, (4, 32))
    model(ids, labels=ids)["loss"].backward()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    mon = GeometricMonitor(model, MonitorConfig(tier1_sample_layers=[0], tier2_twonn_layers=[0]))
    mon.set_probe_batch(ids)
    m = mon.compute_sharpness(step=0, epsilon=0.1, n_sequences=4)
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]), f"{n} changed by the sharpness probe"
    assert "eos/sharpness" in m and 0.5 < m["eos/eps_landed_frac"] < 1.05  # nearest rounding can overshoot
