"""kotodama's capture into anamnesis's on-device reduction (replay_features_gpu).

Gates: cached_forward's attention sink and capture_layers change nothing they are
not asked to (logits, routing, committed blocks and residuals are the default
capture's bytes; the sink receives exactly the weights the default capture keeps;
only the named layers keep projections). And the on-device lane agrees with the
CPU specification, replay_extract_koto then anamnesis's canonical pipeline, on the
same looped AttnRes model and span: the same names in the same order, and the same
values to the arithmetic difference between the two implementations. The anamnesis
tests skip when anamnesis (with its external-capture readout) is not importable.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import pytest
import torch

from kotodama.model.capture import cached_forward
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

TINY = dict(hidden_size=64, num_layers=9, num_attention_heads=4, num_kv_heads=2, head_dim=16,
            intermediate_size=128, vocab_size=199, max_position_embeddings=96, attn_impl="sdpa",
            attn_res=True, attn_res_boundaries=[0, 1, 3, 5, 7], use_liger=False,
            loop_start=1, loop_len=2, loop_times=3)   # 1 prelude + 2x3 core + 2 coda = 9 effective
SAMPLED = [0, 2, 4, 6, 8]
PCA_LAYERS = [2, 4, 6]


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    torch.manual_seed(20261026)
    return KotodamaModel(KotodamaModelConfig(**TINY)).eval()


@pytest.fixture(scope="module")
def ids() -> torch.Tensor:
    torch.manual_seed(5)
    return torch.randint(3, 199, (1, 40), dtype=torch.long)


@torch.no_grad()
def test_the_sink_and_capture_layers_change_nothing_else(model, ids):
    logits, cap, _ = cached_forward(model, ids, light=False, capture_hidden=True)
    seen = {}
    logits2, cap2, _ = cached_forward(model, ids, light=False, capture_hidden=True,
                                      attention_sink=lambda layer, w: seen.__setitem__(layer, w),
                                      capture_layers={2, 6})
    assert torch.equal(logits, logits2)
    for key in ("residual_partial", "committed"):
        assert all(torch.equal(a, b) for a, b in zip(cap[key], cap2[key]))
    assert [(t, l) for t, l, _ in cap["routing"]] == [(t, l) for t, l, _ in cap2["routing"]]
    assert all(torch.equal(a[2], b[2]) for a, b in zip(cap["routing"], cap2["routing"]))
    assert sorted(seen) == list(range(TINY["num_layers"]))
    assert all(torch.equal(seen[layer], cap["attn"][layer]) for layer in seen)
    assert all(a is None for a in cap2["attn"])
    for key in ("keys", "values", "queries", "gate", "attn_out"):
        kept = [layer for layer, x in enumerate(cap2[key]) if x is not None]
        assert kept == [2, 6], key
        assert all(torch.equal(cap[key][layer], cap2[key][layer]) for layer in kept)


def _lane_and_spec(model, ids, prompt_length):
    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
    from anamnesis.extraction.fast.features import GpuFeatureLane
    from anamnesis.extraction.fast.schema import resolve_gpu_schema

    L, D = TINY["num_layers"], TINY["hidden_size"]
    extraction = ExtractionConfig(sampled_layers=SAMPLED, pca_layers=PCA_LAYERS, pca_components=3,
                                  early_layer_cutoff=2, late_layer_cutoff=6)
    battery = dict(include_core_blocks=True, enable_residual_trajectory=True, trajectory_layers=PCA_LAYERS,
                   enable_attention_flow=True, enable_gate_features=True, enable_per_head=True,
                   enable_value_geometry=True, enable_qk_geometry=True, enable_kv_cka=True,
                   contrastive_layers=PCA_LAYERS)
    families = FeaturePipelineConfig(**battery)
    rng = np.random.default_rng(11)
    components = {layer: np.linalg.qr(rng.standard_normal((D, D)))[0][:3].astype(np.float32)
                  for layer in PCA_LAYERS}
    means = {layer: rng.standard_normal(D).astype(np.float32) * 0.1 for layer in PCA_LAYERS}
    pm = (rng.standard_normal((L + 1, ids.shape[1] + 8, D)) * 0.1).astype(np.float32)
    steps = ids.shape[1] - prompt_length - 1
    schema = resolve_gpu_schema(L, steps, extraction, families, components)
    lane = GpuFeatureLane(extraction, families, list(schema.feature_names), pm, components, means,
                          device="cpu", calibration_sha256="a" * 64, replay_path="full")
    spec_families = FeaturePipelineConfig(**battery, enable_attn_res=True)
    return lane, extraction, spec_families, pm, components, means


@torch.no_grad()
def test_the_gpu_lane_agrees_with_the_cpu_specification(model, ids):
    pytest.importorskip("anamnesis.extraction.fast.external")
    from anamnesis.extraction.feature_pipeline import compute_features_with_families_from_data
    from kotodama.model.capture import replay_extract_koto, replay_features_gpu

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        P = 14
        lane, extraction, spec_families, pm, components, means = _lane_and_spec(model, ids, P)
        lane_result = replay_features_gpu(model, ids[0], P, lane)
        raw = replay_extract_koto(model, ids[0], prompt_length=P, positional_means=pm)
        spec = compute_features_with_families_from_data(raw, extraction, spec_families, components, means)
    finally:
        torch.use_deterministic_algorithms(previous)
    assert list(lane_result.feature_names) == list(spec.feature_names)
    a = np.asarray(lane_result.features, np.float64)
    b = np.asarray(spec.features, np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    worst = np.argsort(-np.abs(a - b))[:5]
    report = [(spec.feature_names[i], float(a[i]), float(b[i])) for i in worst]
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=2e-6, err_msg=str(report))
    assert lane_result.metadata["routing_family"] is True
    assert lane_result.metadata["capture"]["producer"] == "kotodama.model.capture"
