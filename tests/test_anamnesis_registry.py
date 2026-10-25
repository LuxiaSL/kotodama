"""kotodama's anamnesis registry file: generated from the tables, and read by anamnesis as installed.

Gates: the tracked file is exactly what the tables generate; the layer rule reproduces the
kotodama-3b set at depth 28 and anamnesis's 70B row at depth 80; every loop twin runs its
partner's weights; and, when anamnesis is importable, anamnesis reads the file, accepts its layer
plan, and computes a finite feature vector with the attn_res family from a looped AttnRes model
through ``kotodama.model.capture`` (the instrument as installed, not as it was when this was written).
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np
import pytest
import torch

from kotodama import anamnesis_registry as reg
from kotodama import presets
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig


def test_the_tracked_file_is_what_the_tables_generate():
    if os.environ.get(reg.WRITE_ENV) == "1":
        reg.write()
    assert reg.REGISTRY_FILE.read_text() == reg.render(), (
        f"{reg.REGISTRY_FILE.name} differs from the tables; regenerate with {reg.WRITE_ENV}=1")


def test_the_rule_reproduces_kotodama_3b():
    plan = reg.layer_plan("3b")
    assert plan["sampled_layers"] == [0, 7, 14, 18, 21, 24, 27]
    assert plan["pca_layers"] == [7, 14, 18, 21, 24]
    assert (plan["early_layer_cutoff"], plan["late_layer_cutoff"]) == (7, 21)
    assert plan["loop_twin_layers"] == []


def test_the_rule_reproduces_the_70b_row():
    assert [reg.rescale_layer(layer, 80) for layer in (0, 8, 16, 20, 24, 28, 31)] == [0, 20, 40, 50, 60, 70, 79]


def test_tsumugi_layer_plan():
    plan = reg.layer_plan("1b-moonshot-loop")
    assert plan["rule_layers"] == [0, 18, 36, 45, 54, 63, 71]
    assert plan["loop_twin_layers"] == [25, 34, 56]
    assert plan["sampled_layers"] == [0, 18, 25, 34, 36, 45, 54, 56, 63, 71]
    assert plan["pca_layers"] == plan["trajectory_layers"] == plan["contrastive_layers"] == [18, 36, 45, 54, 63]


def test_every_loop_twin_runs_its_partners_weights():
    config = KotodamaModelConfig(**presets.shape("1b-moonshot-loop"))
    plan = reg.layer_plan("1b-moonshot-loop")
    for layer in plan["rule_layers"]:
        for twin in reg.loop_twins(config, layer):
            assert config.layer_source(twin) == config.layer_source(layer)
            assert twin in plan["sampled_layers"]
    assert reg.loop_twins(config, 0) == [] and reg.loop_twins(config, 71) == []


def test_rows_restate_no_table():
    row = reg.preset_row("kotodama-tsumugi-1b")
    shape = presets.shape("1b-moonshot-loop")
    assert (row["num_layers"], row["hidden_dim"], row["num_attention_heads"], row["num_kv_heads"],
            row["head_dim"]) == (shape["num_layers"], shape["hidden_size"], shape["num_attention_heads"],
                                 shape["num_kv_heads"], shape["head_dim"])
    assert json.loads(reg.render())["presets"]["kotodama-tsumugi-1b"] == row


# ── against the installed anamnesis ───────────────────────────────────────────

def test_anamnesis_reads_the_file():
    models = pytest.importorskip("anamnesis.config.models")
    from anamnesis.config import ExtractionConfig

    registry = models.load_registry((*models.registry_paths(), reg.REGISTRY_FILE))
    row = registry.presets["kotodama-tsumugi-1b"]
    assert tuple(row.sampled_layers) == tuple(reg.layer_plan("1b-moonshot-loop")["sampled_layers"])
    config = ExtractionConfig(sampled_layers=list(row.sampled_layers), pca_layers=list(row.pca_layers),
                              early_layer_cutoff=row.early_layer_cutoff, late_layer_cutoff=row.late_layer_cutoff)
    config.check_depth(row)


TINY = dict(hidden_size=64, num_layers=9, num_attention_heads=4, num_kv_heads=2, head_dim=16,
            intermediate_size=128, vocab_size=199, max_position_embeddings=96, attn_impl="sdpa",
            attn_res=True, attn_res_boundaries=[0, 1, 3, 5, 7], use_liger=False,
            loop_start=1, loop_len=2, loop_times=3)   # 1 prelude + 2x3 core + 2 coda = 9 effective


@torch.no_grad()
def test_a_looped_attnres_model_reaches_an_anamnesis_vector():
    pytest.importorskip("anamnesis.extraction.feature_pipeline")
    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig
    from anamnesis.extraction.feature_pipeline import compute_features_with_families_from_data
    from kotodama.model.capture import replay_extract_koto

    torch.manual_seed(20261024)
    model = KotodamaModel(KotodamaModelConfig(**TINY)).eval()
    ids = torch.randint(3, 199, (40,), dtype=torch.long)
    sampled = [0, 2, 4, 6, 8]                       # 2, 4, 6 run the same core weights on three passes
    raw = replay_extract_koto(model, ids, prompt_length=16, positional_means=None)
    ext = ExtractionConfig(sampled_layers=sampled, pca_layers=[2, 4, 6], early_layer_cutoff=2,
                           late_layer_cutoff=6, enable_residual_pca=False, enable_knnlm_baseline=False)
    fam = FeaturePipelineConfig(trajectory_layers=[2, 4, 6], contrastive_layers=[2, 4, 6],
                                enable_residual_trajectory=True, enable_attention_flow=True,
                                enable_gate_features=True, enable_temporal_dynamics=True, enable_per_head=True,
                                enable_value_geometry=True, enable_qk_geometry=True, enable_kv_cka=True,
                                enable_attn_res=True)
    first = compute_features_with_families_from_data(raw, ext, fam, None, None)
    v = np.asarray(first.features, np.float32)
    names = list(first.feature_names)
    assert len(v) == len(names) and len(names) == len(set(names))
    assert np.isfinite(v).all()
    assert any(n.startswith("attnres_L2_") for n in names)
    assert "attnres_committed_cos_b0b1_mean" in names
    again = compute_features_with_families_from_data(
        replay_extract_koto(model, ids, prompt_length=16, positional_means=None), ext, fam, None, None)
    assert np.asarray(again.features, np.float32).tobytes() == v.tobytes()
