"""The eval loader decides AttnRes geometry in one place: stated fields win, then the weights (AttnRes on or off),
then the published config's boundaries, then the preset's trained boundaries; published weights load from a
directory, and a published shape that disagrees with the requested one is refused."""
from __future__ import annotations

import json
import logging

import pytest
import torch
import yaml
from safetensors.torch import save_file

from kotodama import presets
from kotodama.eval.model_loader import attn_res_geometry, load_model
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

TINY = dict(hidden_size=64, num_layers=2, num_attention_heads=2, num_kv_heads=1, head_dim=32,
            intermediate_size=128, vocab_size=256, max_position_embeddings=128, attn_impl="sdpa")
WITH_ATTNRES = {"model.attn_res_query.0": torch.zeros(1)}
PLAIN = {"model.layers.0.weight": torch.zeros(1)}


def test_stated_fields_win():
    out = attn_res_geometry({"attn_res": True, "attn_res_boundaries": [0, 5]}, WITH_ATTNRES,
                            {"attn_res_boundaries": [0, 1]}, "3b")
    assert out == {"attn_res": True, "attn_res_boundaries": [0, 5]}
    assert attn_res_geometry({"attn_res": False}, WITH_ATTNRES, {}, "3b") == {"attn_res": False}


def test_attn_res_read_from_the_weights():
    assert attn_res_geometry(None, PLAIN, {}, "3b") == {"attn_res": False}
    assert attn_res_geometry({}, WITH_ATTNRES, {}, "3b")["attn_res"] is True


def test_boundaries_published_then_preset_then_legacy(caplog):
    assert attn_res_geometry(None, WITH_ATTNRES, {"attn_res_boundaries": [0, 2, 4]}, "3b") == {
        "attn_res": True, "attn_res_boundaries": [0, 2, 4]}
    for name in ("3b", "1b-tsumugi", "1b-moonshot-loop", "proxy"):
        out = attn_res_geometry({"attn_res": True}, WITH_ATTNRES, {}, name)
        assert out["attn_res_boundaries"] == list(presets.DEFAULT_BOUNDARIES[name])
    with caplog.at_level(logging.WARNING):
        out = attn_res_geometry(None, WITH_ATTNRES, {}, None)          # a YAML shape: no preset lookup
    assert out == {"attn_res": True, "attn_res_n_blocks": 7}
    assert "7 equal blocks" in caplog.text


@pytest.fixture()
def published(tmp_path):
    torch.manual_seed(0)
    model = KotodamaModel(KotodamaModelConfig(**TINY, attn_res=True, attn_res_boundaries=[0, 1]))
    d = tmp_path / "weights"
    d.mkdir()
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(d / "model.safetensors"))
    (d / "aliases.json").write_text("{}")
    (d / "config.json").write_text(json.dumps({"attn_res": True, "attn_res_boundaries": [0, 1], "step": 3}))
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump({"tiny": TINY}))
    return d, cfg, model


def test_published_directory_loads_with_its_boundaries(published):
    d, cfg, ref = published
    meta: dict = {}
    m = load_model(d, config_path=cfg, config_section="tiny", device="cpu", meta=meta)
    assert m.config.attn_res is True and list(m.config.attn_res_boundaries) == [0, 1]
    assert meta["step"] == 3
    ids = torch.randint(1, 256, (1, 12))
    with torch.no_grad():
        assert torch.allclose(m(ids)["logits"], ref.eval()(ids)["logits"], atol=1e-5)


def test_published_shape_mismatch_is_refused(published):
    d, _, _ = published
    (d / "config.json").write_text(json.dumps({"model_size": "1b-tsumugi", "attn_res_boundaries": [0, 1]}))
    with pytest.raises(ValueError, match="1b-tsumugi"):
        load_model(d, config_section="proxy", device="cpu")
