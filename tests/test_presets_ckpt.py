"""presets = the one shape table; ckpt = load/strip/registry round trips (CPU)."""

from __future__ import annotations

import shutil

import pytest
import torch

from kotodama import ckpt, presets
from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig

# The values the three pre-2026-09 tables (train.py MODEL_CONFIGS, serve.py
# MODEL_CONFIGS, configs/model.yaml) all held. A change here is a model change.
HISTORICAL = {
    "3b": (3072, 28, 24, 8, 128, 8192),
    "7b": (4096, 32, 32, 8, 128, 14336),
    "intermediate": (1024, 28, 8, 4, 128, 2816),
    "proxy": (512, 28, 4, 2, 128, 1408),
    "smoke": (256, 4, 4, 2, 64, 512),
}
KEYS = ("hidden_size", "num_layers", "num_attention_heads", "num_kv_heads", "head_dim", "intermediate_size")


@pytest.mark.parametrize("name", sorted(HISTORICAL))
def test_shapes_are_historical(name):
    s = presets.shape(name)
    assert tuple(s[k] for k in KEYS) == HISTORICAL[name]


def test_aliases_and_boundaries():
    assert presets.shape("full") == presets.shape("model") == presets.shape("3b")
    assert presets.shape("8b") == presets.shape("7b")
    assert presets.DD3B == (0, 1, 3, 7, 15, 19, 24)
    assert presets.DDV1 == (0, 3, 7, 12, 21, 25)
    with pytest.raises(KeyError):
        presets.shape("nope")


def test_shape_is_a_copy():
    presets.shape("3b")["hidden_size"] = 1
    assert presets.shape("3b")["hidden_size"] == 3072


def test_defaults_cover_the_rest():
    cfg = LuxiaModelConfig(**presets.shape("3b"))
    assert (cfg.rope_theta, cfg.norm_eps, cfg.qk_norm, cfg.tie_word_embeddings, cfg.z_loss_weight) == (
        500000.0, 1e-5, True, True, 1e-5)


@pytest.fixture
def tiny_ckpt(tmp_path):
    torch.manual_seed(0)
    model = LuxiaBaseModel(LuxiaModelConfig(**presets.shape("smoke")))
    state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    path = tmp_path / "step_00000010.pt"
    torch.save({"model": state, "step": 10, "tokens_consumed": 1234,
                "optimizer": {"state": {}, "param_groups": []}}, path)
    return path, model.state_dict()


def test_load_and_model_state(tiny_ckpt):
    path, ref = tiny_ckpt
    state = ckpt.model_state(ckpt.load(path))
    assert state.keys() == ref.keys()
    assert all(torch.equal(state[k], ref[k]) for k in ref)


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd binary not installed")
def test_strip_zst_round_trip(tiny_ckpt, tmp_path):
    path, ref = tiny_ckpt
    out = ckpt.strip(path, tmp_path / "out" / "m.pt.zst")
    assert out.exists() and not out.with_suffix("").exists()
    back = ckpt.load(out)
    assert "optimizer" not in back and back["step"] == 10
    assert all(torch.equal(back["model"][k], ref[k]) for k in ref)
    meta = ckpt.info(out)
    assert meta["step"] == 10 and not meta["has_optimizer"]
    with pytest.raises(FileExistsError):
        ckpt.strip(path, out)


def test_strip_bf16(tiny_ckpt, tmp_path):
    path, _ = tiny_ckpt
    out = ckpt.strip(path, tmp_path / "m.pt", bf16=True)
    assert ckpt.info(out)["dtypes"] == ["torch.bfloat16"]


def test_registry_resolution(tiny_ckpt, tmp_path, monkeypatch):
    path, _ = tiny_ckpt
    reg = tmp_path / "private.yaml"
    reg.write_text(f"tiny:\n  path: {path.name}\n  size: smoke\n  boundaries: null\n")
    monkeypatch.setenv("KOTODAMA_DATA_ROOT", str(path.parent))
    monkeypatch.setenv("KOTODAMA_CKPT_REGISTRY", str(reg))
    assert ckpt.resolve("tiny") == path
    assert ckpt.registry()["tiny"].boundaries is None
    assert "kotodama-3b-base-final" in ckpt.registry()
    with pytest.raises(FileNotFoundError):
        ckpt.resolve("kotodama-3b-base-final")  # registered, absent under this data root
    with pytest.raises(FileNotFoundError):
        ckpt.resolve("no-such-name")
