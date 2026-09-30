"""1B ladder checkpoints through the eval/serve loaders: the `overrides` path carries
rope_fp32 without disturbing AttnRes auto-detection, and the server's 1b shape is the
preset with DD-3B boundaries."""
from __future__ import annotations

import pytest
import torch
import yaml

from kotodama import presets
from kotodama.eval.model_loader import load_model
from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig

TINY = dict(hidden_size=64, num_layers=2, num_attention_heads=2, num_kv_heads=1, head_dim=32,
            intermediate_size=128, vocab_size=256, max_position_embeddings=128, attn_impl="sdpa")


@pytest.fixture()
def tiny_ckpt(tmp_path):
    torch.manual_seed(0)
    model = LuxiaBaseModel(LuxiaModelConfig(**TINY, attn_res=True, attn_res_boundaries=[0, 1]))
    ck = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(), "step": 7}, ck)
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(yaml.safe_dump({"tiny": TINY}))
    return ck, cfg, model


def test_overrides_set_rope_fp32_and_keep_autodetect(tiny_ckpt):
    ck, cfg, ref = tiny_ckpt
    m = load_model(ck, config_path=cfg, config_section="tiny", device="cpu",
                   overrides={"rope_fp32": True})
    assert m.config.rope_fp32 is True
    assert m.config.attn_res is True  # auto-detection still ran (no attn_res_config given)
    assert m.rope_cos.dtype == torch.float32
    ids = torch.randint(1, 256, (1, 16))
    with torch.no_grad():
        out = m(ids)["logits"]
    assert torch.isfinite(out).all()


def test_no_overrides_is_unchanged(tiny_ckpt):
    ck, cfg, _ = tiny_ckpt
    m = load_model(ck, config_path=cfg, config_section="tiny", device="cpu")
    assert m.config.rope_fp32 is False


def test_server_1b_shape_is_the_preset_with_dd3b():
    from kotodama.serve import server
    c = server.MODEL_CONFIGS["1b"]
    for k, v in presets.shape("1b").items():
        assert c[k] == v, k
    assert list(c["attn_res_boundaries"]) == [0, 1, 3, 7, 15, 19, 24]
    assert c["attn_res"] is True
    assert server.SIZE_LABELS["1b"] == "1b"
