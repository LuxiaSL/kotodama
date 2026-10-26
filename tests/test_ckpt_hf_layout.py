"""Published weights (Hugging Face layout: model.safetensors + aliases.json + config.json) load through
kotodama.ckpt.load as a normal checkpoint dict, with looped-depth repeats re-tied to their core tensors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from kotodama import ckpt


def _publish(tmp: Path, sd: dict[str, torch.Tensor], aliases: dict[str, str], cfg: dict) -> Path:
    d = tmp / "published"
    d.mkdir()
    save_file({k: v for k, v in sd.items() if k not in aliases}, str(d / "model.safetensors"))
    (d / "aliases.json").write_text(json.dumps(aliases))
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def _state() -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    g = torch.Generator().manual_seed(0)
    core = torch.randn(4, 3, generator=g, dtype=torch.float32).to(torch.bfloat16)
    sd = {"layers.1.ffn.w": core, "layers.3.ffn.w": core, "norm.weight": torch.ones(3, dtype=torch.bfloat16)}
    return sd, {"layers.3.ffn.w": "layers.1.ffn.w"}


def test_directory_loads_and_reties_aliases(tmp_path):
    sd, aliases = _state()
    d = _publish(tmp_path, sd, aliases, {"step": 52450, "tokens_consumed": 110_000_000_000, "model_size": "x"})
    assert ckpt.is_hf_layout(d) and not ckpt.is_hf_ref(d)
    out = ckpt.load(d)
    assert set(out["model"]) == set(sd)
    assert all(torch.equal(out["model"][k], v) for k, v in sd.items())
    assert out["model"]["layers.3.ffn.w"] is out["model"]["layers.1.ffn.w"]   # one tensor, as in the looped model
    assert out["step"] == 52450 and out["tokens_consumed"] == 110_000_000_000 and out["config"]["model_size"] == "x"


def test_hf_ref_fetches_then_loads(tmp_path, monkeypatch):
    sd, aliases = _state()
    d = _publish(tmp_path, sd, aliases, {"step": 7})
    seen = {}

    def fake_snapshot_download(repo, revision=None, allow_patterns=None):
        seen.update(repo=repo, revision=revision, allow=tuple(allow_patterns))
        return str(d)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    out = ckpt.load("hf:Owner/model@main")
    assert seen == {"repo": "Owner/model", "revision": "main", "allow": ckpt.HF_FILES}
    assert out["step"] == 7 and set(out["model"]) == set(sd)
    ckpt.load("hf:Owner/model")
    assert seen["revision"] is None
    with pytest.raises(ValueError):
        ckpt.load("hf:")


def test_dangling_alias_is_an_error(tmp_path):
    sd = {"layers.1.ffn.w": torch.zeros(2, 2), "norm.weight": torch.ones(2)}   # distinct tensors: safetensors refuses shared
    d = _publish(tmp_path, sd, {"layers.9.ffn.w": "layers.7.ffn.w"}, {})
    with pytest.raises(KeyError, match="aliases.json"):
        ckpt.load(d)


def test_plain_checkpoints_still_load(tmp_path):
    p = tmp_path / "c.pt"
    torch.save({"model": {"a": torch.zeros(2)}, "step": 3}, p)
    out = ckpt.load(p)
    assert out["step"] == 3 and torch.equal(out["model"]["a"], torch.zeros(2))
