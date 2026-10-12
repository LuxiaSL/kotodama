"""--attn_lr_scale: the attention projections get their own Muon group whose LR is scaled for a window."""
import pytest
import torch

from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig
from kotodama.training.muon import HybridScheduler, build_hybrid_optimizer


def _tiny() -> LuxiaBaseModel:
    cfg = LuxiaModelConfig(hidden_size=64, num_layers=2, num_attention_heads=4, num_kv_heads=2, head_dim=16,
                           intermediate_size=128, vocab_size=256, attn_impl="sdpa", use_liger=False)
    torch.manual_seed(0)
    return LuxiaBaseModel(cfg)


def test_default_is_one_group_unchanged() -> None:
    muon, _ = build_hybrid_optimizer(_tiny(), muon_lr=0.04)
    assert len(muon.param_groups) == 1


def test_attn_group_split_and_window() -> None:
    model = _tiny()
    muon, adamw = build_hybrid_optimizer(model, muon_lr=0.04, attn_group=True)
    tags = [g["tag"] for g in muon.param_groups]
    assert tags == ["main", "attn"]
    attn_ids = {id(p) for p in muon.param_groups[1]["params"]}
    names = {n for n, p in model.named_parameters() if id(p) in attn_ids}
    assert names and all(".attn." in n and n.endswith("_proj.weight") for n in names)
    assert len(names) == 2 * 4  # q,k,v,o x 2 layers
    sch = HybridScheduler(muon, adamw, warmup_steps=0, total_steps=100, decay_start_pct=0.8,
                          attn_lr_scale=0.1, attn_lr_scale_steps=10)
    sch.step(5)
    assert muon.param_groups[0]["lr"] == pytest.approx(0.04)
    assert muon.param_groups[1]["lr"] == pytest.approx(0.004)
    sch.step(10)
    assert muon.param_groups[1]["lr"] == pytest.approx(0.04)


def test_scale_without_group_is_an_error() -> None:
    muon, adamw = build_hybrid_optimizer(_tiny(), muon_lr=0.04)
    with pytest.raises(ValueError):
        HybridScheduler(muon, adamw, warmup_steps=0, total_steps=100, attn_lr_scale=0.1, attn_lr_scale_steps=10)


def test_attn_group_optimizer_steps() -> None:
    model = _tiny()
    muon, adamw = build_hybrid_optimizer(model, muon_lr=0.04, attn_group=True)
    sch = HybridScheduler(muon, adamw, warmup_steps=0, total_steps=10, attn_lr_scale=0.0, attn_lr_scale_steps=5)
    q = model.layers[0].attn.q_proj.weight
    g = model.layers[0].ffn.gate_proj.weight if hasattr(model.layers[0], "ffn") else None
    q0 = q.detach().clone()
    other = [p for p in muon.param_groups[0]["params"]][0]; o0 = other.detach().clone()
    sch.step(0)
    loss = model(torch.randint(0, 256, (2, 8)))["logits"].float().pow(2).mean()
    loss.backward(); muon.step(); adamw.step()
    assert torch.equal(q, q0)            # frozen (scale 0) inside the window
    assert not torch.equal(other, o0)    # everything else trains
