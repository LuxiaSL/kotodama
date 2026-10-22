"""Muon fp32_master: the fix for bf16 Muon weight decay being a no-op.

Without it, Muon on a bf16 model applies `p.mul_(1 - lr*wd)` (0.9998 at lr 0.02, wd 0.01)
and the update directly to the bf16 weights: the decay rounds back to 1.0 for every element,
and late updates fall below the bf16 spacing of the (undecayed, grown) weights.
These tests pin the bug, the fix's fp32-reference trajectory, resume on live parameters,
dtype preservation across save/load, legacy loading, and legacy-path bit-identity.
"""
from __future__ import annotations

import copy

import pytest
import torch

from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.training.muon import Muon, build_hybrid_optimizer

LR, WD = 0.02, 0.01


def _steps(opt, params, grads):
    for g in grads:
        for p, gg in zip(params, g):
            p.grad = gg.to(p.dtype)
        opt.step()
        opt.zero_grad()


def test_the_bug_bf16_muon_weight_decay_is_a_noop():
    torch.manual_seed(0)
    p = (torch.randn(64, 64) * 0.02).bfloat16()
    before = p.clone()
    opt = Muon([p], lr=LR, weight_decay=WD, fp32_master=False)
    _steps(opt, [p], [[torch.zeros(64, 64)] for _ in range(50)])
    assert torch.equal(p, before), "bf16 Muon weight decay now moves weights — reconsider fp32_master"


def test_fp32_master_applies_weight_decay():
    torch.manual_seed(0)
    p = (torch.randn(64, 64) * 0.02).bfloat16()
    opt = Muon([p], lr=LR, weight_decay=WD, fp32_master=True)
    before = p.float().norm().item()
    _steps(opt, [p], [[torch.zeros(64, 64)] for _ in range(500)])
    ratio = p.float().norm().item() / before
    assert ratio == pytest.approx((1 - LR * WD) ** 500, rel=5e-3)
    assert opt.state[p]["master"].dtype == torch.float32
    assert opt.state[p]["momentum_buffer"].dtype == torch.float32


def test_fp32_master_tracks_fp32_reference():
    torch.manual_seed(1)
    init = (torch.randn(48, 96) * 0.02).bfloat16()
    grads = [[(torch.randn(48, 96) * 1e-2).bfloat16().float()] for _ in range(30)]
    ref_p = init.float()  # true fp32 Muon, same (bf16-representable) inputs
    ref = Muon([ref_p], lr=LR, weight_decay=WD)
    _steps(ref, [ref_p], grads)
    p = init.clone()
    opt = Muon([p], lr=LR, weight_decay=WD, fp32_master=True)
    _steps(opt, [p], grads)
    master = opt.state[p]["master"]
    assert (master - ref_p).abs().max().item() < 1e-6
    assert torch.equal(p, master.bfloat16())
    # and the legacy bf16 path is measurably off the reference
    q = init.clone()
    legacy = Muon([q], lr=LR, weight_decay=WD)
    _steps(legacy, [q], grads)
    assert (q.float() - ref_p).abs().max().item() > 100 * (master - ref_p).abs().max().item()


def _tiny() -> KotodamaModel:
    torch.manual_seed(0)
    return KotodamaModel(KotodamaModelConfig(
        hidden_size=64, num_layers=2, num_attention_heads=2, num_kv_heads=1, head_dim=32,
        intermediate_size=128, vocab_size=256, max_position_embeddings=128,
        attn_res=True, attn_res_boundaries=[0, 1], attn_impl="sdpa",
    )).to(torch.bfloat16)


def test_resume_on_real_model_preserves_fp32_and_is_exact():
    model = _tiny()
    muon, adamw = build_hybrid_optimizer(model, muon_fp32_master=True)
    ids = torch.randint(0, 256, (2, 16))
    for _ in range(3):
        model(ids, labels=ids)["loss"].backward()
        muon.step(); adamw.step(); muon.zero_grad(); adamw.zero_grad()
    sd = copy.deepcopy(muon.state_dict())
    muon2, _ = build_hybrid_optimizer(model, muon_fp32_master=True)
    muon2.load_state_dict(sd)  # writes masters into live nn.Parameters (no_grad path)
    for p, s in muon2.state.items():
        assert s["master"].dtype == torch.float32 and s["momentum_buffer"].dtype == torch.float32
        assert torch.equal(s["master"], muon.state[p]["master"])
        assert torch.equal(p, s["master"].to(p.dtype))


def test_legacy_bf16_state_loads_into_fp32_master():
    model = _tiny()
    legacy, _ = build_hybrid_optimizer(model)
    ids = torch.randint(0, 256, (2, 16))
    model(ids, labels=ids)["loss"].backward()
    legacy.step(); legacy.zero_grad()
    fixed, _ = build_hybrid_optimizer(model, muon_fp32_master=True)
    fixed.load_state_dict(copy.deepcopy(legacy.state_dict()))
    for p, s in fixed.state.items():
        assert s["momentum_buffer"].dtype == torch.float32
        assert torch.equal(s["master"], p.detach().float())


def test_layout_mismatch_refused():
    a = Muon([torch.zeros(4, 4, dtype=torch.bfloat16)], fp32_master=True)
    b = Muon([torch.zeros(4, 4), torch.zeros(4, 4)], fp32_master=True)
    with pytest.raises(ValueError):
        a.load_state_dict(b.state_dict())


def test_legacy_path_unchanged_bitwise():
    torch.manual_seed(3)
    init = (torch.randn(32, 64) * 0.02).bfloat16()
    grads = [[torch.randn(32, 64) * 1e-2] for _ in range(10)]
    p1, p2 = init.clone(), init.clone()
    o1 = Muon([p1], lr=LR, weight_decay=WD)
    o2 = Muon([p2], lr=LR, weight_decay=WD, fp32_master=False)
    _steps(o1, [p1], grads); _steps(o2, [p2], grads)
    assert torch.equal(p1, p2)
