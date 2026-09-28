"""Tests for MasterAdamW — the fp32-master fix for the bf16 norm-freeze bug.

The bug (2026-09-27, the 3B archaeology): under pure-bf16 training, stock
AdamW's ~6e-4 steps on RMSNorm gains at 1.0 round away against bf16's 0.0039
spacing — every norm gain in the shipped 3B was bit-frozen at init for the
entire run. These tests pin: the bug itself (so nobody "simplifies" the fix
away), the fix's fp32-reference-exact trajectory, end-to-end norm movement on
a real tiny model through build_hybrid_optimizer, resume exactness with fp32
state preserved, and graceful legacy-state loading.
"""
from __future__ import annotations

import copy

import pytest
import torch
from torch.optim import AdamW as StockAdamW

from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig
from kotodama.training.muon import MasterAdamW, build_hybrid_optimizer

LR, WD = 6e-4, 0.1


def _grad_steps(opt, params, grads, n):
    for _ in range(n):
        for p, g in zip(params, grads):
            p.grad = g.clone().to(p.dtype)
        opt.step()
        opt.zero_grad()


def test_the_bug_stock_adamw_freezes_bf16_norm_gains():
    """Documentation test: this is the failure the fix exists for."""
    p = torch.ones(64, dtype=torch.bfloat16)
    opt = StockAdamW([p], lr=LR, betas=(0.9, 0.95), weight_decay=WD)
    _grad_steps(opt, [p], [torch.full((64,), 1e-3)], 30)
    assert torch.equal(p, torch.ones(64, dtype=torch.bfloat16)), (
        "stock AdamW no longer freezes bf16 params at 1.0 — if torch fixed "
        "this upstream, MasterAdamW can be reconsidered"
    )


def test_master_adamw_moves_and_tracks_fp32_reference():
    torch.manual_seed(0)
    init = torch.ones(64)
    p_bf16 = init.clone().to(torch.bfloat16)
    p_ref = init.clone()  # pure-fp32 twin under stock AdamW
    fixed = MasterAdamW([p_bf16], lr=LR, betas=(0.9, 0.95), weight_decay=WD)
    ref = StockAdamW([p_ref], lr=LR, betas=(0.9, 0.95), weight_decay=WD)
    grads = [torch.randn(64) * 1e-3 for _ in range(40)]
    for g in grads:
        p_bf16.grad = g.to(torch.bfloat16)
        p_ref.grad = g.clone()
        fixed.step(); ref.step()
        fixed.zero_grad(); ref.zero_grad()
    master = fixed.state[p_bf16]["master"]
    assert master.dtype == torch.float32
    # bf16 grads quantize the inputs, so allow small tolerance vs the fp32 ref
    assert torch.allclose(master, p_ref, rtol=0, atol=5e-5)
    assert not torch.equal(p_bf16, torch.ones(64, dtype=torch.bfloat16)), \
        "param never moved — the freeze is back"
    # the param is exactly the bf16 view of the master
    assert torch.equal(p_bf16, master.to(torch.bfloat16))


def test_hybrid_optimizer_trains_norms_on_real_model():
    torch.manual_seed(0)
    model = LuxiaBaseModel(LuxiaModelConfig(
        hidden_size=64, num_layers=2, num_attention_heads=2, num_kv_heads=1,
        head_dim=32, intermediate_size=128, vocab_size=256,
        max_position_embeddings=128, attn_res=False, attn_impl="sdpa",
    )).to(torch.bfloat16)
    norm_keys = [n for n, p in model.named_parameters()
                 if p.ndim == 1 and "norm" in n]
    assert norm_keys, "no norm params found — split assumption broke"
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    muon_opt, adamw_opt = build_hybrid_optimizer(model)
    assert isinstance(adamw_opt, MasterAdamW)
    ids = torch.randint(0, 256, (2, 32))
    for _ in range(25):
        out = model(ids, labels=ids)
        out["loss"].backward()
        muon_opt.step(); adamw_opt.step()
        muon_opt.zero_grad(); adamw_opt.zero_grad()
    moved = [n for n in norm_keys
             if not torch.equal(dict(model.named_parameters())[n], before[n])]
    assert moved, ("no norm gain moved after 25 steps — the norm-freeze bug "
                   "is back in the hybrid path")
    emb = "embed_tokens.weight"
    assert not torch.equal(dict(model.named_parameters())[emb], before[emb])


def test_state_roundtrip_preserves_fp32_and_resumes_exactly():
    torch.manual_seed(1)
    p1 = torch.ones(16, dtype=torch.bfloat16)
    opt1 = MasterAdamW([p1], lr=LR, weight_decay=WD)
    grads = [torch.randn(16) * 1e-3 for _ in range(6)]
    _grad_steps(opt1, [p1], grads[:3], 1)
    _grad_steps(opt1, [p1], grads[1:2], 1)
    sd = copy.deepcopy(opt1.state_dict())

    p2 = p1.detach().clone()
    opt2 = MasterAdamW([p2], lr=LR, weight_decay=WD)
    opt2.load_state_dict(sd)
    s1, s2 = opt1.state[p1], opt2.state[p2]
    for k in ("master", "exp_avg", "exp_avg_sq"):
        assert s2[k].dtype == torch.float32, f"{k} lost fp32 on load"
        assert torch.equal(s1[k], s2[k])
    assert s1["step"] == s2["step"]
    # one more identical step on both → bitwise-identical params and masters
    _grad_steps(opt1, [p1], grads[2:3], 1)
    _grad_steps(opt2, [p2], grads[2:3], 1)
    assert torch.equal(p1, p2)
    assert torch.equal(opt1.state[p1]["master"], opt2.state[p2]["master"])


def test_legacy_stock_state_loads_gracefully():
    p_old = torch.ones(16, dtype=torch.bfloat16)
    legacy = StockAdamW([p_old], lr=LR, betas=(0.9, 0.95), weight_decay=WD)
    _grad_steps(legacy, [p_old], [torch.full((16,), 1e-3)], 2)
    sd = copy.deepcopy(legacy.state_dict())

    p_new = p_old.detach().clone()
    opt = MasterAdamW([p_new], lr=LR, betas=(0.9, 0.95), weight_decay=WD)
    opt.load_state_dict(sd)  # no masters in the saved state
    s = opt.state[p_new]
    assert s["master"].dtype == torch.float32
    assert torch.equal(s["master"], p_new.to(torch.float32))
    assert s["exp_avg"].dtype == torch.float32  # upcast from legacy
    assert s["step"] == 2
    _grad_steps(opt, [p_new], [torch.full((16,), 1e-3)], 5)  # keeps running


def test_layout_mismatch_refused():
    p = torch.ones(8, dtype=torch.bfloat16)
    opt = MasterAdamW([p], lr=LR)
    other = MasterAdamW([torch.ones(8), torch.ones(4)], lr=LR)
    with pytest.raises(ValueError):
        opt.load_state_dict(other.state_dict())
