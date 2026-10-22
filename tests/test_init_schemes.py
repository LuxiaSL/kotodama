"""Init schemes: determinism, shapes, norms, intended structure; 'standard' is a no-op."""

from __future__ import annotations

import math

import pytest
import torch

from kotodama.model.init_schemes import SCHEMES, apply_init_scheme, depth_gain
from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig
from kotodama.presets import SHAPES


def _model(seed: int = 0) -> LuxiaBaseModel:
    torch.manual_seed(seed)
    cfg = LuxiaModelConfig(**SHAPES["smoke"])
    return LuxiaBaseModel(cfg)


def _state(m: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def test_standard_is_noop():
    m = _model(); before = _state(m)
    apply_init_scheme(m, "standard")
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        apply_init_scheme(_model(), "nope")


@pytest.mark.parametrize("scheme", [s for s in SCHEMES if s != "standard"])
def test_deterministic_and_shape_preserving(scheme):
    a, b = _model(3), _model(3)
    torch.manual_seed(11); apply_init_scheme(a, scheme)
    torch.manual_seed(11); apply_init_scheme(b, scheme)
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    ref = _model(3).state_dict()
    assert all(sa[k].shape == ref[k].shape and sa[k].dtype == ref[k].dtype for k in sa)
    assert all(torch.isfinite(v).all() for v in sa.values())


@pytest.mark.parametrize("scheme", ["mimetic", "orthogonal"])
def test_frobenius_norm_matches_standard(scheme):
    m = _model(); torch.manual_seed(5); apply_init_scheme(m, scheme)
    L = len(m.layers)
    for layer in m.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            w = getattr(layer.attn, name).weight
            scale = 1 / math.sqrt(2 * L) if name == "o_proj" else 1.0
            assert w.norm().item() == pytest.approx(0.02 * scale * math.sqrt(w.numel()), rel=1e-3)


def test_orthogonal_rows_are_orthogonal():
    m = _model(); torch.manual_seed(1); apply_init_scheme(m, "orthogonal")
    w = m.layers[0].attn.q_proj.weight.float()            # (nh*hd) x d
    g = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
    g = g / g.diag().mean()
    assert torch.allclose(g, torch.eye(g.shape[0]), atol=1e-3)


def test_mimetic_qk_has_identity_component():
    m = _model(); torch.manual_seed(2); apply_init_scheme(m, "mimetic")
    at = m.layers[0].attn
    hd, d = at.head_dim, at.q_proj.in_features
    wq = at.q_proj.weight.float()[:hd]; wk = at.k_proj.weight.float()[:hd]
    p = wq.T @ wk                                        # d x d: x·(W_Qᵀ W_K)·x
    diag_mean = p.diag().mean().item(); off_rms = (p - torch.diag(p.diag())).pow(2).mean().sqrt().item()
    assert diag_mean > 0 and diag_mean > 3 * off_rms     # identity-dominated (attend to matching content)
    # standard init has no such structure
    s = _model(); ws, ks = s.layers[0].attn.q_proj.weight.float()[:hd], s.layers[0].attn.k_proj.weight.float()[:hd]
    ps = ws.T @ ks
    assert abs(ps.diag().mean().item()) < 3 * (ps - torch.diag(ps.diag())).pow(2).mean().sqrt().item()


def test_mimetic_vo_negative_identity():
    m = _model(); torch.manual_seed(4); apply_init_scheme(m, "mimetic")
    at = m.layers[0].attn
    hd, group = at.head_dim, at.num_heads // at.num_kv_heads
    wv = at.v_proj.weight.float()[:hd]
    wo = sum(at.o_proj.weight.float()[:, j * hd:(j + 1) * hd] for j in range(group))
    p = wo @ wv
    assert p.diag().mean().item() < 0


def test_depth_graded_gains():
    assert depth_gain(0, 28) == 1.0 and depth_gain(27, 28) == 1.0
    assert depth_gain(14, 28) == pytest.approx(1.5)
    assert 1.0 < depth_gain(8, 28) < 1.5 and 1.0 < depth_gain(21, 28) < 1.5
    m = _model(); before = [l.attn.q_norm.weight.clone() for l in m.layers]
    apply_init_scheme(m, "depth_graded")
    L = len(m.layers)
    for i, (l, b) in enumerate(zip(m.layers, before)):
        g = l.attn.q_norm.weight / b
        assert torch.allclose(g, torch.full_like(g, math.sqrt(depth_gain(i, L))), atol=1e-2)


def test_small_attn_out():
    m = _model(); before = m.layers[1].attn.o_proj.weight.clone()
    apply_init_scheme(m, "small_attn_out")
    assert torch.allclose(m.layers[1].attn.o_proj.weight, before * 0.1)
