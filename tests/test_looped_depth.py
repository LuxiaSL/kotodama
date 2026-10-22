"""Looped depth: loop repeats alias their core position's attention + FFN, nothing else.

Gates: the aliasing is exactly "the same weights at every repeat" (a looped model's state_dict loaded into an
UNLOOPED model of the same effective depth gives identical logits); per-position norms/routing stay unique; the
optimizer sees each shared weight once; param_count and the preset agree with the instantiated model; the capture
walker (instrumented_forward) matches model.forward on a looped model; a state_dict round-trip keeps the tying.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import pytest
import torch

from kotodama import presets
from kotodama.model.capture import hidden_rows, instrumented_forward
from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig

BASE = dict(hidden_size=64, num_layers=7, num_attention_heads=4, num_kv_heads=2, head_dim=16,
            intermediate_size=128, vocab_size=199, max_position_embeddings=96, attn_impl="sdpa",
            attn_res=True, attn_res_boundaries=[0, 2, 5], use_liger=False)
LOOP = dict(loop_start=1, loop_len=2, loop_times=2)        # 1 prelude + 2x2 core + 2 coda = 7 effective, 5 unique


@pytest.fixture(scope="module")
def looped() -> LuxiaBaseModel:
    torch.manual_seed(20261018)
    m = LuxiaBaseModel(LuxiaModelConfig(**BASE, **LOOP)); m.eval()
    return m


@pytest.fixture(scope="module")
def ids() -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randint(3, 199, (2, 24), dtype=torch.long)


def test_layer_source_map():
    cfg = LuxiaModelConfig(**BASE, **LOOP)
    assert [cfg.layer_source(i) for i in range(7)] == [0, 1, 2, 1, 2, 5, 6]
    assert cfg.num_unique_layers == 5


@pytest.mark.parametrize("bad", [dict(loop_start=1, loop_len=2, loop_times=1),      # times < 2
                                 dict(loop_start=4, loop_len=2, loop_times=2),      # runs past num_layers
                                 dict(loop_start=0, loop_len=0, loop_times=2)])
def test_bad_loop_spec_refuses(bad):
    with pytest.raises(ValueError):
        LuxiaModelConfig(**BASE, **bad)


def test_heavy_modules_shared_small_ones_not(looped):
    L = looped.layers
    for i, src in ((3, 1), (4, 2)):
        assert L[i].attn is L[src].attn and L[i].ffn is L[src].ffn
        for name in ("attn_norm", "ffn_norm", "attn_res_norm", "mlp_res_norm"):
            assert getattr(L[i], name) is not getattr(L[src], name), name
        assert L[i].attn_res_query is not L[src].attn_res_query
    assert L[5].attn is not L[1].attn                      # coda is unique


def test_optimizer_sees_each_shared_weight_once(looped):
    names = [n for n, _ in looped.named_parameters()]
    assert len(names) == len({id(p) for p in looped.parameters()})
    assert not any(n.startswith(("layers.3.attn.", "layers.3.ffn.", "layers.4.attn.", "layers.4.ffn.")) for n in names)


def test_param_count_matches_instantiated_model():
    cfg = LuxiaModelConfig(**dict(BASE, attn_res=False), **LOOP)
    m = LuxiaBaseModel(cfg)
    assert sum(p.numel() for p in m.parameters()) == cfg.param_count()


def test_aliasing_equals_explicit_copies(looped, ids):
    """Loading the looped state_dict (which lists every position) into an UNLOOPED model of the same effective depth
    gives identical logits: weight sharing is exactly 'repeat the same weights', nothing more."""
    flat = LuxiaBaseModel(LuxiaModelConfig(**BASE)); flat.eval()
    flat.load_state_dict(looped.state_dict(), strict=True)
    with torch.no_grad():
        a = looped(ids)["logits"]; b = flat(ids)["logits"]
    assert torch.equal(a, b)


def test_capture_walker_matches_forward(looped, ids):
    with torch.no_grad():
        ref = looped(ids)["logits"]
        logits, cap = instrumented_forward(looped, ids, light=True)
    assert (logits.float() - ref.float()).abs().max().item() <= 1e-5
    assert len(hidden_rows(cap, looped.config.num_layers)) == looped.config.num_layers + 1   # embeddings + 1/position


def test_state_dict_roundtrip_keeps_tying(looped, ids):
    fresh = LuxiaBaseModel(LuxiaModelConfig(**BASE, **LOOP)); fresh.eval()
    fresh.load_state_dict(looped.state_dict(), strict=True)
    assert fresh.layers[3].attn is fresh.layers[1].attn
    with torch.no_grad():
        assert torch.equal(fresh(ids)["logits"], looped(ids)["logits"])


def test_backward_accumulates_into_shared_weights(ids):
    torch.manual_seed(0)
    m = LuxiaBaseModel(LuxiaModelConfig(**BASE, **LOOP))
    out = m(ids, labels=ids)
    out["loss"].backward()
    g = m.layers[1].attn.q_proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_preset_1b_deep_loop():
    """The 1b-deep-loop preset: 1b-deep's width and effective depth, 28 unique layers, deep's boundaries."""
    s, d = presets.shape("1b-deep-loop"), presets.shape("1b-deep")
    for k in ("hidden_size", "num_layers", "num_attention_heads", "num_kv_heads", "head_dim", "intermediate_size"):
        assert s[k] == d[k], k
    cfg = LuxiaModelConfig(**s)
    assert cfg.num_layers == 52 and cfg.num_unique_layers == 28
    assert presets.DEFAULT_BOUNDARIES["1b-deep-loop"] == presets.DEFAULT_BOUNDARIES["1b-deep"]
    assert 0.75e9 < cfg.param_count() < 0.85e9


def test_serve_knows_1b_deep_loop():
    from kotodama.serve import server
    c = server.MODEL_CONFIGS["1b-deep-loop"]
    assert c["loop_times"] == 3 and c["attn_res_boundaries"] == list(presets.DEFAULT_BOUNDARIES["1b-deep-loop"])


def test_preset_1b_deep_loopfit():
    """1b-deep-loopfit: the loop sits wholly inside one CF52_FIT block (19→39), 42 unique layers, deep's width/depth."""
    s, d = presets.shape("1b-deep-loopfit"), presets.shape("1b-deep")
    for k in ("hidden_size", "num_layers", "num_attention_heads", "num_kv_heads", "head_dim", "intermediate_size"):
        assert s[k] == d[k], k
    cfg = LuxiaModelConfig(**s)
    assert cfg.num_layers == 52 and cfg.num_unique_layers == 42
    b = presets.DEFAULT_BOUNDARIES["1b-deep-loopfit"]
    assert b == presets.CF52_FIT
    end = s["loop_start"] + s["loop_len"] * s["loop_times"]
    blk = [x for x in b if x <= s["loop_start"]][-1]
    nxt = [x for x in b if x > s["loop_start"]][0]
    assert blk == s["loop_start"] and end == nxt          # the loop spans exactly one block
    assert 1.1e9 < cfg.param_count() < 1.25e9
    from kotodama.serve import server
    c = server.MODEL_CONFIGS["1b-deep-loopfit"]
    assert c["loop_times"] == 2 and c["attn_res_boundaries"] == list(b)


def test_preset_1b_moonshot_loop():
    """1b-moonshot-loop (configs/moonshot-1b-shapeA.yaml): 52 unique (1b-deep's budget), 72 effective, the loop = the CF52_FIT long block 19–38 x2,
    a block boundary at every pass start and at the coda."""
    s, d = presets.shape("1b-moonshot-loop"), presets.shape("1b-deep")
    for k in ("hidden_size", "num_attention_heads", "num_kv_heads", "head_dim", "intermediate_size"):
        assert s[k] == d[k], k
    cfg = LuxiaModelConfig(**s)
    assert cfg.num_layers == 72 and cfg.num_unique_layers == 52
    b = presets.DEFAULT_BOUNDARIES["1b-moonshot-loop"]
    assert b == presets.MOONSHOT_LOOP
    starts = [s["loop_start"] + k * s["loop_len"] for k in range(s["loop_times"])]
    coda = s["loop_start"] + s["loop_len"] * s["loop_times"]
    assert all(x in b for x in starts) and coda in b
    assert abs(cfg.param_count() - LuxiaModelConfig(**d).param_count()) / LuxiaModelConfig(**d).param_count() < 0.01
    from kotodama.serve import server
    c = server.MODEL_CONFIGS["1b-moonshot-loop"]
    assert c["num_layers"] == 72 and c["attn_res_boundaries"] == list(b)
