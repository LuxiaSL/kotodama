"""Who-reads-what routing keys: source layout on the DD-3B structure, telemetry keys ==
the same read off the capture walker, and routing_only capture == full capture."""

from __future__ import annotations

import pytest
import torch

from kotodama import presets
from kotodama.model import routing_read as rr
from kotodama.model.capture import instrumented_forward
from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig

BOUNDARIES = [0, 1, 3]  # 6 layers: blocks L0 | L1-2 | L3-5


def _ranges(bounds: list[int], n_layers: int) -> list[tuple[int, int]]:
    return list(zip(bounds, bounds[1:] + [n_layers]))


# ── layout ─────────────────────────────────────────────────────────────────────

def test_dd3b_layout_labels_and_lookback():
    ranges = _ranges(list(presets.DEFAULT_BOUNDARIES["1b"]), 28)
    assert ranges[0] == (0, 1) and ranges[-1] == (24, 28)
    # layer 0 pre_mlp: embedding + its own block
    s = rr.site_layout(ranges, 0, "pre_mlp", 2)
    assert s.labels == ("emb", "self:L0") and s.rel == ("emb", "self")
    # block-start pre_attn at layer 3: emb, L0 committed; L1-2 just finished (its partial)
    s = rr.site_layout(ranges, 3, "pre_attn", 3)
    assert s.labels == ("emb", "L0", "L1-2") and s.rel == ("emb", "older", "prev")
    # mid-block layer 10 pre_mlp: emb, L0, L1-2, L3-6 committed + self L7-14
    s = rr.site_layout(ranges, 10, "pre_mlp", 5)
    assert s.labels == ("emb", "L0", "L1-2", "L3-6", "self:L7-14")
    assert s.rel == ("emb", "older", "older", "prev", "self")
    one_hot = torch.eye(5)
    assert [rr.lookback(one_hot[j], s) for j in range(5)] == [11.0, 10.0, 8.0, 4.0, 0.0]
    # final: emb + six finished blocks + self L24-27
    s = rr.site_layout(ranges, -1, "final", 8)
    assert s.labels[-2:] == ("L19-23", "self:L24-27") and s.at == 28
    with pytest.raises(ValueError):
        rr.site_layout(ranges, 10, "pre_mlp", 4)


def test_split_sums_to_one_and_token_entropy_bounds():
    ranges = _ranges(BOUNDARIES, 6)
    s = rr.site_layout(ranges, 4, "pre_attn", 4)
    w = torch.softmax(torch.randn(4, 2, 9), dim=0)
    assert abs(sum(rr.relative_split(w, s).values()) - 1.0) < 1e-5
    assert rr.token_entropy(torch.eye(4)[:, :, None].expand(4, 4, 3)) < 1e-6
    assert abs(rr.token_entropy(torch.full((4, 2, 3), 0.25)) - 1.0) < 1e-6


# ── telemetry == capture ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    torch.manual_seed(20260930)
    cfg = LuxiaModelConfig(
        hidden_size=64, num_layers=6, num_attention_heads=4, num_kv_heads=2, head_dim=16,
        intermediate_size=128, vocab_size=199, max_position_embeddings=96, attn_impl="sdpa",
        attn_res=True, attn_res_boundaries=BOUNDARIES, use_liger=False,
    )
    m = LuxiaBaseModel(cfg)
    # random routing queries so the softmaxes are not uniform (init is zero)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "res_query" in n:
                p.normal_(0, 0.5)
    m.eval()
    return m


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(11)
    return torch.randint(3, 199, (2, 24), dtype=torch.long)


def test_telemetry_route_keys_match_capture(model, ids):
    with torch.no_grad():
        model(ids)
    diag = model._last_attn_res_diagnostics
    _, cap = instrumented_forward(model, ids, routing_only=True)
    ranges = model._attn_res_block_ranges
    n_checked = 0
    for sub, layer, w in cap["routing"]:
        site = rr.site_layout(ranges, layer, sub, w.shape[0])
        prefix = "attnres/route/final" if sub == "final" else f"attnres/route/layer_{layer}/{sub}"
        for src, v in rr.relative_split(w, site).items():
            assert diag[f"{prefix}/from_{src}"] == pytest.approx(v, abs=1e-5), (prefix, src)
        assert diag[f"{prefix}/lookback"] == pytest.approx(rr.lookback(w, site), abs=1e-4)
        assert diag[f"{prefix}/token_entropy"] == pytest.approx(rr.token_entropy(w), abs=1e-5)
        n_checked += 1
    # every pre_mlp (6) + every non-first pre_attn (5) + final
    assert n_checked == 12
    assert sum(1 for k in diag if k.startswith("attnres/route/")) > 0
    # existing keys are still there
    assert "attnres/avg_alpha/partial" in diag and "attnres/routing_entropy/layer_0/pre_mlp" in diag


def test_routing_only_capture_equals_full_capture(model, ids):
    lf, cf = instrumented_forward(model, ids)
    lr, cr = instrumented_forward(model, ids, routing_only=True)
    assert torch.allclose(lf, lr, atol=1e-4)
    assert [(s, l) for s, l, _ in cf["routing"]] == [(s, l) for s, l, _ in cr["routing"]]
    for (_, _, a), (_, _, b) in zip(cf["routing"], cr["routing"]):
        assert torch.allclose(a, b, atol=1e-5)
    assert cr["attn"] == [] and cr["residual_partial"] == [] and cr["committed"] == []
    with pytest.raises(ValueError):
        instrumented_forward(model, ids, light=True, routing_only=True)
