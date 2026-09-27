"""Tests for kotodama.eval.induction (the P0.4 induction / prefix-matching score).

Stub models pin the math exactly (a perfect copier scores ~ln|V|, uniform
logits score 0, per-position bookkeeping lands on the right index); tiny real
LuxiaBaseModels (random init, sdpa, CPU) run the behavioral channel and the
capture-backed prefix-matching read end to end.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kotodama.eval.induction import (aggregate_induction, build_repeat_batch,
                                     candidate_ids, prefix_matching_scores,
                                     split_halves, summarize_heads,
                                     token_losses)

VOCAB = 64


class UniformModel:
    """Flat logits: every token costs exactly ln(VOCAB)."""

    def __call__(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": torch.zeros(*ids.shape, VOCAB)}


class CopierModel:
    """Ideal induction: at position t, find the latest s < t with
    ids[s] == ids[t] and put a +50 peak on ids[s+1]; flat otherwise."""

    def __call__(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        B, S = ids.shape
        logits = torch.zeros(B, S, VOCAB)
        for b in range(B):
            last: dict[int, int] = {}
            for t in range(S):
                tok = int(ids[b, t])
                if tok in last and last[tok] + 1 < S:
                    logits[b, t, ids[b, last[tok] + 1]] = 50.0
                last[tok] = t
        return {"logits": logits}


class OracleAt:
    """Peaks on the true next token only at predicting positions in ``pos``."""

    def __init__(self, pos: set[int]):
        self.pos = pos

    def __call__(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        B, S = ids.shape
        logits = torch.zeros(B, S, VOCAB)
        for t in self.pos:
            logits[torch.arange(B), t, ids[:, t + 1]] = 50.0
        return {"logits": logits}


def test_build_repeat_batch_shape_repeat_and_exclusions():
    ids = build_repeat_batch(5, 10, 0, VOCAB, exclude=[0, 1, 2], seed=3,
                             prepend_id=0)
    assert ids.shape == (5, 21)
    assert (ids[:, 0] == 0).all()
    assert torch.equal(ids[:, 1:11], ids[:, 11:21])
    body = ids[:, 1:]
    assert body.min() >= 3 and body.max() < VOCAB
    # seeded: same seed same batch, different seed different batch
    assert torch.equal(ids, build_repeat_batch(5, 10, 0, VOCAB, [0, 1, 2],
                                               seed=3, prepend_id=0))
    assert not torch.equal(ids, build_repeat_batch(5, 10, 0, VOCAB, [0, 1, 2],
                                                   seed=4, prepend_id=0))
    no_pre = build_repeat_batch(2, 4, 10, 20, seed=0)
    assert no_pre.shape == (2, 8) and no_pre.min() >= 10 and no_pre.max() < 20
    assert len(candidate_ids(0, VOCAB, [0, 5, 999])) == VOCAB - 2
    with pytest.raises(ValueError):
        candidate_ids(5, 6, [5])


def test_uniform_model_scores_zero():
    ids = build_repeat_batch(4, 12, 1, VOCAB, seed=0, prepend_id=0)
    losses = token_losses(UniformModel(), ids, batch_size=3)
    assert torch.isnan(losses[:, 0]).all()
    assert torch.allclose(losses[:, 1:], torch.full_like(losses[:, 1:],
                                                         math.log(VOCAB)))
    agg = aggregate_induction(losses, 12, 1, n_candidates=VOCAB - 1)
    assert agg["induction_gain"] == pytest.approx(0.0, abs=1e-9)
    assert agg["induction_gain_normalized"] == pytest.approx(0.0, abs=1e-9)
    assert agg["loss_first"] == pytest.approx(math.log(VOCAB))
    assert agg["uniform_loss"] == pytest.approx(math.log(VOCAB - 1))


def test_perfect_copier_scores_high():
    # distinct tokens inside A so "latest earlier occurrence" is unambiguous
    L = 16
    a = torch.stack([torch.randperm(VOCAB - 1, generator=torch.Generator()
                                    .manual_seed(s))[:L] + 1 for s in range(6)])
    ids = torch.cat([torch.zeros(6, 1, dtype=torch.long), a, a], dim=1)
    agg = aggregate_induction(token_losses(CopierModel(), ids), L, 1)
    assert agg["loss_second"] < 1e-6
    assert agg["loss_first"] == pytest.approx(math.log(VOCAB), rel=1e-6)
    assert agg["induction_gain"] == pytest.approx(math.log(VOCAB), rel=1e-5)
    assert agg["induction_gain_normalized"] == pytest.approx(1.0, abs=1e-5)
    assert all(g > 4.0 for g in agg["per_seq_gain"])
    # j = 0 of the second copy is not inducible: stays at ln V in the curve
    assert agg["curve_second"][0] == pytest.approx(math.log(VOCAB), rel=1e-6)


def test_per_position_bookkeeping_exact():
    L, off = 6, 1
    ids = build_repeat_batch(3, L, 1, VOCAB, seed=9, prepend_id=0)  # S = 13
    # predicting position t scores target t+1. Target A_first[2] = index 3
    # (pos 2); target A_second[4] = index off+L+4 = 11 (pos 10).
    losses = token_losses(OracleAt({2, 10}), ids)
    first, second = split_halves(losses, L, off)
    lnv = math.log(VOCAB)
    agg = aggregate_induction(losses, L, off)
    for j in range(L):
        exp_f = 0.0 if j == 2 else lnv
        exp_s = 0.0 if j == 4 else lnv
        assert agg["curve_first"][j] == pytest.approx(exp_f, abs=1e-6)
        assert agg["curve_second"][j] == pytest.approx(exp_s, abs=1e-6)
    assert torch.equal(first, losses[:, 1:7]) and torch.equal(second,
                                                              losses[:, 7:13])
    # both halves aggregate over j = 1..5: one zero in each -> gain 0
    assert agg["induction_gain"] == pytest.approx(0.0, abs=1e-9)
    # an oracle only on the second copy gives gain = lnv / (L-1)
    agg2 = aggregate_induction(token_losses(OracleAt({10}), ids), L, off)
    assert agg2["induction_gain"] == pytest.approx(lnv / (L - 1), rel=1e-6)
    # without a prepend, j = 0 of the first copy is undefined (NaN in curve)
    ids0 = ids[:, 1:]
    agg0 = aggregate_induction(token_losses(UniformModel(), ids0), L, 0)
    assert agg0["curve_first"][0] is None
    assert agg0["induction_gain"] == pytest.approx(0.0, abs=1e-9)


def _tiny(attn_res: bool) -> torch.nn.Module:
    from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig
    torch.manual_seed(0)
    kw = dict(attn_res=True, attn_res_boundaries=[0, 1]) if attn_res else {}
    return LuxiaBaseModel(LuxiaModelConfig(
        hidden_size=64, num_layers=2, num_attention_heads=2, num_kv_heads=1,
        head_dim=32, intermediate_size=128, vocab_size=VOCAB,
        max_position_embeddings=256, attn_impl="sdpa",  # "auto" -> no CPU kernel
        use_liger=False, **kw)).eval()


@pytest.mark.parametrize("attn_res", [False, True])
def test_integration_real_model(attn_res):
    model = _tiny(attn_res)
    L = 20
    ids = build_repeat_batch(6, L, 0, VOCAB, exclude=[0], seed=1, prepend_id=0)
    losses = token_losses(model, ids, batch_size=4)
    agg = aggregate_induction(losses, L, 1, n_candidates=VOCAB - 1)
    for k in ("loss_first", "loss_second", "induction_gain",
              "induction_gain_normalized", "induction_gain_se"):
        assert np.isfinite(agg[k]), k
    # random init: near-uniform predictions, no copying
    assert abs(agg["loss_first"] - math.log(VOCAB)) < 1.0
    assert abs(agg["induction_gain"]) < 0.5
    assert len(agg["curve_first"]) == L and len(agg["per_seq_gain"]) == 6
    # losses match the model's own labels path (column-shifted mean CE)
    with torch.inference_mode():
        ref = model(ids, labels=ids)["loss"]
    assert float(losses[:, 1:].mean()) == pytest.approx(float(ref), rel=1e-4)

    if not attn_res:
        with pytest.raises(ValueError):
            prefix_matching_scores(model, ids, L, 1)
        return
    scores = prefix_matching_scores(model, ids, L, 1, batch_size=4)
    assert scores.shape == (2, 2)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    summ = summarize_heads(scores, top_k=3)
    assert len(summ["top"]) == 3 and summ["max"] == pytest.approx(scores.max())


def test_positive_control_trained_copier():
    """Falsifier for both channels: a 2-layer AttnRes model trained on
    [A][A] repeats must form induction — behavioral gain near-perfect and the
    prefix-matching mass in layer 1 (layer 0 = previous-token layer)."""
    model = _tiny(attn_res=True).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(150):
        ids = build_repeat_batch(32, 16, 1, VOCAB, seed=1000 + step,
                                 prepend_id=0)
        loss = model(ids, labels=ids)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    ids = build_repeat_batch(16, 16, 1, VOCAB, seed=0, prepend_id=0)
    agg = aggregate_induction(token_losses(model, ids), 16, 1)
    assert agg["induction_gain_normalized"] > 0.8
    scores = prefix_matching_scores(model, ids, 16, 1)
    assert scores[1].max() > 0.5 and scores[0].max() < 0.2


def test_prefix_matching_reads_the_right_cells(monkeypatch):
    """Hand-built attention: all mass from query q onto key q - L + 1 exactly
    in layer 0 head 1 -> that head scores 1, everything else 0."""
    L, off, S = 5, 1, 11
    B, nh = 2, 2
    w0 = torch.zeros(B, nh, S, S)
    for j in range(L - 1):
        w0[:, 1, off + L + j, off + j + 1] = 1.0
    w1 = torch.zeros(B, nh, S, S)
    w1[:, :, :, 0] = 1.0   # sink head: attends BOS only

    import kotodama.model.capture as capture
    monkeypatch.setattr(capture, "instrumented_forward",
                        lambda m, ids: (None, {"attn": [w0[:len(ids)],
                                                        w1[:len(ids)]]}))

    class Cfg:
        attn_res = True

    class Fake:
        config = Cfg()

    ids = torch.zeros(B, S, dtype=torch.long)
    scores = prefix_matching_scores(Fake(), ids, L, off, batch_size=1)
    assert scores.shape == (2, nh)
    assert scores[0, 1] == pytest.approx(1.0)
    assert scores[0, 0] == 0 and (scores[1] == 0).all()
    assert summarize_heads(scores)["top"][0] == {"layer": 0, "head": 1,
                                                 "score": 1.0}
