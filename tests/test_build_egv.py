"""scripts/steer/build_egv.py — the EGV builder takes its gradient in the serving engine's write space.

Gates (CPU, tiny fp32 LOOPED model, MATH SDPA):
  * the site leaf receives a nonzero gradient, fires at exactly one effective position (loop repeats share
    attn/ffn but the wrapper sits on the per-position block), and is removed on exit;
  * the leaf IS the engine's steer row: a teacher-forced ``inject`` of the block-persistent write equals a
    steered DecodeEngine decode step, and the leaf gradient predicts the engine's finite-difference response;
  * the bank written from ``build_site`` loads through ``load_steer_banks`` with the expected keys/shapes/sites,
    its vectors are unit, the ⊥ forms are orthogonal to V7, and the random controls stay in the band.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend

from kotodama.model.decode_engine import DecodeEngine
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
from kotodama.serve.steering import load_steer_banks

_SPEC = importlib.util.spec_from_file_location(
    "build_egv", Path(__file__).resolve().parents[1] / "scripts" / "steer" / "build_egv.py")
egv = importlib.util.module_from_spec(_SPEC)
sys.modules["build_egv"] = egv
_SPEC.loader.exec_module(egv)

# 1 prelude + a 2-layer core run 3x (positions 1..6) + 1 coda = 8 effective, 4 unique; a boundary at every pass.
CFG = dict(hidden_size=64, num_layers=8, num_attention_heads=4, num_kv_heads=2, head_dim=16,
           intermediate_size=128, vocab_size=211, max_position_embeddings=128, attn_impl="sdpa",
           attn_res=True, attn_res_boundaries=[0, 1, 3, 5, 7], use_liger=False, z_loss_weight=0.0,
           loop_start=1, loop_len=2, loop_times=3)
SITE = 4  # pass 2 of the core (shares weights with 2 and 6); steer rows (4, 5, 7)


def _model(seed: int = 20261024) -> KotodamaModel:
    torch.manual_seed(seed)
    m = KotodamaModel(KotodamaModelConfig(**CFG)).eval()
    # random routing queries (zero-init would make every route a uniform average)
    with torch.no_grad():
        for q, _n in m._attn_res_query_norm_pairs():
            q.normal_(0, 0.5)
    m.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def model() -> KotodamaModel:
    return _model()


def _ids(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, CFG["vocab_size"], (n,), generator=g)


def test_leaf_gets_gradient_at_one_position(model):
    ids = _ids(20, 1)
    calls = []
    with egv.site_leaf(model, SITE) as box:
        orig = model.layers[SITE].attn_sublayer

        def counting(*a, **k):
            calls.append(1)
            return orig(*a, **k)

        model.layers[SITE].attn_sublayer = counting
        with torch.enable_grad():
            logits = model(ids[None])["logits"][0].float()
            F = egv.f_entropy(logits, ids, 8)
            (g,) = torch.autograd.grad(F, box["h"])
        model.layers[SITE].attn_sublayer = orig
    assert len(calls) == 1
    assert box["h"].shape == (1, 20, CFG["hidden_size"])
    assert torch.isfinite(g).all() and g[0, 8:].abs().sum() > 0
    assert "attn_sublayer" not in model.layers[SITE].__dict__  # wrapper removed
    # the leaf is detached from upstream: nothing before it is in its graph
    assert box["h"].grad_fn is None and box["h"].is_leaf
    # unwrapped forward is unchanged by having been wrapped
    with torch.no_grad():
        a = model(ids[None])["logits"]
        with egv.site_leaf(model, SITE):
            b = model(ids[None])["logits"]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def _engine(m: KotodamaModel) -> DecodeEngine:
    return DecodeEngine(m, max_seq_len=64, dtype=torch.float32, prefill_backends=[SDPBackend.MATH],
                        steer_enabled=True)


def test_inject_matches_engine_and_gradient_predicts_it(model):
    ids = _ids(17, 2)
    P = 16  # decode step consumes ids[16]; its logits are the last row
    g = torch.Generator().manual_seed(5)
    v = torch.randn(CFG["hidden_size"], generator=g)
    v /= v.norm()
    eng = _engine(model)

    # (a) block-persistent write: engine decode step == teacher-forced inject at positions >= P
    scale = 0.7
    eng.reset()
    eng.prefill(ids[None, :P])
    eng.set_steer([(SITE, v, scale)])
    eng_logits = eng.step_logits(ids[None, P:P + 1].clone())[0, -1].clone()
    eng.set_steer(None)
    with torch.no_grad(), egv.inject(model, [(SITE, v, scale)], start=P):
        tf = model(ids[None])["logits"][0, -1]
    torch.testing.assert_close(eng_logits, tf, rtol=1e-4, atol=1e-4)
    with torch.no_grad():
        plain = model(ids[None])["logits"][0, -1]
    assert (plain - tf).abs().max() > 1e-3  # the write did something

    # (b) single-row write (no persistence): ∇_leaf F · v == engine finite difference along v (fp64)
    model = _model().double()
    v = v.double()
    eng = DecodeEngine(model, max_seq_len=64, dtype=torch.float64, prefill_backends=[SDPBackend.MATH],
                       steer_enabled=True)
    def last_entropy(lg: torch.Tensor) -> torch.Tensor:
        lp = torch.log_softmax(lg.double(), -1)
        return -(lp.exp() * lp).sum()

    eps = 1e-6  # the tiny random model is strongly curved; fp64 keeps the O(eps^2) term below 1e-5

    def eng_F(sign: float) -> float:
        eng.reset()
        eng.prefill(ids[None, :P])
        with torch.inference_mode():
            eng.steer_buf.zero_()
            eng.steer_buf[SITE] = sign * eps * v
        eng._steer_active = True
        out = eng.step_logits(ids[None, P:P + 1].clone())[0, -1].clone()
        eng.set_steer(None)
        return float(last_entropy(out))

    fd = (eng_F(+1) - eng_F(-1)) / (2 * eps)
    with egv.site_leaf(model, SITE) as box, torch.enable_grad():
        lg = model(ids[None])["logits"][0, -1]
        (grad,) = torch.autograd.grad(last_entropy(lg), box["h"])
    an = float(grad[0, -1].double() @ v.double())
    assert abs(an) > 1e-4
    assert abs(fd - an) <= 1e-5 * abs(an), (fd, an)


def test_functionals_selfrep_matches_loop_reference():
    torch.manual_seed(0)
    T, V, p = 14, 23, 5
    ids = torch.randint(0, V, (T,))
    ids[7] = ids[1]  # a generated token that is also in the prompt (excluded)
    ids[9] = ids[6]  # a repeat
    logits = torch.randn(T, V)
    prompt = set(ids[:p].tolist())
    terms = []
    for t in range(p - 1, T - 1):
        prior = sorted({int(x) for x in ids[p:t + 1].tolist() if int(x) not in prompt})
        if prior:
            terms.append(torch.softmax(logits[t], -1)[prior].sum())
    ref = torch.stack(terms).mean()
    torch.testing.assert_close(egv.f_selfrep(logits, ids, p), ref)


def test_functionals_loglik_typdev_match_loop_reference():
    torch.manual_seed(1)
    T, V, p = 12, 19, 4
    ids = torch.randint(0, V, (T,))
    logits = torch.randn(T, V)
    ll, dev = [], []
    for t in range(p - 1, T - 1):
        lp = torch.log_softmax(logits[t], -1)
        H = -(lp.exp() * lp).sum()
        ll.append(lp[ids[t + 1]])
        dev.append((-lp[ids[t + 1]] - H).abs())
    torch.testing.assert_close(egv.f_loglik(logits, ids, p), torch.stack(ll).mean())
    torch.testing.assert_close(egv.f_typdev(logits, ids, p), torch.stack(dev).mean())


def test_perp2_removes_entropy_and_repetition(model):
    seqs = [egv.Seq(_ids(28, 10 + i), 8, i) for i in range(10)]
    res = egv.build_site(model, seqs[:6], seqs[6:], SITE, ["entropy", "selfrep", "loglik", "typdev"], (2, 40))
    s = 2 * SITE
    v7, rep = res.bank[f"V7_s{s}"], res.bank[f"Vrepperp_s{s}"]
    for n in ("Vll", "Vtyp"):
        u = res.bank[f"{n}perp2_s{s}"]
        assert abs(float(np.linalg.norm(u)) - 1) < 1e-5
        assert abs(float(u @ v7)) < 1e-5 and abs(float(u @ rep)) < 1e-5
        meta = res.meta["perp"][n]
        assert 0 < meta["perp2_residual_fraction"] <= meta["perp_residual_fraction"] + 1e-9
        assert -1 - 1e-6 <= meta["cos_perp_to_Vrepperp"] <= 1 + 1e-6
    assert "Vrepperp2_s%d" % s not in res.bank


def test_bank_roundtrip_through_server_loader(model, tmp_path):
    seqs = [egv.Seq(_ids(28, 10 + i), 8, i) for i in range(10)]
    names = ["entropy", "selfrep", "margin"]
    band = (2, 40)
    res = egv.build_site(model, seqs[:6], seqs[6:], SITE, names, band)
    out = tmp_path / "egv.npz"
    np.savez(out, **res.bank)
    bank, meds, aliases = load_steer_banks([str(out)], CFG["num_layers"], CFG["hidden_size"], torch.device("cpu"))
    s = 2 * SITE
    expect = {f"{n}_s{s}" for n in ("V7", "Vrep", "Vmargin", "Vrepperp", "Vmarginperp", "Vmarginperp2",
                                    "Rband1", "Rband2", "Rband3")}
    assert set(bank) == expect
    assert all(site == SITE for site, _ in bank.values())
    assert set(meds) == {SITE} and meds[SITE] == pytest.approx(res.meta["median_norm"]) and meds[SITE] > 0
    assert "control" in aliases
    for _k, (_site, vec) in bank.items():
        assert vec.shape == (CFG["hidden_size"],)
        assert float(vec.norm()) == pytest.approx(1.0, abs=1e-5)
    v7 = bank[f"V7_s{s}"][1]
    for k in (f"Vrepperp_s{s}", f"Vmarginperp_s{s}"):
        assert abs(float(bank[k][1] @ v7)) < 1e-5
    # ⊥ is the Gram-Schmidt residual of the raw member
    vrep = bank[f"Vrep_s{s}"][1]
    c = float(vrep @ v7)
    r = vrep - c * v7
    torch.testing.assert_close(bank[f"Vrepperp_s{s}"][1], r / r.norm(), rtol=0, atol=1e-5)
    assert res.meta["perp"]["Vrep"]["cos_to_V7"] == pytest.approx(c, abs=1e-5)
    # every vector lies in span(U): projecting onto the band changes nothing
    spec = egv.spectrum(model, seqs[:6], SITE)
    U = torch.from_numpy(spec.evecs[:, band[0]:band[1]]).float()
    for _k, (_site, vec) in bank.items():
        torch.testing.assert_close(U @ (U.T @ vec), vec, rtol=0, atol=1e-4)
    d = res.meta["diagnostics"]
    assert set(d) == {"V7", "Vrep", "Vmargin"}
    for v in d.values():
        assert 0 < v["band_mass"] <= 1 and -1 <= v["pairwise_cos"] <= 1 and v["agree"].endswith("/4")


def test_sampler_batched_cached_and_stop_trimmed(model):
    g = torch.Generator().manual_seed(3)
    prompt = _ids(9, 4).tolist()
    rows = egv.sample_batch(model, prompt, 3, 12, 0.7, 1.2, frozenset({0}), g)
    assert len(rows) == 3 and all(len(r) <= 12 and 0 not in r for r in rows)
    # temperature -> 0 reproduces greedy decoding of the full (uncached) forward
    rows0 = egv.sample_batch(model, prompt, 1, 6, 1e-6, 1.0, frozenset({0}), g)[0]
    seq = list(prompt)
    with torch.no_grad():
        for _ in range(len(rows0)):
            seq.append(int(model(torch.tensor([seq]))["logits"][0, -1].argmax()))
    assert rows0 == seq[len(prompt):]
