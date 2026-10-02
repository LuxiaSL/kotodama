"""nca_profile: the boundary algorithm is the archived one (DD-3B's method of record),
and the profile walk works on a no-AttnRes model and refuses an AttnRes one."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import nca_profile as npf  # noqa: E402

from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig  # noqa: E402


def _archived_suggest():
    """suggest_boundaries exactly as committed at tag pre-consolidation-2026-09."""
    try:
        src = subprocess.run(["git", "show", "pre-consolidation-2026-09:scripts/nca_profile.py"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("archived tag not available")
    start = src.index("def suggest_boundaries(")
    end = src.index("\ndef main()", start)
    body = src[start:end]
    body = body[:body.index('    logger.info("Zone-aware')] + "    return boundaries, combined\n"
    ns: dict = {}
    exec(body, ns)
    return ns["suggest_boundaries"]


def test_suggest_boundaries_is_the_archived_algorithm():
    archived = _archived_suggest()
    rng = random.Random(7)
    for _ in range(200):
        n = rng.choice([12, 28, 32])
        prof = {k: [rng.uniform(0, 300) for _ in range(n)]
                for k in ("anisotropy", "attn_entropy_mean", "stable_rank_q", "stable_rank_down")}
        for mbs, mxb in ((2, 6), (1, 7)):
            assert npf.suggest_boundaries(prof, n, mbs, mxb) == archived(prof, n, mbs, mxb)


def _model(attn_res: bool) -> LuxiaBaseModel:
    torch.manual_seed(0)
    cfg = LuxiaModelConfig(hidden_size=64, num_layers=6, num_attention_heads=4, num_kv_heads=2,
                           head_dim=16, intermediate_size=128, vocab_size=10010,
                           max_position_embeddings=64, attn_impl="sdpa", attn_res=attn_res,
                           attn_res_boundaries=[0, 1, 3] if attn_res else None, use_liger=False)
    return LuxiaBaseModel(cfg).eval()


def test_profile_walk_both_rulers():
    m = _model(False)
    probe = torch.randint(0, 10002, (4, 32))
    prof = npf.extract_profile(m, probe, attn_chunk=2)
    assert set(prof) == {"legacy", "true"}
    for r in prof.values():
        assert all(len(r[k]) == 6 for k in npf.PROFILE_KEYS)
    # shared metrics identical across rulers; entropies are the rulers' difference
    assert prof["legacy"]["stable_rank_q"] == prof["true"]["stable_rank_q"]
    assert prof["legacy"]["attn_entropy_mean"] != prof["true"]["attn_entropy_mean"]
    b, _ = npf.suggest_boundaries(prof["true"], 6, 1, 3)
    assert b[0] == 0 and len(b) <= 3


def test_profile_refuses_attnres_model():
    with pytest.raises(SystemExit):
        npf.extract_profile(_model(True), torch.randint(0, 10002, (2, 16)))


# ── 2026-10-01 re-fit additions ───────────────────────────────────────────────

import argparse  # noqa: E402

import fit_boundaries as fb  # noqa: E402


def test_select_boundaries_reduces_to_the_archived_pick():
    rng = random.Random(11)
    for _ in range(200):
        n = rng.choice([12, 28])
        prof = {k: [rng.uniform(0, 300) for _ in range(n)]
                for k in ("anisotropy", "attn_entropy_mean", "stable_rank_q", "stable_rank_down")}
        for mbs, mxb in ((2, 6), (1, 7)):
            b, scores = npf.suggest_boundaries(prof, n, mbs, mxb)
            assert npf.select_boundaries(scores, n, mbs, mxb, forced=(), min_last_block=2) == b


def test_select_boundaries_honours_the_hand_rules():
    rng = random.Random(3)
    for _ in range(100):
        scores = [rng.random() for _ in range(27)]
        b = npf.select_boundaries(scores, 28, min_block_size=2, max_boundaries=7,
                                  forced=(1,), min_last_block=4)
        assert b[:2] == [0, 1] and len(b) <= 7
        assert 28 - b[-1] >= 4
        assert all(y - x >= 1 for x, y in zip(b, b[1:]))
        assert all(y - x >= 2 for x, y in zip(b[1:], b[2:]))   # beyond the forced L0 block


def test_linear_cka_invariances():
    torch.manual_seed(0)
    x = torch.randn(512, 32)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    assert abs(npf.linear_cka(x, x) - 1) < 1e-9
    assert abs(npf.linear_cka(x, 3.0 * x @ q) - 1) < 1e-9      # rotation + isotropic scale
    assert npf.linear_cka(x, torch.randn(512, 32)) < 0.2


def test_settled_at():
    s = [(1, [0, 1, 5]), (2, [0, 1, 6]), (3, [0, 1, 6]), (4, [0, 1, 6]), (5, [0, 1, 7])]
    assert fb.settled_at(s, 3) is None            # moved again at the end: not settled
    assert fb.settled_at(s, 3, tol=1) == 2        # 6,6,6,7 within ±1 of the last (7)
    s2 = s[:-1] + [(5, [0, 1, 6]), (6, [0, 1, 6])]
    assert fb.settled_at(s2, 3) == 2
    assert fb.settled_at(s2, 6) is None
    f = fb.cut_frequency(s2)
    assert f[6] == 1.0 and f[0] == 1.0


def test_fits_end_to_end_tiny():
    m = _model(False)
    hid: list = []
    prof = npf.extract_profile(m, torch.randint(0, 10002, (4, 32)), attn_chunk=2,
                               hidden_sink=hid, hidden_rows=64)
    assert len(hid) == 6 and hid[0].shape == (64, 64)
    args = argparse.Namespace(v1_min_block=1, v1_max_boundaries=4, min_block=2, max_boundaries=4,
                              forced=[1], min_last_block=2, min_mid_block=2)
    fits = fb.fits_for(prof, npf.cka_transition_scores(hid), 6, args)
    assert set(fits) == set(fb.METHODS)
    for f in ("rules_true", "rules_cka"):
        assert fits[f]["boundaries"][:2] == [0, 1]


def test_middle_block_rule():
    rng = random.Random(5)
    bound = 0
    for _ in range(200):
        scores = [rng.random() for _ in range(27)]
        free = npf.select_boundaries(scores, 28, 2, 7, forced=(1,), min_last_block=4)
        mid = npf.select_boundaries(scores, 28, 2, 7, forced=(1,), min_last_block=4, min_mid_block=8)
        assert npf.mid_block_size(mid, 28) >= 8
        bound += free != mid
    assert bound > 0  # the rule actually changes picks on some score profiles
    assert npf.mid_block_size([0, 1, 3, 7, 15, 19, 24], 28) == 8   # DD-3B: L7-14 holds L14
