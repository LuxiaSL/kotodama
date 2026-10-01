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
