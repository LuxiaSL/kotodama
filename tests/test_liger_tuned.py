"""liger_rmsnorm_tuned selects kotodama.model.liger_tuned.LigerRMSNormTuned (Liger's RMSNorm kernels with tuned
launches) and falls back to LigerRMSNorm when liger's internals cannot be imported. GPU parity and timing vs Liger's
own launch: scripts/benchmark/rmsnorm_roofline.py and scripts/utils/liger_tuned_parity.py."""

from __future__ import annotations

import pytest

from kotodama.model import liger_tuned, llama
from kotodama.model.llama import KotodamaModelConfig, _select_norm_class


@pytest.mark.skipif(not llama._LIGER_AVAILABLE, reason="liger-kernel not installed")
def test_selected_only_when_asked():
    assert _select_norm_class(KotodamaModelConfig(use_liger=True)) is llama.LigerRMSNorm
    tuned = _select_norm_class(KotodamaModelConfig(use_liger=True, liger_rmsnorm_tuned=True))
    if liger_tuned.AVAILABLE:
        assert tuned is liger_tuned.LigerRMSNormTuned and issubclass(tuned, llama.LigerRMSNorm)
    else:
        assert tuned is llama.LigerRMSNorm


def test_needs_liger():
    assert _select_norm_class(KotodamaModelConfig(liger_rmsnorm_tuned=True)) is llama.RMSNorm


def test_fallback_when_internals_missing(monkeypatch):
    if not llama._LIGER_AVAILABLE:
        pytest.skip("liger-kernel not installed")
    monkeypatch.setattr(liger_tuned, "AVAILABLE", False)
    assert _select_norm_class(KotodamaModelConfig(use_liger=True, liger_rmsnorm_tuned=True)) is llama.LigerRMSNorm


@pytest.mark.skipif(not llama._LIGER_AVAILABLE, reason="liger-kernel not installed")
def test_scope_qk_tunes_only_the_qk_norms():
    from kotodama.model.llama import GQAttention, TransformerBlock

    cfg = KotodamaModelConfig(use_liger=True, liger_rmsnorm_tuned=True, liger_rmsnorm_tuned_scope="qk",
                              hidden_size=128, num_attention_heads=2, num_kv_heads=2, head_dim=64,
                              intermediate_size=256)
    tuned = liger_tuned.LigerRMSNormTuned if liger_tuned.AVAILABLE else llama.LigerRMSNorm
    assert _select_norm_class(cfg, site="qk") is tuned
    assert _select_norm_class(cfg) is llama.LigerRMSNorm
    blk = TransformerBlock(cfg)
    assert type(blk.attn_norm) is llama.LigerRMSNorm and type(blk.ffn_norm) is llama.LigerRMSNorm
    assert isinstance(blk.attn, GQAttention)
    assert type(blk.attn.q_norm) is tuned and type(blk.attn.k_norm) is tuned


def test_scope_all_is_the_default_and_bad_scope_raises():
    assert KotodamaModelConfig().liger_rmsnorm_tuned_scope == "all"
    with pytest.raises(ValueError):
        KotodamaModelConfig(liger_rmsnorm_tuned_scope="hidden")
