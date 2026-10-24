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
