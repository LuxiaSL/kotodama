"""hm_portrait pure helpers: family matching, alignment, trajectory gist, Spearman."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "hm_portrait", Path(__file__).resolve().parents[1] / "scripts/eval/hm_portrait.py")
hp = importlib.util.module_from_spec(_spec)
sys.modules["hm_portrait"] = hp  # dataclasses resolve their module during class creation
_spec.loader.exec_module(hp)


def test_family_does_not_cross_match_std_sibling():
    keys = ["norm_gain/layer_3/k_norm", "norm_gain_std/layer_3/k_norm",
            "norm_gain/layer_12/k_norm", "norm_gain/layer_3/q_norm"]
    assert hp.layers_for("norm_gain/layer_{i}/k_norm", keys) == [3, 12]
    assert hp.layers_for("norm_gain_std/layer_{i}/k_norm", keys) == [3]
    with pytest.raises(ValueError):
        hp.family_regex("norm_gain/k_norm")


def test_align_keeps_shared_steps_in_order():
    assert hp.align({0: 1.0, 50: 2.0, 100: 3.0}, {50: 9.0, 100: 8.0, 150: 7.0}) == [
        (50, 2.0, 9.0), (100, 3.0, 8.0)]


def test_gist_trajectory_slope_and_settling():
    moving = [(s, 1 + s / 100, 7 - s / 50) for s in range(0, 101, 10)]  # straight line
    g = hp.gist_trajectory(14, moving)
    assert g.start == (1.0, 7.0) and g.end == (2.0, 5.0)
    assert g.recent_slope == pytest.approx(-2.0)
    settled = [(s, min(1 + s / 25, 2.0), 1.0) for s in range(0, 101, 10)]  # stops at step 25
    assert hp.gist_trajectory(0, settled).last_quarter_share == pytest.approx(0.0)
    assert hp.gist_trajectory(0, moving[:1]) is None


def test_spearman():
    assert hp.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert hp.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert hp.spearman([1, 2], [1, 2]) is None
