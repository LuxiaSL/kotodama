"""hm_geodiff: common-step alignment and the per-family gist."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts/eval"
sys.path.insert(0, str(_p))
_spec = importlib.util.spec_from_file_location("hm_geodiff", _p / "hm_geodiff.py")
gd = importlib.util.module_from_spec(_spec)
sys.modules["hm_geodiff"] = gd
_spec.loader.exec_module(gd)


def test_common_steps_intersects_and_clips():
    a = {"k0": {0: 1.0, 50: 1.0, 100: 1.0}, "k1": {150: 1.0}}
    b = {"k0": {0: 2.0, 100: 2.0, 150: 2.0, 200: 2.0}}
    assert gd.common_steps(a, b, None) == [0, 100, 150]
    assert gd.common_steps(a, b, 120) == [0, 100]


def test_grid_fills_missing_with_nan():
    g = gd.grid({"k0": {0: 1.0}, "k1": {50: 2.0}}, ["k0", "k1"], [0, 50])
    assert g[0][0] == 1.0 and math.isnan(g[0][1]) and math.isnan(g[1][0]) and g[1][1] == 2.0


def test_gist_uses_last_finite_step_and_ranks_moves():
    ga = [[1.0, 1.0, math.nan], [2.0, 2.0, math.nan], [3.0, 3.0, math.nan]]
    gb = [[1.0, 1.5, 9.0], [2.0, 2.0, 9.0], [3.0, 1.0, 9.0]]
    g = gd.gist("f/{i}", [0, 1, 2], [0, 50, 100], ga, gb, k=2)
    assert g.step == 50
    assert [t[0] for t in g.top] == [2, 0]          # |Δ| 2.0 then 0.5
    assert math.isclose(g.rel_mean_abs_delta, (0.5 + 0 + 2.0) / 3 / (3.0 - 1.0))
    assert gd.gist("f/{i}", [0], [0], [[math.nan]], [[1.0]]) is None
