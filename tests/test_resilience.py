"""Resilience read: spike detection (one event per burst) and rank-recovery timing."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts/eval"
sys.path.insert(0, str(_p))
_spec = importlib.util.spec_from_file_location("resilience", _p / "resilience.py")
rs = importlib.util.module_from_spec(_spec)
sys.modules["resilience"] = rs
_spec.loader.exec_module(rs)


def _loss():
    s = {i: 3.0 - 0.0001 * i + (0.01 if i % 2 else -0.01) for i in range(0, 1000, 10)}
    s[500] = 3.6   # a spike
    s[510] = 3.4   # same burst
    return s


def test_one_event_per_burst():
    sp = rs.find_spikes(_loss(), k=6, window=20)
    assert [x[0] for x in sp] == [500]


def test_recovery_and_never():
    rank = {i: 1000.0 for i in range(0, 1000, 100)}
    rank.update({600: 900.0, 700: 995.0, 800: 1000.0})
    r0, rmin, rec = rs.recovery(rank, 500, horizon=400, tol=0.01)
    assert (r0, rmin, rec) == (1000.0, 900.0, 700)
    rank2 = {i: 1000.0 for i in range(0, 600, 100)} | {600: 900.0, 700: 920.0, 800: 930.0}
    assert rs.recovery(rank2, 500, horizon=400, tol=0.01)[2] is None
