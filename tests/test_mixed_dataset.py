"""MixedTokenDataset: exact token fraction, deterministic, resumable mid-stream."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from kotodama.data.dataset import MixedTokenDataset, TokenizedDataset

SEQ = 16


def _bin(path, n, fill_base):
    np.concatenate([np.full(SEQ, fill_base + i, dtype=np.uint16) for i in range(n)]).tofile(path)
    return str(path)


def _mixed(tmp_path, frac=0.03):
    p = TokenizedDataset(path=_bin(tmp_path / "lang.bin", 4000, 0), seq_len=SEQ, seed=1)
    s = TokenizedDataset(path=_bin(tmp_path / "dyad.bin", 400, 10000), seq_len=SEQ, seed=1)
    return MixedTokenDataset(p, s, frac)


def _src(t):  # 0 = language, 1 = dyadic (by the fill values)
    return int(int(t[0]) >= 10000)


def test_exact_fraction_and_spacing(tmp_path):
    seq = [_src(t) for t in itertools.islice(iter(_mixed(tmp_path)), 3000)]
    assert sum(seq) == 90  # exactly 3% of 3000
    for k in (100, 1000, 2000):  # never more than one sequence off the target
        assert abs(sum(seq[:k]) - 0.03 * k) <= 1


def test_resume_mid_stream_is_exact(tmp_path):
    ref = [tuple(t.tolist()) for t in itertools.islice(iter(_mixed(tmp_path)), 700)]
    ds = _mixed(tmp_path)
    head = [tuple(t.tolist()) for t in itertools.islice(iter(ds), 300)]
    state = ds.state_dict()
    ds2 = _mixed(tmp_path)
    ds2.load_state_dict(state)
    tail = [tuple(t.tolist()) for t in itertools.islice(iter(ds2), 400)]
    assert head + tail == ref


def test_rejects_bad_fraction(tmp_path):
    p = TokenizedDataset(path=_bin(tmp_path / "a.bin", 10, 0), seq_len=SEQ)
    with pytest.raises(ValueError):
        MixedTokenDataset(p, p, 0.0)
