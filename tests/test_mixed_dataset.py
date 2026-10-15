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


def _mixed(tmp_path, frac=0.03, window=None):
    p = TokenizedDataset(path=_bin(tmp_path / "lang.bin", 4000, 0), seq_len=SEQ, seed=1)
    s = TokenizedDataset(path=_bin(tmp_path / "dyad.bin", 400, 10000), seq_len=SEQ, seed=1)
    return MixedTokenDataset(p, s, frac, window=window)


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


def test_resume_from_unmixed_state_introduces_mix_at_fork(tmp_path):
    """A decay branch can introduce the mix at a fork: resuming a MixedTokenDataset from a plain
    TokenizedDataset state continues the language stream exactly and holds the fraction from there."""
    lang = _bin(tmp_path / "lang.bin", 4000, 0)
    plain = TokenizedDataset(path=lang, seq_len=SEQ, seed=1)
    it = iter(plain)
    for _ in range(250):
        next(it)
    fork_state = plain.state_dict()
    cont = TokenizedDataset(path=lang, seq_len=SEQ, seed=1)
    cont.load_state_dict(fork_state)
    expected_lang = [tuple(t.tolist()) for t in itertools.islice(iter(cont), 500)]

    mixed = _mixed(tmp_path)
    mixed.load_state_dict(fork_state)
    out = list(itertools.islice(iter(mixed), 515))
    got_lang = [tuple(t.tolist()) for t in out if _src(t) == 0]
    assert len(got_lang) >= 490 and got_lang == expected_lang[:len(got_lang)]  # language continues exactly from the fork
    n_dyad = sum(_src(t) for t in out)
    assert abs(n_dyad - 0.03 * len(out)) <= 1  # fraction holds from the resume point
    state = mixed.state_dict()  # and the branch's own checkpoints are mixed-format
    assert set(state) == {"primary", "secondary", "n_total", "n_secondary", "n_window"}


def test_window_confines_mix_exactly(tmp_path):
    """Inside [start, end) the secondary holds the fraction of the window; outside, primary only."""
    seq = [_src(t) for t in itertools.islice(iter(_mixed(tmp_path, 0.4, window=(500, 1500))), 3000)]
    assert sum(seq[:500]) == 0 and sum(seq[1500:]) == 0
    assert sum(seq[500:1500]) == 400                       # exactly 40% of the 1000-sequence window
    for k in (100, 500, 900):                               # evenly spread inside the window
        assert abs(sum(seq[500:500 + k]) - 0.4 * k) <= 1


def test_no_window_is_byte_identical_to_unwindowed(tmp_path):
    a = [tuple(t.tolist()) for t in itertools.islice(iter(_mixed(tmp_path)), 800)]
    b = [tuple(t.tolist()) for t in itertools.islice(iter(_mixed(tmp_path, window=None)), 800)]
    assert a == b


def test_window_resume_mid_window_is_exact(tmp_path):
    ref = [tuple(t.tolist()) for t in itertools.islice(iter(_mixed(tmp_path, 0.4, window=(200, 900))), 1200)]
    ds = _mixed(tmp_path, 0.4, window=(200, 900))
    head = [tuple(t.tolist()) for t in itertools.islice(iter(ds), 550)]
    ds2 = _mixed(tmp_path, 0.4, window=(200, 900))
    ds2.load_state_dict(ds.state_dict())
    assert head + [tuple(t.tolist()) for t in itertools.islice(iter(ds2), 650)] == ref


def test_pre_window_state_loads_as_unwindowed(tmp_path):
    ds = _mixed(tmp_path)
    list(itertools.islice(iter(ds), 300))
    state = ds.state_dict()
    del state["n_window"]                                   # a checkpoint written before windows existed
    ds2 = _mixed(tmp_path)
    ds2.load_state_dict(state)
    assert ds2._n_window == 300


def test_bad_window_rejected(tmp_path):
    with pytest.raises(ValueError):
        _mixed(tmp_path, 0.4, window=(10, 10))


def test_mix_window_tokens_to_sequences():
    from kotodama.training.train import mix_window_seqs
    assert mix_window_seqs(None, 4096) is None
    assert mix_window_seqs("0:2.5B", 2048 * 8) == (0, 2_500_000_000 // (2048 * 8))
    assert mix_window_seqs("2.5B:5B", 2048 * 8) == (2_500_000_000 // 16384, 5_000_000_000 // 16384)
    with pytest.raises(ValueError):
        mix_window_seqs("5B:2.5B", 16384)
    with pytest.raises(ValueError):
        mix_window_seqs("2.5B", 16384)
