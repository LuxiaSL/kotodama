"""The geo-monitor probe draw must not advance the training stream.

train.py builds the probe from the shared dataset on rank 0 only; before
2026-09-27 that skipped rank 0 ahead of its peers, and resuming from the
(rank-0) data position shifted every rank (the 3B's offset-1920 quirk).
The fix snapshots dataset state around the draw — this pins the dataset
behaviour that fix relies on."""
from __future__ import annotations

import numpy as np
import torch

from kotodama.data.dataset import TokenizedDataset


def _dataset(tmp_path, seed=7):
    path = tmp_path / "toks.bin"
    np.arange(64 * 16, dtype=np.uint16).tofile(path)
    return TokenizedDataset(path=str(path), seq_len=16, rank=0, world_size=1, seed=seed)


def _first(ds, n):
    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
    it = iter(loader)
    return [next(it) for _ in range(n)]


def test_probe_draw_then_restore_matches_unprobed_stream(tmp_path):
    reference = _first(_dataset(tmp_path), 3)

    ds = _dataset(tmp_path)
    before = ds.state_dict()
    _first(ds, 5)  # the probe draw
    assert ds.state_dict() != before, "probe really advances the dataset"
    ds.load_state_dict(before)
    after = _first(ds, 3)
    for a, b in zip(reference, after):
        assert torch.equal(a, b)
