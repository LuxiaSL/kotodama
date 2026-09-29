"""The loader must not undo the assembler's shuffle.

curation/assemble_shuffle.py writes input chunk i to position perm[i] with
perm = RandomState(seed).permutation(N). A single-rank TokenizedDataset that shuffled
arange(N) with RandomState(seed + epoch) reproduced perm exactly at epoch 0 and read the
corpus back in source order (1-GPU runs then memorized their stream; 2026-09-28)."""
from __future__ import annotations

import numpy as np

from kotodama.data.dataset import TokenizedDataset

SEQ, N = 8, 500


def test_single_rank_loader_does_not_invert_assembly_shuffle(tmp_path):
    seed = 42
    source = np.repeat(np.arange(N, dtype=np.uint16), SEQ).reshape(N, SEQ)  # chunk i = all i
    perm = np.random.RandomState(seed).permutation(N)
    out = np.empty_like(source)
    out[perm] = source  # assembler: input chunk i -> output position perm[i]
    path = tmp_path / "train.bin"
    out.tofile(path)
    ds = TokenizedDataset(path=str(path), seq_len=SEQ, rank=0, world_size=1, seed=seed)
    order = [int(t[0]) for _, t in zip(range(N), iter(ds))]
    assert sorted(order) == list(range(N))  # still a full epoch, no repeats
    adjacent = sum(1 for a, b in zip(order, order[1:]) if b == a + 1)
    assert adjacent < 10, f"loader reads {adjacent} consecutive source chunks — shuffle undone"
