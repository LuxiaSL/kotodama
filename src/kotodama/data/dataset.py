"""
Tokenized text dataset for pretraining.

Reads pre-tokenized data from a flat binary file of uint16 token IDs.
Each sample is a contiguous chunk of `seq_len` tokens. For distributed
training, each rank gets a non-overlapping partition of the data.

Data format:
    Flat binary file of np.uint16 token IDs (2 bytes per token).
    EOS token (id=0) separates documents within the stream.
    Create with curation/tokenize_parallel.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class PackedSample:
    """A packed sequence with document boundary metadata for attention masking."""
    input_ids: torch.Tensor      # (seq_len,) int64
    cu_seqlens: torch.Tensor     # (num_docs+1,) int32 — cumulative doc lengths
    position_ids: torch.Tensor   # (seq_len,) int32 — per-token position (resets per doc)
    max_seqlen: int              # longest doc in this sequence


def compute_doc_boundaries(tokens: np.ndarray, eos_id: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    """Find document boundaries from EOS tokens and compute cu_seqlens + position_ids.

    EOS belongs to the preceding document (it's the end-of-text marker).
    Each new document starts at position 0 for RoPE.

    Args:
        tokens: (seq_len,) token IDs
        eos_id: EOS token ID (default 0 for SmolLM2)

    Returns:
        cu_seqlens: int32 array [0, end_doc1, end_doc2, ..., seq_len]
        position_ids: int32 array of per-token positions (reset at boundaries)
        max_seqlen: length of longest document
    """
    seq_len = len(tokens)
    eos_positions = np.where(tokens == eos_id)[0]

    if len(eos_positions) == 0:
        cu_seqlens = np.array([0, seq_len], dtype=np.int32)
        position_ids = np.arange(seq_len, dtype=np.int32)
        return cu_seqlens, position_ids, seq_len

    # Document boundaries: each doc ends at EOS (inclusive), next starts at eos+1
    doc_starts = np.empty(len(eos_positions) + 1, dtype=np.int32)
    doc_starts[0] = 0
    doc_starts[1:] = eos_positions + 1

    # Filter out empty documents (consecutive EOS tokens)
    # and boundaries that exceed seq_len
    valid = doc_starts < seq_len
    doc_starts = doc_starts[valid]

    # Build cu_seqlens: [start0, start1, ..., seq_len]
    cu_seqlens = np.empty(len(doc_starts) + 1, dtype=np.int32)
    cu_seqlens[:-1] = doc_starts
    cu_seqlens[-1] = seq_len

    # Position IDs: reset to 0 at each document start.
    # Vectorized: global position minus each token's document-start offset.
    doc_lengths = np.diff(cu_seqlens)
    position_ids = (
        np.arange(seq_len, dtype=np.int32)
        - np.repeat(cu_seqlens[:-1], doc_lengths).astype(np.int32)
    )
    max_seqlen = int(doc_lengths.max()) if len(doc_lengths) > 0 else seq_len
    return cu_seqlens, position_ids, max_seqlen


def collate_packed(batch: list[PackedSample]) -> dict[str, torch.Tensor | int]:
    """Collate PackedSamples for flash_attn_varlen_func.

    Flattens batch into a single token stream with concatenated cu_seqlens
    (offsets adjusted per batch element).

    Returns dict with:
        input_ids: (B, S) int64
        cu_seqlens: (total_docs+1,) int32 — flat across entire batch
        position_ids: (B, S) int32
        max_seqlen: int — max doc length across batch
    """
    batch_size = len(batch)
    seq_len = batch[0].input_ids.shape[0]

    input_ids = torch.stack([s.input_ids for s in batch])
    position_ids = torch.stack([s.position_ids for s in batch])

    # Build flat cu_seqlens: offset each sequence's boundaries by seq_idx * seq_len
    parts = [batch[0].cu_seqlens]
    for i in range(1, batch_size):
        # Skip the leading 0, add offset
        parts.append(batch[i].cu_seqlens[1:] + i * seq_len)

    cu_seqlens = torch.cat(parts)
    max_seqlen = max(s.max_seqlen for s in batch)

    return {
        "input_ids": input_ids,
        "cu_seqlens": cu_seqlens,
        "position_ids": position_ids,
        "max_seqlen": max_seqlen,
    }


def build_block_causal_mask(
    cu_seqlens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build block-diagonal causal mask from cu_seqlens (SDPA fallback).

    Returns (B, 1, S, S) float mask: 0 where attention is allowed, -inf elsewhere.
    """
    # Assign each token to a document ID
    total_len = batch_size * seq_len
    doc_ids = torch.zeros(total_len, dtype=torch.int32, device=device)
    for i in range(len(cu_seqlens) - 1):
        doc_ids[cu_seqlens[i]:cu_seqlens[i + 1]] = i

    doc_ids = doc_ids.view(batch_size, seq_len)

    # same_doc[b, i, j] = True if tokens i and j are in the same document
    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)  # (B, S, S)

    # Causal: j <= i
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Combined: attend within same doc, causally
    allowed = same_doc & causal.unsqueeze(0)  # (B, S, S)

    mask = torch.where(allowed, 0.0, float("-inf"))
    return mask.unsqueeze(1)  # (B, 1, S, S)


class TokenizedDataset(IterableDataset):
    """
    Memory-mapped tokenized text dataset for language model pretraining.

    Each sample is a contiguous chunk of ``seq_len`` tokens read from a
    flat binary file.  For distributed training, the token stream is
    partitioned across ranks so each GPU sees non-overlapping data.

    When ``doc_masking=True``, yields :class:`PackedSample` with document
    boundary metadata (cu_seqlens, position_ids) derived from EOS tokens.
    When ``False``, yields plain tensors for backward compatibility.

    Supports checkpointing: call :meth:`state_dict` /
    :meth:`load_state_dict` to save/restore the read position.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        doc_masking: bool = False,
        eos_id: int = 0,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Tokenized data not found: {self.path}")

        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.doc_masking = doc_masking
        self.eos_id = eos_id

        # Memory-map the data (read-only)
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)
        self.total_sequences = self.total_tokens // seq_len

        if self.total_sequences < world_size:
            raise ValueError(
                f"Data has {self.total_sequences} sequences but world_size={world_size}. "
                f"Need at least {world_size * seq_len} tokens."
            )

        # Partition sequences across ranks (drop remainder)
        seqs_per_rank = self.total_sequences // world_size
        self.start_seq = seqs_per_rank * rank
        self.end_seq = seqs_per_rank * (rank + 1)
        self.num_sequences = self.end_seq - self.start_seq

        # Track position for checkpointing
        self._position: int = 0  # index into shuffled order
        self._epoch: int = 0

        logger.info(
            "Rank %d: %d sequences (%.1fM tokens) from %.1fM total, doc_masking=%s",
            rank,
            self.num_sequences,
            self.num_sequences * seq_len / 1e6,
            self.total_tokens / 1e6,
            doc_masking,
        )

    # -- Checkpointing ---------------------------------------------------------

    def state_dict(self) -> dict[str, int]:
        return {"position": self._position, "epoch": self._epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._position = state["position"]
        self._epoch = state["epoch"]

    # -- Iteration -------------------------------------------------------------

    def __iter__(self) -> Iterator[torch.Tensor | PackedSample]:
        indices = self._shuffled_indices(self._epoch)
        idx = self._position

        while True:
            if idx >= len(indices):
                self._epoch += 1
                self._position = 0
                idx = 0
                indices = self._shuffled_indices(self._epoch)

            seq_idx = indices[idx]
            start = seq_idx * self.seq_len
            end = start + self.seq_len

            raw = self.data[start:end]
            tokens = torch.from_numpy(raw.astype(np.int64)).clone()

            self._position = idx + 1
            idx += 1

            if self.doc_masking:
                cu_seqlens, position_ids, max_seqlen = compute_doc_boundaries(
                    raw, eos_id=self.eos_id
                )
                yield PackedSample(
                    input_ids=tokens,
                    cu_seqlens=torch.from_numpy(cu_seqlens),
                    position_ids=torch.from_numpy(position_ids).long(),
                    max_seqlen=max_seqlen,
                )
            else:
                yield tokens

    def __len__(self) -> int:
        """Number of sequences per rank (one epoch)."""
        return self.num_sequences

    def _shuffled_indices(self, epoch: int) -> np.ndarray:
        """Return a deterministically shuffled array of sequence indices for this rank."""
        rng = np.random.RandomState(self.seed + epoch)
        indices = np.arange(self.start_seq, self.end_seq)
        rng.shuffle(indices)
        return indices


class RandomTokenDataset(IterableDataset):
    """
    Generates random token sequences for testing/smoke tests.

    Produces an infinite stream of random token IDs.  Useful for verifying
    the training loop without requiring actual tokenized data.

    When ``doc_masking=True``, inserts EOS tokens at random intervals to
    simulate document boundaries and yields :class:`PackedSample`.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        seed: int = 42,
        rank: int = 0,
        doc_masking: bool = False,
        eos_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.doc_masking = doc_masking
        self.eos_id = eos_id
        self.rng = np.random.RandomState(seed + rank)

    def state_dict(self) -> dict[str, int]:
        return {}

    def load_state_dict(self, state: dict[str, int]) -> None:
        pass

    def __iter__(self) -> Iterator[torch.Tensor | PackedSample]:
        while True:
            tokens_np = self.rng.randint(1, self.vocab_size, size=(self.seq_len,), dtype=np.int64)

            if self.doc_masking:
                # Insert EOS at random positions to simulate ~8 docs per sequence
                n_docs = self.rng.randint(2, 12)
                eos_positions = np.sort(
                    self.rng.choice(self.seq_len - 1, size=min(n_docs, self.seq_len - 1), replace=False)
                )
                tokens_np[eos_positions] = self.eos_id

                cu_seqlens, position_ids, max_seqlen = compute_doc_boundaries(
                    tokens_np.astype(np.uint16), eos_id=self.eos_id
                )
                yield PackedSample(
                    input_ids=torch.from_numpy(tokens_np).clone(),
                    cu_seqlens=torch.from_numpy(cu_seqlens),
                    position_ids=torch.from_numpy(position_ids).long(),
                    max_seqlen=max_seqlen,
                )
            else:
                yield torch.from_numpy(tokens_np).clone()
