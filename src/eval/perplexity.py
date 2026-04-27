"""Held-out perplexity computation for model evaluation.

Consolidates PPL computation from eval_lang_full.py and other eval scripts.
Uses memmap-based eval data loading for memory efficiency.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.model.llama import LuxiaBaseModel

logger = logging.getLogger(__name__)


def compute_perplexity(
    model: LuxiaBaseModel,
    eval_data_path: Path | str,
    seq_len: int = 2048,
    batch_size: int = 4,
    max_seqs: int = 400,
    device: str | None = None,
) -> dict[str, float]:
    """Compute held-out perplexity on a tokenized eval dataset.

    Loads eval data as a uint16 memmap and computes cross-entropy loss
    over non-overlapping sequences.

    Args:
        model: The model in eval mode.
        eval_data_path: Path to the tokenized eval data (.bin, uint16 memmap).
        seq_len: Sequence length for each evaluation chunk.
        batch_size: Number of sequences per forward pass.
        max_seqs: Maximum total sequences to evaluate.
        device: Device to run on. Defaults to the model's device.

    Returns:
        Dict with keys:
            - loss: Average cross-entropy loss.
            - perplexity: exp(loss).
            - n_tokens: Total tokens evaluated.
            - n_seqs: Number of sequences processed.
            - time_s: Elapsed time in seconds.

    Raises:
        FileNotFoundError: If eval data file doesn't exist.
    """
    eval_data_path = Path(eval_data_path)
    if not eval_data_path.exists():
        raise FileNotFoundError(f"Eval data not found: {eval_data_path}")

    if device is None:
        device = str(next(model.parameters()).device)

    tokens = np.memmap(eval_data_path, dtype=np.uint16, mode="r")
    available_seqs = len(tokens) // seq_len
    n_seqs = min(available_seqs, max_seqs)

    logger.info(
        "Computing perplexity: %d seqs × %d tokens (%.1fM eval tokens)",
        n_seqs,
        seq_len,
        n_seqs * seq_len / 1e6,
    )

    total_loss = 0.0
    total_tokens = 0
    seqs_processed = 0
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, n_seqs, batch_size):
            bs = min(batch_size, n_seqs - i)
            input_ids = torch.zeros(bs, seq_len, dtype=torch.long, device=device)

            for j in range(bs):
                offset = (i + j) * seq_len
                chunk = tokens[offset : offset + seq_len].astype(np.int64)
                input_ids[j] = torch.from_numpy(chunk)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids)
                logits = output["logits"] if isinstance(output, dict) else output

            # Shift for next-token prediction: predict token[t+1] from token[t]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()
            seqs_processed += bs

    elapsed = time.time() - t0
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)

    logger.info(
        "Perplexity: loss=%.4f, ppl=%.2f (%d tokens, %.1fs)",
        avg_loss,
        ppl,
        total_tokens,
        elapsed,
    )

    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "n_tokens": total_tokens,
        "n_seqs": seqs_processed,
        "time_s": elapsed,
    }
