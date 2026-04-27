"""
Tokenize a HuggingFace dataset into a flat binary file for pretraining.

Produces a flat file of uint16 token IDs (2 bytes per token) that can be
memory-mapped by ``src.data.dataset.TokenizedDataset``.

Uses batch tokenization for speed — the HuggingFace tokenizer's internal
Rust implementation already parallelizes across CPU cores when given
batches of text.

Usage::

    # Fast batch tokenization (~6B tokens)
    python scripts/tokenize_data.py \
        --dataset HuggingFaceFW/fineweb-edu \
        --name sample-10BT \
        --split train \
        --output data/fineweb_edu_6b.bin \
        --max_tokens 6_000_000_000 \
        --batch_size 5000

    # Verify the output
    python scripts/tokenize_data.py --verify data/fineweb_edu_6b.bin
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def tokenize_dataset(
    dataset_name: str,
    tokenizer_name: str,
    output_path: str | Path,
    split: str = "train",
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    text_field: str = "text",
    batch_size: int = 1000,
) -> int:
    """
    Tokenize a HuggingFace dataset using batch tokenization.

    The HuggingFace tokenizer's Rust backend parallelizes across CPU cores
    automatically when given batches of text. This is simpler and faster
    than Python-level multiprocessing for this workload.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    logger.info(
        "Loading dataset: %s (name=%s, split=%s, streaming)",
        dataset_name, name, split,
    )
    ds = load_dataset(dataset_name, name=name, split=split, streaming=True)

    logger.info(
        "Batch tokenization → %s (batch_size=%d)", output_path, batch_size,
    )

    total_tokens = 0
    docs_processed = 0
    t0 = time.time()

    with open(output_path, "wb") as f:
        batch_texts: list[str] = []

        for doc in ds:
            text = doc.get(text_field, "")
            if text:
                batch_texts.append(text)

            if len(batch_texts) >= batch_size:
                n_written = _tokenize_and_write_batch(
                    tokenizer, batch_texts, f
                )
                total_tokens += n_written
                docs_processed += len(batch_texts)
                batch_texts = []

                if docs_processed % (batch_size * 10) == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "  %d docs, %.1fM tokens, %.0f tok/s",
                        docs_processed,
                        total_tokens / 1e6,
                        total_tokens / max(elapsed, 1e-6),
                    )

                if max_tokens is not None and total_tokens >= max_tokens:
                    logger.info("Reached max_tokens=%d, stopping", max_tokens)
                    break

        # Flush remaining batch
        if batch_texts and (max_tokens is None or total_tokens < max_tokens):
            n_written = _tokenize_and_write_batch(
                tokenizer, batch_texts, f
            )
            total_tokens += n_written
            docs_processed += len(batch_texts)

    # Trim to exact max_tokens if overshot
    if max_tokens is not None and total_tokens > max_tokens:
        output_path_obj = Path(output_path)
        with open(output_path_obj, "r+b") as f:
            f.truncate(max_tokens * 2)  # uint16 = 2 bytes per token
        total_tokens = max_tokens

    elapsed = time.time() - t0
    file_size_mb = output_path.stat().st_size / 1e6
    logger.info(
        "Done: %d docs, %.1fM tokens, %.1f MB, %.1fs (%.0f tok/s)",
        docs_processed,
        total_tokens / 1e6,
        file_size_mb,
        elapsed,
        total_tokens / max(elapsed, 1e-6),
    )

    return total_tokens


def _tokenize_and_write_batch(
    tokenizer, texts: list[str], f
) -> int:
    """
    Batch-tokenize texts and write to file. Returns number of tokens written.

    Uses the tokenizer's batch encoding which parallelizes internally
    via the Rust tokenizers library.
    """
    # Batch encode — the Rust tokenizer parallelizes this across cores
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    total = 0
    for ids in encoded["input_ids"]:
        if ids and max(ids) < 65536:
            arr = np.array(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            total += len(ids)

    return total


def verify_data(path: str | Path) -> None:
    """Quick verification of a tokenized binary file."""
    path = Path(path)
    data = np.memmap(path, dtype=np.uint16, mode="r")
    total_tokens = len(data)
    file_size_mb = path.stat().st_size / 1e6

    print(f"File: {path}")
    print(f"Tokens: {total_tokens:,} ({total_tokens / 1e6:.1f}M)")
    print(f"Size: {file_size_mb:.1f} MB")
    print(f"Vocab range: [{data.min()}, {data.max()}]")
    print(f"First 20 tokens: {data[:20].tolist()}")
    print(f"Sequences at 2048: {total_tokens // 2048:,}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Tokenize HF dataset → binary")
    p.add_argument("--dataset", type=str, help="HuggingFace dataset name")
    p.add_argument("--name", type=str, default=None, help="Dataset config name")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--tokenizer", type=str, default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--output", type=str, help="Output binary file path")
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--batch_size", type=int, default=1000,
                    help="Docs per batch for tokenizer (default: 1000)")
    p.add_argument("--verify", type=str, default=None)

    args = p.parse_args()

    if args.verify:
        verify_data(args.verify)
        return

    if not args.dataset or not args.output:
        p.error("--dataset and --output are required for tokenization")

    tokenize_dataset(
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer,
        output_path=args.output,
        split=args.split,
        name=args.name,
        max_tokens=args.max_tokens,
        text_field=args.text_field,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
