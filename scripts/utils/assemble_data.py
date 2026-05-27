"""Assemble per-source tokenized binaries into a single training binary.

Reads tokenized uint16 .bin files (one per source), splits each into
fixed-size blocks, shuffles blocks across sources, and writes a single
concatenated binary compatible with ``src.data.dataset.TokenizedDataset``.

Also writes a provenance index so that any training sequence can be
traced back to its source — useful for correlating gradient norm spikes
with data quality issues.

Usage::

    # Assemble all sources
    python scripts/utils/assemble_data.py \
        --input-dir /models/kotodama-data/tokenized \
        --output /models/kotodama-data/assembled/train.bin \
        --seq-len 4096

    # Exclude specific sources
    python scripts/utils/assemble_data.py \
        --input-dir /models/kotodama-data/tokenized \
        --output /models/kotodama-data/assembled/train.bin \
        --exclude stack_v1

    # Cap a source at N tokens
    python scripts/utils/assemble_data.py \
        --input-dir /models/kotodama-data/tokenized \
        --output /models/kotodama-data/assembled/train.bin \
        --cap pesa2o:25000000000

    # Dry run (report sizes without writing)
    python scripts/utils/assemble_data.py \
        --input-dir /models/kotodama-data/tokenized \
        --output /dev/null --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

BLOCK_TOKENS = 1_048_576  # 1M tokens per block (~2MB on disk)
BYTES_PER_TOKEN = 2  # uint16


@dataclass
class SourceInfo:
    name: str
    path: Path
    total_tokens: int
    cap_tokens: Optional[int] = None
    effective_tokens: int = 0
    n_blocks: int = 0

    def __post_init__(self) -> None:
        if self.cap_tokens is not None:
            self.effective_tokens = min(self.total_tokens, self.cap_tokens)
        else:
            self.effective_tokens = self.total_tokens


@dataclass
class Block:
    source: str
    source_offset_tokens: int
    n_tokens: int


@dataclass
class ProvenanceSpan:
    source: str
    global_start_token: int
    global_end_token: int
    source_start_token: int
    n_tokens: int
    block_idx: int


def discover_sources(
    input_dir: Path,
    excludes: set[str],
    caps: dict[str, int],
) -> list[SourceInfo]:
    """Find all .bin files and compute their token counts."""
    sources: list[SourceInfo] = []

    for bin_path in sorted(input_dir.glob("*.bin")):
        name = bin_path.stem
        if name in excludes:
            logger.info("Excluding %s", name)
            continue

        file_size = bin_path.stat().st_size
        if file_size == 0:
            logger.warning("Skipping empty file: %s", bin_path)
            continue

        total_tokens = file_size // BYTES_PER_TOKEN
        cap = caps.get(name)

        src = SourceInfo(
            name=name,
            path=bin_path,
            total_tokens=total_tokens,
            cap_tokens=cap,
        )
        sources.append(src)
        logger.info(
            "  %-25s %12.1fM tokens%s",
            name,
            src.effective_tokens / 1e6,
            f" (capped from {total_tokens / 1e6:.1f}M)" if cap else "",
        )

    return sources


def plan_blocks(
    sources: list[SourceInfo],
    block_tokens: int,
    seq_len: int,
) -> list[Block]:
    """Divide each source into fixed-size blocks, aligned to seq_len."""
    aligned_block = (block_tokens // seq_len) * seq_len
    if aligned_block == 0:
        raise ValueError(f"block_tokens ({block_tokens}) must be >= seq_len ({seq_len})")

    blocks: list[Block] = []

    for src in sources:
        usable = (src.effective_tokens // seq_len) * seq_len
        if usable == 0:
            logger.warning("Source %s has fewer tokens than seq_len, skipping", src.name)
            continue

        offset = 0
        n = 0
        while offset < usable:
            chunk = min(aligned_block, usable - offset)
            if chunk >= seq_len:
                blocks.append(Block(source=src.name, source_offset_tokens=offset, n_tokens=chunk))
                n += 1
            offset += chunk

        src.n_blocks = n
        src.effective_tokens = usable

    return blocks


def shuffle_blocks(blocks: list[Block], seed: int) -> list[Block]:
    """Deterministic shuffle of all blocks across sources."""
    rng = np.random.RandomState(seed)
    indices = np.arange(len(blocks))
    rng.shuffle(indices)
    return [blocks[i] for i in indices]


def write_assembled(
    blocks: list[Block],
    sources: dict[str, SourceInfo],
    output_path: Path,
) -> list[ProvenanceSpan]:
    """Write the assembled binary and return provenance spans."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance: list[ProvenanceSpan] = []

    mmap_cache: dict[str, np.ndarray] = {}

    def get_mmap(name: str) -> np.ndarray:
        if name not in mmap_cache:
            mmap_cache[name] = np.memmap(sources[name].path, dtype=np.uint16, mode="r")
        return mmap_cache[name]

    global_offset = 0
    t0 = time.time()
    total_blocks = len(blocks)

    with open(output_path, "wb") as f:
        for i, block in enumerate(blocks):
            data = get_mmap(block.source)
            start = block.source_offset_tokens
            end = start + block.n_tokens
            chunk = np.array(data[start:end], dtype=np.uint16)
            f.write(chunk.tobytes())

            provenance.append(ProvenanceSpan(
                source=block.source,
                global_start_token=global_offset,
                global_end_token=global_offset + block.n_tokens,
                source_start_token=block.source_offset_tokens,
                n_tokens=block.n_tokens,
                block_idx=i,
            ))
            global_offset += block.n_tokens

            if (i + 1) % 100 == 0 or i == total_blocks - 1:
                elapsed = time.time() - t0
                pct = (i + 1) / total_blocks * 100
                tokens_written = global_offset
                logger.info(
                    "  [%d/%d %.0f%%] %.1fB tokens written, %.1f GB, %.0fs",
                    i + 1, total_blocks, pct,
                    tokens_written / 1e9,
                    tokens_written * BYTES_PER_TOKEN / 1e9,
                    elapsed,
                )

    return provenance


def write_provenance(
    provenance: list[ProvenanceSpan],
    sources: list[SourceInfo],
    output_path: Path,
    seq_len: int,
    seed: int,
) -> None:
    """Write provenance index and manifest JSON files."""
    manifest_path = output_path.parent / "manifest.json"
    provenance_path = output_path.parent / "provenance.json"

    total_tokens = sum(s.effective_tokens for s in sources)
    total_sequences = total_tokens // seq_len

    source_summary = {}
    for src in sources:
        pct = src.effective_tokens / total_tokens * 100 if total_tokens > 0 else 0
        source_summary[src.name] = {
            "total_tokens_available": src.total_tokens,
            "tokens_used": src.effective_tokens,
            "cap_tokens": src.cap_tokens,
            "n_blocks": src.n_blocks,
            "pct_of_total": round(pct, 2),
        }

    manifest = {
        "total_tokens": total_tokens,
        "total_sequences": total_sequences,
        "seq_len": seq_len,
        "n_sources": len(sources),
        "n_blocks": len(provenance),
        "block_size_tokens": BLOCK_TOKENS,
        "shuffle_seed": seed,
        "file_size_bytes": total_tokens * BYTES_PER_TOKEN,
        "sources": source_summary,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest: %s", manifest_path)

    spans = [
        {
            "source": s.source,
            "global_start": s.global_start_token,
            "global_end": s.global_end_token,
            "source_start": s.source_start_token,
            "n_tokens": s.n_tokens,
            "block_idx": s.block_idx,
        }
        for s in provenance
    ]

    with open(provenance_path, "w") as f:
        json.dump(spans, f)
    logger.info("Provenance index: %s (%d spans)", provenance_path, len(spans))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [assemble] %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Assemble tokenized binaries into training data")
    p.add_argument("--input-dir", type=str, required=True,
                    help="Directory containing per-source .bin files")
    p.add_argument("--output", type=str, required=True,
                    help="Output binary path (e.g. assembled/train.bin)")
    p.add_argument("--seq-len", type=int, default=4096,
                    help="Sequence length for alignment (default: 4096)")
    p.add_argument("--block-tokens", type=int, default=BLOCK_TOKENS,
                    help="Tokens per shuffle block (default: 1M)")
    p.add_argument("--seed", type=int, default=42,
                    help="Shuffle seed (default: 42)")
    p.add_argument("--exclude", type=str, nargs="*", default=[],
                    help="Source names to exclude")
    p.add_argument("--cap", type=str, nargs="*", default=[],
                    help="Cap source tokens: source_name:max_tokens")
    p.add_argument("--dry-run", action="store_true",
                    help="Report sizes without writing")

    args = p.parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    excludes = set(args.exclude)

    caps: dict[str, int] = {}
    for cap_str in args.cap:
        try:
            name, tokens = cap_str.split(":")
            caps[name] = int(tokens)
        except ValueError:
            p.error(f"Invalid --cap format: {cap_str}. Use source_name:max_tokens")

    logger.info("Discovering sources in %s...", input_dir)
    sources = discover_sources(input_dir, excludes, caps)

    if not sources:
        logger.error("No sources found")
        return

    logger.info("Planning blocks (block_size=%dK tokens, seq_len=%d)...",
                args.block_tokens // 1000, args.seq_len)
    blocks = plan_blocks(sources, args.block_tokens, args.seq_len)

    total_tokens = sum(s.effective_tokens for s in sources)
    total_seqs = total_tokens // args.seq_len
    logger.info(
        "Total: %.1fB tokens, %d sequences, %d blocks from %d sources",
        total_tokens / 1e9, total_seqs, len(blocks), len(sources),
    )

    if args.dry_run:
        logger.info("DRY RUN — no files written")
        for src in sorted(sources, key=lambda s: -s.effective_tokens):
            pct = src.effective_tokens / total_tokens * 100
            logger.info("  %-25s %8.1fB tokens  %5.1f%%  (%d blocks)",
                        src.name, src.effective_tokens / 1e9, pct, src.n_blocks)
        return

    logger.info("Shuffling %d blocks (seed=%d)...", len(blocks), args.seed)
    blocks = shuffle_blocks(blocks, args.seed)

    logger.info("Writing assembled binary to %s...", output_path)
    t0 = time.time()
    source_map = {s.name: s for s in sources}
    provenance = write_assembled(blocks, source_map, output_path)
    elapsed = time.time() - t0

    logger.info(
        "Assembly complete: %.1fB tokens, %.1f GB, %.0fs (%.1f GB/s)",
        total_tokens / 1e9,
        output_path.stat().st_size / 1e9,
        elapsed,
        output_path.stat().st_size / 1e9 / max(elapsed, 0.001),
    )

    write_provenance(provenance, sources, output_path, args.seq_len, args.seed)

    verify_path = output_path
    data = np.memmap(verify_path, dtype=np.uint16, mode="r")
    logger.info(
        "Verification: %d tokens, vocab range [%d, %d], %d sequences at %d",
        len(data), data.min(), data.max(), len(data) // args.seq_len, args.seq_len,
    )


if __name__ == "__main__":
    main()
