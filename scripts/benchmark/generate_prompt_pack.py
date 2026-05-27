#!/usr/bin/env python3
"""Generate a fixed prompt pack at exact token counts for reproducible benchmarking.

Usage:
    python scripts/benchmark/generate_prompt_pack.py \
        --output scripts/benchmark/prompt_pack.json

Produces prompts at target lengths (16, 128, 1024, 3072 tokens) across
multiple domains by tokenizing seed texts, tiling to fill the target
length, truncating, and decoding back.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"

TARGET_LENGTHS = [16, 128, 1024, 3072]

SEED_TEXTS: dict[str, str] = {
    "narrative": (
        "Once upon a time, in a small village by the sea, there lived an old fisherman "
        "who had spent his entire life studying the tides. Every morning before dawn, he "
        "would walk down to the shore and listen to the waves, trying to understand the "
        "patterns that governed their movement. The villagers thought him eccentric, but "
        "he knew something they did not: the sea had a language of its own, and those who "
        "listened carefully could learn to read its moods. One winter evening, a terrible "
        "storm gathered on the horizon. The fisherman watched the clouds with growing "
        "unease. He had seen storms before, but this one was different. The waves were "
        "behaving in ways he had never observed, crashing against the rocks with a rhythm "
        "that seemed almost deliberate. He gathered his nets and hurried back to warn the "
        "village. The wind howled through the narrow streets as he knocked on every door, "
        "telling each family to move to higher ground. Some listened, others laughed. By "
        "midnight, the water had risen three feet above the highest tide mark anyone could "
        "remember. Those who had heeded the old man's warning watched from the hilltop as "
        "the sea reclaimed the lower village. When morning came, the fisherman stood alone "
        "on the shore, surveying the damage. The sea had spoken, and he had translated its "
        "warning. But he knew this was only the beginning."
    ),
    "encyclopedic": (
        "The history of the Roman Empire can be divided into several distinct periods, "
        "beginning with the founding of Rome in 753 BCE according to traditional dating. "
        "The early monarchy gave way to the Roman Republic in 509 BCE, which lasted until "
        "the rise of Augustus as the first Emperor in 27 BCE. The Principate, as the early "
        "imperial period is known, saw Rome reach its greatest territorial extent under "
        "Trajan in 117 CE, stretching from Britain in the northwest to Mesopotamia in the "
        "east. The empire's administrative structures were remarkably sophisticated for "
        "their time, featuring a complex system of provinces, each governed by appointed "
        "officials who reported to the central authority in Rome. The Roman legal system, "
        "codified in the Twelve Tables and later expanded through imperial edicts and "
        "juristic interpretation, formed the foundation of civil law traditions that "
        "persist in many modern legal systems. The empire's road network, spanning over "
        "250,000 miles at its peak, facilitated trade, military movement, and cultural "
        "exchange across three continents. Roman engineering achievements, including "
        "aqueducts, amphitheaters, and concrete construction techniques, demonstrated a "
        "level of technical sophistication that would not be matched in Europe for over "
        "a thousand years after the fall of the Western Empire in 476 CE."
    ),
    "code": (
        "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
        "from dataclasses import dataclass\nfrom typing import Optional\n\n\n"
        "@dataclass\nclass TransformerConfig:\n"
        '    """Configuration for a standard transformer model."""\n'
        "    hidden_size: int = 768\n"
        "    num_layers: int = 12\n"
        "    num_heads: int = 12\n"
        "    intermediate_size: int = 3072\n"
        "    vocab_size: int = 50257\n"
        "    max_seq_len: int = 1024\n"
        "    dropout: float = 0.1\n\n\n"
        "class MultiHeadAttention(nn.Module):\n"
        "    def __init__(self, config: TransformerConfig) -> None:\n"
        "        super().__init__()\n"
        "        self.num_heads = config.num_heads\n"
        "        self.head_dim = config.hidden_size // config.num_heads\n"
        "        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)\n"
        "        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)\n"
        "        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)\n"
        "        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)\n\n"
        "    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None"
        ") -> torch.Tensor:\n"
        "        bsz, seq_len, _ = x.shape\n"
        "        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)\n"
        "        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)\n"
        "        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)\n"
        "        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)\n"
        "        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, "
        "is_causal=True)\n"
        "        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)\n"
        "        return self.o_proj(attn)\n\n\n"
        "class FeedForward(nn.Module):\n"
        "    def __init__(self, config: TransformerConfig) -> None:\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)\n"
        "        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)\n\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        return self.fc2(F.gelu(self.fc1(x)))\n"
    ),
    "academic": (
        "Abstract: We present a novel approach to understanding the relationship between "
        "neural network depth and generalization performance in overparameterized models. "
        "Recent theoretical work has established that deeper networks can achieve lower "
        "approximation error for certain function classes, but the interplay between depth, "
        "width, and optimization dynamics remains poorly understood. In this paper, we "
        "provide both theoretical and empirical evidence that the effective depth of a "
        "trained network, measured by the rank of intermediate representations, correlates "
        "strongly with generalization performance on held-out data. We introduce a new "
        "metric called the representation rank profile, which tracks how information is "
        "compressed and expanded across layers during training. Our experiments on image "
        "classification benchmarks demonstrate that networks which develop a characteristic "
        "hourglass-shaped rank profile consistently outperform those with monotonically "
        "decreasing or flat profiles. Furthermore, we show that standard regularization "
        "techniques such as dropout and weight decay implicitly encourage this beneficial "
        "rank structure. These findings suggest that the geometry of learned representations, "
        "rather than the raw number of parameters, is the primary determinant of a model's "
        "ability to generalize from training data to unseen examples. Our results have "
        "practical implications for architecture design and training procedures."
    ),
}


def tile_and_truncate(token_ids: list[int], target_len: int) -> list[int]:
    """Tile token IDs to fill target length, then truncate exactly."""
    if len(token_ids) >= target_len:
        return token_ids[:target_len]
    repeats = (target_len // len(token_ids)) + 1
    tiled = (token_ids * repeats)[:target_len]
    return tiled


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed benchmark prompt pack")
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "prompt_pack.json",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    logger.info("Tokenizer loaded: %s (vocab %d)", TOKENIZER_NAME, len(tokenizer))

    prompts: list[dict] = []

    for domain, seed_text in SEED_TEXTS.items():
        seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
        logger.info("Seed '%s': %d tokens from %d chars", domain, len(seed_ids), len(seed_text))

        for target_len in TARGET_LENGTHS:
            ids = tile_and_truncate(seed_ids, target_len)
            text = tokenizer.decode(ids, skip_special_tokens=True)

            verify_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(verify_ids) != target_len:
                logger.warning(
                    "Round-trip mismatch for %s@%d: got %d tokens (storing token IDs directly)",
                    domain, target_len, len(verify_ids),
                )

            prompts.append({
                "name": f"{domain}-{target_len}",
                "domain": domain,
                "target_tokens": target_len,
                "actual_tokens": len(ids),
                "text": text,
                "token_ids": ids,
            })
            logger.info("  %s-%d: %d tokens", domain, target_len, len(ids))

    pack = {
        "metadata": {
            "tokenizer": TOKENIZER_NAME,
            "target_lengths": TARGET_LENGTHS,
            "domains": list(SEED_TEXTS.keys()),
            "num_prompts": len(prompts),
        },
        "prompts": prompts,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(pack, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d prompts to %s", len(prompts), args.output)


if __name__ == "__main__":
    main()
