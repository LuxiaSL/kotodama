"""Model shapes, AttnRes block boundaries, and the tokenizer: the one table.

Everything else (rope theta, norm eps, QK-norm, tied embeddings, z-loss) is a
``LuxiaModelConfig`` default. Training, serving, and eval all build their
configs from here, so a new model size is one entry below.
"""

from __future__ import annotations

from typing import Any

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"

# Block AttnRes boundaries (first layer of each block).
# DD-3B: re-derived from the 3B phase-1 geometry; the shipped 3B uses it.
# DD-v1: the proxy-scale (108M) configuration.
DD3B: tuple[int, ...] = (0, 1, 3, 7, 15, 19, 24)
DDV1: tuple[int, ...] = (0, 3, 7, 12, 21, 25)
BOUNDARIES: dict[str, tuple[int, ...]] = {"dd3b": DD3B, "ddv1": DDV1}

_COMMON = dict(vocab_size=49152, max_position_embeddings=4096)

SHAPES: dict[str, dict[str, Any]] = {
    "3b": dict(hidden_size=3072, num_layers=28, num_attention_heads=24,
               num_kv_heads=8, head_dim=128, intermediate_size=8192, **_COMMON),
    # ~7.18B params at vocab 49,152: the Llama-3-8B skeleton; its extra ~1B is
    # purely the 128K vocab.
    "7b": dict(hidden_size=4096, num_layers=32, num_attention_heads=32,
               num_kv_heads=8, head_dim=128, intermediate_size=14336, **_COMMON),
    # d1024 x 28 (~0.38B with tied embeddings) — the April "intermediate".
    "intermediate": dict(hidden_size=1024, num_layers=28, num_attention_heads=8,
                         num_kv_heads=4, head_dim=128, intermediate_size=2816, **_COMMON),
    # The 108M proxy used for the March–August ablation matrices.
    "proxy": dict(hidden_size=512, num_layers=28, num_attention_heads=4,
                  num_kv_heads=2, head_dim=128, intermediate_size=1408, **_COMMON),
    "smoke": dict(hidden_size=256, num_layers=4, num_attention_heads=4,
                  num_kv_heads=2, head_dim=64, intermediate_size=512,
                  vocab_size=1024, max_position_embeddings=2048),
}

# Older names still found in configs and scripts.
ALIASES: dict[str, str] = {"full": "3b", "model": "3b", "8b": "7b"}

# The boundaries each shape was trained with, where there is a shipped model.
DEFAULT_BOUNDARIES: dict[str, tuple[int, ...]] = {"3b": DD3B, "proxy": DDV1}


def resolve(name: str) -> str:
    """Canonical shape name for ``name`` (accepts aliases)."""
    key = ALIASES.get(name, name)
    if key not in SHAPES:
        raise KeyError(f"unknown model size {name!r}; choose from {sorted(SHAPES) + sorted(ALIASES)}")
    return key


def shape(name: str) -> dict[str, Any]:
    """A fresh copy of the shape kwargs for ``name`` (accepts aliases)."""
    return dict(SHAPES[resolve(name)])


def names() -> list[str]:
    """Every accepted size name, canonical first."""
    return list(SHAPES) + list(ALIASES)
