"""Unified model loading for evaluation and analysis.

Consolidates checkpoint loading patterns from ~10 scripts into one module.
Handles torch.compile prefix stripping, AttnRes auto-detection / explicit
config, YAML config loading, and device placement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from kotodama import ckpt, presets
from kotodama.model.llama import LuxiaBaseModel, LuxiaModelConfig

logger = logging.getLogger(__name__)


def load_model_config(
    config_path: Path | str | None = None,
    section: str = "proxy",
) -> dict[str, Any]:
    """Model architecture kwargs for LuxiaModelConfig.

    With no ``config_path``, ``section`` names a shape in ``kotodama.presets``
    (aliases like "model" -> "3b" accepted). A YAML path is still honoured for
    one-off shapes that are not presets.

    Raises:
        FileNotFoundError: If a given config file doesn't exist.
        KeyError: If the section / preset is unknown.
    """
    if config_path is None:
        return presets.shape(section)
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    if section not in full_config:
        available = list(full_config.keys())
        raise KeyError(
            f"Section '{section}' not found in {config_path}. "
            f"Available: {available}"
        )

    return dict(full_config[section])


def load_model(
    checkpoint_path: Path | str,
    config_path: Path | str | None = None,
    config_section: str = "proxy",
    attn_res_config: dict[str, Any] | None = None,
    device: str = "cuda:0",
) -> LuxiaBaseModel:
    """Load a model from a checkpoint file.

    Handles:
    - torch.compile ``_orig_mod.`` prefix stripping
    - DDP ``module.`` prefix stripping
    - AttnRes configuration (explicit or auto-detected)
    - State dict wrapped in {"model": ...} or bare

    AttnRes handling priority:
    1. If ``attn_res_config`` is provided, use it (preferred, explicit).
    2. Otherwise, auto-detect from state dict keys (fallback, assumes n_blocks=7).

    Args:
        checkpoint_path: A .pt/.pt.zst path or a kotodama.ckpt registry name.
        config_path: Optional model YAML; default = kotodama.presets.
        config_section: Preset name (3b, proxy, ...) or YAML section.
        attn_res_config: Explicit AttnRes kwargs (attn_res, attn_res_n_blocks,
            attn_res_boundaries). Preferred over auto-detection.
        device: Target device for the model.

    Returns:
        Model in eval mode on the specified device.

    Raises:
        FileNotFoundError: If checkpoint or config doesn't exist.
    """
    checkpoint_path = ckpt.decompress_cached(ckpt.resolve(checkpoint_path))

    # Load model architecture config
    cfg = load_model_config(config_path, config_section)

    # Load checkpoint state dict
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cleaned = ckpt.model_state(state)

    # Configure AttnRes
    if attn_res_config is not None:
        cfg.update(attn_res_config)
        logger.info("AttnRes config (explicit): %s", attn_res_config)
    else:
        # Auto-detect from state dict keys
        has_attn_res = any("attn_res_query" in k for k in cleaned)
        if has_attn_res:
            cfg.update({"attn_res": True, "attn_res_n_blocks": 7})
            logger.warning(
                "AttnRes auto-detected from checkpoint keys — using default "
                "n_blocks=7. Pass attn_res_config explicitly for reliability."
            )

    # Build model — filter out YAML keys that aren't LuxiaModelConfig fields
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(LuxiaModelConfig)}
    cfg = {k: v for k, v in cfg.items() if k in valid_fields}
    model = LuxiaBaseModel(LuxiaModelConfig(**cfg))
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    # Report key mismatches (filter out expected AttnRes key mismatches)
    real_missing = [k for k in missing if "attn_res" not in k]
    if real_missing:
        logger.warning("Missing keys in checkpoint: %s", real_missing[:10])
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected[:10])

    model.eval()
    model.to(device)
    if "cuda" in str(device):
        model.bfloat16()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded model: %s (%.1fM params, device=%s, attn_res=%s)",
        checkpoint_path.name,
        param_count / 1e6,
        device,
        cfg.get("attn_res", False),
    )

    return model
