"""Unified model loading for evaluation and analysis.

Consolidates checkpoint loading patterns from ~10 scripts into one module.
Handles torch.compile prefix stripping, AttnRes auto-detection / explicit
config, YAML config loading, and device placement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Registry entry for a named checkpoint."""

    name: str
    path: Path
    attn_res_config: dict[str, Any] | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)


def load_model_config(
    config_path: Path | str = "configs/model.yaml",
    section: str = "proxy",
) -> dict[str, Any]:
    """Load model architecture config from YAML.

    Args:
        config_path: Path to the model config YAML.
        section: Which config section to load (proxy, intermediate, model).

    Returns:
        Dict of model config kwargs suitable for LuxiaModelConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        KeyError: If section is missing from config.
    """
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


def load_checkpoint_registry(
    config_path: Path | str = "configs/checkpoints.yaml",
) -> dict[str, CheckpointInfo]:
    """Load the checkpoint registry from YAML.

    The registry maps checkpoint names to paths and optional AttnRes config.

    Returns:
        Dict mapping checkpoint name → CheckpointInfo.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint registry not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    registry: dict[str, CheckpointInfo] = {}
    for group_name, group_data in raw.items():
        checkpoints = group_data.get("checkpoints", {})
        for ckpt_name, ckpt_data in checkpoints.items():
            attn_res_config = None
            ar_keys = {"attn_res", "attn_res_n_blocks", "attn_res_boundaries"}
            ar_subset = {k: v for k, v in ckpt_data.items() if k in ar_keys}
            if ar_subset:
                attn_res_config = ar_subset

            registry[ckpt_name] = CheckpointInfo(
                name=ckpt_name,
                path=Path(ckpt_data["path"]),
                attn_res_config=attn_res_config,
                group=group_name,
                tags=ckpt_data.get("tags", []),
            )

    return registry


def load_model(
    checkpoint_path: Path | str,
    config_path: Path | str = "configs/model.yaml",
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
        checkpoint_path: Path to the .pt checkpoint file.
        config_path: Path to model YAML config.
        config_section: Section name in YAML (proxy, intermediate, model).
        attn_res_config: Explicit AttnRes kwargs (attn_res, attn_res_n_blocks,
            attn_res_boundaries). Preferred over auto-detection.
        device: Target device for the model.

    Returns:
        Model in eval mode on the specified device.

    Raises:
        FileNotFoundError: If checkpoint or config doesn't exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model architecture config
    cfg = load_model_config(config_path, config_section)

    # Load checkpoint state dict
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = state.get("model", state)

    # Strip torch.compile and DDP prefixes
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in model_state.items():
        k = k.replace("_orig_mod.", "")
        k = k.replace("module.", "")
        cleaned[k] = v

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

    # Build model
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
