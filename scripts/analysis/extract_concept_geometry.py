#!/usr/bin/env python3
"""Unified concept geometry extraction (Track 6a).

Extracts residual stream activations at concept token positions for
manifold geometry analysis. Supports all concept sets from configs/concepts.yaml.

Replaces: extract_manifolds.py, extract_manifolds_v3.py.

Usage::

    # All concept sets, all registry checkpoints
    python -m scripts.extract_concept_geometry --device cpu

    # Specific concept sets and checkpoints
    python -m scripts.extract_concept_geometry --device cpu \\
        --concept-sets months,digits,us_states \\
        --checkpoints P3-Muon-002,NCA-002

    # Specific tier
    python -m scripts.extract_concept_geometry --device cpu --tier 1

    # Single checkpoint by path
    python -m scripts.extract_concept_geometry --device cpu \\
        --checkpoint path/to/ckpt.pt --name my-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.forward import forward_with_states
from src.eval.model_loader import load_checkpoint_registry, load_model

logger = logging.getLogger(__name__)


@dataclass
class ConceptSet:
    """A concept set loaded from config."""

    name: str
    concepts: list[str]
    topology: str
    tier: int
    category: str

    @property
    def n_concepts(self) -> int:
        return len(self.concepts)


def load_concept_config(
    config_path: Path | str = "configs/concepts.yaml",
) -> tuple[list[ConceptSet], list[str], dict[str, Any]]:
    """Load concept sets, templates, and extra metadata from config.

    Returns:
        (concept_sets, templates, extra_meta) where extra_meta contains
        state_coordinates and any other auxiliary data.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    concept_sets = [
        ConceptSet(
            name=cs["name"],
            concepts=cs["concepts"],
            topology=cs["topology"],
            tier=cs["tier"],
            category=cs["category"],
        )
        for cs in raw["concept_sets"]
    ]
    templates = raw["templates"]
    extra_meta = {}
    if "state_coordinates" in raw:
        extra_meta["state_coordinates"] = raw["state_coordinates"]

    return concept_sets, templates, extra_meta


def find_concept_token_position(
    tokenizer: Any,
    template: str,
    concept: str,
    category: str,
) -> tuple[list[int], int, int]:
    """Find token indices for the concept in the filled template.

    Uses prefix-diff method: tokenize full text and prefix (up to concept),
    then the concept tokens are between the two lengths.

    Returns:
        (token_ids, start_idx, end_idx) where start/end are inclusive indices
        into token_ids for the concept tokens.
    """
    # Fill template
    text = template.replace("{X}", concept).replace("{category}", category)

    # Find concept position in text
    concept_pos = text.find(concept)
    if concept_pos < 0:
        # Concept not found (e.g., bare template "{X}" → concept IS the text)
        token_ids = tokenizer.encode(text)
        return token_ids, 0, len(token_ids) - 1

    prefix = text[:concept_pos]

    # Tokenize full text and prefix
    full_ids = tokenizer.encode(text)
    prefix_ids = tokenizer.encode(prefix) if prefix else []

    # Concept tokens = difference
    start_idx = max(0, len(prefix_ids) - 1)  # -1 because BOS overlaps
    if not prefix:
        start_idx = 0

    # Find end by tokenizing prefix + concept
    prefix_concept = text[: concept_pos + len(concept)]
    pc_ids = tokenizer.encode(prefix_concept)
    end_idx = min(len(pc_ids) - 1, len(full_ids) - 1)

    # Safety clamps
    start_idx = max(0, min(start_idx, len(full_ids) - 1))
    end_idx = max(start_idx, min(end_idx, len(full_ids) - 1))

    return full_ids, start_idx, end_idx


def validate_tokenization(
    tokenizer: Any,
    concept_sets: list[ConceptSet],
    templates: list[str],
) -> dict[str, Any]:
    """Pre-check tokenizations and report multi-token concepts."""
    report: dict[str, Any] = {}

    for cs in concept_sets:
        cs_report: dict[str, Any] = {"n_concepts": cs.n_concepts, "multi_token": []}
        for concept in cs.concepts:
            for template in templates:
                ids, start, end = find_concept_token_position(
                    tokenizer, template, concept, cs.category
                )
                n_tokens = end - start + 1
                if n_tokens > 1:
                    cs_report["multi_token"].append({
                        "concept": concept,
                        "template": template,
                        "n_tokens": n_tokens,
                    })

        if cs_report["multi_token"]:
            n_multi = len(set(m["concept"] for m in cs_report["multi_token"]))
            logger.info(
                "  %s: %d/%d concepts are multi-token in at least one template",
                cs.name, n_multi, cs.n_concepts,
            )

        report[cs.name] = cs_report

    return report


@torch.no_grad()
def extract_concept_activations(
    model: Any,
    tokenizer: Any,
    concept_set: ConceptSet,
    templates: list[str],
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Extract residual stream activations at concept token positions.

    For each concept × template pair, runs a forward pass and captures
    the hidden state at the concept's last token position at every layer.

    Returns dict with arrays:
        activations: (n_samples, n_layers+1, hidden_size)
        last_tok_activations: (n_samples, n_layers+1, hidden_size)
        concept_idx: (n_samples,)
        template_idx: (n_samples,)
        token_counts: (n_samples,)
    """
    n_concepts = concept_set.n_concepts
    n_templates = len(templates)
    n_samples = n_concepts * n_templates
    n_layers = model.config.num_layers
    hidden_size = model.config.hidden_size

    activations = np.zeros((n_samples, n_layers + 1, hidden_size), dtype=np.float32)
    last_tok_acts = np.zeros((n_samples, n_layers + 1, hidden_size), dtype=np.float32)
    concept_idx = np.zeros(n_samples, dtype=np.int32)
    template_idx = np.zeros(n_samples, dtype=np.int32)
    token_counts = np.zeros(n_samples, dtype=np.int32)

    sample_i = 0
    for ci, concept in enumerate(concept_set.concepts):
        for ti, template in enumerate(templates):
            ids, start, end = find_concept_token_position(
                tokenizer, template, concept, concept_set.category
            )
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)

            result = forward_with_states(model, input_ids)

            for layer_i, state in enumerate(result.states):
                # Concept position (last token of concept)
                activations[sample_i, layer_i] = state[0, end, :].float().cpu().numpy()
                # Last token position
                activations[sample_i, layer_i] = state[0, end, :].float().cpu().numpy()
                last_tok_acts[sample_i, layer_i] = state[0, -1, :].float().cpu().numpy()

            concept_idx[sample_i] = ci
            template_idx[sample_i] = ti
            token_counts[sample_i] = end - start + 1
            sample_i += 1

    return {
        "activations": activations,
        "last_tok_activations": last_tok_acts,
        "concept_idx": concept_idx,
        "template_idx": template_idx,
        "token_counts": token_counts,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Concept geometry extraction (Track 6a)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("-o", "--output-dir", type=Path, default=None)

    # Concept selection
    p.add_argument("--concept-sets", type=str, default=None,
                   help="Comma-separated concept set names")
    p.add_argument("--tier", type=int, default=None,
                   help="Include only concept sets at this tier or lower")
    p.add_argument("--concepts-config", type=Path, default=Path("configs/concepts.yaml"))

    # Checkpoint selection
    ckpt = p.add_mutually_exclusive_group()
    ckpt.add_argument("--checkpoints", type=str, default=None)
    ckpt.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--attn-res", type=str, default=None)

    # Model config
    p.add_argument("--config", type=Path, default=Path("configs/model.yaml"))
    p.add_argument("--config-section", type=str, default="proxy")

    args = p.parse_args()

    # Load concept config
    concept_sets, templates, extra_meta = load_concept_config(args.concepts_config)

    # Filter concept sets
    if args.concept_sets:
        requested = set(args.concept_sets.split(","))
        concept_sets = [cs for cs in concept_sets if cs.name in requested]
    if args.tier is not None:
        concept_sets = [cs for cs in concept_sets if cs.tier <= args.tier]

    if not concept_sets:
        logger.error("No concept sets selected")
        sys.exit(1)

    logger.info("Concept sets (%d): %s", len(concept_sets), [cs.name for cs in concept_sets])
    logger.info("Templates (%d): %s", len(templates), templates)

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    # Validate tokenization
    tok_report = validate_tokenization(tokenizer, concept_sets, templates)

    # Resolve checkpoints
    runs: list[tuple[str, Path, dict | None]] = []
    if args.checkpoint:
        name = args.name or args.checkpoint.stem
        ar = None
        if args.attn_res:
            ar = {"attn_res": True}
            for part in args.attn_res.split():
                k, v = part.split("=", 1)
                if k == "n_blocks":
                    ar["attn_res_n_blocks"] = int(v)
                elif k == "boundaries":
                    ar["attn_res_boundaries"] = [int(x) for x in v.split(",")]
        runs.append((name, args.checkpoint, ar))
    else:
        registry = load_checkpoint_registry()
        if args.checkpoints:
            names = [n.strip() for n in args.checkpoints.split(",")]
            for n in names:
                if n in registry:
                    runs.append((registry[n].name, registry[n].path, registry[n].attn_res_config))
        else:
            runs = [(i.name, i.path, i.attn_res_config) for i in registry.values()]

    runs = [(n, p, a) for n, p, a in runs if p.exists()]
    missing_runs = [(n, p) for n, p, a in runs if not p.exists()]
    if missing_runs:
        logger.warning("Missing: %s", [n for n, _ in missing_runs])
    if not runs:
        logger.error("No valid checkpoints found")
        sys.exit(1)

    # Output directory
    output_dir = args.output_dir
    if output_dir is None:
        if len(runs) == 1:
            output_dir = Path("analysis") / runs[0][0] / "concept_geometry"
        else:
            output_dir = Path("analysis") / "concept_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract
    total_fwd = sum(cs.n_concepts * len(templates) for cs in concept_sets) * len(runs)
    logger.info("Total forward passes: %d", total_fwd)

    all_data: dict[str, np.ndarray] = {}
    t0 = time.time()

    model_cfg = yaml.safe_load(open(args.config))

    for run_name, ckpt_path, ar_config in runs:
        logger.info("=" * 60)
        logger.info("Loading: %s", run_name)

        cfg = dict(model_cfg[args.config_section])
        if ar_config:
            cfg.update(ar_config)

        model = load_model(
            ckpt_path, config_path=args.config, config_section=args.config_section,
            attn_res_config=ar_config, device=args.device,
        )

        for cs in concept_sets:
            t1 = time.time()
            result = extract_concept_activations(model, tokenizer, cs, templates, args.device)
            prefix = f"{run_name}_{cs.name}"
            for key, arr in result.items():
                all_data[f"{prefix}_{key}"] = arr
            logger.info(
                "  %s: %d samples, %.1fs",
                cs.name, result["activations"].shape[0], time.time() - t1,
            )

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Save activations
    all_data["checkpoint_names"] = np.array([n for n, _, _ in runs])
    all_data["concept_set_names"] = np.array([cs.name for cs in concept_sets])
    all_data["n_layers"] = np.array([model_cfg[args.config_section]["num_layers"]])
    all_data["hidden_size"] = np.array([model_cfg[args.config_section]["hidden_size"]])
    all_data["templates"] = np.array(templates)

    np.savez_compressed(output_dir / "activations.npz", **all_data)
    size_mb = (output_dir / "activations.npz").stat().st_size / 1e6
    logger.info("Saved activations.npz (%.1f MB)", size_mb)

    # Save metadata
    metadata: dict[str, Any] = {
        "checkpoints": {
            name: {"path": str(path), **(ar or {})}
            for name, path, ar in runs
        },
        "concept_sets": {
            cs.name: {
                "concepts": cs.concepts,
                "topology": cs.topology,
                "tier": cs.tier,
                "category": cs.category,
                "n_concepts": cs.n_concepts,
            }
            for cs in concept_sets
        },
        "templates": templates,
        "tokenization": tok_report,
    }
    metadata.update(extra_meta)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Done. Total: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
