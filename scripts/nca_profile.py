"""Extract geometric profile from NCA checkpoint and suggest AttnRes boundaries.

Adapted from metis/scripts/nca_profile.py for the kotodama 3B model.

Phase 2 of the NCA → AttnRes pipeline:
  Phase 1: Raw NCA pretrain → periodic checkpoints
  Phase 2: Pick knee checkpoint, run this script → get boundaries
  Phase 3: NCA cotrain with AttnRes using the derived boundaries

Usage:
    python -m scripts.nca_profile \
        --checkpoint /models/kotodama-data/checkpoints/nca-3b-phase1/step_00003800.pt.zst \
        --model_size 3b \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _stable_rank(weight: torch.Tensor) -> float:
    """Frobenius-norm-based stable rank: ||W||_F^2 / sigma_max^2."""
    w = weight.detach().float()
    fro_sq = w.pow(2).sum().item()
    sigma_max = torch.linalg.svdvals(w)[0].item()
    return fro_sq / max(sigma_max ** 2, 1e-10)


def _anisotropy(hidden: torch.Tensor, max_samples: int = 2048) -> float:
    """Anisotropy: how much variance is captured by the first principal component."""
    h = hidden.detach().float().cpu()
    if h.shape[0] > max_samples:
        idx = torch.randperm(h.shape[0])[:max_samples]
        h = h[idx]
    h = h - h.mean(dim=0, keepdim=True)
    try:
        cov = h.T @ h / h.shape[0]
        eigvals = torch.linalg.eigvalsh(cov)
        total = eigvals.sum().item()
        pc1 = eigvals[-1].item()
        return pc1 / max(total, 1e-10)
    except Exception as e:
        logger.warning("Anisotropy failed at layer: %s", e)
        return 0.0


def _attention_entropy(attn_weights: torch.Tensor) -> tuple[float, float]:
    """Mean and std of per-head attention entropy."""
    # attn_weights: (B, H, T, T)
    eps = 1e-10
    ent = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)  # (B, H, T)
    per_head = ent.mean(dim=(0, 2))  # (H,)
    return per_head.mean().item(), per_head.std().item()


def _dead_unit_fraction(hidden: torch.Tensor) -> float:
    """Fraction of hidden dimensions that are always zero or constant."""
    h = hidden.detach().float().reshape(-1, hidden.shape[-1])
    std = h.std(dim=0)
    return (std < 1e-6).float().mean().item()


def extract_profile(
    model: torch.nn.Module,
    probe_batch: torch.Tensor,
    device: torch.device,
) -> dict[str, list[float]]:
    """Forward pass through all layers, extract per-layer geometric profile."""
    from src.model.llama import LuxiaBaseModel

    num_layers = len(model.layers)
    model.eval()

    profile = {
        "anisotropy": [],
        "attn_entropy_mean": [],
        "attn_entropy_std": [],
        "stable_rank_q": [],
        "stable_rank_k": [],
        "stable_rank_o": [],
        "stable_rank_gate": [],
        "stable_rank_down": [],
        "dead_units": [],
    }

    # Manual forward to capture all hidden states
    # Attention entropy is computed in small chunks to avoid OOM
    attn_chunk = min(8, probe_batch.shape[0])

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Embed
        x = model.embed_tokens(probe_batch)

        # RoPE from model buffers
        bsz, seq_len, _ = x.shape
        rope_cos = model.rope_cos[:seq_len]
        rope_sin = model.rope_sin[:seq_len]

        n_heads = model.config.num_attention_heads
        n_kv = model.config.num_kv_heads
        head_dim = model.config.head_dim
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        for i, layer in enumerate(model.layers):
            # Attention entropy: compute in chunks to avoid OOM on (B, H, T, T)
            ent_means = []
            ent_stds = []
            normed_full = layer.attn_norm(x)
            for c_start in range(0, bsz, attn_chunk):
                c_end = min(c_start + attn_chunk, bsz)
                normed_chunk = normed_full[c_start:c_end]
                cb = c_end - c_start

                q = layer.attn.q_proj(normed_chunk)
                k = layer.attn.k_proj(normed_chunk)
                q = q.view(cb, seq_len, n_heads, head_dim).transpose(1, 2)
                k = k.view(cb, seq_len, n_kv, head_dim).transpose(1, 2)

                n_rep = n_heads // n_kv
                if n_rep > 1:
                    k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(cb, n_heads, seq_len, head_dim)

                scale = head_dim ** -0.5
                attn_w = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_w = attn_w.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                attn_w = F.softmax(attn_w, dim=-1)

                em, es = _attention_entropy(attn_w)
                ent_means.append(em)
                ent_stds.append(es)
                del attn_w, q, k

            profile["attn_entropy_mean"].append(sum(ent_means) / len(ent_means))
            profile["attn_entropy_std"].append(sum(ent_stds) / len(ent_stds))
            del normed_full

            # Full layer forward with RoPE
            x = layer(x, rope_cos, rope_sin)

            # Hidden state metrics
            h_flat = x.detach().float().reshape(-1, x.shape[-1])
            profile["anisotropy"].append(_anisotropy(h_flat))
            profile["dead_units"].append(_dead_unit_fraction(x))

            # Weight stable ranks
            profile["stable_rank_q"].append(_stable_rank(layer.attn.q_proj.weight))
            profile["stable_rank_k"].append(_stable_rank(layer.attn.k_proj.weight))
            profile["stable_rank_o"].append(_stable_rank(layer.attn.o_proj.weight))
            profile["stable_rank_gate"].append(_stable_rank(layer.ffn.gate_proj.weight))
            profile["stable_rank_down"].append(_stable_rank(layer.ffn.down_proj.weight))

            logger.info("  Layer %2d: aniso=%.3f entropy=%.3f SR_q=%.1f",
                        i, profile["anisotropy"][-1], profile["attn_entropy_mean"][-1], profile["stable_rank_q"][-1])

    return profile


def suggest_boundaries(
    profile: dict[str, list[float]],
    num_layers: int,
    min_block_size: int = 2,
    max_boundaries: int = 6,
) -> tuple[list[int], list[float]]:
    """Suggest AttnRes block boundaries using zone-aware analysis.

    Three-signal approach:
    1. Jump scores — large simultaneous shifts across metrics
    2. Zone continuity — penalize splitting inside smooth processing regions
    3. Regime change — reward boundaries where computation character changes
    """
    metrics_to_use = ["anisotropy", "attn_entropy_mean", "stable_rank_q", "stable_rank_down"]
    available = [m for m in metrics_to_use if m in profile and len(profile[m]) == num_layers]

    if not available:
        return [0], []

    # Signal 1: Raw jump scores (normalized delta magnitude per metric)
    raw_jumps = [0.0] * (num_layers - 1)
    for metric in available:
        vals = profile[metric]
        deltas = [abs(vals[i + 1] - vals[i]) for i in range(num_layers - 1)]
        max_delta = max(deltas) if deltas else 1.0
        if max_delta < 1e-10:
            continue
        for i, d in enumerate(deltas):
            raw_jumps[i] += d / max_delta

    # Signal 2: Zone continuity penalty
    # Smooth SR_q regions = continuous processing zone, don't split them
    sr_q = profile.get("stable_rank_q", [0.0] * num_layers)
    window = 3
    local_variance = []
    for i in range(num_layers - 1):
        lo = max(0, i - window // 2)
        hi = min(num_layers, i + window // 2 + 2)
        segment = sr_q[lo:hi]
        mean_val = sum(segment) / len(segment)
        var = sum((x - mean_val) ** 2 for x in segment) / len(segment)
        local_variance.append(var)

    max_var = max(local_variance) if local_variance else 1.0
    if max_var > 1e-10:
        continuity_penalty = [1.0 - (v / max_var) for v in local_variance]
    else:
        continuity_penalty = [0.0] * len(local_variance)

    # Signal 3: Regime change scoring
    # Large SR_q shifts = regime transitions (input→processing, processing→output)
    sr_deltas = [abs(sr_q[i + 1] - sr_q[i]) for i in range(num_layers - 1)]
    max_sr_delta = max(sr_deltas) if sr_deltas else 1.0
    regime_scores = [d / max(max_sr_delta, 1e-10) for d in sr_deltas]

    ent = profile.get("attn_entropy_mean", [0.0] * num_layers)
    ent_deltas = [abs(ent[i + 1] - ent[i]) for i in range(num_layers - 1)]
    max_ent_delta = max(ent_deltas) if ent_deltas else 1.0
    ent_regime = [d / max(max_ent_delta, 1e-10) for d in ent_deltas] if max_ent_delta > 1e-10 else [0.0] * len(ent_deltas)

    # Combined: jump * zone_factor * regime_boost
    combined = []
    for i in range(num_layers - 1):
        zone_factor = 1.0 - 0.5 * continuity_penalty[i]
        regime_boost = 1.0 + 0.5 * regime_scores[i] + 0.3 * ent_regime[i]
        combined.append(raw_jumps[i] * zone_factor * regime_boost)

    # Greedy selection
    boundaries = [0]
    candidates = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)

    for candidate_layer in candidates:
        boundary_layer = candidate_layer + 1
        if boundary_layer >= num_layers - 1:
            continue
        too_close = any(abs(boundary_layer - b) < min_block_size for b in boundaries)
        if too_close:
            continue
        boundaries.append(boundary_layer)
        if len(boundaries) >= max_boundaries:
            break

    boundaries.sort()

    logger.info("Zone-aware boundary suggestion:")
    logger.info("  Raw jump leaders: %s",
                [(i+1, f"{raw_jumps[i]:.2f}") for i in sorted(range(len(raw_jumps)), key=lambda i: raw_jumps[i], reverse=True)[:6]])
    logger.info("  Combined leaders: %s",
                [(i+1, f"{combined[i]:.2f}") for i in sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:6]])

    return boundaries, combined


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description="Extract NCA geometric profile and suggest AttnRes boundaries")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_size", type=str, default="3b")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--probe_tokens", type=int, default=256,
                    help="Number of probe sequences (each seq_len=2048)")
    p.add_argument("--min_block_size", type=int, default=2)
    p.add_argument("--max_boundaries", type=int, default=6)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device)

    # Import model
    from src.training.train import MODEL_CONFIGS
    from src.model.llama import LuxiaModelConfig, LuxiaBaseModel

    model_kwargs = MODEL_CONFIGS[args.model_size].copy()
    config = LuxiaModelConfig(**model_kwargs)
    model = LuxiaBaseModel(config).to(device=device, dtype=torch.bfloat16)
    num_layers = config.num_layers

    logger.info("Model: %s (%d layers, %.1fM params)",
                args.model_size, num_layers, sum(p.numel() for p in model.parameters()) / 1e6)

    # Load checkpoint (handle zstd)
    ckpt_path = Path(args.checkpoint)
    load_path = ckpt_path
    if ckpt_path.name.endswith(".pt.zst"):
        load_path = ckpt_path.with_name(ckpt_path.name.replace(".pt.zst", ".pt"))
        logger.info("Decompressing %s...", ckpt_path.name)
        subprocess.run(["zstd", "-d", str(ckpt_path), "-o", str(load_path), "-f"],
                       check=True, capture_output=True)

    ckpt = torch.load(load_path, map_location=device, weights_only=False)
    if ckpt_path.name.endswith(".pt.zst"):
        load_path.unlink(missing_ok=True)

    model_state = ckpt.get("model", ckpt)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    model.load_state_dict(cleaned, strict=True)
    step = ckpt.get("step", -1)
    logger.info("Loaded checkpoint: step %d", step)
    del ckpt

    # Generate probe batch (random NCA-vocab tokens)
    logger.info("Generating probe batch (%d sequences)...", args.probe_tokens)
    probe = torch.randint(0, 10002, (args.probe_tokens, 2048), device=device)

    # Extract profile
    logger.info("Extracting geometric profile across %d layers...", num_layers)
    profile = extract_profile(model, probe, device)

    # Print per-layer table
    print("\n" + "=" * 100)
    print(f"GEOMETRIC PROFILE — {args.model_size}, step {step}")
    print("=" * 100)

    header = f"{'Layer':>5} | {'Aniso':>7} | {'Ent_m':>7} | {'Ent_s':>7} | {'SR_q':>7} | {'SR_k':>7} | {'SR_o':>7} | {'SR_gate':>8} | {'SR_down':>8} | {'Dead':>6}"
    print(header)
    print("-" * len(header))
    for i in range(num_layers):
        print(
            f"{i:>5} | "
            f"{profile['anisotropy'][i]:>7.3f} | "
            f"{profile['attn_entropy_mean'][i]:>7.3f} | "
            f"{profile['attn_entropy_std'][i]:>7.3f} | "
            f"{profile['stable_rank_q'][i]:>7.1f} | "
            f"{profile['stable_rank_k'][i]:>7.1f} | "
            f"{profile['stable_rank_o'][i]:>7.1f} | "
            f"{profile['stable_rank_gate'][i]:>8.1f} | "
            f"{profile['stable_rank_down'][i]:>8.1f} | "
            f"{profile['dead_units'][i]:>6.3f}"
        )

    # Layer-to-layer deltas
    print("\n" + "=" * 100)
    print("LAYER-TO-LAYER DELTAS (large jumps = potential boundaries)")
    print("=" * 100)
    header2 = f"{'L→L+1':>6} | {'dAniso':>8} | {'dEntropy':>9} | {'dSR_q':>8} | {'dSR_down':>9} | {'JumpScore':>10}"
    print(header2)
    print("-" * len(header2))

    jump_scores = [0.0] * (num_layers - 1)
    for metric in ["anisotropy", "attn_entropy_mean", "stable_rank_q", "stable_rank_down"]:
        vals = profile[metric]
        deltas = [abs(vals[i + 1] - vals[i]) for i in range(num_layers - 1)]
        mx = max(deltas) if deltas else 1.0
        if mx > 1e-10:
            for i, d in enumerate(deltas):
                jump_scores[i] += d / mx

    for i in range(num_layers - 1):
        print(
            f"{i:>2} → {i+1:<2} | "
            f"{profile['anisotropy'][i+1] - profile['anisotropy'][i]:>+8.3f} | "
            f"{profile['attn_entropy_mean'][i+1] - profile['attn_entropy_mean'][i]:>+9.3f} | "
            f"{profile['stable_rank_q'][i+1] - profile['stable_rank_q'][i]:>+8.1f} | "
            f"{profile['stable_rank_down'][i+1] - profile['stable_rank_down'][i]:>+9.1f} | "
            f"{jump_scores[i]:>10.3f}"
        )

    # Suggest boundaries (zone-aware)
    boundaries, combined_scores = suggest_boundaries(
        profile, num_layers,
        min_block_size=args.min_block_size,
        max_boundaries=args.max_boundaries,
    )

    block_sizes = []
    for i in range(len(boundaries)):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else num_layers
        block_sizes.append(end - boundaries[i])

    # Print combined scores with zone indicators
    print("\n" + "=" * 100)
    print("ZONE-AWARE COMBINED SCORES (jump × zone_factor × regime_boost)")
    print("=" * 100)
    for i in range(num_layers - 1):
        bar = "#" * int(combined_scores[i] * 10) if combined_scores else ""
        marker = " ◀ BOUNDARY" if (i + 1) in boundaries else ""
        print(f"  L{i:>2}→{i+1:<2}: {combined_scores[i]:>6.2f} {bar}{marker}")

    print("\n" + "=" * 100)
    print(f"SUGGESTED BOUNDARIES: {boundaries}")
    print(f"Block sizes: {block_sizes}")
    print(f"CLI flag: --attn_res_boundaries {','.join(str(b) for b in boundaries)}")
    print("=" * 100)

    # Save
    output_path = args.output or str(Path(args.checkpoint).parent / "geometric_profile.json")
    profile_data = {
        "checkpoint": str(args.checkpoint),
        "model_size": args.model_size,
        "step": step,
        "num_layers": num_layers,
        "profile": {k: [float(v) for v in vals] for k, vals in profile.items()},
        "suggested_boundaries": boundaries,
        "block_sizes": block_sizes,
        "raw_jump_scores": [float(s) for s in jump_scores],
        "combined_scores": [float(s) for s in combined_scores] if combined_scores else [],
    }
    with open(output_path, "w") as f:
        json.dump(profile_data, f, indent=2)
    logger.info("Saved profile to %s", output_path)


if __name__ == "__main__":
    main()
