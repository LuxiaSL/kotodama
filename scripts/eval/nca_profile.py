#!/usr/bin/env python3
"""Geometric profile of a no-AttnRes NCA checkpoint -> suggested AttnRes block boundaries.

The DD boundary method of record (how DD-3B ``0,1,3,7,15,19,24`` was derived):

    Phase 1  raw NCA pretrain WITHOUT AttnRes -> periodic checkpoints
    Phase 2  pick the knee checkpoint, run this script -> boundaries
    Phase 3  NCA co-train WITH AttnRes on the derived boundaries

Restored from ``scripts/nca_profile.py`` at tag ``pre-consolidation-2026-09`` (dropped in
the consolidation); ``suggest_boundaries`` is kept VERBATIM so a re-derivation at a new
scale runs the same algorithm the 3B did. Two rulers for attention entropy:

- ``legacy``: the archived computation — softmax(q kᵀ/√d) from the RAW q/k projections,
  no QK-norm, no RoPE. This is what DD-3B was fit on; kept for faithful replication.
- ``true``: the model's actual attention (QK-norm + RoPE, fp32 softmax) via the
  parity-tested capture helper ``kotodama.model.capture._eager_attention``.

Boundaries are suggested from each ruler; when they disagree, say so in the read.

Probe: ``random`` = uniform NCA-vocab tokens [0, 10002) as the 3B run used (default), or
``--probe-bin`` = windows of a real token stream (e.g. the NCA eval .bin).

Usage::

    tools/run_py.sh scripts/eval/nca_profile.py --checkpoint <phase1>/step_N.pt \\
        --config-section 1b --rope-fp32 --max-boundaries 7 --min-block-size 1 \\
        --out /tmp/claude-output/nca-profile/step_N.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("nca_profile")

PROFILE_KEYS = ("anisotropy", "attn_entropy_mean", "attn_entropy_std", "stable_rank_q",
                "stable_rank_k", "stable_rank_o", "stable_rank_gate", "stable_rank_down",
                "dead_units")


# ── per-layer metrics (archived definitions) ─────────────────────────────────

def _stable_rank(weight: torch.Tensor) -> float:
    """Frobenius-norm-based stable rank: ||W||_F^2 / sigma_max^2."""
    w = weight.detach().float()
    fro_sq = w.pow(2).sum().item()
    sigma_max = torch.linalg.svdvals(w)[0].item()
    return fro_sq / max(sigma_max ** 2, 1e-10)


def _anisotropy(hidden: torch.Tensor, max_samples: int = 2048, seed: int = 0) -> float:
    """Variance fraction on the first principal component (seeded subsample)."""
    h = hidden.detach().float().cpu()
    if h.shape[0] > max_samples:
        g = torch.Generator().manual_seed(seed)
        h = h[torch.randperm(h.shape[0], generator=g)[:max_samples]]
    h = h - h.mean(dim=0, keepdim=True)
    try:
        eig = torch.linalg.eigvalsh(h.T @ h / h.shape[0])
        return eig[-1].item() / max(eig.sum().item(), 1e-10)
    except Exception as e:  # keep the profile going; a 0 is visible in the table
        logger.warning("anisotropy failed: %s", e)
        return 0.0


def _attention_entropy(attn_weights: torch.Tensor) -> tuple[float, float]:
    """Mean and std over heads of per-head attention entropy. attn_weights: (B, H, T, T)."""
    w = attn_weights.float()
    ent = -(w * (w + 1e-10).log()).sum(dim=-1)   # (B, H, T)
    per_head = ent.mean(dim=(0, 2))
    return per_head.mean().item(), per_head.std().item()


def _dead_unit_fraction(hidden: torch.Tensor) -> float:
    h = hidden.detach().float().reshape(-1, hidden.shape[-1])
    return (h.std(dim=0) < 1e-6).float().mean().item()


def _legacy_attention_weights(attn: Any, x_normed: torch.Tensor) -> torch.Tensor:
    """The archived ruler: raw q/k projections, no QK-norm, no RoPE, causal softmax."""
    B, S, _ = x_normed.shape
    nh, nkv, hd = attn.num_heads, attn.num_kv_heads, attn.head_dim
    q = attn.q_proj(x_normed).view(B, S, nh, hd).transpose(1, 2)
    k = attn.k_proj(x_normed).view(B, S, nkv, hd).transpose(1, 2)
    k = k.repeat_interleave(nh // nkv, dim=1)
    scores = (q @ k.transpose(-2, -1)) * hd ** -0.5
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=scores.device), diagonal=1)
    return torch.softmax(scores.masked_fill(causal, float("-inf")).float(), dim=-1)


@torch.no_grad()
def extract_profile(model: Any, probe: torch.Tensor, attn_chunk: int = 8) -> dict[str, dict[str, list[float]]]:
    """Walk the (no-AttnRes) layers; per-layer profile under both attention-entropy rulers.
    Returns {"legacy": profile, "true": profile}; non-entropy metrics are shared."""
    from kotodama.model.capture import _eager_attention

    if getattr(model.config, "attn_res", False):
        raise SystemExit("nca_profile expects a Phase-1 (no-AttnRes) checkpoint")
    shared = {k: [] for k in PROFILE_KEYS if not k.startswith("attn_entropy")}
    ent = {r: {"attn_entropy_mean": [], "attn_entropy_std": []} for r in ("legacy", "true")}
    x = model.embed_tokens(probe)
    S = probe.shape[1]
    rope_cos, rope_sin = model.rope_cos, model.rope_sin
    for i, layer in enumerate(model.layers):
        normed = layer.attn_norm(x)
        acc = {r: ([], []) for r in ent}
        for c in range(0, probe.shape[0], attn_chunk):
            xc = normed[c:c + attn_chunk]
            m, s = _attention_entropy(_legacy_attention_weights(layer.attn, xc))
            acc["legacy"][0].append(m); acc["legacy"][1].append(s)
            *_, w = _eager_attention(layer.attn, xc, rope_cos, rope_sin)
            m, s = _attention_entropy(w)
            acc["true"][0].append(m); acc["true"][1].append(s)
        for r in ent:
            ent[r]["attn_entropy_mean"].append(float(np.mean(acc[r][0])))
            ent[r]["attn_entropy_std"].append(float(np.mean(acc[r][1])))
        x = layer(x, rope_cos[:S], rope_sin[:S])
        h = x.reshape(-1, x.shape[-1])
        shared["anisotropy"].append(_anisotropy(h))
        shared["dead_units"].append(_dead_unit_fraction(x))
        a, f = layer.attn, layer.ffn
        shared["stable_rank_q"].append(_stable_rank(a.q_proj.weight))
        shared["stable_rank_k"].append(_stable_rank(a.k_proj.weight))
        shared["stable_rank_o"].append(_stable_rank(a.o_proj.weight))
        shared["stable_rank_gate"].append(_stable_rank(f.gate_proj.weight))
        shared["stable_rank_down"].append(_stable_rank(f.down_proj.weight))
        logger.info("L%2d aniso=%.3f ent(legacy)=%.3f ent(true)=%.3f SR_q=%.1f", i,
                    shared["anisotropy"][-1], ent["legacy"]["attn_entropy_mean"][-1],
                    ent["true"]["attn_entropy_mean"][-1], shared["stable_rank_q"][-1])
    return {r: {**shared, **ent[r]} for r in ent}


# ── boundary suggestion (VERBATIM from the archived script) ─────────────────

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
    return boundaries, combined


# ── CLI ───────────────────────────────────────────────────────────────────────

def _probe(args: argparse.Namespace, device: str) -> torch.Tensor:
    if args.probe_bin:
        data = np.memmap(args.probe_bin, dtype=np.uint16, mode="r")
        stride = (len(data) - args.seq_len) // max(args.probe_seqs - 1, 1)
        rows = [data[i * stride:i * stride + args.seq_len].astype(np.int64) for i in range(args.probe_seqs)]
        return torch.tensor(np.stack(rows), device=device)
    g = torch.Generator().manual_seed(args.seed)
    return torch.randint(0, 10002, (args.probe_seqs, args.seq_len), generator=g).to(device)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-section", default="1b")
    ap.add_argument("--rope-fp32", action="store_true")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--probe-seqs", type=int, default=256, help="archived default: 256 x 2048")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--probe-bin", default=None, help="real token windows instead of random NCA-vocab tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-block-size", type=int, default=2)
    ap.add_argument("--max-boundaries", type=int, default=6)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from kotodama.eval.model_loader import load_model
    meta: dict[str, Any] = {}
    model = load_model(args.checkpoint, config_section=args.config_section,
                       attn_res_config={"attn_res": False, "attn_impl": "sdpa"},
                       device=args.device, meta=meta,
                       overrides={"rope_fp32": True} if args.rope_fp32 else None)
    model = model.float() if args.device == "cpu" else model
    n = len(model.layers)
    probe = _probe(args, args.device)
    logger.info("probe %s (%s)", tuple(probe.shape), "bin" if args.probe_bin else "random NCA-vocab")
    profiles = extract_profile(model, probe)
    out: dict[str, Any] = {"checkpoint": args.checkpoint, "step": meta.get("step"),
                           "num_layers": n, "probe": "bin" if args.probe_bin else "random",
                           "probe_shape": list(probe.shape),
                           "min_block_size": args.min_block_size,
                           "max_boundaries": args.max_boundaries, "rulers": {}}
    for ruler, prof in profiles.items():
        b, combined = suggest_boundaries(prof, n, args.min_block_size, args.max_boundaries)
        out["rulers"][ruler] = {"profile": prof, "suggested_boundaries": b,
                                "combined_scores": combined}
        top = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)[:8]
        print(f"[{ruler}] suggested boundaries: {b}   top transitions: "
              + ", ".join(f"L{i}->{i + 1}:{combined[i]:.2f}" for i in top))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
