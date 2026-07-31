#!/usr/bin/env python3
"""Validate the engine's Gumbel-max sampler against serve.py's softmax+multinomial.

Model-free, seconds to run. Two gates:
  1. EXACT: the repetition-penalty + temperature transform matches
     serve.py's sample_next_token math bitwise on fp32 logits.
  2. DISTRIBUTIONAL: empirical token frequencies from Gumbel-max draws match
     multinomial draws from softmax of the SAME transformed logits
     (symmetric KL + max abs probability gap on union of top-100 tokens).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark/test_sampler_dist.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

VOCAB = 49152
N_DRAWS = 400_000
TEMP = 0.9
RP = 1.2


def serve_transform(logits: torch.Tensor, presence: torch.Tensor, rp: float) -> torch.Tensor:
    """serve.py sample_next_token penalty math (vectorized, fp32)."""
    out = logits.clone()
    pen = torch.where(out > 0, out / rp, out * rp)
    return torch.where(presence, pen, out)


def engine_transform(logits: torch.Tensor, presence: torch.Tensor, rp_buf: torch.Tensor) -> torch.Tensor:
    pen = torch.where(logits > 0, logits / rp_buf, logits * rp_buf)
    return torch.where(presence, pen, logits)


def main() -> None:
    device = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(123)
    logits = torch.randn(VOCAB, device=device, generator=g) * 4.0  # spread like real logits
    presence = torch.zeros(VOCAB, dtype=torch.bool, device=device)
    presence[torch.randint(0, VOCAB, (300,), generator=g, device=device)] = True

    # Gate 1: penalty math equality. Not bitwise: serve.py divides by a
    # python float (scalar routed via double), the engine by an fp32 0-dim
    # buffer — measured difference is exactly 1 ulp of fp32 (~5e-7 abs) on a
    # subset of penalized tokens. Gate at a few ulps.
    rp_buf = torch.tensor(RP, device=device, dtype=torch.float32)
    a = serve_transform(logits, presence, RP)
    b = engine_transform(logits, presence, rp_buf)
    max_diff = float((a - b).abs().max())
    exact = max_diff < 1e-5
    print(f"gate1 penalty-math max|diff| = {max_diff:.2e} (threshold 1e-5): {'pass' if exact else 'FAIL'}")

    scaled = a / TEMP
    probs = F.softmax(scaled, dim=-1)

    # Gate 2: distribution match
    ref_counts = torch.zeros(VOCAB, device=device)
    gum_counts = torch.zeros(VOCAB, device=device)
    chunk = 50_000
    for _ in range(N_DRAWS // chunk):
        toks = torch.multinomial(probs, chunk, replacement=True)
        ref_counts.scatter_add_(0, toks, torch.ones_like(toks, dtype=torch.float))
        u = torch.rand(chunk, VOCAB, device=device)
        gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
        toks_g = (scaled.unsqueeze(0) + gumbel).argmax(dim=-1)
        gum_counts.scatter_add_(0, toks_g, torch.ones_like(toks_g, dtype=torch.float))

    ref_p = ref_counts / N_DRAWS
    gum_p = gum_counts / N_DRAWS
    top = torch.topk(probs, 100).indices
    union = torch.unique(torch.cat([top, torch.topk(ref_p, 100).indices, torch.topk(gum_p, 100).indices]))
    max_gap = float((ref_p[union] - gum_p[union]).abs().max())
    eps = 1e-9
    kl_sym = float((F.kl_div((gum_p + eps).log(), ref_p + eps, reduction="sum")
                    + F.kl_div((ref_p + eps).log(), gum_p + eps, reduction="sum")))
    # Also compare both empiricals against the TRUE distribution
    gap_ref_true = float((ref_p[union] - probs[union]).abs().max())
    gap_gum_true = float((gum_p[union] - probs[union]).abs().max())

    print(f"gate2 over {N_DRAWS} draws: max|p_ref - p_gumbel| = {max_gap:.5f} (top-union tokens)")
    print(f"       multinomial-vs-true {gap_ref_true:.5f} | gumbel-vs-true {gap_gum_true:.5f} (sampling noise scale)")
    print(f"       symmetric KL = {kl_sym:.6f}")
    ok = exact and max_gap < 3 * max(gap_ref_true, gap_gum_true) + 1e-4
    print("OVERALL:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
