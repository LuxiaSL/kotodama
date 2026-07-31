#!/usr/bin/env python3
"""Validate the engine's in-graph top-k/top-p against serve.py's sample_next_token.

Model-free, seconds to run. Two gates:

  1. EXACT (deterministic): over a grid of (top_k, top_p, temperature) x many
     random logit vectors (ties forced on a subset), the engine's masking math
     (sort -> kth-value threshold -> renormalized exclusive-cumsum cutoff ->
     unsort) must produce the SAME survivor set as serve.py's sequential
     masked_fill blocks, with surviving logit values equal to a few fp32 ulps
     (0-dim-buffer vs python-scalar division, same tolerance as the round-1
     penalty gate).
  2. DISTRIBUTIONAL: Gumbel-max draws over the engine's final masked logits
     match multinomial draws from serve.py's final softmax (symmetric KL +
     max abs probability gap on the union of top-100 tokens), for k100, p90,
     and the composed k100+p90 law.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark/test_sampler_truncated.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

VOCAB = 49152
N_DRAWS = 400_000
CHUNK = 2048
TEMP = 0.9
RP = 1.2

# (top_k, top_p) laws to gate. 0 / 0.0 = disabled (serve.py convention).
EXACT_GRID = [
    (0, 0.9), (0, 0.95), (0, 0.5), (100, 0.0), (5, 0.0), (1, 0.0),
    (100, 0.9), (2000, 0.95), (0, 0.0), (0, 1.0), (VOCAB + 7, 0.0),
]
DIST_LAWS = [(100, 0.0), (0, 0.9), (100, 0.9)]


def serve_mask(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Verbatim port of serve.py sample_next_token's truncation blocks.

    Input: temperature-scaled fp32 logits (V,). Returns masked logits.
    """
    logits = logits.clone()
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = logits.topk(top_k).values[-1]
        logits = logits.masked_fill(logits < kth_val, float("-inf"))
    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = logits.sort(descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
        sorted_logits[mask] = float("-inf")
        logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)
    return logits


def engine_mask(logits: torch.Tensor, top_k_eff: torch.Tensor, top_p_val: torch.Tensor) -> torch.Tensor:
    """The engine's _step_sampled_truncated masking math (post temp-scale)."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    kth = sorted_logits.index_select(0, (top_k_eff - 1).view(1))
    sorted_logits = torch.where(
        sorted_logits < kth, torch.full_like(sorted_logits, float("-inf")), sorted_logits
    )
    probs = sorted_logits.softmax(dim=-1)
    cutoff = probs.cumsum(dim=-1) - probs >= top_p_val
    sorted_logits = torch.where(
        cutoff, torch.full_like(sorted_logits, float("-inf")), sorted_logits
    )
    return torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)


def to_buffers(top_k: int, top_p: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The engine's _sync_sampling_buffers mapping (disabled -> no-op values).

    Disabled p maps to 2.0, NOT 1.0: fp32 cumsum saturates to exactly 1.0 in
    the distribution tail, so a 1.0 threshold spuriously masks it (caught by
    this test on 2026-07-05: 11,337 of 49,152 tokens masked on a disabled law).
    """
    k = top_k if top_k > 0 else VOCAB
    p = top_p if 0.0 < top_p < 1.0 else 2.0
    return (
        torch.tensor(min(k, VOCAB), device=device, dtype=torch.long),
        torch.tensor(p, device=device, dtype=torch.float32),
    )


def gate_exact(device: torch.device) -> bool:
    g = torch.Generator(device=device.type).manual_seed(1234)
    ok = True
    for trial in range(50):
        logits = torch.randn(VOCAB, device=device, generator=g) * 4.0
        if trial % 3 == 1:
            # Force tie plateaus, including across the k-th position.
            # (.item() detaches the fill value: duplicated indices + an
            # aliasing source tensor is an illegal overlapping index_put_.)
            idx = torch.randint(0, VOCAB, (500,), generator=g, device=device)
            logits[idx] = float(logits[idx[0]].item())
        if trial % 7 == 3:
            logits = logits.round()  # heavy ties everywhere
        scaled = (logits / TEMP).float()
        for top_k, top_p in EXACT_GRID:
            a = serve_mask(scaled, top_k, top_p)
            k_buf, p_buf = to_buffers(top_k, top_p, device)
            b = engine_mask(scaled, k_buf, p_buf)
            surv_a, surv_b = torch.isfinite(a), torch.isfinite(b)
            if not torch.equal(surv_a, surv_b):
                n_a, n_b = int(surv_a.sum()), int(surv_b.sum())
                sym = int((surv_a ^ surv_b).sum())
                print(f"  FAIL survivor-set trial={trial} k={top_k} p={top_p}: "
                      f"serve={n_a} engine={n_b} symdiff={sym}")
                ok = False
                continue
            diff = (a[surv_a] - b[surv_b]).abs().max()
            if float(diff) > 3e-6:  # a few fp32 ulps at logit scale
                print(f"  FAIL survivor-values trial={trial} k={top_k} p={top_p}: maxdiff={float(diff)}")
                ok = False
    print(f"Gate 1 (exact survivor-mask equivalence, {50 * len(EXACT_GRID)} cases): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def gate_dist(device: torch.device) -> bool:
    g = torch.Generator(device=device.type).manual_seed(99)
    logits = torch.randn(VOCAB, device=device, generator=g) * 4.0
    scaled = (logits / TEMP).float()
    ok = True
    for top_k, top_p in DIST_LAWS:
        ref_probs = serve_mask(scaled, top_k, top_p).softmax(dim=-1)
        ref_counts = torch.bincount(
            torch.multinomial(ref_probs, N_DRAWS, replacement=True, generator=g), minlength=VOCAB
        ).float()

        k_buf, p_buf = to_buffers(top_k, top_p, device)
        final = engine_mask(scaled, k_buf, p_buf)
        eng_counts = torch.zeros(VOCAB, device=device)
        done = 0
        while done < N_DRAWS:
            n = min(CHUNK, N_DRAWS - done)
            u = torch.rand((n, VOCAB), device=device, generator=g)
            gumbel = -torch.log((-torch.log(u.clamp_min(1e-20))).clamp_min(1e-20))
            toks = (final.unsqueeze(0) + gumbel).argmax(dim=-1)
            eng_counts += torch.bincount(toks, minlength=VOCAB).float()
            done += n

        p_emp = ref_counts / N_DRAWS
        q_emp = eng_counts / N_DRAWS
        top = torch.unique(torch.cat([p_emp.topk(100).indices, q_emp.topk(100).indices]))
        max_gap = float((p_emp[top] - q_emp[top]).abs().max())
        eps = 1e-9
        m = 0.5 * (p_emp + q_emp) + eps
        skl = float((p_emp * ((p_emp + eps) / m).log()).sum() + (q_emp * ((q_emp + eps) / m).log()).sum())
        # Engine draws must land outside the survivor set exactly never.
        leaked = int(eng_counts[~torch.isfinite(final)].sum())
        line_ok = max_gap < 2.5e-3 and skl < 5e-3 and leaked == 0
        print(f"  k={top_k} p={top_p}: max_prob_gap={max_gap:.5f} symKL={skl:.5f} "
              f"leaked={leaked} {'PASS' if line_ok else 'FAIL'}")
        ok = ok and line_ok
    print(f"Gate 2 (distribution, {N_DRAWS} draws/law): {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    device = torch.device("cuda")
    ok = gate_exact(device)
    ok = gate_dist(device) and ok
    print(f"SAMPLER_TRUNCATED: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
