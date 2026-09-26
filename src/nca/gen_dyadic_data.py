"""ARM-D-lite dataset writer — dyadic NCA training bin + held-out eval items.

Mix (DRAFT-SPEC-dyadic-nca §4, lite): 30% original Class-IV single-rule
trajectories (native 32x32x4 — the aliveness core) + 70% integrated dyadic
family (coupled turns + commitments + perturbations) at CONTEXT-SCALE grids:
16x16, 2 channels, 30 steps -> ~4K tokens/trajectory ~= one training window,
so turn alternation and commit->query lags live INSIDE the context. (The
original 32x32x4 frame costs ~1K tokens/step; dyadic structure would never
fit a window at that scale — this is a deliberate design split.)

Eval items (held-out rule pairs, never in training): ledger items + smoothing
pairs per src/nca/dyadic.py emitters.

Usage (node1, sharded across GPUs):
  for s in 0..5: CUDA_VISIBLE_DEVICES=$s python -m src.nca.gen_dyadic_data \
      --shard $s --n-shards 6 --target-tokens 125_000_000 \
      --out /models/kotodama-data/p5-dyadic &
  # then: python -m src.nca.gen_dyadic_data --finalize --out ...
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .generator import NCAConfig, NCARule, sample_rule_config, simulate_trajectory, tokenize_trajectory
from .dyadic import (DyadicConfig, simulate_dyadic, serialize_dyadic,
                     emit_ledger_items, emit_smoothing_pairs)

ORIG_FRac = 0.30

ORIG_CFG = NCAConfig()  # native 32x32, 4ch, 128 steps, Class-IV gzip band
DYAD_CFG = DyadicConfig(nca=NCAConfig(grid_size=16, n_groups=2, num_steps=30,
                                      burn_in=4, filter_enabled=False))
# v2 (--dense-events): ~5x supervision density — the v1 ledger floor was
# signal starvation (~0.1% of loss on answer tokens). Shorter episodes, more
# commits, re-queries at fresh lags (induction-assisted; first-queries stay
# the pure-retrieval read).
DYAD_CFG_V2 = DyadicConfig(
    nca=NCAConfig(grid_size=16, n_groups=2, num_steps=20, burn_in=4,
                  filter_enabled=False),
    turn_len_min=2, turn_len_max=4,
    commits_per_traj=(6, 10), queries_per_traj=(8, 16),
    query_replacement=True, min_query_lag_steps=3)


def make_rule(cfg: NCAConfig, device: torch.device) -> NCARule:
    rc = sample_rule_config(cfg)
    arch = {k: rc[k] for k in ("kernel_size", "hidden_dim", "num_hidden_layers")}
    return NCARule(d_state=cfg.d_state, n_groups=cfg.n_groups, **arch).to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--target-tokens", type=int, default=125_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1021)
    ap.add_argument("--finalize", action="store_true",
                    help="concat shards + emit eval items, then exit")
    ap.add_argument("--n-eval-ledger", type=int, default=400)
    ap.add_argument("--n-eval-smoothing", type=int, default=200)
    ap.add_argument("--dense-events", action="store_true",
                    help="v2 density (see DYAD_CFG_V2)")
    args = ap.parse_args()
    global DYAD_CFG
    if args.dense_events:
        DYAD_CFG = DYAD_CFG_V2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        shards = sorted(out.glob("shard_*.bin"))
        with (out / "p5_dyadic.bin").open("wb") as f:
            for s in shards:
                f.write(s.read_bytes())
        total = (out / "p5_dyadic.bin").stat().st_size // 2
        # held-out rules for eval (seed offset far from any shard)
        rng = random.Random(args.seed + 999_983)
        torch.manual_seed(args.seed + 999_983)
        pairs = [(make_rule(DYAD_CFG.nca, torch.device("cpu")),
                  make_rule(DYAD_CFG.nca, torch.device("cpu"))) for _ in range(8)]
        led = emit_ledger_items(DYAD_CFG, pairs, args.n_eval_ledger,
                                seed0=5_000_000)
        smo = emit_smoothing_pairs(DYAD_CFG, pairs, args.n_eval_smoothing,
                                   seed0=6_000_000)
        (out / "eval_ledger.jsonl").write_text(
            "\n".join(json.dumps(x) for x in led))
        (out / "eval_smoothing.jsonl").write_text(
            "\n".join(json.dumps(x) for x in smo))
        meta = {"total_tokens": int(total), "n_shards": len(shards),
                "orig_frac": ORIG_FRac, "n_ledger": len(led), "n_smo": len(smo)}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed0 = args.seed + args.shard * 1_000_003
    rng = random.Random(seed0)
    torch.manual_seed(seed0)
    written = 0
    t0 = time.time()
    with (out / f"shard_{args.shard}.bin").open("wb") as f:
        traj_i = 0
        while written < args.target_tokens:
            traj_i += 1
            if rng.random() < ORIG_FRac:
                rule = make_rule(ORIG_CFG, device)
                traj = simulate_trajectory(
                    rule, ORIG_CFG.grid_size, ORIG_CFG.d_state,
                    ORIG_CFG.n_groups, ORIG_CFG.num_steps, ORIG_CFG.burn_in,
                    ORIG_CFG.identity_bias, ORIG_CFG.temperature,
                    batch_size=1, device=device)
                toks = tokenize_trajectory(traj, ORIG_CFG.d_state,
                                           ORIG_CFG.patch_size)
            else:
                rs = make_rule(DYAD_CFG.nca, device)
                ro = make_rule(DYAD_CFG.nca, device)
                dt = simulate_dyadic(DYAD_CFG, rs, ro,
                                     seed=seed0 + traj_i, device=device)
                toks = serialize_dyadic(dt, DYAD_CFG)
            f.write(toks.astype(np.uint16).tobytes())
            written += len(toks)
            if traj_i % 200 == 0:
                rate = written / max(time.time() - t0, 1e-6)
                print(f"shard {args.shard}: {written/1e6:.1f}M tok "
                      f"({rate/1e3:.0f}K/s, eta "
                      f"{(args.target_tokens-written)/max(rate,1):.0f}s)",
                      flush=True)
    print(f"shard {args.shard} done: {written} tokens", flush=True)


if __name__ == "__main__":
    main()
