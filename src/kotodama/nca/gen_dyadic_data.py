"""NCA-v3 dataset writer — mixed aliveness + dyadic training bin, with the
2026-09-26 fixes (SPEC-1B-FORMATION-LADDER §5; supersedes the ARM-D-lite
writer, whose bins live on in git history / the frozen snapshot):

  R1  token-level mix control — ``--orig-frac`` is a fraction of TOKENS,
      enforced by a debt scheduler (the ARM-D-lite writer drew the mix per
      TRAJECTORY: its "70% dyadic" stream was ~5-7% dyadic by tokens).
  R2  last-write-wins ledger + ``--tag-reuse-start/-end`` ramp (unique-tag
      retrieval early → overwrite/state-update demand late; ruling 2026-09-26).
  R3  every episode ends with the EOS 0 separator (``reserve_eos`` alphabet,
      ids shifted +1) — train with ``doc_masking: true`` and episodes stop
      bleeding across windows; eval prefixes finally match training.
  R4  fully seeded: per-episode RNG streams for architecture/band/marking
      draws; global ``random`` is never consumed; two runs with the same args
      are byte-identical.
  R5  sidecar JSONL per shard — every episode's family, offsets, geometry,
      band, per-rule arch/dynamics/gzip, ramp value, marking, event counts.
  R6  gzip band filtering applied to EVERY rule (both families — ARM-D-lite's
      "Class-IV core" was never filtered), with band quotas via ``--bands``.
  R9  mixed-geometry quotas for the aliveness family (``--geometries``) with
      visibility recorded (steps a 4096 window can see).

Turn marking: ``--p-unmarked`` of dyadic episodes drop TURN tokens (handover
must be detected from dynamics — ruling 2026-09-26; the marked/unmarked
contrast is a free in-domain gate).

Usage (sharded across GPUs; CPU works for tests/small doses):
  for s in 0..5: CUDA_VISIBLE_DEVICES=$s python -m kotodama.nca.gen_dyadic_data \
      --shard $s --n-shards 6 --target-tokens 125_000_000 --out <dir> &
  python -m kotodama.nca.gen_dyadic_data --finalize --out <dir>   # concat +
      merged sidecar + eval sets (ledger_unique / ledger_overwrite /
      smoothing) + meta.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .generator import (NCAConfig, NCARule, evaluate_rule_complexity,
                        sample_rule_config, simulate_trajectory,
                        tokenize_trajectory)
from .dyadic import (DyadicConfig, emit_ledger_items, emit_smoothing_pairs,
                     serialize_dyadic, simulate_dyadic)

EOS_ID = 0
WINDOW = 4096  # training context, for the visibility numbers in the sidecar


# ── Quota parsing ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Band:
    lo: float
    hi: float
    weight: float = 1.0


@dataclass(frozen=True)
class Geometry:
    grid: int
    groups: int
    weight: float = 1.0


def parse_bands(spec: str) -> list[Band]:
    out = []
    for part in spec.split(","):
        bits = part.split(":")
        out.append(Band(float(bits[0]), float(bits[1]),
                        float(bits[2]) if len(bits) > 2 else 1.0))
    return out


def parse_geometries(spec: str) -> list[Geometry]:
    out = []
    for part in spec.split(","):
        gspec, _, w = part.partition(":")
        g, grp = gspec.lower().split("x")[0], gspec.lower().split("x")[-1]
        out.append(Geometry(int(g), int(grp), float(w) if w else 1.0))
    return out


def parse_range(spec: str) -> tuple[int, int]:
    lo, _, hi = spec.partition(":")
    return int(lo), int(hi or lo)


def _weighted(rng: random.Random, items):
    total = sum(i.weight for i in items)
    x = rng.random() * total
    for i in items:
        x -= i.weight
        if x <= 0:
            return i
    return items[-1]


# ── Banded rule sampling (R6) ─────────────────────────────────────────────────

def banded_rule(cfg: NCAConfig, rng: random.Random, device: torch.device,
                max_attempts: int) -> tuple[NCARule, dict, float, int, bool]:
    """Rejection-sample a rule whose solo-rollout gzip complexity falls in
    [cfg.gzip_lower, cfg.gzip_upper]. On exhaustion, returns the nearest-miss
    candidate with in_band=False (recorded in the sidecar, never silent)."""
    best = None
    best_dist = float("inf")
    for attempt in range(1, max_attempts + 1):
        params = sample_rule_config(cfg, rng=rng)
        rule = NCARule(d_state=cfg.d_state, n_groups=cfg.n_groups,
                       kernel_size=params["kernel_size"],
                       hidden_dim=params["hidden_dim"],
                       num_hidden_layers=params["num_hidden_layers"]).to(device)
        gz = evaluate_rule_complexity(rule, cfg, params, device)
        if cfg.gzip_lower <= gz <= cfg.gzip_upper:
            return rule, params, gz, attempt, True
        dist = max(cfg.gzip_lower - gz, gz - cfg.gzip_upper)
        if dist < best_dist:
            best, best_dist = (rule, params, gz), dist
    rule, params, gz = best
    return rule, params, gz, max_attempts, False


# ── Config assembly ───────────────────────────────────────────────────────────

def dyad_config(args, band: Band, p_tag_reuse: float,
                mark_turns: bool) -> DyadicConfig:
    return DyadicConfig(
        nca=NCAConfig(grid_size=args.dyad_grid, n_groups=args.dyad_groups,
                      num_steps=args.dyad_steps, burn_in=args.dyad_burn_in,
                      gzip_lower=band.lo, gzip_upper=band.hi,
                      reserve_eos=True),
        turn_len_min=parse_range(args.turn_len)[0],
        turn_len_max=parse_range(args.turn_len)[1],
        mark_turns=mark_turns,
        commits_per_traj=parse_range(args.commits),
        queries_per_traj=parse_range(args.queries),
        p_tag_reuse=p_tag_reuse,
        min_query_lag_steps=args.min_lag,
        other_identity_bias_boost=args.other_bias_boost,
    )


# ── Writer ────────────────────────────────────────────────────────────────────

def write_shard(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed0 = args.seed + args.shard * 1_000_003
    bands = parse_bands(args.bands)
    geometries = parse_geometries(args.geometries)

    written = w_orig = 0
    n_off_band = 0
    t0 = time.time()
    bin_path = out / f"shard_{args.shard}.bin"
    side_path = out / f"shard_{args.shard}.sidecar.jsonl"
    with bin_path.open("wb") as f, side_path.open("w") as sc:
        ep = 0
        while written < args.target_tokens:
            ep += 1
            ep_seed = seed0 + ep
            rng = random.Random(ep_seed)          # all per-episode choices (R4)
            torch.manual_seed(ep_seed * 2 + 1)    # rule-weight stream
            band = _weighted(rng, bands)

            # R1: token-debt mix — pick the family that is behind its target
            orig_turn = (rng.random() < args.orig_frac if written == 0
                         else (w_orig / written) < args.orig_frac)
            rec: dict = {"ep": ep, "seed": ep_seed, "band": [band.lo, band.hi]}

            if orig_turn:
                geo = _weighted(rng, geometries)
                cfg = NCAConfig(grid_size=geo.grid, n_groups=geo.groups,
                                num_steps=args.orig_steps,
                                gzip_lower=band.lo, gzip_upper=band.hi,
                                reserve_eos=True)
                rule, params, gz, tries, ok = banded_rule(
                    cfg, rng, device, args.filter_attempts)
                n_off_band += 0 if ok else 1
                torch.manual_seed(ep_seed)  # simulation stream
                traj = simulate_trajectory(
                    rule, cfg.grid_size, cfg.d_state, cfg.n_groups,
                    cfg.num_steps, cfg.burn_in,
                    params["identity_bias"], params["temperature"],
                    batch_size=1, device=device)
                toks = tokenize_trajectory(traj, cfg.d_state, cfg.patch_size,
                                           offset=1)
                rec.update({
                    "family": "orig",
                    "geometry": {"grid": geo.grid, "groups": geo.groups,
                                 "steps": cfg.num_steps},
                    "steps_per_window": round(WINDOW / cfg.tokens_per_timestep, 2),
                    "rules": [{**params, "gzip": round(gz, 4),
                               "attempts": tries, "in_band": ok}],
                })
            else:
                progress = min(written / max(args.target_tokens, 1), 1.0)
                p_reuse = (args.tag_reuse_start
                           + (args.tag_reuse_end - args.tag_reuse_start)
                           * progress)
                marked = rng.random() >= args.p_unmarked
                dcfg = dyad_config(args, band, p_reuse, marked)
                rule_dyn = []
                pair = []
                for _ in range(2):
                    r, p, gz, tries, ok = banded_rule(
                        dcfg.nca, rng, device, args.filter_attempts)
                    n_off_band += 0 if ok else 1
                    pair.append((r, p))
                    rule_dyn.append({**p, "gzip": round(gz, 4),
                                     "attempts": tries, "in_band": ok})
                dt = simulate_dyadic(
                    dcfg, pair[0][0], pair[1][0], seed=ep_seed, device=device,
                    dyn_self=pair[0][1], dyn_other=pair[1][1])
                toks = serialize_dyadic(dt, dcfg)
                qs = [e for e in dt.events if e["kind"] == "query"]
                rec.update({
                    "family": "dyadic",
                    "geometry": {"grid": dcfg.nca.grid_size,
                                 "groups": dcfg.nca.n_groups,
                                 "steps": dcfg.nca.num_steps},
                    "steps_per_window": round(
                        WINDOW / dcfg.nca.tokens_per_timestep, 2),
                    "rules": rule_dyn,
                    "p_tag_reuse": round(p_reuse, 4),
                    "marked": marked,
                    "n_commits": sum(1 for e in dt.events
                                     if e["kind"] == "commit"),
                    "n_queries": len(qs),
                    "n_post_overwrite": sum(1 for e in qs
                                            if e["post_overwrite"]),
                    "perturbed": any(e["kind"] == "perturb"
                                     for e in dt.events),
                })

            assert toks.size > 0 and int(toks.min()) > EOS_ID, \
                "EOS id leaked into an episode body"
            rec["start"], rec["end"] = written, written + len(toks)
            f.write(toks.astype(np.uint16).tobytes())
            f.write(np.array([EOS_ID], dtype=np.uint16).tobytes())  # R3
            written += len(toks) + 1
            if orig_turn:
                w_orig += len(toks) + 1
            sc.write(json.dumps(rec) + "\n")

            if ep % 100 == 0:
                rate = written / max(time.time() - t0, 1e-6)
                print(f"shard {args.shard}: ep {ep}, {written/1e6:.1f}M tok, "
                      f"orig {w_orig/max(written,1):.2%} "
                      f"(target {args.orig_frac:.0%}), {rate/1e3:.0f}K tok/s",
                      flush=True)

    print(f"shard {args.shard} done: {written} tokens, {ep} episodes, "
          f"orig {w_orig/max(written,1):.2%}, off-band rules {n_off_band}",
          flush=True)


# ── Finalize: concat + merged sidecar + eval sets + meta ─────────────────────

def finalize(args) -> None:
    out = Path(args.out)
    shards = sorted(out.glob("shard_*.bin"))
    offsets = []
    total = 0
    with (out / "nca_v3.bin").open("wb") as f:
        for s in shards:
            offsets.append(total)
            f.write(s.read_bytes())
            total += s.stat().st_size // 2
    with (out / "sidecar.jsonl").open("w") as merged:
        for s, off in zip(shards, offsets):
            side = s.parent / (s.stem + ".sidecar.jsonl")
            for line in side.read_text().splitlines():
                rec = json.loads(line)
                rec["start"] += off
                rec["end"] += off
                rec["shard"] = s.stem
                merged.write(json.dumps(rec) + "\n")

    # eval sets: held-out banded rules, both ledger regimes (ruling: report
    # unique vs overwrite separately) + smoothing pairs
    device = torch.device("cpu")
    band = parse_bands(args.bands)[0]
    rng = random.Random(args.seed + 999_983)
    torch.manual_seed(args.seed + 999_983)
    pairs = []
    for _ in range(args.n_eval_rule_pairs):
        rs, ps, *_ = banded_rule(
            dyad_config(args, band, 0.0, True).nca, rng, device,
            args.filter_attempts)
        ro, po, *_ = banded_rule(
            dyad_config(args, band, 0.0, True).nca, rng, device,
            args.filter_attempts)
        pairs.append((rs, ro, ps, po))
    evals = {
        "eval_ledger_unique.jsonl": emit_ledger_items(
            dyad_config(args, band, 0.0, True), pairs,
            args.n_eval_ledger, seed0=5_000_000),
        "eval_ledger_overwrite.jsonl": emit_ledger_items(
            dyad_config(args, band, max(args.tag_reuse_end, 0.99), True), pairs,
            args.n_eval_ledger, seed0=5_500_000),
        "eval_smoothing.jsonl": emit_smoothing_pairs(
            dyad_config(args, band, 0.0, True), pairs,
            args.n_eval_smoothing, seed0=6_000_000),
    }
    for name, items in evals.items():
        (out / name).write_text("\n".join(json.dumps(x) for x in items))

    meta = {"total_tokens": int(total), "n_shards": len(shards),
            "eos_id": EOS_ID, "reserve_eos": True,
            "args": {k: v for k, v in vars(args).items() if k != "finalize"},
            "eval_counts": {k: len(v) for k, v in evals.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--target-tokens", type=int, default=125_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1021)
    # mix (R1) — fraction of TOKENS
    ap.add_argument("--orig-frac", type=float, default=0.30)
    # ledger ramp (R2, ruling: unique → overwrite)
    ap.add_argument("--tag-reuse-start", type=float, default=0.0)
    ap.add_argument("--tag-reuse-end", type=float, default=0.5)
    # turn marking mix (ruling: ~half unmarked)
    ap.add_argument("--p-unmarked", type=float, default=0.5)
    # complexity bands (R6): "lo:hi[:weight],..."
    ap.add_argument("--bands", default="0.4:0.7")
    # aliveness-family geometry quotas (R9): "GRIDxGRIDxGROUPS:weight,..."
    ap.add_argument("--geometries",
                    default="16x16x2:0.35,16x16x4:0.35,32x32x2:0.15,32x32x4:0.15")
    ap.add_argument("--orig-steps", type=int, default=128)
    # dyadic family shape (fits inside one window by design)
    ap.add_argument("--dyad-grid", type=int, default=16)
    ap.add_argument("--dyad-groups", type=int, default=2)
    ap.add_argument("--dyad-steps", type=int, default=20)
    ap.add_argument("--dyad-burn-in", type=int, default=4)
    ap.add_argument("--turn-len", default="2:4")
    ap.add_argument("--commits", default="6:10")
    ap.add_argument("--queries", default="8:16")
    ap.add_argument("--min-lag", type=int, default=3)
    ap.add_argument("--other-bias-boost", type=float, default=1.0)
    ap.add_argument("--filter-attempts", type=int, default=200)
    # finalize
    ap.add_argument("--finalize", action="store_true",
                    help="concat shards + merged sidecar + eval sets, then exit")
    ap.add_argument("--n-eval-rule-pairs", type=int, default=8)
    ap.add_argument("--n-eval-ledger", type=int, default=400)
    ap.add_argument("--n-eval-smoothing", type=int, default=200)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.finalize:
        finalize(args)
    else:
        write_shard(args)


if __name__ == "__main__":
    main()
