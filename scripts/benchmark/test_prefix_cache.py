#!/usr/bin/env python3
"""Prefix-cache correctness battery for DecodeEngine.prefill_cached.

The failure mode this guards against: a stale or mismatched cache silently
generating against the wrong conversation — poisoned distillation data with
no error anywhere. Three parts:

  1. unit       — match/rollback state machine on a tiny fp32 random model
                  (exact, fast, no checkpoint, no GPU contention concerns)
  2. parity     — real checkpoint: multi-turn conversations, extend-path
                  logits vs full-re-prefill logits, teacher-forced at every
                  turn; deltas must sit in the established noise band
  3. soak       — randomized multi-turn sessions (extend/undo/regenerate/
                  divergent-swap) repeating part-2 gates per turn

Usage:
    python scripts/benchmark/test_prefix_cache.py --unit-only          # anywhere
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark/test_prefix_cache.py \
        --checkpoint /models/.../3b-language-FINAL-step195311.pt [--compile] [--seeds 5]
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

torch.set_num_threads(2)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark.decode_parity import CONFIG_3B, load_model  # noqa: E402
from src.model.decode_engine import DecodeEngine  # noqa: E402
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig  # noqa: E402

# Gates for the real-checkpoint soak. What distillation samples is the HEAD
# of the distribution, so the noise-floor-class bound applies to the top-K
# logits (union of both sides' top-64). The global max over all 49k logits is
# dominated by chaotic near-zero-probability tail tokens whose deltas grow
# with KV-perturbation exposure (measured: tail max 1-2 with argmax PERFECT
# and top-64 deltas at floor level) — it only gets a gross-breakage ceiling
# (real positional/mask bugs measure O(5-30) there WITH argmax chaos).
# NOTE: the soak feeds RANDOM-TOKEN conversations — off-manifold inputs where
# routing/attention dynamics amplify chaotically (a deliberately harsher
# stress than natural text). Measured across 36 turns x 6 seeds: 35 turns at
# 0.19-0.66 (inside the 0.42-0.63 backend noise floor + headroom), one
# deterministic outlier at 0.96 with argmax fully intact. A systematic bug
# (rope offset / mask / KV slice) hits EVERY suffix turn, not one seed.
GATE_TOP64_DELTA_3B = 1.25
GATE_GLOBAL_DELTA_3B = 4.0     # gross-breakage ceiling on the full-vocab max
GATE_GREEDY_AGREE = 0.95       # aggregate free-greedy agreement (near-tie flips only)
# Tiny fp32 model, comparisons where BOTH sides' KV came from the reference
# batch path (only attention backend / mask form differs): near-exact.
GATE_MAX_DELTA_SMALL = 1e-3
# Tiny fp32 model, comparisons where the reused prefix contains STEP-written
# KV rows: the decode step is a different kernel path than batch prefill
# (round-1 equivalence class = bf16 noise floor, not exactness), and a
# random-init model amplifies the mix chaotically through routing. This gate
# only needs to catch gross breakage — a wrong rope offset / mask / position
# produces O(1)+ errors, measured backend-mix noise here is ~2e-2.
GATE_MAX_DELTA_SMALL_STEPMIX = 0.1

_FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        logger.info("  PASS  %s", msg)
    else:
        logger.error("  FAIL  %s", msg)
        _FAILURES.append(msg)


# ───────────────────────────── part 1: unit ─────────────────────────────────

SMALL_CONFIG = dict(
    hidden_size=256, num_layers=4, num_attention_heads=4, num_kv_heads=2,
    head_dim=64, intermediate_size=512, vocab_size=1024,
    max_position_embeddings=512, rope_theta=10000.0, norm_eps=1e-5,
    qk_norm=True, tie_word_embeddings=True, z_loss_weight=0.0,
    use_liger=False, attn_impl="sdpa", attn_res=True,
    attn_res_boundaries=[0, 1],
)


def small_engine(device: torch.device, max_seq_len: int = 128) -> DecodeEngine:
    from torch.nn.attention import SDPBackend

    torch.manual_seed(7)
    model = LuxiaBaseModel(LuxiaModelConfig(**SMALL_CONFIG)).to(device).eval()
    # fp32 + GQA has no flash/efficient kernel — MATH is exact, ideal for units.
    return DecodeEngine(
        model, max_seq_len=max_seq_len, dtype=torch.float32,
        prefill_backends=[SDPBackend.MATH],
    )


def rand_ids(n: int, seed: int, vocab: int = 1024) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, vocab, (1, n), generator=g)


@torch.inference_mode()
def greedy_steps(engine: DecodeEngine, first_logits: torch.Tensor, n: int) -> list[int]:
    """Greedy-decode n tokens via step_logits (records history in-graph)."""
    tok = first_logits.argmax(-1).view(1, 1)
    out = [int(tok.item())]
    for _ in range(n - 1):
        logits = engine.step_logits(tok)
        tok = logits[0, -1].argmax().view(1, 1)
        out.append(int(tok.item()))
    return out


@torch.inference_mode()
def run_unit(device: torch.device) -> None:
    logger.info("== part 1: unit tests (tiny fp32 model) ==")
    eng = small_engine(device)
    eng.min_cached_prefix = 8  # small prompts in these tests

    # fresh engine -> miss
    p1 = rand_ids(32, seed=1)
    logits, info = eng.prefill_cached(p1)
    check(info["prefix_hit"] is False and eng._pos_cpu == 32, "fresh engine: miss + pos=32")
    check(bool((eng.token_history[:32].cpu() == p1[0]).all()), "history == prompt after miss-prefill")

    # generate a few; history must track consumed tokens (tip not in cache)
    gen = greedy_steps(eng, logits, 5)
    # consumed: gen[0..3] forwarded by the 4 step_logits calls; gen[4] is the tip
    check(eng._pos_cpu == 32 + 4, "pos advanced by consumed tokens only")
    check(
        bool((eng.token_history[32:36].cpu() == torch.tensor(gen[:4])).all()),
        "history records consumed generated tokens in-graph",
    )

    # full-continuation -> hit, suffix = generated tip + new turn
    p2 = torch.cat([p1, torch.tensor(gen).view(1, -1), rand_ids(16, seed=2)], dim=1)
    logits2, info2 = eng.prefill_cached(p2)
    check(info2["prefix_hit"] is True, "continuation: hit")
    check(info2["common_prefix"] == 36, f"continuation: common=36 (got {info2['common_prefix']})")
    check(info2["suffix_len"] == p2.shape[1] - 36, "continuation: suffix = tip + new turn")
    check(eng._pos_cpu == p2.shape[1], "continuation: pos == new prompt len")
    # vs fresh full prefill on an identical second engine
    eng_ctl = small_engine(device)
    ref_logits = eng_ctl.prefill(p2)
    d = float((logits2 - ref_logits).abs().max())
    # Reused prefix includes step-written KV rows -> step-vs-batch class gate.
    check(d < GATE_MAX_DELTA_SMALL_STEPMIX,
          f"continuation: extend logits ~= full prefill, step-KV in prefix (maxD={d:.2e})")

    # divergent mid-prompt -> partial hit at divergence point
    p3 = p2.clone()
    p3[0, 20] = (p3[0, 20] + 1) % 1024
    _, info3 = eng.prefill_cached(p3)
    check(info3["prefix_hit"] is True and info3["common_prefix"] == 20,
          f"divergence@20: common=20 (got {info3['common_prefix']})")

    # shrunken prefix (undo): shorter prompt sharing a prefix
    p4 = torch.cat([p3[:, :24], rand_ids(8, seed=3)], dim=1)  # 32 tokens
    _, info4 = eng.prefill_cached(p4)
    check(info4["prefix_hit"] is True and info4["common_prefix"] == 24, "undo: rollback to common=24")
    check(eng._pos_cpu == 32, "undo: pos == shrunken prompt len")
    bias = eng.attn_bias[0, 0, 0]
    check(bool((bias[:32] == 0).all()) and bool((bias[32:] == float("-inf")).all()),
          "undo: bias open exactly [0, 32)")
    ref4 = eng_ctl.prefill(p4)
    # re-extend after undo must still match a fresh prefill; identical prompt
    # -> suffix==1 -> routed through the decode STEP path (step-vs-batch class)
    logits4, _ = eng.prefill_cached(p4)
    d4 = float((logits4 - ref4).abs().max())
    check(d4 < GATE_MAX_DELTA_SMALL_STEPMIX, f"undo+regenerate via step path (maxD={d4:.2e})")

    # identical prompt (regenerate): common capped at plen-1
    _, info5 = eng.prefill_cached(p4)
    check(info5["prefix_hit"] is True and info5["common_prefix"] == p4.shape[1] - 1
          and info5["suffix_len"] == 1, "regenerate: common=plen-1, suffix=1")

    # near-total divergence -> miss
    p6 = rand_ids(32, seed=99)
    _, info6 = eng.prefill_cached(p6)
    check(info6["prefix_hit"] is False, "divergent prompt: miss")

    # min_cached_prefix boundary
    eng.prefill(p1)
    p7 = torch.cat([p1[:, :7], rand_ids(25, seed=5)], dim=1)   # common 7 < 8
    _, info7 = eng.prefill_cached(p7)
    check(info7["prefix_hit"] is False, "common < min_cached_prefix: miss")
    eng.prefill(p1)
    p8 = torch.cat([p1[:, :8], rand_ids(24, seed=6)], dim=1)   # common 8 >= 8
    _, info8 = eng.prefill_cached(p8)
    check(info8["prefix_hit"] is True and info8["common_prefix"] == 8,
          "common == min_cached_prefix: hit")

    # overflow: too-long prompt raises and resets
    try:
        eng.prefill_cached(rand_ids(128, seed=7))
        check(False, "overflow raises")
    except ValueError:
        check(eng._pos_cpu == 0, "overflow: raised + state reset")

    # capacity for subsequent decode unaffected: extend to near-full then step to cap
    eng2 = small_engine(device, max_seq_len=64)
    eng2.min_cached_prefix = 8
    eng2.prefill(rand_ids(40, seed=8))
    big = torch.cat([rand_ids(40, seed=8), rand_ids(23, seed=9)], dim=1)  # 63 = max-1
    _, infob = eng2.prefill_cached(big)
    check(infob["prefix_hit"] is True and eng2._pos_cpu == 63, "extend to max_seq_len-1 ok")
    eng2.step_logits(torch.tensor([[5]], device=device))  # consumes slot 63
    try:
        eng2.step_logits(torch.tensor([[5]], device=device))
        check(False, "capacity assert after extend")
    except RuntimeError:
        check(True, "capacity assert after extend")


# ──────────────────────── parts 2+3: real checkpoint ────────────────────────

@torch.inference_mode()
def forced_step_logits(engine: DecodeEngine, tokens: list[int], device: torch.device) -> list[torch.Tensor]:
    out = []
    for t in tokens:
        logits = engine.step_logits(torch.tensor([[t]], device=device))
        out.append(logits[0, -1].float().clone().cpu())
    return out


def top_delta(a: torch.Tensor, b: torch.Tensor, k: int = 64) -> float:
    """max |a-b| over the union of both sides' top-k token indices."""
    idx = torch.cat([a.topk(k).indices, b.topk(k).indices]).unique()
    return float((a[idx] - b[idx]).abs().max())


@torch.inference_mode()
def greedy_with_logits(
    engine: DecodeEngine, first_logits: torch.Tensor, n: int
) -> tuple[list[int], list[torch.Tensor]]:
    """Greedy decode recording per-step logits (cloned out of graph memory)."""
    tok = first_logits.argmax(-1).view(1, 1)
    toks = [int(tok.item())]
    step_logits: list[torch.Tensor] = []
    for _ in range(n - 1):
        logits = engine.step_logits(tok)
        step_logits.append(logits[0, -1].float().clone().cpu())
        tok = logits[0, -1].argmax().view(1, 1)
        toks.append(int(tok.item()))
    return toks, step_logits


@torch.inference_mode()
def run_conversation(
    cached: DecodeEngine,
    control: DecodeEngine,
    device: torch.device,
    seed: int,
    n_turns: int,
    gen_tokens: int,
    vocab: int,
    randomize_ops: bool,
) -> dict[str, Any]:
    """One multi-turn session; per-turn extend-vs-full forced-logit deltas."""
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)

    def rand_turn(n: int) -> torch.Tensor:
        return torch.randint(4, vocab, (1, n), generator=g)

    prompt = rand_turn(48 + rng.randrange(64))
    stats: dict[str, Any] = {"turns": [], "max_delta": 0.0, "hits": 0}

    for turn in range(n_turns):
        # cached path: prefill_cached + greedy decode, recording logits
        logits_c, info = cached.prefill_cached(prompt)
        gen, steps_c = greedy_with_logits(cached, logits_c, gen_tokens)

        # control: full prefill + teacher-force the SAME token stream, so
        # every step's logits are directly comparable.
        logits_f = control.prefill(prompt)
        forced = forced_step_logits(control, gen[:-1], device)
        pairs = [(logits_c[0], logits_f[0])] + list(zip(steps_c, forced))
        d_top = max(top_delta(a.float(), b.float()) for a, b in pairs)
        d_global = max(float((a - b).abs().max()) for a, b in pairs)
        agree = sum(int(int(l.argmax()) == t) for l, t in zip(forced, gen[1:]))
        stats["turns"].append({
            "turn": turn, "plen": int(prompt.shape[1]), **info,
            "top64_delta": round(d_top, 4),
            "global_delta": round(d_global, 4),
            "greedy_agree": f"{agree}/{len(forced)}",
        })
        stats["max_top"] = max(stats.get("max_top", 0.0), d_top)
        stats["max_delta"] = max(stats["max_delta"], d_global)
        stats["agree_n"] = stats.get("agree_n", 0) + agree
        stats["agree_d"] = stats.get("agree_d", 0) + len(forced)
        stats["hits"] += int(info["prefix_hit"])

        # build next prompt
        full = torch.cat([prompt, torch.tensor(gen).view(1, -1)], dim=1)
        op = rng.choice(["extend", "extend", "undo", "regen"]) if (randomize_ops and turn > 0) else "extend"
        if op == "extend":
            prompt = torch.cat([full, rand_turn(24 + rng.randrange(48))], dim=1)
        elif op == "undo":
            keep = max(32, int(full.shape[1] * 0.6))
            prompt = torch.cat([full[:, :keep], rand_turn(24)], dim=1)
        else:  # regen: same prompt again
            prompt = full
        if prompt.shape[1] > cached.max_seq_len - gen_tokens - 8:
            break
    return stats


def run_real(checkpoint: str, device: torch.device, compile_engine: bool,
             seeds: int, n_turns: int) -> None:
    logger.info("== parts 2+3: real-checkpoint parity + soak ==")
    model = load_model(checkpoint, device)
    cached = DecodeEngine(model)
    control = DecodeEngine(model)
    if compile_engine:
        cached.compile_step("max-autotune")
        control.compile_step("max-autotune")
        ids = torch.randint(4, model.config.vocab_size, (1, 16)).to(device)
        cached.prefill(ids); cached.step_logits(torch.tensor([[100]], device=device))
        control.prefill(ids); control.step_logits(torch.tensor([[100]], device=device))

    vocab = model.config.vocab_size
    worst_top = 0.0
    worst_global = 0.0
    total_hits = 0
    agree_n = agree_d = 0
    for s in range(seeds):
        stats = run_conversation(
            cached, control, device, seed=1000 + s, n_turns=n_turns,
            gen_tokens=24, vocab=vocab, randomize_ops=(s > 0),
        )
        for t in stats["turns"]:
            logger.info(
                "  seed=%d turn=%d plen=%4d hit=%-5s common=%4d suffix=%3d "
                "top64D=%.4f globalD=%.4f agree=%s",
                1000 + s, t["turn"], t["plen"], t["prefix_hit"], t["common_prefix"],
                t["suffix_len"], t["top64_delta"], t["global_delta"], t["greedy_agree"],
            )
        worst_top = max(worst_top, stats.get("max_top", 0.0))
        worst_global = max(worst_global, stats["max_delta"])
        agree_n += stats.get("agree_n", 0)
        agree_d += stats.get("agree_d", 0)
        total_hits += stats["hits"]
    check(worst_top <= GATE_TOP64_DELTA_3B,
          f"soak: worst top-64 delta {worst_top:.4f} <= {GATE_TOP64_DELTA_3B} (noise-floor class)")
    check(worst_global <= GATE_GLOBAL_DELTA_3B,
          f"soak: worst global delta {worst_global:.4f} <= {GATE_GLOBAL_DELTA_3B} (gross-breakage ceiling)")
    agree_frac = agree_n / max(agree_d, 1)
    check(agree_frac >= GATE_GREEDY_AGREE,
          f"soak: greedy agreement {agree_n}/{agree_d} ({agree_frac:.3f}) >= {GATE_GREEDY_AGREE}")
    check(total_hits >= seeds * (n_turns - 1) * 0.7,
          f"soak: cache actually exercised (hits={total_hits})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_unit(device)
    if not args.unit_only:
        if not args.checkpoint:
            parser.error("--checkpoint required unless --unit-only")
        run_real(args.checkpoint, device, args.compile, args.seeds, args.turns)

    if _FAILURES:
        logger.error("OVERALL: FAIL (%d)", len(_FAILURES))
        for f in _FAILURES:
            logger.error("  - %s", f)
        return 1
    logger.info("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
