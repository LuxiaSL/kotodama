#!/usr/bin/env python3
"""Step-0 serving profile: decode + prefill + extend, kernel-bucketed.

Re-ranks the serving levers (SERVING-TRANSFER-kernel-findings-2026-07-05.md
S1/S2/S3) by measuring, on one idle GPU:

  1. DECODE (fast engine, compiled max-autotune like serve.py): free-running
     ms/token at ctx {512, 2048} with the fused sampled step, plus a
     torch.profiler window -> kernel-time sum vs wall (dispatch-gap metric)
     and a top-kernel bucket table.
  2. PREFILL (engine.prefill = llama.py reference path; Triton routing per
     the fleet env unless KOTODAMA_NO_TRITON_ATTNRES=1): wall at several
     prompt lengths + profiler bucket split {p1, restack/copy, sdpa, mm, other}.
  3. EXTEND (engine.prefill_cached): common=1024 with suffix {1, 16, 64} —
     the multi-turn / n-of-5 reroll shape SFT curation traffic lives on.
  4. REFERENCE decode step (eager model forward w/ past_kv) — the path every
     truncated-law (top_p/top_k) request falls back to today.
  5. TOP-P COST microbench: full-vocab sort+softmax+cumsum+mask, the marginal
     in-graph cost of un-cliffing truncated laws in the engine.

Emits PROFILE_SUMMARY: {...} on the last line for machine parsing; full JSON
via --output. Read-only wrt serve.py — builds its own engine.

Canonical invocation (gpu-host, via run_py.sh for warm caches + unbuffered logs):
    CUDA_VISIBLE_DEVICES=0 bash tools/run_py.sh scripts/benchmark/profile_engine_v2.py \
        --checkpoint /models/kotodama-data/tmp/kotodama_checkpoints/3b-language-FINAL-step195311.pt \
        --output scripts/benchmark/results/profiles/engine_v2_baseline.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Match serve.py's engine build env (round-1 deploy recipe).
os.environ.setdefault("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", "1")

import torch
from torch.profiler import ProfilerActivity, profile

torch.set_num_threads(2)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark.decode_parity import load_model  # noqa: E402
from src.model.decode_engine import DecodeEngine, SamplingParams  # noqa: E402

RECIPE = SamplingParams(temperature=0.9, repetition_penalty=1.2)


# ── Kernel bucketing ─────────────────────────────────────────────────────────

BUCKETS = [
    ("p1_routing", ("phase_1", "phase1")),
    ("p2_routing", ("phase_2", "phase2")),
    ("attention", ("fmha", "flash", "cudnn", "attention", "sdpa", "attn")),
    ("gemm", ("gemm", "nvjet", "cutlass", "matmul", "_mm", "mm_", "gemv", "addmm", "dot")),
    ("memcpy_stack", ("memcpy", "copy_", "cat_", "stack", "elementwise_kernel_with_index")),
]


def bucket_of(kernel_name: str) -> str:
    low = kernel_name.lower()
    for bucket, needles in BUCKETS:
        if any(n in low for n in needles):
            return bucket
    return "other"


def profile_window(fn, n_iters: int, device: torch.device) -> dict[str, Any]:
    """Run fn() n_iters times under torch.profiler; return wall/kernel split.

    gap_ms = wall - sum(device kernel time): at B=1 single-stream kernels
    barely overlap, so the sum is a fair busy-time proxy; the gap is host
    dispatch + launch + sync overhead.
    """
    torch.cuda.synchronize(device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn()
        torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter() - t0) * 1000

    by_bucket: dict[str, float] = defaultdict(float)
    by_kernel: dict[str, float] = defaultdict(float)
    kernel_events = 0
    for ev in prof.key_averages():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.self_device_time_total > 0:
            ms = ev.self_device_time_total / 1000.0
            by_bucket[bucket_of(ev.key)] += ms
            by_kernel[ev.key] += ms
            kernel_events += ev.count
    kernel_ms = sum(by_bucket.values())
    top = sorted(by_kernel.items(), key=lambda kv: -kv[1])[:15]
    return {
        "iters": n_iters,
        "wall_ms": wall_ms,
        "wall_ms_per_iter": wall_ms / n_iters,
        "kernel_ms": kernel_ms,
        "kernel_ms_per_iter": kernel_ms / n_iters,
        "gap_ms_per_iter": (wall_ms - kernel_ms) / n_iters,
        "gap_frac": (wall_ms - kernel_ms) / wall_ms if wall_ms > 0 else 0.0,
        "kernel_launches_per_iter": kernel_events / n_iters,
        "buckets_ms_per_iter": {k: v / n_iters for k, v in sorted(by_bucket.items(), key=lambda kv: -kv[1])},
        "top_kernels_ms_per_iter": [(k, v / n_iters) for k, v in top],
    }


# ── Section 1+2: decode ──────────────────────────────────────────────────────

def rand_ids(n: int, vocab: int, device: torch.device, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, vocab, (1, n), generator=g).to(device)


@torch.inference_mode()
def bench_decode(engine: DecodeEngine, ctx: int, device: torch.device,
                 timed_steps: int, profile_steps: int) -> dict[str, Any]:
    ids = rand_ids(ctx, engine.config.vocab_size, device)
    engine.prefill(ids)
    tok = engine.sample_first(engine.prefill(ids), RECIPE)
    for _ in range(10):  # warm replay
        tok = engine.step(tok, RECIPE)
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(timed_steps):
        tok = engine.step(tok, RECIPE)
    torch.cuda.synchronize(device)
    ms_sampled = (time.perf_counter() - t0) * 1000 / timed_steps

    # Forward-only (step_logits) to isolate the fused sampler's cost.
    t0 = time.perf_counter()
    for _ in range(timed_steps):
        logits = engine.step_logits(engine.cur_token)
    torch.cuda.synchronize(device)
    ms_forward = (time.perf_counter() - t0) * 1000 / timed_steps
    del logits

    prof = profile_window(lambda: engine.step(engine.cur_token, RECIPE), profile_steps, device)
    return {
        "ctx": ctx,
        "ms_per_tok_sampled": ms_sampled,
        "tok_s_sampled": 1000.0 / ms_sampled,
        "ms_per_tok_forward_only": ms_forward,
        "sampler_ms": ms_sampled - ms_forward,
        "profile": prof,
    }


# ── Section 3: prefill + extend ──────────────────────────────────────────────

@torch.inference_mode()
def bench_prefill(engine: DecodeEngine, lens: list[int], device: torch.device,
                  profile_len: int) -> dict[str, Any]:
    out: dict[str, Any] = {"wall_ms": {}}
    for n in lens:
        ids = rand_ids(n, engine.config.vocab_size, device, seed=100 + n)
        engine.prefill(ids)  # shape warm (cuDNN/Triton bucket)
        times = []
        for _ in range(3):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            engine.prefill(ids)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)
        out["wall_ms"][str(n)] = sorted(times)[1]  # median of 3
    ids = rand_ids(profile_len, engine.config.vocab_size, device, seed=100 + profile_len)
    out["profile_len"] = profile_len
    out["profile"] = profile_window(lambda: engine.prefill(ids), 3, device)
    return out


@torch.inference_mode()
def bench_extend(engine: DecodeEngine, common: int, suffixes: list[int],
                 device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {"common": common, "wall_ms": {}}
    base = rand_ids(common + max(suffixes), engine.config.vocab_size, device, seed=42)
    # Warm both extend variants first (serve.py does this at startup): the
    # cold suffix-1 path measured 54ms vs 3ms warmed (2026-07-05 probe).
    engine.prefill(base[:, :64])
    engine.prefill_cached(base[:, :96])   # multi-token extend
    engine.prefill_cached(base[:, :97])   # suffix-1 (regenerate) variant
    torch.cuda.synchronize(device)
    for suf in suffixes:
        ids = base[:, : common + suf]
        times = []
        for _ in range(4):
            engine.prefill(base[:, :common])          # seed the cache
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _, info = engine.prefill_cached(ids)
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)
            assert info["prefix_hit"], f"expected prefix hit, got {info}"
        out["wall_ms"][str(suf)] = sorted(times)[len(times) // 2]
    return out


# ── Section 4: reference decode step (the truncated-law fallback path) ───────

@torch.inference_mode()
def bench_reference_decode(model, ctx: int, device: torch.device, steps: int) -> dict[str, Any]:
    ids = rand_ids(ctx, model.config.vocab_size, device, seed=13)
    out = model(ids, use_cache=True)
    past_kv = out["past_kv"]
    tok = out["logits"][0, -1:].argmax(-1).view(1, 1)
    for _ in range(5):
        out = model(tok, use_cache=True, past_kv=past_kv)
        past_kv = out["past_kv"]
        tok = out["logits"][0, -1:].argmax(-1).view(1, 1)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        out = model(tok, use_cache=True, past_kv=past_kv)
        past_kv = out["past_kv"]
        tok = out["logits"][0, -1:].argmax(-1).view(1, 1)
    torch.cuda.synchronize(device)
    ms = (time.perf_counter() - t0) * 1000 / steps
    return {"ctx": ctx, "ms_per_tok": ms, "tok_s": 1000.0 / ms}


# ── Section 5: top-p marginal cost ───────────────────────────────────────────

@torch.inference_mode()
def bench_top_p_cost(vocab: int, device: torch.device, iters: int = 50) -> dict[str, Any]:
    """serve.py sample_next_token top-p math (lines ~638-641) on a (V,) tensor."""
    logits = torch.randn(vocab, device=device, dtype=torch.float32)

    def top_p_mask() -> tuple[torch.Tensor, torch.Tensor]:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cumulative = probs.cumsum(dim=-1)
        mask = cumulative - probs >= 0.9
        return sorted_logits.masked_fill(mask, float("-inf")), sorted_idx

    for _ in range(10):
        top_p_mask()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        top_p_mask()
    torch.cuda.synchronize(device)
    return {"vocab": vocab, "us_per_call": (time.perf_counter() - t0) * 1e6 / iters}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--decode-ctx", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--decode-steps", type=int, default=200)
    ap.add_argument("--profile-steps", type=int, default=20)
    ap.add_argument("--prefill-lens", type=int, nargs="+", default=[128, 512, 1024, 3000])
    ap.add_argument("--profile-prefill-len", type=int, default=3000)
    ap.add_argument("--extend-common", type=int, default=1024)
    ap.add_argument("--extend-suffixes", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--reference-steps", type=int, default=30)
    ap.add_argument("--no-compile", action="store_true", help="Skip engine compile (debug only)")
    args = ap.parse_args()

    device = torch.device("cuda")
    triton_routing = not os.environ.get("KOTODAMA_NO_TRITON_ATTNRES")
    logger.info("Triton AttnRes routing (prefill): %s", "ON (fleet config)" if triton_routing else "OFF (eager)")

    model = load_model(args.checkpoint, device)
    engine = DecodeEngine(model, max_seq_len=4096)
    if not args.no_compile:
        t0 = time.perf_counter()
        engine.compile_step(mode="max-autotune")  # matches serve.py:464
        # Trigger the compile with one real step.
        engine.prefill(rand_ids(64, engine.config.vocab_size, device))
        tok = engine.sample_first(engine.prefill(rand_ids(64, engine.config.vocab_size, device)), RECIPE)
        engine.step(tok, RECIPE)
        engine.step_logits(engine.cur_token)
        torch.cuda.synchronize(device)
        logger.info("Engine compiled in %.1fs", time.perf_counter() - t0)

    results: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "triton_routing_prefill": triton_routing,
        "compiled": not args.no_compile,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
    }

    results["decode"] = []
    for ctx in args.decode_ctx:
        logger.info("Decode bench @ ctx=%d ...", ctx)
        r = bench_decode(engine, ctx, device, args.decode_steps, args.profile_steps)
        results["decode"].append(r)
        logger.info(
            "  ctx=%d: %.3f ms/tok (%.0f tok/s) | fwd-only %.3f | sampler %.3f | "
            "kernel %.3f | gap %.3f (%.0f%%) | %.0f launches/tok",
            ctx, r["ms_per_tok_sampled"], r["tok_s_sampled"], r["ms_per_tok_forward_only"],
            r["sampler_ms"], r["profile"]["kernel_ms_per_iter"], r["profile"]["gap_ms_per_iter"],
            r["profile"]["gap_frac"] * 100, r["profile"]["kernel_launches_per_iter"],
        )

    logger.info("Prefill bench (routing=%s) ...", "triton" if triton_routing else "eager")
    results["prefill"] = bench_prefill(engine, args.prefill_lens, device, args.profile_prefill_len)
    for n, ms in results["prefill"]["wall_ms"].items():
        logger.info("  prefill T=%s: %.1f ms", n, ms)
    logger.info("  prefill buckets @T=%d: %s", args.profile_prefill_len,
                {k: round(v, 2) for k, v in results["prefill"]["profile"]["buckets_ms_per_iter"].items()})

    logger.info("Extend bench (common=%d) ...", args.extend_common)
    results["extend"] = bench_extend(engine, args.extend_common, args.extend_suffixes, device)
    for suf, ms in results["extend"]["wall_ms"].items():
        logger.info("  extend suffix=%s: %.1f ms", suf, ms)

    logger.info("Reference (eager) decode step — the truncated-law fallback ...")
    results["reference_decode"] = bench_reference_decode(model, 512, device, args.reference_steps)
    logger.info("  reference: %.1f ms/tok (%.0f tok/s)",
                results["reference_decode"]["ms_per_tok"], results["reference_decode"]["tok_s"])

    results["top_p_cost"] = bench_top_p_cost(model.config.vocab_size, device)
    logger.info("  top-p sort+cumsum on %d vocab: %.0f us/call",
                results["top_p_cost"]["vocab"], results["top_p_cost"]["us_per_call"])

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Wrote %s", out_path)

    d0 = results["decode"][0]
    print("PROFILE_SUMMARY: " + json.dumps({
        "tok_s_ctx" + str(d0["ctx"]): round(d0["tok_s_sampled"], 1),
        "gap_frac": round(d0["profile"]["gap_frac"], 3),
        "launches_per_tok": round(d0["profile"]["kernel_launches_per_iter"], 1),
        "prefill_ms": results["prefill"]["wall_ms"],
        "extend_ms": results["extend"]["wall_ms"],
        "reference_tok_s": round(results["reference_decode"]["tok_s"], 1),
        "top_p_us": round(results["top_p_cost"]["us_per_call"], 1),
        "triton_prefill": triton_routing,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
