#!/usr/bin/env python3
"""Benchmark DecodeEngine under the decode_waterfall protocol.

Measures free-running ms/token at several context lengths, forward-only and
with the on-GPU sampler, eager or compiled. Optionally profiles a window.

Usage:
    KOTODAMA_NO_TRITON_ATTNRES=1 CUDA_VISIBLE_DEVICES=0 python scripts/benchmark/bench_engine.py \
        --checkpoint /models/.../step_00000358.pt --compile --output outputs/profiles/engine_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.profiler import ProfilerActivity, profile

torch.set_num_threads(2)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark.decode_parity import CONFIG_3B, load_model  # noqa: E402
from src.model.decode_engine import DecodeEngine, SamplingParams  # noqa: E402


@torch.inference_mode()
def bench_forward(engine: DecodeEngine, ctx: int, n_tokens: int, device: torch.device) -> float:
    """Free-running forward-only ms/token at a given context length."""
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(4, engine.config.vocab_size, (1, ctx), generator=g).to(device)
    engine.prefill(ids)
    tok = torch.tensor([[100]], device=device)
    # warm a few steps at this context
    for _ in range(5):
        logits = engine.step_logits(tok)
        tok = logits[0, -1:].argmax(-1, keepdim=True)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        logits = engine.step_logits(tok)
        tok = logits[0, -1:].argmax(-1, keepdim=True)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000 / n_tokens


@torch.inference_mode()
def bench_with_sampler(engine: DecodeEngine, ctx: int, n_tokens: int, device: torch.device) -> float:
    """Free-running ms/token including the on-GPU sampler (serving-realistic)."""
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(4, engine.config.vocab_size, (1, ctx), generator=g).to(device)
    params = SamplingParams(temperature=0.9, repetition_penalty=1.2)
    logits = engine.prefill(ids)
    tok = engine.sample_first(logits, params)
    for _ in range(5):
        tok = engine.step(tok, params)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        tok = engine.step(tok, params)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000 / n_tokens


@torch.inference_mode()
def profile_window(engine: DecodeEngine, ctx: int, n_steps: int, device: torch.device) -> dict[str, Any]:
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(4, engine.config.vocab_size, (1, ctx), generator=g).to(device)
    engine.prefill(ids)
    tok = torch.tensor([[100]], device=device)
    for _ in range(3):
        logits = engine.step_logits(tok)
        tok = logits[0, -1:].argmax(-1, keepdim=True)
    torch.cuda.synchronize(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n_steps):
            logits = engine.step_logits(tok)
            tok = logits[0, -1:].argmax(-1, keepdim=True)
        torch.cuda.synchronize(device)

    try:
        cuda_dt = torch.autograd.DeviceType.CUDA
    except AttributeError:
        cuda_dt = None
    total_us = 0.0
    launches = 0
    top: list[tuple[str, float, int]] = []
    for evt in prof.key_averages():
        if cuda_dt is not None and evt.device_type != cuda_dt:
            continue
        us = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
        if us <= 0:
            continue
        total_us += us
        launches += evt.count
        top.append((str(evt.key), us, evt.count))
    top.sort(key=lambda t: -t[1])
    return {
        "kernel_ms_per_step": round(total_us / 1000 / n_steps, 3),
        "kernel_launches_per_step": round(launches / n_steps, 1),
        "top_kernels": [
            {"name": n[:110], "us_per_step": round(us / n_steps, 1), "calls_per_step": round(c / n_steps, 1)}
            for n, us, c in top[:20]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-lens", type=int, nargs="+", default=[128, 512, 1024, 2048])
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    engine = DecodeEngine(model)

    if args.compile:
        engine.compile_step(args.compile_mode)
        ids = torch.randint(4, model.config.vocab_size, (1, 64)).to(device)
        engine.prefill(ids)
        tok = torch.tensor([[100]], device=device)
        t0 = time.time()
        for i in range(6):
            t_step = time.time()
            logits = engine.step_logits(tok)
            tok = logits[0, -1:].argmax(-1, keepdim=True)
            torch.cuda.synchronize(device)
            el = time.time() - t_step
            if el > 1.0:
                logger.info("  compile step %d: %.1fs", i, el)
        logger.info("Compile warmup total %.1fs", time.time() - t0)

    results: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "compiled": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "decode_tokens": args.decode_tokens,
        "rows": [],
    }
    for ctx in args.context_lens:
        fwd = bench_forward(engine, ctx, args.decode_tokens, device)
        samp = bench_with_sampler(engine, ctx, args.decode_tokens, device)
        results["rows"].append({
            "ctx": ctx,
            "forward_ms_per_tok": round(fwd, 3),
            "forward_tok_s": round(1000 / fwd, 1),
            "sampler_ms_per_tok": round(samp, 3),
            "sampler_tok_s": round(1000 / samp, 1),
        })
        logger.info(
            "ctx=%4d  forward: %6.2f ms/tok (%6.1f tok/s)   +sampler: %6.2f ms/tok (%6.1f tok/s)",
            ctx, fwd, 1000 / fwd, samp, 1000 / samp,
        )

    if args.profile:
        results["profile_window"] = profile_window(engine, 512, 8, device)
        pw = results["profile_window"]
        logger.info("profile @512: kernel %.2f ms/step, %d launches/step",
                    pw["kernel_ms_per_step"], pw["kernel_launches_per_step"])

    results["peak_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
    logger.info("Peak memory: %.2f GB", results["peak_memory_gb"])

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        logger.info("Written: %s", out)


if __name__ == "__main__":
    main()
