#!/usr/bin/env python3
"""Sweep OMP/torch thread counts to find optimal CPU threading for decode.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/benchmark/thread_sweep.py \
        --checkpoint /path/to/checkpoint.pt.zst
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
from transformers import AutoTokenizer

torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
DDV1_BOUNDARIES = [0, 3, 7, 12, 21, 25]

PROXY_CONFIG = dict(
    hidden_size=512, num_layers=28, num_attention_heads=4, num_kv_heads=2,
    head_dim=128, intermediate_size=1408, vocab_size=49152,
    max_position_embeddings=4096, rope_theta=500000.0, norm_eps=1e-5,
    qk_norm=True, tie_word_embeddings=True, z_loss_weight=0.0,
    use_liger=False, attn_impl="sdpa", attn_res=True,
    attn_res_boundaries=DDV1_BOUNDARIES,
)

THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]


def load_model(checkpoint_path: str, device: torch.device) -> LuxiaBaseModel:
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.suffix == ".zst":
        import zstandard as zstd
        print(f"Decompressing {ckpt_path}...")
        dctx = zstd.ZstdDecompressor()
        with open(ckpt_path, "rb") as f:
            ckpt = torch.load(io.BytesIO(dctx.decompress(f.read())), map_location="cpu", weights_only=False)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = LuxiaModelConfig(**PROXY_CONFIG)
    model = LuxiaBaseModel(config)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    print(f"Loaded step {ckpt.get('step', '?')}")
    return model.to(device).eval().bfloat16()


@torch.inference_mode()
def bench_decode(
    model: LuxiaBaseModel,
    input_ids: torch.Tensor,
    num_threads: int,
    n_decode: int = 64,
    n_warmup: int = 3,
) -> dict:
    device = input_ids.device

    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)

    # Warmup
    for _ in range(n_warmup):
        _ = model(input_ids, use_cache=True)
        torch.cuda.synchronize(device)

    # Prefill
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = model(input_ids, use_cache=True)
    torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - t0) * 1000

    past_kv = out["past_kv"]
    tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

    # Decode
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_decode):
        out = model(tok, use_cache=True, past_kv=past_kv)
        torch.cuda.synchronize(device)
        past_kv = out["past_kv"]
        tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
    decode_total = (time.perf_counter() - t0) * 1000

    per_token = decode_total / n_decode
    tok_s = n_decode / (decode_total / 1000)

    return {
        "threads": num_threads,
        "prefill_ms": round(prefill_ms, 1),
        "decode_per_tok_ms": round(per_token, 2),
        "tok_s": round(tok_s, 1),
    }


@torch.inference_mode()
def bench_compiled(
    model: LuxiaBaseModel,
    input_ids: torch.Tensor,
    num_threads: int,
    n_decode: int = 64,
    n_warmup: int = 5,
) -> dict:
    """Benchmark with torch.compile (separate because compile is destructive)."""
    device = input_ids.device

    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)

    compiled = torch.compile(model, dynamic=True)

    # Extra warmup for compile (first calls trigger tracing)
    print(f"  Compiling (warmup with {n_warmup} passes)...")
    for i in range(n_warmup):
        out = compiled(input_ids, use_cache=True)
        torch.cuda.synchronize(device)
        if i == 0:
            # Also warm the decode path
            past_kv = out["past_kv"]
            tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
            for _ in range(3):
                out2 = compiled(tok, use_cache=True, past_kv=past_kv)
                torch.cuda.synchronize(device)
                past_kv = out2["past_kv"]
                tok = out2["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

    # Prefill
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = compiled(input_ids, use_cache=True)
    torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - t0) * 1000

    past_kv = out["past_kv"]
    tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

    # Decode
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_decode):
        out = compiled(tok, use_cache=True, past_kv=past_kv)
        torch.cuda.synchronize(device)
        past_kv = out["past_kv"]
        tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
    decode_total = (time.perf_counter() - t0) * 1000

    per_token = decode_total / n_decode
    tok_s = n_decode / (decode_total / 1000)

    return {
        "threads": num_threads,
        "compiled": True,
        "prefill_ms": round(prefill_ms, 1),
        "decode_per_tok_ms": round(per_token, 2),
        "tok_s": round(tok_s, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    ids = tokenizer.encode("The history of science " * 50, add_special_tokens=False)[:args.prompt_tokens]
    input_ids = torch.tensor([ids], device=device)
    print(f"Input: {input_ids.shape[1]} tokens, decode: {args.decode_tokens} tokens\n")

    # Thread sweep (no compile)
    print("=" * 70)
    print("THREAD SWEEP (no compile)")
    print("=" * 70)
    print(f"{'Threads':>8}  {'Prefill':>10}  {'Decode/tok':>12}  {'Tok/s':>8}")
    print("-" * 42)

    thread_results = []
    for threads in THREAD_COUNTS:
        r = bench_decode(model, input_ids, threads, n_decode=args.decode_tokens)
        thread_results.append(r)
        print(f"{threads:>8d}  {r['prefill_ms']:>8.1f}ms  {r['decode_per_tok_ms']:>10.2f}ms  {r['tok_s']:>7.1f}")

    # Find best thread count
    best = min(thread_results, key=lambda x: x["decode_per_tok_ms"])
    print(f"\nBest: {best['threads']} threads -> {best['decode_per_tok_ms']}ms/tok ({best['tok_s']} tok/s)")

    if not args.skip_compile:
        # Compile test at best thread count (and at 1 thread for comparison)
        print(f"\n{'=' * 70}")
        print("COMPILE TEST")
        print("=" * 70)

        compile_thread_counts = sorted(set([1, best["threads"]]))
        compile_results = []
        for threads in compile_thread_counts:
            print(f"\nthreads={threads}, compiled=True:")
            try:
                r = bench_compiled(model, input_ids, threads, n_decode=args.decode_tokens)
                compile_results.append(r)
                print(f"  prefill={r['prefill_ms']:.1f}ms  decode={r['decode_per_tok_ms']:.2f}ms/tok  {r['tok_s']:.1f} tok/s")
            except Exception as e:
                print(f"  FAILED: {e}")
                compile_results.append({"threads": threads, "compiled": True, "error": str(e)})

        all_results = thread_results + compile_results
    else:
        all_results = thread_results

    # Summary
    print(f"\n{'=' * 70}")
    print("FULL SUMMARY")
    print("=" * 70)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
