#!/usr/bin/env python3
"""SDPA backend matrix: isolate cuDNN plan-selection overhead.

Tests decode with different SDP backends to determine whether the 300ms/tok
tail is cuDNN-specific. Also tests shape-cache priming and randomized order.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/benchmark/sdpa_backend_matrix.py \
        --checkpoint /path/to/checkpoint.pt.zst
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
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

PROMPT_LENGTHS = [16, 128, 512, 1024, 2048, 3072]
DECODE_LENGTHS = [64, 256]


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
    prompt_len: int,
    n_decode: int,
    backend_ctx=None,
    n_warmup: int = 2,
) -> dict:
    device = input_ids.device
    seq = input_ids[:, :prompt_len]

    def run_prefill_decode():
        out = model(seq, use_cache=True)
        torch.cuda.synchronize(device)
        past_kv = out["past_kv"]
        tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

        decode_times = []
        for _ in range(n_decode):
            t = time.perf_counter()
            out = model(tok, use_cache=True, past_kv=past_kv)
            torch.cuda.synchronize(device)
            decode_times.append((time.perf_counter() - t) * 1000)
            past_kv = out["past_kv"]
            tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
        return decode_times

    if backend_ctx is not None:
        with backend_ctx:
            # Warmup
            for _ in range(n_warmup):
                model(seq, use_cache=True)
                torch.cuda.synchronize(device)
            # Timed
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(seq, use_cache=True)
            torch.cuda.synchronize(device)
            prefill_ms = (time.perf_counter() - t0) * 1000

            decode_times = run_prefill_decode()
    else:
        for _ in range(n_warmup):
            model(seq, use_cache=True)
            torch.cuda.synchronize(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model(seq, use_cache=True)
        torch.cuda.synchronize(device)
        prefill_ms = (time.perf_counter() - t0) * 1000

        decode_times = run_prefill_decode()

    first_10 = sum(decode_times[:10]) / min(10, len(decode_times))
    last_10 = sum(decode_times[-10:]) / min(10, len(decode_times)) if len(decode_times) >= 10 else first_10
    avg = sum(decode_times) / len(decode_times)

    return {
        "prompt_len": prompt_len,
        "n_decode": n_decode,
        "prefill_ms": round(prefill_ms, 1),
        "decode_avg_ms": round(avg, 2),
        "first10_ms": round(first_10, 2),
        "last10_ms": round(last_10, 2),
        "tok_s": round(n_decode / (sum(decode_times) / 1000), 1),
        "slowdown": round(last_10 / first_10, 2) if first_10 > 0 else 0,
    }


def print_header():
    print(f"{'Prompt':>8} {'Decode':>8} {'Prefill':>10} {'Avg/tok':>10} {'First10':>10} {'Last10':>10} {'Slowdown':>10} {'Tok/s':>8}")
    print("-" * 80)


def print_row(r: dict):
    print(
        f"{r['prompt_len']:>8d} {r['n_decode']:>8d} "
        f"{r['prefill_ms']:>8.1f}ms {r['decode_avg_ms']:>8.2f}ms "
        f"{r['first10_ms']:>8.2f}ms {r['last10_ms']:>8.2f}ms "
        f"{r['slowdown']:>9.2f}x {r['tok_s']:>7.1f}"
    )


@torch.inference_mode()
def warm_all_shapes(model, input_ids, max_total=3328, device=None):
    """Prime cuDNN plan cache by running decode across all KV lengths."""
    print("Priming shape cache (full range)...")
    seq = input_ids[:, :16]
    out = model(seq, use_cache=True)
    torch.cuda.synchronize(device)
    past_kv = out["past_kv"]
    tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
    for i in range(min(max_total, 3500)):
        out = model(tok, use_cache=True, past_kv=past_kv)
        torch.cuda.synchronize(device)
        past_kv = out["past_kv"]
        tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
    print(f"  Primed {min(max_total, 3500)} decode shapes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    seed = "The history of science is filled with remarkable discoveries. " * 200
    ids = tokenizer.encode(seed, add_special_tokens=False)[:4000]
    input_ids = torch.tensor([ids], device=device)
    print(f"Max tokens: {input_ids.shape[1]}\n")

    all_results: dict[str, list[dict]] = {}

    # ── Test 1: Backend matrix ──────────────────────────────────────────────

    backends = {
        "default": None,
        "no_cudnn": sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]),
        "math_only": sdpa_kernel([SDPBackend.MATH]),
        "flash_only": sdpa_kernel([SDPBackend.FLASH_ATTENTION]),
        "efficient_only": sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]),
    }

    test_cases = [(128, 64), (512, 64), (128, 256)]

    for backend_name, backend_ctx in backends.items():
        print(f"\n{'=' * 80}")
        print(f"BACKEND: {backend_name}")
        print("=" * 80)
        print_header()

        results = []
        for prompt_len, decode_len in test_cases:
            try:
                r = bench_decode(model, input_ids, prompt_len, decode_len, backend_ctx)
                r["backend"] = backend_name
                results.append(r)
                print_row(r)
            except Exception as e:
                print(f"{'FAILED':>8} {prompt_len}+{decode_len}: {e}")
                results.append({
                    "prompt_len": prompt_len, "n_decode": decode_len,
                    "backend": backend_name, "error": str(e),
                })

        all_results[backend_name] = results

    # ── Test 2: Shape-cache priming ─────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("SHAPE-CACHE PRIMING TEST")
    print("=" * 80)
    print("Running full decode warmup to prime cuDNN shape cache...")

    warm_all_shapes(model, input_ids, max_total=3328, device=device)

    print("\nAfter priming:")
    print_header()

    primed_results = []
    for prompt_len, decode_len in [(128, 256), (512, 64), (16, 64), (1024, 64)]:
        r = bench_decode(model, input_ids, prompt_len, decode_len)
        r["primed"] = True
        primed_results.append(r)
        print_row(r)

    all_results["primed_default"] = primed_results

    # ── Test 3: Randomized order ────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("RANDOMIZED ORDER (after fresh model reload)")
    print("=" * 80)

    # Reload model to clear any cached state
    model2 = load_model(args.checkpoint, device)
    random_cases = [(16, 64), (128, 64), (512, 64), (1024, 64), (2048, 64), (3072, 64)]
    random.shuffle(random_cases)
    print(f"Order: {random_cases}")
    print_header()

    random_results = []
    for prompt_len, decode_len in random_cases:
        r = bench_decode(model2, input_ids, prompt_len, decode_len)
        r["order"] = random_cases.index((prompt_len, decode_len))
        random_results.append(r)
        print_row(r)

    all_results["randomized"] = random_results
    del model2

    # ── Summary ─────────────────────────────────────────────────────────────

    print(f"\n{'=' * 80}")
    print("FULL RESULTS JSON")
    print("=" * 80)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
