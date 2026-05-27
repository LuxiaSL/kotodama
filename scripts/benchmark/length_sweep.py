#!/usr/bin/env python3
"""Sweep prompt/decode lengths at best threading + compile settings.

Tests KV cache scaling by measuring decode latency as context grows.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/benchmark/length_sweep.py \
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
def bench_at_length(
    model: LuxiaBaseModel,
    input_ids: torch.Tensor,
    prompt_len: int,
    n_decode: int,
    compiled_model: torch.nn.Module | None = None,
    n_warmup: int = 2,
) -> dict:
    device = input_ids.device
    seq = input_ids[:, :prompt_len]
    m = compiled_model if compiled_model is not None else model

    # Warmup
    for _ in range(n_warmup):
        _ = m(seq, use_cache=True)
        torch.cuda.synchronize(device)

    # Prefill
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = m(seq, use_cache=True)
    torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - t0) * 1000

    past_kv = out["past_kv"]
    tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

    # Measure per-token decode latency at different points
    decode_times: list[float] = []
    torch.cuda.synchronize(device)
    for i in range(n_decode):
        t_step = time.perf_counter()
        out = m(tok, use_cache=True, past_kv=past_kv)
        torch.cuda.synchronize(device)
        decode_times.append((time.perf_counter() - t_step) * 1000)
        past_kv = out["past_kv"]
        tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

    first_10 = sum(decode_times[:10]) / min(10, len(decode_times))
    last_10 = sum(decode_times[-10:]) / min(10, len(decode_times)) if len(decode_times) >= 10 else first_10
    avg = sum(decode_times) / len(decode_times)
    total_decode_ms = sum(decode_times)

    return {
        "prompt_len": prompt_len,
        "n_decode": n_decode,
        "compiled": compiled_model is not None,
        "prefill_ms": round(prefill_ms, 1),
        "decode_avg_ms": round(avg, 2),
        "decode_first10_ms": round(first_10, 2),
        "decode_last10_ms": round(last_10, 2),
        "decode_total_ms": round(total_decode_ms, 1),
        "tok_s": round(n_decode / (total_decode_ms / 1000), 1),
        "slowdown_ratio": round(last_10 / first_10, 2) if first_10 > 0 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    # Build a 4000-token input for slicing
    seed = "The history of science is filled with remarkable discoveries. " * 200
    ids = tokenizer.encode(seed, add_special_tokens=False)[:4000]
    input_ids = torch.tensor([ids], device=device)
    print(f"Max input length: {input_ids.shape[1]} tokens, threads={args.threads}\n")

    prompt_lengths = [16, 128, 512, 1024, 2048, 3072]
    decode_lengths = [64, 256]

    # No compile
    print("=" * 80)
    print("WITHOUT torch.compile")
    print("=" * 80)
    print(f"{'Prompt':>8} {'Decode':>8} {'Prefill':>10} {'Avg/tok':>10} {'First10':>10} {'Last10':>10} {'Slowdown':>10} {'Tok/s':>8}")
    print("-" * 80)

    no_compile_results = []
    for plen in prompt_lengths:
        for dlen in decode_lengths:
            if plen + dlen > 4095:
                continue
            r = bench_at_length(model, input_ids, plen, dlen)
            no_compile_results.append(r)
            print(f"{plen:>8d} {dlen:>8d} {r['prefill_ms']:>8.1f}ms {r['decode_avg_ms']:>8.2f}ms {r['decode_first10_ms']:>8.2f}ms {r['decode_last10_ms']:>8.2f}ms {r['slowdown_ratio']:>9.2f}x {r['tok_s']:>7.1f}")

    if args.skip_compile:
        all_results = {"no_compile": no_compile_results}
        print(f"\n{'=' * 80}")
        print(json.dumps(all_results, indent=2))
        return

    # Compile
    print(f"\n{'=' * 80}")
    print("WITH torch.compile(dynamic=True)")
    print("=" * 80)

    compiled_model = torch.compile(model, dynamic=True)

    # Warmup compile at a few shapes
    print("Warming up compiled model...")
    for plen in [16, 128, 1024]:
        seq = input_ids[:, :plen]
        out = compiled_model(seq, use_cache=True)
        torch.cuda.synchronize(device)
        tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
        for _ in range(3):
            out2 = compiled_model(tok, use_cache=True, past_kv=out["past_kv"])
            torch.cuda.synchronize(device)
            out = out2
            tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
    print("Compile warmup done.\n")

    print(f"{'Prompt':>8} {'Decode':>8} {'Prefill':>10} {'Avg/tok':>10} {'First10':>10} {'Last10':>10} {'Slowdown':>10} {'Tok/s':>8}")
    print("-" * 80)

    compile_results = []
    for plen in prompt_lengths:
        for dlen in decode_lengths:
            if plen + dlen > 4095:
                continue
            r = bench_at_length(model, input_ids, plen, dlen, compiled_model=compiled_model)
            compile_results.append(r)
            print(f"{plen:>8d} {dlen:>8d} {r['prefill_ms']:>8.1f}ms {r['decode_avg_ms']:>8.2f}ms {r['decode_first10_ms']:>8.2f}ms {r['decode_last10_ms']:>8.2f}ms {r['slowdown_ratio']:>9.2f}x {r['tok_s']:>7.1f}")

    # Dump all
    all_results = {"no_compile": no_compile_results, "compile": compile_results}
    print(f"\n{'=' * 80}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
