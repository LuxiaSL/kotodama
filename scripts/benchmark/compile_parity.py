#!/usr/bin/env python3
"""Compare eager vs compiled decode logits for Gate 1 parity.

Verifies that torch.compile(dynamic=True) produces identical logits
to eager mode during cached autoregressive decode.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/benchmark/compile_parity.py \
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
import torch.nn.functional as F

torch.set_num_threads(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

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
def compare_logits(
    model: LuxiaBaseModel,
    compiled_model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_len: int,
    n_decode: int,
    device: torch.device,
) -> dict:
    seq = input_ids[:, :prompt_len]

    # Eager prefill
    eager_out = model(seq, use_cache=True)
    eager_kv = eager_out["past_kv"]
    eager_logits_prefill = eager_out["logits"][0, -1]
    tok = eager_logits_prefill.argmax().unsqueeze(0).unsqueeze(0)

    # Compiled prefill (same model, just to get matching KV cache)
    # Actually, we want to use eager prefill + compiled decode,
    # matching the serve.py strategy
    compiled_kv = eager_kv  # same KV cache from eager prefill

    results = []
    for step in range(n_decode):
        # Eager decode
        eager_out = model(tok, use_cache=True, past_kv=eager_kv)
        eager_logits = eager_out["logits"][0, -1].float()
        eager_kv = eager_out["past_kv"]

        # Compiled decode (from same KV cache state)
        compiled_out = compiled_model(tok, use_cache=True, past_kv=compiled_kv)
        compiled_logits = compiled_out["logits"][0, -1].float()
        compiled_kv = compiled_out["past_kv"]

        # Compare
        abs_diff = (eager_logits - compiled_logits).abs()
        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()

        eager_probs = F.softmax(eager_logits, dim=-1)
        compiled_probs = F.softmax(compiled_logits, dim=-1)
        kl = (eager_probs * (eager_probs / compiled_probs).log()).sum().item()

        top1_match = eager_logits.argmax() == compiled_logits.argmax()

        eager_top10 = set(eager_logits.topk(10).indices.tolist())
        compiled_top10 = set(compiled_logits.topk(10).indices.tolist())
        top10_overlap = len(eager_top10 & compiled_top10) / 10

        results.append({
            "step": step,
            "pos": prompt_len + step,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "kl_div": kl,
            "top1_match": bool(top1_match),
            "top10_overlap": top10_overlap,
        })

        # Advance with the same token (eager's choice) for both
        tok = eager_logits.argmax().unsqueeze(0).unsqueeze(0)

    return {
        "prompt_len": prompt_len,
        "n_decode": n_decode,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = load_model(args.checkpoint, device)

    print("Creating compiled model...")
    compiled_model = torch.compile(model, dynamic=True)

    # Warmup compiled decode (triggers compilation)
    print("Warming up compile...")
    dummy = torch.zeros(1, 8, dtype=torch.long, device=device)
    out = model(dummy, use_cache=True)
    kv = out["past_kv"]
    tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
    for _ in range(4):
        out = compiled_model(tok, use_cache=True, past_kv=kv)
        torch.cuda.synchronize(device)
        kv = out["past_kv"]
        tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
    print("Compile warmed up\n")

    # Build input
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    seed = "The history of science is filled with remarkable discoveries. " * 200
    ids = tokenizer.encode(seed, add_special_tokens=False)[:4000]
    input_ids = torch.tensor([ids], device=device)

    # Test across prompt lengths and decode lengths
    test_cases = [
        (16, 64),
        (128, 64),
        (512, 32),
        (1024, 32),
        (2048, 16),
        (3072, 16),
    ]

    all_results = []
    all_pass = True

    print(f"{'Prompt':>8} {'Decode':>8} {'MaxAbs':>12} {'MeanAbs':>12} {'MaxKL':>12} {'Top1':>8} {'Top10':>8}")
    print("-" * 80)

    for plen, dlen in test_cases:
        result = compare_logits(model, compiled_model, input_ids, plen, dlen, device)
        all_results.append(result)

        max_abs = max(r["max_abs_diff"] for r in result["results"])
        mean_abs = max(r["mean_abs_diff"] for r in result["results"])
        max_kl = max(r["kl_div"] for r in result["results"])
        top1_all = all(r["top1_match"] for r in result["results"])
        min_top10 = min(r["top10_overlap"] for r in result["results"])

        # torch.compile reorders ops for performance, shifting bf16 rounding.
        # KL up to ~5e-3 is expected; top-1 match is the authoritative signal.
        status = "PASS" if top1_all and max_kl < 5e-3 else "FAIL"
        if status == "FAIL":
            all_pass = False

        top1_str = "ALL" if top1_all else "MISS"
        print(f"{plen:>8d} {dlen:>8d} {max_abs:>12.6f} {mean_abs:>12.8f} {max_kl:>12.2e} "
              f"{top1_str:>8s} {min_top10:>7.0%}  {status}")

    print(f"\n{'=' * 80}")
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'=' * 80}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path) + ".json", "w") as f:
            json.dump({"pass": all_pass, "results": all_results}, f, indent=2)
        print(f"Results saved to {output_path}.json")


if __name__ == "__main__":
    main()
