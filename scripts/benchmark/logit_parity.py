#!/usr/bin/env python3
"""Gate 1: Full-forward vs cached-forward logit parity test.

For each prompt, compares:
  1. Full forward over prompt[:t] (ground truth)
  2. Incremental cached forward up to position t

Tracks max/mean absolute logit diff, KL divergence, top-1/10/50 overlap.
Tests at boundary lengths that cover AttnRes boundaries and power-of-2 edges.

Usage:
    python scripts/benchmark/logit_parity.py \
        --checkpoint checkpoints/fullcorpus-ddv1/step_00081252.pt.zst \
        --device cuda:7

    # Quick mode (fewer positions)
    python scripts/benchmark/logit_parity.py \
        --checkpoint checkpoints/fullcorpus-ddv1/step_00081252.pt.zst \
        --quick
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
DDV1_BOUNDARIES = [0, 3, 7, 12, 21, 25]

PROXY_CONFIG = dict(
    hidden_size=512,
    num_layers=28,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=128,
    intermediate_size=1408,
    vocab_size=49152,
    max_position_embeddings=4096,
    rope_theta=500000.0,
    norm_eps=1e-5,
    qk_norm=True,
    tie_word_embeddings=True,
    z_loss_weight=0.0,
    use_liger=False,
    attn_impl="sdpa",
    attn_res=True,
    attn_res_boundaries=DDV1_BOUNDARIES,
)

# Positions to test: AttnRes boundaries, powers of 2, near-boundaries
FULL_TEST_POSITIONS = [
    1, 2, 3, 7, 12, 21, 25, 31, 32,
    127, 128, 255, 256, 511, 512,
    1023, 1024, 2047, 2048, 3071, 3072, 4095,
]

QUICK_TEST_POSITIONS = [1, 3, 7, 12, 25, 32, 128, 512, 1024, 2048]


def load_checkpoint(checkpoint_path: str, device: torch.device) -> LuxiaBaseModel:
    """Load model from checkpoint, handling zstd compression."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if ckpt_path.suffix == ".zst":
        import zstandard as zstd
        logger.info("Decompressing zstd checkpoint: %s", ckpt_path)
        dctx = zstd.ZstdDecompressor()
        with open(ckpt_path, "rb") as f_in:
            decompressed = dctx.decompress(f_in.read())
        ckpt = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
        del decompressed
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    config = LuxiaModelConfig(**PROXY_CONFIG)
    model = LuxiaBaseModel(config)

    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)
    logger.info("Loaded checkpoint step %s (%s tokens)", ckpt.get("step", "?"), ckpt.get("tokens_consumed", "?"))

    model = model.to(device).eval().bfloat16()
    return model


@dataclass
class ParityResult:
    position: int
    max_abs_diff: float
    mean_abs_diff: float
    kl_divergence: float
    top1_match: bool
    top10_overlap: int
    top50_overlap: int
    full_top1_token: int
    cached_top1_token: int


@torch.inference_mode()
def compare_at_position(
    model: LuxiaBaseModel,
    input_ids: torch.Tensor,
    position: int,
    device: torch.device,
) -> ParityResult:
    """Compare full-forward vs cached-forward logits at a given position.

    Full forward: feed prompt[:position+1], take logits at position.
    Cached forward: feed tokens one-at-a-time with KV cache up to position.
    """
    seq = input_ids[:, :position + 1]

    # Full forward (ground truth)
    full_out = model(seq)
    full_logits = full_out["logits"][0, -1].float()

    # Cached forward: prefill with seq[:position], then decode last token
    if position == 0:
        # Single token: cached path is identical to full forward
        cached_out = model(seq, use_cache=True)
        cached_logits = cached_out["logits"][0, -1].float()
    else:
        prefill_out = model(seq[:, :-1], use_cache=True)
        past_kv = prefill_out["past_kv"]
        last_token = seq[:, -1:]
        decode_out = model(last_token, use_cache=True, past_kv=past_kv)
        cached_logits = decode_out["logits"][0, -1].float()

    # Metrics
    abs_diff = (full_logits - cached_logits).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    full_probs = F.softmax(full_logits, dim=-1)
    cached_probs = F.softmax(cached_logits, dim=-1)
    kl = F.kl_div(cached_probs.log(), full_probs, reduction="sum").item()

    full_top1 = full_logits.argmax().item()
    cached_top1 = cached_logits.argmax().item()

    full_top10 = set(full_logits.topk(10).indices.tolist())
    cached_top10 = set(cached_logits.topk(10).indices.tolist())

    full_top50 = set(full_logits.topk(50).indices.tolist())
    cached_top50 = set(cached_logits.topk(50).indices.tolist())

    return ParityResult(
        position=position,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        kl_divergence=kl,
        top1_match=(full_top1 == cached_top1),
        top10_overlap=len(full_top10 & cached_top10),
        top50_overlap=len(full_top50 & cached_top50),
        full_top1_token=full_top1,
        cached_top1_token=cached_top1,
    )


@torch.inference_mode()
def run_extended_decode_test(
    model: LuxiaBaseModel,
    input_ids: torch.Tensor,
    num_decode_steps: int,
    device: torch.device,
) -> list[ParityResult]:
    """Test parity across multiple decode steps after prefill.

    Prefills the full prompt, then decodes greedily for num_decode_steps,
    comparing each step's logits against a full-forward recomputation.
    """
    prompt_len = input_ids.shape[1]
    results: list[ParityResult] = []

    # Prefill
    prefill_out = model(input_ids, use_cache=True)
    past_kv = prefill_out["past_kv"]
    next_token_id = prefill_out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
    generated_ids = [next_token_id.item()]

    for step in range(num_decode_steps):
        decode_out = model(next_token_id, use_cache=True, past_kv=past_kv)
        cached_logits = decode_out["logits"][0, -1].float()
        past_kv = decode_out["past_kv"]

        # Full forward for ground truth
        full_seq = torch.cat([
            input_ids,
            torch.tensor([generated_ids], device=device),
        ], dim=1)
        full_out = model(full_seq)
        full_logits = full_out["logits"][0, -1].float()

        abs_diff = (full_logits - cached_logits).abs()
        full_top1 = full_logits.argmax().item()
        cached_top1 = cached_logits.argmax().item()
        full_top10 = set(full_logits.topk(10).indices.tolist())
        cached_top10 = set(cached_logits.topk(10).indices.tolist())
        full_top50 = set(full_logits.topk(50).indices.tolist())
        cached_top50 = set(cached_logits.topk(50).indices.tolist())

        full_probs = F.softmax(full_logits, dim=-1)
        cached_probs = F.softmax(cached_logits, dim=-1)
        kl = F.kl_div(cached_probs.log(), full_probs, reduction="sum").item()

        results.append(ParityResult(
            position=prompt_len + step,
            max_abs_diff=abs_diff.max().item(),
            mean_abs_diff=abs_diff.mean().item(),
            kl_divergence=kl,
            top1_match=(full_top1 == cached_top1),
            top10_overlap=len(full_top10 & cached_top10),
            top50_overlap=len(full_top50 & cached_top50),
            full_top1_token=full_top1,
            cached_top1_token=cached_top1,
        ))

        next_token_id = torch.tensor([[full_top1]], device=device)
        generated_ids.append(full_top1)

    return results


def format_results(results: list[ParityResult], header: str) -> str:
    """Format parity results as a Markdown table."""
    lines = [
        header,
        "",
        "| Position | Max Abs Diff | Mean Abs Diff | KL Div | Top-1 Match | Top-10 | Top-50 |",
        "|---:|---:|---:|---:|:---|---:|---:|",
    ]

    all_match = True
    for r in results:
        match_str = "yes" if r.top1_match else f"**NO** (full={r.full_top1_token}, cached={r.cached_top1_token})"
        if not r.top1_match:
            all_match = False
        lines.append(
            f"| {r.position} | {r.max_abs_diff:.2e} | {r.mean_abs_diff:.2e} | "
            f"{r.kl_divergence:.2e} | {match_str} | {r.top10_overlap}/10 | {r.top50_overlap}/50 |"
        )

    lines.append("")
    if all_match:
        lines.append("All top-1 tokens match.")
    else:
        mismatches = sum(1 for r in results if not r.top1_match)
        lines.append(f"**WARNING: {mismatches}/{len(results)} positions have top-1 mismatch.**")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 1: Full-forward vs cached-forward logit parity")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--quick", action="store_true", help="Test fewer positions")
    parser.add_argument("--decode-steps", type=int, default=32,
                        help="Number of decode steps for extended decode test")
    parser.add_argument("--output", "-o", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    test_positions = QUICK_TEST_POSITIONS if args.quick else FULL_TEST_POSITIONS

    logger.info("Loading model...")
    model = load_checkpoint(args.checkpoint, device)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Build a long prompt for testing (need at least max(test_positions)+1 tokens)
    max_pos = max(test_positions)
    seed_text = (
        "The history of science is filled with remarkable discoveries that changed our "
        "understanding of the world. From the ancient Greeks who first proposed atomic "
        "theory to modern physicists exploring quantum mechanics, each generation has "
        "built upon the knowledge of those who came before. "
    ) * 100  # repeat to ensure enough tokens
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if len(seed_ids) < max_pos + 1:
        logger.error("Seed text too short: %d tokens, need %d", len(seed_ids), max_pos + 1)
        raise SystemExit(1)

    input_ids = torch.tensor([seed_ids[:max_pos + 1]], device=device)
    logger.info("Test sequence: %d tokens", input_ids.shape[1])

    # Filter positions that fit within our sequence
    valid_positions = [p for p in test_positions if p <= max_pos]

    # Run parity tests at each position
    all_results: list[ParityResult] = []
    logger.info("Testing %d positions...", len(valid_positions))
    t0 = time.perf_counter()

    for pos in valid_positions:
        result = compare_at_position(model, input_ids, pos, device)
        all_results.append(result)
        status = "PASS" if result.top1_match else "FAIL"
        logger.info(
            "  pos=%4d: %s  max_diff=%.2e  kl=%.2e  top10=%d/10",
            pos, status, result.max_abs_diff, result.kl_divergence, result.top10_overlap,
        )

    elapsed = time.perf_counter() - t0
    logger.info("Position tests done in %.1fs", elapsed)

    # Extended decode test: prefill short prompt, then decode and compare
    logger.info("Running extended decode test (%d steps)...", args.decode_steps)
    short_prompt = input_ids[:, :128]
    decode_results = run_extended_decode_test(model, short_prompt, args.decode_steps, device)
    for r in decode_results:
        status = "PASS" if r.top1_match else "FAIL"
        logger.info(
            "  decode pos=%4d: %s  max_diff=%.2e  kl=%.2e",
            r.position, status, r.max_abs_diff, r.kl_divergence,
        )

    # Format output
    sections = [
        format_results(all_results, "## Prefill Parity (full-forward vs cached-forward at various positions)"),
        "",
        format_results(decode_results, "## Extended Decode Parity (128-token prefill + greedy decode)"),
    ]
    summary = "\n".join(["# Gate 1: Logit Parity Report", ""] + sections)
    print("\n" + summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        json_path = args.output.with_suffix(".json")
        json_data = {
            "checkpoint": args.checkpoint,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prefill_parity": [
                {
                    "position": r.position,
                    "max_abs_diff": r.max_abs_diff,
                    "mean_abs_diff": r.mean_abs_diff,
                    "kl_divergence": r.kl_divergence,
                    "top1_match": r.top1_match,
                    "top10_overlap": r.top10_overlap,
                    "top50_overlap": r.top50_overlap,
                }
                for r in all_results
            ],
            "decode_parity": [
                {
                    "position": r.position,
                    "max_abs_diff": r.max_abs_diff,
                    "mean_abs_diff": r.mean_abs_diff,
                    "kl_divergence": r.kl_divergence,
                    "top1_match": r.top1_match,
                    "top10_overlap": r.top10_overlap,
                    "top50_overlap": r.top50_overlap,
                }
                for r in decode_results
            ],
        }
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        logger.info("Results saved to %s", json_path)

        md_path = args.output.with_suffix(".md")
        with open(md_path, "w") as f:
            f.write(summary)
        logger.info("Summary saved to %s", md_path)

    # Final verdict
    all_pass = all(r.top1_match for r in all_results + decode_results)
    if all_pass:
        logger.info("GATE 1 PASSED: All positions have top-1 match.")
    else:
        n_fail = sum(1 for r in all_results + decode_results if not r.top1_match)
        n_total = len(all_results) + len(decode_results)
        logger.error("GATE 1 FAILED: %d/%d positions have top-1 mismatch.", n_fail, n_total)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
