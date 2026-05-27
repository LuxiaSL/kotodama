#!/usr/bin/env python3
"""Profile single-token decode with torch.profiler.

Captures kernel-level timing breakdown for prefill and decode steps,
showing exactly where time is spent (KV concat, AttnRes routing,
attention, MLP, etc.)

Usage:
    python scripts/benchmark/profile_decode.py \
        --checkpoint checkpoints/fullcorpus-ddv1/step_00081252.pt.zst \
        --device cuda:7

    # Custom prompt/gen length
    python scripts/benchmark/profile_decode.py \
        --checkpoint checkpoints/fullcorpus-ddv1/step_00081252.pt.zst \
        --prompt-tokens 128 --decode-tokens 64

    # Export Chrome trace (view at chrome://tracing)
    python scripts/benchmark/profile_decode.py \
        --checkpoint checkpoints/fullcorpus-ddv1/step_00081252.pt.zst \
        --trace-dir scripts/benchmark/traces/
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule
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


def load_checkpoint(checkpoint_path: str, device: torch.device) -> LuxiaBaseModel:
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
    logger.info("Loaded checkpoint step %s", ckpt.get("step", "?"))
    return model.to(device).eval().bfloat16()


@torch.inference_mode()
def profile_generation(
    model: LuxiaBaseModel,
    input_ids: torch.Tensor,
    num_decode_tokens: int,
    device: torch.device,
    trace_dir: Path | None = None,
) -> None:
    """Profile prefill + decode with torch.profiler."""

    # Warmup (outside profiler)
    logger.info("Warming up (3 forward passes)...")
    for _ in range(3):
        model(input_ids, use_cache=True)
    torch.cuda.synchronize(device)

    logger.info("Profiling: %d prompt tokens, %d decode tokens", input_ids.shape[1], num_decode_tokens)

    trace_handler = None
    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_handler = torch.profiler.tensorboard_trace_handler(str(trace_dir))

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
        on_trace_ready=trace_handler,
    ) as prof:
        # Prefill
        with record_function("PREFILL"):
            output = model(input_ids, use_cache=True)
            torch.cuda.synchronize(device)

        past_kv = output["past_kv"]
        next_token = output["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

        # Decode loop
        for step in range(num_decode_tokens):
            with record_function(f"DECODE_STEP_{step}"):
                output = model(next_token, use_cache=True, past_kv=past_kv)
                torch.cuda.synchronize(device)

            past_kv = output["past_kv"]
            next_token = output["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

    # Print summary tables
    print("\n" + "=" * 80)
    print("KERNEL TIME BREAKDOWN (sorted by CUDA time)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\n" + "=" * 80)
    print("KERNEL TIME BREAKDOWN (sorted by CPU time)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    print("\n" + "=" * 80)
    print("GROUPED BY INPUT SHAPE (top CUDA consumers)")
    print("=" * 80)
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))

    # Custom summary: prefill vs decode
    print("\n" + "=" * 80)
    print("PHASE SUMMARY")
    print("=" * 80)
    events = prof.key_averages()
    prefill_events = [e for e in prof.events() if "PREFILL" in str(getattr(e, 'name', ''))]
    decode_events = [e for e in prof.events() if "DECODE_STEP" in str(getattr(e, 'name', ''))]

    for event in events:
        name = event.key
        if name in ("PREFILL",) or name.startswith("DECODE_STEP"):
            cuda_us = event.cuda_time_total
            cpu_us = event.cpu_time_total
            count = event.count
            print(f"  {name}: CUDA={cuda_us/1000:.1f}ms  CPU={cpu_us/1000:.1f}ms  count={count}")

    if trace_dir:
        # Also export Chrome trace
        chrome_path = trace_dir / "chrome_trace.json"
        prof.export_chrome_trace(str(chrome_path))
        logger.info("Chrome trace saved to %s (open at chrome://tracing)", chrome_path)
        logger.info("TensorBoard traces saved to %s", trace_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile decode with torch.profiler")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--trace-dir", type=Path, default=None,
                        help="Directory for Chrome/TensorBoard traces")
    args = parser.parse_args()

    device = torch.device(args.device)

    logger.info("Loading model...")
    model = load_checkpoint(args.checkpoint, device)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Build prompt at target length
    seed = "The history of science is filled with remarkable discoveries. " * 50
    ids = tokenizer.encode(seed, add_special_tokens=False)[:args.prompt_tokens]
    input_ids = torch.tensor([ids], device=device)
    logger.info("Input: %d tokens", input_ids.shape[1])

    profile_generation(model, input_ids, args.decode_tokens, device, args.trace_dir)


if __name__ == "__main__":
    main()
