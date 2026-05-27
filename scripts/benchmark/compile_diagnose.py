#!/usr/bin/env python3
"""Diagnose torch.compile graph breaks in _forward_attn_res_cached.

Runs torch._dynamo.explain() on the cached forward path to identify
exactly which operations cause graph breaks and recompilation.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/benchmark/compile_diagnose.py \
        --checkpoint /path/to/checkpoint.pt.zst
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
import torch._dynamo

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
def diagnose_full_model(model, device):
    """Run explain() on the full model forward with use_cache=True."""
    print("\n" + "=" * 80)
    print("DIAGNOSIS: Full model forward (use_cache=True, prefill)")
    print("=" * 80)

    input_ids = torch.zeros(1, 32, dtype=torch.long, device=device)
    explanation = torch._dynamo.explain(model)(input_ids, use_cache=True)
    print(f"\nGraph breaks: {explanation.graph_count - 1}")
    print(f"Graph count: {explanation.graph_count}")
    if hasattr(explanation, 'break_reasons'):
        print(f"\nBreak reasons:")
        for i, reason in enumerate(explanation.break_reasons):
            print(f"  [{i}] {reason}")
    print(f"\nFull explanation:\n{explanation}")


@torch.inference_mode()
def diagnose_cached_forward(model, device):
    """Run explain() on _forward_attn_res_cached directly."""
    print("\n" + "=" * 80)
    print("DIAGNOSIS: _forward_attn_res_cached (decode step)")
    print("=" * 80)

    # First do a real prefill to get past_kv
    input_ids = torch.zeros(1, 32, dtype=torch.long, device=device)
    out = model(input_ids, use_cache=True)
    past_kv = out["past_kv"]

    # Now explain a single decode step
    embed_single = model.embed_tokens(torch.zeros(1, 1, dtype=torch.long, device=device))

    explanation = torch._dynamo.explain(model._forward_attn_res_cached)(
        embed_single, None, past_kv
    )
    print(f"\nGraph breaks: {explanation.graph_count - 1}")
    print(f"Graph count: {explanation.graph_count}")
    if hasattr(explanation, 'break_reasons'):
        print(f"\nBreak reasons:")
        for i, reason in enumerate(explanation.break_reasons):
            print(f"  [{i}] {reason}")
    print(f"\nFull explanation:\n{explanation}")


@torch.inference_mode()
def diagnose_single_layer(model, device):
    """Run explain() on a single TransformerBlock with use_cache=True."""
    print("\n" + "=" * 80)
    print("DIAGNOSIS: Single TransformerBlock (use_cache=True)")
    print("=" * 80)

    layer = model.layers[0]
    x = torch.randn(1, 1, 512, device=device, dtype=torch.bfloat16)
    rope_cos = model.rope_cos
    rope_sin = model.rope_sin

    # Without past_kv first
    explanation = torch._dynamo.explain(layer)(
        x, rope_cos, rope_sin, None, None, True
    )
    print(f"\nWithout past_kv:")
    print(f"  Graph breaks: {explanation.graph_count - 1}")
    print(f"  Graph count: {explanation.graph_count}")

    # With past_kv
    out, kv = layer(x, rope_cos, rope_sin, None, None, True)
    explanation2 = torch._dynamo.explain(layer)(
        x, rope_cos, rope_sin, None, kv, True
    )
    print(f"\nWith past_kv:")
    print(f"  Graph breaks: {explanation2.graph_count - 1}")
    print(f"  Graph count: {explanation2.graph_count}")
    if hasattr(explanation2, 'break_reasons') and explanation2.break_reasons:
        print(f"  Break reasons:")
        for i, reason in enumerate(explanation2.break_reasons):
            print(f"    [{i}] {reason}")


@torch.inference_mode()
def diagnose_route_static(model, device):
    """Run explain() on _route_static."""
    print("\n" + "=" * 80)
    print("DIAGNOSIS: _route_static")
    print("=" * 80)

    max_s = model._attn_res_max_sources
    buf = torch.randn(max_s, 1, 1, 512, device=device, dtype=torch.bfloat16)
    query = model.layers[0].attn_res_query
    norm = model.layers[0].attn_res_norm
    mask = model._attn_res_masks[0]

    explanation = torch._dynamo.explain(model._route_static)(
        buf, query, norm, mask, 1
    )
    print(f"\nGraph breaks: {explanation.graph_count - 1}")
    print(f"Graph count: {explanation.graph_count}")
    if hasattr(explanation, 'break_reasons') and explanation.break_reasons:
        print(f"  Break reasons:")
        for i, reason in enumerate(explanation.break_reasons):
            print(f"    [{i}] {reason}")


@torch.inference_mode()
def test_compile_decode(model, device, n_tokens=32):
    """Actually compile and run decode to see recompilation behavior."""
    print("\n" + "=" * 80)
    print(f"TEST: Compile + decode {n_tokens} tokens")
    print("=" * 80)

    torch._dynamo.reset()

    compiled = torch.compile(model, dynamic=True)

    # Prefill
    input_ids = torch.zeros(1, 16, dtype=torch.long, device=device)
    t0 = time.perf_counter()
    out = compiled(input_ids, use_cache=True)
    torch.cuda.synchronize(device)
    print(f"Prefill compile+run: {(time.perf_counter() - t0)*1000:.0f}ms")

    past_kv = out["past_kv"]
    tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

    # Decode tokens one at a time
    times = []
    for i in range(n_tokens):
        t = time.perf_counter()
        out = compiled(tok, use_cache=True, past_kv=past_kv)
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - t) * 1000
        times.append(elapsed)
        past_kv = out["past_kv"]
        tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

        if elapsed > 100:
            print(f"  Token {i}: {elapsed:.0f}ms (SLOW - likely recompilation)")
        elif i < 5 or elapsed > 50:
            print(f"  Token {i}: {elapsed:.1f}ms")

    print(f"\nSummary:")
    print(f"  First token: {times[0]:.0f}ms")
    print(f"  Tokens 1-5 avg: {sum(times[1:6])/5:.1f}ms")
    if len(times) > 10:
        print(f"  Tokens 5-end avg: {sum(times[5:])/len(times[5:]):.1f}ms")
    print(f"  Max: {max(times):.0f}ms")
    print(f"  Min (after first): {min(times[1:]):.1f}ms")

    # Test different prompt lengths to check for recompilation
    print(f"\n--- Testing different prompt lengths after warmup ---")
    for plen in [64, 256, 1024]:
        ids = torch.zeros(1, plen, dtype=torch.long, device=device)
        t0 = time.perf_counter()
        out2 = compiled(ids, use_cache=True)
        torch.cuda.synchronize(device)
        prefill_ms = (time.perf_counter() - t0) * 1000

        past2 = out2["past_kv"]
        tok2 = out2["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

        decode_times = []
        for _ in range(8):
            t = time.perf_counter()
            out2 = compiled(tok2, use_cache=True, past_kv=past2)
            torch.cuda.synchronize(device)
            decode_times.append((time.perf_counter() - t) * 1000)
            past2 = out2["past_kv"]
            tok2 = out2["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

        recompiled = "RECOMPILE" if prefill_ms > 1000 else "ok"
        decode_recompiled = "RECOMPILE" if max(decode_times) > 100 else "ok"
        print(f"  prompt={plen:>4d}: prefill={prefill_ms:>8.1f}ms ({recompiled}), "
              f"decode_avg={sum(decode_times)/len(decode_times):.1f}ms "
              f"decode_max={max(decode_times):.1f}ms ({decode_recompiled})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-compile-test", action="store_true",
                        help="Skip the actual compile+decode test (faster)")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = load_model(args.checkpoint, device)

    diagnose_single_layer(model, device)
    diagnose_route_static(model, device)
    diagnose_cached_forward(model, device)

    if not args.skip_compile_test:
        test_compile_decode(model, device, n_tokens=32)


if __name__ == "__main__":
    main()
