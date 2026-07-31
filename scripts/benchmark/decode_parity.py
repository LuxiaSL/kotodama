#!/usr/bin/env python3
"""Parity gate: DecodeEngine vs the reference llama.py cached path.

Two comparisons per prompt:
  1. Teacher-forced logit parity: feed the REFERENCE greedy trajectory into
     both implementations; per-step max|Δlogit|, top-k overlap. This is the
     pure numerics measure (identical inputs at every step).
  2. Free greedy agreement: each implementation decodes greedily on its own;
     report first divergence step and the reference top1-top2 margin there
     (divergence at a near-tie is expected bf16 behavior, not a bug).

Run with KOTODAMA_NO_TRITON_ATTNRES=1 so the reference uses the same PyTorch
routing math the engine replicates.

Usage:
    KOTODAMA_NO_TRITON_ATTNRES=1 CUDA_VISIBLE_DEVICES=0 python scripts/benchmark/decode_parity.py \
        --checkpoint /models/.../step_00000358.pt --output outputs/profiles/parity_v1.json
"""

from __future__ import annotations

import argparse
import io
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

torch.set_num_threads(2)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model import llama as llama_mod
from src.model.decode_engine import DecodeEngine
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

DD3B_BOUNDARIES = [0, 1, 3, 7, 15, 19, 24]
CONFIG_3B = dict(
    hidden_size=3072, num_layers=28, num_attention_heads=24, num_kv_heads=8,
    head_dim=128, intermediate_size=8192, vocab_size=49152,
    max_position_embeddings=4096, rope_theta=500000.0, norm_eps=1e-5,
    qk_norm=True, tie_word_embeddings=True, z_loss_weight=0.0,
    use_liger=False, attn_impl="sdpa", attn_res=True,
    attn_res_boundaries=DD3B_BOUNDARIES,
)


def load_model(checkpoint: str, device: torch.device) -> LuxiaBaseModel:
    config = LuxiaModelConfig(**CONFIG_3B)
    model = LuxiaBaseModel(config)
    ckpt_path = Path(checkpoint)
    logger.info("Loading checkpoint: %s", ckpt_path)
    if ckpt_path.suffix == ".zst":
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(ckpt_path, "rb") as f_in:
            decompressed = dctx.decompress(f_in.read())
        ckpt = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
        del decompressed
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    return model.to(device).eval().bfloat16()


@torch.inference_mode()
def reference_greedy(
    model: LuxiaBaseModel, input_ids: torch.Tensor, n_steps: int, device: torch.device
) -> tuple[list[int], list[torch.Tensor]]:
    """Greedy decode with the reference cached path. Returns (tokens, per-step logits fp32 cpu)."""
    out = model(input_ids, use_cache=True)
    past_kv = out["past_kv"]
    logits_list: list[torch.Tensor] = [out["logits"][0, -1].float().cpu()]
    tokens: list[int] = [int(out["logits"][0, -1].argmax().item())]
    tok = torch.tensor([[tokens[-1]]], device=device)
    for _ in range(n_steps - 1):
        out = model(tok, use_cache=True, past_kv=past_kv)
        past_kv = out["past_kv"]
        logits_list.append(out["logits"][0, -1].float().cpu())
        tokens.append(int(out["logits"][0, -1].argmax().item()))
        tok = torch.tensor([[tokens[-1]]], device=device)
    return tokens, logits_list


@torch.inference_mode()
def reference_forced(
    model: LuxiaBaseModel, input_ids: torch.Tensor, forced: list[int], device: torch.device
) -> list[torch.Tensor]:
    """Reference logits along a forced token trajectory."""
    out = model(input_ids, use_cache=True)
    past_kv = out["past_kv"]
    logits_list: list[torch.Tensor] = [out["logits"][0, -1].float().cpu()]
    for t in forced[:-1]:
        out = model(torch.tensor([[t]], device=device), use_cache=True, past_kv=past_kv)
        past_kv = out["past_kv"]
        logits_list.append(out["logits"][0, -1].float().cpu())
    return logits_list


@torch.inference_mode()
def engine_forced(engine: DecodeEngine, input_ids: torch.Tensor, forced: list[int], device: torch.device) -> list[torch.Tensor]:
    """Engine logits along a forced token trajectory.

    Prefill is PINNED to the reference path: this harness gates the DECODE
    step from an identical starting cache (the round-1 design — engine
    prefill used to copy reference KV exactly). The block prefill/extend
    paths have their own battery (bench_block_prefill.py); letting them in
    here injects known-benign near-tie dust that trips the exact argmax gate
    (observed 2026-07-05: 63/64 at plen=16).
    """
    prefill_logits = engine.prefill_reference(input_ids)
    logits_list: list[torch.Tensor] = [prefill_logits[0].float().cpu()]
    for t in forced[:-1]:
        logits = engine.step_logits(torch.tensor([[t]], device=device))
        logits_list.append(logits[0, -1].float().cpu())
    return logits_list


def compare_steps(ref: list[torch.Tensor], eng: list[torch.Tensor], topk: int = 32) -> dict[str, Any]:
    max_abs = []
    argmax_match = []
    topk_overlap = []
    for r, e in zip(ref, eng):
        max_abs.append(float((r - e).abs().max()))
        argmax_match.append(bool(r.argmax() == e.argmax()))
        rt = set(r.topk(topk).indices.tolist())
        et = set(e.topk(topk).indices.tolist())
        topk_overlap.append(len(rt & et) / topk)
    return {
        "n_steps": len(max_abs),
        "max_abs_logit_delta": round(max(max_abs), 5),
        "mean_abs_logit_delta": round(sum(max_abs) / len(max_abs), 5),
        "argmax_agreement": round(sum(argmax_match) / len(argmax_match), 4),
        "min_top32_overlap": round(min(topk_overlap), 4),
        "per_step_max_delta_first8": [round(x, 5) for x in max_abs[:8]],
        "per_step_max_delta_last8": [round(x, 5) for x in max_abs[-8:]],
    }


def first_divergence(ref_tokens: list[int], eng_tokens: list[int], ref_logits: list[torch.Tensor]) -> dict[str, Any]:
    for i, (r, e) in enumerate(zip(ref_tokens, eng_tokens)):
        if r != e:
            top2 = ref_logits[i].topk(2).values
            return {
                "step": i,
                "ref_token": r,
                "eng_token": e,
                "ref_top1_top2_margin": round(float(top2[0] - top2[1]), 5),
            }
    return {"step": None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-lens", type=int, nargs="+", default=[16, 256, 1024])
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--compile", action="store_true", help="Test the compiled engine step")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Triton routing in reference: %s (want False for clean parity)",
                llama_mod._TRITON_ATTN_RES_AVAILABLE)

    model = load_model(args.checkpoint, device)
    engine = DecodeEngine(model)
    if args.compile:
        engine.compile_step(args.compile_mode)
        # trigger compilation outside timing
        ids = torch.randint(4, model.config.vocab_size, (1, 16)).to(device)
        engine.prefill(ids)
        t0 = time.time()
        engine.step_logits(torch.tensor([[100]], device=device))
        engine.step_logits(torch.tensor([[101]], device=device))
        logger.info("Compile warmup done in %.1fs", time.time() - t0)

    results: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "compiled": args.compile,
        "steps": args.steps,
        "prompts": {},
    }

    all_pass = True
    for plen in args.prompt_lens:
        g = torch.Generator().manual_seed(42 + plen)
        input_ids = torch.randint(4, model.config.vocab_size, (1, plen), generator=g).to(device)

        ref_tokens, ref_logits = reference_greedy(model, input_ids, args.steps, device)
        eng_logits_forced = engine_forced(engine, input_ids, ref_tokens, device)
        forced_cmp = compare_steps(ref_logits, eng_logits_forced)

        eng_tokens = engine.generate_greedy(input_ids, args.steps)
        div = first_divergence(ref_tokens, eng_tokens, ref_logits)

        results["prompts"][str(plen)] = {
            "teacher_forced": forced_cmp,
            "greedy_first_divergence": div,
            "greedy_match_steps": div["step"] if div["step"] is not None else args.steps,
        }
        status = "PASS" if forced_cmp["argmax_agreement"] >= 0.99 and forced_cmp["min_top32_overlap"] >= 0.95 else "CHECK"
        if status != "PASS":
            all_pass = False
        logger.info(
            "plen=%4d  forced: maxΔ=%.4f argmax=%.3f top32=%.3f | greedy diverges@%s (margin %s)  [%s]",
            plen, forced_cmp["max_abs_logit_delta"], forced_cmp["argmax_agreement"],
            forced_cmp["min_top32_overlap"], div["step"], div.get("ref_top1_top2_margin"), status,
        )

    results["all_pass"] = all_pass
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        logger.info("Written: %s", out)
    logger.info("OVERALL: %s", "PASS" if all_pass else "CHECK FAILURES ABOVE")


if __name__ == "__main__":
    main()
