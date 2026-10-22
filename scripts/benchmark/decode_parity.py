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
        --checkpoint <checkpoint.pt> --output outputs/profiles/parity_v1.json
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

torch.set_num_threads(2)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kotodama import ckpt
from kotodama.model import llama as llama_mod
from kotodama.model.decode_engine import DecodeEngine
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

DD3B_BOUNDARIES = [0, 1, 3, 7, 15, 19, 24]
CONFIG_3B = dict(
    hidden_size=3072, num_layers=28, num_attention_heads=24, num_kv_heads=8,
    head_dim=128, intermediate_size=8192, vocab_size=49152,
    max_position_embeddings=4096, rope_theta=500000.0, norm_eps=1e-5,
    qk_norm=True, tie_word_embeddings=True, z_loss_weight=0.0,
    use_liger=False, attn_impl="sdpa", attn_res=True,
    attn_res_boundaries=DD3B_BOUNDARIES,
)


def load_model(checkpoint: str, device: torch.device, model_size: str = "3b",
               rope_fp32: bool = False) -> KotodamaModel:
    """Build the served architecture for `model_size` (server.MODEL_CONFIGS — the serving table) and load."""
    if model_size == "3b":
        cfg = dict(CONFIG_3B)
    else:
        from kotodama.serve.server import MODEL_CONFIGS
        cfg = dict(MODEL_CONFIGS[model_size])
    if rope_fp32:
        cfg["rope_fp32"] = True
    model = KotodamaModel(KotodamaModelConfig(**cfg))
    logger.info("Loading checkpoint: %s (model_size=%s, rope_fp32=%s)", checkpoint, model_size, rope_fp32)
    state = ckpt.load(checkpoint)
    model.load_state_dict(state.get("model", state), strict=True)
    del state
    return model.to(device).eval().bfloat16()


@torch.inference_mode()
def full_forward_argmax(model: KotodamaModel, input_ids: torch.Tensor, tokens: list[int],
                        device: torch.device) -> list[int]:
    """No-cache oracle: one full forward over prompt + trajectory; argmax at each generating position."""
    seq = torch.cat([input_ids, torch.tensor([tokens], device=device)], dim=1)
    logits = model(seq)["logits"][0]
    return logits[input_ids.shape[1] - 1:-1].argmax(-1).tolist()


@torch.inference_mode()
def reference_greedy(
    model: KotodamaModel, input_ids: torch.Tensor, n_steps: int, device: torch.device
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
    model: KotodamaModel, input_ids: torch.Tensor, forced: list[int], device: torch.device
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
    step from an identical starting cache. The block prefill/extend
    paths have their own battery (bench_block_prefill.py); letting them in
    here injects known-benign near-tie dust that trips the exact argmax gate
    (e.g. 63/64 at plen=16).
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
    parser.add_argument("--model_size", default="3b",
                        help="Served architecture (kotodama.serve.server.MODEL_CONFIGS key, e.g. 1b-moonshot-loop)")
    parser.add_argument("--rope_fp32", action="store_true", help="Model config rope_fp32 (the 1B ladder lineage)")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Triton routing in reference: %s (want False for clean parity)",
                llama_mod._TRITON_ATTN_RES_AVAILABLE)

    model = load_model(args.checkpoint, device, args.model_size, args.rope_fp32)
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
        "model_size": args.model_size,
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

        # Full-forward oracle (no KV cache anywhere): does each cached path's greedy token equal the argmax of
        # one full forward over prompt + that path's own trajectory?
        ff_eng = full_forward_argmax(model, input_ids, eng_tokens, device)
        ff_ref = full_forward_argmax(model, input_ids, ref_tokens, device)
        full_fwd = {
            "engine_vs_full_forward_match": round(sum(a == b for a, b in zip(eng_tokens, ff_eng)) / len(eng_tokens), 4),
            "reference_vs_full_forward_match": round(sum(a == b for a, b in zip(ref_tokens, ff_ref)) / len(ref_tokens), 4),
        }

        results["prompts"][str(plen)] = {
            "teacher_forced": forced_cmp,
            "full_forward": full_fwd,
            "greedy_first_divergence": div,
            "greedy_match_steps": div["step"] if div["step"] is not None else args.steps,
        }
        status = "PASS" if forced_cmp["argmax_agreement"] >= 0.99 and forced_cmp["min_top32_overlap"] >= 0.95 else "CHECK"
        if status != "PASS":
            all_pass = False
        logger.info(
            "plen=%4d  forced: maxΔ=%.4f argmax=%.3f top32=%.3f | greedy diverges@%s (margin %s) | "
            "full-forward argmax match: engine %.3f, reference %.3f  [%s]",
            plen, forced_cmp["max_abs_logit_delta"], forced_cmp["argmax_agreement"],
            forced_cmp["min_top32_overlap"], div["step"], div.get("ref_top1_top2_margin"),
            full_fwd["engine_vs_full_forward_match"], full_fwd["reference_vs_full_forward_match"], status,
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
