#!/usr/bin/env python3
"""LAMBADA decomposition probe: rank/entropy/in-context analysis of the final-word
conditional, comparing kotodama checkpoints and HF reference models.

Separates three failure stories for LAMBADA accuracy deficits:
  - "soft commitment": gold token ranked 2-10 with high conditional entropy
  - "missing stored knowledge": gold ranked deep, mostly on items whose target
    does NOT appear in the context
  - "broken in-context binding": failures concentrated on items whose target DOES
    appear verbatim in context (copy/induction should solve these)

Usage (gpu-host, from ~/workspace/kotodama):
    # kotodama checkpoint
    python -m scripts.analysis.lambada_probe --model kotodama \
        --checkpoint /models/kotodama-data/3b-language-FINAL-step195311.pt.zst \
        -o analysis/lambada_probe/kotodama_final.json
    # HF reference
    python -m scripts.analysis.lambada_probe --model EleutherAI/pythia-2.8b \
        -o analysis/lambada_probe/pythia28.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_kotodama(checkpoint: str, device: str):
    from transformers import AutoTokenizer

    from src.eval.model_loader import load_model

    model = load_model(
        checkpoint_path=checkpoint,
        config_path="configs/model.yaml",
        config_section="model",
        attn_res_config={"attn_res": True,
                         "attn_res_boundaries": [0, 1, 3, 7, 15, 19, 24]},
        device=device,
    )
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    def logits_fn(ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(x)
        return out["logits"][0].float()

    return tok, logits_fn


def load_hf(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16).to(device).eval()

    def logits_fn(ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(x)
        return out.logits[0].float()

    return tok, logits_fn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="'kotodama' or HF repo id")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    logger.info("LAMBADA items: %d", len(ds))

    if args.model == "kotodama":
        if not args.checkpoint:
            raise SystemExit("--checkpoint required for kotodama")
        tok, logits_fn = load_kotodama(args.checkpoint, args.device)
        name = Path(args.checkpoint).stem
    else:
        tok, logits_fn = load_hf(args.model, args.device)
        name = args.model

    rows = []
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    for i in range(n):
        text = ds[i]["text"]
        try:
            context, word = text.rsplit(" ", 1)
        except ValueError:
            continue
        tgt_ids = tok.encode(" " + word, add_special_tokens=False)
        ctx_ids = tok.encode(context, add_special_tokens=False)
        if not tgt_ids or not ctx_ids:
            continue

        full = ctx_ids + tgt_ids
        logits = logits_fn(full)

        # final-word conditional = logits at last context position
        cond = logits[len(ctx_ids) - 1]
        logp = F.log_softmax(cond, dim=-1)
        p_ = logp.exp()
        gold = tgt_ids[0]
        rank = int((cond > cond[gold]).sum().item()) + 1
        entropy = float(-(p_ * logp).sum().item())
        top1 = int(cond.argmax().item())

        # full-continuation greedy match (lm-eval acc definition)
        greedy_ok = True
        for j, t in enumerate(tgt_ids):
            pos = len(ctx_ids) - 1 + j
            if int(logits[pos].argmax().item()) != t:
                greedy_ok = False
                break

        # target-in-context split (word-boundary, exact + casefold)
        in_ctx = bool(re.search(r"\b" + re.escape(word) + r"\b", context))
        in_ctx_ci = bool(re.search(r"\b" + re.escape(word) + r"\b", context,
                                   re.IGNORECASE))

        rows.append({
            "i": i, "word": word, "gold_first": gold, "rank": rank,
            "gold_logp": float(logp[gold].item()), "entropy": entropy,
            "top1_correct": top1 == gold, "greedy_correct": greedy_ok,
            "n_tgt_tokens": len(tgt_ids), "in_ctx": in_ctx,
            "in_ctx_ci": in_ctx_ci, "ctx_tokens": len(ctx_ids),
        })
        if (i + 1) % 500 == 0:
            logger.info("%d/%d", i + 1, n)

    def agg(sub: list[dict]) -> dict:
        if not sub:
            return {"n": 0}
        ranks = np.array([r["rank"] for r in sub])
        wrong = ranks[~np.array([r["top1_correct"] for r in sub])]
        return {
            "n": len(sub),
            "acc_top1_first": float(np.mean([r["top1_correct"] for r in sub])),
            "acc_greedy_full": float(np.mean([r["greedy_correct"] for r in sub])),
            "gold_ppl_first": float(np.exp(-np.mean([r["gold_logp"] for r in sub]))),
            "median_rank": float(np.median(ranks)),
            "median_rank_when_wrong": float(np.median(wrong)) if len(wrong) else None,
            "top5": float(np.mean(ranks <= 5)),
            "top50": float(np.mean(ranks <= 50)),
            "top500": float(np.mean(ranks <= 500)),
            "mean_entropy": float(np.mean([r["entropy"] for r in sub])),
        }

    result = {
        "model": name,
        "overall": agg(rows),
        "target_in_context": agg([r for r in rows if r["in_ctx_ci"]]),
        "target_not_in_context": agg([r for r in rows if not r["in_ctx_ci"]]),
        "frac_in_context": float(np.mean([r["in_ctx_ci"] for r in rows])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": result, "rows": rows}, f)
    logger.info("wrote %s", args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
