#!/usr/bin/env python3
"""BLiMP critical-token and margin probe.

For minimal pairs, computes BOTH scoring variants per item:
  - full-sentence logprob margin (standard lm-eval scoring)
  - critical-span margin: logprob over only the differing token span,
    conditioned on the (identical) prefix — isolates the decision point
    from spillover/suffix likelihood differences.

Distinguishes: real structural failure (critical margin wrong sign) vs
register/spillover intrusion (full margin wrong, critical margin right) vs
softness compression (correct sign, margins shrunk toward zero).

Usage (gpu-host, from ~/workspace/kotodama):
    python -m scripts.analysis.blimp_margin_probe --model kotodama \
        --checkpoint /models/kotodama-data/3b-language-FINAL-step195311.pt.zst \
        -o analysis/blimp_probe/kotodama.json
    python -m scripts.analysis.blimp_margin_probe --model EleutherAI/pythia-2.8b \
        -o analysis/blimp_probe/pythia28.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SUBTASKS = [
    "principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
    "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
    "principle_A_reconstruction",
    "left_branch_island_simple_question", "sentential_negation_npi_scope",
    "only_npi_licensor_present",
]


def common_affix_lens(a: list[int], b: list[int]) -> tuple[int, int]:
    """Longest common token prefix and suffix lengths (non-overlapping)."""
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while (s < min(len(a), len(b)) - p) and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    return p, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    from datasets import load_dataset

    if args.model == "kotodama":
        from transformers import AutoTokenizer

        from src.eval.model_loader import load_model
        model = load_model(args.checkpoint, config_path="configs/model.yaml",
                           config_section="model",
                           attn_res_config={"attn_res": True,
                                            "attn_res_boundaries": [0, 1, 3, 7, 15, 19, 24]},
                           device=args.device)
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

        def logprobs(ids: list[int]) -> torch.Tensor:
            x = torch.tensor([ids], dtype=torch.long, device=args.device)
            with torch.no_grad():
                lg = model(x)["logits"][0].float()
            return F.log_softmax(lg, dim=-1)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16).to(args.device).eval()

        def logprobs(ids: list[int]) -> torch.Tensor:
            x = torch.tensor([ids], dtype=torch.long, device=args.device)
            with torch.no_grad():
                lg = model(x).logits[0].float()
            return F.log_softmax(lg, dim=-1)

    bos = tok.eos_token_id  # EOT-as-context, mirroring harness sentence scoring

    def seq_logprob_spans(sent: str) -> tuple[list[int], torch.Tensor]:
        ids = tok.encode(sent, add_special_tokens=False)
        lp = logprobs([bos] + ids)
        # token i of ids is predicted at position i (after bos prepend)
        tok_lps = torch.stack([lp[i, t] for i, t in enumerate(ids)])
        return ids, tok_lps

    rows = []
    for st in SUBTASKS:
        ds = load_dataset("blimp", st, split="train")
        for i, ex in enumerate(ds):
            g_ids, g_lps = seq_logprob_spans(ex["sentence_good"])
            b_ids, b_lps = seq_logprob_spans(ex["sentence_bad"])
            p, s = common_affix_lens(g_ids, b_ids)
            g_crit = float(g_lps[p:len(g_ids) - s].sum().item()) if len(g_ids) - s > p else 0.0
            b_crit = float(b_lps[p:len(b_ids) - s].sum().item()) if len(b_ids) - s > p else 0.0
            rows.append({
                "subtask": st, "i": i,
                "full_margin": float(g_lps.sum().item() - b_lps.sum().item()),
                "crit_margin": g_crit - b_crit,
                "crit_len_good": len(g_ids) - s - p,
                "prefix_len": p,
            })
        logger.info("%s done (%d items)", st, len(ds))

    # aggregate
    agg = {}
    for st in SUBTASKS:
        sub = [r for r in rows if r["subtask"] == st]
        fm = np.array([r["full_margin"] for r in sub])
        cm = np.array([r["crit_margin"] for r in sub])
        agg[st] = {
            "n": len(sub),
            "acc_full": float((fm > 0).mean()),
            "acc_crit": float((cm > 0).mean()),
            "median_full_margin": float(np.median(fm)),
            "median_crit_margin": float(np.median(cm)),
            "mean_abs_full_margin": float(np.abs(fm).mean()),
            "mean_abs_crit_margin": float(np.abs(cm).mean()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": args.model, "agg": agg, "rows": rows}, open(args.output, "w"))
    logger.info("wrote %s", args.output)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
