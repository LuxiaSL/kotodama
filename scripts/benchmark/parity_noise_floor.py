#!/usr/bin/env python3
"""Parity noise floor: the reference path against itself across SDPA backends.

Runs llama.py's cached decode twice on identical inputs — once forced to the
FLASH backend, once to EFFICIENT — and reports teacher-forced logit deltas.
This is the delta magnitude attributable purely to attention-backend choice,
i.e. the honest yardstick for judging DecodeEngine parity numbers.

Usage:
    KOTODAMA_NO_TRITON_ATTNRES=1 CUDA_VISIBLE_DEVICES=0 python scripts/benchmark/parity_noise_floor.py \
        --checkpoint /models/.../ckpt.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmark.decode_parity import compare_steps, load_model, reference_forced, reference_greedy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-lens", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    for plen in args.prompt_lens:
        g = torch.Generator().manual_seed(42 + plen)
        input_ids = torch.randint(4, model.config.vocab_size, (1, plen), generator=g).to(device)

        # Unforced reference dispatches to flash_fwd_splitkv at decode
        # (verified in profile traces); forcing flash globally hits an
        # unsupported corner, so the flash side runs unforced.
        ref_tokens, ref_logits = reference_greedy(model, input_ids, args.steps, device)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            eff_logits = reference_forced(model, input_ids, ref_tokens, device)
        flash_again = reference_forced(model, input_ids, ref_tokens, device)

        cmp_backends = compare_steps(ref_logits, eff_logits)
        cmp_self = compare_steps(ref_logits, flash_again)
        print(f"plen={plen} flash-vs-efficient: maxΔ={cmp_backends['max_abs_logit_delta']:.4f} "
              f"argmax={cmp_backends['argmax_agreement']:.3f} top32={cmp_backends['min_top32_overlap']:.3f}")
        print(f"plen={plen} flash-vs-flash (determinism check): maxΔ={cmp_self['max_abs_logit_delta']:.4f} "
              f"argmax={cmp_self['argmax_agreement']:.3f}")


if __name__ == "__main__":
    main()
