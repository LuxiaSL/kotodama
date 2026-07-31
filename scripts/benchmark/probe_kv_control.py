#!/usr/bin/env python3
"""CONTROL: how much do two KNOWN-CORRECT prefill paths differ in cached KV?

Compares, at T=16 (same ids as the block probes):
  R1: llama prefill, Triton phase-1 routing        (fleet family)
  R2: llama prefill, eager PyTorch routing          (monkeypatched)
  BLK: eager DecodeEngine._forward_block            (the accused)

Reports worst relative delta per pair per layer band. If R1-vs-R2 shows the
same deep-layer divergence scale as BLK-vs-R*, the divergence is trunk dust
amplification (any kernel-schedule change reshuffles it) and the block math
is within the established family. No compile involved.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import src.model.llama as llama_mod  # noqa: E402
from scripts.benchmark.decode_parity import load_model  # noqa: E402
from src.model.decode_engine import DecodeEngine  # noqa: E402


def snap(engine: DecodeEngine, n: int):
    return [(engine.k_caches[li][:, :, :n].clone(), engine.v_caches[li][:, :, :n].clone())
            for li in range(engine.n_layers)]


def worst_by_band(ref, blk, n_layers: int) -> dict[str, float]:
    bands = {"L00-06": (0, 7), "L07-15": (7, 16), "L16-27": (16, n_layers)}
    out = {}
    for name, (lo, hi) in bands.items():
        w = 0.0
        for li in range(lo, hi):
            for i in (0, 1):
                rel = (blk[li][i] - ref[li][i]).abs() / (ref[li][i].abs() + 0.5)
                w = max(w, float(rel.max()))
        out[name] = round(w, 3)
    return out


@torch.inference_mode()
def main() -> int:
    ckpt = sys.argv[1]
    device = torch.device("cuda")
    model = load_model(ckpt, device)
    engine = DecodeEngine(model, max_seq_len=4096)

    n = 16
    g = torch.Generator().manual_seed(1016)
    ids = torch.randint(4, engine.config.vocab_size, (1, n), generator=g).to(device)

    assert llama_mod._TRITON_ATTN_RES_AVAILABLE, "run WITHOUT KOTODAMA_NO_TRITON_ATTNRES"
    engine.prefill_reference(ids)
    r1 = snap(engine, n)

    llama_mod._TRITON_ATTN_RES_AVAILABLE = False
    engine.prefill_reference(ids)
    r2 = snap(engine, n)
    llama_mod._TRITON_ATTN_RES_AVAILABLE = True

    engine.reset()
    engine.pos.zero_()
    engine._forward_block(ids)
    blk = snap(engine, n)

    logger.info("CONTROL R1(triton-llama) vs R2(eager-llama): %s", worst_by_band(r1, r2, engine.n_layers))
    logger.info("BLK vs R1(triton-llama):                     %s", worst_by_band(r1, blk, engine.n_layers))
    logger.info("BLK vs R2(eager-llama):                      %s", worst_by_band(r2, blk, engine.n_layers))

    # Also the logit-level effect of each pair (last-position, fp32).
    lg1 = engine.prefill_reference(ids).view(-1)
    llama_mod._TRITON_ATTN_RES_AVAILABLE = False
    lg2 = engine.prefill_reference(ids).view(-1)
    llama_mod._TRITON_ATTN_RES_AVAILABLE = True
    engine.reset(); engine.pos.zero_()
    lgb = engine._forward_block(ids)[0, -1].float().view(-1)
    logger.info("logits maxΔ: R1-R2 %.4f | BLK-R1 %.4f | BLK-R2 %.4f",
                float((lg1 - lg2).abs().max()), float((lgb - lg1).abs().max()), float((lgb - lg2).abs().max()))

    print("KV_CONTROL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
