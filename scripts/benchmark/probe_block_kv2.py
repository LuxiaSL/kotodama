#!/usr/bin/env python3
"""Discriminate block-forward KV bug: math vs compile/cudagraph.

Legs:
  A. EAGER _forward_block(T=16) vs reference prefill    — isolates block MATH
  B. COMPILED _forward_block(T=16) vs reference         — adds compile/cudagraph
  C. EAGER extend (prefill_reference 1024 + eager block suffix 64) vs eager
     MATH extend                                        — extend math
  D. COMPILED extend                                    — the failing deployment path

Each leg: all-layer scan, report worst (layer, k|v, head, pos, dim), whether
ref/blk is ~zero at that coordinate, and 5 values around it.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", "1")

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark.decode_parity import load_model  # noqa: E402
from src.model.decode_engine import DecodeEngine  # noqa: E402


def snap(engine: DecodeEngine, n: int):
    return [(engine.k_caches[li][:, :, :n].clone(), engine.v_caches[li][:, :, :n].clone())
            for li in range(engine.n_layers)]


def report(tag: str, ref, blk) -> None:
    worst, where = 0.0, None
    for li, ((rk, rv), (bk, bv)) in enumerate(zip(ref, blk)):
        for what, r, b in (("k", rk, bk), ("v", rv, bv)):
            rel = (b - r).abs() / (r.abs() + 0.5)
            m = float(rel.max())
            if m > worst:
                worst = m
                idx = torch.unravel_index(rel.argmax(), rel.shape)
                where = (li, what, tuple(int(x) for x in idx), r, b)
    if where is None:
        logger.info("%s: identical", tag)
        return
    li, what, (_, h, p, d), r, b = where
    logger.info("%s: worst rel %.3f at L%02d %s head=%d pos=%d dim=%d | ref=%.4f blk=%.4f",
                tag, worst, li, what, h, p, d, float(r[0, h, p, d]), float(b[0, h, p, d]))
    logger.info("   ref row: %s", [round(float(x), 3) for x in r[0, h, p, max(0, d - 2): d + 3]])
    logger.info("   blk row: %s", [round(float(x), 3) for x in b[0, h, p, max(0, d - 2): d + 3]])
    per_pos = (b - r).abs().amax(dim=(0, 1, 3)) if what else None
    logger.info("   abs per-pos max (%s, L%02d): %s", what, li,
                [round(float(x), 2) for x in (b - r).abs().amax(dim=(0, 1, 3))])


@torch.inference_mode()
def main() -> int:
    ckpt = sys.argv[1]
    device = torch.device("cuda")
    model = load_model(ckpt, device)
    engine = DecodeEngine(model, max_seq_len=4096)

    n = 16
    g = torch.Generator().manual_seed(1016)
    ids = torch.randint(4, engine.config.vocab_size, (1, n), generator=g).to(device)
    big = torch.randint(4, engine.config.vocab_size, (1, 1088), generator=g).to(device)

    # Reference legs (no compiled block exists yet -> prefill_cached is eager MATH)
    engine.prefill_reference(ids)
    ref16 = snap(engine, n)
    engine.prefill_reference(big[:, :1024])
    _, info = engine.prefill_cached(big)
    assert info["prefix_hit"]
    ref_ext = snap(engine, 1088)

    # A: eager block math, T=16
    engine.reset()
    engine.pos.zero_()
    engine._forward_block(ids)
    blk16_eager = snap(engine, n)
    report("A eager block T=16", ref16, blk16_eager)

    # C: eager block extend
    engine.prefill_reference(big[:, :1024])
    engine.pos.fill_(1024)
    engine._forward_block(big[:, 1024:1088].contiguous())
    blk_ext_eager = snap(engine, 1088)
    report("C eager block extend", ref_ext, blk_ext_eager)

    # Compile and warm
    engine.compile_step(mode="max-autotune")
    engine.warm_blocks()

    # B: compiled block prefill T=16
    engine.prefill(ids)
    blk16_c = snap(engine, n)
    report("B compiled block T=16", ref16, blk16_c)

    # D: compiled block extend
    engine.prefill_reference(big[:, :1024])
    _, info = engine.prefill_cached(big)
    logger.info("D extend info: %s", info)
    blk_ext_c = snap(engine, 1088)
    report("D compiled block extend", ref_ext, blk_ext_c)

    print("KV_PROBE2_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
