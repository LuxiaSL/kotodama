#!/usr/bin/env python3
"""Locate the block-vs-reference KV delta: which tensor/layer/position/dim.

Minimal repro from bench_block_prefill: T=16 single block, no overlap.
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


@torch.inference_mode()
def main() -> int:
    ckpt = sys.argv[1]
    device = torch.device("cuda")
    model = load_model(ckpt, device)
    engine = DecodeEngine(model, max_seq_len=4096)
    engine.compile_step(mode="max-autotune")
    engine.warm_blocks()

    n = 16
    g = torch.Generator().manual_seed(1016)
    ids = torch.randint(4, engine.config.vocab_size, (1, n), generator=g).to(device)

    engine.prefill_reference(ids)
    ref = [(engine.k_caches[li][:, :, :n].clone(), engine.v_caches[li][:, :, :n].clone())
           for li in range(engine.n_layers)]

    engine.prefill(ids)
    blk = [(engine.k_caches[li][:, :, :n].clone(), engine.v_caches[li][:, :, :n].clone())
           for li in range(engine.n_layers)]

    for li in range(engine.n_layers):
        for what, i in (("k", 0), ("v", 1)):
            d = (blk[li][i] - ref[li][i]).abs()
            mx = float(d.max())
            if mx > 0.05:
                # locate: (1, KV, n, dh)
                idx = torch.unravel_index(d.argmax(), d.shape)
                _, h, p, dd = (int(x) for x in idx)
                per_pos = d.amax(dim=(0, 1, 3)).tolist()
                logger.info(
                    "L%02d %s: max %.3f at head=%d pos=%d dim=%d | per-pos max: %s",
                    li, what, mx, h, p, dd,
                    " ".join(f"{v:.2f}" for v in per_pos),
                )
                logger.info("   ref vals around: %s", ref[li][i][0, h, p, max(0, dd - 2): dd + 3].tolist())
                logger.info("   blk vals around: %s", blk[li][i][0, h, p, max(0, dd - 2): dd + 3].tolist())
                break  # one report per layer is enough
        else:
            continue
        if li > 4:
            break

    print("KV_PROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
