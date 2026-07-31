#!/usr/bin/env python3
"""Data-difficulty trajectory under FIXED final weights — large-n, noise-controlled.

Resolves the residual: is the late online train/loss rise genuine DATA drift, or a
small-sample artifact? Reconstructs the EXACT sequences rank-0 consumed at each
given step, scores them all through the FINAL bf16 weights (doc-masked, chunked so
large n fits), and reports a token-weighted mean per step. Stationary => permutation
proof holds (rise is a forward subtlety); rising => data genuinely drifts.

Usage (gpu-host):
    source ~/workspace/.venv-shared/bin/activate
    KOTODAMA_NO_TRITON_ATTNRES=1 python scripts/analysis/loss_trajectory.py \
        --checkpoint /models/kotodama-data/3b-language-FINAL-step195311.pt.zst \
        --train-bin /models/kotodama-data/train.bin \
        --steps 100000,130000,155000,175000,195000 --n-seqs 256 --chunk 16
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import collate_packed, compute_doc_boundaries, PackedSample
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("loss_trajectory")

CONFIG_3B = dict(
    hidden_size=3072, num_layers=28, num_attention_heads=24, num_kv_heads=8, head_dim=128,
    intermediate_size=8192, vocab_size=49152, max_position_embeddings=4096, rope_theta=500000.0,
    norm_eps=1e-5, qk_norm=True, tie_word_embeddings=True, z_loss_weight=0.0, use_liger=False,
    attn_impl="auto", attn_res=True, attn_res_boundaries=[0, 1, 3, 7, 15, 19, 24],
)
SEQ_LEN, SEED, WORLD, SEQS_PER_STEP, EOS_ID = 4096, 42, 8, 60, 0


def rank0_perm(train_bin: str) -> np.ndarray:
    data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    spr = (len(data) // SEQ_LEN) // WORLD
    rng = np.random.RandomState(SEED + 0)
    idx = np.arange(0, spr)
    rng.shuffle(idx)
    return idx


def consumed_samples(train_bin: str, perm: np.ndarray, step: int, n: int) -> list[PackedSample]:
    data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    out = []
    for sid in perm[step * SEQS_PER_STEP: step * SEQS_PER_STEP + n]:
        raw = np.asarray(data[sid * SEQ_LEN:(sid + 1) * SEQ_LEN])
        cu, pos, mx = compute_doc_boundaries(raw, eos_id=EOS_ID)
        out.append(PackedSample(torch.from_numpy(raw.astype(np.int64)),
                                torch.from_numpy(cu), torch.from_numpy(pos).long(), mx))
    return out


@torch.no_grad()
def chunked_loss(model, samples: list[PackedSample], device: str, chunk: int) -> tuple[float, float]:
    """Token-weighted mean CE over all samples, processed in chunks. Returns (mean, sem)."""
    chunk_losses, chunk_w = [], []
    for i in range(0, len(samples), chunk):
        sub = samples[i:i + chunk]
        b = collate_packed(sub)
        ids = b["input_ids"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, labels=ids, cu_seqlens=b["cu_seqlens"].to(device),
                        position_ids=b["position_ids"].to(device), max_seqlen=b["max_seqlen"])
        chunk_losses.append(float(out["loss"].item()))
        chunk_w.append(len(sub))
    w = np.array(chunk_w, dtype=float)
    l = np.array(chunk_losses, dtype=float)
    mean = float((l * w).sum() / w.sum())
    # SEM across chunks (equal-ish weights), rough uncertainty
    sem = float(l.std() / max(np.sqrt(len(l)), 1)) if len(l) > 1 else 0.0
    return mean, sem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--steps", default="100000,130000,155000,175000,195000")
    ap.add_argument("--n-seqs", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    perm = rank0_perm(args.train_bin)
    logger.info("permutation built; steps=%s n=%d chunk=%d", steps, args.n_seqs, args.chunk)

    p = Path(args.checkpoint)
    logger.info("decompressing %s ...", p.name)
    import zstandard as zstd
    with open(p, "rb") as f:
        raw = zstd.ZstdDecompressor().stream_reader(f).read()
    ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    config = LuxiaModelConfig(**CONFIG_3B)
    model = LuxiaBaseModel(config)
    model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    model = model.to(device=args.device, dtype=torch.bfloat16).eval()
    logger.info("final weights loaded (step %s)", ckpt.get("step"))

    print(f"\n{'step':>8} {'n':>5} {'loss(final wts)':>16} {'sem':>7}   online_train/loss(for ref)")
    results = {}
    for s in steps:
        t0 = time.time()
        samples = consumed_samples(args.train_bin, perm, s, args.n_seqs)
        mean, sem = chunked_loss(model, samples, args.device, args.chunk)
        results[s] = mean
        logger.info("step %d: loss=%.4f sem=%.4f (%.0fs, %d docs)", s, mean, sem,
                    time.time() - t0, sum(len(x.cu_seqlens) - 1 for x in samples))
        print(f"{s:>8} {args.n_seqs:>5} {mean:>16.4f} {sem:>7.4f}")

    if len(steps) >= 2:
        lo, hi = min(steps), max(steps)
        print(f"\nDATA-DRIFT (fixed final weights): loss@{hi} - loss@{lo} = {results[hi]-results[lo]:+.4f}")
        print("  ~0  => data stationary (permutation proof holds; rise is a forward subtlety)")
        print("  +   => data genuinely drifts harder late (the rise is real data)")


if __name__ == "__main__":
    main()
