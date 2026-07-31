#!/usr/bin/env python3
"""Loss-divergence investigation for the 3B cooldown.

Adjudicates: is the late-run train/loss rise (while the fixed-probe loss fell)
DATA (streaming distribution drift) or NUMERICS (fp8 / compiled-path error)?

Method — fixed weights from one checkpoint, vary two axes independently:
  DATA axis    : bf16-eager loss on sequences rank-0 consumed near step A vs step B.
                 (permutation proof says these must be ~equal; this is the empirical backstop)
  NUMERICS axis: same batch, bf16-eager vs fp8-eager (vs fp8-compiled).
                 bf16-eager = TRUTH; fp8-eager isolates fp8 quant; compiled adds fusion delta.

All variants run in EVAL mode with materialized-logits CE and identical doc-masking,
so the only thing that moves within each axis is the axis itself.

Also dumps per-Linear input-activation absmax (the fp8 tensorwise scale driver) for
the streaming batch vs the tame probe batch — the hypothesised mechanism.

Usage (gpu-host, after run-end, GPUs free):
    source ~/workspace/.venv-shared/bin/activate
    python scripts/analysis/loss_investigation.py \
        --checkpoint /models/kotodama-data/3b-language-FINAL-step195311.pt.zst \
        --train-bin /models/kotodama-data/train.bin \
        --steps 155000,195000 --n-seqs 32 \
        --variants bf16,fp8 --out outputs/loss_investigation_final.json

    # add fp8c to --variants for the compiled add-on check (heavier, optional)

Dry-run the CPU plumbing without a GPU forward:
    python scripts/analysis/loss_investigation.py --dry-run --train-bin /models/kotodama-data/train.bin
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import collate_packed, compute_doc_boundaries, PackedSample
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("loss_investigation")

# DD-3B architecture (mirror serve.py CONFIG_3B). use_liger=False -> plain F.cross_entropy CE.
CONFIG_3B = dict(
    hidden_size=3072, num_layers=28, num_attention_heads=24, num_kv_heads=8, head_dim=128,
    intermediate_size=8192, vocab_size=49152, max_position_embeddings=4096, rope_theta=500000.0,
    norm_eps=1e-5, qk_norm=True, tie_word_embeddings=True, z_loss_weight=0.0, use_liger=False,
    attn_impl="auto", attn_res=True, attn_res_boundaries=[0, 1, 3, 7, 15, 19, 24],  # auto->fa2 varlen (match training masking)
)

# Data-loader constants (must match training: TokenizedDataset + 3b-language.yaml)
SEQ_LEN = 4096
SEED = 42
WORLD = 8
SEQS_PER_STEP = 60        # rank 0: micro_batch 10 * grad_accum 6
EOS_ID = 0


def rank0_permutation(train_bin: str) -> np.ndarray:
    """Reproduce rank-0's epoch-0 shuffled sequence order exactly."""
    data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    total_seqs = len(data) // SEQ_LEN
    seqs_per_rank = total_seqs // WORLD
    rng = np.random.RandomState(SEED + 0)               # seed + epoch(0)
    idx = np.arange(0, seqs_per_rank)
    rng.shuffle(idx)
    logger.info("train.bin: %d seqs total, %d/rank; permutation built", total_seqs, seqs_per_rank)
    return idx


def consumed_batch(train_bin: str, perm: np.ndarray, step: int, n_seqs: int) -> PackedSample:
    """The exact n_seqs rank-0 consumed starting at `step` (doc-masked, like training)."""
    data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    start = step * SEQS_PER_STEP
    seq_ids = perm[start:start + n_seqs]
    samples = []
    for sid in seq_ids:
        raw = np.asarray(data[sid * SEQ_LEN:(sid + 1) * SEQ_LEN])
        cu, pos, mx = compute_doc_boundaries(raw, eos_id=EOS_ID)
        samples.append(PackedSample(
            input_ids=torch.from_numpy(raw.astype(np.int64)),
            cu_seqlens=torch.from_numpy(cu), position_ids=torch.from_numpy(pos).long(), max_seqlen=mx))
    return collate_packed(samples)


def build_bf16_model(ckpt_model_state: dict, device: str, use_liger: bool = False) -> LuxiaBaseModel:
    config = LuxiaModelConfig(**{**CONFIG_3B, "use_liger": use_liger})
    model = LuxiaBaseModel(config)
    model.load_state_dict(ckpt_model_state, strict=True)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def to_fp8(model: LuxiaBaseModel) -> LuxiaBaseModel:
    """Convert Linears to Float8Linear, matching training (default dynamic tensorwise)."""
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    convert_to_float8_training(model, config=Float8LinearConfig())
    return model


@torch.no_grad()
def loss_on(model: LuxiaBaseModel, batch: dict[str, Any], device: str, doc_mask: bool) -> float:
    ids = batch["input_ids"].to(device)
    kw: dict[str, Any] = {"labels": ids}
    if doc_mask:
        kw["cu_seqlens"] = batch["cu_seqlens"].to(device)
        kw["position_ids"] = batch["position_ids"].to(device)
        kw["max_seqlen"] = batch["max_seqlen"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids, **kw)
    return float(out["loss"].item())


@torch.no_grad()
def activation_absmax(model: LuxiaBaseModel, batch: dict[str, Any], device: str) -> dict[str, float]:
    """Per-Linear input-activation absmax (the fp8 tensorwise-scale driver)."""
    stats: dict[str, float] = {}
    hooks = []
    def mk(name):
        def hook(_m, inp, _o):
            x = inp[0]
            stats[name] = max(stats.get(name, 0.0), float(x.detach().abs().max().item()))
        return hook
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(mk(name)))
    try:
        loss_on(model, batch, device, doc_mask=True)
    finally:
        for h in hooks:
            h.remove()
    return stats


def load_checkpoint(path: str) -> dict:
    p = Path(path)
    if p.suffix == ".zst" or p.name.endswith(".pt.zst"):
        import zstandard as zstd
        logger.info("decompressing %s ...", p.name)
        with open(p, "rb") as f:
            raw = zstd.ZstdDecompressor().stream_reader(f).read()
        return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    return torch.load(p, map_location="cpu", weights_only=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--steps", default="155000,195000", help="two step indices: data axis A,B")
    ap.add_argument("--n-seqs", type=int, default=32)
    ap.add_argument("--variants", default="bf16,fp8", help="subset of bf16,fp8,fp8c")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/loss_investigation.json")
    ap.add_argument("--dry-run", action="store_true", help="CPU plumbing check: reconstruction + shapes only")
    args = ap.parse_args()

    step_a, step_b = (int(s) for s in args.steps.split(","))
    perm = rank0_permutation(args.train_bin)

    # Reconstruct the two streaming batches (CPU, cheap)
    t0 = time.time()
    batch_a = consumed_batch(args.train_bin, perm, step_a, args.n_seqs)
    batch_b = consumed_batch(args.train_bin, perm, step_b, args.n_seqs)
    logger.info("reconstructed consumed batches @%d,%d in %.1fs (shape %s)",
                step_a, step_b, time.time() - t0, tuple(batch_a["input_ids"].shape))

    if args.dry_run:
        # Validate the custom plumbing without a GPU forward.
        from torchao.float8 import convert_to_float8_training  # import check
        tiny = LuxiaModelConfig(**{**CONFIG_3B, "num_layers": 2, "hidden_size": 256, "intermediate_size": 512,
                                   "num_attention_heads": 4, "num_kv_heads": 2, "attn_res_boundaries": [0, 1]})
        m = LuxiaBaseModel(tiny)
        n_lin = sum(1 for _, mod in m.named_modules() if isinstance(mod, torch.nn.Linear))
        logger.info("DRY-RUN OK: tiny model %d Linears; batch docs=%d; torchao import OK",
                    n_lin, len(batch_a["cu_seqlens"]) - 1)
        logger.info("DRY-RUN: batch_a step %d first ids %s", step_a, batch_a["input_ids"][0, :8].tolist())
        return

    ckpt = load_checkpoint(args.checkpoint)
    model_state = ckpt.get("model", ckpt)
    probe = ckpt.get("probe_batch")
    probe_batch = None
    if probe is not None:
        pb = probe[:args.n_seqs]
        probe_batch = {"input_ids": pb if pb.dim() == 2 else pb.unsqueeze(0)}
    logger.info("checkpoint loaded (step field=%s); probe_batch=%s",
                ckpt.get("step"), None if probe_batch is None else tuple(probe_batch["input_ids"].shape))

    variants = args.variants.split(",")
    results: dict[str, Any] = {"checkpoint": args.checkpoint, "steps": [step_a, step_b],
                               "n_seqs": args.n_seqs, "losses": {}, "act_absmax": {}}

    def run_variant(tag: str) -> None:
        logger.info("=== variant %s ===", tag)
        # trainflce: reproduce the TRAINING loss path — train mode + fused Liger FLCE
        # (isolates the fused-CE/train-mode head from the eval/materialized-CE path).
        use_liger = tag in ("trainflce", "trainflce_fp8")
        model = build_bf16_model(model_state, args.device, use_liger=use_liger)
        if tag in ("fp8", "fp8c", "trainflce_fp8"):
            model = to_fp8(model)
        if tag == "fp8c":
            model = torch.compile(model)
        if use_liger:
            model.train()   # gates self.fused_linear_ce_loss in forward()
        row = {
            f"stream@{step_a}": loss_on(model, batch_a, args.device, doc_mask=True),
            f"stream@{step_b}": loss_on(model, batch_b, args.device, doc_mask=True),
        }
        if probe_batch is not None:
            row["probe"] = loss_on(model, probe_batch, args.device, doc_mask=False)
        results["losses"][tag] = row
        logger.info("%s losses: %s", tag, {k: round(v, 4) for k, v in row.items()})
        # activation outliers (bf16 only, streaming vs probe)
        if tag == "bf16":
            results["act_absmax"][f"stream@{step_b}"] = activation_absmax(model, batch_b, args.device)
            if probe_batch is not None:
                pb = {**probe_batch}  # probe has no doc mask; reuse hook via doc_mask path w/o cu
                results["act_absmax"]["probe"] = _act_absmax_nomask(model, probe_batch, args.device)
        del model
        torch.cuda.empty_cache()

    for v in variants:
        run_variant(v)

    # Verdict summary
    lj = results["losses"]
    if "bf16" in lj:
        d = lj["bf16"].get(f"stream@{step_b}", 0) - lj["bf16"].get(f"stream@{step_a}", 0)
        logger.info("DATA axis (bf16): stream@%d - stream@%d = %+.4f  (≈0 ⇒ data stationary)", step_b, step_a, d)
    if "bf16" in lj and "fp8" in lj:
        n = lj["fp8"].get(f"stream@{step_b}", 0) - lj["bf16"].get(f"stream@{step_b}", 0)
        logger.info("NUMERICS axis (fp8-bf16 on stream@%d): %+.4f  (large ⇒ fp8 quant drives the rise)", step_b, n)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.out)


@torch.no_grad()
def _act_absmax_nomask(model: LuxiaBaseModel, batch: dict[str, Any], device: str) -> dict[str, float]:
    stats: dict[str, float] = {}
    hooks = []
    def mk(name):
        def hook(_m, inp, _o):
            stats[name] = max(stats.get(name, 0.0), float(inp[0].detach().abs().max().item()))
        return hook
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(mk(name)))
    try:
        ids = batch["input_ids"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(ids, labels=ids)
    finally:
        for h in hooks:
            h.remove()
    return stats


if __name__ == "__main__":
    main()
