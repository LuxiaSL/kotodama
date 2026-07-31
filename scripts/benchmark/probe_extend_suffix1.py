#!/usr/bin/env python3
"""Diagnose the suffix-1 extend anomaly (measured 54ms vs ~2.5ms docstring claim).

Phase-times prefill_cached's suffix==1 path and contrasts the compiled step's
latency in three contexts:
  A. steady-state decode (replay after replay)          — expect ~2.5 ms
  B. first compiled step right after an eager prefill    — re-record suspect
  C. the full prefill_cached suffix-1 call, decomposed:
     _common_prefix_len (host sync) / bias+pos writes / step_logits

Run with TORCH_LOGS=cudagraphs to surface re-record/warmup reasons.

Usage (gpu-host):
    CUDA_VISIBLE_DEVICES=0 TORCH_LOGS=cudagraphs bash tools/run_py.sh \
        scripts/benchmark/probe_extend_suffix1.py \
        --checkpoint /models/kotodama-data/tmp/kotodama_checkpoints/3b-language-FINAL-step195311.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", "1")

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark.decode_parity import load_model  # noqa: E402
from src.model.decode_engine import DecodeEngine, SamplingParams  # noqa: E402

COMMON = 1024


def rand_ids(n: int, vocab: int, device: torch.device, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, vocab, (1, n), generator=g).to(device)


def timed(device: torch.device, fn, *args, **kwargs):
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize(device)
    return out, (time.perf_counter() - t0) * 1000


@torch.inference_mode()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")

    model = load_model(args.checkpoint, device)
    engine = DecodeEngine(model, max_seq_len=4096)
    engine.compile_step(mode="max-autotune")
    params = SamplingParams(temperature=0.9, repetition_penalty=1.2)
    base = rand_ids(COMMON + 1, engine.config.vocab_size, device)

    # Warm everything the serve.py warmup warms.
    tok = engine.sample_first(engine.prefill(base[:, :64]), params)
    for _ in range(4):
        tok = engine.step(tok, params)
    engine.step_logits(engine.cur_token)
    engine.prefill(base[:, :64])
    engine.prefill_cached(base[:, :65])   # multi-token extend warm... (48-token style)
    engine.prefill_cached(base[:, :65])   # suffix-1 warm (same shape twice)
    torch.cuda.synchronize(device)

    # A. steady-state compiled step
    engine.prefill(base[:, :COMMON])
    tok = engine.sample_first(engine.prefill(base[:, :COMMON]), params)
    for _ in range(5):
        tok = engine.step(tok, params)
    times = []
    for _ in range(10):
        _, ms = timed(device, engine.step_logits, engine.cur_token)
        times.append(ms)
    logger.info("A. steady-state step_logits: %s ms", [round(t, 2) for t in times])

    # B. compiled step immediately after an eager prefill, repeated
    for rep in range(3):
        engine.prefill(base[:, :COMMON])
        _, ms1 = timed(device, engine.step_logits, engine.cur_token)
        _, ms2 = timed(device, engine.step_logits, engine.cur_token)
        _, ms3 = timed(device, engine.step_logits, engine.cur_token)
        logger.info("B. rep%d step-after-prefill: 1st %.2f | 2nd %.2f | 3rd %.2f ms", rep, ms1, ms2, ms3)

    # C. prefill_cached suffix-1 decomposition
    for rep in range(3):
        engine.prefill(base[:, :COMMON])
        torch.cuda.synchronize(device)
        ids = base[:, : COMMON + 1]

        t0 = time.perf_counter()
        common = engine._common_prefix_len(ids)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        common = min(common, ids.shape[1] - 1)
        suffix = ids[:, common:].to(device)
        engine.presence.zero_()
        engine.attn_bias.fill_(float("-inf"))
        engine.attn_bias[..., :common] = 0.0
        engine.pos.fill_(common)
        engine._pos_cpu = common
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()

        logits = engine.step_logits(suffix.contiguous())
        torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        del logits
        logger.info(
            "C. rep%d suffix-1 decomposed: match %.2f | state-writes %.2f | step %.2f | total %.2f ms",
            rep, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t3 - t0) * 1e3,
        )

    # D. same, but feed the step through the static cur_token buffer instead
    # of a fresh contiguous slice (candidate fix).
    for rep in range(3):
        engine.prefill(base[:, :COMMON])
        torch.cuda.synchronize(device)
        ids = base[:, : COMMON + 1]
        common = min(engine._common_prefix_len(ids), ids.shape[1] - 1)
        engine.presence.zero_()
        engine.attn_bias.fill_(float("-inf"))
        engine.attn_bias[..., :common] = 0.0
        engine.pos.fill_(common)
        engine._pos_cpu = common
        engine.cur_token.copy_(ids[:, common:])
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        logits = engine.step_logits(engine.cur_token)
        torch.cuda.synchronize(device)
        logger.info("D. rep%d step via static cur_token: %.2f ms", rep, (time.perf_counter() - t0) * 1e3)
        del logits

    print("PROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
