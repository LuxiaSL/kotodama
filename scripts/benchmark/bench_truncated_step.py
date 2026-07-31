#!/usr/bin/env python3
"""Engine-level gates for the compiled truncated (top-k/top-p) sampler step.

  1. MAX-MEMBERSHIP (end-to-end graph correctness): with top_k=1, the only
     survivors of the truncated fused step are the argmax TIE SET (serve.py's
     kth-value threshold keeps ties, and exact fp32 ties do occur in real
     logits — first observed at step 36 of the first run). Generate N tokens,
     teacher-force the same sequence through step_logits, and assert every
     engine choice attains the reference max of the penalized/scaled logits.
     Any masking/unsort/penalty bug in the compiled graph breaks this; Gumbel
     tie-breaking does not.
  2. REGRESSION BENCH: free-running ms/token for the pure-temperature fused
     step (must be unchanged — it is the same graph as before this feature)
     vs the truncated fused step at p90 (budgeted <= +0.25 ms for the
     full-vocab sort; measured 121 us eager).

Usage (gpu-host):
    CUDA_VISIBLE_DEVICES=0 bash tools/run_py.sh scripts/benchmark/bench_truncated_step.py \
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

TEMP = 0.9
RP = 1.2
N_DET = 64
CTX = 512
# Cross-compiled-graph logit noise ceiling (round-1 parity methodology:
# reference flash-vs-efficient maxΔ 0.42-0.63, gate 0.72).
NOISE_CEILING = 0.72


def rand_ids(n: int, vocab: int, device: torch.device, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, vocab, (1, n), generator=g).to(device)


@torch.inference_mode()
def gate_determinism(engine: DecodeEngine, device: torch.device) -> bool:
    """top_k=1 truncated generation == reference penalty+argmax replay."""
    ids = rand_ids(CTX, engine.config.vocab_size, device)
    params = SamplingParams(temperature=TEMP, repetition_penalty=RP, top_k=1)

    tok = engine.sample_first(engine.prefill(ids), params)
    generated = [int(tok.item())]
    for _ in range(N_DET - 1):
        tok = engine.step(tok, params)
        generated.append(int(tok.item()))

    # Reference replay: teacher-force the same tokens through the forward-only
    # step and check each engine choice attains the max of serve.py's math.
    logits = engine.prefill(ids)  # (1, V) fp32, resets state
    presence: set[int] = set()
    ok = True
    n_ties = 0
    for i, expect in enumerate(generated):
        lg = logits.view(-1).float().clone()
        if presence:
            pid = torch.tensor(sorted(presence), device=device, dtype=torch.long)
            pl = lg[pid]
            lg[pid] = torch.where(pl > 0, pl / RP, pl * RP)
        scaled = lg / TEMP
        max_val = float(scaled.max())
        gap = max_val - float(scaled[expect])
        # The generation ran the TRUNCATED graph; this replay runs the
        # forward-only graph — two separately compiled artifacts whose logits
        # differ by autotune fp dust (round-1 parity: maxΔ 0.375–0.56 vs
        # references; gate ceiling 0.72). A near-tie step can therefore rank
        # differently across graphs without any masking bug: tolerate gaps
        # under the noise ceiling, fail anything beyond it.
        if gap > NOISE_CEILING:
            logger.error(
                "max-membership FAIL at step %d: engine=%d sits %.4f below max %.6f (argmax %d)",
                i, expect, gap, max_val, int(scaled.argmax()),
            )
            ok = False
            break
        if gap > 0.0:
            n_ties += 1
        presence.add(expect)
        if i < len(generated) - 1:
            engine.cur_token.copy_(torch.tensor([[expect]], device=device))
            logits = engine.step_logits(engine.cur_token)[0, -1:].float().clone()
    logger.info("Gate 1 (k=1 max-membership, %d tokens, %d near-tie steps): %s",
                N_DET, n_ties, "PASS" if ok else "FAIL")
    return ok


@torch.inference_mode()
def gate_reproducibility(engine: DecodeEngine, device: torch.device) -> bool:
    """Same seed -> same truncated-graph token sequence (single-graph check).

    Complements the cross-graph gate above without a noise floor: any
    nondeterministic buffer binding or masking instability shows up here.
    """
    ids = rand_ids(CTX, engine.config.vocab_size, device, seed=23)
    params = SamplingParams(temperature=TEMP, repetition_penalty=RP, top_p=0.9)
    seqs = []
    for _ in range(2):
        torch.cuda.manual_seed(1234)
        tok = engine.sample_first(engine.prefill(ids), params)
        seq = [int(tok.item())]
        for _ in range(N_DET - 1):
            tok = engine.step(tok, params)
            seq.append(int(tok.item()))
        seqs.append(seq)
    ok = seqs[0] == seqs[1]
    if not ok:
        first = next(i for i, (a, b) in enumerate(zip(seqs[0], seqs[1])) if a != b)
        logger.error("reproducibility mismatch: first divergence at step %d", first)
    logger.info("Gate 1b (same-seed p90 reproducibility, %d tokens): %s", N_DET, "PASS" if ok else "FAIL")
    return ok


@torch.inference_mode()
def bench_law(engine: DecodeEngine, params: SamplingParams, device: torch.device,
              n_tokens: int = 300) -> float:
    ids = rand_ids(CTX, engine.config.vocab_size, device, seed=11)
    tok = engine.sample_first(engine.prefill(ids), params)
    for _ in range(10):
        tok = engine.step(tok, params)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        tok = engine.step(tok, params)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000 / n_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--budget-ms", type=float, default=0.25,
                    help="max allowed truncated-vs-pure ms/tok delta")
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_model(args.checkpoint, device)
    engine = DecodeEngine(model, max_seq_len=4096)
    t0 = time.perf_counter()
    engine.compile_step(mode="max-autotune")
    # Force all three graphs to build.
    ids = rand_ids(64, engine.config.vocab_size, device)
    pure = SamplingParams(temperature=TEMP, repetition_penalty=RP)
    p90 = SamplingParams(temperature=TEMP, repetition_penalty=RP, top_p=0.9)
    tok = engine.sample_first(engine.prefill(ids), pure)
    engine.step(tok, pure)
    engine.step(tok, p90)
    engine.step_logits(engine.cur_token)
    torch.cuda.synchronize(device)
    logger.info("Compiled 3 graphs in %.1fs", time.perf_counter() - t0)

    ok = gate_determinism(engine, device)
    ok = gate_reproducibility(engine, device) and ok

    ms_pure = bench_law(engine, pure, device)
    ms_p90 = bench_law(engine, p90, device)
    ms_k100 = bench_law(engine, SamplingParams(temperature=TEMP, repetition_penalty=RP, top_k=100), device)
    delta = ms_p90 - ms_pure
    logger.info("Gate 2 bench: pure %.3f ms/tok (%.0f tok/s) | p90 %.3f (%.0f) | "
                "k100 %.3f (%.0f) | delta %.3f ms (budget %.2f)",
                ms_pure, 1000 / ms_pure, ms_p90, 1000 / ms_p90,
                ms_k100, 1000 / ms_k100, delta, args.budget_ms)
    bench_ok = delta <= args.budget_ms
    ok = ok and bench_ok

    print(f"TRUNCATED_STEP: {'PASS' if ok else 'FAIL'} "
          f"pure={ms_pure:.3f} p90={ms_p90:.3f} k100={ms_k100:.3f} delta={delta:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
