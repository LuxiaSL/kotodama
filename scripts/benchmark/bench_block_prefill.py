#!/usr/bin/env python3
"""Gates + bench for the compiled block-forward prefill/extend.

  1. PARITY: block prefill vs reference eager prefill at several lengths
     (incl. overlap-remainder plans): last-position logits within the
     cross-graph noise ceiling (argmax agreement or near-tie), KV caches
     allclose in bf16, token_history/pos/bias state identical.
  2. EXTEND PARITY: prefill_cached (block path) vs the eager MATH extend on
     the same (common, suffix): logits + appended-KV agreement.
  3. DECODE HANDOFF: block prefill -> 32 greedy compiled decode steps vs
     reference prefill -> same steps: token sequences must agree modulo
     near-tie divergence (free-greedy methodology from decode_parity).
  4. BENCH: prefill wall at T in {128, 512, 1024, 3000} and extend wall at
     (1024, +16/+64), block vs reference paths.

Usage (gpu-host):
    CUDA_VISIBLE_DEVICES=0 bash tools/run_py.sh scripts/benchmark/bench_block_prefill.py \
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
from src.model.decode_engine import DecodeEngine  # noqa: E402

NOISE_CEILING = 0.72     # round-1 parity gate (cross-kernel logit noise)
# KV gate is SELF-CALIBRATING against a same-run control, because this trunk
# amplifies ANY kernel-schedule perturbation into multi-unit deep-layer KV
# deltas (measured 2026-07-05: the two ESTABLISHED references — llama with
# Triton vs eager routing — differ from each other by worst-rel up to 9.2 at
# T=16, MORE than blocks differ from either; logit maxΔ R1-R2 0.297 vs
# BLK-ref 0.14-0.26, all under the 0.72 gate). Fixed elementwise tolerances
# across path families are meaningless here; the gate asks instead: are
# blocks FARTHER from the references than the references are from each
# other? Structural bugs (rope offset, unwritten rows) are 10-100x the
# control; dust families are ~1x.
# The KV screen is a LOOSE structural detector only (unwritten rows / rope
# offsets = 10-100x control; measured multi-chunk dust compounding tops out
# at ~3.7x control at T=512 because chunk N amplifies chunk N-1's dust).
# The TIGHT gate is Gate 5: teacher-forced logit parity — the round-1
# standard, position-by-position, at the 0.72 noise ceiling.
KV_CONTROL_FACTOR = 6.0
KV_REL_FLOOR = 0.02      # mean-statistic floor: dust means ~1e-3, structural ~1
PREFILL_LENS = [16, 128, 273, 512, 1024, 3000]   # 273 exercises overlap
EXTEND_CASES = [(1024, 16), (1024, 64), (1024, 47)]  # 47 exercises overlap


def rand_ids(n: int, vocab: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, vocab, (1, n), generator=g).to(device)


def logits_agree(a: torch.Tensor, b: torch.Tensor, tag: str) -> bool:
    """Argmax agreement, or near-tie within the noise ceiling."""
    ia, ib = int(a.argmax()), int(b.argmax())
    if ia == ib:
        return True
    gap = float(b.max() - b.view(-1)[ia])
    if gap <= NOISE_CEILING:
        logger.info("  %s: argmax differs at near-tie (gap %.4f) — tolerated", tag, gap)
        return True
    logger.error("  %s: argmax %d vs %d, gap %.4f > %.2f", tag, ia, ib, gap, NOISE_CEILING)
    return False


@torch.inference_mode()
def snapshot_kv(engine: DecodeEngine, upto: int) -> list[torch.Tensor]:
    return [torch.stack([engine.k_caches[li][:, :, :upto].clone(),
                         engine.v_caches[li][:, :, :upto].clone()])
            for li in range(engine.n_layers)]


def rel_worst(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    """Worst PER-POSITION MEAN relative delta across layers/k/v.

    The house gate-design lesson (round-2 prefix cache): never gate the
    global max — the extreme tail is chaotic under KV perturbation and its
    max-statistic grows with sample count. A structural bug (unwritten row,
    rope offset) makes an ENTIRE position's row wrong -> per-position mean
    jumps ~100x dust; amplified-dust tails barely move a 57k-element mean.
    """
    worst = 0.0
    for x, y in zip(a, b):
        rel = (x - y).abs() / (y.abs() + 0.5)   # (2-in-stack..., KV, n, dh) or (1, KV, n, dh)
        per_pos = rel.mean(dim=(0, 1, 3)) if rel.ndim == 4 else rel.mean(dim=(0, 1, 2, 4))
        worst = max(worst, float(per_pos.max()))
    return worst


def kv_agree(a: list[torch.Tensor], b: list[torch.Tensor], control: float, tag: str) -> bool:
    worst = rel_worst(a, b)
    gate = max(KV_CONTROL_FACTOR * control, KV_REL_FLOOR)
    ok = worst <= gate
    (logger.info if ok else logger.error)(
        "  %s: worst per-pos mean-rel KV delta %.4f (gate %.4f = max(%.1fx control %.4f, %.2f)) %s",
        tag, worst, gate, KV_CONTROL_FACTOR, control, KV_REL_FLOOR, "PASS" if ok else "FAIL")
    return ok


@torch.inference_mode()
def reference_control(engine: DecodeEngine, ids: torch.Tensor) -> float:
    """Same-length ref-vs-ref KV divergence: llama Triton vs eager routing.

    This is the noise floor the trunk itself produces between two correct
    paths; the block gate is calibrated against it.
    """
    import src.model.llama as llama_mod
    n = ids.shape[1]
    engine.prefill_reference(ids)
    r1 = snapshot_kv(engine, n)
    had_triton = llama_mod._TRITON_ATTN_RES_AVAILABLE
    llama_mod._TRITON_ATTN_RES_AVAILABLE = False
    try:
        engine.prefill_reference(ids)
    finally:
        llama_mod._TRITON_ATTN_RES_AVAILABLE = had_triton
    r2 = snapshot_kv(engine, n)
    return max(rel_worst(r1, r2), rel_worst(r2, r1))


@torch.inference_mode()
def gate_prefill_parity(engine: DecodeEngine, device: torch.device) -> bool:
    ok = True
    for n in PREFILL_LENS:
        ids = rand_ids(n, engine.config.vocab_size, device, seed=1000 + n)
        control = reference_control(engine, ids)
        ref_logits = engine.prefill_reference(ids)
        ref_kv = snapshot_kv(engine, n)
        ref_hist = engine.token_history[:n].clone()

        blk_logits = engine.prefill(ids)
        used_blocks = (
            engine._chunk_plan(0, n) is not None and n <= engine.block_span_max
        )
        blk_kv = snapshot_kv(engine, n)

        tag = f"prefill T={n} ({'blocks' if used_blocks else 'reference-fallback'})"
        ok &= logits_agree(blk_logits.view(-1), ref_logits.view(-1), tag)
        ok &= kv_agree(blk_kv, ref_kv, control, tag)
        if not torch.equal(engine.token_history[:n], ref_hist):
            logger.error("  %s: token_history mismatch", tag)
            ok = False
        if engine._pos_cpu != n:
            logger.error("  %s: pos %d != %d", tag, engine._pos_cpu, n)
            ok = False
        opened = float(engine.attn_bias[..., :n].max()) == 0.0 and \
            float(engine.attn_bias[..., n:].max()) == float("-inf")
        if not opened:
            logger.error("  %s: attn_bias state wrong", tag)
            ok = False
    logger.info("Gate 1 (block prefill parity, %d lengths): %s", len(PREFILL_LENS), "PASS" if ok else "FAIL")
    return ok


@torch.inference_mode()
def gate_extend_parity(engine: DecodeEngine, device: torch.device) -> bool:
    ok = True
    saved_block = engine._compiled_block
    for common, suf in EXTEND_CASES:
        ids = rand_ids(common + suf, engine.config.vocab_size, device, seed=2000 + suf)
        control = reference_control(engine, ids)

        # Eager MATH extend (blocks disabled for the reference leg).
        engine._compiled_block = None
        engine.prefill_reference(ids[:, :common])
        ref_logits, ref_info = engine.prefill_cached(ids)
        assert ref_info["prefix_hit"], ref_info
        ref_kv = snapshot_kv(engine, common + suf)

        # Block extend.
        engine._compiled_block = saved_block
        engine.prefill_reference(ids[:, :common])
        blk_logits, blk_info = engine.prefill_cached(ids)
        assert blk_info["prefix_hit"], blk_info
        blk_kv = snapshot_kv(engine, common + suf)

        tag = f"extend common={common} suffix={suf}"
        ok &= logits_agree(blk_logits.view(-1), ref_logits.view(-1), tag)
        ok &= kv_agree(blk_kv, ref_kv, control, tag)
    logger.info("Gate 2 (block extend parity, %d cases): %s", len(EXTEND_CASES), "PASS" if ok else "FAIL")
    return ok


def _greedy_walk(engine: DecodeEngine, first_logits: torch.Tensor, n_steps: int) -> tuple[list[int], list[float]]:
    """Greedy decode; returns (tokens, per-step top1-top2 margins)."""
    lg = first_logits.view(-1).float()
    top2 = lg.topk(2).values
    tok = lg.argmax().view(1, 1)
    seq, margins = [int(tok.item())], [float(top2[0] - top2[1])]
    for _ in range(n_steps - 1):
        engine.cur_token.copy_(tok)
        logits = engine.step_logits(engine.cur_token)[0, -1].float()
        top2 = logits.topk(2).values
        tok = logits.argmax().view(1, 1)
        seq.append(int(tok.item()))
        margins.append(float(top2[0] - top2[1]))
    return seq, margins


def _judge_handoff(tag: str, ref: tuple[list[int], list[float]], blk: tuple[list[int], list[float]], n: int) -> bool:
    """Free-greedy methodology (decode_parity): sequences must match up to the
    first divergence, which is acceptable ONLY at a near-tie (reference
    top1-top2 margin < 1.0 at that step). Cross-path fp dust legitimately
    flips near-ties; a structural bug diverges at a decisive step."""
    if ref[0] == blk[0]:
        logger.info("%s: %d/%d tokens identical", tag, n, n)
        return True
    div = next(i for i, (a, b) in enumerate(zip(ref[0], blk[0])) if a != b)
    margin = ref[1][div]
    ok = margin < 1.0
    (logger.info if ok else logger.error)(
        "%s: diverged at step %d/%d with ref margin %.3f — %s",
        tag, div, n, margin, "near-tie, acceptable" if ok else "DECISIVE step = structural")
    return ok


@torch.inference_mode()
def gate_decode_handoff(engine: DecodeEngine, device: torch.device) -> bool:
    """Block prefill -> greedy decode vs reference prefill -> greedy decode."""
    ids = rand_ids(512, engine.config.vocab_size, device, seed=77)
    ref = _greedy_walk(engine, engine.prefill_reference(ids), 32)
    blk = _greedy_walk(engine, engine.prefill(ids), 32)
    ok = _judge_handoff("  decode handoff", ref, blk, 32)
    logger.info("Gate 3 (decode handoff, 32 greedy steps): %s", "PASS" if ok else "FAIL")
    return ok


@torch.inference_mode()
def gate_extend_handoff(engine: DecodeEngine, device: torch.device) -> bool:
    """Block EXTEND -> greedy decode vs eager extend -> greedy decode.

    The deployment surface for blocks is prefix-cache extends; this gates the
    full path: cached common prefix + block-chunked suffix + compiled decode.
    """
    common, suf = 1024, 64
    ids = rand_ids(common + suf, engine.config.vocab_size, device, seed=88)
    saved_block = engine._compiled_block
    walks = []
    for use_blocks in (False, True):
        engine._compiled_block = saved_block if use_blocks else None
        engine.prefill_reference(ids[:, :common])
        logits, info = engine.prefill_cached(ids)
        assert info["prefix_hit"], info
        walks.append(_greedy_walk(engine, logits, 16))
    engine._compiled_block = saved_block
    ok = _judge_handoff("  extend handoff", walks[0], walks[1], 16)
    logger.info("Gate 4 (extend handoff, 16 greedy steps): %s", "PASS" if ok else "FAIL")
    return ok


@torch.inference_mode()
def gate_forced_trace(engine: DecodeEngine, device: torch.device) -> bool:
    """Teacher-forced logit parity after block prefill vs reference prefill.

    The round-1 methodology, position-by-position: force the SAME 32 tokens
    through the compiled decode step from each prefill's cache; per-step
    max|Δlogit| must stay under the 0.72 cross-kernel noise ceiling. A
    structural defect in ANY block-written cache row perturbs every
    subsequent step's logits far beyond noise.
    """
    ok = True
    for n in (128, 512):  # multi-chunk plans — the contested surface
        ids = rand_ids(n, engine.config.vocab_size, device, seed=6000 + n)
        forced = rand_ids(32, engine.config.vocab_size, device, seed=6100 + n)
        traces = []
        for use_blocks in (False, True):
            if use_blocks:
                engine.prefill(ids)
            else:
                engine.prefill_reference(ids)
            step_logits = []
            for i in range(32):
                engine.cur_token.copy_(forced[:, i: i + 1])
                step_logits.append(engine.step_logits(engine.cur_token)[0, -1].float().clone())
            traces.append(step_logits)
        worst = max(float((a - b).abs().max()) for a, b in zip(*traces))
        line_ok = worst <= NOISE_CEILING
        (logger.info if line_ok else logger.error)(
            "  forced trace T=%d: worst per-step max|Δlogit| %.4f (gate %.2f) %s",
            n, worst, NOISE_CEILING, "PASS" if line_ok else "FAIL")
        ok = ok and line_ok
    logger.info("Gate 5 (teacher-forced logit parity, 32 steps x 2 lengths): %s", "PASS" if ok else "FAIL")
    return ok


@torch.inference_mode()
def bench(engine: DecodeEngine, device: torch.device) -> None:
    def timed(fn) -> float:
        times = []
        for _ in range(3):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000)
        return sorted(times)[1]

    for n in [128, 512, 1024, 3000]:
        ids = rand_ids(n, engine.config.vocab_size, device, seed=3000 + n)
        engine.prefill(ids)            # warm shape
        blk = timed(lambda: engine.prefill(ids))
        engine.prefill_reference(ids)
        ref = timed(lambda: engine.prefill_reference(ids))
        logger.info("BENCH prefill T=%d: blocks %.1f ms | reference %.1f ms (%.1fx)", n, blk, ref, ref / blk)

    for common, suf in [(1024, 16), (1024, 64)]:
        ids = rand_ids(common + suf, engine.config.vocab_size, device, seed=4000 + suf)

        def run_extend():
            engine.prefill(ids[:, :common])
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            engine.prefill_cached(ids)
            torch.cuda.synchronize(device)
            return (time.perf_counter() - t0) * 1000

        run_extend()  # warm
        ext = sorted(run_extend() for _ in range(3))[1]
        logger.info("BENCH extend common=%d suffix=%d: blocks %.1f ms", common, suf, ext)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")

    model = load_model(args.checkpoint, device)
    engine = DecodeEngine(model, max_seq_len=4096)
    t0 = time.perf_counter()
    engine.compile_step(mode="max-autotune")
    engine.warm_blocks()
    ids = rand_ids(64, engine.config.vocab_size, device, seed=5)
    engine.prefill(ids)
    engine.cur_token.copy_(torch.tensor([[100]], device=device))
    engine.step_logits(engine.cur_token)
    torch.cuda.synchronize(device)
    logger.info("Compiled + warmed in %.1fs", time.perf_counter() - t0)

    ok = gate_prefill_parity(engine, device)
    ok = gate_extend_parity(engine, device) and ok
    ok = gate_decode_handoff(engine, device) and ok
    ok = gate_extend_handoff(engine, device) and ok
    ok = gate_forced_trace(engine, device) and ok
    bench(engine, device)
    print(f"BLOCK_PREFILL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
