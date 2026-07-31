#!/usr/bin/env python3
"""Push-button validation battery for DecodeEngine changes.

Chains, as subprocesses (each leg needs its own process: the reference
routing implementation is chosen by KOTODAMA_NO_TRITON_ATTNRES at llama.py
import time):

  1. decode_parity.py vs eager-routing reference   (KOTODAMA_NO_TRITON_ATTNRES=1)
  2. decode_parity.py vs Triton-routing reference  (fleet config)      [skipped by --quick]
  3. bench_engine.py at ctx {512, 2048}                                [skipped by --skip-bench]

Gates (derived from the round-1 measured noise floor, parity_noise_floor.py:
reference flash-vs-efficient maxΔ 0.42-0.63, top32 ~0.94):

  * teacher-forced argmax agreement >= 0.99       (every prompt, every leg)
  * teacher-forced max|Δlogit|      <= 0.72       (noise ceiling 0.63 + headroom)
  * top-32 overlap                  >= 0.88       (gross-breakage detector only:
        known-good builds measure 0.90-0.97 min depending on kernel selection —
        tail-logit order is backend noise; a real bug craters this to <0.5)
  * free greedy: divergence only at near-ties     (ref top1-top2 margin < 1.0)
  * optional --min-tok-s floor on sampler tok/s @ ctx 512

Prints one final line `VALIDATE_SUMMARY: {...}` for machine parsing; writes
full JSON via --output. Exit 0 = PASS, 1 = gate FAIL, 2 = harness error.

Canonical invocation (gpu-host):
    source ~/workspace/.venv-shared/bin/activate
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark/validate_engine.py \
        --checkpoint /models/kotodama-data/tmp/kotodama_checkpoints/3b-language-FINAL-step195311.pt
Inductor cache/CD-tuning env is set automatically if absent. Engine flags
(e.g. KOTODAMA_CUSTOM_GEMV) are inherited by every leg — set them at the
battery invocation to validate a variant.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[1]

# ── Gates (see module docstring for provenance) ──────────────────────────────
GATE_ARGMAX_MIN = 0.99
GATE_MAX_DELTA = 0.72
GATE_TOP32_MIN = 0.88
GATE_GREEDY_NEAR_TIE_MARGIN = 1.0

# GPU host defaults — harmless elsewhere (dir is created if missing).
DEFAULT_INDUCTOR_CACHE = "/models/kotodama-data/tmp/inductor_cache"
LEG_TIMEOUT_S = 2400


def _base_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", DEFAULT_INDUCTOR_CACHE)
    env.setdefault("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING", "1")
    return env


def _run_leg(name: str, cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    logger.info("[%s] %s", name, " ".join(cmd))
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=str(REPO_ROOT), timeout=LEG_TIMEOUT_S,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except subprocess.TimeoutExpired:
        logger.error("[%s] TIMEOUT after %ds", name, LEG_TIMEOUT_S)
        return 124, ""
    dur = time.time() - t0
    logger.info("[%s] exit=%d in %.0fs", name, proc.returncode, dur)
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        logger.error("[%s] tail:\n%s", name, tail)
    return proc.returncode, proc.stdout


def _gate_parity(leg_name: str, parity: dict[str, Any]) -> list[str]:
    """Return a list of human-readable gate failures (empty = pass)."""
    failures: list[str] = []
    for plen, res in parity.get("prompts", {}).items():
        tf = res["teacher_forced"]
        if tf["argmax_agreement"] < GATE_ARGMAX_MIN:
            failures.append(
                f"{leg_name} plen={plen}: argmax {tf['argmax_agreement']} < {GATE_ARGMAX_MIN}"
            )
        if tf["max_abs_logit_delta"] > GATE_MAX_DELTA:
            failures.append(
                f"{leg_name} plen={plen}: maxDelta {tf['max_abs_logit_delta']} > {GATE_MAX_DELTA}"
            )
        if tf["min_top32_overlap"] < GATE_TOP32_MIN:
            failures.append(
                f"{leg_name} plen={plen}: top32 {tf['min_top32_overlap']} < {GATE_TOP32_MIN}"
            )
        div = res.get("greedy_first_divergence", {})
        if div.get("step") is not None:
            margin = div.get("ref_top1_top2_margin")
            if margin is None or margin >= GATE_GREEDY_NEAR_TIE_MARGIN:
                failures.append(
                    f"{leg_name} plen={plen}: greedy diverged @step {div['step']} "
                    f"with margin {margin} (>= {GATE_GREEDY_NEAR_TIE_MARGIN}, not a near-tie)"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--quick", action="store_true",
                        help="eager-ref leg only, fewer prompts/steps (iteration loop)")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--min-tok-s", type=float, default=None,
                        help="fail if sampler tok/s @ctx512 falls below this")
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--output", default=None, help="full JSON report path")
    args = parser.parse_args()

    base_env = _base_env()
    Path(base_env["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    py = sys.executable
    tmpdir = Path(tempfile.mkdtemp(prefix="validate_engine_"))
    report: dict[str, Any] = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "compile_mode": args.compile_mode,
        "quick": args.quick,
        "engine_flags": {k: v for k, v in os.environ.items() if k.startswith("KOTODAMA_")},
        "legs": {},
    }
    failures: list[str] = []
    harness_errors: list[str] = []

    parity_args = ["--compile", "--compile-mode", args.compile_mode,
                   "--checkpoint", args.checkpoint]
    if args.quick:
        parity_args += ["--prompt-lens", "16", "256", "--steps", "32"]

    # Leg 1: eager-routing reference (selfsim/distill server parity).
    legs: list[tuple[str, dict[str, str]]] = [
        ("parity_vs_eager_ref", {**base_env, "KOTODAMA_NO_TRITON_ATTNRES": "1"}),
    ]
    # Leg 2: Triton-routing reference (fleet parity).
    if not args.quick:
        env_triton = {k: v for k, v in base_env.items() if k != "KOTODAMA_NO_TRITON_ATTNRES"}
        legs.append(("parity_vs_triton_ref", env_triton))

    for leg_name, env in legs:
        out_json = tmpdir / f"{leg_name}.json"
        rc, _ = _run_leg(
            leg_name,
            [py, str(BENCH_DIR / "decode_parity.py"), *parity_args, "--output", str(out_json)],
            env,
        )
        if rc != 0 or not out_json.exists():
            harness_errors.append(f"{leg_name}: leg failed to produce output (exit {rc})")
            continue
        parity = json.loads(out_json.read_text())
        report["legs"][leg_name] = parity
        failures.extend(_gate_parity(leg_name, parity))

    # Leg 3: block prefill/extend battery (2026-07-05 paths) — its own gates
    # (control-calibrated KV screen + teacher-forced 0.72 trace + near-tie
    # handoffs); decode_parity above deliberately pins prefill to reference.
    if not args.quick:
        rc, tail = _run_leg(
            "block_prefill",
            [py, str(BENCH_DIR / "bench_block_prefill.py"), "--checkpoint", args.checkpoint],
            base_env,
        )
        block_pass = rc == 0 and "BLOCK_PREFILL: PASS" in tail
        report["legs"]["block_prefill"] = {"pass": block_pass, "exit": rc}
        if not block_pass:
            failures.append(f"block_prefill: battery reported FAIL (exit {rc})")

    # Leg 4: throughput.
    if not args.skip_bench:
        out_json = tmpdir / "bench.json"
        rc, _ = _run_leg(
            "bench",
            [py, str(BENCH_DIR / "bench_engine.py"),
             "--checkpoint", args.checkpoint,
             "--compile", "--compile-mode", args.compile_mode,
             "--context-lens", "512", "2048",
             "--output", str(out_json)],
            base_env,
        )
        if rc != 0 or not out_json.exists():
            harness_errors.append(f"bench: leg failed to produce output (exit {rc})")
        else:
            bench = json.loads(out_json.read_text())
            report["legs"]["bench"] = bench
            tok_s: Optional[float] = None
            for row in bench.get("rows", []):
                if row["ctx"] == 512:
                    tok_s = row["sampler_tok_s"]
            report["sampler_tok_s_512"] = tok_s
            if args.min_tok_s is not None and (tok_s is None or tok_s < args.min_tok_s):
                failures.append(f"bench: sampler tok/s @512 = {tok_s} < floor {args.min_tok_s}")

    report["gate_failures"] = failures
    report["harness_errors"] = harness_errors
    verdict = "PASS" if not failures and not harness_errors else "FAIL"
    report["verdict"] = verdict

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        logger.info("Full report: %s", out)

    for f in failures + harness_errors:
        logger.error("GATE: %s", f)
    summary = {
        "verdict": verdict,
        "label": args.label,
        "tok_s_512": report.get("sampler_tok_s_512"),
        "max_delta": max(
            (leg["prompts"][p]["teacher_forced"]["max_abs_logit_delta"]
             for ln, leg in report["legs"].items() if ln.startswith("parity")
             for p in leg.get("prompts", {})),
            default=None,
        ),
        "n_failures": len(failures) + len(harness_errors),
    }
    print(f"VALIDATE_SUMMARY: {json.dumps(summary)}")
    return 0 if verdict == "PASS" else (2 if harness_errors and not failures else 1)


if __name__ == "__main__":
    sys.exit(main())
