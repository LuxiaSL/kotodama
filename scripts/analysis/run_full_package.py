"""Run all analysis tracks on a single checkpoint, producing a self-contained package.

Usage:
    cd pretraining
    python -m scripts.analysis.run_full_package \
        --checkpoint checkpoints/fullcorpus-ddv1/step_00019200.pt.zst \
        --name fullcorpus-Q1 \
        --metrics data/fullcorpus_ddv1_metrics.jsonl \
        --geo-metrics data/fullcorpus_ddv1_geo_metrics.jsonl

    # Skip heavy tracks for quick iteration:
    python -m scripts.analysis.run_full_package \
        --checkpoint ... --name ... --metrics ... \
        --skip activation_geometry,concept_geometry
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_TRACKS = [
    "training_analysis",
    "generation",
    "text_quality",
    "activation_geometry",
    "concept_geometry",
    "lm_eval",
    "report",
]

DEFAULT_ATTN_RES = "boundaries=0,3,7,12,21,25"
DEFAULT_LM_EVAL_TASKS = (
    "hellaswag,piqa,arc_easy,boolq,lambada_openai,winogrande,wikitext,copa,sciq"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all analysis tracks on a checkpoint"
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path (.pt or .pt.zst)")
    parser.add_argument("--name", required=True, help="Package name (e.g., fullcorpus-Q1)")
    parser.add_argument("--metrics", required=True, help="Training metrics JSONL")
    parser.add_argument("--geo-metrics", default=None, help="Geometric metrics JSONL (optional, merged with --metrics)")
    parser.add_argument("--output-dir", default="analysis/packages", help="Parent dir for packages")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--config-section", default="proxy")
    parser.add_argument("--attn-res", default=DEFAULT_ATTN_RES)
    parser.add_argument("--eval-data", default="data/fineweb_edu_eval_5m.bin")
    parser.add_argument("--lm-eval-tasks", default=DEFAULT_LM_EVAL_TASKS)
    parser.add_argument("--prompt-set", default="fullcorpus")
    parser.add_argument("--skip", default="", help="Comma-separated tracks to skip")
    parser.add_argument("--only", default="", help="Comma-separated tracks to run (overrides --skip)")
    return parser.parse_args()


def extract_step(checkpoint_path: str) -> int:
    match = re.search(r"step_(\d+)", checkpoint_path)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract step number from: {checkpoint_path}")


def filter_metrics_to_step(src: Path, max_step: int, dst: Path) -> None:
    """Write a filtered copy of a JSONL file containing only entries up to max_step."""
    count = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            try:
                d = json.loads(line)
                if d.get("step", 0) <= max_step:
                    fout.write(line)
                    count += 1
            except json.JSONDecodeError:
                continue
    logger.info("Filtered %s: %d entries (step <= %d)", src.name, count, max_step)


def run_track(name: str, cmd: list[str], timeout: int = 3600) -> tuple[bool, float]:
    """Run a track as a subprocess. Returns (success, elapsed_seconds).

    Streams stdout/stderr to the parent process so Scheduler sees log activity
    and doesn't flag the job as hung during long-running tracks.
    """
    logger.info("=" * 60)
    logger.info("TRACK: %s", name)
    logger.info("CMD: %s", " ".join(cmd))
    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        last_heartbeat = time.time()
        output_lines: list[str] = []
        while True:
            line = proc.stdout.readline()
            if line:
                print(f"  [{name}] {line}", end="", flush=True)
                output_lines.append(line)
                last_heartbeat = time.time()
            elif proc.poll() is not None:
                break
            else:
                # No output — emit heartbeat every 5 min to keep log mtime fresh
                if time.time() - last_heartbeat > 300:
                    elapsed_so_far = time.time() - start
                    print(
                        f"  [{name}] [HEARTBEAT {elapsed_so_far:.0f}s] still running...",
                        flush=True,
                    )
                    last_heartbeat = time.time()
                time.sleep(0.5)

            if time.time() - start > timeout:
                proc.kill()
                elapsed = time.time() - start
                logger.error("TIMEOUT (%s, %.1fs)", name, elapsed)
                return False, elapsed

        elapsed = time.time() - start
        if proc.returncode != 0:
            tail = "".join(output_lines[-50:])
            logger.error("FAILED (%s, %.1fs):\n%s", name, elapsed, tail[-2000:])
            return False, elapsed
        logger.info("OK (%s, %.1fs)", name, elapsed)
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        logger.error("ERROR (%s, %.1fs): %s", name, elapsed, e)
        return False, elapsed


def main() -> None:
    args = parse_args()
    step = extract_step(args.checkpoint)
    tokens_b = step * 2097152 / 1e9

    skip_set = set(s.strip() for s in args.skip.split(",") if s.strip())
    only_set = set(s.strip() for s in args.only.split(",") if s.strip())

    def should_run(track: str) -> bool:
        if only_set:
            return track in only_set
        return track not in skip_set

    pkg_dir = Path(args.output_dir) / args.name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata
    metadata = {
        "name": args.name,
        "checkpoint": args.checkpoint,
        "step": step,
        "tokens_B": round(tokens_b, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_section": args.config_section,
        "attn_res": args.attn_res,
        "device": args.device,
    }
    with open(pkg_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    timings: dict[str, tuple[bool, float]] = {}
    total_start = time.time()

    # --- Phase 1: CPU (metrics analysis) ---
    if should_run("training_analysis"):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            filtered_files = []

            metrics_src = Path(args.metrics)
            filtered_m = tmp_path / "metrics_filtered.jsonl"
            filter_metrics_to_step(metrics_src, step, filtered_m)
            filtered_files.append(str(filtered_m))

            if args.geo_metrics:
                geo_src = Path(args.geo_metrics)
                filtered_g = tmp_path / "geo_metrics_filtered.jsonl"
                filter_metrics_to_step(geo_src, step, filtered_g)
                filtered_files.append(str(filtered_g))

            output_json = pkg_dir / "training_summary.json"
            cmd = [
                sys.executable, "-m", "scripts.analysis.analyze_run",
                *filtered_files,
                "--full",
                "--name", args.name,
                "-o", str(output_json),
            ]
            ok, elapsed = run_track("training_analysis", cmd)
            timings["training_analysis"] = (ok, elapsed)

    # --- Phase 2: GPU tracks ---
    if should_run("generation"):
        gen_output = pkg_dir / "generations.json"
        cmd = [
            sys.executable, "-m", "scripts.analysis.eval_generate",
            "--checkpoint", args.checkpoint,
            "--name", args.name,
            "--attn-res", args.attn_res,
            "--prompt-set", args.prompt_set,
            "--config", args.config,
            "--config-section", args.config_section,
            "--device", args.device,
            "-o", str(gen_output),
        ]
        if Path(args.eval_data).exists():
            cmd.extend(["--eval-data", args.eval_data])
        else:
            cmd.append("--no-ppl")
            logger.warning("Eval data not found at %s, skipping perplexity", args.eval_data)
        ok, elapsed = run_track("generation", cmd, timeout=1800)
        timings["generation"] = (ok, elapsed)

    if should_run("activation_geometry"):
        ag_dir = pkg_dir / "activation_geometry"
        ag_dir.mkdir(exist_ok=True)
        cmd = [
            sys.executable, "-m", "scripts.analysis.extract_activation_geometry",
            "--checkpoint", args.checkpoint,
            "--name", args.name,
            "--attn-res", args.attn_res,
            "--config", args.config,
            "--config-section", args.config_section,
            "--device", args.device,
            "-o", str(ag_dir),
        ]
        if Path(args.eval_data).exists():
            cmd.extend(["--eval-data", args.eval_data])
        ok, elapsed = run_track("activation_geometry_extract", cmd, timeout=3600)
        timings["activation_geometry_extract"] = (ok, elapsed)

    if should_run("concept_geometry"):
        cg_dir = pkg_dir / "concept_geometry"
        cg_dir.mkdir(exist_ok=True)
        cmd = [
            sys.executable, "-m", "scripts.analysis.extract_concept_geometry",
            "--checkpoint", args.checkpoint,
            "--name", args.name,
            "--attn-res", args.attn_res,
            "--config", args.config,
            "--config-section", args.config_section,
            "--device", args.device,
            "-o", str(cg_dir),
        ]
        ok, elapsed = run_track("concept_geometry_extract", cmd, timeout=3600)
        timings["concept_geometry_extract"] = (ok, elapsed)

    if should_run("lm_eval"):
        lm_eval_dir = pkg_dir / "lm_eval"
        lm_eval_dir.mkdir(exist_ok=True)
        cmd = [
            sys.executable, "-m", "scripts.eval.run_lm_eval",
            "--checkpoint", args.checkpoint,
            "--tasks", args.lm_eval_tasks,
            "--device", args.device,
            "--output-dir", str(lm_eval_dir),
            "--config", args.config,
            "--config-section", args.config_section,
        ]
        ok, elapsed = run_track("lm_eval", cmd, timeout=3600)
        timings["lm_eval"] = (ok, elapsed)

        # Move results up to package level for easy access
        ckpt_stem = Path(args.checkpoint).stem
        if ckpt_stem.endswith(".pt"):
            ckpt_stem = ckpt_stem[:-3]
        nested_results = lm_eval_dir / ckpt_stem / "results.json"
        if not nested_results.exists():
            # Try with .pt suffix (lm_eval adapter strips .zst but keeps .pt)
            nested_results = lm_eval_dir / (ckpt_stem + ".pt") / "results.json"
        if nested_results.exists():
            shutil.copy2(nested_results, pkg_dir / "lm_eval_results.json")

    # --- Phase 3: CPU post-processing ---
    if should_run("text_quality"):
        gen_json = pkg_dir / "generations.json"
        if gen_json.exists():
            cmd = [
                sys.executable, "-m", "scripts.analysis.analyze_text_quality",
                "--input", str(gen_json),
                "--output", str(pkg_dir / "text_quality.json"),
            ]
            ok, elapsed = run_track("text_quality", cmd)
            timings["text_quality"] = (ok, elapsed)
        else:
            logger.warning("Skipping text_quality: no generations.json")

    if should_run("activation_geometry"):
        ag_dir = pkg_dir / "activation_geometry"
        npz_files = list(ag_dir.glob("*.npz"))
        if npz_files:
            cmd = [
                sys.executable, "-m", "scripts.analysis.visualize_activation_geometry",
                "--input", str(ag_dir),
            ]
            ok, elapsed = run_track("activation_geometry_viz", cmd)
            timings["activation_geometry_viz"] = (ok, elapsed)

    if should_run("concept_geometry"):
        cg_dir = pkg_dir / "concept_geometry"
        if (cg_dir / "activations.npz").exists():
            cmd = [
                sys.executable, "-m", "scripts.analysis.analyze_concept_geometry",
                "--input", str(cg_dir),
                "-o", str(cg_dir / "results.json"),
            ]
            ok, elapsed = run_track("concept_geometry_analyze", cmd, timeout=1800)
            timings["concept_geometry_analyze"] = (ok, elapsed)

            if (cg_dir / "results.json").exists():
                cmd = [
                    sys.executable, "-m", "scripts.analysis.visualize_concept_geometry",
                    "--input", str(cg_dir),
                ]
                ok, elapsed = run_track("concept_geometry_viz", cmd)
                timings["concept_geometry_viz"] = (ok, elapsed)

    # --- Phase 4: Report ---
    if should_run("report"):
        cmd = [
            sys.executable, "-m", "scripts.analysis.generate_report",
            args.name,
            "--analysis-dir", str(pkg_dir.parent),
            "-o", str(pkg_dir / "report.html"),
        ]
        ok, elapsed = run_track("report", cmd)
        timings["report"] = (ok, elapsed)

    # --- Summary ---
    total_elapsed = time.time() - total_start
    metadata["timings"] = {k: {"ok": v[0], "seconds": round(v[1], 1)} for k, v in timings.items()}
    metadata["total_seconds"] = round(total_elapsed, 1)
    with open(pkg_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("PACKAGE COMPLETE: %s", pkg_dir)
    logger.info("Total time: %.1fs", total_elapsed)
    for track, (ok, secs) in timings.items():
        status = "OK" if ok else "FAILED"
        logger.info("  %s: %s (%.1fs)", track, status, secs)

    files = list(pkg_dir.rglob("*"))
    json_files = [f for f in files if f.suffix == ".json"]
    npz_files = [f for f in files if f.suffix == ".npz"]
    png_files = [f for f in files if f.suffix == ".png"]
    logger.info("  Output: %d JSONs, %d NPZs, %d PNGs", len(json_files), len(npz_files), len(png_files))


if __name__ == "__main__":
    main()
