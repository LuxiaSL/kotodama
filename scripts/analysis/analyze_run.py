#!/usr/bin/env python3
"""Unified training run analyzer (Track 1+2).

Replaces: analyze_metrics.py, interaction_analysis_v2.py,
analyze_lang_diagnostics.py, viz_alpha_rank.py.

Usage::

    # Single run analysis
    python -m scripts.analyze_run data/nca_proxy_muon002_full_metrics.jsonl --summary
    python -m scripts.analyze_run data/nca_proxy_muon002_full_metrics.jsonl --dynamics
    python -m scripts.analyze_run data/nca_proxy_muon002_full_metrics.jsonl --full

    # Cross-run comparison
    python -m scripts.analyze_run data/*_metrics.jsonl --compare

    # Factorial analysis (2x2)
    python -m scripts.analyze_run data/sweep_p3_muon002_metrics.jsonl \\
        data/attnres_p3_muon002_metrics.jsonl \\
        data/nca_proxy_muon002_full_metrics.jsonl \\
        data/attnres_nca002_metrics.jsonl \\
        --factorial "NCA:P3,NCA AttnRes:P3,P3-AR"

    # Compare against reference
    python -m scripts.analyze_run data/metrics.jsonl --reference analysis/baseline/training_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.metrics_io import load_metrics
from src.eval.run_analysis import (
    compare_runs,
    endpoint_summary,
    factorial_analysis,
    geometric_health,
    reference_comparison,
    training_dynamics,
)

logger = logging.getLogger(__name__)


def _infer_name(path: Path) -> str:
    """Infer a short run name from a metrics file path."""
    stem = path.stem
    # Strip common suffixes
    for suffix in ("_metrics", "_full_metrics", "_full"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _parse_factorial(spec: str) -> dict[str, tuple[str, str]]:
    """Parse factorial spec string into factor dict.

    Format: "FactorA:without,with FactorB:without,with"
    Example: "NCA:P3,NCA AttnRes:P3,P3-AR"

    Returns:
        Dict mapping factor name → (without_run, with_run).
    """
    factors: dict[str, tuple[str, str]] = {}
    for part in spec.split():
        if ":" not in part:
            raise ValueError(f"Invalid factor spec: '{part}'. Expected 'Name:without,with'")
        name, vals = part.split(":", 1)
        levels = vals.split(",")
        if len(levels) != 2:
            raise ValueError(f"Factor '{name}' needs exactly 2 levels, got {len(levels)}")
        factors[name] = (levels[0], levels[1])
    return factors


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="Unified training run analyzer (Track 1+2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("files", nargs="+", type=Path, help="JSONL metrics file(s)")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path")
    p.add_argument("--name", type=str, default=None, help="Run name override")

    # Analysis modes
    modes = p.add_argument_group("analysis modes")
    modes.add_argument("--summary", action="store_true", help="Endpoint summary")
    modes.add_argument("--dynamics", action="store_true", help="Training dynamics analysis")
    modes.add_argument("--health", action="store_true", help="Geometric health profile")
    modes.add_argument("--compare", action="store_true", help="Cross-run comparison")
    modes.add_argument("--factorial", type=str, default=None,
                       metavar="SPEC", help="Factorial analysis: 'Factor1:a,b Factor2:c,d'")
    modes.add_argument("--reference", type=Path, default=None,
                       metavar="PATH", help="Reference summary JSON for comparison")
    modes.add_argument("--full", action="store_true", help="All single-run analyses")

    # Options
    p.add_argument("--metrics", type=str, default=None,
                   help="Additional metric keys for dynamics (comma-separated)")

    args = p.parse_args()

    # Default: if no mode specified, show summary
    any_mode = (args.summary or args.dynamics or args.health or args.compare
                or args.factorial or args.reference or args.full)
    if not any_mode:
        args.summary = True

    # --full enables all single-run modes
    if args.full:
        args.summary = True
        args.dynamics = True
        args.health = True

    # Load data
    files = [f for f in args.files if f.exists()]
    missing = [f for f in args.files if not f.exists()]
    if missing:
        logger.warning("Missing files: %s", [str(f) for f in missing])
    if not files:
        logger.error("No valid input files found")
        sys.exit(1)

    # Build named datasets
    datasets: dict[str, dict[int, dict[str, float]]] = {}
    for f in files:
        name = args.name if (args.name and len(files) == 1) else _infer_name(f)
        # Handle duplicate names
        if name in datasets:
            name = f"{name}_{f.parent.name}"
        datasets[name] = load_metrics(f)

    # Run analyses
    result: dict[str, Any] = {}
    first_name = next(iter(datasets))
    first_data = datasets[first_name]

    if args.summary:
        result["summary"] = endpoint_summary(first_data)

    if args.dynamics:
        extra_metrics = args.metrics.split(",") if args.metrics else None
        result["dynamics"] = training_dynamics(first_data, metric_keys=extra_metrics)

    if args.health:
        result["geometric_health"] = geometric_health(first_data)

    if args.compare and len(datasets) >= 2:
        result["comparison"] = compare_runs(datasets)

    if args.factorial:
        try:
            factors = _parse_factorial(args.factorial)
        except ValueError as e:
            logger.error("Invalid factorial spec: %s", e)
            sys.exit(1)
        result["factorial"] = factorial_analysis(datasets, factors)

    if args.reference:
        if not args.reference.exists():
            logger.error("Reference file not found: %s", args.reference)
            sys.exit(1)
        with open(args.reference) as f:
            ref = json.load(f)
        # Support both flat reference dicts and nested {"summary": {...}}
        if "summary" in ref:
            ref = ref["summary"]
        result["reference_comparison"] = reference_comparison(first_data, ref)

    # Determine output path
    output_path = args.output
    if output_path is None:
        if args.compare or args.factorial:
            output_dir = Path("analysis/comparisons")
            names = "_vs_".join(list(datasets.keys())[:3])
            if len(datasets) > 3:
                names += f"_+{len(datasets) - 3}"
            output_path = output_dir / f"{names}.json"
        else:
            output_dir = Path("analysis") / first_name
            output_path = output_dir / "training_summary.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info("Wrote %s", output_path)

    # Print concise summary to stdout
    _print_summary(result, datasets)


def _print_summary(result: dict[str, Any], datasets: dict[str, Any]) -> None:
    """Print a concise human-readable summary to stdout."""
    if "summary" in result:
        s = result["summary"]
        print(f"\n{'=' * 60}")
        print("ENDPOINT SUMMARY")
        print(f"{'=' * 60}")
        if "loss_final" in s:
            print(f"  Loss: {s.get('loss_initial', '?'):.4f} → {s['loss_final']:.4f} (min {s.get('loss_min', '?'):.4f})")
        if "rankme_final" in s:
            print(f"  RankMe: {s.get('rankme_initial', '?'):.1f} → {s['rankme_final']:.1f} (rebound {s.get('rankme_rebound_ratio', 0):.2f}x)")
        if "ww_alpha_final" in s:
            print(f"  WW alpha: {s['ww_alpha_final']:.2f}, healthy: {s.get('ww_healthy_frac_final', 0):.0%}")
        if "twonn_id_final" in s:
            print(f"  TwoNN ID: {s['twonn_id_final']:.1f}")
        if "avg_tokens_per_sec" in s:
            print(f"  Throughput: {s['avg_tokens_per_sec']:.0f} tok/s")
        if "tokens_B" in s:
            print(f"  Tokens: {s['tokens_B']:.3f}B, steps: {s.get('max_step', '?')}")

    if "dynamics" in result:
        d = result["dynamics"]
        print(f"\n{'=' * 60}")
        print("TRAINING DYNAMICS")
        print(f"{'=' * 60}")
        for key, metric_data in d.get("metrics", {}).items():
            n_jumps = len(metric_data.get("jumps", []))
            n_plateaus = len(metric_data.get("plateaus", []))
            n_inflections = len(metric_data.get("inflection_points", []))
            stability = metric_data.get("stability_score", None)
            stab_str = f", stability={stability:.3f}" if stability is not None else ""
            print(f"  {key}: {metric_data['n_points']} pts, {n_jumps} jumps, "
                  f"{n_plateaus} plateaus, {n_inflections} inflections{stab_str}")

    if "geometric_health" in result:
        gh = result["geometric_health"]
        print(f"\n{'=' * 60}")
        print("GEOMETRIC HEALTH")
        print(f"{'=' * 60}")
        layers = gh.get("layers", [])
        print(f"  Sampled layers: {layers}")
        if "rankme_stability" in gh:
            print(f"  RankMe stability (late_std/early_std): {gh['rankme_stability']:.3f}")
        if "depth_gradient" in gh:
            dg = gh["depth_gradient"]
            for metric, vals in dg.items():
                if metric == "layers":
                    continue
                parts = [f"{k}={v:.3f}" if v is not None else f"{k}=—" for k, v in vals.items()]
                print(f"  Depth gradient {metric}: {', '.join(parts)}")

    if "comparison" in result:
        c = result["comparison"]
        print(f"\n{'=' * 60}")
        print(f"COMPARISON (baseline: {c['baseline']})")
        print(f"{'=' * 60}")
        runs = c.get("runs", {})
        keys = ["loss_final", "rankme_final", "ww_alpha_final", "twonn_id_final"]
        header = f"{'Metric':<25}" + "".join(f"{n:>15}" for n in runs.keys())
        print(header)
        for key in keys:
            row = f"{key:<25}"
            for name, s in runs.items():
                v = s.get(key)
                row += f"{v:>15.4f}" if v is not None else f"{'—':>15}"
            best = c.get("best", {}).get(key, "")
            row += f"  ← {best}" if best else ""
            print(row)

    if "factorial" in result:
        fa = result["factorial"]
        print(f"\n{'=' * 60}")
        print(f"FACTORIAL ANALYSIS (step {fa.get('step', '?')})")
        print(f"{'=' * 60}")
        effects = fa.get("effects", {})
        # Show key metrics
        for key in ["train/loss", "geo/rankme_last"]:
            if key in effects:
                e = effects[key]
                factor_keys = [k for k in e if k.endswith("_main")]
                parts = [f"{k}={e[k]:+.4f}" for k in factor_keys]
                parts.append(f"interaction={e['interaction']:+.4f}")
                parts.append(f"({e['classification']})")
                print(f"  {key}: {' '.join(parts)}")


if __name__ == "__main__":
    main()
