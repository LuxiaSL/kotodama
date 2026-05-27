"""Compare multiple analysis packages and produce cross-checkpoint analysis.

Usage:
    cd pretraining
    python -m scripts.analysis.compare_packages \
        --packages analysis/packages/fullcorpus-Q1,fullcorpus-Q2,fullcorpus-Q3,fullcorpus-Q4 \
        --output analysis/comparisons/fullcorpus-quartiles
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONVERGENCE_THRESHOLD = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple analysis packages"
    )
    parser.add_argument(
        "--packages",
        required=True,
        help="Comma-separated package directory paths or names (resolved under --packages-dir)",
    )
    parser.add_argument(
        "--packages-dir",
        default="analysis/packages",
        help="Parent directory to resolve package names",
    )
    parser.add_argument(
        "--output",
        default="analysis/comparisons/comparison",
        help="Output directory",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=CONVERGENCE_THRESHOLD,
        help="Relative delta threshold for convergence detection",
    )
    return parser.parse_args()


def load_package(pkg_dir: Path) -> dict:
    """Load all JSONs from a package directory."""
    pkg: dict = {"path": str(pkg_dir)}

    for name in ["metadata", "training_summary", "text_quality", "lm_eval_results"]:
        json_path = pkg_dir / f"{name}.json"
        if json_path.exists():
            with open(json_path) as f:
                pkg[name] = json.load(f)
        else:
            pkg[name] = None

    # Concept geometry results
    cg_results = pkg_dir / "concept_geometry" / "results.json"
    if cg_results.exists():
        with open(cg_results) as f:
            pkg["concept_geometry"] = json.load(f)
    else:
        pkg["concept_geometry"] = None

    return pkg


def extract_training_metrics(pkg: dict) -> dict[str, float]:
    """Extract key training metrics from a package."""
    metrics: dict[str, float] = {}
    summary = pkg.get("training_summary", {})
    if not summary:
        return metrics

    s = summary.get("summary", {})
    for key in ["loss_final", "loss_min", "ppl_final", "rankme_final",
                 "ww_alpha_final", "ww_healthy_frac_final", "twonn_id_final",
                 "avg_tokens_per_sec"]:
        if key in s:
            metrics[key] = s[key]

    return metrics


def extract_geometric_metrics(pkg: dict) -> dict[str, float]:
    """Extract geometric health metrics from training_summary."""
    metrics: dict[str, float] = {}
    summary = pkg.get("training_summary", {})
    if not summary:
        return metrics

    geo = summary.get("geometric_health", {})
    if not geo:
        return metrics

    # Extract depth gradients
    for key in ["attn_entropy_mean", "anisotropy", "stable_rank_q_proj"]:
        for pos in ["first", "mid", "last"]:
            full_key = f"depth_gradient_{key}_{pos}"
            dg = geo.get(f"depth_gradient_{key}", {})
            if isinstance(dg, dict) and pos in dg:
                metrics[full_key] = dg[pos]

    return metrics


def extract_lm_eval_metrics(pkg: dict) -> dict[str, float]:
    """Extract lm-eval benchmark scores."""
    metrics: dict[str, float] = {}
    results = pkg.get("lm_eval_results", {})
    if not results:
        return metrics

    for task_name, task_data in results.items():
        if not isinstance(task_data, dict):
            continue
        for key, val in task_data.items():
            if key.endswith(",none") and isinstance(val, (int, float)):
                metric_name = f"{task_name}/{key.replace(',none', '')}"
                metrics[metric_name] = val

    return metrics


def extract_text_quality_metrics(pkg: dict) -> dict[str, float]:
    """Extract text quality profile metrics."""
    metrics: dict[str, float] = {}
    tq = pkg.get("text_quality", {})
    if not tq:
        return metrics

    profiles = tq.get("model_profiles", {})
    for model_name, profile in profiles.items():
        if isinstance(profile, dict):
            for key, val in profile.items():
                if isinstance(val, (int, float)):
                    metrics[key] = val
        break  # Only first model (single checkpoint per package)

    return metrics


def compute_convergence(
    names: list[str],
    values: list[float],
    threshold: float,
) -> dict:
    """Detect when a metric stops changing significantly."""
    if len(values) < 2:
        return {"converged": False, "deltas": []}

    deltas = []
    for i in range(1, len(values)):
        if values[i - 1] != 0:
            rel_delta = abs(values[i] - values[i - 1]) / abs(values[i - 1])
        else:
            rel_delta = abs(values[i] - values[i - 1])
        deltas.append({
            "from": names[i - 1],
            "to": names[i],
            "absolute": round(values[i] - values[i - 1], 6),
            "relative": round(rel_delta, 6),
        })

    # Find first point where 2+ consecutive deltas are below threshold
    converge_at = None
    for i in range(len(deltas) - 1):
        if deltas[i]["relative"] < threshold and deltas[i + 1]["relative"] < threshold:
            converge_at = deltas[i]["from"]
            break

    return {
        "converged": converge_at is not None,
        "convergence_point": converge_at,
        "deltas": deltas,
        "values": {n: round(v, 6) for n, v in zip(names, values)},
    }


def main() -> None:
    args = parse_args()
    pkg_names = [p.strip() for p in args.packages.split(",")]

    packages_dir = Path(args.packages_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve and load packages
    packages: list[dict] = []
    for name in pkg_names:
        pkg_path = Path(name)
        if not pkg_path.is_dir():
            pkg_path = packages_dir / name
        if not pkg_path.is_dir():
            logger.error("Package not found: %s", name)
            continue

        logger.info("Loading package: %s", pkg_path)
        pkg = load_package(pkg_path)
        packages.append(pkg)

    if len(packages) < 2:
        logger.error("Need at least 2 packages to compare (got %d)", len(packages))
        return

    # Sort by step
    packages.sort(key=lambda p: p.get("metadata", {}).get("step", 0))

    names = [p.get("metadata", {}).get("name", f"pkg_{i}") for i, p in enumerate(packages)]
    steps = [p.get("metadata", {}).get("step", 0) for p in packages]
    tokens = [p.get("metadata", {}).get("tokens_B", 0) for p in packages]

    # Build comparison
    comparison: dict = {
        "checkpoints": [
            {"name": n, "step": s, "tokens_B": t}
            for n, s, t in zip(names, steps, tokens)
        ],
        "training": {},
        "geometric": {},
        "lm_eval": {},
        "text_quality": {},
    }

    # Gather all metrics per category
    extractors = [
        ("training", extract_training_metrics),
        ("geometric", extract_geometric_metrics),
        ("lm_eval", extract_lm_eval_metrics),
        ("text_quality", extract_text_quality_metrics),
    ]

    all_metrics_by_category: dict[str, dict[str, list[float | None]]] = {}

    for category, extractor in extractors:
        category_data: dict[str, list[float | None]] = {}
        for pkg in packages:
            metrics = extractor(pkg)
            for key, val in metrics.items():
                if key not in category_data:
                    category_data[key] = [None] * len(packages)
                idx = packages.index(pkg)
                category_data[key][idx] = val

        comparison[category] = {
            k: [round(v, 6) if v is not None else None for v in vals]
            for k, vals in sorted(category_data.items())
        }
        all_metrics_by_category[category] = category_data

    # Save comparison
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Wrote comparison.json")

    # Convergence analysis
    convergence: dict = {}
    for category, category_data in all_metrics_by_category.items():
        for metric_key, values in category_data.items():
            clean_vals = [v for v in values if v is not None]
            clean_names = [n for n, v in zip(names, values) if v is not None]
            if len(clean_vals) >= 2:
                full_key = f"{category}/{metric_key}"
                convergence[full_key] = compute_convergence(
                    clean_names, clean_vals, args.convergence_threshold
                )

    with open(output_dir / "convergence.json", "w") as f:
        json.dump(convergence, f, indent=2)
    logger.info("Wrote convergence.json")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: {' vs '.join(names)}")
    print(f"{'=' * 70}")

    for category in ["training", "lm_eval"]:
        data = comparison.get(category, {})
        if not data:
            continue
        print(f"\n  {category.upper()}:")
        header = f"  {'metric':<40} " + "  ".join(f"{n:>12}" for n in names)
        print(header)
        print("  " + "-" * len(header))
        for key, vals in sorted(data.items()):
            formatted = []
            for v in vals:
                if v is None:
                    formatted.append(f"{'—':>12}")
                elif abs(v) < 0.01:
                    formatted.append(f"{v:>12.6f}")
                else:
                    formatted.append(f"{v:>12.4f}")
            print(f"  {key:<40} {'  '.join(formatted)}")

    # Convergence summary
    converged = {k: v for k, v in convergence.items() if v.get("converged")}
    if converged:
        print(f"\n  CONVERGED METRICS ({len(converged)}/{len(convergence)}):")
        for key, info in sorted(converged.items()):
            print(f"    {key}: plateaued at {info['convergence_point']}")

    not_converged = {k: v for k, v in convergence.items() if not v.get("converged")}
    if not_converged:
        still_moving = []
        for key, info in not_converged.items():
            if info["deltas"]:
                last_delta = info["deltas"][-1]
                if last_delta["relative"] > args.convergence_threshold:
                    still_moving.append((key, last_delta["relative"]))
        if still_moving:
            still_moving.sort(key=lambda x: x[1], reverse=True)
            print(f"\n  STILL MOVING (last delta > {args.convergence_threshold}):")
            for key, rel in still_moving[:15]:
                print(f"    {key}: {rel:.4f} relative change")

    logger.info("Comparison complete: %s", output_dir)


if __name__ == "__main__":
    main()
