"""Unified training run analysis engine.

Pure computation functions — no I/O, CLI, or side effects.
Operates on step-indexed metric dicts from metrics_io.load_metrics().

Six analysis functions, each returning a JSON-serializable dict:
  A. endpoint_summary    — loss, rankme, ww, twonn, throughput
  B. training_dynamics    — windowed slopes, second derivatives, phase transitions
  C. geometric_health     — per-layer profiles at landmarks, depth gradient
  D. compare_runs         — side-by-side endpoint table with deltas
  E. factorial_analysis   — main effects + interaction terms
  F. reference_comparison — delta vs reference at matched landmarks
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from src.eval.metrics_io import extract_series, get_value

logger = logging.getLogger(__name__)

# Type alias for step-indexed metrics
StepData = dict[int, dict[str, float]]

# Default metrics for dynamics analysis
DEFAULT_DYNAMICS_METRICS = ["train/loss", "geo/rankme_last"]

# Landmark percentages (of max step)
LANDMARK_PCTS = [0.10, 0.25, 0.50, 0.75, 0.90]

# Window sizes for multi-resolution slope analysis
SLOPE_WINDOWS = [100, 500, 1000, 5000]

# Projections tracked for stable rank
STABLE_RANK_PROJECTIONS = ["q_proj", "k_proj", "o_proj", "gate_proj", "down_proj"]


# =============================================================================
# A. Endpoint Summary
# =============================================================================


def endpoint_summary(data: StepData) -> dict[str, Any]:
    """Compute endpoint summary for a single training run.

    Consolidates analyze_metrics.py:summarize_run().

    Returns dict with loss, perplexity, rankme, ww_alpha, twonn, throughput.
    """
    summary: dict[str, Any] = {}

    max_step = max(data.keys()) if data else 0
    summary["max_step"] = max_step

    # Loss trajectory
    loss = extract_series(data, "train/loss")
    if loss:
        summary["loss_initial"] = loss[0][1]
        summary["loss_final"] = loss[-1][1]
        summary["loss_min"] = min(v for _, v in loss)
        summary["loss_min_step"] = min(loss, key=lambda x: x[1])[0]

    # Perplexity
    ppl = extract_series(data, "train/perplexity")
    if ppl:
        summary["ppl_final"] = ppl[-1][1]

    # Throughput (skip warmup)
    tps = extract_series(data, "perf/tokens_per_sec")
    if tps:
        stable = tps[2:] if len(tps) > 2 else tps
        summary["avg_tokens_per_sec"] = sum(v for _, v in stable) / len(stable)

    # RankMe trajectory
    rankme = extract_series(data, "geo/rankme_last")
    if rankme:
        summary["rankme_initial"] = rankme[0][1]
        summary["rankme_min"] = min(v for _, v in rankme)
        summary["rankme_min_step"] = min(rankme, key=lambda x: x[1])[0]
        summary["rankme_final"] = rankme[-1][1]
        min_val = max(summary["rankme_min"], 1e-6)
        summary["rankme_rebound_ratio"] = summary["rankme_final"] / min_val

    # WeightWatcher alpha
    ww = extract_series(data, "geo/ww_alpha_mean")
    if ww:
        summary["ww_alpha_initial"] = ww[0][1]
        summary["ww_alpha_final"] = ww[-1][1]
    ww_healthy = extract_series(data, "geo/ww_alpha_healthy_frac")
    if ww_healthy:
        summary["ww_healthy_frac_final"] = ww_healthy[-1][1]

    # TwoNN ID — try last layer first, then discover
    twonn = extract_series(data, "geo/twonn_id/layer_27")
    if not twonn:
        # Discover available TwoNN keys
        all_keys: set[str] = set()
        for step_data in data.values():
            all_keys.update(step_data.keys())
        twonn_keys = sorted(k for k in all_keys if k.startswith("geo/twonn_id/"))
        if twonn_keys:
            twonn = extract_series(data, twonn_keys[-1])
    if twonn:
        summary["twonn_id_final"] = twonn[-1][1]

    # Tokens consumed
    tok = extract_series(data, "data/tokens_consumed_B")
    if tok:
        summary["tokens_B"] = tok[-1][1]

    return summary


# =============================================================================
# B. Training Dynamics (NEW)
# =============================================================================


def training_dynamics(
    data: StepData,
    metric_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Multi-resolution windowed analysis of training dynamics.

    For each metric, computes:
    - Slopes at multiple window sizes at landmark steps
    - Second derivatives (slope of slopes)
    - Rolling statistics (mean, std, p25, p75)
    - Phase transition detection (plateaus, jumps)

    Args:
        data: Step-indexed metrics.
        metric_keys: Metrics to analyze. Defaults to loss + rankme.

    Returns:
        Dict with per-metric dynamics analysis.
    """
    if metric_keys is None:
        metric_keys = list(DEFAULT_DYNAMICS_METRICS)

    max_step = max(data.keys()) if data else 0
    landmarks = {
        f"pct_{int(p * 100)}": int(max_step * p)
        for p in LANDMARK_PCTS
    }

    result: dict[str, Any] = {"max_step": max_step, "landmarks": landmarks}
    metrics_results: dict[str, Any] = {}

    for key in metric_keys:
        series = extract_series(data, key)
        if len(series) < 10:
            continue

        steps = [s for s, _ in series]
        values = [v for _, v in series]

        metric_result: dict[str, Any] = {"n_points": len(series)}

        # Windowed slopes at landmark steps
        slopes_at_landmarks: dict[str, dict[str, float | None]] = {}
        for lm_name, lm_step in landmarks.items():
            slopes_at_landmarks[lm_name] = {}
            for window in SLOPE_WINDOWS:
                slope = _slope_at_step(steps, values, lm_step, window)
                slopes_at_landmarks[lm_name][f"w{window}"] = slope
        metric_result["slopes_at_landmarks"] = slopes_at_landmarks

        # Full windowed slopes at finest resolution (1000-step windows)
        full_slopes = _windowed_slopes(steps, values, window=1000)

        # Second derivative (slope of slopes)
        if len(full_slopes) >= 5:
            slope_steps = [s for s, _ in full_slopes]
            slope_vals = [v for _, v in full_slopes]
            second_deriv = _windowed_slopes(slope_steps, slope_vals, window=max(1, len(slope_steps) // 5))
            metric_result["inflection_points"] = _find_sign_changes(second_deriv)
        else:
            metric_result["inflection_points"] = []

        # Rolling statistics (500-step windows)
        rolling = _rolling_stats(steps, values, window=500)
        if rolling:
            metric_result["rolling_stats"] = rolling

            # Stability score: late std / early std
            n = len(rolling)
            early_stds = [r["std"] for r in rolling[: n // 4] if r["std"] > 0]
            late_stds = [r["std"] for r in rolling[3 * n // 4 :] if r["std"] > 0]
            if early_stds and late_stds:
                early_avg = sum(early_stds) / len(early_stds)
                late_avg = sum(late_stds) / len(late_stds)
                metric_result["stability_score"] = late_avg / max(early_avg, 1e-10)

        # Phase transition detection
        metric_result["plateaus"] = _detect_plateaus(full_slopes)
        metric_result["jumps"] = _detect_jumps(steps, values)

        # Sign changes in slope (convergence → divergence)
        metric_result["slope_sign_changes"] = _find_sign_changes(full_slopes)

        metrics_results[key] = metric_result

    result["metrics"] = metrics_results
    return result


def _slope_at_step(
    steps: list[int],
    values: list[float],
    target_step: int,
    window: int,
) -> float | None:
    """Compute slope over a window centered on target_step."""
    half = window // 2
    lo, hi = target_step - half, target_step + half

    # Collect points in window
    pts = [(s, v) for s, v in zip(steps, values) if lo <= s <= hi]
    if len(pts) < 3:
        return None

    return _linear_slope(pts)


def _windowed_slopes(
    steps: list[int],
    values: list[float],
    window: int,
) -> list[tuple[int, float]]:
    """Compute slopes at each position using a sliding window."""
    if len(steps) < 3:
        return []

    results: list[tuple[int, float]] = []
    half = window // 2

    for i in range(len(steps)):
        center = steps[i]
        lo, hi = center - half, center + half
        pts = [(s, v) for s, v in zip(steps, values) if lo <= s <= hi]
        if len(pts) >= 3:
            slope = _linear_slope(pts)
            if slope is not None:
                results.append((center, slope))

    return results


def _linear_slope(pts: list[tuple[int, float]]) -> float | None:
    """OLS slope from (x, y) pairs."""
    n = len(pts)
    if n < 2:
        return None

    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-20:
        return None

    return (n * sxy - sx * sy) / denom


def _rolling_stats(
    steps: list[int],
    values: list[float],
    window: int,
) -> list[dict[str, float]]:
    """Compute rolling mean/std/p25/p75 over non-overlapping windows."""
    if not steps:
        return []

    min_step, max_step = steps[0], steps[-1]
    results: list[dict[str, float]] = []

    # Build lookup for fast windowing
    step_to_val = dict(zip(steps, values))

    for win_start in range(min_step, max_step, window):
        win_end = win_start + window
        win_vals = [
            v for s, v in zip(steps, values) if win_start <= s < win_end
        ]
        if len(win_vals) < 2:
            continue

        win_vals_sorted = sorted(win_vals)
        n = len(win_vals_sorted)
        mean = sum(win_vals) / n
        variance = sum((v - mean) ** 2 for v in win_vals) / (n - 1)
        std = variance**0.5

        p25_idx = max(0, int(n * 0.25) - 1)
        p75_idx = min(n - 1, int(n * 0.75))

        results.append({
            "step_center": win_start + window // 2,
            "mean": mean,
            "std": std,
            "p25": win_vals_sorted[p25_idx],
            "p75": win_vals_sorted[p75_idx],
            "n": n,
        })

    return results


def _detect_plateaus(
    slopes: list[tuple[int, float]],
    threshold: float = 1e-7,
    min_consecutive: int = 3,
) -> list[dict[str, Any]]:
    """Detect regions where |slope| < threshold for min_consecutive windows."""
    if not slopes:
        return []

    plateaus: list[dict[str, Any]] = []
    run_start: int | None = None
    run_count = 0

    for step, slope in slopes:
        if abs(slope) < threshold:
            if run_start is None:
                run_start = step
            run_count += 1
        else:
            if run_count >= min_consecutive and run_start is not None:
                plateaus.append({
                    "start_step": run_start,
                    "end_step": step,
                    "duration_steps": step - run_start,
                })
            run_start = None
            run_count = 0

    # Handle trailing plateau
    if run_count >= min_consecutive and run_start is not None and slopes:
        plateaus.append({
            "start_step": run_start,
            "end_step": slopes[-1][0],
            "duration_steps": slopes[-1][0] - run_start,
        })

    return plateaus


def _detect_jumps(
    steps: list[int],
    values: list[float],
    sigma_threshold: float = 3.0,
    warmup_frac: float = 0.05,
) -> list[dict[str, Any]]:
    """Detect step-over-step jumps exceeding sigma_threshold × running std."""
    if len(steps) < 20:
        return []

    jumps: list[dict[str, Any]] = []
    warmup_n = max(10, int(len(steps) * warmup_frac))

    # Compute running mean and std of deltas
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    if not deltas:
        return []

    for i in range(warmup_n, len(deltas)):
        # Running stats from deltas so far (excluding current)
        past = deltas[:i]
        mean_delta = sum(past) / len(past)
        var_delta = sum((d - mean_delta) ** 2 for d in past) / len(past)
        std_delta = max(var_delta**0.5, 1e-10)

        deviation = abs(deltas[i] - mean_delta) / std_delta
        if deviation > sigma_threshold:
            jumps.append({
                "step": steps[i + 1],
                "delta": deltas[i],
                "sigma": round(deviation, 2),
            })

    return jumps


def _find_sign_changes(
    series: list[tuple[int, float]],
) -> list[dict[str, Any]]:
    """Find steps where the series changes sign."""
    changes: list[dict[str, Any]] = []
    for i in range(1, len(series)):
        prev_step, prev_val = series[i - 1]
        curr_step, curr_val = series[i]
        if prev_val * curr_val < 0:
            direction = "positive→negative" if prev_val > 0 else "negative→positive"
            changes.append({"step": curr_step, "direction": direction})
    return changes


# =============================================================================
# C. Geometric Health Profile
# =============================================================================


def geometric_health(data: StepData) -> dict[str, Any]:
    """Per-layer geometric health profile at landmark steps.

    Auto-discovers sampled layers, extracts per-layer metrics at landmarks,
    and computes cross-layer depth gradients.
    """
    max_step = max(data.keys()) if data else 0
    if max_step == 0:
        return {}

    # Discover sampled layers from metric keys
    layers = _discover_layers(data)
    if not layers:
        return {"error": "no geometric layer metrics found"}

    # Find steps that actually have geo data
    geo_steps = sorted(
        s for s in data if any(k.startswith("geo/layer_") for k in data[s])
    )
    if not geo_steps:
        return {"error": "no geometric data at any step"}

    # Landmark steps — snap to nearest step WITH geo data
    landmark_pcts = {"early": 0.10, "quarter": 0.25, "mid": 0.50, "three_quarter": 0.75, "late": 0.90, "final": 1.0}
    landmarks = {}
    for name, pct in landmark_pcts.items():
        target = int(max_step * pct)
        landmarks[name] = min(geo_steps, key=lambda s: abs(s - target))

    result: dict[str, Any] = {
        "layers": layers,
        "landmarks": landmarks,
    }

    # Per-layer profiles at each landmark
    profiles: dict[str, dict[str, Any]] = {}
    for lm_name, lm_step in landmarks.items():
        if lm_step is None:
            continue
        profile: dict[str, Any] = {"step": lm_step}
        for layer in layers:
            layer_data: dict[str, float | None] = {}
            prefix = f"geo/layer_{layer}"

            # Stable rank per projection
            for proj in STABLE_RANK_PROJECTIONS:
                layer_data[f"stable_rank_{proj}"] = get_value(data, lm_step, f"{prefix}/stable_rank_{proj}")

            # Attention entropy
            layer_data["attn_entropy_mean"] = get_value(data, lm_step, f"{prefix}/attn_entropy_mean")
            layer_data["attn_entropy_std"] = get_value(data, lm_step, f"{prefix}/attn_entropy_std")

            # Anisotropy and dead units
            layer_data["anisotropy"] = get_value(data, lm_step, f"{prefix}/anisotropy")
            layer_data["dead_units"] = get_value(data, lm_step, f"{prefix}/dead_units")

            profile[f"layer_{layer}"] = layer_data
        profiles[lm_name] = profile

    result["profiles"] = profiles

    # Cross-layer depth gradient at final landmark
    final_step = landmarks.get("final")
    if final_step is not None and len(layers) >= 3:
        first, mid, last = layers[0], layers[len(layers) // 2], layers[-1]
        gradient: dict[str, Any] = {
            "layers": {"first": first, "mid": mid, "last": last},
        }
        for metric_suffix in ["attn_entropy_mean", "anisotropy", "stable_rank_q_proj"]:
            vals = {}
            for label, layer in [("first", first), ("mid", mid), ("last", last)]:
                vals[label] = get_value(data, final_step, f"geo/layer_{layer}/{metric_suffix}")
            gradient[metric_suffix] = vals
        result["depth_gradient"] = gradient

    # Stability scores per layer (late std / early std for rankme)
    rankme = extract_series(data, "geo/rankme_last")
    if len(rankme) >= 10:
        n = len(rankme)
        early_vals = [v for _, v in rankme[: n // 4]]
        late_vals = [v for _, v in rankme[3 * n // 4 :]]
        if early_vals and late_vals:
            early_std = _std(early_vals)
            late_std = _std(late_vals)
            result["rankme_stability"] = late_std / max(early_std, 1e-10)

    return result


def _discover_layers(data: StepData) -> list[int]:
    """Find all layer indices that have geometric metrics."""
    layer_pattern = re.compile(r"geo/layer_(\d+)/")
    layers: set[int] = set()
    for step_data in data.values():
        for key in step_data:
            m = layer_pattern.match(key)
            if m:
                layers.add(int(m.group(1)))
    return sorted(layers)


def _nearest_step(data: StepData, target: int) -> int | None:
    """Find the step in data closest to target."""
    if not data:
        return None
    return min(data.keys(), key=lambda s: abs(s - target))


# =============================================================================
# D. Cross-Run Comparison
# =============================================================================


def compare_runs(
    datasets: dict[str, StepData],
) -> dict[str, Any]:
    """Side-by-side comparison of endpoint summaries across runs.

    Consolidates analyze_metrics.py:compare_runs().

    Args:
        datasets: Mapping of run name → step-indexed metrics.

    Returns:
        Dict with per-run summaries and delta table.
    """
    summaries = {name: endpoint_summary(data) for name, data in datasets.items()}

    # Compute deltas relative to first run
    run_names = list(summaries.keys())
    if len(run_names) < 2:
        return {"runs": summaries}

    baseline_name = run_names[0]
    baseline = summaries[baseline_name]

    # Direction awareness: lower is better for loss/ppl, higher for rankme/twonn
    lower_is_better = {"loss_final", "loss_min", "ppl_final", "ww_alpha_final"}

    deltas: dict[str, dict[str, float | None]] = {}
    comparison_keys = [
        "loss_final", "loss_min", "ppl_final", "rankme_final", "rankme_min",
        "rankme_rebound_ratio", "ww_alpha_final", "ww_healthy_frac_final",
        "twonn_id_final", "avg_tokens_per_sec", "tokens_B",
    ]

    for name in run_names[1:]:
        run_deltas: dict[str, float | None] = {}
        for key in comparison_keys:
            base_val = baseline.get(key)
            run_val = summaries[name].get(key)
            if base_val is not None and run_val is not None:
                delta = run_val - base_val
                run_deltas[key] = delta
            else:
                run_deltas[key] = None
        deltas[name] = run_deltas

    # Find best per metric
    best: dict[str, str] = {}
    for key in comparison_keys:
        vals = {name: s.get(key) for name, s in summaries.items() if s.get(key) is not None}
        if vals:
            if key in lower_is_better:
                best[key] = min(vals, key=lambda n: vals[n])
            else:
                best[key] = max(vals, key=lambda n: vals[n])

    return {
        "baseline": baseline_name,
        "runs": summaries,
        "deltas": deltas,
        "best": best,
    }


# =============================================================================
# E. Factorial Analysis
# =============================================================================


def factorial_analysis(
    datasets: dict[str, StepData],
    factors: dict[str, tuple[str, str]],
    step: int | None = None,
) -> dict[str, Any]:
    """2×2 factorial analysis: main effects + interaction.

    Consolidates interaction_analysis_v2.py.

    Args:
        datasets: Mapping of run name → step-indexed metrics.
            Must contain exactly 4 runs matching the 2×2 grid.
        factors: Mapping of factor name → (without_run, with_run).
            E.g., {"NCA": ("P3", "NCA"), "AttnRes": ("P3", "P3-AR")}
            The 2×2 grid is:
              - (without_A, without_B) = baseline
              - (without_A, with_B) = factor B only
              - (with_A, without_B) = factor A only
              - (with_A, with_B) = both factors
        step: Step at which to compute effects. Default: max common step.

    Returns:
        Dict with main effects, interaction terms, and classification.
    """
    if len(factors) != 2:
        return {"error": "factorial_analysis requires exactly 2 factors"}

    factor_names = list(factors.keys())
    a_name, b_name = factor_names[0], factor_names[1]
    a_without, a_with = factors[a_name]
    b_without, b_with = factors[b_name]

    # Identify the 4 cells
    # baseline = a_without AND b_without
    # For a 2x2 with factors A and B:
    #   cell_00 = neither (baseline)
    #   cell_10 = A only
    #   cell_01 = B only
    #   cell_11 = both A and B
    cell_00 = a_without  # baseline
    cell_10 = a_with     # A only
    cell_01 = b_with     # B only
    # The fourth cell: need to find the run that has both factors
    # This is the run that is NOT cell_00, cell_10, or cell_01
    remaining = set(datasets.keys()) - {cell_00, cell_10, cell_01}
    if not remaining:
        return {"error": f"Cannot identify 2×2 fourth cell from runs: {list(datasets.keys())}"}
    cell_11 = next(iter(remaining))

    for cell_name in [cell_00, cell_10, cell_01, cell_11]:
        if cell_name not in datasets:
            return {"error": f"Run '{cell_name}' not found in datasets"}

    # Determine step
    if step is None:
        common_steps = set.intersection(
            *(set(d.keys()) for d in datasets.values())
        )
        step = max(common_steps) if common_steps else max(
            max(d.keys()) for d in datasets.values()
        )

    # Discover all numeric metric keys at this step
    all_keys: set[str] = set()
    for name in [cell_00, cell_10, cell_01, cell_11]:
        s = _nearest_step(datasets[name], step)
        if s is not None and s in datasets[name]:
            all_keys.update(datasets[name][s].keys())

    # Compute effects for each metric
    effects: dict[str, dict[str, Any]] = {}
    for key in sorted(all_keys):
        v00 = _get_at_nearest(datasets[cell_00], step, key)
        v10 = _get_at_nearest(datasets[cell_10], step, key)
        v01 = _get_at_nearest(datasets[cell_01], step, key)
        v11 = _get_at_nearest(datasets[cell_11], step, key)

        if any(v is None for v in [v00, v10, v01, v11]):
            continue

        a_main = ((v10 + v11) / 2) - ((v00 + v01) / 2)
        b_main = ((v01 + v11) / 2) - ((v00 + v10) / 2)

        # Interaction: observed - expected under additivity
        a_alone = v10 - v00
        b_alone = v01 - v00
        expected = v00 + a_alone + b_alone
        interaction = v11 - expected

        # Classify
        if abs(interaction) < abs(a_main + b_main) * 0.05:
            classification = "additive"
        elif interaction > 0:
            classification = "super-additive"
        else:
            classification = "sub-additive"

        effects[key] = {
            "values": {cell_00: v00, cell_10: v10, cell_01: v01, cell_11: v11},
            f"{a_name}_main": a_main,
            f"{b_name}_main": b_main,
            "interaction": interaction,
            "classification": classification,
        }

    return {
        "step": step,
        "factors": {a_name: [a_without, a_with], b_name: [b_without, b_with]},
        "cells": {
            "baseline": cell_00,
            f"{a_name}_only": cell_10,
            f"{b_name}_only": cell_01,
            "both": cell_11,
        },
        "effects": effects,
    }


def _get_at_nearest(data: StepData, target: int, key: str) -> float | None:
    """Get a metric value at the nearest step to target."""
    step = _nearest_step(data, target)
    if step is None:
        return None
    return get_value(data, step, key)


# =============================================================================
# F. Reference Trajectory Comparison
# =============================================================================


def reference_comparison(
    data: StepData,
    reference: dict[str, Any],
    threshold_pct: float = 10.0,
) -> dict[str, Any]:
    """Compare current run against a reference summary.

    Consolidates analyze_lang_diagnostics.py pattern.

    Args:
        data: Step-indexed metrics for current run.
        reference: Reference summary dict (output of endpoint_summary or
            a manually constructed reference with metric values).
        threshold_pct: Flag metrics deviating more than this percentage.

    Returns:
        Dict with matched comparisons and flagged deviations.
    """
    current = endpoint_summary(data)

    comparisons: dict[str, dict[str, Any]] = {}
    flagged: list[dict[str, Any]] = []

    # Compare all numeric fields present in both
    for key in reference:
        ref_val = reference[key]
        cur_val = current.get(key)

        if not isinstance(ref_val, (int, float)) or ref_val is None:
            continue
        if cur_val is None:
            continue

        delta = cur_val - ref_val
        if abs(ref_val) > 1e-10:
            pct_change = (delta / abs(ref_val)) * 100
        else:
            pct_change = 0.0

        entry = {
            "reference": ref_val,
            "current": cur_val,
            "delta": delta,
            "pct_change": round(pct_change, 2),
        }
        comparisons[key] = entry

        if abs(pct_change) > threshold_pct:
            flagged.append({"metric": key, **entry})

    return {
        "comparisons": comparisons,
        "flagged": flagged,
        "threshold_pct": threshold_pct,
    }


# =============================================================================
# Helpers
# =============================================================================


def _std(values: list[float]) -> float:
    """Sample standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance**0.5
