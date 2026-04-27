"""JSONL metrics loading and querying utilities.

Consolidates metrics loading patterns from interaction_analysis_v2.py,
analyze_dynamics.py, analyze_lang_diagnostics.py, and other analysis scripts.

Handles line-by-line JSONL parsing, step-based merging, nearest-step
lookup, and time-series extraction.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def load_metrics(
    path: Path | str,
    merge: bool = True,
) -> dict[int, dict[str, float]]:
    """Load a JSONL metrics file into a step-indexed dict.

    Each line is a JSON object with at least a "step" field. When merge=True
    (default), multiple records for the same step are merged into a single
    dict (later values overwrite earlier ones for the same key).

    Args:
        path: Path to the .jsonl metrics file.
        merge: If True, merge multiple records per step. If False, only the
            last record per step is kept.

    Returns:
        Dict mapping step number → metric dict.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    by_step: dict[int, dict[str, float]] = {}
    n_lines = 0
    n_errors = 0

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_errors += 1
                if n_errors <= 5:
                    logger.warning("Malformed JSON at line %d in %s", line_num, path)
                continue

            n_lines += 1
            step = rec.pop("step", None)
            if step is None:
                continue
            step = int(step)

            # Remove non-metric fields
            rec.pop("timestamp", None)

            if merge and step in by_step:
                by_step[step].update(rec)
            else:
                by_step[step] = dict(rec)

    if n_errors > 5:
        logger.warning("Total malformed lines in %s: %d", path, n_errors)

    logger.info(
        "Loaded %s: %d steps from %d lines (max step %d)",
        path.name,
        len(by_step),
        n_lines,
        max(by_step.keys()) if by_step else -1,
    )

    return by_step


def get_nearest_step(
    data: dict[int, dict[str, float]],
    target: int,
    tolerance: int = 500,
) -> int | None:
    """Find the step closest to target within tolerance.

    Args:
        data: Step-indexed metrics dict.
        target: Target step number.
        tolerance: Maximum allowed distance from target.

    Returns:
        The nearest step, or None if no step is within tolerance.
    """
    if not data:
        return None

    if target in data:
        return target

    best_step = None
    best_dist = tolerance + 1

    for step in data:
        dist = abs(step - target)
        if dist < best_dist:
            best_dist = dist
            best_step = step

    return best_step if best_dist <= tolerance else None


def extract_series(
    data: dict[int, dict[str, float]],
    key: str,
) -> list[tuple[int, float]]:
    """Extract a time series for a specific metric key.

    Returns sorted (step, value) pairs for all steps where the key exists.

    Args:
        data: Step-indexed metrics dict.
        key: The metric key to extract (e.g., "train/loss", "geo/rankme_last").

    Returns:
        List of (step, value) tuples, sorted by step.
    """
    series = []
    for step in sorted(data.keys()):
        if key in data[step]:
            series.append((step, data[step][key]))
    return series


def extract_keys_matching(
    data: dict[int, dict[str, float]],
    pattern: str,
) -> list[str]:
    """Find all metric keys matching a regex pattern.

    Searches across all steps to find the union of matching keys.

    Args:
        data: Step-indexed metrics dict.
        pattern: Regular expression to match against key names.

    Returns:
        Sorted list of unique matching keys.
    """
    compiled = re.compile(pattern)
    matching: set[str] = set()
    for step_data in data.values():
        for key in step_data:
            if compiled.search(key):
                matching.add(key)
    return sorted(matching)


def get_value(
    data: dict[int, dict[str, float]],
    step: int,
    key: str,
) -> float | None:
    """Get a single metric value at a specific step.

    Convenience function for interactive analysis.

    Args:
        data: Step-indexed metrics dict.
        step: Step number.
        key: Metric key.

    Returns:
        The value, or None if step or key is missing.
    """
    if step in data and key in data[step]:
        return data[step][key]
    return None


def load_multiple(
    paths: dict[str, Path | str],
    merge: bool = True,
) -> dict[str, dict[int, dict[str, float]]]:
    """Load multiple metrics files into a named dict.

    Args:
        paths: Mapping of run name → JSONL file path.
        merge: Whether to merge records per step.

    Returns:
        Dict mapping run name → step-indexed metrics.
    """
    result: dict[str, dict[int, dict[str, float]]] = {}
    for name, path in paths.items():
        try:
            result[name] = load_metrics(path, merge=merge)
        except FileNotFoundError:
            logger.warning("Skipping %s: file not found at %s", name, path)
    return result
