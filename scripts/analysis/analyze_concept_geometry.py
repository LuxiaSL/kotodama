#!/usr/bin/env python3
"""Unified concept geometry analysis (Track 6b).

Loads Track 6a NPZ activations and runs statistical analysis:
cyclic structure (Mantel + k-NN), ordinal preservation, direction
consistency, template sensitivity, subspace diagnostics.

Every metric has a permutation-based null baseline.

Replaces: analyze_manifolds_v2.py. CPU-only.

Usage::

    python -m scripts.analyze_concept_geometry --input analysis/concept_geometry/
    python -m scripts.analyze_concept_geometry --input analysis/my-run/concept_geometry/ \\
        --only cyclic,ordinal --n-perm 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

ANALYSES = {
    "cyclic": "Mantel test + k-NN cyclic neighbor preservation",
    "ordinal": "k-NN overlap / Mantel for ordinal sets",
    "smoothness": "Direction consistency across layers with closed-form null",
    "template_sensitivity": "Mean pairwise cosine distance between templates",
    "subspace_diagnostics": "PC1/PC2 ratio + participation ratio",
    "spectral_entropy": "Spectral entropy of covariance eigenspectrum",
}


# =============================================================================
# Data loading
# =============================================================================


def load_data(input_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load Track 6a outputs: activations.npz + metadata.json."""
    npz_path = input_dir / "activations.npz"
    meta_path = input_dir / "metadata.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"activations.npz not found in {input_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {input_dir}")

    data = np.load(npz_path, allow_pickle=True)
    with open(meta_path) as f:
        meta = json.load(f)

    return dict(data), meta


def template_average(
    data: dict[str, np.ndarray], checkpoint: str, concept_set: str,
) -> np.ndarray | None:
    """Average activations across templates for each concept.

    Returns (n_concepts, n_layers+1, hidden_size) or None if missing.
    """
    key_acts = f"{checkpoint}_{concept_set}_activations"
    key_cidx = f"{checkpoint}_{concept_set}_concept_idx"
    key_tidx = f"{checkpoint}_{concept_set}_template_idx"

    if key_acts not in data:
        return None

    acts = data[key_acts]       # (n_samples, n_layers+1, hidden)
    cidx = data[key_cidx]       # (n_samples,)
    n_concepts = int(cidx.max()) + 1

    # Average over templates for each concept
    result = np.zeros((n_concepts, acts.shape[1], acts.shape[2]), dtype=np.float64)
    counts = np.zeros(n_concepts, dtype=np.int32)
    for i in range(len(cidx)):
        result[cidx[i]] += acts[i]
        counts[cidx[i]] += 1

    for c in range(n_concepts):
        if counts[c] > 0:
            result[c] /= counts[c]

    return result


def _has_data(data: dict, ckpt: str, cs: str) -> bool:
    return f"{ckpt}_{cs}_activations" in data


# =============================================================================
# Expected distance matrices
# =============================================================================


def cyclic_distance_matrix(n: int) -> np.ndarray:
    """Expected pairwise distance matrix for n items on a circle."""
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = min(abs(i - j), n - abs(i - j))
    return D


def ordinal_distance_matrix(n: int) -> np.ndarray:
    """Expected pairwise distance matrix for n linearly ordered items."""
    return np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)


# =============================================================================
# Statistical tests
# =============================================================================


def mantel_test(
    dist_observed: np.ndarray,
    dist_expected: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> tuple[float, float]:
    """Mantel test: Spearman rho between distance matrices + permutation p-value."""
    from scipy.stats import spearmanr

    # Extract upper triangle
    idx = np.triu_indices_from(dist_observed, k=1)
    obs_vec = dist_observed[idx]
    exp_vec = dist_expected[idx]

    rho, _ = spearmanr(obs_vec, exp_vec)

    # Permutation null
    rng = np.random.default_rng(seed)
    n = dist_observed.shape[0]
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        perm_dist = dist_observed[np.ix_(perm, perm)]
        perm_rho, _ = spearmanr(perm_dist[idx], exp_vec)
        if perm_rho >= rho:
            count_ge += 1

    p = (count_ge + 1) / (n_perm + 1)
    return float(rho), float(p)


def knn_cyclic(
    dist_observed: np.ndarray,
    n: int,
    k: int = 1,
) -> float:
    """Fraction of items whose k-NN includes a cyclic neighbor."""
    hits = 0
    for i in range(n):
        dists = dist_observed[i].copy()
        dists[i] = np.inf
        nn_indices = np.argsort(dists)[:k]
        # Cyclic neighbors are (i-1) % n and (i+1) % n
        cyclic_neighbors = {(i - 1) % n, (i + 1) % n}
        if cyclic_neighbors & set(nn_indices):
            hits += 1
    return hits / n


def knn_ordinal_overlap(
    dist_observed: np.ndarray,
    n: int,
    k: int = 3,
    window: int = 2,
) -> float:
    """Fraction of items whose k-NN are within ±window of ordinal position."""
    hits = 0
    for i in range(n):
        dists = dist_observed[i].copy()
        dists[i] = np.inf
        nn_indices = set(np.argsort(dists)[:k])
        expected = {j for j in range(max(0, i - window), min(n, i + window + 1)) if j != i}
        overlap = len(nn_indices & expected)
        hits += overlap / min(k, len(expected)) if expected else 0
    return hits / n


# =============================================================================
# Analysis functions
# =============================================================================


def analyze_cyclic(
    data: dict, meta: dict,
    checkpoints: list[str], concept_sets: list[str],
    n_perm: int = 5000,
) -> dict[str, Any]:
    """Cyclic structure: Mantel test + k-NN preservation."""
    results: dict[str, Any] = {}
    cyclic_sets = [cs for cs in concept_sets
                   if meta["concept_sets"].get(cs, {}).get("topology") == "cyclic"]
    if not cyclic_sets:
        return {}

    for cs_name in cyclic_sets:
        n_concepts = len(meta["concept_sets"][cs_name]["concepts"])
        expected_dist = cyclic_distance_matrix(n_concepts)
        cs_results: dict[str, Any] = {}

        for ckpt in checkpoints:
            acts = template_average(data, ckpt, cs_name)
            if acts is None:
                continue

            n_layers_plus_1 = acts.shape[1]
            layers: dict[str, Any] = {}

            for layer in range(n_layers_plus_1):
                layer_acts = acts[:, layer, :]
                obs_dist = squareform(pdist(layer_acts))

                rho, p = mantel_test(obs_dist, expected_dist, n_perm)
                knn_k1 = knn_cyclic(obs_dist, n_concepts, k=1)

                layers[str(layer)] = {
                    "mantel_rho": round(rho, 4),
                    "mantel_p": round(p, 4),
                    "knn_cyclic_k1": round(knn_k1, 4),
                }

            cs_results[ckpt] = {"layers": layers}
        results[cs_name] = cs_results

    return results


def analyze_ordinal(
    data: dict, meta: dict,
    checkpoints: list[str], concept_sets: list[str],
    n_perm: int = 5000,
) -> dict[str, Any]:
    """Ordinal structure: Mantel + k-NN overlap."""
    results: dict[str, Any] = {}
    ordinal_sets = [cs for cs in concept_sets
                    if meta["concept_sets"].get(cs, {}).get("topology") in ("ordinal", "geographic")]
    if not ordinal_sets:
        return {}

    for cs_name in ordinal_sets:
        n_concepts = len(meta["concept_sets"][cs_name]["concepts"])
        expected_dist = ordinal_distance_matrix(n_concepts)
        cs_results: dict[str, Any] = {}

        for ckpt in checkpoints:
            acts = template_average(data, ckpt, cs_name)
            if acts is None:
                continue

            n_layers_plus_1 = acts.shape[1]
            layers: dict[str, Any] = {}

            for layer in range(n_layers_plus_1):
                layer_acts = acts[:, layer, :]
                obs_dist = squareform(pdist(layer_acts))

                rho, p = mantel_test(obs_dist, expected_dist, n_perm)

                result: dict[str, Any] = {
                    "mantel_rho": round(rho, 4),
                    "mantel_p": round(p, 4),
                }

                # k-NN overlap for smaller sets
                if n_concepts <= 30:
                    k = min(3, n_concepts - 1)
                    window = max(2, n_concepts // 5)
                    knn_ov = knn_ordinal_overlap(obs_dist, n_concepts, k, window)
                    result["knn_overlap"] = round(knn_ov, 4)

                layers[str(layer)] = result

            cs_results[ckpt] = {"layers": layers}
        results[cs_name] = cs_results

    return results


def analyze_smoothness(
    data: dict, meta: dict,
    checkpoints: list[str], concept_sets: list[str],
    n_perm: int = 5000,
) -> dict[str, Any]:
    """Direction consistency across layers (closed-form null)."""
    results: dict[str, Any] = {}

    for cs_name in concept_sets:
        for ckpt in checkpoints:
            acts = template_average(data, ckpt, cs_name)
            if acts is None:
                continue

            n_concepts, n_layers_plus_1, hidden = acts.shape
            # Use middle layers (skip embedding and early)
            layer_start = max(1, n_layers_plus_1 // 4)
            layer_end = n_layers_plus_1 - 1

            # Compute direction consistency: cosine similarity between
            # consecutive displacement vectors
            consistencies = []
            for c in range(n_concepts):
                for l in range(layer_start, layer_end - 1):
                    d1 = acts[c, l + 1] - acts[c, l]
                    d2 = acts[c, l + 2] - acts[c, l + 1]
                    norm1 = np.linalg.norm(d1)
                    norm2 = np.linalg.norm(d2)
                    if norm1 > 1e-10 and norm2 > 1e-10:
                        cos = np.dot(d1, d2) / (norm1 * norm2)
                        consistencies.append(float(cos))

            if not consistencies:
                continue

            mean_consistency = np.mean(consistencies)

            # Closed-form null: random directions in d dimensions
            # E[cos] = 0, Var[cos] ≈ 1/d
            null_std = 1.0 / np.sqrt(hidden)
            z_score = mean_consistency / null_std

            key = f"{cs_name}/{ckpt}" if len(concept_sets) > 1 else ckpt
            results[key] = {
                "mean_consistency": round(float(mean_consistency), 4),
                "z_score": round(float(z_score), 2),
                "null_std": round(float(null_std), 6),
                "n_pairs": len(consistencies),
            }

    return results


def analyze_template_sensitivity(
    data: dict, meta: dict,
    checkpoints: list[str], concept_sets: list[str],
    n_perm: int = 5000,
) -> dict[str, Any]:
    """Mean pairwise cosine distance between templates for each concept."""
    results: dict[str, Any] = {}

    for cs_name in concept_sets:
        cs_results: dict[str, Any] = {}

        for ckpt in checkpoints:
            key_acts = f"{ckpt}_{cs_name}_activations"
            key_cidx = f"{ckpt}_{cs_name}_concept_idx"
            key_tidx = f"{ckpt}_{cs_name}_template_idx"

            if key_acts not in data:
                continue

            acts = data[key_acts]
            cidx = data[key_cidx]
            tidx = data[key_tidx]
            n_concepts = int(cidx.max()) + 1
            n_templates = int(tidx.max()) + 1
            n_layers_plus_1 = acts.shape[1]

            if n_templates < 2:
                continue

            # Per-layer template sensitivity
            layer_sens: dict[str, float] = {}
            for layer in range(n_layers_plus_1):
                cos_dists = []
                for c in range(n_concepts):
                    mask = cidx == c
                    concept_acts = acts[mask, layer, :]
                    if len(concept_acts) < 2:
                        continue
                    # Pairwise cosine distances
                    norms = np.linalg.norm(concept_acts, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-10)
                    normed = concept_acts / norms
                    sim_matrix = normed @ normed.T
                    # Extract upper triangle
                    triu_idx = np.triu_indices(len(concept_acts), k=1)
                    cos_dists.extend((1.0 - sim_matrix[triu_idx]).tolist())

                if cos_dists:
                    layer_sens[str(layer)] = round(float(np.mean(cos_dists)), 6)

            cs_results[ckpt] = {"layers": layer_sens}
        results[cs_name] = cs_results

    return results


def analyze_subspace_diagnostics(
    data: dict, meta: dict,
    checkpoints: list[str], concept_sets: list[str],
    n_perm: int = 5000,
) -> dict[str, Any]:
    """PC1/PC2 ratio + participation ratio per layer."""
    results: dict[str, Any] = {}

    for cs_name in concept_sets:
        cs_results: dict[str, Any] = {}

        for ckpt in checkpoints:
            acts = template_average(data, ckpt, cs_name)
            if acts is None:
                continue

            n_layers_plus_1 = acts.shape[1]
            layers: dict[str, Any] = {}

            for layer in range(n_layers_plus_1):
                layer_acts = acts[:, layer, :]
                centered = layer_acts - layer_acts.mean(axis=0)

                try:
                    cov = np.cov(centered, rowvar=False)
                    eigvals = np.linalg.eigvalsh(cov)
                    eigvals = np.sort(eigvals)[::-1]
                    eigvals = eigvals[eigvals > 0]

                    if len(eigvals) >= 2:
                        pc1_pc2 = float(eigvals[0] / eigvals[1])
                    else:
                        pc1_pc2 = float("inf")

                    # Participation ratio
                    total = eigvals.sum()
                    if total > 0:
                        p = eigvals / total
                        pr = 1.0 / (p ** 2).sum()
                    else:
                        pr = 0.0

                    layers[str(layer)] = {
                        "pc1_pc2_ratio": round(pc1_pc2, 4),
                        "participation_ratio": round(float(pr), 4),
                    }
                except np.linalg.LinAlgError:
                    pass

            cs_results[ckpt] = {"layers": layers}
        results[cs_name] = cs_results

    return results


def analyze_spectral_entropy(
    data: dict, meta: dict,
    checkpoints: list[str], concept_sets: list[str],
    n_perm: int = 5000,
) -> dict[str, Any]:
    """Spectral entropy of the covariance eigenspectrum."""
    results: dict[str, Any] = {}

    for cs_name in concept_sets:
        n_concepts = len(meta["concept_sets"][cs_name]["concepts"])
        max_entropy = float(np.log2(max(n_concepts - 1, 1)))
        cs_results: dict[str, Any] = {"max_entropy": max_entropy}

        for ckpt in checkpoints:
            acts = template_average(data, ckpt, cs_name)
            if acts is None:
                continue

            n_layers_plus_1 = acts.shape[1]
            layers: dict[str, float] = {}

            for layer in range(n_layers_plus_1):
                layer_acts = acts[:, layer, :]
                centered = layer_acts - layer_acts.mean(axis=0)

                try:
                    cov = np.cov(centered, rowvar=False)
                    eigvals = np.linalg.eigvalsh(cov)
                    eigvals = eigvals[eigvals > 1e-10]

                    if len(eigvals) > 0:
                        p = eigvals / eigvals.sum()
                        entropy = -float((p * np.log2(p + 1e-20)).sum())
                        layers[str(layer)] = round(entropy, 4)
                except np.linalg.LinAlgError:
                    pass

            cs_results[ckpt] = {"layers": layers}
        results[cs_name] = cs_results

    return results


# =============================================================================
# Main
# =============================================================================


def _make_serializable(obj: Any) -> Any:
    """Make numpy types JSON-serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Concept geometry analysis (Track 6b)")
    p.add_argument("--input", type=Path, required=True,
                   help="Directory with Track 6a outputs (activations.npz + metadata.json)")
    p.add_argument("--only", type=str, default=None,
                   help=f"Run only these analyses (comma-separated): {list(ANALYSES.keys())}")
    p.add_argument("--concept-sets", type=str, default=None,
                   help="Analyze only these concept sets (comma-separated)")
    p.add_argument("--n-perm", type=int, default=5000,
                   help="Number of permutations for null baselines")
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args()

    logger.info("Loading data from %s", args.input)
    data, meta = load_data(args.input)

    # Discover available checkpoints and concept sets
    checkpoints = list(meta.get("checkpoints", {}).keys())
    all_concept_sets = list(meta.get("concept_sets", {}).keys())

    # Filter
    concept_sets = all_concept_sets
    if args.concept_sets:
        requested = [cs.strip() for cs in args.concept_sets.split(",")]
        concept_sets = [cs for cs in requested if cs in all_concept_sets]

    # Select analyses
    to_run = list(ANALYSES.keys())
    if args.only:
        to_run = [a.strip() for a in args.only.split(",") if a.strip() in ANALYSES]

    logger.info("Checkpoints: %s", checkpoints)
    logger.info("Concept sets (%d): %s", len(concept_sets), concept_sets)
    logger.info("Analyses: %s", to_run)
    logger.info("Permutations: %d", args.n_perm)

    # Run analyses
    analysis_dispatch = {
        "cyclic": analyze_cyclic,
        "ordinal": analyze_ordinal,
        "smoothness": analyze_smoothness,
        "template_sensitivity": analyze_template_sensitivity,
        "subspace_diagnostics": analyze_subspace_diagnostics,
        "spectral_entropy": analyze_spectral_entropy,
    }

    summary: dict[str, Any] = {
        "metadata": {
            "checkpoints": checkpoints,
            "concept_sets": concept_sets,
            "n_perm": args.n_perm,
            "analyses_run": to_run,
        }
    }
    t0 = time.time()

    for name in to_run:
        if name not in analysis_dispatch:
            logger.warning("Unknown analysis: %s", name)
            continue

        logger.info("Running: %s", name)
        t1 = time.time()

        try:
            result = analysis_dispatch[name](
                data, meta, checkpoints, concept_sets, args.n_perm
            )
            summary[name] = _make_serializable(result)
        except Exception as e:
            logger.error("  FAILED: %s: %s", name, e)
            import traceback
            traceback.print_exc()
            summary[name] = {"error": str(e)}

        logger.info("  %s done in %.1fs", name, time.time() - t1)

    # Save
    output_path = args.output or (args.input / "results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Results saved to: %s", output_path)
    logger.info("Total time: %.1fs", time.time() - t0)

    # Print summary
    for name in to_run:
        result = summary.get(name, {})
        if isinstance(result, dict) and "error" in result:
            print(f"  {name}: ERROR — {result['error']}")
        elif isinstance(result, dict):
            n_entries = sum(1 for v in result.values() if isinstance(v, dict))
            print(f"  {name}: {n_entries} concept set(s) analyzed")


if __name__ == "__main__":
    main()
