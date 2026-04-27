#!/usr/bin/env python3
"""Unified concept geometry visualization (Track 6c).

Produces plots from Track 6a activations and Track 6b analysis results. CPU-only.

Replaces: plot_manifolds_paper_v3.py, plot_manifolds_paper_v3_extra.py.

Usage::

    python -m scripts.visualize_concept_geometry --input analysis/concept_geometry/
    python -m scripts.visualize_concept_geometry --input analysis/concept_geometry/ --only mantel,pca
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.plot_utils import get_abbrev, get_color, setup_style

logger = logging.getLogger(__name__)

PLOT_TYPES = {
    "mantel": "Mantel rho profiles across layers",
    "pca": "PCA scatter of concept embeddings",
    "sensitivity": "Template sensitivity heatmap",
    "spectral": "Spectral entropy profiles",
    "subspace": "PC1/PC2 ratio and participation ratio",
}


def plot_mantel_profiles(
    results: dict[str, Any],
    meta: dict[str, Any],
    plot_dir: Path,
) -> None:
    """Mantel rho across layers for cyclic and ordinal concept sets."""
    import matplotlib.pyplot as plt

    for topology in ["cyclic", "ordinal"]:
        topo_data = results.get(topology, {})
        if not topo_data:
            continue

        for cs_name, cs_data in topo_data.items():
            fig, ax = plt.subplots(figsize=(10, 5))

            for ckpt, ckpt_data in cs_data.items():
                layers_data = ckpt_data.get("layers", {})
                layer_indices = sorted(int(k) for k in layers_data.keys())
                rhos = [layers_data[str(l)]["mantel_rho"] for l in layer_indices]
                ax.plot(layer_indices, rhos, color=get_color(ckpt),
                        label=get_abbrev(ckpt), linewidth=1.5, marker=".", markersize=3)

            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Layer (0 = embedding)")
            ax.set_ylabel("Mantel ρ (Spearman)")
            ax.set_title(f"{cs_name} ({topology}) — Structure Preservation")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / f"mantel_{cs_name}.png")
            plt.close(fig)
            logger.info("  Saved mantel_%s.png", cs_name)


def plot_pca_concepts(
    data: dict[str, np.ndarray],
    meta: dict[str, Any],
    plot_dir: Path,
) -> None:
    """PCA scatter of concept embeddings at key layers."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    checkpoints = list(meta.get("checkpoints", {}).keys())
    concept_sets_meta = meta.get("concept_sets", {})
    n_layers = int(data.get("n_layers", [28])[0])

    # Plot at layers: embedding (0), mid, last
    plot_layers = [0, n_layers // 2, n_layers]

    for cs_name, cs_info in concept_sets_meta.items():
        concepts = cs_info["concepts"]
        n_concepts = len(concepts)

        for ckpt in checkpoints[:3]:  # Limit to 3 checkpoints for readability
            key = f"{ckpt}_{cs_name}_activations"
            cidx_key = f"{ckpt}_{cs_name}_concept_idx"
            tidx_key = f"{ckpt}_{cs_name}_template_idx"

            if key not in data:
                continue

            acts = data[key]
            cidx = data[cidx_key]

            # Template-average
            avg = np.zeros((n_concepts, acts.shape[1], acts.shape[2]), dtype=np.float64)
            counts = np.zeros(n_concepts)
            for i in range(len(cidx)):
                avg[cidx[i]] += acts[i]
                counts[cidx[i]] += 1
            for c in range(n_concepts):
                if counts[c] > 0:
                    avg[c] /= counts[c]

            fig, axes = plt.subplots(1, len(plot_layers), figsize=(5 * len(plot_layers), 4.5))

            for ax, layer in zip(axes, plot_layers):
                layer_acts = avg[:, layer, :]
                if layer_acts.shape[0] < 3:
                    continue

                pca = PCA(n_components=2)
                projected = pca.fit_transform(layer_acts)

                # Color by ordinal position
                colors = np.arange(n_concepts)
                sc = ax.scatter(projected[:, 0], projected[:, 1],
                                c=colors, cmap="hsv" if cs_info["topology"] == "cyclic" else "viridis",
                                s=40, edgecolors="k", linewidths=0.3)

                # Label points
                for i, concept in enumerate(concepts):
                    label = concept[:4] if len(concept) > 4 else concept
                    ax.annotate(label, (projected[i, 0], projected[i, 1]),
                                fontsize=5, ha="center", va="bottom")

                layer_label = "Embed" if layer == 0 else f"L{layer - 1}"
                var = pca.explained_variance_ratio_
                ax.set_title(f"{layer_label} ({var[0]:.0%}+{var[1]:.0%})")

            fig.suptitle(f"{cs_name} — {get_abbrev(ckpt)}", fontsize=12)
            fig.tight_layout()
            fig.savefig(plot_dir / f"pca_{cs_name}_{ckpt}.png")
            plt.close(fig)
            logger.info("  Saved pca_%s_%s.png", cs_name, ckpt)


def plot_template_sensitivity(
    results: dict[str, Any],
    meta: dict[str, Any],
    plot_dir: Path,
) -> None:
    """Template sensitivity across layers as line plots."""
    import matplotlib.pyplot as plt

    ts_data = results.get("template_sensitivity", {})
    if not ts_data:
        return

    for cs_name, cs_data in ts_data.items():
        fig, ax = plt.subplots(figsize=(10, 5))

        for ckpt, ckpt_data in cs_data.items():
            layers_data = ckpt_data.get("layers", {})
            layer_indices = sorted(int(k) for k in layers_data.keys())
            values = [layers_data[str(l)] for l in layer_indices]
            ax.plot(layer_indices, values, color=get_color(ckpt),
                    label=get_abbrev(ckpt), linewidth=1.5)

        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Cosine Distance")
        ax.set_title(f"{cs_name} — Template Sensitivity")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"template_sensitivity_{cs_name}.png")
        plt.close(fig)
        logger.info("  Saved template_sensitivity_%s.png", cs_name)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Concept geometry visualization (Track 6c)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--only", type=str, default=None,
                   help=f"Comma-separated: {list(PLOT_TYPES.keys())}")
    p.add_argument("--style", type=str, default="publication")
    args = p.parse_args()

    if not args.input.exists():
        logger.error("Input directory not found: %s", args.input)
        sys.exit(1)

    setup_style(args.style)

    plot_dir = args.input / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    npz_path = args.input / "activations.npz"
    meta_path = args.input / "metadata.json"
    results_path = args.input / "results.json"

    data = dict(np.load(npz_path, allow_pickle=True)) if npz_path.exists() else {}
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    results = json.load(open(results_path)) if results_path.exists() else {}

    plot_types = set(PLOT_TYPES.keys())
    if args.only:
        plot_types = set(args.only.split(","))

    if "mantel" in plot_types and results:
        logger.info("Plotting: mantel profiles")
        plot_mantel_profiles(results, meta, plot_dir)

    if "pca" in plot_types and data:
        logger.info("Plotting: PCA concept embeddings")
        plot_pca_concepts(data, meta, plot_dir)

    if "sensitivity" in plot_types and results:
        logger.info("Plotting: template sensitivity")
        plot_template_sensitivity(results, meta, plot_dir)

    logger.info("Done. Plots in %s", plot_dir)


if __name__ == "__main__":
    main()
