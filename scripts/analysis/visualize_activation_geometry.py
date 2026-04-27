#!/usr/bin/env python3
"""Unified activation geometry visualization (Track 5b).

Produces plots from Track 5a NPZ outputs. CPU-only.

Replaces: visualize_shapes.py, visualize_shapes_v2.py.

Usage::

    python -m scripts.visualize_activation_geometry --input analysis/my-run/activation_geometry/
    python -m scripts.visualize_activation_geometry --input analysis/activation_geometry/ --only pca,eigenspectra
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.plot_utils import get_abbrev, get_color, get_marker, setup_style

logger = logging.getLogger(__name__)

PLOT_TYPES = {
    "pca": "PCA of point clouds (2D projections)",
    "eigenspectra": "Singular value spectra per weight type",
    "trajectories": "Layer trajectories (PCA + distance)",
    "attention": "Attention weight heatmaps",
    "head_entropy": "Per-head entropy heatmap + profile",
    "effective_rank": "Participation ratio per weight type",
    "procrustes": "Procrustes-aligned cross-run overlays",
}


def plot_pca(input_dir: Path, plot_dir: Path) -> None:
    """PCA scatter plots of mean-pooled point clouds."""
    pc_path = input_dir / "point_clouds.npz"
    if not pc_path.exists():
        logger.warning("Skipping PCA: point_clouds.npz not found")
        return

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    data = np.load(pc_path, allow_pickle=True)
    run_names = list(data["run_names"])
    n_layers = int(data["n_layers"][0])

    # Plot embedding + 3 evenly spaced layers + last layer
    plot_layers = sorted(set([-1, 0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]))

    fig, axes = plt.subplots(1, len(plot_layers), figsize=(5 * len(plot_layers), 4.5))
    if len(plot_layers) == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, plot_layers):
        # Collect all runs for this layer
        all_clouds = []
        labels = []
        for run_name in run_names:
            key = f"{run_name}_layer_{layer_idx}"
            if key in data:
                all_clouds.append(data[key])
                labels.append(run_name)

        if not all_clouds:
            continue

        combined = np.concatenate(all_clouds, axis=0).astype(np.float32)
        pca = PCA(n_components=2)
        projected = pca.fit_transform(combined)

        offset = 0
        for run_name, cloud in zip(labels, all_clouds):
            n = len(cloud)
            pts = projected[offset: offset + n]
            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=get_color(run_name), marker=get_marker(run_name),
                s=8, alpha=0.4, label=get_abbrev(run_name),
            )
            offset += n

        layer_label = "Embed" if layer_idx == -1 else f"L{layer_idx}"
        var = pca.explained_variance_ratio_
        ax.set_title(f"{layer_label} ({var[0]:.0%}+{var[1]:.0%})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    axes[0].legend(fontsize=7, markerscale=2)
    fig.suptitle("Point Cloud PCA Overview", fontsize=13)
    fig.tight_layout()
    fig.savefig(plot_dir / "point_clouds_pca_overview.png")
    plt.close(fig)
    logger.info("  Saved point_clouds_pca_overview.png")


def plot_eigenspectra(input_dir: Path, plot_dir: Path) -> None:
    """Singular value spectra per weight type."""
    es_path = input_dir / "eigenspectra.npz"
    if not es_path.exists():
        logger.warning("Skipping eigenspectra: not found")
        return

    import matplotlib.pyplot as plt

    data = np.load(es_path, allow_pickle=True)
    run_names = list(data["run_names"])
    weight_types = list(data["weight_types"])
    n_layers = int(data["n_layers"][0])

    # One subplot per weight type, showing last layer
    fig, axes = plt.subplots(1, len(weight_types), figsize=(3 * len(weight_types), 3.5))
    if len(weight_types) == 1:
        axes = [axes]

    last_layer = n_layers - 1
    for ax, wtype in zip(axes, weight_types):
        for run_name in run_names:
            key = f"{run_name}_{wtype}_layer_{last_layer}"
            if key in data:
                sv = data[key]
                ax.semilogy(sv, color=get_color(run_name), alpha=0.7,
                            label=get_abbrev(run_name), linewidth=1)
        ax.set_title(wtype, fontsize=9)
        ax.set_xlabel("Index")

    axes[0].set_ylabel("Singular Value")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"Eigenspectra (Layer {last_layer})", fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_dir / "eigenspectra_last_layer.png")
    plt.close(fig)
    logger.info("  Saved eigenspectra_last_layer.png")


def plot_trajectories(input_dir: Path, plot_dir: Path) -> None:
    """Layer trajectory distance and PCA plots."""
    tr_path = input_dir / "trajectories.npz"
    if not tr_path.exists():
        logger.warning("Skipping trajectories: not found")
        return

    import matplotlib.pyplot as plt

    data = np.load(tr_path, allow_pickle=True)
    run_names = list(data["run_names"])
    n_prompts = int(data["n_prompts"][0])

    # Plot L2 distance from embedding for prompt 0
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_name in run_names:
        key = f"{run_name}_prompt_0"
        if key not in data:
            continue
        traj = data[key]  # (n_layers+1, hidden)
        embedding = traj[0]
        distances = [np.linalg.norm(traj[i] - embedding) for i in range(len(traj))]
        ax.plot(distances, color=get_color(run_name), label=get_abbrev(run_name), linewidth=1.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Distance from Embedding")
    ax.set_title("Trajectory Distance (Prompt 0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "trajectories_distance.png")
    plt.close(fig)
    logger.info("  Saved trajectories_distance.png")


def plot_head_entropy(input_dir: Path, plot_dir: Path) -> None:
    """Per-head entropy heatmap and profile."""
    he_path = input_dir / "head_entropy.npz"
    if not he_path.exists():
        logger.warning("Skipping head entropy: not found")
        return

    import matplotlib.pyplot as plt

    data = np.load(he_path, allow_pickle=True)
    run_names = list(data["run_names"])
    n_layers = int(data["n_layers"][0])
    n_heads = int(data["n_heads"][0])

    for run_name in run_names:
        matrix = np.zeros((n_layers, n_heads))
        for layer in range(n_layers):
            key = f"{run_name}_layer_{layer}"
            if key in data:
                matrix[layer] = data[key]

        fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.4), max(4, n_layers * 0.2)))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"Head Entropy: {get_abbrev(run_name)}")
        fig.colorbar(im, ax=ax, label="Entropy (nats)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"head_entropy_{run_name}.png")
        plt.close(fig)
        logger.info("  Saved head_entropy_%s.png", run_name)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Activation geometry visualization (Track 5b)")
    p.add_argument("--input", type=Path, required=True, help="Directory with Track 5a NPZ files")
    p.add_argument("--only", type=str, default=None,
                   help=f"Comma-separated plot types: {list(PLOT_TYPES.keys())}")
    p.add_argument("--style", type=str, default="publication",
                   choices=["publication", "dark", "notebook"])
    args = p.parse_args()

    if not args.input.exists():
        logger.error("Input directory not found: %s", args.input)
        sys.exit(1)

    setup_style(args.style)

    plot_dir = args.input / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_types = set(PLOT_TYPES.keys())
    if args.only:
        plot_types = set(args.only.split(","))

    dispatchers = {
        "pca": plot_pca,
        "eigenspectra": plot_eigenspectra,
        "trajectories": plot_trajectories,
        "head_entropy": plot_head_entropy,
    }

    for ptype in sorted(plot_types):
        if ptype in dispatchers:
            logger.info("Plotting: %s", ptype)
            dispatchers[ptype](args.input, plot_dir)
        elif ptype in PLOT_TYPES:
            logger.info("Skipping %s (not yet implemented)", ptype)

    logger.info("Done. Plots in %s", plot_dir)


if __name__ == "__main__":
    main()
