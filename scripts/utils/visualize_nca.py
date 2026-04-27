"""
Visualize NCA trajectories for qualitative inspection.

Generates a few NCA rules with the same pipeline as the data generator,
then creates grid visualizations showing the temporal evolution of each
channel. Outputs PNG files showing:

1. Multi-channel grid snapshots at key timesteps
2. Full trajectory strips (time → horizontal, space → vertical)
3. Complexity distribution across a batch of rules

Usage::

    # Generate visualizations from fresh rules
    python scripts/visualize_nca.py --device cuda:0

    # Visualize from existing trajectory data
    python scripts/visualize_nca.py --from-data data/nca_trajectories.bin
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# We need matplotlib but it may not be installed on the server
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def make_colormap(n_states: int) -> "ListedColormap":
    """Create a discrete colormap for cell states."""
    if n_states <= 10:
        # Hand-picked distinctive colors for up to 10 states
        colors = [
            "#1a1a2e", "#16213e", "#0f3460", "#e94560", "#533483",
            "#00b4d8", "#06d6a0", "#ffd166", "#ef476f", "#118ab2",
        ][:n_states]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, n_states))
    return ListedColormap(colors)


def visualize_trajectory(
    trajectory: np.ndarray,
    title: str,
    output_path: str,
    d_state: int = 10,
    timesteps_to_show: list[int] | None = None,
) -> None:
    """
    Visualize a single multi-channel NCA trajectory.

    trajectory: (T, G, H, W) integer cell states
    """
    T, G, H, W = trajectory.shape
    cmap = make_colormap(d_state)

    if timesteps_to_show is None:
        # Show ~8 evenly spaced timesteps
        n_show = min(8, T)
        timesteps_to_show = [int(i * (T - 1) / (n_show - 1)) for i in range(n_show)]

    n_times = len(timesteps_to_show)

    fig, axes = plt.subplots(
        G, n_times,
        figsize=(n_times * 1.8, G * 1.8),
        squeeze=False,
    )

    for col, t in enumerate(timesteps_to_show):
        for row in range(G):
            ax = axes[row, col]
            ax.imshow(
                trajectory[t, row],
                cmap=cmap, vmin=0, vmax=d_state - 1,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t={t}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"ch{row}", fontsize=9)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path)


def visualize_trajectory_strip(
    trajectory: np.ndarray,
    title: str,
    output_path: str,
    d_state: int = 10,
    row_idx: int = None,
) -> None:
    """
    Visualize trajectory as a spacetime strip — one row of the grid
    across all timesteps. Time flows left→right, space is vertical.

    trajectory: (T, G, H, W)
    """
    T, G, H, W = trajectory.shape
    if row_idx is None:
        row_idx = H // 2  # middle row

    cmap = make_colormap(d_state)

    fig, axes = plt.subplots(G, 1, figsize=(min(T * 0.08, 20), G * 2), squeeze=False)

    for g in range(G):
        # Extract one row across time: (T, W)
        strip = trajectory[:, g, row_idx, :]  # (T, W)
        ax = axes[g, 0]
        ax.imshow(
            strip.T, cmap=cmap, vmin=0, vmax=d_state - 1,
            interpolation="nearest", aspect="auto",
        )
        ax.set_ylabel(f"ch{g}", fontsize=9)
        ax.set_xticks([0, T // 4, T // 2, 3 * T // 4, T - 1])
        ax.set_xticklabels([0, T // 4, T // 2, 3 * T // 4, T - 1], fontsize=7)
        if g == G - 1:
            ax.set_xlabel("time step", fontsize=9)

    fig.suptitle(f"{title} — spacetime strip (row {row_idx})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", output_path)


def visualize_from_generator(
    output_dir: str,
    device: str = "cpu",
    n_rules: int = 6,
    seed: int = 42,
) -> None:
    """Generate a few NCA rules and visualize their trajectories."""
    import torch
    sys.path.insert(0, ".")
    from src.nca.generator import (
        NCAConfig, NCARule, sample_rule_config,
        simulate_trajectory, evaluate_rule_complexity,
    )

    random.seed(seed)
    torch.manual_seed(seed)

    config = NCAConfig(
        grid_size=32,
        d_state=10,
        n_groups=4,
        num_steps=128,
        burn_in=10,
        mixed_complexity=True,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)

    # Generate rules with different complexity levels
    rules_viz = []
    candidates = 0
    complexities = []

    logger.info("Generating rules for visualization...")
    while len(rules_viz) < n_rules:
        params = sample_rule_config(config)
        rule = NCARule(
            d_state=config.d_state,
            n_groups=config.n_groups,
            kernel_size=params["kernel_size"],
            hidden_dim=params["hidden_dim"],
            num_hidden_layers=params["num_hidden_layers"],
        ).to(dev)
        candidates += 1

        ratio = evaluate_rule_complexity(rule, config, params, dev)
        complexities.append(ratio)

        if config.gzip_lower <= ratio <= config.gzip_upper:
            rules_viz.append((rule, params, ratio))
            logger.info(
                "  Rule %d: kernel=%d, depth=%d, gzip=%.3f",
                len(rules_viz), params["kernel_size"],
                params["num_hidden_layers"], ratio,
            )

    # Simulate and visualize each accepted rule
    for i, (rule, params, ratio) in enumerate(rules_viz):
        traj = simulate_trajectory(
            rule=rule,
            grid_size=config.grid_size,
            d_state=config.d_state,
            n_groups=config.n_groups,
            num_steps=config.num_steps,
            burn_in=config.burn_in,
            identity_bias=params["identity_bias"],
            temperature=params["temperature"],
            batch_size=1,
            device=dev,
        )
        traj_np = traj[0].cpu().numpy()  # (T, G, H, W)

        title = (
            f"Rule {i+1}: kernel={params['kernel_size']}, "
            f"depth={params['num_hidden_layers']}, "
            f"gzip={ratio:.3f}"
        )

        # Grid snapshots
        visualize_trajectory(
            traj_np, title,
            str(out / f"nca_rule_{i+1}_grid.png"),
            d_state=config.d_state,
        )

        # Spacetime strip
        visualize_trajectory_strip(
            traj_np, title,
            str(out / f"nca_rule_{i+1}_strip.png"),
            d_state=config.d_state,
        )

    # Complexity distribution plot
    if len(complexities) > 10:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(complexities, bins=50, color="#0f3460", alpha=0.7, edgecolor="white")
        ax.axvspan(config.gzip_lower, config.gzip_upper, alpha=0.2, color="#06d6a0",
                   label=f"Accepted range [{config.gzip_lower}, {config.gzip_upper}]")
        ax.set_xlabel("Gzip compression ratio")
        ax.set_ylabel("Count")
        ax.set_title(
            f"NCA Rule Complexity Distribution "
            f"({len(rules_viz)} accepted / {candidates} tested, "
            f"{100*len(rules_viz)/candidates:.0f}%)"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(out / "complexity_distribution.png"), dpi=150)
        plt.close()
        logger.info("Saved: %s", out / "complexity_distribution.png")

    logger.info("All visualizations saved to %s", out)


def visualize_from_data(
    data_path: str,
    output_dir: str,
    d_state: int = 10,
    n_groups: int = 4,
    grid_size: int = 32,
    patch_size: int = 2,
    n_trajectories: int = 4,
) -> None:
    """Decode and visualize trajectories from a binary token file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokens = np.memmap(data_path, dtype=np.uint16, mode="r")
    logger.info("Loaded %d tokens from %s", len(tokens), data_path)

    START_TOKEN = d_state ** (patch_size ** 2)
    END_TOKEN = START_TOKEN + 1
    nph = grid_size // patch_size
    npw = grid_size // patch_size
    patches_per_frame = nph * npw
    tokens_per_frame = patches_per_frame + 2  # START + patches + END
    tokens_per_timestep = tokens_per_frame * n_groups
    num_steps = 128  # default

    tokens_per_traj = tokens_per_timestep * num_steps

    for traj_idx in range(min(n_trajectories, len(tokens) // tokens_per_traj)):
        offset = traj_idx * tokens_per_traj
        traj_tokens = tokens[offset:offset + tokens_per_traj]

        # Decode back to grid states
        trajectory = np.zeros((num_steps, n_groups, grid_size, grid_size), dtype=np.int64)

        pos = 0
        for t in range(num_steps):
            for g in range(n_groups):
                # Skip START token
                if traj_tokens[pos] != START_TOKEN:
                    logger.warning("Expected START at pos %d, got %d", pos, traj_tokens[pos])
                pos += 1

                # Read patches
                for ph in range(nph):
                    for pw in range(npw):
                        token = int(traj_tokens[pos])
                        # Decode patch: token = sum(val_i * d^i)
                        patch = np.zeros(patch_size * patch_size, dtype=np.int64)
                        for i in range(patch_size * patch_size):
                            patch[i] = token % d_state
                            token //= d_state
                        trajectory[t, g, ph*patch_size:(ph+1)*patch_size,
                                   pw*patch_size:(pw+1)*patch_size] = patch.reshape(patch_size, patch_size)
                        pos += 1

                # Skip END token
                pos += 1

        title = f"Trajectory {traj_idx + 1} (from data)"
        visualize_trajectory(
            trajectory, title,
            str(out / f"data_traj_{traj_idx+1}_grid.png"),
            d_state=d_state,
        )
        visualize_trajectory_strip(
            trajectory, title,
            str(out / f"data_traj_{traj_idx+1}_strip.png"),
            d_state=d_state,
        )

    logger.info("All visualizations saved to %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(description="Visualize NCA trajectories")
    p.add_argument("--output-dir", default="outputs/nca_viz")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-rules", type=int, default=6)
    p.add_argument("--from-data", type=str, default=None,
                   help="Visualize from existing binary data file")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not HAS_MPL:
        print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
        sys.exit(1)

    if args.from_data:
        visualize_from_data(args.from_data, args.output_dir)
    else:
        visualize_from_generator(args.output_dir, args.device, args.n_rules, args.seed)


if __name__ == "__main__":
    main()
