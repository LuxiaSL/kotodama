"""Visualization defaults for analysis pipeline plots.

Consolidates color schemes, marker styles, abbreviation maps, and style
setup from visualization scripts into one module.
"""

from __future__ import annotations

from typing import Any

# Checkpoint color scheme — consistent across all plots
CHECKPOINT_COLORS: dict[str, str] = {
    # Proxy sweep
    "P1-AdamW": "#7f7f7f",       # gray
    "P3-Muon-002": "#1f77b4",    # blue
    "P4-Muon-003": "#aec7e8",    # light blue
    # NCA proxy
    "NCA-002": "#2ca02c",        # green
    "NCA-003": "#98df8a",         # light green
    # AttnRes sweep
    "P3-AttnRes": "#ff7f0e",     # orange
    "NCA-AttnRes": "#d62728",    # red
    # Language full
    "Lang-Baseline": "#9467bd",  # purple
    "Lang-DDv1": "#e377c2",      # pink
    # Fallback palette for dynamic checkpoint lists
    "_fallback": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ],
}

# Marker styles per group
CHECKPOINT_MARKERS: dict[str, str] = {
    "P1-AdamW": "s",        # square
    "P3-Muon-002": "o",     # circle
    "P4-Muon-003": "^",     # triangle up
    "NCA-002": "D",          # diamond
    "NCA-003": "v",          # triangle down
    "P3-AttnRes": "P",      # plus (filled)
    "NCA-AttnRes": "*",     # star
    "Lang-Baseline": "X",   # x (filled)
    "Lang-DDv1": "H",       # hexagon
}

# Short names for compact labels
CHECKPOINT_ABBREVS: dict[str, str] = {
    "P1-AdamW": "P1-AW",
    "P3-Muon-002": "P3",
    "P4-Muon-003": "P4",
    "NCA-002": "NCA",
    "NCA-003": "NCA-3",
    "P3-AttnRes": "P3-AR",
    "NCA-AttnRes": "NCA-AR",
    "Lang-Baseline": "Baseline",
    "Lang-DDv1": "DD-v1",
}

# Weight projection type labels
WEIGHT_TYPE_LABELS: dict[str, str] = {
    "q_proj": "Q",
    "k_proj": "K",
    "v_proj": "V",
    "o_proj": "O",
    "gate_proj": "Gate",
    "up_proj": "Up",
    "down_proj": "Down",
}


def get_color(name: str) -> str:
    """Get the color for a checkpoint name, with fallback."""
    if name in CHECKPOINT_COLORS:
        return CHECKPOINT_COLORS[name]
    # Deterministic fallback based on name hash
    fallback = CHECKPOINT_COLORS["_fallback"]
    idx = hash(name) % len(fallback)
    return fallback[idx]


def get_marker(name: str) -> str:
    """Get the marker for a checkpoint name, with fallback."""
    return CHECKPOINT_MARKERS.get(name, "o")


def get_abbrev(name: str) -> str:
    """Get the short name for a checkpoint, with fallback to original."""
    return CHECKPOINT_ABBREVS.get(name, name)


def setup_style(style: str = "publication") -> None:
    """Set up matplotlib style for the target context.

    Args:
        style: One of "publication" (white bg, serif), "dark" (dark_background),
            "notebook" (larger fonts).
    """
    import matplotlib.pyplot as plt

    if style == "dark":
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    # Common tweaks
    params: dict[str, Any] = {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }

    if style == "publication":
        params.update({
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.figsize": (8, 5),
        })
    elif style == "notebook":
        params.update({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "figure.figsize": (10, 6),
        })

    plt.rcParams.update(params)
