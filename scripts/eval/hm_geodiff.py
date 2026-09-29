#!/usr/bin/env python3
"""Geometry diff of two runs from heimdall telemetry: where did the layer-wise picture move?

For every per-layer family (attention entropy, stable ranks, AttnRes routing entropy /
alphas / block norms, norm gains, Jacobian sigma, ...) render run A and run B as
layer x step heatmaps on ONE shared color scale, plus B − A on a diverging scale over
the steps both runs logged. Writes one PNG per family and a text gist (per family:
the layers that moved most at the last common step, and the mean |Δ| relative to the
family's spread) so the read works for humans and agents alike.

Pairs only compare fairly at matched schedules (same warmup / LR at each step); say
so in the read when they do not. ``--max-step`` clips to the span both runs share.

Usage:
  python scripts/eval/hm_geodiff.py --job <A> --job <B> --out-dir /tmp/claude-output/geodiff \
      [--max-step 300] [--family 'geo/layer_{i}/attn_entropy_mean' ...]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hm_portrait as hp  # noqa: E402

Grid = list[list[float]]  # [layer][step]

DEFAULT_FAMILIES = [
    "geo/layer_{i}/attn_entropy_mean",
    "geo/layer_{i}/stable_rank_q_proj",
    "geo/layer_{i}/stable_rank_k_proj",
    "geo/layer_{i}/stable_rank_o_proj",
    "geo/layer_{i}/stable_rank_gate_proj",
    "geo/layer_{i}/stable_rank_down_proj",
    "geo/layer_{i}/anisotropy",
    "geo/twonn_id/layer_{i}",
    "eoc/jacobian_sigma/layer_{i}/attn",
    "eoc/jacobian_sigma/layer_{i}/mlp",
    "attnres/routing_entropy/layer_{i}/pre_attn",
    "attnres/routing_entropy/layer_{i}/pre_mlp",
    "attnres/avg_alpha/block_{i}",
    "attnres/final_alpha/block_{i}",
    "attnres/block_norm/{i}",
    "norm_gain/layer_{i}/q_norm",
    "norm_gain/layer_{i}/k_norm",
    "norm_gain/layer_{i}/attn_norm",
    "norm_gain/layer_{i}/ffn_norm",
    "norm_gain/layer_{i}/attn_res_norm",
    "norm_gain/layer_{i}/mlp_res_norm",
]


# ── pure helpers (tested) ─────────────────────────────────────────────────────


def common_steps(a: dict[str, hp.Series], b: dict[str, hp.Series], max_step: int | None) -> list[int]:
    """Steps at which BOTH runs logged at least one layer of the family."""
    sa = set().union(*(s.keys() for s in a.values())) if a else set()
    sb = set().union(*(s.keys() for s in b.values())) if b else set()
    return sorted(s for s in sa & sb if max_step is None or s <= max_step)


def grid(data: dict[str, hp.Series], keys: list[str], steps: list[int]) -> Grid:
    return [[data[k].get(s, math.nan) for s in steps] for k in keys]


@dataclass(frozen=True)
class FamilyGist:
    family: str
    step: int                      # last common step
    rel_mean_abs_delta: float      # mean |B−A| / (spread of A∪B) at that step
    top: list[tuple[int, float, float]]  # (layer, A, B) for the largest |Δ|


def gist(family: str, layers: list[int], steps: list[int], ga: Grid, gb: Grid, k: int = 4) -> FamilyGist | None:
    """Summarize the last common step with finite values in both runs."""
    for j in range(len(steps) - 1, -1, -1):
        pairs = [(l, ga[r][j], gb[r][j]) for r, l in enumerate(layers)
                 if not math.isnan(ga[r][j]) and not math.isnan(gb[r][j])]
        if pairs:
            vals = [v for _, x, y in pairs for v in (x, y)]
            spread = (max(vals) - min(vals)) or 1e-12
            rel = sum(abs(y - x) for _, x, y in pairs) / len(pairs) / spread
            top = sorted(pairs, key=lambda p: -abs(p[2] - p[1]))[:k]
            return FamilyGist(family, steps[j], rel, top)
    return None


# ── rendering ─────────────────────────────────────────────────────────────────


def slug(family: str) -> str:
    return family.replace("{i}", "i").replace("/", "__")


def plot(family: str, jobs: tuple[str, str], layers: list[int], steps: list[int],
         ga: Grid, gb: Grid, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    a, b = np.array(ga, dtype=float), np.array(gb, dtype=float)
    both = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    vmin, vmax = (float(both.min()), float(both.max())) if both.size else (0.0, 1.0)
    d = b - a
    dm = float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    ext = (steps[0], steps[-1], -0.5, len(layers) - 0.5)
    for ax, g, title, cmap, lo, hi in (
        (axes[0], a, f"A {jobs[0]}", "magma", vmin, vmax),
        (axes[1], b, f"B {jobs[1]}", "magma", vmin, vmax),
        (axes[2], d, "B − A", "RdBu_r", -dm, dm),
    ):
        im = ax.imshow(g, aspect="auto", origin="lower", cmap=cmap, vmin=lo, vmax=hi, extent=ext,
                       interpolation="nearest")
        ax.set_yticks(range(len(layers)), [str(l) for l in layers], fontsize=6)
        ax.set_xlabel("step")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(family, fontsize=11)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", action="append", required=True, help="exactly two heimdall job ids: A then B")
    ap.add_argument("--family", action="append", help="family pattern with {i} (repeatable; default = geometry set)")
    ap.add_argument("--max-step", type=int)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--url", default=os.environ.get(
        "KOTODAMA_SCHEDULER_URL", os.environ.get("HEIMDALL_URL", "http://127.0.0.1:7000")))
    a = ap.parse_args()
    if len(a.job) != 2:
        print("FAIL: pass exactly two --job (A then B)")
        return 1
    base, (ja, jb) = a.url.rstrip("/"), a.job
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        keys_a, keys_b = hp.job_keys(base, ja), hp.job_keys(base, jb)
    except (OSError, KeyError) as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1

    lines = [f"geodiff A={ja} B={jb} max_step={a.max_step}",
             "family | last common step | mean|Δ|/spread | largest moves (layer: A -> B)"]
    for fam in a.family or DEFAULT_FAMILIES:
        layers = sorted(set(hp.layers_for(fam, keys_a)) & set(hp.layers_for(fam, keys_b)))
        if not layers:
            lines.append(f"{fam} | not in both runs")
            continue
        fkeys = [fam.format(i=l) for l in layers]
        try:
            da, db = hp.fetch(base, ja, fkeys), hp.fetch(base, jb, fkeys)
        except (OSError, KeyError) as e:
            lines.append(f"{fam} | fetch failed: {type(e).__name__}: {e}")
            continue
        steps = common_steps(da, db, a.max_step)
        if len(steps) < 2:
            lines.append(f"{fam} | <2 common steps")
            continue
        ga, gb = grid(da, fkeys, steps), grid(db, fkeys, steps)
        plot(fam, (ja, jb), layers, steps, ga, gb, out / f"{slug(fam)}.png")
        g = gist(fam, layers, steps, ga, gb)
        if g is None:
            lines.append(f"{fam} | no finite overlap")
            continue
        moves = ", ".join(f"L{l}: {x:.3g}->{y:.3g}" for l, x, y in g.top)
        lines.append(f"{fam} | {g.step} | {g.rel_mean_abs_delta:.2f} | {moves}")
    text = "\n".join(lines)
    (out / "gist.txt").write_text(text + "\n")
    print(text)
    print(f"PNGs + gist: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
