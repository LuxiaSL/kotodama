#!/usr/bin/env python3
"""Render heimdall Telemetry families to a PNG + a text gist (for humans AND agents).

Two modes, both reading the coordinator's HTTP API:

  comet   --x FAMILY --y FAMILY : per-layer trajectory of family X vs family Y over
          training (the Telemetry "portrait" panel), colored by step, dot = latest.
  heatmap --x FAMILY            : layer x step heatmap of one family.

FAMILY is a key pattern with ``{i}`` for the layer, e.g.
``norm_gain/layer_{i}/q_norm`` or ``geo/layer_{i}/attn_entropy_mean``. Layers are
auto-detected from the run's keys (only layers present in every family are used).

The gist is the quantitative read: per layer start -> end, total change, the
recent slope (dy/dx over the last quarter of the trajectory, comet mode), how
much of the movement happened in the last quarter ("still moving?"), and across
layers the Spearman rank correlation of the final values (comet mode).

Examples:
  python scripts/eval/hm_portrait.py --job <id> --x 'norm_gain/layer_{i}/q_norm' \
      --y 'geo/layer_{i}/attn_entropy_mean' --out /tmp/claude-output/qgain-vs-entropy.png
  python scripts/eval/hm_portrait.py --job <A0> --job <A1> --x ... --y ...   # side by side
  python scripts/eval/hm_portrait.py --job <id> --x 'norm_gain/layer_{i}/q_norm' --out heat.png
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass

Series = dict[int, float]  # step -> value


# ── pure helpers (tested) ─────────────────────────────────────────────────────


def family_regex(pattern: str) -> re.Pattern[str]:
    if "{i}" not in pattern:
        raise ValueError(f"family pattern needs '{{i}}' for the layer: {pattern!r}")
    head, tail = pattern.split("{i}", 1)
    return re.compile("^" + re.escape(head) + r"(\d+)" + re.escape(tail) + "$")


def layers_for(pattern: str, keys: list[str]) -> list[int]:
    rx = family_regex(pattern)
    return sorted({int(m.group(1)) for k in keys if (m := rx.match(k))})


def align(x: Series, y: Series) -> list[tuple[int, float, float]]:
    return [(s, x[s], y[s]) for s in sorted(set(x) & set(y))]


def spearman(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((p - ma) * (q - mb) for p, q in zip(ra, rb))
    va = sum((p - ma) ** 2 for p in ra) ** 0.5
    vb = sum((q - mb) ** 2 for q in rb) ** 0.5
    return None if va == 0 or vb == 0 else cov / (va * vb)


@dataclass(frozen=True)
class TrajGist:
    layer: int
    start: tuple[float, float]
    end: tuple[float, float]
    recent_slope: float | None  # dy/dx over the last quarter
    last_quarter_share: float  # fraction of total path length in the last quarter


def gist_trajectory(layer: int, pts: list[tuple[int, float, float]]) -> TrajGist | None:
    if len(pts) < 2:
        return None
    seg = [((pts[k + 1][1] - pts[k][1]) ** 2 + (pts[k + 1][2] - pts[k][2]) ** 2) ** 0.5
           for k in range(len(pts) - 1)]
    total = sum(seg) or 1e-12
    q = max(1, len(pts) * 3 // 4)
    tail = pts[q - 1:]
    dx, dy = tail[-1][1] - tail[0][1], tail[-1][2] - tail[0][2]
    return TrajGist(
        layer=layer,
        start=(pts[0][1], pts[0][2]),
        end=(pts[-1][1], pts[-1][2]),
        recent_slope=(dy / dx) if abs(dx) > 1e-9 else None,
        last_quarter_share=sum(seg[q - 1:]) / total,
    )


# ── heimdall API ──────────────────────────────────────────────────────────────


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def fetch(base: str, job: str, keys: list[str]) -> dict[str, Series]:
    out: dict[str, Series] = {k: {} for k in keys}
    for i in range(0, len(keys), 12):
        chunk = keys[i:i + 12]
        q = urllib.parse.urlencode({"keys": ",".join(chunk), "downsample": 5000})
        for p in _get(f"{base}/api/v1/jobs/{job}/metrics?{q}")["points"]:
            for k in chunk:
                v = p.get(k)
                if isinstance(v, (int, float)):
                    out[k][int(p["step"])] = float(v)
    return out


def job_keys(base: str, job: str) -> list[str]:
    return list(_get(f"{base}/api/v1/jobs/{job}/metrics")["keys"])


# ── modes ─────────────────────────────────────────────────────────────────────


def comet(base: str, jobs: list[str], xf: str, yf: str, out: str | None) -> int:
    per_job: dict[str, dict[int, list[tuple[int, float, float]]]] = {}
    for job in jobs:
        keys = job_keys(base, job)
        layers = sorted(set(layers_for(xf, keys)) & set(layers_for(yf, keys)))
        if not layers:
            print(f"{job}: no layers carry both {xf} and {yf}")
            return 1
        data = fetch(base, job, [xf.format(i=l) for l in layers] + [yf.format(i=l) for l in layers])
        per_job[job] = {l: align(data[xf.format(i=l)], data[yf.format(i=l)]) for l in layers}

    for job, trajs in per_job.items():
        steps = max((t[-1][0] for t in trajs.values() if t), default=0)
        print(f"== {job}: {xf}  vs  {yf}  (to step {steps})")
        print(f"   {'layer':>5}  {'start (x, y)':>20}  {'end (x, y)':>20}  {'Δx':>8} {'Δy':>8}"
              f"  {'recent dy/dx':>12}  {'last-¼ share':>12}")
        finals_x, finals_y = [], []
        for l, pts in trajs.items():
            g = gist_trajectory(l, pts)
            if g is None:
                continue
            finals_x.append(g.end[0]); finals_y.append(g.end[1])
            slope = f"{g.recent_slope:+.3f}" if g.recent_slope is not None else "—"
            print(f"   L{l:<4}  ({g.start[0]:8.3f},{g.start[1]:9.3f})  ({g.end[0]:8.3f},{g.end[1]:9.3f})"
                  f"  {g.end[0] - g.start[0]:+8.3f} {g.end[1] - g.start[1]:+8.3f}"
                  f"  {slope:>12}  {g.last_quarter_share:12.0%}")
        rho = spearman(finals_x, finals_y)
        print(f"   across layers: Spearman(final x, final y) = "
              f"{'n/a' if rho is None else f'{rho:+.2f}'}  (n={len(finals_x)})")

    if out:
        _plot_comet(per_job, xf, yf, out)
        print(f"PNG: {out}")
    return 0


def heatmap(base: str, jobs: list[str], xf: str, out: str | None) -> int:
    per_job: dict[str, tuple[list[int], list[int], list[list[float]]]] = {}
    for job in jobs:
        keys = job_keys(base, job)
        layers = layers_for(xf, keys)
        if not layers:
            print(f"{job}: no keys match {xf}")
            return 1
        data = fetch(base, job, [xf.format(i=l) for l in layers])
        steps = sorted(set().union(*(data[xf.format(i=l)].keys() for l in layers)))
        grid = [[data[xf.format(i=l)].get(s, float("nan")) for s in steps] for l in layers]
        per_job[job] = (layers, steps, grid)
        print(f"== {job}: {xf}  ({len(layers)} layers, steps {steps[0]}..{steps[-1]})")
        print(f"   {'layer':>5}  {'start':>9}  {'end':>9}  {'min':>9}  {'max':>9}  {'last-¼ Δ':>9}")
        for l, row in zip(layers, grid):
            vals = [v for v in row if v == v]
            if not vals:
                continue
            q = vals[len(vals) * 3 // 4]
            print(f"   L{l:<4}  {vals[0]:9.4f}  {vals[-1]:9.4f}  {min(vals):9.4f}  {max(vals):9.4f}"
                  f"  {vals[-1] - q:+9.4f}")
    if out:
        _plot_heat(per_job, xf, out)
        print(f"PNG: {out}")
    return 0


# ── plotting ──────────────────────────────────────────────────────────────────


def _plot_comet(per_job: dict, xf: str, yf: str, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    n = len(per_job)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 6), squeeze=False, sharex=True, sharey=True)
    smax = max((t[-1][0] for tr in per_job.values() for t in tr.values() if t), default=1)
    for ax, (job, trajs) in zip(axes[0], per_job.items()):
        for l, pts in trajs.items():
            if len(pts) < 2:
                continue
            xy = [(p[1], p[2]) for p in pts]
            segs = [[xy[k], xy[k + 1]] for k in range(len(xy) - 1)]
            lc = LineCollection(segs, cmap="inferno", norm=plt.Normalize(0, smax), linewidths=1.6)
            lc.set_array([p[0] for p in pts[1:]])
            ax.add_collection(lc)
            ax.plot(*xy[-1], "o", color="white", mec="black", ms=5, zorder=3)
            ax.annotate(f"L{l}", xy[-1], xytext=(4, 2), textcoords="offset points", fontsize=8)
        ax.autoscale()
        ax.set_title(job, fontsize=10)
        ax.set_xlabel(xf)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(yf)
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, smax))
    fig.colorbar(sm, ax=axes[0].tolist(), label="step", fraction=0.03)
    fig.savefig(out, dpi=110, bbox_inches="tight")


def _plot_heat(per_job: dict, xf: str, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(per_job)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), squeeze=False)
    for ax, (job, (layers, steps, grid)) in zip(axes[0], per_job.items()):
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="magma",
                       extent=(steps[0], steps[-1], -0.5, len(layers) - 0.5))
        ax.set_yticks(range(len(layers)), [f"L{l}" for l in layers], fontsize=7)
        ax.set_xlabel("step")
        ax.set_title(f"{job}: {xf}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.savefig(out, dpi=110, bbox_inches="tight")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", action="append", required=True, help="heimdall job id (repeat to compare)")
    ap.add_argument("--x", required=True, help="family pattern with {i}")
    ap.add_argument("--y", help="second family (comet mode); omit for a heatmap")
    ap.add_argument("--out", help="PNG path")
    ap.add_argument("--url", default=os.environ.get(
        "KOTODAMA_SCHEDULER_URL", os.environ.get("HEIMDALL_URL", "http://127.0.0.1:7000")))
    a = ap.parse_args()
    base = a.url.rstrip("/")
    try:
        return comet(base, a.job, a.x, a.y, a.out) if a.y else heatmap(base, a.job, a.x, a.out)
    except (OSError, KeyError, ValueError) as e:
        print(f"FAIL: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
