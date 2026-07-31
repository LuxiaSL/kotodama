#!/usr/bin/env python3
"""End-of-run cooldown settlement analysis for the 3B language run.

Answers: did the late-run phenomena settle cleanly by step 195,311?
  - L7/mlp EoC sigma spike (peaked ~60k sigma at 93.5-96.5k, self-resolved)
  - L14 reorganization (anisotropy differentiation, compressed write pathways)
  - 140-160k loss waves (EMA-residual amplitude)
  - Cosine cooldown (decay start step 175,780): probe loss, sharpness,
    geometry, routing.

Compares pre-decay reference window vs the final cooldown window.

Usage::

    python -m scripts.analysis.cooldown_settlement \
        --metrics data/3b-lang-metrics.jsonl \
        --geo-metrics data/3b-lang-geo_metrics.jsonl \
        --out outputs/3b_lang_cooldown_final.png

Note: train/loss here is the fp8+compile training-path estimator and is
known to be biased late in the run — wave-amplitude trends are reported,
absolute loss claims are not made from it (use bf16 evals for quality).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DECAY_START = 175_780
FINAL_STEP = 195_311
SPIKE_LAYER_KEY = "eoc/jacobian_sigma/layer_7/mlp"


@dataclass
class Series:
    """A (step, value) series for one metric key."""

    steps: np.ndarray
    values: np.ndarray

    def window(self, lo: int, hi: int) -> "Series":
        mask = (self.steps >= lo) & (self.steps <= hi)
        return Series(self.steps[mask], self.values[mask])

    @property
    def empty(self) -> bool:
        return len(self.steps) == 0

    def mean(self) -> float:
        return float(np.mean(self.values)) if not self.empty else float("nan")

    def last(self) -> float:
        return float(self.values[-1]) if not self.empty else float("nan")

    def slope_per_10k(self) -> float:
        """Least-squares slope, units per 10k steps."""
        if len(self.steps) < 3:
            return float("nan")
        coeffs = np.polyfit(self.steps.astype(float), self.values, 1)
        return float(coeffs[0] * 10_000)


@dataclass
class MetricsStore:
    """All series keyed by metric name."""

    series: dict[str, Series] = field(default_factory=dict)

    @classmethod
    def from_jsonl(cls, path: Path, keys: list[str] | None = None) -> "MetricsStore":
        rows: dict[str, list[tuple[int, float]]] = {}
        n_bad = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                step = d.get("step")
                if step is None:
                    continue
                for k, v in d.items():
                    if k in ("step", "timestamp"):
                        continue
                    if keys is not None and k not in keys:
                        continue
                    if isinstance(v, (int, float)):
                        rows.setdefault(k, []).append((int(step), float(v)))
        if n_bad:
            logger.warning("%s: skipped %d unparseable lines", path.name, n_bad)
        store = cls()
        for k, pairs in rows.items():
            pairs.sort(key=lambda p: p[0])
            steps = np.array([p[0] for p in pairs], dtype=np.int64)
            vals = np.array([p[1] for p in pairs], dtype=np.float64)
            store.series[k] = Series(steps, vals)
        return store

    def get(self, key: str) -> Series:
        return self.series.get(key, Series(np.array([], dtype=np.int64), np.array([])))


def ema_residual_amplitude(s: Series, halflife_steps: float = 1500.0,
                           window: int = 3000) -> Series:
    """Rolling amplitude (std) of residuals around an EMA trend.

    Reproduces the loss-wave quantification: EMA-residual amplitude in
    overlapping windows. Returns a Series of (window-center step, amplitude).
    """
    if len(s.steps) < 50:
        return Series(np.array([], dtype=np.int64), np.array([]))
    # EMA with decay matched to median step spacing
    spacing = float(np.median(np.diff(s.steps))) or 1.0
    alpha = 1.0 - 0.5 ** (spacing / halflife_steps)
    ema = np.empty_like(s.values)
    ema[0] = s.values[0]
    for i in range(1, len(s.values)):
        ema[i] = alpha * s.values[i] + (1 - alpha) * ema[i - 1]
    resid = s.values - ema
    centers: list[int] = []
    amps: list[float] = []
    lo, hi = int(s.steps[0]), int(s.steps[-1])
    for c in range(lo + window, hi - window // 2, window // 2):
        m = (s.steps >= c - window // 2) & (s.steps < c + window // 2)
        if m.sum() >= 20:
            centers.append(c)
            amps.append(float(np.std(resid[m])))
    return Series(np.array(centers, dtype=np.int64), np.array(amps))


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}" if np.isfinite(x) else "—"


def compare_windows(store: MetricsStore, keys: list[str], pre: tuple[int, int],
                    post: tuple[int, int]) -> list[tuple[str, float, float, float]]:
    """Per-key (pre-mean, post-mean, post-slope/10k)."""
    out: list[tuple[str, float, float, float]] = []
    for k in keys:
        s = store.get(k)
        if s.empty:
            continue
        out.append((k, s.window(*pre).mean(), s.window(*post).mean(),
                    s.window(*post).slope_per_10k()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, default=Path("data/3b-lang-metrics.jsonl"))
    p.add_argument("--geo-metrics", type=Path,
                   default=Path("data/3b-lang-geo_metrics.jsonl"))
    p.add_argument("--out", type=Path,
                   default=Path("outputs/3b_lang_cooldown_final.png"))
    p.add_argument("--json-out", type=Path,
                   default=Path("analysis/3b-lang-final/cooldown_settlement.json"))
    args = p.parse_args()

    for f in (args.metrics, args.geo_metrics):
        if not f.exists():
            raise FileNotFoundError(f)

    train = MetricsStore.from_jsonl(args.metrics, keys=["train/loss", "train/lr",
                                                        "train/grad_norm", "train/z_loss"])
    geo = MetricsStore.from_jsonl(args.geo_metrics)

    pre_win = (165_000, DECAY_START)          # last ~10k of peak LR
    post_win = (190_000, FINAL_STEP)          # last ~5k of cooldown
    cool_win = (DECAY_START, FINAL_STEP)

    print("=" * 72)
    print(f"COOLDOWN SETTLEMENT — decay start {DECAY_START:,}, final {FINAL_STEP:,}")
    print(f"  pre-decay ref window {pre_win[0]:,}-{pre_win[1]:,}; "
          f"end window {post_win[0]:,}-{post_win[1]:,}")
    print("=" * 72)

    # --- 1. Fixed-probe loss through the cooldown -------------------------
    probe = geo.get("eos/L0_probe")
    print("\n[1] Fixed-probe loss (eos/L0_probe) — the trustworthy estimator")
    if not probe.empty:
        pre, cool = probe.window(*pre_win), probe.window(*cool_win)
        # split cooldown into halves for acceleration check
        mid = (DECAY_START + FINAL_STEP) // 2
        c1, c2 = probe.window(DECAY_START, mid), probe.window(mid, FINAL_STEP)
        print(f"  pre-decay mean {fmt(pre.mean())}, slope {fmt(pre.slope_per_10k())}/10k")
        print(f"  cooldown   mean {fmt(cool.mean())}, slope {fmt(cool.slope_per_10k())}/10k")
        print(f"    1st half slope {fmt(c1.slope_per_10k())}/10k → "
              f"2nd half slope {fmt(c2.slope_per_10k())}/10k "
              f"({'accelerating' if c2.slope_per_10k() < c1.slope_per_10k() else 'flattening'})")
        print(f"  final value {fmt(probe.last())} at step {probe.steps[-1]:,} "
              f"(run min {fmt(float(np.min(probe.values)))})")
    else:
        print("  MISSING eos/L0_probe")

    # --- 2. L7/mlp sigma spike recurrence check ---------------------------
    print("\n[2] L7/mlp EoC sigma (spike layer; resolved sigma≈1.9 after 96.5k)")
    l7 = geo.get(SPIKE_LAYER_KEY)
    if not l7.empty:
        post_spike = l7.window(100_000, FINAL_STEP)
        if not post_spike.empty:
            mx_i = int(np.argmax(post_spike.values))
            print(f"  100k→end: max {fmt(float(np.max(post_spike.values)), 2)} "
                  f"at step {post_spike.steps[mx_i]:,}, "
                  f"end-window mean {fmt(post_spike.window(*post_win).mean(), 2)}")
        recur = post_spike.values > 10.0
        print(f"  recurrences >10 after 100k: {int(recur.sum())}")
    else:
        print("  MISSING", SPIKE_LAYER_KEY)
    smax = geo.get("eoc/sigma_max")
    if not smax.empty:
        print(f"  eoc/sigma_max: pre-decay mean {fmt(smax.window(*pre_win).mean(), 1)} → "
              f"end mean {fmt(smax.window(*post_win).mean(), 1)} "
              f"(last {fmt(smax.last(), 1)})")

    # --- 3. Loss-wave amplitude through decay ------------------------------
    print("\n[3] Loss-wave amplitude (EMA-residual std, fp8-path train/loss)")
    loss = train.get("train/loss")
    amp = ema_residual_amplitude(loss)
    if not amp.empty:
        def wmean(lo: int, hi: int) -> float:
            return amp.window(lo, hi).mean()
        print(f"  run-typical (60-130k): {fmt(wmean(60_000, 130_000))}")
        print(f"  wave window (140-156k): {fmt(wmean(140_000, 156_000))}")
        print(f"  pre-decay (165-175.8k): {fmt(wmean(*pre_win))}")
        print(f"  cooldown (175.8k-end): {fmt(wmean(*cool_win))}")
        print(f"  last 5k: {fmt(wmean(*post_win))}")
        print("  (prediction was: decay damps excursions)")

    # --- 4. Geometry pre- vs post-decay ------------------------------------
    print("\n[4] Geometry: pre-decay vs end-of-cooldown (mean | end-window slope/10k)")
    geo_keys = [
        "geo/rankme_last",
        "geo/layer_0/anisotropy", "geo/layer_7/anisotropy",
        "geo/layer_14/anisotropy", "geo/layer_21/anisotropy",
        "geo/layer_27/anisotropy",
        "geo/layer_21/attn_entropy_mean", "geo/layer_27/attn_entropy_mean",
        "geo/layer_14/stable_rank_o_proj", "geo/layer_14/stable_rank_down_proj",
        "geo/twonn_id/layer_7", "geo/twonn_id/layer_14", "geo/twonn_id/layer_21",
        "eos/sharpness", "eos/embed_grad_frac",
    ]
    rows = compare_windows(geo, geo_keys, pre_win, post_win)
    for k, pre_m, post_m, post_sl in rows:
        delta = post_m - pre_m
        rel = delta / abs(pre_m) * 100 if pre_m else float("nan")
        print(f"  {k:42s} {fmt(pre_m,3):>9} → {fmt(post_m,3):>9} "
              f"({rel:+.1f}%)  slope {fmt(post_sl,3)}")

    dead = [geo.get(f"geo/layer_{i}/dead_units") for i in (0, 7, 14, 21, 27)]
    dead_end = [d.window(*cool_win) for d in dead if not d.empty]
    dead_max = max((float(np.max(d.values)) for d in dead_end if not d.empty),
                   default=float("nan"))
    print(f"  dead units max over cooldown (all sampled layers): {fmt(dead_max, 4)}")

    # --- 5. AttnRes routing through decay -----------------------------------
    print("\n[5] AttnRes routing (frozen since ~40k — did decay move it?)")
    route_keys = ["attnres/final_alpha/partial", "attnres/final_alpha/block_6",
                  "attnres/final_alpha/block_0",
                  "attnres/routing_entropy/layer_16/pre_mlp",
                  "attnres/routing_entropy/layer_17/pre_mlp"]
    for k, pre_m, post_m, post_sl in compare_windows(geo, route_keys, pre_win, post_win):
        print(f"  {k:42s} {fmt(pre_m,3):>9} → {fmt(post_m,3):>9}  slope {fmt(post_sl,4)}")

    # --- 6. Sanity: z-loss / grad norm --------------------------------------
    print("\n[6] Optimizer-side sanity")
    for k in ("train/z_loss", "train/grad_norm"):
        s = train.get(k)
        if not s.empty:
            print(f"  {k:20s} pre {fmt(s.window(*pre_win).mean(),5)} → "
                  f"end {fmt(s.window(*post_win).mean(),5)}")

    # --- JSON dump -----------------------------------------------------------
    summary = {
        "decay_start": DECAY_START, "final_step": FINAL_STEP,
        "probe_loss": {
            "pre_mean": probe.window(*pre_win).mean() if not probe.empty else None,
            "cooldown_slope_per_10k": probe.window(*cool_win).slope_per_10k()
            if not probe.empty else None,
            "final": probe.last() if not probe.empty else None,
        },
        "l7_mlp_recurrences_gt10_post100k": int((l7.window(100_000, FINAL_STEP).values > 10).sum())
        if not l7.empty else None,
        "geometry_pre_vs_post": {k: {"pre": pm, "post": qm, "slope": sl}
                                 for k, pm, qm, sl in rows},
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote %s", args.json_out)

    # --- Figure ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 3, figsize=(17, 11))
        fig.suptitle("3B language run — cooldown settlement (final step 195,311)",
                     fontsize=13)

        def vline(ax):
            ax.axvline(DECAY_START / 1e3, color="crimson", ls="--", lw=0.8,
                       label="decay start")

        zoom = (150_000, FINAL_STEP + 500)

        ax = axes[0][0]
        s = probe.window(*zoom)
        if not s.empty:
            ax.plot(s.steps / 1e3, s.values, lw=1.0, color="tab:blue")
        vline(ax); ax.set_title("fixed-probe loss (eos/L0_probe)"); ax.legend()

        ax = axes[0][1]
        s = loss.window(*zoom)
        if not s.empty:
            ax.plot(s.steps / 1e3, s.values, lw=0.3, alpha=0.5, color="gray")
        vline(ax); ax.set_title("train/loss (fp8 path — biased, shape only)")

        ax = axes[0][2]
        if not amp.empty:
            ax.plot(amp.steps / 1e3, amp.values, lw=1.2, color="tab:purple")
        vline(ax); ax.set_title("loss-wave amplitude (EMA-resid std, full run)")

        ax = axes[1][0]
        s = smax.window(*zoom)
        if not s.empty:
            ax.plot(s.steps / 1e3, s.values, lw=0.9, color="tab:orange", label="sigma_max")
        s = l7.window(*zoom)
        if not s.empty:
            ax.plot(s.steps / 1e3, s.values, lw=0.9, color="tab:green", label="L7/mlp")
        ax.set_yscale("log"); vline(ax); ax.legend(); ax.set_title("EoC Jacobian sigma")

        ax = axes[1][1]
        sh = geo.get("eos/sharpness").window(*zoom)
        if not sh.empty:
            ax.plot(sh.steps / 1e3, sh.values, lw=0.9, color="tab:red")
        vline(ax); ax.set_title("EoS sharpness")

        ax = axes[1][2]
        rk = geo.get("geo/rankme_last").window(*zoom)
        if not rk.empty:
            ax.plot(rk.steps / 1e3, rk.values, lw=1.0, color="tab:cyan")
        vline(ax); ax.set_title("RankMe (last layer)")

        ax = axes[2][0]
        for i, c in zip((0, 7, 14, 21, 27), ("k", "tab:green", "tab:blue", "tab:orange", "tab:red")):
            s = geo.get(f"geo/layer_{i}/anisotropy").window(*zoom)
            if not s.empty:
                ax.plot(s.steps / 1e3, s.values, lw=0.9, color=c, label=f"L{i}")
        vline(ax); ax.legend(ncol=2, fontsize=7); ax.set_title("anisotropy by layer")

        ax = axes[2][1]
        for i, c in zip((21, 27), ("tab:orange", "tab:red")):
            s = geo.get(f"geo/layer_{i}/attn_entropy_mean").window(*zoom)
            if not s.empty:
                ax.plot(s.steps / 1e3, s.values, lw=0.9, color=c, label=f"L{i}")
        vline(ax); ax.legend(); ax.set_title("deep attn entropy (BOS-sink guard)")

        ax = axes[2][2]
        for k, c in (("attnres/final_alpha/partial", "tab:blue"),
                     ("attnres/final_alpha/block_6", "tab:orange"),
                     ("attnres/final_alpha/block_0", "tab:green")):
            s = geo.get(k).window(*zoom)
            if not s.empty:
                ax.plot(s.steps / 1e3, s.values, lw=0.9, color=c,
                        label=k.split("/")[-1])
        vline(ax); ax.legend(fontsize=7); ax.set_title("AttnRes final-mix alpha")

        for row in axes:
            for ax in row:
                ax.set_xlabel("step (k)")
                ax.grid(alpha=0.25)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130)
        logger.info("Wrote %s", args.out)
    except Exception:
        logger.exception("Figure generation failed (analysis text still valid)")


if __name__ == "__main__":
    main()
