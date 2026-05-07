#!/usr/bin/env python3
"""Generate a self-contained HTML analysis report from pipeline outputs.

Reads JSON/PNG outputs from Tracks 1-6 and produces a single HTML file
with embedded charts, plots, and formatted data.

Usage::

    python scripts/analysis/generate_report.py owt-ddv1
    python scripts/analysis/generate_report.py owt-ddv1 --theme light
    python scripts/analysis/generate_report.py owt-ddv1 -o report.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class ReportData:
    def __init__(self, run_name: str, run_dir: Path) -> None:
        self.run_name = run_name
        self.run_dir = run_dir
        self.summary: dict[str, Any] | None = None
        self.dynamics: dict[str, Any] | None = None
        self.geometric_health: dict[str, Any] | None = None
        self.generations: dict[str, Any] | None = None
        self.text_quality: dict[str, Any] | None = None
        self.activation_plots: list[Path] = []
        self.concept_results: dict[str, Any] | None = None
        self.concept_plots: list[Path] = []


def load_report_data(run_name: str, analysis_dir: Path) -> ReportData:
    run_dir = analysis_dir / run_name
    data = ReportData(run_name, run_dir)

    ts_path = run_dir / "training_summary.json"
    if ts_path.exists():
        ts = json.loads(ts_path.read_text())
        data.summary = ts.get("summary")
        data.dynamics = ts.get("dynamics")
        data.geometric_health = ts.get("geometric_health")

    gen_path = run_dir / "generations.json"
    if gen_path.exists():
        data.generations = json.loads(gen_path.read_text())

    tq_path = run_dir / "text_quality.json"
    if tq_path.exists():
        data.text_quality = json.loads(tq_path.read_text())

    ag_plots = run_dir / "activation_geometry" / "plots"
    if ag_plots.exists():
        data.activation_plots = sorted(ag_plots.glob("*.png"))

    cg_results = run_dir / "concept_geometry" / "results.json"
    if cg_results.exists():
        data.concept_results = json.loads(cg_results.read_text())
    cg_plots = run_dir / "concept_geometry" / "plots"
    if cg_plots.exists():
        data.concept_plots = sorted(cg_plots.glob("*.png"))

    return data


# ---------------------------------------------------------------------------
# Plot generation (matplotlib → base64)
# ---------------------------------------------------------------------------

def _embed_fig(fig: Any) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    import matplotlib.pyplot as plt
    plt.close(fig)
    return f'<img src="data:image/png;base64,{b64}" class="chart" />'


def _embed_image(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" class="plot" />'


def render_loss_chart(dynamics: dict[str, Any]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss_data = dynamics.get("metrics", {}).get("train/loss", {}).get("rolling_stats", [])
    if not loss_data:
        return ""

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    steps = [d["step_center"] for d in loss_data]
    means = [d["mean"] for d in loss_data]
    p25 = [d["p25"] for d in loss_data]
    p75 = [d["p75"] for d in loss_data]

    ax.fill_between(steps, p25, p75, alpha=0.15, color="#58a6ff")
    ax.plot(steps, means, color="#58a6ff", linewidth=2)
    ax.set_xlabel("Step", color="#8b949e")
    ax.set_ylabel("Loss", color="#8b949e")
    ax.set_title("Training Loss", color="#e6edf3", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.1, color="#8b949e")

    return _embed_fig(fig)


def render_rankme_chart(dynamics: dict[str, Any]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rm_data = dynamics.get("metrics", {}).get("geo/rankme_last", {}).get("rolling_stats", [])
    if not rm_data:
        return ""

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    steps = [d["step_center"] for d in rm_data]
    means = [d["mean"] for d in rm_data]
    p25 = [d["p25"] for d in rm_data]
    p75 = [d["p75"] for d in rm_data]

    ax.fill_between(steps, p25, p75, alpha=0.15, color="#3fb950")
    ax.plot(steps, means, color="#3fb950", linewidth=2)
    ax.set_xlabel("Step", color="#8b949e")
    ax.set_ylabel("RankMe", color="#8b949e")
    ax.set_title("RankMe (Effective Rank)", color="#e6edf3", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.1, color="#8b949e")

    return _embed_fig(fig)


def render_depth_gradient_chart(geo_health: dict[str, Any]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    dg = geo_health.get("depth_gradient", {})
    layers_info = dg.get("layers", {})
    if not layers_info:
        return ""

    metrics = {k: v for k, v in dg.items() if k != "layers"}
    if not metrics:
        return ""

    plt.style.use("dark_background")
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.patch.set_facecolor("#0d1117")
    if n == 1:
        axes = [axes]

    positions = ["first", "mid", "last"]
    layer_labels = [f"L{layers_info.get(p, '?')}" for p in positions]
    colors = ["#58a6ff", "#d2a8ff", "#f97583"]

    for ax, (metric_name, vals) in zip(axes, metrics.items()):
        ax.set_facecolor("#0d1117")
        bar_vals = [vals.get(p, 0) for p in positions]
        bars = ax.bar(layer_labels, bar_vals, color=colors, alpha=0.8, width=0.5)
        ax.set_title(metric_name.replace("_", " ").title(), color="#e6edf3", fontsize=11)
        ax.tick_params(colors="#8b949e")
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, v in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", color="#8b949e", fontsize=9)

    fig.suptitle("Depth Gradient (First → Mid → Last Layer)", color="#e6edf3", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _embed_fig(fig)


def render_geo_profiles_chart(geo_health: dict[str, Any]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = geo_health.get("profiles", {})
    layers = geo_health.get("layers", [])
    if not profiles or not layers:
        return ""

    metric_key = "stable_rank_q_proj"
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    cmap = ["#58a6ff", "#79c0ff", "#d2a8ff", "#f97583", "#ffa657"]
    for i, (landmark, profile) in enumerate(profiles.items()):
        vals = []
        for layer_idx in layers:
            layer_data = profile.get(f"layer_{layer_idx}", {})
            vals.append(layer_data.get(metric_key, 0))
        color = cmap[i % len(cmap)]
        ax.plot(layers, vals, marker="o", markersize=4, linewidth=1.5,
                color=color, label=f"{landmark} (step {profile.get('step', '?')})", alpha=0.85)

    ax.set_xlabel("Layer", color="#8b949e")
    ax.set_ylabel("Stable Rank (Q proj)", color="#8b949e")
    ax.set_title("Stable Rank Across Depth Over Training", color="#e6edf3", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    ax.tick_params(colors="#8b949e")
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.1, color="#8b949e")

    return _embed_fig(fig)


def render_text_quality_radar(text_quality: dict[str, Any]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    profiles = text_quality.get("model_profiles", {})
    if not profiles:
        return ""

    profile = next(iter(profiles.values()))

    radar_metrics = [
        ("lexical.ttr_100", "Vocabulary"),
        ("coherence.topic_drift_mean", "Coherence"),
        ("coherence.novelty_mean", "Novelty"),
        ("structural.discourse_marker_density", "Discourse"),
        ("structural.structural_variety_entropy", "Structure"),
        ("repetition.unique_5gram_ratio", "Non-Repetition"),
        ("creativity.specificity_score", "Specificity"),
        ("creativity.temporal_marker_density", "Narrative"),
    ]

    values = []
    labels = []
    for key, label in radar_metrics:
        v = profile.get(key)
        if v is not None:
            values.append(v)
            labels.append(label)

    if len(values) < 3:
        return ""

    max_vals = [max(abs(v), 0.001) for v in values]
    normalized = [v / m for v, m in zip(values, max_vals)]

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = normalized + [normalized[0]]
    angles += [angles[0]]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.fill(angles, values_plot, alpha=0.15, color="#58a6ff")
    ax.plot(angles, values_plot, color="#58a6ff", linewidth=2)
    ax.scatter(angles[:-1], normalized, color="#58a6ff", s=40, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="#e6edf3", fontsize=10)
    ax.tick_params(axis="y", colors="#8b949e")
    ax.set_title("Text Quality Profile", color="#e6edf3", fontsize=14, fontweight="bold", pad=20)
    ax.spines["polar"].set_color("#30363d")
    ax.grid(color="#30363d", alpha=0.3)

    return _embed_fig(fig)


# ---------------------------------------------------------------------------
# HTML section renderers
# ---------------------------------------------------------------------------

def _metric_card(label: str, value: str, subtitle: str = "", color: str = "blue") -> str:
    sub = f'<div class="metric-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    return f"""<div class="metric-card" data-color="{color}">
        <div class="metric-value">{html.escape(value)}</div>
        <div class="metric-label">{html.escape(label)}</div>
        {sub}
    </div>"""


def render_hero(summary: dict[str, Any]) -> str:
    cards = []
    loss = summary.get("loss_final")
    if loss is not None:
        cards.append(_metric_card("Final Loss", f"{loss:.4f}",
                                  f"min {summary.get('loss_min', 0):.4f} @ step {summary.get('loss_min_step', '?')}", color="blue"))

    ppl = summary.get("ppl_final")
    if ppl is not None:
        cards.append(_metric_card("Perplexity", f"{ppl:.2f}", color="blue"))

    rankme = summary.get("rankme_final")
    if rankme is not None:
        rebound = summary.get("rankme_rebound_ratio", 0)
        cards.append(_metric_card("RankMe", f"{rankme:.1f}", f"{rebound:.2f}x rebound", color="green"))

    ww = summary.get("ww_alpha_final")
    if ww is not None:
        healthy = summary.get("ww_healthy_frac_final", 0)
        cards.append(_metric_card("WW Alpha", f"{ww:.2f}", f"{healthy:.0%} healthy", color="green"))

    twonn = summary.get("twonn_id_final")
    if twonn is not None:
        cards.append(_metric_card("TwoNN ID", f"{twonn:.1f}", color="purple"))

    tps = summary.get("avg_tokens_per_sec")
    if tps is not None:
        cards.append(_metric_card("Throughput", f"{tps / 1e6:.2f}M tok/s", color="orange"))

    tokens = summary.get("tokens_B")
    steps = summary.get("max_step")
    if tokens is not None:
        cards.append(_metric_card("Training", f"{tokens:.2f}B tokens", f"{steps} steps", color="orange"))

    return f'<section id="hero"><div class="metric-grid">{"".join(cards)}</div></section>'


def render_training_section(dynamics: dict[str, Any]) -> str:
    parts = ['<section id="training"><h2>Training Dynamics</h2>']
    parts.append('<div class="chart-row">')
    parts.append(f'<div class="chart-container">{render_loss_chart(dynamics)}</div>')
    parts.append(f'<div class="chart-container">{render_rankme_chart(dynamics)}</div>')
    parts.append("</div>")

    loss_dyn = dynamics.get("metrics", {}).get("train/loss", {})
    inflections = loss_dyn.get("inflection_points", [])
    stability = loss_dyn.get("stability_score")
    jumps = loss_dyn.get("jumps", [])
    plateaus = loss_dyn.get("plateaus", [])

    parts.append('<details><summary>Dynamics Detail</summary><div class="detail-content">')
    parts.append(f'<p>Stability score: <strong>{stability:.4f}</strong> (lower = more stable late training)</p>')
    parts.append(f"<p>Jumps: {len(jumps)} | Plateaus: {len(plateaus)} | Inflection points: {len(inflections)}</p>")

    if inflections:
        parts.append("<table><thead><tr><th>Step</th><th>Direction</th></tr></thead><tbody>")
        for ip in inflections:
            parts.append(f'<tr><td>{ip["step"]}</td><td>{html.escape(ip["direction"])}</td></tr>')
        parts.append("</tbody></table>")

    slopes = loss_dyn.get("slopes_at_landmarks", {})
    if slopes:
        parts.append("<h4>Loss Slope at Landmarks</h4>")
        parts.append("<table><thead><tr><th>Landmark</th><th>Step</th><th>Slope (500w)</th></tr></thead><tbody>")
        for name, info in slopes.items():
            step = info.get("step", "?")
            slope_500 = info.get("slopes", {}).get("500", "—")
            slope_str = f"{slope_500:.6f}" if isinstance(slope_500, (int, float)) else str(slope_500)
            parts.append(f"<tr><td>{html.escape(name)}</td><td>{step}</td><td>{slope_str}</td></tr>")
        parts.append("</tbody></table>")

    parts.append("</div></details>")
    parts.append("</section>")
    return "\n".join(parts)


def render_geometry_section(geo_health: dict[str, Any]) -> str:
    parts = ['<section id="geometry"><h2>Geometric Health</h2>']

    parts.append('<div class="chart-row">')
    parts.append(f'<div class="chart-container">{render_depth_gradient_chart(geo_health)}</div>')
    parts.append(f'<div class="chart-container">{render_geo_profiles_chart(geo_health)}</div>')
    parts.append("</div>")

    stability = geo_health.get("rankme_stability")
    if stability is not None:
        parts.append(f'<p class="stat">RankMe late stability: <strong>{stability:.4f}</strong></p>')

    profiles = geo_health.get("profiles", {})
    if profiles:
        parts.append('<details><summary>Per-Layer Profiles at Landmarks</summary><div class="detail-content">')
        for landmark, profile in profiles.items():
            step = profile.get("step", "?")
            parts.append(f"<h4>{landmark.replace('_', ' ').title()} (step {step})</h4>")
            parts.append("<table><thead><tr><th>Layer</th><th>SR q</th><th>SR k</th><th>SR o</th>"
                         "<th>Entropy μ</th><th>Entropy σ</th><th>Anisotropy</th><th>Dead</th></tr></thead><tbody>")
            layers = geo_health.get("layers", [])
            for layer_idx in layers:
                ld = profile.get(f"layer_{layer_idx}", {})
                parts.append(f"<tr><td>{layer_idx}</td>"
                             f"<td>{ld.get('stable_rank_q_proj', 0):.1f}</td>"
                             f"<td>{ld.get('stable_rank_k_proj', 0):.1f}</td>"
                             f"<td>{ld.get('stable_rank_o_proj', 0):.1f}</td>"
                             f"<td>{ld.get('attn_entropy_mean', 0):.3f}</td>"
                             f"<td>{ld.get('attn_entropy_std', 0):.3f}</td>"
                             f"<td>{ld.get('anisotropy', 0):.4f}</td>"
                             f"<td>{ld.get('dead_units', 0):.4f}</td></tr>")
            parts.append("</tbody></table>")
        parts.append("</div></details>")

    parts.append("</section>")
    return "\n".join(parts)


def render_generations_section(generations: dict[str, Any]) -> str:
    parts = ['<section id="generations"><h2>Generated Text</h2>']

    config = generations.get("config", {})
    parts.append(f'<p class="config">T={config.get("temperature", "?")} | '
                 f'top_p={config.get("top_p", "none")} | '
                 f'max_tokens={config.get("max_tokens", "?")} | '
                 f'n_samples={config.get("n_samples", "?")}</p>')

    for run in generations.get("runs", []):
        ppl_str = f' | eval_ppl={run["eval_ppl"]:.2f}' if "eval_ppl" in run else ""
        parts.append(f'<h3>{html.escape(run.get("name", "unknown"))}{ppl_str}</h3>')

        current_prompt_idx = -1
        for sample in run.get("samples", []):
            if sample["prompt_idx"] != current_prompt_idx:
                if current_prompt_idx >= 0:
                    parts.append("</div>")
                current_prompt_idx = sample["prompt_idx"]
                parts.append(f'<div class="prompt-group">')
                parts.append(f'<div class="prompt">{html.escape(sample["prompt"])}</div>')

            stopped_class = "eos" if sample.get("stopped_by") == "eos" else "max"
            parts.append(f'<div class="continuation">'
                         f'<span class="sample-meta">[{sample["n_tokens"]} tok, '
                         f'<span class="stopped-{stopped_class}">{sample.get("stopped_by", "?")}</span>]</span>'
                         f'{html.escape(sample["continuation"])}</div>')

        if current_prompt_idx >= 0:
            parts.append("</div>")

    parts.append("</section>")
    return "\n".join(parts)


def render_text_quality_section(text_quality: dict[str, Any]) -> str:
    parts = ['<section id="text-quality"><h2>Text Quality</h2>']

    parts.append(render_text_quality_radar(text_quality))

    distinctive = text_quality.get("most_distinctive", {})
    if distinctive:
        for run_name, features in distinctive.items():
            parts.append(f"<h3>Distinctive Features: {html.escape(run_name)}</h3>")
            parts.append("<table><thead><tr><th>Metric</th><th>Value</th><th>Z-Score</th></tr></thead><tbody>")
            for feat in features[:15]:
                z = feat.get("z_score", 0)
                z_class = "positive" if z > 0 else "negative"
                parts.append(f'<tr><td>{html.escape(feat["metric"])}</td>'
                             f'<td>{feat.get("value", 0):.4f}</td>'
                             f'<td class="z-{z_class}">{z:+.2f}</td></tr>')
            parts.append("</tbody></table>")

    profiles = text_quality.get("model_profiles", {})
    if profiles:
        parts.append('<details><summary>Full Metric Profiles</summary><div class="detail-content">')
        for run_name, profile in profiles.items():
            parts.append(f"<h4>{html.escape(run_name)}</h4>")
            categories: dict[str, list[tuple[str, float]]] = {}
            for key, val in sorted(profile.items()):
                cat = key.split(".")[0] if "." in key else "other"
                categories.setdefault(cat, []).append((key, val))

            for cat, items in sorted(categories.items()):
                parts.append(f"<h5>{cat.title()}</h5>")
                parts.append("<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>")
                for key, val in items:
                    parts.append(f"<tr><td>{html.escape(key)}</td><td>{val:.4f}</td></tr>")
                parts.append("</tbody></table>")
        parts.append("</div></details>")

    parts.append("</section>")
    return "\n".join(parts)


def render_activation_section(plots: list[Path]) -> str:
    parts = ['<section id="activation-geometry"><h2>Activation Geometry</h2>']
    parts.append('<div class="plot-grid">')
    for p in plots:
        name = p.stem.replace("_", " ").title()
        parts.append(f'<div class="plot-card"><h4>{html.escape(name)}</h4>{_embed_image(p)}</div>')
    parts.append("</div></section>")
    return "\n".join(parts)


def render_concept_section(results: dict[str, Any] | None, plots: list[Path]) -> str:
    parts = ['<section id="concept-geometry"><h2>Concept Geometry</h2>']

    if results:
        for analysis_type in ["cyclic", "ordinal"]:
            data = results.get(analysis_type, {})
            if not data:
                continue
            parts.append(f"<h3>{analysis_type.title()} Structure</h3>")
            parts.append("<table><thead><tr><th>Concept Set</th><th>Mantel ρ</th><th>p-value</th></tr></thead><tbody>")
            for concept_set, info in data.items():
                if isinstance(info, dict):
                    rho = info.get("mantel_rho", info.get("rho", "—"))
                    pval = info.get("p_value", info.get("p", "—"))
                    rho_str = f"{rho:.4f}" if isinstance(rho, (int, float)) else str(rho)
                    p_str = f"{pval:.4f}" if isinstance(pval, (int, float)) else str(pval)
                    parts.append(f"<tr><td>{html.escape(concept_set)}</td><td>{rho_str}</td><td>{p_str}</td></tr>")
            parts.append("</tbody></table>")

    if plots:
        parts.append('<div class="plot-grid">')
        for p in plots:
            name = p.stem.replace("_", " ").title()
            parts.append(f'<div class="plot-card"><h4>{html.escape(name)}</h4>{_embed_image(p)}</div>')
        parts.append("</div>")

    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = """
:root {
    --bg: #0d1117;
    --surface: #161b22;
    --card: #21262d;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --blue: #58a6ff;
    --green: #3fb950;
    --orange: #ffa657;
    --red: #f97583;
    --purple: #d2a8ff;
    --nav-h: 42px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px;
}
header { border-bottom: 1px solid var(--border); padding: 20px 0; margin-bottom: 30px; }
header h1 { font-size: 28px; color: var(--text); }
header .meta { color: var(--text-dim); font-size: 14px; margin-top: 4px; }
nav { position: sticky; top: 0; background: var(--bg); border-bottom: 1px solid var(--border);
      padding: 10px 0; margin-bottom: 20px; z-index: 100; display: flex; gap: 12px;
      flex-wrap: wrap; align-items: center; height: var(--nav-h); }
nav a { color: var(--text-dim); text-decoration: none; font-size: 13px; padding: 4px 10px;
        border-radius: 6px; transition: background 0.15s, color 0.15s; }
nav a:hover { background: var(--card); color: var(--text); }
nav a.active { background: var(--card); color: var(--blue); font-weight: 600; }
section { margin-bottom: 48px; scroll-margin-top: calc(var(--nav-h) + 12px); }
h2 { font-size: 22px; color: var(--text); border-bottom: 1px solid var(--border);
     padding-bottom: 8px; margin-bottom: 16px; }
h3 { font-size: 17px; color: var(--text); margin: 16px 0 8px; }
h4 { font-size: 14px; color: var(--text-dim); margin: 12px 0 6px; }
h5 { font-size: 13px; color: var(--purple); margin: 10px 0 4px; text-transform: uppercase; letter-spacing: 0.5px; }
p.stat { color: var(--text-dim); margin: 8px 0; }
p.config { color: var(--text-dim); font-family: monospace; font-size: 13px;
           background: var(--card); padding: 8px 12px; border-radius: 6px; margin-bottom: 16px; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 12px; }
.metric-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
               padding: 16px; text-align: center; transition: border-color 0.15s; }
.metric-card:hover { border-color: var(--text-dim); }
.metric-value { font-size: 26px; font-weight: 700; color: var(--blue); }
.metric-card[data-color="green"] .metric-value { color: var(--green); }
.metric-card[data-color="orange"] .metric-value { color: var(--orange); }
.metric-card[data-color="purple"] .metric-value { color: var(--purple); }
.metric-label { font-size: 12px; color: var(--text-dim); text-transform: uppercase;
                letter-spacing: 0.5px; margin-top: 4px; }
.metric-sub { font-size: 11px; color: var(--text-dim); margin-top: 2px; }
.chart-row { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 900px) { .chart-row { grid-template-columns: 1fr 1fr; } }
.chart-container { background: var(--surface); border: 1px solid var(--border);
                   border-radius: 8px; padding: 16px; }
.chart { max-width: 100%; height: auto; border-radius: 4px; display: block; }
.plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px; }
.plot-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
.plot-card h4 { margin: 0 0 8px; }
.plot, .chart { max-width: 100%; height: auto; border-radius: 4px; cursor: zoom-in; }
table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 13px; }
thead th { background: var(--card); color: var(--text-dim); text-align: left;
           padding: 8px 10px; border-bottom: 1px solid var(--border); font-weight: 600;
           text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px;
           position: sticky; top: var(--nav-h); z-index: 10; cursor: pointer; }
thead th:hover { color: var(--text); }
tbody td { padding: 6px 10px; border-bottom: 1px solid var(--border); font-family: monospace; font-size: 12px; }
tbody tr:hover { background: var(--card); }
details { margin: 12px 0; }
summary { cursor: pointer; color: var(--blue); font-size: 14px; padding: 8px 0;
          user-select: none; transition: color 0.15s; }
summary:hover { color: var(--purple); }
summary::marker { color: var(--text-dim); }
.detail-content { padding: 12px 0; animation: fadeIn 0.2s ease-out; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: none; } }
.prompt-group { background: var(--surface); border: 1px solid var(--border);
                border-radius: 8px; margin: 12px 0; overflow: hidden; }
.prompt { padding: 12px 16px; background: var(--card); font-weight: 600;
          border-bottom: 1px solid var(--border); font-size: 14px;
          border-left: 3px solid var(--blue); }
.continuation { padding: 12px 16px; font-family: "SF Mono", "Fira Code", monospace;
                font-size: 13px; line-height: 1.7; white-space: pre-wrap;
                border-bottom: 1px solid var(--border); color: var(--text);
                border-left: 3px solid var(--border); position: relative; }
.continuation:last-child { border-bottom: none; }
.sample-meta { font-size: 11px; color: var(--text-dim); float: right; font-family: sans-serif; }
.stopped-eos { color: var(--green); }
.stopped-max { color: var(--orange); }
.z-positive { color: var(--green); font-weight: 600; }
.z-negative { color: var(--red); font-weight: 600; }
.controls { display: flex; gap: 8px; margin-left: auto; }
.controls button { background: var(--card); border: 1px solid var(--border); color: var(--text-dim);
                   padding: 4px 12px; border-radius: 6px; cursor: pointer; font-size: 12px;
                   transition: background 0.15s, color 0.15s; }
.controls button:hover { background: var(--surface); color: var(--text); }
.back-to-top { position: fixed; bottom: 24px; right: 24px;
               background: var(--card); border: 1px solid var(--border);
               color: var(--text-dim); border-radius: 50%; width: 40px; height: 40px;
               display: flex; align-items: center; justify-content: center;
               cursor: pointer; opacity: 0; transition: opacity 0.2s;
               font-size: 18px; z-index: 200; }
.back-to-top.visible { opacity: 1; }
.back-to-top:hover { color: var(--text); border-color: var(--text-dim); }
.lightbox { position: fixed; inset: 0; background: rgba(0,0,0,0.92);
            display: flex; align-items: center; justify-content: center;
            z-index: 1000; cursor: zoom-out; animation: fadeIn 0.15s ease-out; }
.lightbox img { max-width: 95vw; max-height: 95vh; border-radius: 8px; }
@media print {
    :root { --bg: #fff; --surface: #f6f8fa; --card: #eee; --text: #111; --text-dim: #555; --border: #ddd; }
    nav, .controls, .back-to-top { display: none !important; }
    details[open] .detail-content { max-height: none; }
    body { max-width: none; }
    .chart, .plot { cursor: default; }
}
"""

JS = """
function toggleAll(expand) {
    document.querySelectorAll('details').forEach(d => d.open = expand);
}

// Smooth scroll nav links
document.querySelectorAll('nav a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
        e.preventDefault();
        const el = document.querySelector(a.getAttribute('href'));
        if (el) el.scrollIntoView({behavior: 'smooth'});
    });
});

// Scroll-spy: highlight active nav link
const spy = new IntersectionObserver(entries => {
    entries.forEach(e => {
        const link = document.querySelector('nav a[href=\"#' + e.target.id + '\"]');
        if (link) link.classList.toggle('active', e.isIntersecting);
    });
}, { rootMargin: '-20% 0px -70% 0px' });
document.querySelectorAll('section[id]').forEach(s => spy.observe(s));

// Back to top
const btt = document.querySelector('.back-to-top');
if (btt) {
    window.addEventListener('scroll', () => btt.classList.toggle('visible', window.scrollY > 400));
    btt.addEventListener('click', () => window.scrollTo({top: 0, behavior: 'smooth'}));
}

// Lightbox for plots and charts
document.querySelectorAll('.plot, .chart').forEach(img => {
    img.addEventListener('click', () => {
        const overlay = document.createElement('div');
        overlay.className = 'lightbox';
        overlay.innerHTML = '<img src=\"' + img.src + '\" />';
        overlay.addEventListener('click', () => overlay.remove());
        document.body.appendChild(overlay);
    });
});

// Sortable table headers
document.querySelectorAll('th').forEach(th => {
    th.addEventListener('click', () => {
        const table = th.closest('table');
        if (!table) return;
        const tbody = table.querySelector('tbody');
        if (!tbody) return;
        const idx = Array.from(th.parentNode.children).indexOf(th);
        const rows = Array.from(tbody.rows);
        const asc = th.dataset.sort !== 'asc';
        rows.sort((a, b) => {
            const av = a.cells[idx]?.textContent || '', bv = b.cells[idx]?.textContent || '';
            const an = parseFloat(av), bn = parseFloat(bv);
            return isNaN(an) || isNaN(bn) ? av.localeCompare(bv) * (asc?1:-1) : (an-bn) * (asc?1:-1);
        });
        rows.forEach(r => tbody.appendChild(r));
        table.querySelectorAll('th').forEach(t => delete t.dataset.sort);
        th.dataset.sort = asc ? 'asc' : 'desc';
    });
});
"""


def render_html(data: ReportData) -> str:
    sections: list[tuple[str, str, str]] = []

    if data.summary:
        sections.append(("hero", "Overview", render_hero(data.summary)))
    if data.dynamics:
        sections.append(("training", "Training", render_training_section(data.dynamics)))
    if data.geometric_health:
        sections.append(("geometry", "Geometry", render_geometry_section(data.geometric_health)))
    if data.generations:
        sections.append(("generations", "Generations", render_generations_section(data.generations)))
    if data.text_quality:
        sections.append(("text-quality", "Text Quality", render_text_quality_section(data.text_quality)))
    if data.activation_plots:
        sections.append(("activation-geometry", "Activations", render_activation_section(data.activation_plots)))
    if data.concept_results or data.concept_plots:
        sections.append(("concept-geometry", "Concepts", render_concept_section(data.concept_results, data.concept_plots)))

    nav_links = " ".join(f'<a href="#{sid}">{label}</a>' for sid, label, _ in sections)
    body = "\n".join(content for _, _, content in sections)

    ts = data.summary or {}
    meta_parts = []
    if ts.get("tokens_B"):
        meta_parts.append(f'{ts["tokens_B"]:.2f}B tokens')
    if ts.get("max_step"):
        meta_parts.append(f'{ts["max_step"]} steps')
    meta_parts.append(f'Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    meta = " | ".join(meta_parts)

    available = [label for _, label, _ in sections]
    missing_tracks = []
    for label in ["Overview", "Training", "Geometry", "Generations", "Text Quality", "Activations", "Concepts"]:
        if label not in available:
            missing_tracks.append(label)
    missing_note = ""
    if missing_tracks:
        missing_note = f'<p class="stat">Tracks not yet available: {", ".join(missing_tracks)}</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analysis: {html.escape(data.run_name)}</title>
<style>{CSS}</style>
</head>
<body>
<header>
    <h1>{html.escape(data.run_name)}</h1>
    <p class="meta">{html.escape(meta)}</p>
    {missing_note}
</header>
<nav>{nav_links}
    <div class="controls">
        <button onclick="toggleAll(true)">Expand All</button>
        <button onclick="toggleAll(false)">Collapse All</button>
    </div>
</nav>
{body}
<div class="back-to-top" title="Back to top">&#8593;</div>
<script>{JS}</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")

    p = argparse.ArgumentParser(description="Generate HTML analysis report")
    p.add_argument("run_name", type=str, help="Run name (directory under analysis/)")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    args = p.parse_args()

    data = load_report_data(args.run_name, args.analysis_dir)

    has_any = any([data.summary, data.dynamics, data.geometric_health,
                   data.generations, data.text_quality, data.activation_plots,
                   data.concept_results, data.concept_plots])
    if not has_any:
        logger.error("No analysis outputs found for '%s' in %s", args.run_name, args.analysis_dir)
        sys.exit(1)

    found = []
    if data.summary: found.append("T1+2 summary")
    if data.dynamics: found.append("T1+2 dynamics")
    if data.geometric_health: found.append("T1+2 geometry")
    if data.generations: found.append("T3 generations")
    if data.text_quality: found.append("T4 text quality")
    if data.activation_plots: found.append(f"T5 plots ({len(data.activation_plots)})")
    if data.concept_results: found.append("T6 concept results")
    if data.concept_plots: found.append(f"T6 plots ({len(data.concept_plots)})")
    logger.info("Found: %s", ", ".join(found))

    report_html = render_html(data)

    output_path = args.output or (data.run_dir / "report.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html)
    logger.info("Report written to %s (%.1f KB)", output_path, len(report_html) / 1024)


if __name__ == "__main__":
    main()
