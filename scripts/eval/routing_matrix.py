#!/usr/bin/env python3
"""Who reads what: the full AttnRes routing matrix (site × source) of checkpoints, offline.

Telemetry keeps the relative split per site (``attnres/route/…``); this reads the
whole matrix — every routing site (each layer's pre_attn / pre_mlp, and the final
aggregation) × every source (embedding, each finished block, the site's own
in-progress block) — on held-out text, per domain and per token-position bucket,
through the parity-tested capture walker (``instrumented_forward(routing_only=True)``).

Per checkpoint it writes ``<out-dir>/<run>/step_<N>.json`` and appends one line to
``<out-dir>/<run>/points.jsonl`` in the hm_backfill ``--points`` shape, with the
telemetry's relative keys under an ``offline/`` prefix (held-out text, fp32 — not
the training batch, so kept apart from the live keys)::

    offline/attnres/route/layer_{i}/{pre_attn,pre_mlp}/{from_emb,from_prev,from_older,from_self,
                                                        lookback,token_entropy}

Text: Uncheatable Eval domains (fresh web text, never trained on), the first
documents in data order until ``--tokens-per-domain``, packed into ``[BOS] + chunk``
sequences of ``--seq-len`` — the same text at every checkpoint.

Usage (a run's weights trajectory, CPU)::

    tools/run_py.sh scripts/eval/routing_matrix.py --run a0-fixed \\
        --checkpoint <run>/weights/step_00000119.pt --checkpoint <run>/weights/step_00001192.pt \\
        --config-section 1b --attn-res-boundaries dd3b --rope-fp32 --threads 24 \\
        --out-dir /tmp/claude-output/routing-matrix
    # plots over finished JSONs (no model):
    python scripts/eval/routing_matrix.py --plot --out-dir /tmp/claude-output/routing-matrix
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("routing_matrix")

POS_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("0", 0, 1), ("1-15", 1, 16), ("16-127", 16, 128), ("128+", 128, 1 << 30))
TOKENS_PER_STEP = 2_097_152  # ladder global batch; only for tokens_seen on the points


@dataclass
class SiteInfo:
    layer: int
    sublayer: str
    labels: list[str]
    rel: list[str]
    end_layer: list[int]


class Acc:
    """Running sums per site: weight per source (overall + per position bucket),
    per-token entropy, token counts."""

    def __init__(self) -> None:
        self.w: list[torch.Tensor] = []
        self.pos: list[dict[str, torch.Tensor]] = []
        self.pos_n: dict[str, int] = {b: 0 for b, _, _ in POS_BUCKETS}
        self.ent: list[float] = []
        self.n = 0

    def add(self, routing: list[tuple[str, int, torch.Tensor]], rr: Any) -> None:
        T = routing[0][2].shape[-1]
        if not self.w:
            self.w = [torch.zeros(w.shape[0], dtype=torch.float64) for _, _, w in routing]
            self.pos = [{b: torch.zeros(w.shape[0], dtype=torch.float64) for b, _, _ in POS_BUCKETS}
                        for _, _, w in routing]
            self.ent = [0.0] * len(routing)
        for k, (_, _, w) in enumerate(routing):
            w2 = w.double().reshape(w.shape[0], -1)          # (n_src, B*T), B = 1
            self.w[k] += w2.sum(1)
            self.ent[k] += rr.token_entropy(w) * w2.shape[1]
            for b, lo, hi in POS_BUCKETS:
                if lo < T:
                    self.pos[k][b] += w2[:, lo:min(hi, T)].sum(1)
        for b, lo, hi in POS_BUCKETS:
            self.pos_n[b] += max(0, min(hi, T) - lo)
        self.n += T

    def result(self) -> dict[str, Any]:
        return {
            "n_tokens": self.n,
            "mean": [(w / self.n).tolist() for w in self.w],
            "token_entropy": [e / self.n for e in self.ent],
            "pos": {b: [(p[b] / self.pos_n[b]).tolist() for p in self.pos] if self.pos_n[b] else None
                    for b, _, _ in POS_BUCKETS},
        }


def route_points(sites: list[SiteInfo], mean: list[list[float]], ent: list[float],
                 rr: Any, prefix: str = "offline/attnres/route") -> dict[str, float]:
    """Telemetry-shaped relative keys from a site-mean matrix."""
    out: dict[str, float] = {}
    for s, m, e in zip(sites, mean, ent):
        site = rr.Site(s.layer, s.sublayer, s.layer if s.sublayer != "final" else max(s.end_layer),
                       tuple(s.labels), tuple(s.end_layer), tuple(s.rel))
        key = f"{prefix}/final" if s.sublayer == "final" else f"{prefix}/layer_{s.layer}/{s.sublayer}"
        w = torch.tensor(m)
        for src, v in rr.relative_split(w, site).items():
            out[f"{key}/from_{src}"] = v
        out[f"{key}/lookback"] = rr.lookback(w, site)
        out[f"{key}/token_entropy"] = e
    return out


def step_of(path: str, meta: dict[str, Any]) -> int:
    import re
    if meta.get("step") is not None:
        return int(meta["step"])
    m = re.search(r"step[_-]?0*(\d+)", Path(path).name)
    if not m:
        raise SystemExit(f"{path}: no step in checkpoint or file name; pass --step")
    return int(m.group(1))


def build_sequences(domains: dict[str, list[str]], encode: Any, bos: int, seq_len: int,
                    tokens_per_domain: int) -> dict[str, list[list[int]]]:
    """First docs in data order until the budget; packed into [BOS] + (seq_len-1) chunks."""
    out: dict[str, list[list[int]]] = {}
    for name, docs in domains.items():
        stream: list[int] = []
        for d in docs:
            stream.extend(encode(d))
            if len(stream) >= tokens_per_domain:
                break
        stream = stream[:tokens_per_domain]
        body = seq_len - 1
        out[name] = [[bos] + stream[i:i + body] for i in range(0, len(stream) - body + 1, body)]
        if not out[name]:
            raise SystemExit(f"domain {name}: fewer than {body} tokens available")
    return out


def run_capture(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from kotodama.eval.model_loader import load_model
    from kotodama.eval.uncheatable import DEFAULT_DATASET, load_domains, resolve_arch
    from kotodama.model import routing_read as rr
    from kotodama.model.capture import instrumented_forward
    from kotodama.presets import TOKENIZER_NAME

    torch.set_num_threads(args.threads)
    domains = load_domains(args.data_dir or os.environ.get("KOTODAMA_UNCHEATABLE_DATA", DEFAULT_DATASET))
    if args.domains:
        want = [d.strip() for d in args.domains.split(",") if d.strip()]
        missing = [d for d in want if d not in domains]
        if missing:
            raise SystemExit(f"unknown domains {missing}; have {sorted(domains)}")
        domains = {d: domains[d] for d in want}
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    seqs = build_sequences(domains, lambda t: tok(t, add_special_tokens=False)["input_ids"],
                           int(tok.bos_token_id), args.seq_len, args.tokens_per_domain)
    logger.info("text: %d domains × %d seqs × %d tokens", len(seqs),
                min(len(v) for v in seqs.values()), args.seq_len)

    run_dir = Path(args.out_dir) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    for ckpt_path in args.checkpoint:
        section, bounds = resolve_arch(ckpt_path, args.config_section, args.attn_res_boundaries)
        model_cfg: dict[str, Any] = {"attn_res": True, "attn_res_boundaries": bounds,
                                     "attn_impl": "sdpa"}
        meta: dict[str, Any] = {}
        t0 = time.time()
        try:
            model = load_model(ckpt_path, config_section=section, attn_res_config=model_cfg,
                               device=args.device, meta=meta,
                               overrides={"rope_fp32": True} if args.rope_fp32 else None)
        except Exception as e:  # keep the sweep going; the missing step is visible in the output
            logger.error("SKIP %s: load failed: %s: %s", ckpt_path, type(e).__name__, e)
            continue
        model = model.float()
        step = args.step if args.step is not None else step_of(ckpt_path, meta)
        out_json = run_dir / f"step_{step:08d}.json"
        if out_json.exists() and not args.overwrite:
            logger.info("exists, skipping: %s", out_json)
            continue
        ranges = [tuple(r) for r in model._attn_res_block_ranges]
        sites: list[SiteInfo] | None = None
        per_dom: dict[str, Acc] = {}
        with torch.no_grad():
            for dom, dseqs in seqs.items():
                acc = Acc()
                for s in dseqs:
                    ids = torch.tensor([s], dtype=torch.long, device=args.device)
                    _, cap = instrumented_forward(model, ids, routing_only=True)
                    routing = [(sub, layer, w.cpu()) for sub, layer, w in cap["routing"]]
                    if sites is None:
                        sites = []
                        for sub, layer, w in routing:
                            lay = rr.site_layout(ranges, layer, sub, w.shape[0])
                            sites.append(SiteInfo(layer, sub, list(lay.labels), list(lay.rel),
                                                  list(lay.end_layer)))
                    acc.add(routing, rr)
                per_dom[dom] = acc
        assert sites is not None
        dom_res = {d: a.result() for d, a in per_dom.items()}
        # domain-balanced mean: every domain weighs the same
        n_sites = len(sites)
        bal_mean = [[sum(dom_res[d]["mean"][k][j] for d in dom_res) / len(dom_res)
                     for j in range(len(sites[k].labels))] for k in range(n_sites)]
        bal_ent = [sum(dom_res[d]["token_entropy"][k] for d in dom_res) / len(dom_res)
                   for k in range(n_sites)]
        bal_pos = {b: [[sum(dom_res[d]["pos"][b][k][j] for d in dom_res) / len(dom_res)
                        for j in range(len(sites[k].labels))] for k in range(n_sites)]
                   for b, _, _ in POS_BUCKETS if all(dom_res[d]["pos"][b] for d in dom_res)}
        result = {
            "run": args.run, "checkpoint": str(ckpt_path), "step": step,
            "tokens_seen": step * TOKENS_PER_STEP, "block_ranges": ranges,
            "seq_len": args.seq_len, "tokens_per_domain": args.tokens_per_domain,
            "dtype": "fp32", "sites": [asdict(s) for s in sites],
            "all": {"mean": bal_mean, "token_entropy": bal_ent, "pos": bal_pos},
            "domains": dom_res,
        }
        out_json.write_text(json.dumps(result))
        with (run_dir / "points.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, "tokens": step * TOKENS_PER_STEP,
                                "metrics": route_points(sites, bal_mean, bal_ent, rr)}) + "\n")
        logger.info("%s step %d: %d sites, %.0fs -> %s", args.run, step, n_sites,
                    time.time() - t0, out_json)
        del model


# ── plots ─────────────────────────────────────────────────────────────────────

def _matrix(res: dict[str, Any], key: str = "mean", bucket: str | None = None):
    """(site_labels, source_labels, rows) with sources aligned to absolute columns:
    emb, each finished block, then 'self' (the in-progress block)."""
    blocks = [s for s in res["sites"] if s["sublayer"] == "final"][0]["labels"]
    cols = blocks[:-1] + ["self"]
    vals = res["all"][key] if bucket is None else res["all"]["pos"][bucket]
    rows, names = [], []
    for s, m in zip(res["sites"], vals):
        row = [math.nan] * len(cols)
        for lab, v in zip(s["labels"], m):
            row[cols.index("self") if lab.startswith("self:") else cols.index(lab)] = v
        rows.append(row)
        names.append("final" if s["sublayer"] == "final"
                     else f"L{s['layer']} {'attn' if s['sublayer'] == 'pre_attn' else 'mlp'}")
    return names, cols, rows


def run_plot(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(args.out_dir)
    runs = sorted(d.name for d in root.iterdir() if d.is_dir() and list(d.glob("step_*.json")))
    if not runs:
        raise SystemExit(f"no step_*.json under {root}")
    latest = {r: json.loads(sorted((root / r).glob("step_*.json"))[-1].read_text()) for r in runs}
    # 1) final-checkpoint matrix per run, side by side
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 12), squeeze=False)
    for ax, r in zip(axes[0], runs):
        names, cols, rows = _matrix(latest[r])
        im = ax.imshow(np.array(rows), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=60, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=6)
        ax.set_title(f"{r} step {latest[r]['step']}", fontsize=9)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.4, label="token-mean routing weight")
    fig.savefig(root / "matrix_latest.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    # 2) relative split over training, per run: one panel per relative source, layer × step
    for r in runs:
        pts = [json.loads(line) for line in (root / r / "points.jsonl").read_text().splitlines()]
        pts.sort(key=lambda p: p["step"])
        steps = [p["step"] for p in pts]
        fig, axes = plt.subplots(2, 6, figsize=(24, 8), squeeze=False)
        for row, sub in enumerate(("pre_attn", "pre_mlp")):
            for c, q in enumerate(("from_emb", "from_prev", "from_older", "from_self",
                                   "lookback", "token_entropy")):
                grid = np.array([[p["metrics"].get(f"offline/attnres/route/layer_{i}/{sub}/{q}", np.nan)
                                  for p in pts] for i in range(28)])
                ax = axes[row][c]
                vmax = None if q == "lookback" else 1
                im = ax.imshow(grid, aspect="auto", cmap="magma" if q == "lookback" else "viridis",
                               vmin=0, vmax=vmax, origin="lower")
                ax.set_title(f"{sub} {q}", fontsize=9)
                ax.set_xticks(range(0, len(steps), max(1, len(steps) // 6)),
                              [steps[i] for i in range(0, len(steps), max(1, len(steps) // 6))],
                              fontsize=7, rotation=45)
                ax.set_ylabel("layer", fontsize=8)
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f"{r}: who reads what over training (offline, held-out text)")
        fig.savefig(root / f"relative_{r}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"plots -> {root}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plot", action="store_true", help="render plots from existing JSONs")
    ap.add_argument("--run", help="run label (output subdir)")
    ap.add_argument("--checkpoint", action="append", default=[])
    ap.add_argument("--config-section", default=None)
    ap.add_argument("--attn-res-boundaries", default=None)
    ap.add_argument("--rope-fp32", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--domains", default=None)
    ap.add_argument("--tokens-per-domain", type=int, default=4092)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--step", type=int, default=None, help="override (single checkpoint)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.plot:
        run_plot(args)
        return
    if not args.run or not args.checkpoint:
        ap.error("--run and at least one --checkpoint are required (or --plot)")
    if args.step is not None and len(args.checkpoint) > 1:
        ap.error("--step overrides a single checkpoint only")
    run_capture(args)


if __name__ == "__main__":
    sys.exit(main())
