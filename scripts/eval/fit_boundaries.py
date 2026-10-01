#!/usr/bin/env python3
"""Fit AttnRes block boundaries on a no-AttnRes run, snapshot by snapshot, until they settle.

The 2026-10-01 re-fit (Luxia): boundaries are read where LANGUAGE circuits settle, on a
no-AttnRes counterfactual (``configs/ladder-cf-lang.yaml``; leg B adds a safe-dose NCA
phase first), with the probe on held-out language text — not on NCA geometry with a
random-token probe, as DD-3B was. Per snapshot, four fits:

- ``v1_legacy``: the archived algorithm (``nca_profile.suggest_boundaries``) on the
  archived entropy ruler (raw q/k, no QK-norm/RoPE) — the DD-3B procedure, for reference;
- ``v1_true``: same algorithm, real attention entropy;
- ``rules_true``: same transition scores, picked under the DD-3B hand rules made explicit
  (``--forced 1`` isolates the embedding layer, ``--min-last-block`` cushions the output);
- ``rules_cka``: independent signal — 1 - linear CKA between adjacent layers' outputs,
  picked under the same rules;
- ``*_nomid``: the rules fits without the middle-block rule (``--min-mid-block``: the
  block holding the midpoint layer keeps >= K layers, Luxia's "contiguous middle"), to
  show whether that rule binds or the data already agrees.

Stop rule (``summary.json``): a fit is SETTLED at the first snapshot whose boundaries are
identical for ``--settle`` consecutive snapshots (also reported with ±1-layer tolerance).

Usage::

    tools/run_py.sh scripts/eval/fit_boundaries.py --run cf-lang \\
        --checkpoint <ladder>/cf-lang/weights/step_*.pt --config-section 1b --rope-fp32 \\
        --out-dir /tmp/claude-output/boundary-fit
    python scripts/eval/fit_boundaries.py --summarize --run cf-lang --out-dir ...
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nca_profile as npf  # noqa: E402

logger = logging.getLogger("fit_boundaries")
METHODS = ("v1_legacy", "v1_true", "rules_true", "rules_cka", "rules_true_nomid", "rules_cka_nomid")


def fits_for(profiles: dict[str, dict[str, list[float]]], cka_scores: list[float], n: int,
             args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ruler in ("legacy", "true"):
        b, combined = npf.suggest_boundaries(profiles[ruler], n, args.v1_min_block, args.v1_max_boundaries)
        out[f"v1_{ruler}"] = {"boundaries": b, "scores": combined}
    true_scores = out["v1_true"]["scores"]
    rules = dict(min_block_size=args.min_block, max_boundaries=args.max_boundaries,
                 forced=tuple(args.forced), min_last_block=args.min_last_block)
    for name, sc in (("true", true_scores), ("cka", cka_scores)):
        out[f"rules_{name}"] = {"boundaries": npf.select_boundaries(
            sc, n, min_mid_block=args.min_mid_block, **rules), "scores": sc}
        # the same pick without the middle rule: does the rule bind, or does the data agree?
        out[f"rules_{name}_nomid"] = {"boundaries": npf.select_boundaries(sc, n, **rules), "scores": sc}
    return out


def settled_at(series: list[tuple[int, list[int]]], k: int, tol: int = 0) -> int | None:
    """First step whose boundaries match the next k-1 snapshots (each boundary within tol)."""
    def same(a: list[int], b: list[int]) -> bool:
        return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))
    for i in range(len(series) - k + 1):
        if all(same(series[i][1], series[j][1]) for j in range(i + 1, i + k)):
            return series[i][0]
    return None


def summarize(run_dir: Path, k: int) -> dict[str, Any]:
    snaps = sorted((json.loads(p.read_text()) for p in run_dir.glob("step_*.json")),
                   key=lambda r: r["step"])
    if not snaps:
        raise SystemExit(f"no step_*.json in {run_dir}")
    summ: dict[str, Any] = {"run": run_dir.name, "steps": [s["step"] for s in snaps], "methods": {}}
    for m in METHODS:
        series = [(s["step"], s["fits"][m]["boundaries"]) for s in snaps]
        summ["methods"][m] = {"series": series, "settled_at": settled_at(series, k),
                              "settled_at_tol1": settled_at(series, k, tol=1),
                              "latest": series[-1][1]}
    (run_dir / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"== {run_dir.name}: boundaries per snapshot (settle = {k} identical in a row)")
    for m, d in summ["methods"].items():
        print(f"  {m:10s} settled_at={d['settled_at']} (±1: {d['settled_at_tol1']})  latest={d['latest']}")
        for step, b in d["series"]:
            print(f"      {step:6d}  {b}")
    return summ


def probe_language(args: argparse.Namespace) -> torch.Tensor:
    """Held-out language: Uncheatable docs (first in data order), [BOS] + chunk sequences,
    round-robin over domains — the same text at every snapshot."""
    from transformers import AutoTokenizer

    from kotodama.eval.uncheatable import DEFAULT_DATASET, load_domains
    from kotodama.presets import TOKENIZER_NAME
    import routing_matrix as rmx

    domains = load_domains(args.data_dir or DEFAULT_DATASET)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    per = max(1, -(-args.probe_seqs // len(domains)))
    seqs = rmx.build_sequences(domains, lambda t: tok(t, add_special_tokens=False)["input_ids"],
                               int(tok.bos_token_id), args.seq_len, per * (args.seq_len - 1))
    rows = [s for i in range(per) for d in sorted(seqs) for s in seqs[d][i:i + 1]][:args.probe_seqs]
    return torch.tensor(rows, dtype=torch.long)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", nargs="*", default=[])
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--config-section", default="1b")
    ap.add_argument("--rope-fp32", action="store_true")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--probe-seqs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--probe-bin", default=None, help="raw token windows instead of Uncheatable text")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--hidden-rows", type=int, default=4096)
    # archived-algorithm settings (v1_*): 7 boundaries / min block 1 = what DD-3B's shape needs
    ap.add_argument("--v1-max-boundaries", type=int, default=7)
    ap.add_argument("--v1-min-block", type=int, default=1)
    # explicit hand rules (rules_*)
    ap.add_argument("--max-boundaries", type=int, default=7)
    ap.add_argument("--min-block", type=int, default=2)
    ap.add_argument("--forced", type=int, nargs="*", default=[1])
    ap.add_argument("--min-last-block", type=int, default=4)
    ap.add_argument("--min-mid-block", type=int, default=8,
                    help="block containing the midpoint layer keeps >= this many layers "
                         "(DD-3B's middle block L7-14 = 8; 0 disables)")
    ap.add_argument("--settle", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.out_dir) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(run_dir, args.settle)
        return 0
    if not args.checkpoint:
        ap.error("--checkpoint required (or --summarize)")
    torch.set_num_threads(args.threads)

    from kotodama.eval.model_loader import load_model

    if args.probe_bin:
        ns = argparse.Namespace(probe_bin=args.probe_bin, seq_len=args.seq_len,
                                probe_seqs=args.probe_seqs, seed=0)
        probe = npf._probe(ns, "cpu")
    else:
        probe = probe_language(args)
    logger.info("probe %s (%s)", tuple(probe.shape), args.probe_bin or "uncheatable text")
    for ck in args.checkpoint:
        m = re.search(r"step[_-]?0*(\d+)", Path(ck).name)
        step = int(m.group(1)) if m else -1
        out = run_dir / f"step_{step:08d}.json"
        if out.exists() and not args.overwrite:
            logger.info("exists, skipping %s", out)
            continue
        try:
            model = load_model(ck, config_section=args.config_section,
                               attn_res_config={"attn_res": False, "attn_impl": "sdpa"},
                               device=args.device,
                               overrides={"rope_fp32": True} if args.rope_fp32 else None)
        except Exception as e:  # keep the sweep going; the gap is visible in the summary
            logger.error("SKIP %s: %s: %s", ck, type(e).__name__, e)
            continue
        if args.device == "cpu":
            model = model.float()
        hiddens: list[torch.Tensor] = []
        profiles = npf.extract_profile(model, probe.to(args.device), hidden_sink=hiddens,
                                       hidden_rows=args.hidden_rows)
        n = len(model.layers)
        fits = fits_for(profiles, npf.cka_transition_scores(hiddens), n, args)
        out.write_text(json.dumps({"run": args.run, "checkpoint": ck, "step": step, "num_layers": n,
                                   "probe_shape": list(probe.shape), "settings": {
                                       k: getattr(args, k) for k in ("v1_max_boundaries", "v1_min_block",
                                                                     "max_boundaries", "min_block", "forced",
                                                                     "min_last_block", "min_mid_block")},
                                   "profiles": profiles, "fits": fits}))
        logger.info("step %d: %s", step, {k: v["boundaries"] for k, v in fits.items()})
        del model
    summarize(run_dir, args.settle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
