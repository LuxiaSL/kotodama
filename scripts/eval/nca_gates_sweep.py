"""NCA gate sweep: score every checkpoint of a run -> the formation curve.

Runs ``kotodama.eval.nca_gates`` on each ``step_*.pt[.zst]`` under a
checkpoint dir (plus, optionally, the run's untrained init as step 0), writes
one JSON per step, and a summary (JSON + TSV + printed table) of ledger acc vs
tokens against chance. Resumable: steps with an existing JSON are reused.

AttnRes boundaries are always passed explicitly (the loader's auto-detect
assumes n_blocks=7, wrong for DD-v1): default = the shape's
``presets.DEFAULT_BOUNDARIES`` entry.

Usage::

  TMPDIR=<roomy scratch> tools/run_py.sh -m scripts.eval.nca_gates_sweep \
      --ckpt-dir <run ckpt dir> --eval-dir <v3 data dir> --init-seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch

from kotodama import ckpt, presets
from kotodama.eval.nca_gates import LEDGER_FILES, score_eval_dir

STEP_RE = re.compile(r"^step_(\d+)\.pt(\.zst)?$")
INIT_STEP = 0  # the untrained init, when --init-seed is given


def list_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """(step, path) sorted by step; a plain .pt wins over its .pt.zst twin."""
    found: dict[int, Path] = {}
    for p in ckpt_dir.iterdir():
        m = STEP_RE.match(p.name)
        if m and (int(m.group(1)) not in found or not m.group(2)):
            found[int(m.group(1))] = p
    return sorted(found.items())


def parse_boundaries(spec: str | None, section: str) -> list[int] | None:
    """None = AttnRes off. Accepts a preset name, 'off', or '0,3,7,...'."""
    if spec is None:
        try:
            b = presets.DEFAULT_BOUNDARIES.get(presets.resolve(section))
        except KeyError:
            raise SystemExit(f"--config-section {section!r} is not a preset: "
                             "pass --boundaries explicitly") from None
        return list(b) if b else None
    if spec == "off":
        return None
    if spec in presets.BOUNDARIES:
        return list(presets.BOUNDARIES[spec])
    return [int(x) for x in spec.split(",")]


def model_overrides(boundaries: list[int] | None, attn_impl: str) -> dict[str, Any]:
    ov: dict[str, Any] = {"attn_res": boundaries is not None, "attn_impl": attn_impl}
    if boundaries is not None:
        ov["attn_res_boundaries"] = boundaries
    return ov


def build_init(section: str, overrides: dict[str, Any], seed: int,
               device: str, config_path: Path | None = None) -> torch.nn.Module:
    """The run's step-0 weights: train.py seeds with --seed, then builds."""
    from kotodama.eval.model_loader import load_model_config
    from kotodama.model.llama import KotodamaModel, KotodamaModelConfig
    torch.manual_seed(seed)
    model = KotodamaModel(KotodamaModelConfig(
        **{**load_model_config(config_path, section), "qk_norm": True,
           **overrides})).eval().to(device)
    return model.bfloat16() if "cuda" in device else model


def ckpt_tokens(path: Path) -> int | None:
    """tokens_consumed recorded by train.py (mmap load: no full read)."""
    try:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        return int(state["tokens_consumed"])
    except Exception:  # legacy / non-zip checkpoints: tokens stay unknown
        return None


def summarize(per_step: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in sorted(per_step, key=lambda r: r["step"]):
        row: dict[str, Any] = {"step": r["step"], "tokens": r.get("tokens"),
                               "is_init": str(r.get("checkpoint", "")).startswith("init")}
        for name in LEDGER_FILES:
            agg = r["ledger"].get(name, {}).get("aggregate", {}).get("all")
            if agg:
                key = name.removeprefix("eval_").removesuffix(".jsonl")
                row[f"{key}_acc"] = agg["acc"]
                row[f"{key}_medrank"] = agg["med_rank"]
                row[f"{key}_n"] = agg["n"]
        smo = (r.get("smoothing") or {}).get("aggregate")
        row["si"] = smo["si_mean"] if smo else None
        rows.append(row)
    return rows


def format_table(rows: list[dict[str, Any]], chance: float) -> str:
    cols = [k for k in rows[0] if k.endswith("_acc")] if rows else []
    lines = ["step\ttokens\t" + "\t".join(cols) + "\tsi"]
    for r in rows:
        tok = f"{r['tokens'] / 1e6:.2f}M" if r["tokens"] is not None else "?"
        si = f"{r['si']:.3f}" if r["si"] is not None else "-"
        lines.append(f"{r['step']}\t{tok}\t"
                     + "\t".join(f"{r.get(c, float('nan')):.3f}" for c in cols)
                     + f"\t{si}")
    trained = [r for r in rows if not r["is_init"]]
    for c in cols:
        if not trained:
            break
        best = max(trained, key=lambda r: r.get(c, -1.0))
        n = best.get(c.replace("_acc", "_n"), 0)
        bar = chance + 1.645 * math.sqrt(chance * (1 - chance) / max(n, 1))
        lines.append(f"# {c}: max {best[c]:.3f} @ step {best['step']} | chance "
                     f"{chance:.3f} (L-P1: {'ABOVE' if best[c] > chance else 'NOT above'}) "
                     f"| one-sided p<.05 bar at n={n}: {bar:.3f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <ckpt-dir>/gates")
    ap.add_argument("--config-section", default="proxy",
                    help="preset shape (or a section of --config-path)")
    ap.add_argument("--config-path", type=Path, default=None,
                    help="model YAML for non-preset shapes (smoke tests)")
    ap.add_argument("--boundaries", default=None,
                    help="preset name | 'off' | '0,3,7,...'; default = shape's")
    ap.add_argument("--init-seed", type=int, default=None,
                    help="also score the untrained init (train.py --seed) as step 0")
    ap.add_argument("--attn-impl", default=None,
                    help="default: auto on cuda, sdpa on cpu")
    ap.add_argument("--device", default=None)
    ap.add_argument("--reserve-eos", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--force", action="store_true", help="rescore existing steps")
    ap.add_argument("--keep-decompressed", action="store_true")
    args = ap.parse_args()
    # load_model reports missing/unexpected keys as warnings: keep them visible
    logging.basicConfig(level=logging.INFO)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    overrides = model_overrides(
        parse_boundaries(args.boundaries, args.config_section),
        args.attn_impl or ("auto" if "cuda" in device else "sdpa"))
    out_dir = args.out_dir or args.ckpt_dir / "gates"
    out_dir.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[int, Path | None]] = list_checkpoints(args.ckpt_dir)
    if args.init_seed is not None:
        todo.insert(0, (INIT_STEP, None))
    if not todo:
        print(f"ERROR: no step_*.pt[.zst] under {args.ckpt_dir}", file=sys.stderr)
        return 1

    per_step = []
    for step, path in todo:
        out = out_dir / f"gates_step{step:08d}.json"
        if out.exists() and not args.force:
            per_step.append(json.loads(out.read_text()))
            continue
        tag = f"step{step}"
        if path is None:
            model = build_init(args.config_section, overrides, args.init_seed,
                               device, args.config_path)
            tokens, src = 0, f"init(seed={args.init_seed})"
        else:
            from kotodama.eval.model_loader import load_model
            plain = ckpt.decompress_cached(path)
            model = load_model(plain, config_path=args.config_path,
                               config_section=args.config_section,
                               attn_res_config=overrides, device=device)
            tokens, src = ckpt_tokens(plain), str(path)
        res = {"step": step, "tokens": tokens, "checkpoint": src,
               **score_eval_dir(model, args.eval_dir, tag, device, args.reserve_eos)}
        out.write_text(json.dumps(res))
        per_step.append(res)
        del model
        # decompress_cached writes to $TMPDIR/kotodama_checkpoints/; drop our
        # copy (15 proxy ckpts would pile up there), never a file beside the run
        if (path is not None and not args.keep_decompressed
                and plain.parent.name == "kotodama_checkpoints"
                and plain.parent != path.parent):
            plain.unlink(missing_ok=True)

    rows = summarize(per_step)
    chance = next((a["aggregate"]["chance_acc"] for r in per_step
                   for a in r["ledger"].values()), 0.1)
    table = format_table(rows, chance)
    (out_dir / "summary.json").write_text(json.dumps(
        {"chance_acc": chance, "rows": rows}, indent=2))
    (out_dir / "summary.tsv").write_text(table + "\n")
    print(table)
    print(f"wrote {out_dir}/summary.{{json,tsv}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
