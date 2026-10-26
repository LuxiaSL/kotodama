"""Binding-suite CLI.

    python -m kotodama.eval.binding list
    python -m kotodama.eval.binding run --model koto:/ckpts/step_00041960.pt --name tsumugi-a88B \\
        --batteries scorecard --out binding-scores [--prompts-dir DIR] [--device cuda] [--batch 32]
    python -m kotodama.eval.binding run --model hf:EleutherAI/pythia-1.4b --name pythia-1.4b --batteries all --out …
    python -m kotodama.eval.binding report --scores binding-scores [--models a,b] [--json OUT.json]
    python -m kotodama.eval.binding capacity --scores binding-scores --params params.tsv [--ours a,b]
    python -m kotodama.eval.binding gen-capacity --out capacity-v1.jsonl

``run`` loads the model ONCE and scores every requested battery into ``OUT/<battery>/<name>.jsonl`` (resumable).
Prompt-row batteries are read by ``report``; bindprobes batteries by ``python -m kotodama.eval.bindprobes aggregate
--grids OUT/<battery>``. Model specs: see :mod:`kotodama.eval.binding.models`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

from kotodama.eval.binding import batteries as bt
from kotodama.eval.binding import readers
from kotodama.eval.binding.score import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m kotodama.eval.binding", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="batteries and groups")
    r = sub.add_parser("run", help="score batteries with one model")
    r.add_argument("--model", required=True, help="koto:CKPT[?size=…&bounds=…&attn_res=…] | hf:PATH")
    r.add_argument("--name", required=True, help="model name in rows and output file names")
    r.add_argument("--batteries", required=True, help="comma list of batteries / groups (see `list`)")
    r.add_argument("--out", required=True)
    r.add_argument("--prompts-dir", default=None)
    r.add_argument("--device", default="cuda")
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--no-pin-check", action="store_true", help="score files that differ from their pins (debug)")
    p = sub.add_parser("report", help="the scorecard from scored prompt-row batteries")
    p.add_argument("--scores", required=True, help="a run --out directory")
    p.add_argument("--models", default="", help="comma list (default: every model found)")
    p.add_argument("--json", default="", help="also write per-model reader results here")
    c = sub.add_parser("capacity", help="capacity curves, K50, and the K50 ∝ N^α size law")
    c.add_argument("--scores", required=True, help="a run --out directory (reads <scores>/capacity-v1)")
    c.add_argument("--params", required=True, help="TSV: name, params in billions, note")
    c.add_argument("--ours", default="", help="comma names placed on the law instead of fitted into it")
    g = sub.add_parser("gen-capacity", help="write the capacity-v1 prompt rows")
    g.add_argument("--out", required=True)
    return ap


def cmd_run(a: argparse.Namespace) -> int:
    from kotodama.eval.binding import models, score

    bats = bt.resolve(a.batteries)
    root = bt.prompts_dir(a.prompts_dir)
    paths = {b.name: bt.path_for(b, root, check=not a.no_pin_check) for b in bats}   # fail before loading weights
    model = models.load(models.ModelSpec.parse(a.model), device=a.device)
    out = Path(a.out)
    for b in bats:
        dest = out / b.name / f"{a.name}.jsonl"
        if b.kind == "rows":
            score.score_prompt_rows(model, read_jsonl(paths[b.name]), dest, a.name, max_batch=a.batch)
        else:
            from kotodama.eval.bindprobes.items import CONDITIONS, load_items
            items = load_items(paths[b.name])
            if b.families:
                items = [it for it in items if it.family in b.families]
            score.score_bindprobe_items(model, items, b.conditions or CONDITIONS, dest, a.name, max_batch=a.batch)
    return 0


def _models_in(scores: Path) -> list[str]:
    return sorted({p.stem for p in scores.glob("*/*.jsonl")})


def cmd_report(a: argparse.Namespace) -> int:
    scores = Path(a.scores)
    names = [m for m in a.models.split(",") if m] or _models_in(scores)
    rows_out: list[tuple[str, dict[str, str]]] = []
    dump: dict[str, Any] = {}
    for m in names:
        res: dict[str, Any] = {}
        for b in bt.BATTERIES.values():
            f = scores / b.name / f"{m}.jsonl"
            if b.reader is None or not f.exists():
                continue
            rows = read_jsonl(f)
            if rows:
                res[b.reader] = readers.READERS[b.reader](rows)
        if res:
            rows_out.append((m, readers.scorecard_row(res)))
            dump[m] = res
    cols = readers.SCORECARD_COLUMNS
    print("| model | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for m, row in rows_out:
        print(f"| {m} | " + " | ".join(row[c] for c in cols) + " |")
    if a.json:
        Path(a.json).write_text(json.dumps(dump, indent=1, default=str))
    return 0


def cmd_capacity(a: argparse.Namespace) -> int:
    from kotodama.eval.binding import capacity as cap

    params: dict[str, tuple[float, str]] = {}
    for line in Path(a.params).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, n, *note = line.split("\t")
            params[name] = (float(n), note[0] if note else "")
    ours = {x for x in a.ours.split(",") if x}
    sums = {}
    for f in sorted((Path(a.scores) / "capacity-v1").glob("*.jsonl")):
        rows = read_jsonl(f)
        if rows:
            sums[f.stem] = cap.summarize(rows)
    order = sorted(sums, key=lambda m: params.get(m, (math.inf, ""))[0])
    print(f"{'model':28} {'N(B)':>6} {'C':>6} {'K50':>6} {'beta':>5} {'AUC':>5} {'acc@8':>6} | "
          f"{'C D256':>6} {'K50fix':>6} {'K50own':>6} {'acc@8':>6} | multi")
    for m in order:
        s = sums[m]
        n = params.get(m, (math.nan, ""))[0]
        flag = "" if s.usable else "  (ceiling ~chance: excluded)"
        print(f"{m:28} {n:6.2f} {s.ceiling:6.3f} {s.fit.label():>6} {s.fit.beta:5.2f} {s.auc:5.3f} "
              f"{(s.acc8 or math.nan):6.3f} | {(s.ceiling_d256 or math.nan):6.3f} "
              f"{s.fit_d256_fixed.label() if s.fit_d256_fixed else '—':>6} "
              f"{s.fit_d256_own.label() if s.fit_d256_own else 'n/a':>6} {(s.acc8_d256 or math.nan):6.3f} | "
              f"{s.multi_token_rows}{flag}")
    if sums:
        ks = sorted(next(iter(sums.values())).curve)
        print("\nnormalised recall r(K), D=0:\n" + f"{'model':28} " + " ".join(f"{k:>5}" for k in ks))
        for m in order:
            print(f"{m:28} " + " ".join(f"{sums[m].curve.get(k, math.nan):5.2f}" for k in ks))
    pts = [(params[m][0], s.fit.k50) for m, s in sums.items()
           if m in params and m not in ours and s.usable and not s.fit.censored]
    try:
        law = cap.fit_law(pts)
    except ValueError as e:
        print(f"\nlaw: {e}")
        return 0
    print(f"\nlaw on {law.n} baselines: ln K50 = {law.c:.3f} + {law.alpha:.3f} ln N(B)   R² {law.r2:.2f}   "
          f"(paper, its own harness: α = .82)")
    if law.n < 5 or law.r2 < 0.5 or law.alpha < 0.1:
        print(f"  no N_eff: the law is too weak to invert (need ≥ 5 baselines, R² ≥ .5, α ≥ .1)")
        return 0
    for m in sorted(ours & set(sums)):
        s = sums[m]
        actual = params.get(m, (math.nan, ""))[0]
        if s.fit.censored:
            print(f"  {m}: K50 > {cap.K_GRID[-1]} (censored) → N_eff > {law.n_eff(cap.K_GRID[-1]):.1f}B (actual {actual:.2f}B)")
        else:
            print(f"  {m}: K50 {s.fit.k50:.1f} → N_eff {law.n_eff(s.fit.k50):.2f}B (actual {actual:.2f}B)")
    return 0


def cmd_gen_capacity(a: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from kotodama import presets
    from kotodama.eval.binding import capacity as cap

    tok = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
    rows = cap.generate(lambda s: len(tok.encode(s, add_special_tokens=False)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pin = bt.BATTERIES["capacity-v1"].sha256
    got = bt.sha256(out)
    print(f"wrote {len(rows)} rows → {out}; sha256 {got[:16]}… {'= pin' if got == pin else '≠ pin ' + pin[:16] + '…'}")
    return 0 if got == pin else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    a = build_parser().parse_args(argv)
    if a.cmd == "list":
        for b in bt.BATTERIES.values():
            print(f"{b.name:20} {b.kind:10} {b.file:42} {b.what}")
        print("\ngroups: " + "; ".join(f"{k} = {','.join(v)}" for k, v in bt.GROUPS.items() if k != "all"))
        return 0
    return {"run": cmd_run, "report": cmd_report, "capacity": cmd_capacity,
            "gen-capacity": cmd_gen_capacity}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
