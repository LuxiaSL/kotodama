"""Binding-suite CLI.

    python -m kotodama.eval.binding list
    python -m kotodama.eval.binding run --model koto:/ckpts/step_00041960.pt --name tsumugi-a88B \\
        --batteries scorecard --out binding-scores [--prompts-dir DIR] [--device cuda] [--batch 32]
    python -m kotodama.eval.binding run --model hf:EleutherAI/pythia-1.4b --name pythia-1.4b --batteries all --out …
    python -m kotodama.eval.binding report --scores binding-scores [--models a,b] [--json OUT.json]
    python -m kotodama.eval.binding capacity --scores binding-scores --params params.tsv [--ours a,b]
    python -m kotodama.eval.binding gen-capacity --out capacity-v1.jsonl
    python -m kotodama.eval.binding gen-fluid --out DIR
    python -m kotodama.eval.binding fluid --scores binding-scores [--models a,b] [--stat midpoint|auc] [--json OUT]
    python -m kotodama.eval.binding check-fluid-tokens [--tokenizers a,b]
    python -m kotodama.eval.binding gen-formupdate --out DIR
    python -m kotodama.eval.binding formupdate --scores binding-scores [--models a,b] [--json OUT]
        [--reference a,b,c --release R --anneal X]

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
    gf = sub.add_parser("gen-fluid", help="write every fluid battery's prompt rows (v1 and v2) into a directory")
    gf.add_argument("--out", required=True)
    fr = sub.add_parser("fluid", help="fluid / crystallized read-out: ladder midpoints, AUC, F3 stale rate, scores")
    fr.add_argument("--scores", required=True, help="a run --out directory")
    fr.add_argument("--models", default="", help="comma list (default: every model found)")
    fr.add_argument("--stat", choices=("midpoint", "auc"), default="midpoint")
    fr.add_argument("--order", choices=("mean_then_z", "z_then_mean"), default="mean_then_z")
    fr.add_argument("--json", default="", help="also write per-model summaries and scores here")
    fr.add_argument("--set", default="v2", choices=("v1", "v2"),
                    help="which battery per family (fluid.FLUID_SETS; v2 = the controlled F2 / F3)")
    ct = sub.add_parser("check-fluid-tokens", help="check the fluid answer pools are one token in every tokenizer")
    ct.add_argument("--tokenizers", default="", help="comma list of HF tokenizer names (default: the zoo)")
    gu = sub.add_parser("gen-formupdate", help="write the formupdate-v1 and -v2 prompt rows into a directory")
    gu.add_argument("--out", required=True)
    fu = sub.add_parser("formupdate", help="form / interference / update costs with paired-bootstrap intervals")
    fu.add_argument("--scores", required=True, help="a run --out directory (reads <scores>/<battery>)")
    fu.add_argument("--battery", default="formupdate-v2", choices=("formupdate-v1", "formupdate-v2"))
    fu.add_argument("--models", default="", help="comma list (default: every model found)")
    fu.add_argument("--json", default="", help="also write per-model summaries (and predictions) here")
    fu.add_argument("--boot", type=int, default=10_000, help="bootstrap resamples")
    fu.add_argument("--reference", default="", help="comma list: the reference set for the predictions")
    fu.add_argument("--release", default="", help="the release model compared against the reference set")
    fu.add_argument("--anneal", default="", help="the checkpoint whose normalized update cost is compared to the "
                                                 "release's")
    return ap


def cmd_run(a: argparse.Namespace) -> int:
    from kotodama.eval.binding import models, score

    bats = bt.resolve(a.batteries)
    root = bt.prompts_dir(a.prompts_dir)
    paths = {b.name: bt.path_for(b, root, check=not a.no_pin_check) for b in bats}   # fail before loading weights
    if not a.no_pin_check:
        for b in bats:
            bt.check_render(b, paths[b.name])
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


def cmd_gen_fluid(a: argparse.Namespace) -> int:
    from kotodama.eval.binding import fluid as fl

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for name in fl.BY_BATTERY:
        rows = fl.generate(name)
        bad = fl.audit_failures(fl.audit(rows))
        b = bt.BATTERIES[name]
        dest = out / b.file
        dest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        got = bt.sha256(dest)
        same = got == b.sha256
        ok = ok and same and not bad
        verdict = "= pin" if same else f"≠ pin {b.sha256[:16]}…"
        print(f"{name:18} {len(rows):5} rows → {dest}; sha256 {got} {verdict}"
              f"{'' if not bad else '; AUDIT: ' + '; '.join(bad)}")
    return 0 if ok else 1


def _fmt(x: Any, fmt: str = "{:.3f}") -> str:
    return "—" if x is None else fmt.format(x)


def cmd_fluid(a: argparse.Namespace) -> int:
    from kotodama.eval.binding import fluid as fl

    scores = Path(a.scores)
    chosen = fl.FLUID_SETS[a.set]                       # family key → battery
    print(f"battery set {a.set}: " + ", ".join(chosen.values()))
    names = [m for m in a.models.split(",") if m] or sorted(
        {p.stem for b in chosen.values() for p in (scores / b).glob("*.jsonl")})
    sums: dict[str, dict[str, fl.FamilySummary]] = {}
    for m in names:
        for key, b in chosen.items():
            f = scores / b / f"{m}.jsonl"
            rows = read_jsonl(f) if f.exists() else []
            if rows:
                sums.setdefault(m, {})[key] = fl.summarize(rows)
    if not sums:
        print(f"no fluid rows under {scores}")
        return 1
    res = fl.fluid_scores(sums, stat=a.stat, order=a.order)
    keys = ("F1", "F2", "F3", "F4", "C0")
    print(f"{'model':28} " + " ".join(f"{k + ' mid':>7} {'AUC':>5}" for k in keys)
          + f" {'stale':>6} {'C0@0':>5} {'C0@6':>5} | {'fluid':>6} {'cryst':>6} {'gap':>6}")
    for m in names:
        if m not in sums:
            continue
        s = sums[m]
        cells = []
        for k in keys:
            x = s.get(k)
            cells.append(f"{x.midpoint.label():>7} {x.auc:5.3f}" if x else f"{'—':>7} {'—':>5}")
        st = s["F3"].stale["rate"] if "F3" in s and s["F3"].stale else None
        shots = (s["C0"].by_shots or {}).get("all", {}) if "C0" in s else {}
        z = " ".join(f"{_fmt(res[m][k], '{:+.2f}'):>6}" for k in ("fluid", "crystallized", "gap"))
        print(f"{m:28} " + " ".join(cells) + f" {_fmt(st):>6} {_fmt(shots.get(0)):>5} {_fmt(shots.get(6)):>5} | {z}")
    if a.json:
        Path(a.json).write_text(json.dumps({"stat": a.stat, "order": a.order, "scores": res,
                                            "summaries": {m: {k: x.as_dict() for k, x in s.items()}
                                                          for m, s in sums.items()}}, indent=1, default=str))
    return 0


def cmd_check_fluid_tokens(a: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from kotodama.eval.binding import fluid as fl

    names = [x for x in a.tokenizers.split(",") if x] or list(fl.ZOO_TOKENIZERS)
    encoders = {}
    for n in names:
        tok = AutoTokenizer.from_pretrained(n)
        encoders[n] = lambda s, t=tok: t.encode(s, add_special_tokens=False)
    bad = fl.check_pools(encoders)
    for line in bad:
        print(line)
    total = sum(len(v) for v in fl.POOLS.values())
    print(f"{total} pool words × {len(names)} tokenizers: {'PASS' if not bad else f'{len(bad)} failures'}")
    return 0 if not bad else 1


def cmd_gen_formupdate(a: argparse.Namespace) -> int:
    from kotodama.eval.binding import formupdate as fu

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for version in fu.VERSIONS:
        rows = fu.generate(version=version)
        bad = fu.audit_failures(fu.audit(rows)) + fu.pairing_failures(rows)
        b = bt.BATTERIES[version]
        dest = out / b.file
        dest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        got = bt.sha256(dest)
        same = got == b.sha256
        ok = ok and same and not bad
        print(f"{b.name} {len(rows)} rows → {dest}; sha256 {got} {'= pin' if same else f'≠ pin {b.sha256[:16]}…'}"
              f"{'' if not bad else '; AUDIT: ' + '; '.join(bad)}")
    return 0 if ok else 1


def _ci(s: Any, stat: str) -> str:
    ci = s.ci[stat]["ci"]
    x = s.point[stat]
    return f"{_fmt(x, '{:+.3f}')} [{_fmt(ci[0] if ci else None, '{:+.3f}')}, {_fmt(ci[1] if ci else None, '{:+.3f}')}]"


def cmd_formupdate(a: argparse.Namespace) -> int:
    from kotodama.eval.binding import formupdate as fu

    root = Path(a.scores) / a.battery
    names = [m for m in a.models.split(",") if m] or sorted(p.stem for p in root.glob("*.jsonl"))
    sums: dict[str, fu.Summary] = {}
    for m in names:
        f = root / f"{m}.jsonl"
        rows = read_jsonl(f) if f.exists() else []
        if rows:
            sums[m] = fu.summarize(rows, n_boot=a.boot)
    if not sums:
        print(f"no {a.battery} rows under {root}")
        return 1
    print(f"{'model':28} {'form [95% CI]':>24} {'interference':>24} {'update':>24} {'norm interf.':>24} "
          f"{'norm update':>24} {'stale':>6} {'stale/err':>9} {'upd C≠last':>24}")
    for m, s in sums.items():
        st = s.pooled
        print(f"{m:28} {_ci(s, 'form'):>24} {_ci(s, 'interference'):>24} {_ci(s, 'update'):>24} "
              f"{_ci(s, 'norm_interference'):>24} {_ci(s, 'norm_update'):>24} {_fmt(st['stale_rate']):>6} "
              f"{_fmt(st['stale_of_errors']):>9} {_ci(s, 'update_nonrecent'):>24}"
              f"{'' if not s.incomplete else f'  ({s.incomplete} incomplete base items left out)'}")
    matched = {m: s.recency_split["position_matched"] for m, s in sums.items()
               if "position_matched" in s.recency_split}
    if matched:
        print("\nposition-matched base items (gold a middle mention in A, B and C):")
        for m, x in matched.items():
            print(f"{m:28} n {x['n']:4}  acc A/B/C " + "/".join(f"{x['acc'][c]:.3f}" for c in fu.CONDITIONS)
                  + f"  interference {x['interference']:+.3f}  update {x['update']:+.3f}")
    print("\nacc A / B / C by K, U, register:")
    for m, s in sums.items():
        parts = [f"K{k} " + "/".join(f"{x['acc'][c]:.2f}" for c in fu.CONDITIONS) for k, x in s.by_k.items() if x["n"]]
        parts += [f"U{u} " + "/".join(f"{x['acc'][c]:.2f}" for c in fu.CONDITIONS) for u, x in s.by_u.items()
                  if x["n"]]
        parts += [f"{r} " + "/".join(f"{x['acc'][c]:.2f}" for c in fu.CONDITIONS) for r, x in s.by_register.items()
                  if x["n"]]
        print(f"{m:28} " + "  ".join(parts))
    preds = None
    if a.reference or a.release or a.anneal:
        if not (a.reference and a.release and a.anneal):
            print("predictions need all of --reference, --release and --anneal")
            return 1
        preds = fu.predictions(sums, [x for x in a.reference.split(",") if x], a.release, a.anneal)
        print(f"\npredictions (release {a.release}, anneal {a.anneal}; reference {a.reference}):")
        for p in ("P1", "P2", "P3"):
            d = preds[p]
            print(f"  {p} {d['stat']:32} release {_fmt(d['release'], '{:+.3f}')} {d['op']} median "
                  f"{_fmt(d['median'], '{:+.3f}')} → {d['holds']}"
                  f"{'' if not d['undefined'] else '  (undefined: ' + ','.join(d['undefined']) + ')'}")
        d = preds["P4"]
        print(f"  P4 {d['stat']:32} anneal {_fmt(d['anneal'], '{:+.3f}')} < release {_fmt(d['release'], '{:+.3f}')}"
              f" → {d['holds']}")
    if a.json:
        Path(a.json).write_text(json.dumps({"summaries": {m: s.as_dict() for m, s in sums.items()},
                                            "predictions": preds}, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    a = build_parser().parse_args(argv)
    if a.cmd == "list":
        for b in bt.BATTERIES.values():
            print(f"{b.name:20} {b.kind:10} {b.file:42} {b.what}")
        print("\ngroups: " + "; ".join(f"{k} = {','.join(v)}" for k, v in bt.GROUPS.items() if k != "all"))
        return 0
    return {"run": cmd_run, "report": cmd_report, "capacity": cmd_capacity,
            "gen-capacity": cmd_gen_capacity, "gen-fluid": cmd_gen_fluid, "fluid": cmd_fluid,
            "check-fluid-tokens": cmd_check_fluid_tokens, "gen-formupdate": cmd_gen_formupdate,
            "formupdate": cmd_formupdate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
