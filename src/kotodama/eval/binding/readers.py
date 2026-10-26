"""Per-battery readers (each battery computes a different statistic) and the one scorecard built from them.

Readers take scored rows (``score.prompt_row`` shape) for ONE model and return plain dicts:
  joinlite      static stages: lookup (3rd-person) accuracy, I / you accuracy (P2 and P3), and the P2 speaker-swap
                shift with a paired-bootstrap CI (the stage gauge: > +.02 at the CI's low end binds, < −.02 at its high
                end inverts)
  swapdyn       dynamic stage: the speaker-swap shift on the transfer prompts, per condition (0shot / 3shot) and
                query kind (self / addressee)
  decomp        stage-5 decomposition: accuracy per variant × condition (layer 1 = 0-shot ``third``)
  capacity      K50 / AUC / D=256 (:func:`kotodama.eval.binding.capacity.summarize`)
  fluid-*, cryst  per-family ladder midpoint / AUC, F3 stale-value rate, C0 accuracy by shots
                (:func:`kotodama.eval.binding.fluid.summarize`)
  formupdate    form acc(A), interference and update costs (raw and normalized) with paired-bootstrap intervals,
                C's stale rate (:func:`kotodama.eval.binding.formupdate.summarize`)
The arithmetic is that of the readers the recorded gauge numbers came from, so a suite number equals the recorded
one for the same rows.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

Row = Mapping[str, Any]
BIND = 0.02


def _pnorm(r: Row, obj: str) -> float:
    lps = r["cand_lps"]
    mx = max(lps.values())
    return math.exp(lps[obj] - mx) / sum(math.exp(v - mx) for v in lps.values())


def _ok(r: Row, gold_key: str = "gold") -> bool:
    return max(r["cand_lps"], key=r["cand_lps"].get) == r[gold_key]


def boot_ci(xs: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rng.choice(xs) for _ in range(k)) / k for _ in range(n))
    return means[int(.025 * n)], means[int(.975 * n)]


def reading(lo: float, hi: float) -> str:
    return "binds" if lo > BIND else ("INVERTS" if hi < -BIND else "ignores")


def _shift(d: Sequence[float]) -> dict[str, Any]:
    mu = sum(d) / len(d)
    lo, hi = boot_ci(d)
    return {"n": len(d), "shift": mu, "ci": [lo, hi], "reading": reading(lo, hi)}


def joinlite(rows: Sequence[Row]) -> dict[str, Any]:
    by = {(r["scene"], r["qkind"], r["arm"]): r for r in rows}
    out: dict[str, Any] = {}
    for P in (2, 3):
        sc = sorted({r["scene"] for r in rows if r["P"] == P})
        if not sc:
            continue

        def acc(q: str, arm: str = "orig") -> float:
            return sum(_ok(by[(s, q, arm)]) for s in sc if (s, q, arm) in by) / len(sc)

        third_ok = [s for s in sc if _ok(by[(s, "third", "orig")])]
        cell: dict[str, Any] = {"scenes": len(sc), "lookup": acc("third"), "I": acc("self"), "you": acc("addressee"),
                                "you_given_lookup": sum(_ok(by[(s, "addressee", "orig")]) for s in third_ok)
                                / max(1, len(third_ok)), "chance": 1 / (4 if P == 3 else 3)}
        if P == 2:
            for q in ("self", "addressee"):
                d = [_pnorm(by[(s, q, "orig")], by[(s, q, "orig")]["gold"])
                     - _pnorm(by[(s, q, "swap")], by[(s, q, "orig")]["gold"]) for s in sc]
                cell[f"swap_{q}"] = _shift(d)
            inv = sum(max(by[(s, "addressee", "orig")]["cand_lps"], key=by[(s, "addressee", "orig")]["cand_lps"].get)
                      == by[(s, "self", "orig")]["gold"] for s in sc)
            cell["you_to_speaker_obj"] = inv / len(sc)
        out[f"P{P}"] = cell
    return out


def swapdyn(rows: Sequence[Row]) -> dict[str, Any]:
    pair: dict[tuple[str, str, str], dict[str, Row]] = defaultdict(dict)
    for r in rows:
        if r.get("multi_token"):
            continue
        pair[(r["condition"], r["qkind"], r["base_id"])][r["arm"]] = r
    out: dict[str, Any] = {}
    for cond in ("0shot", "3shot"):
        for q in ("addressee", "self"):
            ps = [v for (c, qq, _), v in pair.items() if c == cond and qq == q and "orig" in v and "swap" in v]
            if not ps:
                continue
            po = [_pnorm(v["orig"], v["orig"]["gold_orig"]) for v in ps]
            pw = [_pnorm(v["swap"], v["orig"]["gold_orig"]) for v in ps]
            cell = _shift([x - y for x, y in zip(po, pw)])
            cell.update(p_orig=sum(po) / len(po), p_swap=sum(pw) / len(pw),
                        argmax_orig=sum(_ok(v["orig"], "gold_orig") for v in ps) / len(ps),
                        argmax_swap=sum(_ok(v["swap"], "gold_orig") for v in ps) / len(ps))
            out[f"{cond}/{q}"] = cell
    return out


def decomp(rows: Sequence[Row]) -> dict[str, Any]:
    by: dict[tuple[str, str], list[bool]] = defaultdict(list)
    chance: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        key = (r["variant"], r["condition"])
        by[key].append(_ok(r))
        chance[key].append(1 / len(r["candidates"]))
    return {f"{v}/{c}": {"n": len(x), "acc": sum(x) / len(x), "chance": sum(chance[(v, c)]) / len(x)}
            for (v, c), x in sorted(by.items())}


def capacity(rows: Sequence[Row]) -> dict[str, Any]:
    from kotodama.eval.binding.capacity import summarize

    s = summarize(rows)
    return {"ceiling": s.ceiling, "K50": s.fit.k50, "K50_censored": s.fit.censored, "K50_label": s.fit.label(),
            "beta": s.fit.beta, "auc": s.auc, "acc@8": s.acc8, "ceiling_d256": s.ceiling_d256,
            "K50_d256_fixed": s.fit_d256_fixed.label() if s.fit_d256_fixed else None,
            "K50_d256_own": s.fit_d256_own.label() if s.fit_d256_own else None,
            "acc@8_d256": s.acc8_d256, "usable": s.usable, "multi_token_rows": s.multi_token_rows,
            "curve": s.curve}


def fluid(rows: Sequence[Row]) -> dict[str, Any]:
    from kotodama.eval.binding.fluid import summarize

    return summarize(rows).as_dict()


def formupdate(rows: Sequence[Row]) -> dict[str, Any]:
    from kotodama.eval.binding.formupdate import summarize

    return summarize(rows).as_dict()


READERS = {"joinlite": joinlite, "swapdyn": swapdyn, "decomp": decomp, "capacity": capacity,
           "fluid-morph": fluid, "fluid-algebra": fluid, "fluid-rebind": fluid, "fluid-seqind": fluid,
           "cryst": fluid, "formupdate": formupdate}


def _f(x: Any, fmt: str = "{:+.3f}") -> str:
    return "—" if x is None else fmt.format(x)


SCORECARD_COLUMNS = ("static I", "static you", "dyn-you 0shot", "dyn-you 3shot", "decomp third",
                     "decomp deictic", "capacity K50", "K50 D256")


def scorecard_row(results: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """One scorecard row from per-battery reader results (keyed by reader name)."""
    jl, sd, dc, cp = (results.get(k) or {} for k in ("joinlite", "swapdyn", "decomp", "capacity"))
    p2 = jl.get("P2", {})

    def shift(cell: Mapping[str, Any]) -> str:
        if not cell:
            return "—"
        return f"{cell['shift']:+.3f} {cell['reading'][:4]}"

    return {
        "static I": shift(p2.get("swap_self", {})),
        "static you": shift(p2.get("swap_addressee", {})),
        "dyn-you 0shot": shift(sd.get("0shot/addressee", {})),
        "dyn-you 3shot": shift(sd.get("3shot/addressee", {})),
        "decomp third": _f(dc.get("third/0shot", {}).get("acc"), "{:.3f}"),
        "decomp deictic": _f(dc.get("deictic/0shot", {}).get("acc"), "{:.3f}"),
        "capacity K50": cp.get("K50_label", "—"),
        "K50 D256": cp.get("K50_d256_fixed") or "—",
    }
