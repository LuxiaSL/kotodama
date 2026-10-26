"""Binding capacity: how many entity→value bindings fit in context before recall falls to chance.

After Ravulapalli & Chadha, "In-Context Binding Capacity in Language Models" (arXiv 2609.30634). A trial is K lines
``Name: task``, an optional block of unrelated filler (about D tokens), then ``The task for Name is:``. Candidates are
the whole task pool (closed set, first token; chance = 1/25). The paper publishes neither templates nor pools, so this
is a faithful re-build, not a copy. Compare models only on this harness, and fit the size law on our own rows.

Choices the paper leaves open, fixed here:
  - tasks: 25 common words, each ONE token after a leading space in SmolLM2 (ours), Pythia, OLMo-1, OLMo-2,
    Qwen2.5, Qwen3 and Llama-3. Distinct while K ≤ 25; past that the pool repeats in shuffled
    rounds (the queried entity still has exactly one task).
  - entities: first names, disjoint from the tasks (multi-token is fine; only the answer is read).
  - query position: uniform over the K lines.
  - filler: templated neutral sentences (no corpus text, no names, no task words). D is measured in OUR tokenizer
    and recorded per row; every model gets the same strings.
  - grid: K ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}, D ∈ {0, 256}, 6 seeds × 40 trials per cell.

Read-out (:func:`summarize`), per model, D = 0 unless noted:
  C     accuracy at K = 1 (the ceiling); b = chance
  r(K)  (acc(K) − b) / (C − b), the paper's fixed-ceiling normalisation
  K50   midpoint of r̂(K) = 1 / (1 + exp(β (ln K − ln K50))), least squares over the K grid; CENSORED when
        r(64) > .5 or the fitted midpoint lies beyond K = 64 (the curve never crosses inside the grid, so the
        midpoint is an extrapolation, not a measurement)
  AUC   mean of r(K) clipped to [0, 1] over the log-spaced K grid (threshold-free)
  D=256 K50 against the D = 0 ceiling (fixed) and against its own K = 1 ceiling (own; n/a when that ceiling is
        within .2 of chance)
Models whose C − b < .2 are reported but kept out of the size law (the paper's Pythia-70M rule).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

VERSION = "capacity-v1"
TASKS: tuple[str, ...] = (
    "laundry", "dishes", "garden", "cooking", "shopping", "painting", "windows", "trash", "mail", "dog",
    "car", "plants", "bills", "lunch", "dinner", "breakfast", "garage", "kitchen", "roof", "fence",
    "pool", "lawn", "taxes", "report", "budget",
)
NAMES: tuple[str, ...] = (
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Isabel", "Jack", "Karen", "Leo", "Maria",
    "Nathan", "Olivia", "Peter", "Quinn", "Rachel", "Samuel", "Tina", "Victor", "Wendy", "Xavier", "Yara", "Zach",
    "Amelia", "Brian", "Chloe", "Daniel", "Elena", "Felix", "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Marcus",
    "Nora", "Oscar", "Paula", "Ryan", "Sophie", "Thomas", "Ursula", "Vera", "William", "Adam", "Bella", "Caleb",
    "Diana", "Ethan", "Fiona", "George", "Helen", "Isaac", "Jasmine", "Kyle", "Lily", "Mason", "Naomi", "Owen",
    "Priya", "Robert", "Sarah", "Tyler", "Uma", "Vincent", "Willow", "Yusuf", "Zoe",
)
K_GRID: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
D_GRID: tuple[int, ...] = (0, 256)
SEEDS = 6
TRIALS = 40
MIN_HEADROOM = 0.2   # C − chance below this ⇒ excluded from the law / own-ceiling read

_ADJ = ("quiet", "narrow", "old", "bright", "wide", "small", "tall", "calm", "grey", "warm", "cold", "long")
_NOUN = ("river", "hill", "road", "bridge", "field", "tower", "valley", "forest", "harbor", "station", "market", "lake")
_VERB = ("runs past", "sits beside", "looks over", "stretches toward", "lies below", "rises above", "winds around")
_PLACE = ("the town", "the coast", "the old mill", "the northern pass", "the square", "the edge of the map",
          "the eastern ridge", "the train line")


def _filler(rng: random.Random, n_tokens: int, count: Callable[[str], int]) -> tuple[str, int]:
    out: list[str] = []
    while True:
        out.append(f"The {rng.choice(_ADJ)} {rng.choice(_NOUN)} {rng.choice(_VERB)} {rng.choice(_PLACE)}.")
        n = count(" ".join(out))
        if n >= n_tokens:
            return " ".join(out), n


def _tasks_for(rng: random.Random, k: int) -> list[str]:
    out: list[str] = []
    while len(out) < k:
        r = list(TASKS)
        rng.shuffle(r)
        out.extend(r)
    return out[:k]


def generate(count_tokens: Callable[[str], int], k_grid: Sequence[int] = K_GRID, d_grid: Sequence[int] = D_GRID,
             seeds: int = SEEDS, trials: int = TRIALS) -> list[dict[str, Any]]:
    """The battery's prompt rows. ``count_tokens`` measures filler length (our tokenizer, no special tokens)."""
    if len(set(TASKS)) != 25 or set(TASKS) & {n.lower() for n in NAMES}:
        raise AssertionError("task pool must be 25 distinct words disjoint from the names")
    if max(k_grid) > len(NAMES):
        raise ValueError(f"K up to {max(k_grid)} needs that many names (have {len(NAMES)})")
    cands = [" " + w for w in TASKS]
    rows: list[dict[str, Any]] = []
    for d in d_grid:
        for k in k_grid:
            for seed in range(seeds):
                rng = random.Random(f"{VERSION}:{k}:{d}:{seed}")
                for t in range(trials):
                    names = rng.sample(NAMES, k)
                    tasks = _tasks_for(rng, k)
                    j = rng.randrange(k)
                    body = "".join(f"{n}: {o}\n" for n, o in zip(names, tasks))
                    dn = 0
                    if d:
                        block, dn = _filler(rng, d, count_tokens)
                        body += f"\n{block}\n\n"
                    rows.append({
                        "id": f"cap1|K{k}|D{d}|s{seed}|t{t}",
                        "prompt": body + f"The task for {names[j]} is:", "candidates": cands, "gold": tasks[j],
                        "K": k, "D": d, "d_tokens": dn, "seed": seed, "qpos": j, "q_is_last": j == k - 1,
                        "q_is_first": j == 0, "gold_repeats": tasks.count(tasks[j]) - 1,
                    })
    return rows


@dataclass(frozen=True)
class Fit:
    k50: float
    beta: float
    sse: float
    censored: bool

    def label(self) -> str:
        return f">{K_GRID[-1]}" if self.censored else f"{self.k50:.1f}"


def fit_logistic(ks: Sequence[int], r: Sequence[float]) -> Fit:
    """Least-squares logistic in ln K (grid search: ln K50 over [ln .5, ln 1024], β over [.1, 8])."""
    lk = [math.log(k) for k in ks]
    best = (math.inf, 0.0, 0.0)
    lo, hi = math.log(0.5), math.log(1024)
    for i in range(241):
        m = lo + i * (hi - lo) / 240
        for j in range(1, 81):
            b = j * 0.1
            sse = sum((1 / (1 + math.exp(b * (x - m))) - y) ** 2 for x, y in zip(lk, r))
            if sse < best[0]:
                best = (sse, m, b)
    sse, m, b = best
    k50 = math.exp(m)
    return Fit(k50, b, sse, r[-1] > 0.5 or k50 > ks[-1])


@dataclass
class Summary:
    model: str
    n_rows: int
    chance: float
    ceiling: float
    fit: Fit
    auc: float
    acc8: Optional[float]
    ceiling_d256: Optional[float]
    fit_d256_fixed: Optional[Fit]
    fit_d256_own: Optional[Fit]
    acc8_d256: Optional[float]
    multi_token_rows: int
    curve: dict[int, float] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.ceiling - self.chance >= MIN_HEADROOM


def _acc(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], float]:
    by: dict[tuple[int, int], list[bool]] = defaultdict(list)
    for r in rows:
        lps = r["cand_lps"]
        by[(r["K"], r["D"])].append(max(lps, key=lps.get) == r["gold"])
    return {k: sum(v) / len(v) for k, v in by.items()}


def _norm(acc: Mapping[tuple[int, int], float], d: int, ceiling: float, chance: float) -> tuple[list[int], list[float]]:
    ks = sorted(k for (k, dd) in acc if dd == d)
    return ks, [(acc[(k, d)] - chance) / max(ceiling - chance, 1e-9) for k in ks]


def summarize(rows: Sequence[Mapping[str, Any]]) -> Summary:
    if not rows:
        raise ValueError("no rows")
    chance = 1 / len(rows[0]["candidates"])
    acc = _acc(rows)
    c0 = acc[(1, 0)]
    ks, r0 = _norm(acc, 0, c0, chance)
    s = Summary(model=str(rows[0].get("model", "?")), n_rows=len(rows), chance=chance, ceiling=c0,
                fit=fit_logistic(ks, r0), auc=sum(min(max(x, 0.0), 1.0) for x in r0) / len(r0),
                acc8=acc.get((8, 0)), ceiling_d256=acc.get((1, 256)), fit_d256_fixed=None, fit_d256_own=None,
                acc8_d256=acc.get((8, 256)), multi_token_rows=sum(bool(r.get("multi_token")) for r in rows),
                curve=dict(zip(ks, r0)))
    if s.ceiling_d256 is not None:
        k2, rf = _norm(acc, 256, c0, chance)
        s.fit_d256_fixed = fit_logistic(k2, rf)
        if s.ceiling_d256 - chance >= MIN_HEADROOM:
            _, ro = _norm(acc, 256, s.ceiling_d256, chance)
            s.fit_d256_own = fit_logistic(k2, ro)
    return s


@dataclass(frozen=True)
class Law:
    c: float
    alpha: float
    r2: float
    n: int

    def n_eff(self, k50: float) -> float:
        """Parameter count (billions) whose law-predicted K50 equals ``k50``."""
        return math.exp((math.log(k50) - self.c) / self.alpha)


def fit_law(points: Sequence[tuple[float, float]]) -> Law:
    """ln K50 = c + α ln N over (N in billions, K50) points (ordinary least squares)."""
    if len(points) < 3:
        raise ValueError(f"need ≥ 3 points for the size law, got {len(points)}")
    xs = [math.log(n) for n, _ in points]
    ys = [math.log(k) for _, k in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    alpha = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    c = my - alpha * mx
    ss_res = sum((y - (c + alpha * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return Law(c, alpha, 1 - ss_res / ss_tot if ss_tot else float("nan"), len(points))
