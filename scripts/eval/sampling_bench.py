#!/usr/bin/env python3
"""Sampling-parameter bench: temperature x top_p x repetition penalty, measured on a live server.

For every setting in the grid it measures two things against one or more running
``kotodama.serve.server`` replicas (native ``/generate`` + ``/score_tokens``):

1. Open-text degeneration. N built-in openers (short human-style beginnings across
   genres, written for this bench) x L new tokens. Per setting:
     d1 / d2 / d3   distinct-n: unique word n-grams / total word n-grams, averaged per
                    continuation (word = \\w+ run or one punctuation mark, lowercased).
                    Short continuations score high trivially — read next to len.
     pd2            pooled distinct-2 over all of the setting's continuations.
     loop           fraction of continuations where some word 4-gram occurs >= 3 times.
     lrn            mean longest repeated word n-gram (largest n with an n-gram seen twice).
     len / eos      mean continuation length in tokens / fraction that ended before L (EOS).
     nll            mean per-token NLL of the continuation under the plain model
                    (temperature 1, no penalty, no truncation; ``/score_tokens``).
                    Two references sit beside it: greedy (temperature 0, penalty 1.0)
                    continuations — the "maximally likely, often looping" floor — and
                    ancestral samples (T=1, top_p off, penalty 1.0), whose expected NLL is
                    the model's own entropy. Far below ancestral = drifting toward
                    degenerate text; far above = sampling off the model's manifold.
2. Generative binding. Items from prompt/candidates/gold batteries (joinlite, swapdyn)
   scored by GENERATION: K samples of up to 8 tokens per item; the read is the candidate
   whose word appears FIRST in the continuation (or none). Per setting:
     acc            fraction of reads equal to the gold (items that have a gold).
     none           fraction of reads that name no candidate.
     shift          swap pairs (same item, query speaker swapped; ids differ only in the
                    arm suffix): mean of 1[orig read == orig gold] - 1[swap read == orig
                    gold]. > 0: the named object moves with the speaker (binds); ~0: the
                    speaker is ignored; < 0: inverted.
   The penalty applies to GENERATED tokens only, so it cannot touch the first answer
   token; binding varies with temperature / top_p, and with the penalty only through
   reads that answer after a preamble.

The server has no per-request seed, so every number is a sample mean; 95% bootstrap CIs
(over openers / items / pairs) are reported for the headline metrics.

Work is split into (setting, open|bind) units spread over the endpoints, one request in
flight per endpoint (each replica is single-stream). Results append to
``OUT/settings/<setting>.jsonl`` and a rerun resumes. ``--time_budget_min`` stops taking
new units after the budget; the summary covers what finished. Central settings run first
(the grid's extreme values last), so a cut-short run still covers the core.

Usage:
  python scripts/eval/sampling_bench.py --endpoints http://HOST:PORT[,http://HOST:PORT...] \\
      --out DIR --binding_files joinlite-v1.jsonl,swapdyn-v1.jsonl \\
      [--temps 0.7,0.8,0.9,1.0,1.1] [--top_ps 0,0.9,0.95] [--reps 1.0,1.05,1.1,1.2,1.35] \\
      [--n_openers 40] [--max_new_tokens 300] [--binding_groups 150] [--binding_samples 2] \\
      [--time_budget_min 180]
  python scripts/eval/sampling_bench.py --out DIR --summarize_only
Outputs: DIR/summary.md, DIR/summary.csv (one row per setting), DIR/settings/*.jsonl (raw).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import queue
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

log = logging.getLogger("sampling_bench")

# ── Openers (written for this bench; not drawn from any corpus) ───────────────

OPENERS: list[tuple[str, str]] = [
    ("story", "The lighthouse keeper had not spoken to anyone in eleven days when the boat appeared on the horizon."),
    ("story", "My grandmother kept a jar of buttons on the windowsill, and every button had a story."),
    ("story", "Nobody in the village could remember who had planted the orchard, but everyone agreed it was haunted."),
    ("story", "The train stopped in the middle of a field at three in the morning, and the conductor walked off into the dark."),
    ("story", "When the power went out, Mara lit the last candle and opened the letter she had been avoiding all week."),
    ("story", "The robot was designed to sort mail. It had never been asked what it wanted."),
    ("dialogue", "A chat between Sam and Priya.\nSam: Did you ever finish that book I lent you?\nPriya:"),
    ("dialogue", "Interviewer: So tell me, what made you leave the city?\nFarmer:"),
    ("dialogue", "\"You're late,\" said the old man, without looking up from his chessboard.\n\"I know,\" I said."),
    ("dialogue", "A conversation between a teacher and a student after class.\nStudent: I don't get why the sky is blue.\nTeacher:"),
    ("dialogue", "Customer: Hi, I bought this kettle last week and it already stopped working.\nClerk:"),
    ("essay", "There is a particular kind of silence that only exists in libraries, and I have spent most of my life chasing it."),
    ("essay", "Cities are usually described by their buildings, but I think they are better understood through their sidewalks."),
    ("essay", "The case for learning a second language is rarely about usefulness. It is about seeing your own language from the outside."),
    ("essay", "Every generation believes it invented boredom. The evidence suggests otherwise."),
    ("essay", "Most advice about productivity assumes that time is the scarce resource. For most people, attention is."),
    ("howto", "How to repot a houseplant without killing it.\n\nStep 1:"),
    ("howto", "Making bread at home is easier than it looks. You only need four ingredients:"),
    ("howto", "If your bicycle chain keeps slipping, the problem is usually one of three things."),
    ("howto", "A beginner's guide to reading a topographic map.\n\nThe first thing to notice is"),
    ("howto", "Here is the simplest way to clean a cast iron pan after cooking:"),
    ("news", "Local officials announced on Tuesday that the river bridge, closed since the spring floods, will reopen next month."),
    ("news", "A small museum in the north of the country has recovered a painting that went missing more than forty years ago."),
    ("news", "Researchers studying a colony of crows say the birds appear to hold grudges against specific people."),
    ("news", "The town council voted late Thursday to turn the abandoned rail yard into a public park."),
    ("poetry", "I wrote your name on the frost of the window,\nand watched the morning take it back."),
    ("poetry", "The river does not hurry,\nand yet it is never late."),
    ("poetry", "Late autumn, and the maples\nhave begun their slow confession."),
    ("poetry", "Ode to the Kitchen Table\n\nYou have held"),
    ("code", "# Return the n-th Fibonacci number using an iterative loop.\ndef fibonacci(n):\n"),
    ("code", "# Parse a date string like '2024-03-15' and return (year, month, day) as ints.\ndef parse_date(s):\n"),
    ("code", "// Debounce: return a function that delays calling fn until wait ms have passed\n// since the last call.\nfunction debounce(fn, wait) {\n"),
    ("code", "def is_palindrome(text: str) -> bool:\n    \"\"\"Return True if text reads the same forwards and backwards, ignoring case and spaces.\"\"\"\n"),
    ("forum", "Has anyone else had their sourdough starter suddenly stop rising? Mine was fine for months and now nothing."),
    ("forum", "Long time lurker, first post. I'm thinking about switching careers at 35 and could use some honest opinions."),
    ("forum", "Re: best budget hiking boots\n\nI've been wearing the same pair for three years and"),
    ("forum", "Quick question for the gardeners here: is it too late to plant garlic in November?"),
    ("forum", "Update: I finally fixed the noise in my car. You will not believe what it was."),
    ("letter", "Dear Aunt Ruth,\n\nThank you for the scarf. It arrived on the coldest day of the year, which felt like"),
    ("recipe", "Grandma's Lemon Cake\n\nIngredients:\n"),
]

# ── Metrics (pure functions; CPU-tested) ──────────────────────────────────────

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def words(text: str) -> list[str]:
    """Lowercased word/punctuation tokens."""
    return [w.lower() for w in _WORD.findall(text)]


def ngrams(toks: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(toks: Sequence[str], n: int) -> Optional[float]:
    """Unique n-grams / total n-grams (None when there are no n-grams)."""
    g = ngrams(toks, n)
    return len(set(g)) / len(g) if g else None


def has_loop(toks: Sequence[str], n: int = 4, min_count: int = 3) -> bool:
    """True if some n-gram occurs at least ``min_count`` times."""
    c = Counter(ngrams(toks, n))
    return bool(c) and max(c.values()) >= min_count


def longest_repeated_ngram(toks: Sequence[str]) -> int:
    """Largest n such that some n-gram occurs at least twice (0 if no token repeats)."""
    best = 0
    n = 1
    while n <= len(toks) // 2 + 1:
        seen: set[tuple[str, ...]] = set()
        found = False
        for g in ngrams(toks, n):
            if g in seen:
                found = True
                break
            seen.add(g)
        if not found:
            break
        best = n
        n += 1
    return best


def first_candidate(text: str, candidates: Sequence[str]) -> Optional[str]:
    """The (stripped) candidate whose word appears first in ``text``; None if none does.

    Whole-word, case-insensitive. On a tie at the same offset the longer candidate wins.
    """
    best: Optional[tuple[int, int, str]] = None
    for c in candidates:
        w = c.strip()
        if not w:
            continue
        m = re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", text, re.IGNORECASE)
        if m is None:
            continue
        key = (m.start(), -len(w), w)
        if best is None or key < best:
            best = key
    return best[2] if best else None


_ARMS = ("orig3", "orig", "swap")


def pair_key(item_id: str, arm: Optional[str]) -> Optional[str]:
    """Swap-pair key: the id with its arm suffix ('-orig' / '|swap' ...) removed.

    None when the item has no orig/swap arm or the id does not end in it."""
    if arm not in ("orig", "swap"):
        return None
    for sep in ("-", "|", "_"):
        suf = sep + arm
        if item_id.endswith(suf):
            return item_id[: -len(suf)]
    return None


def item_gold(item: dict[str, Any]) -> Optional[str]:
    """The item's own gold (joinlite: 'gold'; swapdyn orig arm: 'gold_orig'; swapdyn swap
    arms have no single gold)."""
    if item.get("gold"):
        return str(item["gold"]).strip()
    if item.get("arm") == "orig" and item.get("gold_orig"):
        return str(item["gold_orig"]).strip()
    return None


def bootstrap_ci(xs: Sequence[float], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return (math.nan, math.nan)
    if len(xs) == 1:
        return (xs[0], xs[0])
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(xs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def mean(xs: Iterable[Optional[float]]) -> float:
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else math.nan


# ── Grid ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Setting:
    temperature: float
    top_p: float          # 0 = off
    repetition_penalty: float

    @property
    def tag(self) -> str:
        p = "off" if self.top_p in (0.0, 1.0) else f"{self.top_p:g}"
        return f"t{self.temperature:g}_p{p}_r{self.repetition_penalty:g}"

    def params(self) -> dict[str, float]:
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": 0,
                "repetition_penalty": self.repetition_penalty}


GREEDY = Setting(0.0, 0.0, 1.0)
ANCESTRAL = Setting(1.0, 0.0, 1.0)


def build_grid(temps: Sequence[float], top_ps: Sequence[float], reps: Sequence[float]) -> list[Setting]:
    """Full grid, central values first: settings whose values sit at an axis's extreme
    (first/last of a >=4-value axis) are scheduled after the core."""
    def edge(v: float, axis: Sequence[float]) -> int:
        s = sorted(axis)
        return int(len(s) >= 4 and v in (s[0], s[-1]))
    grid = [Setting(t, p, r) for t in temps for p in top_ps for r in reps]
    return sorted(grid, key=lambda s: edge(s.temperature, temps) + edge(s.repetition_penalty, reps))


# ── HTTP ──────────────────────────────────────────────────────────────────────

class ServerError(RuntimeError):
    pass


def post(url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        if 400 <= e.code < 500:
            raise ValueError(f"{url}: HTTP {e.code}: {body}") from e   # request bug: no retry
        raise ServerError(f"{url}: HTTP {e.code}: {body}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise ServerError(f"{url}: {e}") from e


def post_retry(url: str, payload: dict[str, Any], tries: int = 4) -> dict[str, Any]:
    for k in range(tries):
        try:
            return post(url, payload)
        except ServerError as e:
            if k == tries - 1:
                raise
            wait = 5 * 2 ** k
            log.warning("%s; retry in %ds", e, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


# ── Records ───────────────────────────────────────────────────────────────────

class Store:
    """Append-only JSONL per setting; a record is keyed by (kind, item, sample)."""

    def __init__(self, out: Path) -> None:
        self.dir = out / "settings"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def path(self, tag: str) -> Path:
        return self.dir / f"{tag}.jsonl"

    def load(self, tag: str) -> list[dict[str, Any]]:
        p = self.path(tag)
        if not p.exists():
            return []
        rows = []
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue   # a torn last line from an interrupted run
        return rows

    def done_keys(self, tag: str) -> set[tuple[str, str, int]]:
        return {(r["kind"], r["item"], r["sample"]) for r in self.load(tag)}

    def append(self, tag: str, rows: list[dict[str, Any]]) -> None:
        with self.lock, self.path(tag).open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


# ── Work units ────────────────────────────────────────────────────────────────

@dataclass
class Unit:
    setting: Setting
    kind: str   # "open" | "bind"


def run_open(ep: str, s: Setting, openers: list[tuple[str, str]], max_new: int,
             store: Store, nll_batch: int) -> None:
    done = store.done_keys(s.tag)
    todo = [(i, g, p) for i, (g, p) in enumerate(openers) if ("open", f"o{i:02d}", 0) not in done]
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        res = post_retry(f"{ep}/score_tokens", {"items": [
            {"id": r["item"], "prompt_ids": r["prompt_ids"], "completion_ids": r["completion_ids"]}
            for r in pending]})["results"]
        for r, sc in zip(pending, res):
            r["nll"] = sc["mean_nll"]
            r.pop("prompt_ids")
        store.append(s.tag, pending)
        pending.clear()

    for i, genre, prompt in todo:
        out = post_retry(f"{ep}/generate", {"prompt": prompt, "max_new_tokens": max_new, **s.params()})
        pending.append({"kind": "open", "item": f"o{i:02d}", "sample": 0, "genre": genre,
                        "text": out["text"], "n_tokens": out["completion_tokens"],
                        "prompt_ids": out["prompt_ids"], "completion_ids": out["completion_ids"],
                        "tps": out.get("tokens_per_second")})
        if len(pending) >= nll_batch:
            flush()
    flush()


def run_bind(ep: str, s: Setting, items: list[dict[str, Any]], samples: int, max_new: int,
             store: Store) -> None:
    done = store.done_keys(s.tag)
    buf: list[dict[str, Any]] = []
    for it in items:
        for k in range(samples):
            if ("bind", it["id"], k) in done:
                continue
            out = post_retry(f"{ep}/generate", {"prompt": it["prompt"], "max_new_tokens": max_new,
                                                **s.params()})
            buf.append({"kind": "bind", "item": it["id"], "sample": k, "file": it["_file"],
                        "text": out["text"], "read": first_candidate(out["text"], it["candidates"])})
            if len(buf) >= 50:
                store.append(s.tag, buf)
                buf = []
    store.append(s.tag, buf)


def worker(ep: str, q: "queue.Queue[Unit]", args: argparse.Namespace, openers, items,
           store: Store, deadline: float, failed: list[str]) -> None:
    fails = 0
    while time.time() < deadline:
        try:
            u = q.get_nowait()
        except queue.Empty:
            return
        t0 = time.time()
        try:
            if u.kind == "open":
                run_open(ep, u.setting, openers, args.max_new_tokens, store, args.nll_batch)
            else:
                run_bind(ep, u.setting, items, args.binding_samples, args.binding_max_tokens, store)
            fails = 0
            log.info("[%s] %s/%s done in %.0fs (%d units left)", ep, u.setting.tag, u.kind,
                     time.time() - t0, q.qsize())
        except ServerError as e:
            fails += 1
            log.error("[%s] %s/%s failed: %s", ep, u.setting.tag, u.kind, e)
            q.put(u)   # another endpoint (or this one, later) picks it up; partial rows are kept
            if fails >= 3:
                log.error("[%s] giving up on this endpoint", ep)
                failed.append(ep)
                return
        except Exception as e:   # a request bug (4xx) or a parsing error: skip the unit, keep going
            log.exception("[%s] %s/%s skipped: %s", ep, u.setting.tag, u.kind, e)
    log.info("[%s] time budget reached", ep)


# ── Binding items ─────────────────────────────────────────────────────────────

def load_binding(files: Sequence[str], groups_per_file: int, seed: int = 0) -> list[dict[str, Any]]:
    """Per file, a seeded sample of ``groups_per_file`` swap groups (both arms kept
    together; unpaired items are their own group). 0 = all items."""
    out: list[dict[str, Any]] = []
    for f in files:
        rows = [json.loads(line) for line in open(f) if line.strip()]
        name = Path(f).stem
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            r["_file"] = name
            groups[pair_key(r["id"], r.get("arm")) or r["id"]].append(r)
        keys = sorted(groups)
        if groups_per_file and groups_per_file < len(keys):
            keys = sorted(random.Random(seed).sample(keys, groups_per_file))
        for k in keys:
            out.extend(groups[k])
    return out


# ── Summary ───────────────────────────────────────────────────────────────────

def summarize_open(rows: list[dict[str, Any]], max_new: int) -> dict[str, Any]:
    rows = [r for r in rows if r["kind"] == "open"]
    if not rows:
        return {}
    ws = [words(r["text"]) for r in rows]
    loops = [float(has_loop(w)) for w in ws]
    nll = [r.get("nll") for r in rows]
    pooled2 = [g for w in ws for g in ngrams(w, 2)]
    lo, hi = bootstrap_ci(loops)
    nlo, nhi = bootstrap_ci([x for x in nll if x is not None])
    return {
        "n_open": len(rows),
        "d1": mean(distinct_n(w, 1) for w in ws), "d2": mean(distinct_n(w, 2) for w in ws),
        "d3": mean(distinct_n(w, 3) for w in ws),
        "pd2": len(set(pooled2)) / len(pooled2) if pooled2 else math.nan,
        "loop": mean(loops), "loop_lo": lo, "loop_hi": hi,
        "lrn": mean(float(longest_repeated_ngram(w)) for w in ws),
        "len": mean(float(r["n_tokens"]) for r in rows),
        "eos": mean(float(r["n_tokens"] < max_new) for r in rows),
        "nll": mean(nll), "nll_lo": nlo, "nll_hi": nhi,
    }


def summarize_bind(rows: list[dict[str, Any]], items_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in rows if r["kind"] == "bind" and r["item"] in items_by_id]
    if not rows:
        return {}
    acc: list[float] = []
    acc_by_file: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        g = item_gold(items_by_id[r["item"]])
        if g is not None:
            hit = float(r["read"] is not None and r["read"].lower() == g.lower())
            acc.append(hit)
            acc_by_file[r["file"]].append(hit)
    none = [float(r["read"] is None) for r in rows]
    # swap pairs: per (pair, sample)
    by_pair: dict[tuple[str, int], dict[str, Optional[str]]] = defaultdict(dict)
    gold_orig: dict[str, str] = {}
    for r in rows:
        it = items_by_id[r["item"]]
        pk = pair_key(it["id"], it.get("arm"))
        if pk is None:
            continue
        by_pair[(pk, r["sample"])][it["arm"]] = r["read"]
        if it["arm"] == "orig":
            g = item_gold(it)
            if g is not None:
                gold_orig[pk] = g.lower()
    shift_by_pair: dict[str, list[float]] = defaultdict(list)
    for (pk, _k), arms in by_pair.items():
        if "orig" in arms and "swap" in arms and pk in gold_orig:
            g = gold_orig[pk]
            o = float((arms["orig"] or "").lower() == g)
            w = float((arms["swap"] or "").lower() == g)
            shift_by_pair[pk].append(o - w)
    shifts = [mean(v) for v in shift_by_pair.values()]
    alo, ahi = bootstrap_ci(acc)
    slo, shi = bootstrap_ci(shifts)
    out: dict[str, Any] = {"n_bind": len(rows), "acc": mean(acc), "acc_lo": alo, "acc_hi": ahi,
                           "none": mean(none), "shift": mean(shifts), "shift_lo": slo,
                           "shift_hi": shi, "n_pairs": len(shifts)}
    for f, v in sorted(acc_by_file.items()):
        out[f"acc_{f}"] = mean(v)
    return out


def _fmt(v: Any, nd: int = 3) -> str:
    if isinstance(v, float):
        return "—" if math.isnan(v) else f"{v:.{nd}f}"
    return "—" if v is None else str(v)


def pareto(rows: list[dict[str, Any]], keys: list[tuple[str, int]]) -> list[dict[str, Any]]:
    """Rows not dominated on keys [(name, +1 maximize | -1 minimize)]."""
    ok = [r for r in rows if all(not math.isnan(r.get(k, math.nan)) for k, _ in keys)]
    front = []
    for a in ok:
        dominated = any(all(b[k] * d >= a[k] * d for k, d in keys) and
                        any(b[k] * d > a[k] * d for k, d in keys) for b in ok if b is not a)
        if not dominated:
            front.append(a)
    return front


def write_summary(out: Path, settings: list[Setting], items: list[dict[str, Any]],
                  max_new: int, note: str = "") -> None:
    store = Store(out)
    items_by_id = {it["id"]: it for it in items}
    table: list[dict[str, Any]] = []
    for s in [GREEDY, ANCESTRAL] + [x for x in settings if x not in (GREEDY, ANCESTRAL)]:
        rows = store.load(s.tag)
        if not rows:
            continue
        rec: dict[str, Any] = {"setting": s.tag, "temperature": s.temperature, "top_p": s.top_p,
                               "rep": s.repetition_penalty}
        rec.update(summarize_open(rows, max_new))
        rec.update(summarize_bind(rows, items_by_id))
        table.append(rec)
    if not table:
        log.warning("nothing to summarize under %s", out)
        return
    head = ["setting", "temperature", "top_p", "rep"]
    cols = head + sorted({k for r in table for k in r} - set(head))
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in table:
            w.writerow(r)

    ref = {r["setting"]: r for r in table}
    g_nll = ref.get(GREEDY.tag, {}).get("nll", math.nan)
    a_nll = ref.get(ANCESTRAL.tag, {}).get("nll", math.nan)
    grid_rows = [r for r in table if r["setting"] not in (GREEDY.tag,)]
    lines = ["# Sampling bench summary", ""]
    if note:
        lines += [note, ""]
    lines += [f"NLL references: greedy (t0, rep 1.0) = {_fmt(g_nll)}; ancestral (t1, top_p off, "
              f"rep 1.0) = {_fmt(a_nll)} nats/token. Open text: {max_new} new tokens per opener.", "",
              "| setting | loop [95% CI] | d2 | pd2 | lrn | len | eos | nll [95% CI] | Δnll vs ancestral "
              "| bind acc [95% CI] | none | swap shift [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        dn = r.get("nll", math.nan) - a_nll if not math.isnan(a_nll) else math.nan
        lines.append(
            f"| {r['setting']} | {_fmt(r.get('loop', math.nan))} [{_fmt(r.get('loop_lo', math.nan), 2)},"
            f"{_fmt(r.get('loop_hi', math.nan), 2)}] | {_fmt(r.get('d2', math.nan))} "
            f"| {_fmt(r.get('pd2', math.nan))} | {_fmt(r.get('lrn', math.nan), 1)} "
            f"| {_fmt(r.get('len', math.nan), 0)} | {_fmt(r.get('eos', math.nan), 2)} "
            f"| {_fmt(r.get('nll', math.nan))} [{_fmt(r.get('nll_lo', math.nan), 2)},"
            f"{_fmt(r.get('nll_hi', math.nan), 2)}] | {_fmt(dn, 2)} "
            f"| {_fmt(r.get('acc', math.nan))} [{_fmt(r.get('acc_lo', math.nan), 2)},"
            f"{_fmt(r.get('acc_hi', math.nan), 2)}] | {_fmt(r.get('none', math.nan), 2)} "
            f"| {_fmt(r.get('shift', math.nan))} [{_fmt(r.get('shift_lo', math.nan), 2)},"
            f"{_fmt(r.get('shift_hi', math.nan), 2)}] |")

    def top(key: str, d: int, n: int = 5) -> str:
        v = [r for r in grid_rows if not math.isnan(r.get(key, math.nan))]
        v.sort(key=lambda r: -d * r[key])
        return ", ".join(f"{r['setting']} ({_fmt(r[key])})" for r in v[:n])

    lines += ["", "## Leaders", "",
              f"- fewest loops: {top('loop', -1)}",
              f"- lowest NLL (most model-likely): {top('nll', -1)}",
              f"- highest d2: {top('d2', +1)}",
              f"- highest binding acc: {top('acc', +1)}",
              f"- largest swap shift: {top('shift', +1)}", ""]
    front = pareto(grid_rows, [("loop", -1), ("nll", -1), ("acc", +1)])
    front.sort(key=lambda r: (r["loop"], r["nll"]))
    lines += ["## Pareto front (loop ↓, nll ↓, binding acc ↑)", ""]
    lines += [f"- {r['setting']}: loop {_fmt(r['loop'])}, nll {_fmt(r['nll'])}, d2 {_fmt(r.get('d2', math.nan))}, "
              f"acc {_fmt(r['acc'])}, shift {_fmt(r.get('shift', math.nan))}" for r in front]
    lines += ["", "Reading: low loop with NLL near the ancestral reference is the target; NLL far "
              "below it with low loop usually means truncation (top_p / low T) is buying likelihood "
              "with diversity (watch d2/pd2); NLL far above it means the sampler is leaving the "
              "model's distribution. The penalty cannot affect the first answer token, so binding "
              "differences across rep values at fixed (T, top_p) are sampling noise unless they "
              "exceed the CI.", ""]
    (out / "summary.md").write_text("\n".join(lines))
    log.info("wrote %s and %s", out / "summary.md", out / "summary.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--endpoints", default="", help="comma-separated server base URLs")
    ap.add_argument("--out", required=True, help="output directory (resumable)")
    ap.add_argument("--temps", default="0.7,0.8,0.9,1.0,1.1")
    ap.add_argument("--top_ps", default="0,0.9,0.95", help="0 = off")
    ap.add_argument("--reps", default="1.0,1.05,1.1,1.2,1.35")
    ap.add_argument("--n_openers", type=int, default=len(OPENERS))
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--nll_batch", type=int, default=16)
    ap.add_argument("--binding_files", default="", help="comma-separated battery jsonl files")
    ap.add_argument("--binding_groups", type=int, default=150,
                    help="swap groups sampled per battery file (0 = all items)")
    ap.add_argument("--binding_samples", type=int, default=2)
    ap.add_argument("--binding_max_tokens", type=int, default=8)
    ap.add_argument("--no_references", action="store_true",
                    help="skip the greedy + ancestral open-text references")
    ap.add_argument("--time_budget_min", type=float, default=0.0, help="0 = unlimited")
    ap.add_argument("--summarize_only", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "bench_config.json"
    if args.summarize_only:
        if not cfg_path.exists():
            log.error("%s missing: run the bench first", cfg_path)
            return 2
        cfg = json.loads(cfg_path.read_text())
        for k in ("temps", "top_ps", "reps", "binding_files", "binding_groups", "max_new_tokens"):
            setattr(args, k, cfg[k])
    else:
        cfg_path.write_text(json.dumps({k: getattr(args, k) for k in (
            "temps", "top_ps", "reps", "binding_files", "binding_groups", "binding_samples",
            "max_new_tokens", "n_openers")}, indent=1))

    settings = build_grid(floats(args.temps), floats(args.top_ps), floats(args.reps))
    bfiles = [f for f in args.binding_files.split(",") if f.strip()]
    try:
        items = load_binding(bfiles, args.binding_groups)
    except FileNotFoundError as e:
        log.error("binding file missing: %s", e)
        return 2
    openers = OPENERS[: args.n_openers]

    if not args.summarize_only:
        eps = [e.rstrip("/") for e in args.endpoints.split(",") if e.strip()]
        if not eps:
            log.error("--endpoints is required unless --summarize_only")
            return 2
        for e in eps:
            try:
                with urllib.request.urlopen(f"{e}/health", timeout=10) as r:
                    h = json.loads(r.read())
                if not h.get("model_loaded"):
                    raise ServerError("model not loaded")
            except (OSError, ServerError, ValueError) as exc:
                log.error("endpoint %s not healthy: %s", e, exc)
                return 2
        log.info("%d settings x (%d openers + %d binding items x %d samples) on %d endpoints",
                 len(settings), len(openers), len(items), args.binding_samples, len(eps))
        q: "queue.Queue[Unit]" = queue.Queue()
        if not args.no_references:
            q.put(Unit(GREEDY, "open"))
            if ANCESTRAL not in settings:
                q.put(Unit(ANCESTRAL, "open"))
        for s in settings:
            q.put(Unit(s, "open"))
            if items:
                q.put(Unit(s, "bind"))
        deadline = time.time() + (args.time_budget_min * 60 if args.time_budget_min > 0 else 1e12)
        store = Store(out)
        failed: list[str] = []
        threads = [threading.Thread(target=worker, args=(e, q, args, openers, items, store,
                                                         deadline, failed), daemon=True)
                   for e in eps]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        left = q.qsize()
        if left:
            log.warning("%d units not run (budget or endpoint failures: %s)", left, failed)
    note = ""
    write_summary(out, settings, items, args.max_new_tokens, note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
