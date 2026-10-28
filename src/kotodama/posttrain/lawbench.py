"""Sampling-law bench: which law (sampler dials and steering dials) gives the best text from a base model, per job.

A *cell* is one law: temperature, repetition penalty, truncation (top_p / min_p), penalty scope, plus an optional
steering stack (bank vectors with a dose, a write rule and a span). Every cell runs the same fixed *panels* through
one OpenAI-compatible endpoint (a ``kotodama.serve.gateway`` fleet of the tsumugi profile: per-row laws and
per-request levers, so all cells share batches):

* ``completion`` panels: one prompt, two seeded draws (``n`` 2), e.g. the seed chain's blurb step or a prose
  opening. Rows ``{"id", "prompt", "stop"?, "max_tokens"?, "check"?, "reference"?}``; ``check`` holds the form
  limits a draw must meet (``min_chars``, ``max_chars``, ``min_sentences``, ``forbid`` regex); ``reference`` is a
  held-out human continuation (the damage set reads it).
* ``dialogue`` panels: self-play seeds (``kotodama.posttrain.selfplay``: ``{"id", "cast", "header", "turns"?}``)
  grown ONE path, no search: every turn draws two candidates, the first is taken as drawn (the second only when the
  first is empty after the stray-label cut) and the second is kept as the turn's sibling. The mechanical vetoes are
  recorded as meters, never acted on, so the law alone shapes the scene.

Draws are seeded per (item, turn) and the seed is shared by every cell, so cells differ by their law, not by their
noise. Stage A (``stage-a``) is four one-dimensional ladders from the base law: temperature, repetition penalty, the
V7 entropy direction and the Vrep-perp repetition direction (each vector ladder with random-direction controls at
the same doses), read against DAMAGE: the mean NLL increase of a fixed continuation set under the cell's law
(teacher-forced; ``scripts/eval/law_damage.py``), since a matched dose is not a matched perturbation.

    python -m kotodama.posttrain.lawbench stage-a --out cells.jsonl [--plan plan.json]
    python -m kotodama.posttrain.lawbench generate --cells cells.jsonl --panel seed=seed.jsonl \\
        --panel dialogue=dlg.jsonl --panel prose=prose.jsonl --endpoint http://HOST:PORT --out <run>/ [--limit 4]
    python -m kotodama.posttrain.lawbench fluency <run>/ --panel seed=seed.jsonl ... --endpoint http://HOST:PORT
    python -m kotodama.posttrain.lawbench damage-set <run>/ --panel prose=prose.jsonl --out <run>/damage_set.jsonl
    python -m kotodama.posttrain.lawbench summary <run>/ --panel seed=seed.jsonl ... [--damage <run>/damage.json] [--bt bt.json]

``generate`` writes ``<run>/cells.jsonl`` and ``<run>/gen/<panel>.jsonl`` (one row per (cell, item); rerunning resumes);
``fluency`` adds ``<run>/fluency/<panel>.json`` (each row's mean NLL under the plain model, ``/score_tokens``);
``summary`` writes ``<run>/summary.json`` and prints the per-panel frontier table (meters, fluency, damage, blind-read
strength when a Bradley-Terry file is given).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from kotodama.posttrain.narration import out_of_frame
from kotodama.posttrain.selfplay import (PromptTooLong, SelfPlayConfig, evaluate, is_template,
                                         next_speaker, oai_completer, parse_seed, speech_key, spoken_text,
                                         split_completion)
from kotodama.posttrain.transcript import Transcript, Turn, render_prompt, stop_sequences
from kotodama.serve.laws import law

logger = logging.getLogger(__name__)

BASE_LAW = "tsumugi"
RULES = ("K", "N", "S")
SPANS = ("decode", "uniform")
SCOPES = ("generated", "row")


# ── cells ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cell:
    """One sampling law. ``steer``: ((bank key or alias, dose), ...); ``injection``: a raw lever instead, as the
    server's ``injection`` field (e.g. ``{"kind": "act", "sites", "neurons", "mu", "alpha"}``); ``control``: a
    random-direction control."""

    id: str
    ladder: str
    temperature: float
    repetition_penalty: float
    top_p: float = 0.0
    min_p: float = 0.0
    penalty_scope: str = "generated"
    steer: tuple[tuple[str, float], ...] = ()
    steer_rule: str = "K"
    steer_span: str = "decode"
    control: bool = False
    injection: Optional[dict[str, Any]] = field(default=None, compare=True, hash=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.+-]{1,64}", self.id):
            raise ValueError(f"cell id {self.id!r}: 1-64 of [A-Za-z0-9_.+-]")
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError(f"cell {self.id}: temperature {self.temperature} outside (0, 2]")
        if not 1.0 <= self.repetition_penalty <= 2.0:
            raise ValueError(f"cell {self.id}: repetition_penalty {self.repetition_penalty} outside [1, 2]")
        if not (0.0 <= self.top_p <= 1.0 and 0.0 <= self.min_p < 1.0):
            raise ValueError(f"cell {self.id}: top_p in [0, 1] and min_p in [0, 1)")
        if self.penalty_scope not in SCOPES or self.steer_rule not in RULES or self.steer_span not in SPANS:
            raise ValueError(f"cell {self.id}: penalty_scope {SCOPES}, steer_rule {RULES}, steer_span {SPANS}")
        for name, dose in self.steer:
            if not name or not math.isfinite(dose):
                raise ValueError(f"cell {self.id}: bad steering member {(name, dose)!r}")
        if self.injection is not None:
            if self.steer or not isinstance(self.injection, dict) or "sites" not in self.injection:
                raise ValueError(f"cell {self.id}: an injection is a dict with sites, and excludes bank vectors")

    @property
    def dose(self) -> float:
        """The first steering member's dose (0 for an unsteered cell): a vector ladder's coordinate."""
        return float(self.steer[0][1]) if self.steer else 0.0

    def sampling(self) -> dict[str, Any]:
        """The law as ``kotodama.posttrain.selfplay.oai_completer`` takes it."""
        return {"temperature": self.temperature, "top_k": 0, "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty, "min_p": self.min_p,
                "penalty_scope": self.penalty_scope}

    def vectors(self) -> list[dict[str, Any]]:
        return [{"name": n, "alpha": a} for n, a in self.steer]

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["steer"] = [list(m) for m in self.steer]
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Cell":
        known = {f for f in cls.__dataclass_fields__}
        extra = sorted(set(row) - known)
        if extra:
            raise ValueError(f"cell row has unknown keys {extra}")
        kw = dict(row)
        kw["steer"] = tuple((str(n), float(a)) for n, a in row.get("steer", []))
        return cls(**kw)


def _fmt(x: float) -> str:
    return f"{x:+.3f}".rstrip("0").rstrip(".").replace("+", "p").replace("-", "m")


# The stage-A plan: four ladders from the base law, and random-direction controls for each vector ladder. Bank keys
# are the tsumugi steering bank's (site 58 = after layer 29's attention; Rband* are random unit directions in the
# same band the eigen-directions were drawn from).
STAGE_A: dict[str, Any] = {
    "temperature": [0.6, 0.7, 0.8, 0.9, 1.0],
    "repetition_penalty": [1.0, 1.1, 1.35],
    "vector_ladders": [
        {"name": "V7", "key": "V7_s58", "rule": "K", "doses": [-0.15, -0.10, -0.05, 0.05, 0.10, 0.15],
         "random_per_dose": 3},
        {"name": "Vrep", "key": "Vrepperp_s58", "rule": "N", "doses": [-0.15, -0.30, -0.50, -0.80]},
        {"name": "Vrep", "key": "Vrepperp_s58", "rule": "S", "doses": [-0.30, -0.60, -1.0]},
        {"name": "Vrep", "key": "Vrepperp_s58", "rule": "K", "doses": [-0.05, -0.10]},
    ],
    "random_keys": ["Rband1_s58", "Rband2_s58", "Rband3_s58"],
    "random_per_dose": 2,
}


def stage_a(plan: Optional[dict[str, Any]] = None, base: str = BASE_LAW) -> list[Cell]:
    """The stage-A cells: the base law, its temperature and repetition-penalty ladders, each vector ladder at the
    base law, and ``random_per_dose`` random-direction controls per vector dose (same rule, dose and sign; the
    random keys rotate so every key serves every ladder; a ladder's own ``random_per_dose`` adds controls on the
    keys its dose has not used)."""
    p = {**STAGE_A, **(plan or {})}
    b = law(base)
    T0, rp0 = float(b["temperature"]), float(b["repetition_penalty"])
    cells = [Cell("base", "base", T0, rp0)]
    cells += [Cell(f"T{t:g}", "T", float(t), rp0) for t in p["temperature"] if not math.isclose(t, T0)]
    cells += [Cell(f"rp{r:g}", "rp", T0, float(r)) for r in p["repetition_penalty"] if not math.isclose(r, rp0)]
    keys = list(p["random_keys"])
    k = 0
    for lad in p["vector_ladders"]:
        rule = lad.get("rule", "K")
        for d in lad["doses"]:
            cells.append(Cell(f"{lad['name']}{rule}{_fmt(d)}", lad["name"], T0, rp0, steer=((lad["key"], float(d)),),
                              steer_rule=rule))
            used: list[str] = []
            for j in range(int(lad.get("random_per_dose", p["random_per_dose"]))):
                if j < int(p["random_per_dose"]):
                    key = keys[k % len(keys)]
                    k += 1
                else:  # a ladder's extra controls take the keys this dose has not used yet (the rotation is kept)
                    key = next((x for x in keys if x not in used), keys[j % len(keys)])
                used.append(key)
                cells.append(Cell(f"R{lad['name']}{rule}{_fmt(d)}_{j}", f"rand:{lad['name']}", T0, rp0,
                                  steer=((key, float(d)),), steer_rule=rule, control=True))
    ids = [c.id for c in cells]
    if len(set(ids)) != len(ids):
        raise ValueError("stage-A plan produced duplicate cell ids")
    return cells


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSONL file; a final line cut mid-write (an interrupted run) is skipped."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("%s: skipped an unreadable line", path)
    return rows


def load_cells(path: Path) -> list[Cell]:
    return [Cell.from_row(r) for r in read_jsonl(path)]


# ── panels ───────────────────────────────────────────────────────────────────


@dataclass
class Panel:
    name: str
    kind: str  # completion | dialogue
    items: list[dict[str, Any]]


def load_panel(name: str, path: Path) -> Panel:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"panel {name}: no rows in {path}")
    kinds = {"dialogue" if "cast" in r else "completion" for r in rows}
    if len(kinds) != 1:
        raise ValueError(f"panel {name}: mixes dialogue seeds and completion prompts")
    ids = [str(r.get("id", i)) for i, r in enumerate(rows)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"panel {name}: duplicate item ids")
    for i, r in zip(ids, rows):
        r["id"] = i
        if "cast" not in r and not isinstance(r.get("prompt"), str):
            raise ValueError(f"panel {name}: item {i} has no prompt")
    return Panel(name, kinds.pop(), rows)


def item_seed(item: str, step: int = 0) -> int:
    """The draw seed of (item, step), shared by every cell (common random numbers across laws)."""
    return zlib.crc32(f"{item}:{step}".encode()) & 0x7FFFFFFF


# ── generation ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenConfig:
    draws: int = 2
    completion_max_tokens: int = 256
    max_turns: int = 40
    turn_max_tokens: int = 256
    window_chars: int = 10000
    max_prompt_chars: int = 14000


def _ended(raw: str, fr: str, hit: bool, stops: list[str]) -> tuple[str, str]:
    return split_completion(raw, stops, fr, hit)


async def gen_completion(item: dict[str, Any], complete: Any, cfg: GenConfig, user: str) -> dict[str, Any]:
    stops = list(item.get("stop") or [])
    raw = await complete(item["prompt"], cfg.draws, stops, seed=item_seed(item["id"]), user=user,
                         max_tokens=int(item.get("max_tokens", cfg.completion_max_tokens)))
    draws = []
    for t, fr, hit in raw:
        text, ended = _ended(t, fr, hit, stops)
        draws.append({"text": text, "ended_by": ended})
    return {"draws": draws}


async def gen_dialogue(item: dict[str, Any], complete: Any, cfg: GenConfig, user: str) -> dict[str, Any]:
    """One path of up to ``max_turns`` turns: draw 0 taken as drawn (draw 1 when draw 0 is empty after the cut),
    the other kept as the sibling; vetoes recorded only."""
    from kotodama.posttrain.loom import window_turns

    seed = parse_seed(item, 0)
    if seed is None:
        raise ValueError(f"dialogue item {item['id']}: needs a cast of >= 2 names")
    sp = SelfPlayConfig(candidates=cfg.draws, max_tokens=cfg.turn_max_tokens)
    turns: list[Turn] = list(seed.turns)
    n_seed = len(turns)
    stops = stop_sequences(seed.cast)
    rec_turns: list[dict[str, Any]] = []
    finish = "max_turns"
    for k in range(cfg.max_turns):
        speaker = next_speaker(seed.cast, turns)
        seen = window_turns(tuple(turns), seed.header, n_seed, cfg.window_chars)
        prompt = render_prompt(Transcript(seen, seed.header), speaker)
        if len(prompt) > cfg.max_prompt_chars:
            finish = "length_cap"
            break
        try:
            raw = await complete(prompt, cfg.draws, stops, seed=item_seed(item["id"], k + 1), user=user,
                                 max_tokens=cfg.turn_max_tokens)
        except PromptTooLong:
            finish = "length_cap"
            break
        cands = [evaluate(*_ended(t, fr, hit, stops), speaker, list(seen), seed.cast, seed.header, sp)
                 for t, fr, hit in raw]
        pick = next((i for i, c in enumerate(cands) if c.text), None)
        if pick is None:
            rec_turns.append({"speaker": speaker, "text": "", "ended_by": cands[0].ended_by if cands else "empty",
                              "vetoes": ["empty"], "siblings": []})
            finish = "empty"
            break
        c = cands[pick]
        rec_turns.append({"speaker": speaker, "text": c.text, "ended_by": c.ended_by, "vetoes": c.vetoes,
                          "echo_frac": round(c.echo_frac, 4), "selfcopy_frac": round(c.selfcopy_frac, 4),
                          "picked_draw": pick,
                          "siblings": [{"text": o.text, "ended_by": o.ended_by} for i, o in enumerate(cands)
                                       if i != pick]})
        turns.append(Turn(speaker, c.text))
        if c.ended_by == "eos":
            finish = "eos"
            break
    return {"cast": seed.cast, "header": seed.header, "seed_turns": [asdict(t) for t in turns[:n_seed]],
            "turns": rec_turns, "finish": finish}


async def generate(cells: list[Cell], panels: list[Panel], endpoint: str, out: Path, cfg: GenConfig,
                   concurrency: int, limit: int = 0, transport: Any = None,
                   stamp: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Every (cell, panel item) not yet in ``out/gen/<panel>.jsonl``; items of a panel are taken in file order
    (``limit`` > 0: the first ``limit``). Jobs interleave cells, so a partial run covers every cell evenly. ``stamp``:
    provenance (engine commit, engine flags) written into every row as ``prov``."""
    (out / "gen").mkdir(parents=True, exist_ok=True)
    have_cells = {c["id"]: c for c in read_jsonl(out / "cells.jsonl")}
    for c in cells:
        if c.id in have_cells and Cell.from_row(have_cells[c.id]) != c:
            raise SystemExit(f"cell {c.id} differs from the one already in {out / 'cells.jsonl'}")
    with open(out / "cells.jsonl", "a", encoding="utf-8") as f:
        for c in cells:
            if c.id not in have_cells:
                f.write(json.dumps(c.to_row()) + "\n")
    completers: dict[str, Any] = {}
    closers = []
    for c in cells:
        comp, close = oai_completer([endpoint], c.sampling(), cfg.turn_max_tokens, transport=transport,
                                    steer=c.vectors() or None, steer_rule=c.steer_rule)
        completers[c.id] = (comp, c)
        closers.append(close)
    jobs: list[tuple[Panel, dict[str, Any], Cell]] = []
    files = {}
    for p in panels:
        done = {(r["cell"], r["item"]) for r in read_jsonl(out / "gen" / f"{p.name}.jsonl")}
        files[p.name] = open(out / "gen" / f"{p.name}.jsonl", "a", encoding="utf-8")
        items = p.items[:limit] if limit else p.items
        for it in items:
            for c in cells:
                if (c.id, it["id"]) not in done:
                    jobs.append((p, it, c))
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    stats = {"done": 0, "failed": 0, "todo": len(jobs)}

    async def one(p: Panel, it: dict[str, Any], c: Cell) -> None:
        comp, _ = completers[c.id]
        user = f"lawbench|{c.id}|{p.name}|{it['id']}"

        async def call(prompt: str, n: int, stops: list[str], **kw: Any) -> Any:
            if c.steer_span != "decode":
                kw["steer_span"] = c.steer_span
            if c.injection is not None:
                kw["injection"] = c.injection
            return await comp(prompt, n, stops, **kw)

        async with sem:
            try:
                body = await (gen_dialogue if p.kind == "dialogue" else gen_completion)(it, call, cfg, user)
            except Exception as e:  # one row's failure must not stop the bench
                stats["failed"] += 1
                logger.error("%s / %s / %s failed: %r", p.name, c.id, it["id"], e)
                return
        files[p.name].write(json.dumps({"panel": p.name, "kind": p.kind, "cell": c.id, "item": it["id"], **body,
                                        **({"prov": stamp} if stamp else {})},
                                       ensure_ascii=False) + "\n")
        files[p.name].flush()
        stats["done"] += 1
        if stats["done"] % 100 == 0:
            logger.info("%d/%d rows (%.0fs, %d failed)", stats["done"], len(jobs), time.time() - t0, stats["failed"])

    try:
        # interleave: item-major, cells inside, so every cell advances together
        await asyncio.gather(*(one(*j) for j in jobs))
    finally:
        for f in files.values():
            f.close()
        for close in closers:
            await close()
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


# ── meters (pure) ────────────────────────────────────────────────────────────

_WORD = re.compile(r"[a-z0-9']+")
_SENT = re.compile(r"[.!?](?:[\"'”’)]*)(?:\s|$)")
_STOPWORDS = frozenset("the a an and or but to of in on at for with is are was were be been it its this that you your "
                       "i me my we our he she his her they them their what do does did not no yes so just have has "
                       "had will would can could dont im youre thats there here well oh okay know like get got go "
                       "going about from into out then than when where who why how all one some any very".split())


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(ws: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def repeat_share(text: str, n: int = 4) -> float:
    """Share of the text's word n-grams that already occurred earlier in it (0 for a text shorter than n)."""
    seen: set[tuple[str, ...]] = set()
    rep = 0
    grams = ngrams(words(text), n)
    for g in grams:
        rep += g in seen
        seen.add(g)
    return rep / len(grams) if grams else 0.0


def text_loop_onset(text: str, n: int = 4, span: int = 40) -> Optional[float]:
    """Where a text falls into a loop, as a fraction of its n-grams: the first n-gram from which at least half of the
    next ``span`` repeat an earlier one; None when it never does."""
    grams = ngrams(words(text), n)
    seen: set[tuple[str, ...]] = set()
    rep: list[bool] = []
    for g in grams:
        rep.append(g in seen)
        seen.add(g)
    for i in range(len(rep)):
        win = rep[i:i + span]
        if len(win) >= span // 2 and 2 * sum(win) >= len(win) and rep[i]:
            return round(i / len(rep), 4)
    return None


def distinctness(a: str, b: str, n: int = 3) -> Optional[float]:
    """1 - Jaccard overlap of two texts' word n-gram sets (None when either has none)."""
    x, y = set(ngrams(words(a), n)), set(ngrams(words(b), n))
    if not x or not y:
        return None
    return 1.0 - len(x & y) / len(x | y)


def form_check(text: str, check: Optional[dict[str, Any]]) -> Optional[bool]:
    """Whether a draw meets an item's form limits (None when the item states none)."""
    if not check:
        return None
    t = " ".join(text.split())
    if len(t) < int(check.get("min_chars", 0)) or len(t) > int(check.get("max_chars", 10 ** 9)):
        return False
    if len(_SENT.findall(t)) < int(check.get("min_sentences", 0)):
        return False
    return not (check.get("forbid") and re.search(check["forbid"], t))


def content_words(text: str) -> set[str]:
    return {w for w in speech_key(text).split() if len(w) >= 3 and w not in _STOPWORDS}


def uptake(turns: list[dict[str, Any]], k: int, header: str = "", back: int = 3) -> Optional[bool]:
    """Whether turn ``k`` takes up content another speaker INTRODUCED in their last ``back`` turns: a content word
    of those turns that neither the header nor any turn before them used. None when they introduced nothing."""
    me = turns[k]["speaker"]
    recent = [j for j in range(k - 1, -1, -1) if turns[j]["speaker"] != me][:back]
    if not recent:
        return None
    first = min(recent)
    before = content_words(header).union(*(content_words(t["text"]) for t in turns[:first]))
    introduced = set().union(*(content_words(turns[j]["text"]) for j in recent)) - before
    if not introduced:
        return None
    return bool(content_words(turns[k]["text"]) & introduced)


def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    v = [float(x) for x in xs if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def completion_meters(row: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    draws = row["draws"]
    first = draws[0]["text"] if draws else ""
    onset = text_loop_onset(first)
    return {
        "words": len(words(first)),
        "repeat_share": round(repeat_share(first), 4),
        "loop": onset is not None,
        "loop_onset": onset,
        "eos_share": _mean(d["ended_by"] == "eos" for d in draws),
        "cap_share": _mean(d["ended_by"] == "length" for d in draws),
        "form_pass": _mean(form_check(d["text"], item.get("check")) for d in draws) if item.get("check") else None,
        "sibling_distinct": distinctness(draws[0]["text"], draws[1]["text"]) if len(draws) > 1 else None,
        # verbatim recall of a held-out reference: the share of the draw's word 8-grams found in it
        "ref_overlap": _overlap(first, item["reference"]) if item.get("reference") else None,
    }


def _overlap(text: str, ref: str, n: int = 8) -> Optional[float]:
    grams = ngrams(words(text), n)
    if not grams:
        return None
    have = set(ngrams(words(ref), n))
    return round(sum(g in have for g in grams) / len(grams), 4)


def dialogue_meters(row: dict[str, Any], template_window: int = 12, template_max: int = 3) -> dict[str, Any]:
    from kotodama.posttrain.loom import echo_question, loop_onset

    turns = [t for t in row["turns"] if t["text"]]
    n = len(turns)
    keys = [speech_key(t["text"]) for t in turns]
    rep = [bool(keys[i]) and keys[i] in keys[max(0, i - 30):i] for i in range(n)]
    tmpl = [is_template(t["text"], [Turn(x["speaker"], x["text"]) for x in turns[:i]], template_window, template_max)
            if i else False for i, t in enumerate(turns)]
    cast = row.get("cast", [])
    narr = [spoken_text(t["text"], cast).strip() != t["text"].strip() for t in turns]
    off = [out_of_frame(t["text"]) for t in turns]
    all_draws = [t["ended_by"] for t in row["turns"]] + [s["ended_by"] for t in row["turns"] for s in t["siblings"]]
    sib = [distinctness(t["text"], t["siblings"][0]["text"]) for t in turns if t["siblings"]]
    ups = [uptake(turns, k, row.get("header", "")) for k in range(n)]
    echo = [("echo" in t.get("vetoes", [])) or (k > 0 and echo_question(t["text"], turns[k - 1]["text"]))
            for k, t in enumerate(turns)]
    onset = loop_onset(turns) if n else 0
    vetoed = [t for t in row["turns"] if t.get("vetoes")]
    return {
        "turns": n,
        "finish": row.get("finish"),
        "loop": onset < n,
        "loop_onset": onset if onset < n else None,
        "repeat_share": _mean(rep),
        "template_share": _mean(tmpl),
        "narration_share": _mean(narr),
        # character speech or scene narration, not text about a work or a page (narration.out_of_frame)
        "in_frame": _mean(not x for x in off),
        "eos_share": _mean(e == "eos" for e in all_draws),
        "cap_share": _mean(e == "length" for e in all_draws),
        "sibling_distinct": _mean(sib),
        "uptake": _mean(ups),
        "echo_share": _mean(echo),
        "veto_share": round(len(vetoed) / max(len(row["turns"]), 1), 4),
        "words_per_turn": _mean(len(words(t["text"])) for t in turns),
    }


def row_meters(row: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return dialogue_meters(row) if row["kind"] == "dialogue" else completion_meters(row, item)


# ── fluency (plain-model NLL of what each cell wrote) ────────────────────────


def fluency_items(rows: list[dict[str, Any]], items: dict[str, dict[str, Any]], tokenizer: Any,
                  max_prompt_tokens: int = 3000, max_completion_tokens: int = 512) -> list[dict[str, Any]]:
    """One ``/score_tokens`` item per generated text: a completion draw 0 given its prompt; every dialogue turn
    given the transcript before it (header and seed turns included)."""
    out = []

    def add(rid: str, prompt: str, text: str) -> None:
        p = tokenizer.encode(prompt, add_special_tokens=False)[-max_prompt_tokens:]
        c = tokenizer.encode(text, add_special_tokens=False)[:max_completion_tokens]
        if p and c:
            out.append({"id": rid, "prompt_ids": p, "completion_ids": c})

    for r in rows:
        key = f"{r['cell']}|{r['item']}"
        if r["kind"] == "completion":
            if r["draws"] and r["draws"][0]["text"]:
                add(key, items[r["item"]]["prompt"], r["draws"][0]["text"])
            continue
        prior = [Turn(t["speaker"], t["text"]) for t in r.get("seed_turns", [])]
        for k, t in enumerate(r["turns"]):
            if not t["text"]:
                continue
            add(f"{key}|{k}", render_prompt(Transcript(tuple(prior), r.get("header", "")), t["speaker"]),
                " " + t["text"])
            prior.append(Turn(t["speaker"], t["text"]))
    return out


def score_parallel(items: list[dict[str, Any]], score: Any, workers: int,
                   chunk: int = 512) -> dict[str, list[float]]:
    """``score`` (a list of ``/score_tokens`` items -> ``{id: [sum_nll, n_tokens]}``) over chunks of ``items``,
    ``workers`` chunks at a time; the merged per-item results."""
    from concurrent.futures import ThreadPoolExecutor

    parts = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    out: dict[str, list[float]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for res in ex.map(score, parts):
            out.update(res)
    return out


def fluency_by_row(scored: dict[str, list[float]]) -> dict[str, float]:
    """``{cell|item[|turn]: [sum_nll, n_tokens]}`` -> token-weighted mean NLL per ``cell|item``."""
    acc: dict[str, list[float]] = {}
    for k, (s, n) in scored.items():
        key = "|".join(k.split("|")[:2])
        a = acc.setdefault(key, [0.0, 0.0])
        a[0] += s
        a[1] += n
    return {k: round(s / n, 4) for k, (s, n) in acc.items() if n}


# ── the law transform (damage readout) ───────────────────────────────────────


def law_logprobs(logits: "Any", targets: "Any", temperature: Sequence[float], repetition_penalty: Sequence[float],
                 penalty_ids: Sequence[Sequence[int]], top_p: Sequence[float], min_p: Sequence[float]) -> "Any":
    """Per-row log-probability of ``targets`` under each row's sampling law, the arithmetic of the server's
    sampler (``kotodama.serve.server.sample_next_token``): the repetition penalty on ``penalty_ids`` (positive
    logits divided, negative multiplied), then temperature, then top-p and min-p truncation. ``logits`` (R, V),
    ``targets`` (R,); a target the truncation removes gets -inf."""
    import torch

    x = logits.float().clone()
    R = x.shape[0]
    for r in range(R):
        rp = float(repetition_penalty[r])
        if rp != 1.0 and len(penalty_ids[r]):
            ids = torch.tensor(sorted(set(int(i) for i in penalty_ids[r])), device=x.device, dtype=torch.long)
            v = x[r, ids]
            x[r, ids] = torch.where(v > 0, v / rp, v * rp)
    x = x / torch.tensor([float(t) for t in temperature], device=x.device).view(R, 1)
    for r in range(R):
        tp, mp = float(top_p[r]), float(min_p[r])
        if 0.0 < tp < 1.0:
            s, idx = x[r].sort(descending=True)
            p = s.softmax(-1)
            drop = (p.cumsum(-1) - p) >= tp
            s[drop] = float("-inf")
            x[r] = s.scatter(0, idx, s)
        if mp > 0.0:
            x[r] = x[r].masked_fill(x[r] < x[r].max() + math.log(mp), float("-inf"))
    lp = x.log_softmax(-1)
    return lp.gather(1, targets.view(R, 1).to(lp.device)).view(R)


def damage_set(run: Path, panels: dict[str, Panel], base_cell: str = "base",
               turns_per_dialogue: Sequence[int] = (1, 5, 9)) -> list[dict[str, Any]]:
    """The fixed continuation set of the damage readout: a completion panel's held-out ``reference`` when its item
    has one, else the base cell's first draw; for dialogue, the base cell's turns at ``turns_per_dialogue`` with the
    transcript before each as the prompt. Every cell is scored on the same set."""
    out = []
    for name, p in panels.items():
        rows = {r["item"]: r for r in read_jsonl(run / "gen" / f"{name}.jsonl") if r["cell"] == base_cell}
        for it in p.items:
            r = rows.get(it["id"])
            if p.kind == "completion":
                cont = it.get("reference") or (r["draws"][0]["text"] if r and r["draws"] else "")
                if cont:
                    out.append({"id": f"{name}|{it['id']}", "panel": name, "prompt": it["prompt"],
                                "continuation": cont})
                continue
            if r is None:
                continue
            prior = [Turn(t["speaker"], t["text"]) for t in r.get("seed_turns", [])]
            for k, t in enumerate(r["turns"]):
                if k in turns_per_dialogue and t["text"]:
                    out.append({"id": f"{name}|{it['id']}|{k}", "panel": name,
                                "prompt": render_prompt(Transcript(tuple(prior), r.get("header", "")), t["speaker"]),
                                "continuation": " " + t["text"]})
                prior.append(Turn(t["speaker"], t["text"]))
    return out


# ── summary ──────────────────────────────────────────────────────────────────

def summarize(run: Path, panels: dict[str, Panel], damage: Optional[dict[str, Any]] = None,
              bt: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Per panel, per cell: mean meters over items, mean fluency NLL, damage, and Bradley-Terry strength."""
    cells = {c["id"]: c for c in read_jsonl(run / "cells.jsonl")}
    out: dict[str, Any] = {}
    for name, p in panels.items():
        items = {it["id"]: it for it in p.items}
        flu_path = run / "fluency" / f"{name}.json"
        flu = json.loads(flu_path.read_text())["rows"] if flu_path.exists() else {}
        per: dict[str, dict[str, list[Any]]] = {}
        for r in read_jsonl(run / "gen" / f"{name}.jsonl"):
            if r["item"] not in items:
                continue
            m = row_meters(r, items[r["item"]])
            m["nll"] = flu.get(f"{r['cell']}|{r['item']}")
            d = per.setdefault(r["cell"], {})
            for k, v in m.items():
                if isinstance(v, (bool, int, float)) or v is None:
                    d.setdefault(k, []).append(None if v is None else float(v))
        table = {}
        for cid, d in per.items():
            c = cells.get(cid, {})
            row = {"ladder": c.get("ladder"), "control": c.get("control", False), "n": len(d.get("nll", [])),
                   **{k: _mean(v) for k, v in d.items()}}
            if damage and cid in damage.get("cells", {}):
                dm = damage["cells"][cid]
                row["damage"] = dm.get("panels", {}).get(name, dm.get("damage"))
            if bt and name in bt and cid in bt[name].get("strength", {}):
                row["bt"] = bt[name]["strength"][cid]
            table[cid] = row
        out[name] = table
    return out


def frontier_table(table: dict[str, dict[str, Any]], cols: Sequence[str]) -> str:
    order = sorted(table, key=lambda c: (str(table[c].get("ladder")), table[c].get("damage") or 0.0, c))
    lines = ["| cell | ladder | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for c in order:
        r = table[c]
        vals = ["" if r.get(k) is None else (f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k])) for k in cols]
        lines.append(f"| {c} | {r.get('ladder')} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _panels(specs: list[str]) -> list[Panel]:
    out = []
    for s in specs:
        name, sep, path = s.partition("=")
        if not sep or not re.fullmatch(r"[a-z0-9_-]+", name):
            raise SystemExit(f"--panel {s!r}: expected name=path (name of [a-z0-9_-])")
        out.append(load_panel(name, Path(path)))
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [lawbench] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="sampling-law bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage-a", help="write the stage-A cells")
    s.add_argument("--out", required=True)
    s.add_argument("--plan", default=None, help="json overriding keys of the stage-A plan")
    s.add_argument("--base_law", default=BASE_LAW)
    g = sub.add_parser("generate")
    g.add_argument("--cells", required=True)
    g.add_argument("--only", default="", help="comma list of cell ids (default: every cell)")
    g.add_argument("--panel", action="append", required=True, help="name=path (repeatable)")
    g.add_argument("--endpoint", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--limit", type=int, default=0, help="first N items of each panel (0 = all)")
    g.add_argument("--concurrency", type=int, default=256)
    g.add_argument("--max_turns", type=int, default=GenConfig.max_turns)
    g.add_argument("--expect_name", default="", help="refuse unless the endpoint's /info name is this")
    g.add_argument("--stamp", action="append", default=[], help="key=value provenance written into every row")
    f = sub.add_parser("fluency")
    f.add_argument("run")
    f.add_argument("--panel", action="append", required=True)
    f.add_argument("--endpoint", required=True)
    f.add_argument("--workers", type=int, default=8, help="scoring requests in flight (spread over the fleet)")
    d = sub.add_parser("damage-set")
    d.add_argument("run")
    d.add_argument("--panel", action="append", required=True)
    d.add_argument("--out", required=True)
    m = sub.add_parser("summary")
    m.add_argument("run")
    m.add_argument("--panel", action="append", required=True)
    m.add_argument("--damage", default=None)
    m.add_argument("--bt", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "stage-a":
        plan = json.loads(Path(a.plan).read_text()) if a.plan else None
        cells = stage_a(plan, a.base_law)
        Path(a.out).write_text("".join(json.dumps(c.to_row()) + "\n" for c in cells))
        logger.info("%d cells -> %s", len(cells), a.out)
        return 0
    if a.cmd == "generate":
        if a.expect_name:
            import httpx

            name = httpx.get(a.endpoint.rstrip("/") + "/info", timeout=30).json().get("name", "")
            if name != a.expect_name:
                raise SystemExit(f"REFUSING: {a.endpoint} serves {name!r}, not {a.expect_name!r}")
        cells = load_cells(Path(a.cells))
        if a.only:
            keep = set(a.only.split(","))
            cells = [c for c in cells if c.id in keep]
        cfg = GenConfig(max_turns=a.max_turns)
        if not all("=" in kv for kv in a.stamp):
            raise SystemExit("--stamp takes key=value")
        stamp = dict(kv.split("=", 1) for kv in a.stamp) or None
        stats = asyncio.run(generate(cells, _panels(a.panel), a.endpoint, Path(a.out), cfg, a.concurrency, a.limit,
                                     stamp=stamp))
        logger.info("done: %s", stats)
        return 1 if stats["failed"] else 0
    panels = {p.name: p for p in _panels(a.panel)}
    run = Path(a.run)
    if a.cmd == "fluency":
        from transformers import AutoTokenizer

        from kotodama.posttrain.corpus import score_fluency
        from kotodama.presets import TOKENIZER_NAME

        tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        (run / "fluency").mkdir(exist_ok=True)
        for name, p in panels.items():
            items = {it["id"]: it for it in p.items}
            fi = fluency_items(read_jsonl(run / "gen" / f"{name}.jsonl"), items, tok)
            res = score_parallel(fi, lambda part: score_fluency(part, a.endpoint, per_item=True)["items"],
                                 a.workers)
            rows = fluency_by_row(res)
            tot = [sum(v[0] for v in res.values()), sum(v[1] for v in res.values())]
            out = {"texts": len(fi), "tokens": int(tot[1]), "mean_nll": round(tot[0] / max(tot[1], 1), 4),
                   "rows": rows}
            (run / "fluency" / f"{name}.json").write_text(json.dumps(out))
            logger.info("%s: %d texts, mean NLL %s", name, len(fi), out["mean_nll"])
        return 0
    if a.cmd == "damage-set":
        rows = damage_set(run, panels)
        Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        logger.info("%d continuations -> %s", len(rows), a.out)
        return 0
    if a.cmd == "summary":
        damage = json.loads(Path(a.damage).read_text()) if a.damage else None
        bt = json.loads(Path(a.bt).read_text()) if a.bt else None
        summ = summarize(run, panels, damage, bt)
        (run / "summary.json").write_text(json.dumps(summ, indent=1))
        for name, table in summ.items():
            cols = ["damage", "nll", "bt", "loop", "repeat_share", "sibling_distinct", "eos_share"]
            cols += (["template_share", "narration_share", "in_frame", "uptake", "echo_share", "turns"]
                     if panels[name].kind == "dialogue" else ["form_pass", "ref_overlap", "words"])
            print(f"\n## {name}\n" + frontier_table(table, cols))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
