"""Scene seeds as a conditional chain: three drawn roots, then every later piece written by the model given the rest.

Each seed draws only its roots: an ideonomic theme, how the people are bound (a pair or a cast of 3-4 relationship
entry), and a register, all from ``kotodama.posttrain.seedpools`` pools; plus its arm tags (friction, secret arm,
cast arm), which enter the prompts as the notebook's own "Drawn:" line, and names from the names pool. The model then
fills a playwright's-notebook entry one field at a time, each step a short completion of the entry so far after
three worked entries:

    people (one line per name: who they are, fitting the bond) -> needs (from the theme) -> wants today -> the secret
    one keeps from another -> place -> what has just happened there -> the clock -> the opening page

so coherence comes from conditioning on everything already written, and diversity from the roots and the sampling.
Each short step draws two candidates and keeps the first that passes its check (length, garbling, a field label
leaking in, a group written as one person, a malformed secret line), with one redraw; a chain that still has
nothing ends there and is kept with its failed step. The page draws ``page_candidates`` and passes
``kotodama.posttrain.seedpools.gate_page`` (a cast needs all wants but one on the page).

    python -m kotodama.posttrain.seeds chain --endpoint URL --out DIR --pools POOLS_DIR [--n 3000]
    python -m kotodama.posttrain.seeds pick  --out DIR [--verdicts VERDICTS.jsonl]
    python -m kotodama.posttrain.seeds chain-report --out DIR

``--form native`` grows a different chain from the same roots (theme, relationship, and a plain genre label in
place of the register): the model writes a season-brochure programme note for the play, then the printed cast list
(name, age, one-line note; one line per drawn name), then the stage direction that opens the play, each a few-shot
document form in which premises occur naturally. The header is that cast list and stage direction; wants, secret
and clock are never asked for, only tagged afterwards when legible on the page.

``chain`` writes ``<out>/chains.jsonl`` (every chain, passed or not, with per-step attempts) and resumes by id.
``pick`` writes ``<out>/seeds.all.jsonl`` (every passing chain as a loom seed, minus the rejects of an optional
verdicts file: rows ``{id, verdict: keep|reject, ...}``) and ``<out>/seeds.jsonl``, a pick stratified by arm: each
(secret arm, cast arm) cell keeps its designed share of the largest total every cell can fill, friction kinds
interleaved within a cell, so the gates' different pass rates per arm do not set the arm mix.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import itertools
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from kotodama.posttrain import seedpools as sp

logger = logging.getLogger(__name__)

CHAIN_HEAD = ("A playwright's notebook of one-act plays. Each entry starts from what was drawn: a theme, a register, "
              "how the people are bound to each other, and how the play should go. The playwright then works the play "
              "out in order: who the people are, what each needs underneath, what each wants today, the secret one "
              "keeps from another, the place and what has just happened there to put them in one room, the clock, "
              "and last the play's opening page.\n\n")
SHORT_STEPS = ("place", "happening", "clock")
_FIELD = re.compile(r"\b(?:Theme|Register|Bound|Drawn|People|Needs|Wants today|Secret|Place|Happening|Clock|Page|"
                    r"No\.)\s*:", re.I)
_GROUP = re.compile(r"^(?:a |an |the )?(?:group|couple|pair|family|team|crowd|band|class|bunch|committee|board|"
                    r"crew|gang|trio|duo|panel|club|staff)\b|^(?:a |the )?(?:newlywed|married) couple\b|"
                    r"^(?:two|three|four|five|several|some|many)\b|^(?:people|men|women|children|kids|students)\b",
                    re.I)
_SECRET = re.compile(r"^\s*(.+?) keeps it from (.+?):\s*(.+)$")
_PAGE_LEAK = re.compile(r"(?m)^(?:People|Needs|Wants today)\s*:")
LIMITS = {"person": (10, 160), "need": (8, 120), "want": (8, 200), "place": (10, 160), "happening": (10, 200),
          "clock": (10, 200), "secret": (8, 240)}
CELL_SHARE = {("plain", "pair"): 0.35, ("hinted", "pair"): 0.35, ("plain", "cast"): 0.15, ("hinted", "cast"): 0.15}
CELL_SHARE_NATIVE = {("none", "pair"): 0.7, ("none", "cast"): 0.3}  # the native form has no secret arm


def is_group(person: str) -> bool:
    """A character line that describes several people ("a couple who...", "two nurses", "a group of academics")."""
    return bool(_GROUP.search(person.strip()))


def drawn_line(n_people: int, friction: str, secret_arm: str) -> str:
    count = {2: "two people", 3: "three people", 4: "four people"}.get(n_people, f"{n_people} people")
    return f"{count}; {sp.FRICTION_TEXT[friction]}; {sp.SECRET_TEXT[secret_arm]}"


def render_entry(no: int, st: dict[str, Any]) -> str:
    """The notebook entry as far as ``st`` is filled (no trailing newline after the last filled field)."""
    labels = [n["label"] for n in st["names"]]
    out = [f"No. {no}", f"Theme: {st['theme']}", f"Register: {st['register']}", f"Bound: {st['relationship']}",
           f"Drawn: {drawn_line(len(st['names']), st['friction'], st['secret_arm'])}"]
    if st.get("people"):
        out.append("People:")
        out += [f"- {n['text']} — {p}" for n, p in zip(st["names"], st["people"])]
    if st.get("needs"):
        out.append("Needs:")
        out += [f"- {lb}: {x}" for lb, x in zip(labels, st["needs"])]
    if st.get("wants"):
        out.append("Wants today:")
        out += [f"- {lb}: {x}" for lb, x in zip(labels, st["wants"])]
    if st.get("secret"):
        s = st["secret"]
        out.append(f"Secret: {s['holder']} keeps it from {s['target']}: {s['text']}")
    for key, title in (("place", "Place"), ("happening", "Happening"), ("clock", "Clock")):
        if st.get(key):
            out.append(f"{title}: {st[key]}")
    if st.get("page"):
        out += ["Page:", st["page"], sp.END_MARK]
    return "\n".join(out)


def example_state(ex: sp.HeaderExample) -> dict[str, Any]:
    names = [{"text": full, "label": sp.name_label(full) or full.split()[0]} for full, _c, _n in ex.people]
    return {"theme": ex.theme, "register": ex.register, "relationship": ex.relationship, "friction": ex.friction,
            "secret_arm": ex.secret_arm, "names": names, "people": [c for _f, c, _n in ex.people],
            "needs": [n for _f, _c, n in ex.people], "wants": list(ex.wants),
            "secret": {"holder": names[ex.holder]["label"], "target": names[ex.target]["label"], "text": ex.secret},
            "place": ex.setting, "happening": ex.inciting, "clock": ex.stakes, "page": ex.page}


EXAMPLES_TEXT = "".join(render_entry(i + 1, example_state(ex)) + "\n\n" for i, ex in enumerate(sp.HEADER_EXAMPLES))


def step_prompt(st: dict[str, Any], step: str, i: int = 0) -> str:
    """The document for one step: the head, the worked entries, the entry so far, and the open field."""
    no = len(sp.HEADER_EXAMPLES) + 1
    body = render_entry(no, st)
    labels = [n["label"] for n in st["names"]]
    if step == "person":
        lead = "People:\n" if i == 0 else ""
        tail = f"\n{lead}- {st['names'][i]['text']} —"
    elif step in ("need", "want"):
        title = "Needs:" if step == "need" else "Wants today:"
        lead = f"{title}\n" if i == 0 else ""
        tail = f"\n{lead}- {labels[i]}:"
    elif step == "secret":
        tail = "\nSecret:"
    elif step in SHORT_STEPS:
        tail = f"\n{step.capitalize()}:"
    elif step == "page":
        tail = "\nPage:\n"
    else:
        raise ValueError(f"unknown chain step {step!r}")
    return CHAIN_HEAD + EXAMPLES_TEXT + body + tail


def check_short(step: str, text: str, st: dict[str, Any]) -> tuple[Optional[Any], str]:
    """(value, "") for a usable completion of a short step, else (None, reason)."""
    t = " ".join(text.strip().split())
    if step == "secret":
        m = _SECRET.match(t)
        labels = [n["label"] for n in st["names"]]
        if not m:
            return None, "secret_format"
        holder, target, body = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if holder not in labels or target not in labels or holder == target:
            return None, "secret_people"
        t = body
        value: Any = {"holder": holder, "target": target, "text": body}
    else:
        value = t
    lo, hi = LIMITS["person" if step == "person" else step]
    if len(t) < lo:
        return None, "too_short"
    if len(t) > hi:
        return None, "too_long"
    if sp.mojibake(t) or re.search(r"\S{40,}", t):
        return None, "garbled"
    if _FIELD.search(t) or " — " in t and step != "person":
        return None, "field_leak"
    if step == "person" and is_group(t):
        return None, "group_as_person"
    return value, ""


async def run_step(complete: sp.Complete, st: dict[str, Any], step: str, i: int = 0, draws: int = 2,
                   n: int = 2) -> tuple[Optional[Any], list[str]]:
    """Draw ``n`` candidates up to ``draws`` times; the first usable value, and every rejection reason seen."""
    reasons: list[str] = []
    prompt = step_prompt(st, step, i)
    for _ in range(draws):
        for text, _finish, hit in await complete(prompt, n, ["\n"]):
            if hit:
                reasons.append("overlong")
                continue
            value, why = check_short(step, text, st)
            if value is not None:
                return value, reasons
            reasons.append(why)
    return None, reasons


def gate_chain_page(page: str, st: dict[str, Any]) -> Optional[str]:
    if _PAGE_LEAK.search(page):
        return "label_leak"
    c = {"secret_arm": st["secret_arm"],
         "characters": [{"name": n["text"], "label": n["label"], "character": {"text": p}, "need": {"text": nd}}
                        for n, p, nd in zip(st["names"], st["people"], st["needs"])]}
    parsed = {"wants": st["wants"], "secret": st["secret"]["text"], "page": page}
    k = len(st["names"])
    return sp.gate_page(parsed, c, wants_required=k if k == 2 else k - 1)


async def grow_chain(complete: sp.Complete, complete_page: sp.Complete, score: Optional[sp.Score],
                     st: dict[str, Any], page_candidates: int = 4, page_draws: int = 2) -> dict[str, Any]:
    """Fill ``st`` step by step; returns it with ``status`` (pass | fail), ``failed_step``, ``rejections`` and up to
    two rejected pages (``page_rejected``: [reason, text]) for reading what the gate refused."""
    st["rejections"] = {}
    k = len(st["names"])
    plan = ([("person", i) for i in range(k)] + [("need", i) for i in range(k)] + [("want", i) for i in range(k)]
            + [("secret", 0)] + [(s, 0) for s in SHORT_STEPS])
    for step, i in plan:
        value, reasons = await run_step(complete, st, step, i)
        if reasons:
            st["rejections"][f"{step}{i if step in ('person', 'need', 'want') else ''}"] = reasons
        if value is None:
            st.update(status="fail", failed_step=step)
            return st
        key = {"person": "people", "need": "needs", "want": "wants"}.get(step)
        if key:
            st.setdefault(key, []).append(value)
        else:
            st[step] = value
    page_reasons: list[str] = []
    st["page_rejected"] = []
    for _ in range(page_draws):
        outs = await complete_page(step_prompt(st, "page"), page_candidates, [sp.END_MARK, "\nNo. "])
        for text, _finish, hit in outs:
            if hit:
                page_reasons.append("overlong")
                continue
            page = text.split(sp.END_MARK)[0].split("\nNo. ")[0].strip()
            why = gate_chain_page(page, st)
            if why:
                page_reasons.append(why)
                if len(st["page_rejected"]) < 2:
                    st["page_rejected"].append([why, page])
                continue
            st["page"] = page
            break
        if st.get("page"):
            break
    st["rejections"]["page"] = page_reasons
    if not st.get("page"):
        st.update(status="fail", failed_step="page")
        return st
    if score is not None:
        try:
            st["nll"] = -(await score([(sp.DOC_START, st["page"])]))[0]
        except Exception as e:  # the NLL is a measurement, never a reason to lose the chain
            logger.error("scoring %s failed: %s", st["id"], e)
    st.update(status="pass", failed_step=None)
    return st


def draw_roots(pools: dict[str, list[dict[str, Any]]], n: int, seed: int, cast_share: float = 0.3) -> list[dict]:
    """``n`` chain starts: arms by ``seedpools.arm_schedule``; theme, relationship (by cast arm), register and names
    drawn least-used first; no two starts share all three roots."""
    rng = random.Random(seed)
    uses: collections.Counter = collections.Counter()
    seen: set[tuple[str, str, str]] = set()
    out = []
    for i, (friction, secret_arm, cast_arm) in enumerate(sp.arm_schedule(n, cast_share, rng)):
        k = 2 if cast_arm == "pair" else rng.choice((3, 4))
        for _ in range(50):
            theme = sp._pick(rng, pools["theme"], uses)[0]
            rel = sp._pick(rng, pools["relationship_pair" if cast_arm == "pair" else "relationship_cast"], uses)[0]
            reg = sp._pick(rng, pools["register"], uses)[0]
            if (theme["id"], rel["id"], reg["id"]) not in seen:
                break
        seen.add((theme["id"], rel["id"], reg["id"]))
        names = sp._pick_names(rng, pools["names"], uses, k)
        for e in (theme, rel, reg, *names):
            uses[e["id"]] += 1
        out.append({"id": f"ch{i:05d}", "friction": friction, "secret_arm": secret_arm, "cast_arm": cast_arm,
                    "theme": theme["text"], "relationship": rel["text"], "register": reg["text"],
                    "roots": {"theme": theme["id"], "relationship": rel["id"], "register": reg["id"],
                              "names": [x["id"] for x in names]},
                    "names": [{"text": x["text"], "label": sp.name_label(x["text"])} for x in names]})
    return out


async def run_chains(complete: sp.Complete, complete_page: sp.Complete, score: Optional[sp.Score],
                     starts: list[dict[str, Any]], out: Path, concurrency: int = 128,
                     page_candidates: int = 4, page_draws: int = 2, form: str = "slots") -> dict[str, Any]:
    """Grow every start not already in ``<out>/chains.jsonl`` (appended as each finishes)."""
    path = out / "chains.jsonl"
    done = {r["id"] for r in sp.read_jsonl(path)}
    todo = [s for s in starts if s["id"] not in done]
    logger.info("chains: %d starts, %d done, %d to grow", len(starts), len(done), len(todo))
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    count = {"n": 0, "pass": 0}
    f = path.open("a", encoding="utf-8")

    async def one(s: dict[str, Any]) -> None:
        async with sem:
            try:
                if form == "native":
                    st = await grow_native(complete, score, dict(s))
                else:
                    st = await grow_chain(complete, complete_page, score, dict(s), page_candidates, page_draws)
            except Exception as e:  # one chain's failure must not stop the run
                logger.error("chain %s failed: %s", s["id"], e)
                return
            f.write(json.dumps(st, ensure_ascii=False) + "\n")
            f.flush()
            count["n"] += 1
            count["pass"] += st["status"] == "pass"
            if count["n"] % 100 == 0:
                logger.info("chains %d/%d, %d pass (%.0fs)", count["n"], len(todo), count["pass"], time.time() - t0)

    try:
        await asyncio.gather(*(one(s) for s in todo))
    finally:
        f.close()
    rows = sp.read_jsonl(path)
    return {"chains": len(rows), "pass": sum(r["status"] == "pass" for r in rows),
            "failed_step": dict(collections.Counter(r["failed_step"] for r in rows if r["status"] != "pass"))}


def chain_seed(st: dict[str, Any]) -> dict[str, Any]:
    """A loom seed (``kotodama.posttrain.selfplay.parse_seed`` reads it) from a passing chain."""
    if st.get("form") == "native":
        return {"id": f"sc-{st['id']}", "cast": [n["label"] for n in st["names"]], "header": st["page"],
                "form": "native", "friction": st["friction"], "cast_arm": st["cast_arm"], "secret_arm": "none",
                "characters": [{"name": n["text"], "label": n["label"], "character": p}
                               for n, p in zip(st["names"], st["people"])],
                "genre": st["genre"], "theme": st["theme"], "relationship": st["relationship"],
                "blurb": st["blurb"], "scene": st["scene"], "tags": st.get("tags", {}), "roots": st["roots"],
                "nll": st.get("nll")}
    chars = [{"name": n["text"], "label": n["label"], "character": p, "need": nd, "want": w}
             for n, p, nd, w in zip(st["names"], st["people"], st["needs"], st["wants"])]
    return {"id": f"sc-{st['id']}", "cast": [n["label"] for n in st["names"]], "header": st["page"],
            "friction": st["friction"], "cast_arm": st["cast_arm"], "secret_arm": st["secret_arm"],
            "characters": chars, "secret": st["secret"], "theme": st["theme"], "register": st["register"],
            "relationship": st["relationship"], "place": st["place"], "happening": st["happening"],
            "clock": st["clock"], "roots": st["roots"], "nll": st.get("nll"),
            "secret_overlap": round(sp.overlap(st["secret"]["text"], st["page"]), 3)}


def stratified(rows: list[dict[str, Any]], seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The largest pick in which every (secret arm, cast arm) cell holds its ``CELL_SHARE`` (rounded down), friction
    kinds interleaved within each cell."""
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        cells[(r["secret_arm"], r["cast_arm"])].append(r)
    shares = CELL_SHARE_NATIVE if rows and all(r["secret_arm"] == "none" for r in rows) else CELL_SHARE
    total = min((len(cells.get(c, [])) / share for c, share in shares.items()), default=0)
    out: list[dict[str, Any]] = []
    for cell, share in shares.items():
        by_f: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for r in cells.get(cell, []):
            by_f[r["friction"]].append(r)
        for v in by_f.values():
            rng.shuffle(v)
        inter = [r for group in itertools.zip_longest(*(by_f[f] for f in sp.FRICTIONS)) for r in group
                 if r is not None]
        out += inter[:math.floor(total * share)]
    stats = {"available": {f"{a}/{b}": len(cells.get((a, b), [])) for a, b in shares},
             "picked": dict(collections.Counter(f"{r['secret_arm']}/{r['cast_arm']}" for r in out)),
             "picked_friction": dict(collections.Counter(r["friction"] for r in out)), "total": len(out)}
    return out, stats


def pick(out: Path, verdicts: Optional[Path] = None, nll_cut: float = 0.05, seed: int = 0) -> dict[str, Any]:
    """``<out>/seeds.all.jsonl`` + stratified ``<out>/seeds.jsonl`` + ``<out>/pick.stats.json``."""
    chains = [r for r in sp.read_jsonl(out / "chains.jsonl") if r["status"] == "pass"]
    rows = [chain_seed(r) for r in chains]
    nlls = sorted(r["nll"] for r in rows if r.get("nll") is not None)
    cut = nlls[max(0, math.ceil(len(nlls) * (1 - nll_cut)) - 1)] if nlls and nll_cut > 0 else None
    by_nll = [r for r in rows if cut is None or r.get("nll") is None or r["nll"] <= cut]
    rejected: dict[str, str] = {}
    if verdicts is not None:
        for v in sp.read_jsonl(verdicts):
            if v.get("verdict") == "reject":
                rejected[v["id"]] = v.get("category", "other")
    kept = [r for r in by_nll if r["id"] not in rejected]
    sp.write_jsonl(out / "seeds.all.jsonl", kept)
    picked, sstats = stratified(kept, seed)
    sp.write_jsonl(out / "seeds.jsonl", picked)
    stats = {"passing_chains": len(rows), "nll_cut_share": nll_cut, "nll_cut_value": cut,
             "cut_by_nll": len(rows) - len(by_nll), "filter_rejects": sum(r["id"] in rejected for r in by_nll),
             "kept_all": len(kept), "stratified": sstats}
    (out / "pick.stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def chain_report(out: Path, sample: int = 40, reject_sample: int = 20, verdicts: Optional[Path] = None,
                 seed: int = 0) -> dict[str, Any]:
    """``<out>/chain-report.json`` (pass rates by arm and stage, page gate reasons, filter verdicts), a reading sample
    of picked seeds (``<out>/reading-sample.md``) and of filter rejects (``<out>/filter-rejects-sample.md``)."""
    chains = sp.read_jsonl(out / "chains.jsonl")
    native = any(c.get("form") == "native" for c in chains)
    order = [*NATIVE_STEPS, "page"] if native else ["person", "need", "want", "secret", *SHORT_STEPS, "page"]
    by_arm: dict[str, Any] = {}
    for arm_key in ("cast_arm", "friction") if native else ("cast_arm", "secret_arm", "friction"):
        for arm in sorted({c[arm_key] for c in chains}):
            sub = [c for c in chains if c[arm_key] == arm]
            fails = collections.Counter(c["failed_step"] for c in sub if c["status"] != "pass")
            reach, stages = len(sub), {}
            for s in order:
                stages[s] = round((reach - fails[s]) / reach, 3) if reach else None
                reach -= fails[s]
            by_arm[f"{arm_key}={arm}"] = {"chains": len(sub), "pass": sum(c["status"] == "pass" for c in sub),
                                          "pass_rate": round(sum(c["status"] == "pass" for c in sub) / len(sub), 3),
                                          "stage_pass_rate": stages}
    page_reasons = collections.Counter(x for c in chains for x in c.get("rejections", {}).get("page", []))
    step_reasons = collections.Counter(f"{k.rstrip('0123456789')}:{x}" for c in chains
                                       for k, xs in c.get("rejections", {}).items() if k != "page" for x in xs)
    seeds = sp.read_jsonl(out / "seeds.jsonl")
    rep: dict[str, Any] = {"chains": len(chains), "pass": sum(c["status"] == "pass" for c in chains),
                           "by_arm": by_arm, "page_rejections": dict(page_reasons),
                           "step_rejections": dict(step_reasons.most_common()),
                           "pick": json.loads((out / "pick.stats.json").read_text())
                           if (out / "pick.stats.json").exists() else None,
                           "header_nn_similarity": sp.quantiles(sp.nearest_neighbour_sims([s["header"]
                                                                                          for s in seeds]))}
    vrows = sp.read_jsonl(verdicts) if verdicts is not None else []
    if vrows:
        rep["filter"] = {"verdicts": dict(collections.Counter(v["verdict"] for v in vrows)),
                         "reject_categories": dict(collections.Counter(v.get("category") for v in vrows
                                                                       if v["verdict"] == "reject")),
                         "reject_rate_by_arm": _reject_by_arm(vrows, chains)}
    (out / "chain-report.json").write_text(json.dumps(rep, indent=1))
    rng = random.Random(seed)
    _sample_md(out / "reading-sample.md", rng.sample(seeds, min(sample, len(seeds))), f"Reading sample: kept seeds "
               f"({min(sample, len(seeds))} of {len(seeds)} picked)", {})
    if vrows:
        rej = [v for v in vrows if v["verdict"] == "reject"]
        by_id = {f"sc-{c['id']}": c for c in chains if c["status"] == "pass"}
        rows = [chain_seed(by_id[v["id"]]) for v in rng.sample(rej, min(reject_sample, len(rej))) if v["id"] in by_id]
        _sample_md(out / "filter-rejects-sample.md", rows, f"Filter rejects ({len(rows)} of {len(rej)})",
                   {v["id"]: v for v in rej})
    return rep


def _reject_by_arm(vrows: list[dict[str, Any]], chains: list[dict[str, Any]]) -> dict[str, Any]:
    arms = {f"sc-{c['id']}": (c["cast_arm"], c["secret_arm"]) for c in chains}
    tally: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for v in vrows:
        if v["id"] in arms:
            for key in (f"cast_arm={arms[v['id']][0]}", f"secret_arm={arms[v['id']][1]}"):
                tally[key][0] += v["verdict"] == "reject"
                tally[key][1] += 1
    return {k: round(r / n, 3) for k, (r, n) in sorted(tally.items())}


def _sample_md(path: Path, rows: list[dict[str, Any]], title: str, verdicts: dict[str, dict[str, Any]]) -> None:
    md = [f"# {title}\n"]
    for s in rows:
        nll = s.get("nll")
        md.append(f"## {s['id']} · {s['cast_arm']} · {s['friction']} · secret {s['secret_arm']} · "
                  f"nll {None if nll is None else round(nll, 3)}\n")
        v = verdicts.get(s["id"])
        if v:
            md.append(f"**filter: {v['verdict']} / {v.get('category')}**: {v.get('reason', '')}\n")
        if s.get("form") == "native":
            md.append(f"*genre:* {s['genre']}  \n*theme:* {s['theme']}  \n*bound:* {s['relationship']}  \n"
                      f"*blurb:* {s['blurb']}  \n*tags:* {s['tags']}\n")
            md.append("```\n" + s["header"] + "\n```\n")
            continue
        md.append(f"*theme:* {s['theme']}  \n*register:* {s['register']}  \n*bound:* {s['relationship']}  \n"
                  f"*place:* {s['place']}  \n*happening:* {s['happening']}  \n*clock:* {s['clock']}\n")
        for ch in s["characters"]:
            md.append(f"- **{ch['name']}** — {ch['character']}; needs {ch['need']}; wants {ch['want']}")
        md.append(f"- secret ({s['secret']['holder']} → {s['secret']['target']}): {s['secret']['text']}\n")
        md.append("```\n" + s["header"] + "\n```\n")
    path.write_text("\n".join(md), encoding="utf-8")


# ── native form: blurb -> cast list -> opening stage direction ──────────────

GENRES = ("comedy", "drama", "thriller", "romance", "mystery", "farce", "tragedy", "family drama", "satire",
          "ghost story", "workplace comedy", "crime drama", "coming-of-age story", "black comedy", "melodrama",
          "western", "science fiction", "kitchen-sink drama", "romantic comedy", "political drama")
NATIVE_HEAD = ("The Lantern Theatre: new writing, this season and next. Each listing gives the genre, what the play "
               "is about and who it concerns, then the programme note, the characters as printed in the script, "
               "and the stage direction that opens the play.\n\n")
NATIVE_STEPS = ("blurb", "character", "scene")


@dataclass(frozen=True)
class NativeExample:
    genre: str
    theme: str
    relationship: str
    blurb: str
    characters: tuple[tuple[str, str], ...]  # (full name, "age, one-line note")
    scene: str


NATIVE_EXAMPLES: tuple[NativeExample, ...] = (
    NativeExample(
        "comedy", "owing a debt and the illusion of standing alone",
        "a piano teacher and the former student who now out-earns her",
        "Ines Moreau has taught piano for forty years and never asked anyone for anything. On the eve of her move "
        "into sheltered housing, her most famous former pupil turns up with a van, a bottle of wine and a plan for "
        "her old upright. What he has not mentioned is that the piano is already sold, to pay the rent she never "
        "told anyone she owed. A comedy about pride, debt and the people we refuse to need.",
        (("Ines Moreau", "seventy-one, a retired piano teacher who corrects everyone's posture"),
         ("Kwame Asante", "thirty-eight, her former pupil, now a session pianist with a van parked outside")),
        "Ines's flat, late afternoon in March. Everything is boxed except one chair and the piano. Kwame stands in "
        "the doorway holding a bottle of wine. The movers are due back at six."),
    NativeExample(
        "thriller", "small kindnesses and the belief that they end where they land",
        "two night-shift workers who have never had a full conversation",
        "At three in the morning a coach pulls out of a motorway services without one of its passengers, a "
        "ten-year-old boy. The night cashier wants to drive him to the ferry herself; the security guard wants to "
        "wait for the police, as the procedure says. Only one of them knows the boy hid on purpose. A taut "
        "two-hander about rules, mercy and a promise made to a child.",
        (("Dolores Vega", "fifty-five, the night cashier, who sings along to the till"),
         ("Bram Visser", "twenty-six, a security guard who follows every rule to the letter")),
        "A motorway service station, three in the morning, January. A child's rucksack with a name tag sits on the "
        "counter. Dolores has her car keys in her hand; Bram is on hold to the police. The boy's coach sails from "
        "the ferry port at half past five."),
    NativeExample(
        "ghost story", "inheritance and the hope that children teach their parents",
        "a widowed father, his grown daughter, and the estate agent selling the family house",
        "Every viewing of the old house on Mill Lane has ended the same way: the buyers leave before they reach the "
        "kitchen, and nobody will say why. Anna Hale drives up overnight to sell her childhood home out from under "
        "her stubborn father, and finds him laying three places at the table and the estate agent refusing to go "
        "upstairs. A quiet ghost story about grief, property and what a house remembers.",
        (("Peter Hale", "sixty-eight, a retired signalman who will not leave"),
         ("Anna Hale", "thirty-nine, his daughter, a paramedic who has driven up overnight"),
         ("Joseph Okoro", "forty-two, the estate agent, polite and very keen to sell")),
        "The kitchen of a terraced house on Mill Lane, an hour before dawn in November. The kettle is coming to the "
        "boil. Peter sets out three cups; Anna still has her coat on; Joseph waits in the hall with the keys. The "
        "first viewing is at nine."),
    NativeExample(
        "farce", "the illusion of a fresh start",
        "three sisters and the eldest's fiance, on the night before the wedding",
        "The night before Ruth Callaghan's wedding, her two sisters arrive at the farmhouse with a hip flask, a "
        "suitcase and a secret apiece. By midnight the cake has been eaten by a calf, the groom has lost both rings "
        "in the yard, and the youngest has announced she is not going back to Sydney alone. A farce about family, "
        "cattle and the impossibility of starting over in the house you grew up in.",
        (("Ruth Callaghan", "forty-four, the eldest, a dairy farmer who never sits down"),
         ("Maeve Callaghan", "thirty-six, the middle sister, a solicitor with a hip flask"),
         ("Orla Callaghan", "twenty-nine, the youngest, back from Australia with news"),
         ("Tom Brennan", "forty-seven, the fiance, a vet who has lost the rings")),
        "The farmhouse kitchen, eleven at night, the eve of the wedding. A half-iced cake stands on the table under "
        "a tea towel. Tom is on his hands and knees by the door with a torch; Maeve pours from her hip flask into "
        "three teacups. The church expects them at ten."),
)


def native_entry(no: int, st: dict[str, Any]) -> str:
    """A season listing as far as ``st`` is filled: the roots line, then blurb, characters and scene."""
    names = [n["text"] for n in st["names"]]
    who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    out = [f"No. {no}. {st['genre'].capitalize()}, about {st['theme']}; {st['relationship']}: {who}."]
    if st.get("blurb"):
        out.append(st["blurb"])
    if st.get("people"):
        out.append("Characters:")
        out += [f"{n['text']}, {p}" for n, p in zip(st["names"], st["people"])]
    if st.get("scene"):
        out += ["The play opens:", st["scene"]]
    return "\n".join(out)


def _native_example_state(ex: NativeExample) -> dict[str, Any]:
    return {"genre": ex.genre, "theme": ex.theme, "relationship": ex.relationship, "blurb": ex.blurb,
            "names": [{"text": n, "label": sp.name_label(n)} for n, _ in ex.characters],
            "people": [c for _, c in ex.characters], "scene": ex.scene}


NATIVE_EXAMPLES_TEXT = "".join(native_entry(i + 1, _native_example_state(ex)) + "\n\n"
                               for i, ex in enumerate(NATIVE_EXAMPLES))


def native_prompt(st: dict[str, Any], step: str, i: int = 0) -> str:
    no = len(NATIVE_EXAMPLES) + 1
    body = native_entry(no, st)
    if step == "blurb":
        tail = "\n"
    elif step == "character":
        tail = ("\nCharacters:" if i == 0 else "") + f"\n{st['names'][i]['text']},"
    elif step == "scene":
        tail = "\nThe play opens:\n"
    else:
        raise ValueError(f"unknown native step {step!r}")
    return NATIVE_HEAD + NATIVE_EXAMPLES_TEXT + body + tail


NATIVE_LIMITS = {"blurb": (150, 900), "character": (8, 160), "scene": (60, 700)}
_SENTENCE = re.compile(r"[.!?](?:\s|$)")
_CLOCK = re.compile(r"\b(?:at|by|until|before)\s+(?:\d{1,2}(?:[:.]\d\d)?|one|two|three|four|five|six|seven|eight|"
                    r"nine|ten|eleven|twelve|noon|midnight|dawn|dusk)\b|o'clock|\bdue\b|\bin (?:an|one|two|three|"
                    r"ten|twenty|thirty) (?:hour|minute)s?\b", re.I)


def check_native(step: str, text: str, st: dict[str, Any]) -> tuple[Optional[str], str]:
    t = " ".join(text.strip().split())
    lo, hi = NATIVE_LIMITS[step]
    if len(t) < lo:
        return None, "too_short"
    if len(t) > hi:
        return None, "too_long"
    if sp.mojibake(t) or re.search(r"\S{40,}", t):
        return None, "garbled"
    if re.search(r"\b(?:Characters|The play opens|No\. \d)\b", t):
        return None, "field_leak"
    if step == "blurb" and len(_SENTENCE.findall(t)) < 2:
        return None, "one_sentence"
    if step == "character":
        note = re.sub(r"^[\w-]+,\s*", "", t)  # the age, then the note
        if is_group(note):
            return None, "group_as_person"
    if step == "scene" and not any(n["label"] in t for n in st["names"]):
        return None, "nobody_on_stage"
    return t, ""


async def grow_native(complete: sp.Complete, score: Optional[sp.Score], st: dict[str, Any],
                      draws: int = 3, n: int = 2) -> dict[str, Any]:
    """Blurb, then one cast line per name, then the opening stage direction; each step the first of ``n``
    candidates (up to ``draws`` draws) that passes ``check_native``. The header is the cast lines and the stage
    direction; it then passes the mechanical part of ``seedpools.gate_page``."""
    st["form"] = "native"
    st["rejections"] = {}
    plan = [("blurb", 0)] + [("character", i) for i in range(len(st["names"]))] + [("scene", 0)]
    for step, i in plan:
        value, reasons = None, []
        for _ in range(draws):
            for text, _f, hit in await complete(native_prompt(st, step, i), n, ["\n"]):
                if hit:
                    reasons.append("overlong")
                    continue
                value, why = check_native(step, text, st)
                if value is not None:
                    break
                reasons.append(why)
            if value is not None:
                break
        if reasons:
            st["rejections"][f"{step}{i if step == 'character' else ''}"] = reasons
        if value is None:
            st.update(status="fail", failed_step=step)
            return st
        if step == "character":
            st.setdefault("people", []).append(value)
        else:
            st[step] = value
    page = "\n".join(f"{nm['text']}, {p}" for nm, p in zip(st["names"], st["people"])) + "\n\n" + st["scene"]
    c = {"secret_arm": "hinted",
         "characters": [{"name": nm["text"], "label": nm["label"], "character": {"text": p}, "need": {"text": ""}}
                        for nm, p in zip(st["names"], st["people"])]}
    why = sp.gate_page({"wants": [], "secret": "", "page": page}, c, wants_required=0)
    st["rejections"]["page"] = [why] if why else []
    if why:
        st.update(status="fail", failed_step="page", page_rejected=[[why, page]])
        return st
    st["page"] = page
    st["tags"] = {"clock_on_page": bool(_CLOCK.search(st["scene"])),
                  "wants_on_page": sum(bool(re.search(r"\bwants?\b|\bhopes?\b|\bneeds?\b", p))
                                       for p in st["people"])}
    if score is not None:
        try:
            st["nll"] = -(await score([(sp.DOC_START, page)]))[0]
        except Exception as e:  # the NLL is a measurement, never a reason to lose the chain
            logger.error("scoring %s failed: %s", st["id"], e)
    st.update(status="pass", failed_step=None)
    return st


def assign_genres(starts: list[dict[str, Any]], seed: int) -> None:
    """A plain genre label per start, drawn from ``GENRES`` by the start's id (stable across reruns)."""
    for s in starts:
        s.setdefault("genre", random.Random(f"{seed}:{s['id']}:genre").choice(GENRES))


# ── CLI ──────────────────────────────────────────────────────────────────────


def main_chain(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="conditional-chain scene seeds (python -m kotodama.posttrain.seeds chain)")
    ap.add_argument("--endpoint", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pools", required=True, help="directory holding pools/{theme,relationship_pair,"
                                                   "relationship_cast,register,names}.jsonl")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0, help="grow only the first N starts (0 = all)")
    ap.add_argument("--form", choices=["slots", "native"], default="slots",
                    help="slots: people/needs/wants/secret/place/happening/clock/page; native: programme note, "
                         "cast list, opening stage direction")
    ap.add_argument("--cast_share", type=float, default=0.3)
    ap.add_argument("--law", default="tsumugi")
    ap.add_argument("--temperature", type=float, default=1.0, help="short steps")
    ap.add_argument("--page_temperature", type=float, default=0.8, help="the opening page (rendering, not invention)")
    ap.add_argument("--page_repetition_penalty", type=float, default=None,
                    help="override the law's repetition penalty for the page, which must repeat names and facts")
    ap.add_argument("--page_candidates", type=int, default=4)
    ap.add_argument("--page_draws", type=int, default=2, help="page draws until one candidate passes the gate")
    ap.add_argument("--no_score", action="store_true")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    starts_path = out / "starts.jsonl"
    starts = sp.read_jsonl(starts_path)
    if not starts:
        pdir = Path(a.pools) / "pools"
        pools = {k: sp.read_jsonl(pdir / f"{k}.jsonl")
                 for k in ("theme", "relationship_pair", "relationship_cast", "register", "names")}
        missing = [k for k, v in pools.items() if not v]
        if missing:
            raise SystemExit(f"pools missing or empty under {pdir}: {missing}")
        starts = draw_roots(pools, a.n, a.seed, a.cast_share)
        sp.write_jsonl(starts_path, starts)
    if a.form == "native":
        assign_genres(starts, a.seed)
    if a.limit:
        starts = starts[:a.limit]
    complete, close, sampling = sp._completer(a.endpoint, a.law, a.temperature, 96)
    complete_page, close_page, page_sampling = sp._completer(a.endpoint, a.law, a.page_temperature, 480,
                                                             a.page_repetition_penalty)
    score, sclose = (None, None)
    if not a.no_score:
        from kotodama.posttrain.loom import http_scorer

        score, sclose = http_scorer(a.endpoint, max_completion_tokens=1024)

    async def go() -> dict[str, Any]:
        try:
            return await run_chains(complete, complete_page, score, starts, out, a.concurrency, a.page_candidates,
                                    a.page_draws, a.form)
        finally:
            await close()
            await close_page()
            if sclose is not None:
                await sclose()

    stats = asyncio.run(go())
    (out / "chain.run.json").write_text(json.dumps({"args": vars(a), "sampling": sampling,
                                                    "page_sampling": page_sampling, **stats}, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


def main_pick(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="passing chains -> stratified seeds (python -m kotodama.posttrain.seeds "
                                             "pick)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verdicts", default=None, help="JSONL of {id, verdict: keep|reject, category, reason}")
    ap.add_argument("--nll_cut", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    print(json.dumps(pick(Path(a.out), Path(a.verdicts) if a.verdicts else None, a.nll_cut, a.seed), indent=1))
    return 0


def main_chain_report(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="chain report + reading samples (python -m kotodama.posttrain.seeds "
                                             "chain-report)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verdicts", default=None)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--reject_sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rep = chain_report(Path(a.out), a.sample, a.reject_sample, Path(a.verdicts) if a.verdicts else None, a.seed)
    print(json.dumps(rep, indent=1))
    return 0


COMMANDS = {"chain": main_chain, "pick": main_pick, "chain-report": main_chain_report}
