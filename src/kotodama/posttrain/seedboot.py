"""Scene seeds with no hand-written examples: frames, pools and listings all come from the model's own text.

Every phrase in a generator's context imprints the data it produces, so this pipeline puts no written-for-the-purpose
example text in front of the model:

``boot-frames``  The model writes the introduction of a document from a bare title line ("Catalogue", "Contents",
                 ...), ``per_title`` draws each. A reader (outside this repository) marks which read as coherent
                 introductions; ``boot-frames-pick`` keeps the lowest-NLL coherent intro per title, up to ``k``.
                 Each kept intro is a frame; every seed records its frame id.
``boot-organon`` Pools as Gunkel organons (https://ideonomy.mit.edu): a titled, numbered list whose first entries are
                 drawn from a real organon list (``--lists``), continued by the model. Themes continue an illusions
                 list with the open entry starting from one side of an antisyzygy (the crossing); names continue a
                 guest list whose first entries are names the model wrote earlier.
``boot``         One bootstrap pass. Each seed's roots are a theme from the pool, a relationship crossing of two real
                 organon atoms (an antisyzygy and a life moment), and names; its listing is the model's continuation
                 of the frame, then of the roots line: the listing text, a cast line per name after "Characters:", and
                 the stage direction after "The play opens:". Pass 0 shows the frame and the roots line only; later
                 passes put 2-4 of the model's own kept listings from earlier passes between the frame and the new
                 entry, different ones per seed (``--exemplar_select kmedoids`` draws them from a health-gated k-medoids pool
                 in a text embedding). The genre is whatever the listing names, tagged afterwards. Arms: the header
                 carries the listing text above the cast list (``note``) or not (``bare``).
``boot-fan``     The frame-belief readout: ``n`` draws at each frame's boundary (where the first listing would start),
                 for an outside classifier to label by document type and genre.
``boot-variety`` Distribution shift between frames and passes, led by the held-out AUC of a length-residualized
                 logistic probe per pair of groups (GroupKFold by theme root, the length-only AUC beside it), in a
                 pluggable embedding space (``--embedder tfidf`` |
                 ``minilm`` | ``npz:PATH`` for vectors computed elsewhere, keyed by seed id) with a pluggable distance
                 (``energy`` | ``mmd``) and a document-level permutation null; within-group dispersion; the guards
                 beside every number (fluency: mean NLL under the unframed model, repeated 4-gram share, 8-gram loops;
                 reader: an outside filter's keep rate per group when ``--verdicts`` is given).

    python -m kotodama.posttrain.seeds boot-frames  --endpoint URL --out DIR
    python -m kotodama.posttrain.seeds boot-frames-pick --out DIR --verdicts FRAME_VERDICTS.jsonl
    python -m kotodama.posttrain.seeds boot-organon --endpoint URL --out DIR --lists LISTS --names_seed NAMES_JSONL
    python -m kotodama.posttrain.seeds boot --endpoint URL --out DIR --pass 0 --n 30
    python -m kotodama.posttrain.seeds boot --endpoint URL --out DIR --pass 1 --n 30 --verdicts V0.jsonl
    python -m kotodama.posttrain.seeds boot-fan --endpoint URL --out DIR --n 64
    python -m kotodama.posttrain.seeds boot-variety --out DIR --passes 0,1 [--verdicts V.jsonl ...]
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from kotodama.posttrain import seedchain as sc
from kotodama.posttrain import seedpools as sp

logger = logging.getLogger(__name__)

FRAME_TITLES = ("Catalogue", "Contents", "Index", "Programme", "Anthology", "Listings")
LISTING_TAIL = "Characters:"
SCENE_TAIL = "The play opens:"
GENRE_WORDS = ("comedy", "drama", "tragedy", "farce", "thriller", "romance", "mystery", "satire", "melodrama",
               "fable", "fantasy", "horror", "western", "musical", "monologue", "ghost story", "love story",
               "crime", "adventure", "science fiction", "opera", "tragicomedy", "parable", "allegory", "fairy tale",
               "detective story", "black comedy", "burlesque", "elegy", "pantomime")
_GENRE = re.compile(r"\b((?:[a-z][a-z-]+ ){0,2}(?:" + "|".join(sorted(GENRE_WORDS, key=len, reverse=True)) + r"))\b",
                    re.I)
ORGANONS = {
    # kind: (title lines, list file in --lists, crossing list, prefix side)
    "theme": [("The Study of Illusions\n\n68 Examples of Illusions", "illusion_examples"),
              ("The Study of Illusions\n\n76 Causes, Sources, and Bases of Illusions", "illusion_causes")],
}


# ── shared pieces ────────────────────────────────────────────────────────────


def genre_tag(text: str) -> str:
    """The genre the listing names (its last match: listings close on "A comedy about ..."), or "none"."""
    hits = _GENRE.findall(text)
    return hits[-1].lower() if hits else "none"


def repetition_guards(text: str) -> dict[str, float]:
    """Fluency guards on a header: the share of word 4-grams that repeat an earlier one, and whether any word 8-gram
    occurs twice (a loop)."""
    w = re.findall(r"\w+", text.lower())
    g4 = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    g8 = collections.Counter(tuple(w[i:i + 8]) for i in range(len(w) - 7))
    rep4 = 1 - len(set(g4)) / len(g4) if g4 else 0.0
    return {"rep4": round(rep4, 4), "loop8": bool(g8) and max(g8.values()) > 1}


def numbered_list(title: str, items: Sequence[str], open_prefix: str = "") -> str:
    """The title, a blank line, the items numbered from 1, and the open next entry (its number, then
    ``open_prefix`` when given; no trailing space, so the model writes the word boundary itself)."""
    body = "".join(f"{i + 1}. {x}\n" for i, x in enumerate(items))
    head = f"{title}\n\n{body}{len(items) + 1}."
    return f"{head} {open_prefix}" if open_prefix else head


_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")


def parse_numbered(text: str, hit_cap: bool, lead: str = "") -> list[str]:
    """Entries of a completion that continues an open numbered entry: the first line (after ``lead``), then each
    following line of the form "N. entry", up to the first other line. A last line cut by the cap is dropped."""
    lines = (lead + text).split("\n")
    if hit_cap and lines:
        lines = lines[:-1]
    out: list[str] = []
    for i, ln in enumerate(lines):
        if i == 0:
            out.append(sp.clean_entry(ln))
            continue
        m = _NUMBERED.match(ln)
        if not m:
            break
        out.append(sp.clean_entry(m.group(2)))
    return out


# ── frames ───────────────────────────────────────────────────────────────────


def trim_intro(text: str, hit_cap: bool) -> str:
    """An intro as written; one cut by the token cap ends at its last paragraph break, else its last sentence end,
    so a frame never stops mid-sentence."""
    t = text.strip()
    if not hit_cap:
        return t
    if "\n\n" in t:
        return t[:t.rindex("\n\n")].strip()
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", t)]
    return t[:ends[-1]].strip() if ends else ""


async def write_frames(complete: sp.Complete, out: Path, titles: Sequence[str], per_title: int,
                       score: Optional[sp.Score]) -> list[dict[str, Any]]:
    """``per_title`` introductions per bare title line; writes ``<out>/frames.candidates.jsonl``."""
    rows: list[dict[str, Any]] = []
    outs = await asyncio.gather(*(complete(f"{t}\n\n", per_title, ["\n\n\n", "\n1.", "\n1 "], user=f"frame:{t}")
                                  for t in titles))
    for t, raw in zip(titles, outs):
        for k, (text, _f, hit) in enumerate(raw):
            intro = trim_intro(text, hit)
            g = repetition_guards(intro)
            ok = 80 <= len(intro) <= 1500 and not sp.mojibake(intro) and not g["loop8"] and g["rep4"] < 0.3
            rows.append({"id": f"{t.lower()}-{k}", "title": t, "intro": intro, "mechanical_ok": ok,
                         "truncated": hit})
    if score is not None:
        lps = await score([(sp.DOC_START, f"{r['title']}\n\n{r['intro']}") for r in rows])
        for r, lp in zip(rows, lps):
            r["nll"] = -lp
    sp.write_jsonl(out / "frames.candidates.jsonl", rows)
    return rows


def pick_frames(out: Path, verdicts: Path, k: int, control: int = 0) -> list[dict[str, Any]]:
    """Coherent (verdict keep) mechanically-sound intros, up to ``k`` per title (lowest NLL first): a large pool, since
    each seed draws its frame at random and no single context's imprint should be shared corpus-wide; frame ids
    ``F0..``; writes ``<out>/frames.jsonl`` and every rejected candidate with the reader's reason to
    ``<out>/frames.rejected.jsonl``. The reader's choice is itself a selection imprint, so ``control`` adds that many
    reader-rejected frames as a control arm (``FC0..``, ``control: true``): the lowest-NLL mechanically-sound
    rejects not called incoherent (borderline ones)."""
    cands = sp.read_jsonl(out / "frames.candidates.jsonl")
    vrows = {v["id"]: v for v in sp.read_jsonl(verdicts)}
    keep = {i for i, v in vrows.items() if v.get("verdict") == "keep"}
    sp.write_jsonl(out / "frames.rejected.jsonl",
                   [{**r, "verdict": vrows.get(r["id"], {}).get("verdict", "unjudged"),
                     "category": vrows.get(r["id"], {}).get("category"),
                     "reason": vrows.get(r["id"], {}).get("reason")} for r in cands if r["id"] not in keep])
    by_title: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in sorted(cands, key=lambda r: r.get("nll", 0.0)):
        if r["mechanical_ok"] and r["id"] in keep and len(by_title[r["title"]]) < k:
            by_title[r["title"]].append(r)
    chosen = [r for t in sorted(by_title) for r in by_title[t]]
    frames = [{"frame": f"F{i}", "source": r["id"], "title": r["title"], "text": f"{r['title']}\n\n{r['intro']}",
               "nll": r.get("nll"), "control": False} for i, r in enumerate(chosen)]
    borderline = sorted((r for r in cands if r["mechanical_ok"] and r["id"] in vrows and r["id"] not in keep
                         and vrows[r["id"]].get("category") != "incoherent"), key=lambda r: r.get("nll", 0.0))
    frames += [{"frame": f"FC{i}", "source": r["id"], "title": r["title"], "text": f"{r['title']}\n\n{r['intro']}",
                "nll": r.get("nll"), "control": True, "reader_reason": vrows[r["id"]].get("reason")}
               for i, r in enumerate(borderline[:control])]
    sp.write_jsonl(out / "frames.jsonl", frames)
    return frames


# ── organon pools ────────────────────────────────────────────────────────────


async def grow_organon(complete: sp.Complete, kind: str, lists: dict[str, list[str]], target: int,
                       rng: random.Random, seed_items: Sequence[str] = (), shown: int = 8,
                       parallel: int = 32, per_request: int = 2, max_rounds: int = 40,
                       threshold: float = 0.4) -> dict[str, Any]:
    """A pool continuing organon documents (``theme``) or a guest list (``names``); filtered by
    ``seedpools.filter_pool`` (hygiene, duplicates, dissimilarity)."""
    axis = sp.AXES["theme" if kind == "theme" else "names"]
    state = sp.PoolState(kind)
    res = sp.filter_pool(state, axis, threshold)
    for rnd in range(max_rounds):
        if len(res["kept"]) >= target:
            break
        reqs = []
        for _ in range(parallel):
            if kind == "theme":
                title, list_name = rng.choice(ORGANONS["theme"])
                items = rng.sample(lists[list_name], min(shown, len(lists[list_name])))
                side = rng.choice(rng.choice(lists["antisyzygies"]).split(" and ")).strip()
                prefix = side[:1].upper() + side[1:]
                reqs.append((numbered_list(title, items, prefix), prefix, [list_name, side]))
            else:
                items = rng.sample(list(seed_items), min(shown, len(seed_items)))
                reqs.append((numbered_list("Guest List", items), "", None))
        outs = await asyncio.gather(*(complete(p, per_request, ["\n\n"]) for p, _l, _c in reqs))
        for (_p, lead, cross), raw in zip(reqs, outs):
            state.requests += 1
            for text, _f, hit in raw:
                for j, entry in enumerate(parse_numbered(text, hit, lead)):
                    state.lines.append({"text": entry, "crossing": cross if j == 0 else None,
                                        "source": "model" if j == 0 else "model-continued", "line": j})
        res = sp.filter_pool(state, axis, threshold)
        logger.info("organon %s round %d: %d lines, %d kept", kind, rnd + 1, len(state.lines), len(res["kept"]))
    res["kept"] = res["kept"][:target]
    res["counts"].update(final=len(res["kept"]), requests=state.requests, threshold=threshold)
    return res


# ── bootstrap passes ─────────────────────────────────────────────────────────


def roots_line(n: int, st: dict[str, Any]) -> str:
    """"n. Names — theme; relationship; moment", empty parts left out."""
    names = [x["text"] for x in st["names"]]
    who = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    parts = [p for p in (st.get("theme", ""), st.get("pair", ""), st.get("moment", "").lower()) if p]
    return f"{n}. {who} — {'; '.join(parts)}"


def as_exemplar(row: dict[str, Any]) -> dict[str, Any]:
    """A kept seed row as a listing the boot prompt can show: boot rows as they are; a ``seedchain`` native row
    (programme note ``blurb``, cast ``characters``, ``scene``) mapped onto the same fields, its relationship in the
    relationship slot and no moment."""
    if row.get("form") == "boot":
        return row
    chars = row.get("characters", [])
    return {"id": row["id"], "names": [{"text": c["name"], "label": c["label"]} for c in chars],
            "theme": row.get("theme", ""), "pair": row.get("relationship", ""), "moment": "",
            "listing": row.get("blurb") or row.get("listing", ""), "people": [c["character"] for c in chars],
            "scene": row.get("scene", ""), "nll": row.get("nll"), "guards": row.get("guards", {})}


def render_listing(n: int, st: dict[str, Any]) -> str:
    """A listing as far as ``st`` is filled."""
    out = [roots_line(n, st)]
    if st.get("listing"):
        out.append(st["listing"])
    if st.get("people"):
        out.append(LISTING_TAIL)
        out += [f"{x['text']}, {p}" for x, p in zip(st["names"], st["people"])]
    if st.get("scene"):
        out += [SCENE_TAIL, st["scene"]]
    return "\n".join(out)


def boot_prompt(frame: str, exemplars: Sequence[dict[str, Any]], st: dict[str, Any], step: str, i: int = 0) -> str:
    body = "".join(render_listing(j + 1, ex) + "\n\n" for j, ex in enumerate(exemplars))
    entry = render_listing(len(exemplars) + 1, st)
    if step == "listing":
        tail = "\n"
    elif step == "character":
        tail = (f"\n{LISTING_TAIL}" if i == 0 else "") + f"\n{st['names'][i]['text']},"
    elif step == "scene":
        tail = f"\n{SCENE_TAIL}\n"
    else:
        raise ValueError(f"unknown boot step {step!r}")
    return f"{frame}\n\n{body}{entry}{tail}"


LIMITS = {"listing": (60, 1000), "character": (8, 160), "scene": (60, 700)}


def check_step(step: str, text: str, st: dict[str, Any]) -> tuple[Optional[str], str]:
    t = text.strip() if step == "listing" else " ".join(text.strip().split())
    lo, hi = LIMITS[step]
    if len(t) < lo:
        return None, "too_short"
    if len(t) > hi:
        return None, "too_long"
    if sp.mojibake(t) or re.search(r"\S{40,}", t):
        return None, "garbled"
    if LISTING_TAIL in t or SCENE_TAIL in t:
        return None, "field_leak"
    if step == "listing" and _NUMBERED.match(t.split("\n")[-1] or ""):
        return None, "next_entry"
    if step == "character" and sc.is_group(re.sub(r"^[\w-]+,\s*", "", t)):
        return None, "group_as_person"
    if step == "scene" and not any(x["label"] in t for x in st["names"]):
        return None, "nobody_on_stage"
    return t, ""


def header_of(st: dict[str, Any]) -> str:
    cast = "\n".join(f"{x['text']}, {p}" for x, p in zip(st["names"], st["people"]))
    body = f"{cast}\n\n{st['scene']}"
    return f"{st['listing']}\n\n{body}" if st["arm"] == "note" else body


async def grow_boot(complete: sp.Complete, score: Optional[sp.Score], st: dict[str, Any], frame: str,
                    exemplars: Sequence[dict[str, Any]], draws: int = 3, n: int = 2) -> dict[str, Any]:
    st["rejections"] = {}
    k = len(exemplars) + 1
    plan = [("listing", 0)] + [("character", i) for i in range(len(st["names"]))] + [("scene", 0)]
    for step, i in plan:
        # a listing is followed by its cast tail; a model that has seen listings writes straight into it
        stops = ["\n\n", f"\n{k + 1}.", f"\n{LISTING_TAIL}"] if step == "listing" else ["\n"]
        value, reasons = None, []
        for _ in range(draws):
            for text, _f, hit in await complete(boot_prompt(frame, exemplars, st, step, i), n, stops,
                                                user=st["id"]):
                if hit:
                    reasons.append("overlong")
                    continue
                cut = min((i for i in (text.find(x) for x in stops) if i >= 0), default=len(text))
                value, why = check_step(step, text[:cut], st)  # the server returns text through its stop
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
    header = header_of(st)
    c = {"secret_arm": "hinted",
         "characters": [{"name": x["text"], "label": x["label"], "character": {"text": p}, "need": {"text": ""}}
                        for x, p in zip(st["names"], st["people"])]}
    why = sp.gate_page({"wants": [], "secret": "", "page": header if st["arm"] == "bare" else
                        header.split("\n\n", 1)[1]}, c, wants_required=0)
    if not why and copies_exemplar(header, exemplars):
        why = "copy"  # a run of 6+ words lifted from one of the listings this seed was shown
    if why:
        st.update(status="fail", failed_step="page", rejections={**st["rejections"], "page": [why]})
        return st
    st["header"] = header
    st["genre"] = genre_tag(st["listing"])
    st["guards"] = repetition_guards(header)
    if score is not None:
        try:
            st["nll"] = -(await score([(sp.DOC_START, header)]))[0]
        except Exception as e:  # the NLL is a guard reading, never a reason to lose the seed
            logger.error("scoring %s failed: %s", st["id"], e)
    st.update(status="pass", failed_step=None)
    return st


COPY_RUN = 6
THEATRE_META = re.compile(r"\b(?:theat(?:re|er)s?|rehears\w*|stage[sd]?|playwright|actors?|actress(?:es)?|"
                          r"audiences?|curtain|scripts?|drama (?:club|school|class)|director|backstage|"
                          r"performances?|the play\b)", re.I)


def word_runs(text: str, n: int = COPY_RUN) -> set[tuple[str, ...]]:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def copies_exemplar(header: str, exemplars: Sequence[dict[str, Any]], n: int = COPY_RUN) -> bool:
    """The header shares a run of ``n`` or more words with one of the listings shown to it."""
    runs = word_runs(header, n)
    return any(runs & word_runs(render_listing(1, e), n) for e in exemplars)


def draw_boot_starts(pools: dict[str, list[dict[str, Any]]], lists: dict[str, list[str]], frames: list[dict],
                     n: int, seed: int, pass_no: int, cast_share: float = 0.3,
                     lineage: str = "") -> list[dict[str, Any]]:
    """Roots and arms for ``n`` seeds: theme least-used from the pool, relationship = (antisyzygy, life moment)
    atoms, names; cast arm by quota, note/bare arm alternating; the frame drawn at random per seed from the whole
    pool (rotation: no frame's imprint is shared by the corpus), its title recorded for grouping. Frames tagged with
    a ``pool`` are drawn pool first (uniformly), then frame within the pool."""
    rng = random.Random(f"{seed}:{pass_no}")
    frame_rng = random.Random(f"{seed}:{pass_no}:frame")  # its own stream: the frame pool never shifts the roots
    uses: collections.Counter = collections.Counter()
    sched = sp.arm_schedule(n, cast_share, rng)
    by_pool: dict[str, list[dict]] = collections.defaultdict(list)
    for f in frames:
        by_pool[f.get("pool", "")].append(f)
    pool_names = sorted(by_pool)
    out = []
    for i, (_f, _s, cast_arm) in enumerate(sched):
        k = 2 if cast_arm == "pair" else rng.choice((3, 4))
        theme = sp._pick(rng, pools["theme"], uses)[0]
        names = sp._pick_names(rng, pools["names"], uses, k)
        for e in (theme, *names):
            uses[e["id"]] += 1
        out.append({"id": f"p{pass_no}-{i:05d}", "pass": pass_no, "cast_arm": cast_arm,
                    "arm": ("note", "bare")[i % 2],
                    **_frame_fields(frame_rng.choice(by_pool[frame_rng.choice(pool_names)])), "lineage": lineage,
                    "theme": theme["text"], "pair": rng.choice(lists["antisyzygies"]),
                    "moment": rng.choice(lists["moments"]),
                    "roots": {"theme": theme["id"], "names": [x["id"] for x in names]},
                    "names": [{"text": x["text"], "label": sp.name_label(x["text"])} for x in names]})
    return out


def _frame_fields(f: dict[str, Any]) -> dict[str, Any]:
    return {"frame": f["frame"], "frame_title": f.get("title", ""), "frame_control": bool(f.get("control")),
            "frame_pool": f.get("pool", "")}


def exemplar_pool(kept: list[dict[str, Any]], select: str, size: int, seed: int,
                  embedder: str = "tfidf", rank: int = 32) -> list[dict[str, Any]]:
    """The listings later passes may show (each seed then draws its own 2-4 from them): all kept (``random``), or
    ``kmedoids``: a health gate (drop the worst NLL decile and any 8-gram loop), then ``size`` medoids in a whitened
    rank-``rank`` projection of the text embedding. Selection never reads signatures, so signatures stay free to
    audit it."""
    if select == "random" or len(kept) <= size:
        return list(kept)
    import numpy as np

    nll = sorted(r["nll"] for r in kept if r.get("nll") is not None)
    cut = nll[int(0.9 * (len(nll) - 1))] if nll else None
    healthy = [r for r in kept if not r.get("guards", {}).get("loop8")
               and (cut is None or r.get("nll") is None or r["nll"] <= cut)]
    if len(healthy) <= size:
        return healthy
    x = embed([render_listing(1, r) for r in healthy], embedder, [r["id"] for r in healthy])
    x = x - x.mean(0)
    u, sv, _ = np.linalg.svd(x, full_matrices=False)
    z = u[:, :min(rank, u.shape[1])]  # whitened: unit variance along each kept direction
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    rng = random.Random(seed)
    medoids = rng.sample(range(len(healthy)), size)
    for _ in range(20):  # alternate assignment and medoid update (PAM-lite)
        assign = np.argmin(d[:, medoids], axis=1)
        new = []
        for c in range(size):
            members = np.where(assign == c)[0]
            new.append(int(members[np.argmin(d[np.ix_(members, members)].sum(1))]) if len(members) else medoids[c])
        if new == medoids:
            break
        medoids = new
    return [healthy[j] for j in medoids]


async def run_boot(complete: sp.Complete, score: Optional[sp.Score], starts: list[dict[str, Any]],
                   frames: dict[str, str], exemplars: list[dict[str, Any]], out: Path, shots: tuple[int, int],
                   seed: int, concurrency: int = 128) -> dict[str, Any]:
    path = out / "chains.jsonl"
    done = {r["id"] for r in sp.read_jsonl(path)}
    todo = [s for s in starts if s["id"] not in done]
    logger.info("boot: %d starts, %d done, %d to grow, %d exemplars", len(starts), len(done), len(todo),
                len(exemplars))
    sem = asyncio.Semaphore(concurrency)
    f = path.open("a", encoding="utf-8")
    t0 = time.time()
    count = {"n": 0}

    async def one(s: dict[str, Any]) -> None:
        async with sem:
            rng = random.Random(f"{seed}:{s['id']}:shots")
            ex = rng.sample(exemplars, min(rng.randint(*shots), len(exemplars))) if exemplars else []
            st = dict(s, exemplars=[e["id"] for e in ex])
            try:
                st = await grow_boot(complete, score, st, frames[s["frame"]], ex)
            except Exception as e:  # one seed's failure must not stop the pass
                logger.error("seed %s failed: %s", s["id"], e)
                return
            f.write(json.dumps(st, ensure_ascii=False) + "\n")
            f.flush()
            count["n"] += 1
            if count["n"] % 50 == 0:
                logger.info("boot %d/%d (%.0fs)", count["n"], len(todo), time.time() - t0)

    try:
        await asyncio.gather(*(one(s) for s in todo))
    finally:
        f.close()
    rows = sp.read_jsonl(path)
    seeds = [boot_seed(r) for r in rows if r["status"] == "pass"]
    sp.write_jsonl(out / "seeds.all.jsonl", seeds)
    return {"chains": len(rows), "pass": len(seeds),
            "failed_step": dict(collections.Counter(r["failed_step"] for r in rows if r["status"] != "pass"))}


def boot_seed(st: dict[str, Any]) -> dict[str, Any]:
    """A loom seed (``kotodama.posttrain.selfplay.parse_seed`` reads it) with its provenance."""
    return {"id": st["id"], "cast": [x["label"] for x in st["names"]], "header": st["header"], "form": "boot",
            "pass": st["pass"], "lineage": st.get("lineage", ""), "frame": st["frame"],
            "frame_pool": st.get("frame_pool", ""),
            "frame_title": st.get("frame_title", ""),
            "frame_control": st.get("frame_control", False), "arm": st["arm"], "cast_arm": st["cast_arm"],
            "exemplars": st.get("exemplars", []), "genre": st["genre"], "theme": st["theme"], "pair": st["pair"],
            "moment": st["moment"], "listing": st["listing"], "scene": st["scene"],
            "characters": [{"name": x["text"], "label": x["label"], "character": p}
                           for x, p in zip(st["names"], st["people"])],
            "names": st["names"], "people": st["people"], "roots": st["roots"], "nll": st.get("nll"),
            "guards": st.get("guards", {})}


# ── readouts ─────────────────────────────────────────────────────────────────


async def frame_fan(complete: sp.Complete, frames: list[dict[str, Any]], n: int, out: Path,
                    exemplars: Sequence[dict[str, Any]] = (), shots: tuple[int, int] = (2, 4), seed: int = 0,
                    name: str = "fan.jsonl") -> None:
    """``n`` draws at each frame's boundary (where the first listing would start), or, with ``exemplars``, after the
    frame and a fresh 2-4 of them per draw (where the next listing would start); writes ``<out>/<name>``
    ({id, frame, text, exemplars})."""
    rng = random.Random(seed)
    jobs = []
    for fr in frames:
        if not exemplars:
            jobs.append((fr, [], f"{fr['text']}\n\n", n))
            continue
        for k in range(n):
            ex = rng.sample(list(exemplars), min(rng.randint(*shots), len(exemplars)))
            body = "".join(render_listing(j + 1, e) + "\n\n" for j, e in enumerate(ex))
            jobs.append((fr, ex, f"{fr['text']}\n\n{body}{len(ex) + 1}.", 1))
    outs = await asyncio.gather(*(complete(p, m, ["\n\n"], user=f"fan:{fr['frame']}:{i}")
                                  for i, (fr, _e, p, m) in enumerate(jobs)))
    rows, counts = [], collections.Counter()
    for (fr, ex, _p, _m), raw in zip(jobs, outs):
        for t, _f, _h in raw:
            k = counts[fr["frame"]]
            counts[fr["frame"]] += 1
            rows.append({"id": f"fan-{fr['frame']}-{k}", "frame": fr["frame"], "text": t.strip(),
                         "exemplars": [e["id"] for e in ex]})
    sp.write_jsonl(out / name, rows)


def _tfidf_dense(texts: Sequence[str]) -> Any:
    import numpy as np

    x = sp.tfidf(texts)
    return np.asarray(x.todense(), dtype=np.float32)


def _minilm(texts: Sequence[str]) -> Any:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = "sentence-transformers/all-MiniLM-L6-v2"
    tok, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
    vecs = []
    with torch.no_grad():
        for k in range(0, len(texts), 64):
            enc = tok(list(texts[k:k + 64]), padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            vecs.append(torch.nn.functional.normalize(v, dim=-1).numpy())
    return np.concatenate(vecs).astype(np.float32)


def embed(texts: Sequence[str], embedder: str, ids: Sequence[str]) -> Any:
    """Unit-norm row vectors. ``tfidf``: character n-gram TF-IDF; ``minilm``: mean-pooled
    sentence-transformers/all-MiniLM-L6-v2; ``npz:PATH``: vectors computed elsewhere (arrays ``ids``, ``vectors``),
    looked up by seed id, so any representation (activation signatures included) plugs in."""
    import numpy as np

    if embedder == "tfidf":
        x = _tfidf_dense(texts)
    elif embedder == "minilm":
        x = _minilm(texts)
    elif embedder.startswith("npz:"):
        z = np.load(embedder[4:], allow_pickle=False)
        index = {str(i): j for j, i in enumerate(z["ids"])}
        missing = [i for i in ids if i not in index]
        if missing:
            raise KeyError(f"{len(missing)} ids have no vector in {embedder[4:]}, e.g. {missing[:3]}")
        x = np.asarray(z["vectors"], dtype=np.float32)[[index[i] for i in ids]]
    else:
        raise ValueError(f"unknown embedder {embedder!r} (tfidf | minilm | npz:PATH)")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _dist(x: Any) -> Any:
    import numpy as np

    g = x @ x.T
    return np.sqrt(np.maximum(0.0, 2 - 2 * g))


def energy_distance(d: Any, a: Any, b: Any) -> float:
    """2 E|X-Y| - E|X-X'| - E|Y-Y'| from a precomputed distance matrix and two index sets."""
    import numpy as np

    return float(2 * d[np.ix_(a, b)].mean() - d[np.ix_(a, a)].mean() - d[np.ix_(b, b)].mean())


def mmd_rbf(d: Any, a: Any, b: Any) -> float:
    """Biased MMD^2 with an RBF kernel at the median-distance bandwidth."""
    import numpy as np

    sig = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
    k = np.exp(-(d ** 2) / (2 * sig ** 2))
    return float(k[np.ix_(a, a)].mean() + k[np.ix_(b, b)].mean() - 2 * k[np.ix_(a, b)].mean())


METRICS: dict[str, Callable[[Any, Any, Any], float]] = {"energy": energy_distance, "mmd": mmd_rbf}


def contrast(d: Any, a: list[int], b: list[int], metric: str, perms: int, rng: random.Random) -> dict[str, float]:
    """The metric between groups ``a`` and ``b`` and its document-level permutation null (labels shuffled across the
    pooled documents): {value, null_mean, null_q95, p}."""
    import numpy as np

    f = METRICS[metric]
    val = f(d, np.array(a), np.array(b))
    pooled = a + b
    null = []
    for _ in range(perms):
        rng.shuffle(pooled)
        null.append(f(d, np.array(pooled[:len(a)]), np.array(pooled[len(a):])))
    null_arr = np.array(null)
    return {"value": round(val, 5), "null_mean": round(float(null_arr.mean()), 5),
            "null_q95": round(float(np.quantile(null_arr, 0.95)), 5),
            "p": round(float((1 + (null_arr >= val).sum()) / (1 + perms)), 4)}


def probe_auc(x: Any, a: list[int], b: list[int], groups: Sequence[str], lengths: Sequence[int],
              folds: int = 5) -> dict[str, Optional[float]]:
    """Held-out AUC of a logistic probe telling group ``a`` from ``b`` (GroupKFold by ``groups``, so one root never
    sits on both sides of a split), on features residualized on log length within each training fold; beside it the
    AUC of log length alone (the length rake). 0.5 = indistinguishable. None when there are too few groups."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    idx = np.array(a + b)
    y = np.array([0] * len(a) + [1] * len(b))
    g = np.array([groups[i] for i in idx])
    if len(set(g)) < folds or min(len(a), len(b)) < folds:
        return {"auc": None, "length_auc": None}
    xs = x[idx]
    ll = np.log1p(np.array([lengths[i] for i in idx], dtype=np.float64))[:, None]
    scores, lscores = np.zeros(len(idx)), np.zeros(len(idx))
    for tr, te in GroupKFold(n_splits=folds).split(xs, y, g):
        design = np.hstack([np.ones((len(tr), 1)), ll[tr]])
        coef, *_ = np.linalg.lstsq(design, xs[tr], rcond=None)
        res = lambda rows: xs[rows] - np.hstack([np.ones((len(rows), 1)), ll[rows]]) @ coef  # noqa: E731
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(res(tr), y[tr])
        scores[te] = clf.decision_function(res(te))
        lscores[te] = LogisticRegression().fit(ll[tr], y[tr]).decision_function(ll[te])
    return {"auc": round(float(roc_auc_score(y, scores)), 4), "length_auc": round(float(roc_auc_score(y, lscores)), 4)}


def dispersion(d: Any, idx: list[int]) -> float:
    import numpy as np

    if len(idx) < 2:
        return 0.0
    sub = d[np.ix_(idx, idx)]
    return float(sub.sum() / (len(idx) * (len(idx) - 1)))


def variety(rows: list[dict[str, Any]], embedder: str = "tfidf", metric: str = "energy", perms: int = 200,
            verdicts: Optional[dict[str, str]] = None, field: str = "header", seed: int = 0,
            keys: Sequence[str] = ("frame_pool", "frame_title", "frame", "pass", "arm")) -> dict[str, Any]:
    """Per-group (frame, pass, arm) dispersion and guards, and pairwise contrasts with their nulls."""
    import numpy as np

    rng = random.Random(seed)
    x = embed([r[field] for r in rows], embedder, [r["id"] for r in rows])
    d = _dist(x)
    roots = [str(r.get("roots", {}).get("theme", r["id"])) for r in rows]
    lengths = [len(r[field]) for r in rows]
    out: dict[str, Any] = {"embedder": embedder, "metric": metric, "field": field, "n": len(rows), "groups": {},
                           "contrasts": {}}
    for key in keys:
        groups: dict[str, list[int]] = collections.defaultdict(list)
        for i, r in enumerate(rows):
            groups[str(r.get(key, ""))].append(i)
        for g, idx in sorted(groups.items()):
            sub = [rows[i] for i in idx]
            nll = [r["nll"] for r in sub if r.get("nll") is not None]
            kept = [verdicts.get(r["id"]) for r in sub] if verdicts else []
            out["groups"][f"{key}={g}"] = {
                "n": len(idx), "dispersion": round(dispersion(d, idx), 4),
                "mean_cos": round(float(1 - (d[np.ix_(idx, idx)] ** 2 / 2).sum() / max(1, len(idx) * (len(idx) - 1))),
                                  4) if len(idx) > 1 else None,
                "nll_mean": round(float(np.mean(nll)), 4) if nll else None,
                "rep4_mean": round(float(np.mean([r["guards"].get("rep4", 0) for r in sub])), 4),
                "loop8": sum(bool(r["guards"].get("loop8")) for r in sub),
                "filter_keep": (round(sum(v == "keep" for v in kept) / max(1, sum(v is not None for v in kept)), 3)
                                if verdicts else None),
                "genres": dict(collections.Counter(r["genre"] for r in sub).most_common(6))}
        names = sorted(groups)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if len(groups[a]) >= 3 and len(groups[b]) >= 3:
                    c = contrast(d, list(groups[a]), list(groups[b]), metric, perms, rng)
                    c.update(probe_auc(x, list(groups[a]), list(groups[b]), roots, lengths))
                    out["contrasts"][f"{key}:{a}|{b}"] = c
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────


def _lists(path: str) -> dict[str, list[str]]:
    return {p.stem: sp.load_list(p) for p in sorted(Path(path).glob("*.txt"))}


def _client(a: argparse.Namespace, max_tokens: int) -> tuple[Any, Any]:
    complete, close, _ = sp._completer(a.endpoint, a.law, a.temperature, max_tokens, a.repetition_penalty)
    return complete, close


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--endpoint", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--law", default="tsumugi")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)


def main_frames(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="model-written frames (python -m kotodama.posttrain.seeds boot-frames)")
    _common(ap)
    ap.add_argument("--titles", default=",".join(FRAME_TITLES))
    ap.add_argument("--per_title", type=int, default=8)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    complete, close = _client(a, 240)
    from kotodama.posttrain.loom import http_scorer

    score, sclose = http_scorer(a.endpoint, max_completion_tokens=1024)

    async def go() -> list[dict[str, Any]]:
        try:
            return await write_frames(complete, out, a.titles.split(","), a.per_title, score)
        finally:
            await close()
            await sclose()

    rows = asyncio.run(go())
    print(json.dumps({"candidates": len(rows), "mechanical_ok": sum(r["mechanical_ok"] for r in rows)}))
    return 0


def main_frames_pick(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="keep coherent frames (python -m kotodama.posttrain.seeds "
                                             "boot-frames-pick)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verdicts", required=True)
    ap.add_argument("--k", type=int, default=8, help="frames kept per title")
    ap.add_argument("--control", type=int, default=0, help="reader-rejected borderline frames kept as a control arm")
    a = ap.parse_args(argv)
    frames = pick_frames(Path(a.out), Path(a.verdicts), a.k, a.control)
    print(json.dumps([{k: f[k] for k in ("frame", "source", "title", "control")} for f in frames], indent=1))
    return 0


def main_organon(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="organon pools (python -m kotodama.posttrain.seeds boot-organon)")
    _common(ap)
    ap.add_argument("--lists", required=True, help="directory of public organon lists, <name>.txt")
    ap.add_argument("--names_seed", required=True, help="JSONL of model-written names ({text}) the guest list opens "
                                                         "with")
    ap.add_argument("--per_pool", type=int, default=300)
    ap.add_argument("--famous_cut", type=float, default=0.15)
    a = ap.parse_args(argv)
    out = Path(a.out)
    lists = _lists(a.lists)
    seed_names = [r["text"] for r in sp.read_jsonl(Path(a.names_seed))]
    complete, close = _client(a, 160)
    from kotodama.posttrain.loom import http_scorer

    score, sclose = http_scorer(a.endpoint)

    async def go() -> dict[str, Any]:
        stats = {}
        try:
            for kind in ("theme", "names"):
                path = out / "pools" / f"{kind}.jsonl"
                if path.exists():
                    continue
                target = a.per_pool if kind == "theme" else math.ceil(a.per_pool / (1 - a.famous_cut))
                res = await grow_organon(complete, kind, lists, target, random.Random(f"{a.seed}:{kind}"),
                                         seed_names)
                if kind == "names":
                    res = await sp.cut_famous(res, sp.AXES["names"], score, a.famous_cut, a.per_pool)
                rows = [{"id": f"{kind}:{i:04d}", "axis": kind, "text": r["text"], "crossing": r["crossing"],
                         "source": r.get("source")} for i, r in enumerate(res["kept"])]
                sp.write_jsonl(path, rows)
                sp.write_jsonl(out / "pools" / f"{kind}.rejects.jsonl", res["rejects"])
                stats[kind] = {**res["counts"],
                               "reasons": dict(collections.Counter(r["reason"] for r in res["rejects"]))}
            return stats
        finally:
            await close()
            await sclose()

    stats = asyncio.run(go())
    (out / "pools" / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


def main_boot(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="one bootstrap pass (python -m kotodama.posttrain.seeds boot)")
    _common(ap)
    ap.add_argument("--lists", required=True)
    ap.add_argument("--pass", dest="pass_no", type=int, required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--shots", default="2,4", help="few-shot listings per seed on passes >= 1: min,max")
    ap.add_argument("--verdicts", action="append", default=[],
                    help="filter verdicts of earlier passes ({id, verdict}); their kept seeds are the exemplars")
    ap.add_argument("--exemplar_select", choices=["random", "kmedoids"], default="random")
    ap.add_argument("--exemplar_pool", type=int, default=64)
    ap.add_argument("--exemplar_file", default=None,
                    help="JSONL of kept seed rows (boot or seedchain native) to draw few-shot from on THIS pass, "
                         "pass 0 included; each seed draws a fresh 2-4 from the whole file")
    ap.add_argument("--lineage", default="", help="generation-lineage tag written on every row")
    ap.add_argument("--from_passes", default=None,
                    help="comma list of earlier passes whose kept seeds are the exemplars (default: the previous one)")
    ap.add_argument("--concurrency", type=int, default=128)
    a = ap.parse_args(argv)
    out = Path(a.out)
    pdir = out / f"pass{a.pass_no}"
    pdir.mkdir(parents=True, exist_ok=True)
    lists = _lists(a.lists)
    pools = {k: sp.read_jsonl(out / "pools" / f"{k}.jsonl") for k in ("theme", "names")}
    frames = sp.read_jsonl(out / "frames.jsonl")
    if not frames or not all(pools.values()):
        raise SystemExit(f"frames or pools missing under {out} (boot-frames-pick, boot-organon first)")
    starts = draw_boot_starts(pools, lists, frames, a.n, a.seed, a.pass_no, lineage=a.lineage)
    exemplars: list[dict[str, Any]] = []
    if a.exemplar_file:
        exemplars = [as_exemplar(r) for r in sp.read_jsonl(Path(a.exemplar_file))]
    elif a.pass_no > 0:
        keep = {v["id"] for vf in a.verdicts for v in sp.read_jsonl(Path(vf)) if v.get("verdict") == "keep"}
        src = [int(x) for x in a.from_passes.split(",")] if a.from_passes else [a.pass_no - 1]
        earlier = [r for p in src for r in sp.read_jsonl(out / f"pass{p}" / "seeds.all.jsonl")]
        kept = [r for r in earlier if r["id"] in keep]
        if not kept:
            raise SystemExit("no kept seeds from earlier passes to draw exemplars from (pass --verdicts)")
        exemplars = exemplar_pool(kept, a.exemplar_select, a.exemplar_pool, a.seed)
    lo, hi = (int(x) for x in a.shots.split(","))
    complete, close = _client(a, 240)
    from kotodama.posttrain.loom import http_scorer

    score, sclose = http_scorer(a.endpoint, max_completion_tokens=1024)

    async def go() -> dict[str, Any]:
        try:
            return await run_boot(complete, score, starts, {f["frame"]: f["text"] for f in frames}, exemplars,
                                  pdir, (lo, hi), a.seed, a.concurrency)
        finally:
            await close()
            await sclose()

    stats = asyncio.run(go())
    (pdir / "run.json").write_text(json.dumps({"args": vars(a), "exemplars": [e["id"] for e in exemplars],
                                               **stats}, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


def main_fan(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="frame-boundary fan (python -m kotodama.posttrain.seeds boot-fan)")
    _common(ap)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--frames", default=None, help="frames JSONL ({frame, text}); default <out>/frames.jsonl")
    ap.add_argument("--exemplar_file", default=None, help="draw after the frame and a fresh 2-4 of these listings")
    ap.add_argument("--name", default="fan.jsonl")
    a = ap.parse_args(argv)
    complete, close = _client(a, 120)
    frames = sp.read_jsonl(Path(a.frames) if a.frames else Path(a.out) / "frames.jsonl")
    ex = [as_exemplar(r) for r in sp.read_jsonl(Path(a.exemplar_file))] if a.exemplar_file else []

    async def go() -> None:
        try:
            await frame_fan(complete, frames, a.n, Path(a.out), ex, seed=a.seed, name=a.name)
        finally:
            await close()

    asyncio.run(go())
    return 0


def main_variety(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="variety between frames/passes/arms (python -m kotodama.posttrain.seeds "
                                             "boot-variety)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", default="0")
    ap.add_argument("--embedder", default="tfidf")
    ap.add_argument("--metric", choices=sorted(METRICS), default="energy")
    ap.add_argument("--field", default="header")
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--verdicts", action="append", default=[])
    ap.add_argument("--keys", default="frame_pool,frame_title,frame,pass,arm", help="row fields to group by")
    ap.add_argument("--kept_only", action="store_true", help="only rows the verdicts keep")
    a = ap.parse_args(argv)
    out = Path(a.out)
    rows = [r for p in a.passes.split(",") for r in sp.read_jsonl(out / f"pass{p}" / "seeds.all.jsonl")]
    verdicts = {v["id"]: v["verdict"] for vf in a.verdicts for v in sp.read_jsonl(Path(vf))} or None
    if a.kept_only and verdicts:
        rows = [r for r in rows if verdicts.get(r["id"]) == "keep"]
    rep = variety(rows, a.embedder, a.metric, a.perms, verdicts, a.field, keys=a.keys.split(","))
    name = (f"variety.{a.embedder.replace(':', '_').replace('/', '_')}.{a.metric}.{a.field}"
            f"{'.kept' if a.kept_only else ''}.json")
    (out / name).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


def drift(out: Path, passes: Sequence[int], verdicts: dict[str, str],
          exemplar_files: Optional[dict[int, Path]] = None) -> dict[str, Any]:
    """Per pass (and per frame pool within it): chains, gate passes, filter keeps, copy rejections, median header
    length, the share of passing listings that state a genre, the share of passing headers sharing a run of
    ``COPY_RUN`` words with a shown exemplar, and the theatre-about-theatre share (``THEATRE_META`` in the header)."""
    import statistics

    rep: dict[str, Any] = {}
    for p in passes:
        rows = sp.read_jsonl(out / f"pass{p}" / "chains.jsonl")
        src: dict[str, dict[str, Any]] = {}
        if exemplar_files and p in exemplar_files:
            src = {r["id"]: as_exemplar(r) for r in sp.read_jsonl(exemplar_files[p])}
        elif p > 0:
            src = {r["id"]: r for r in sp.read_jsonl(out / f"pass{p - 1}" / "seeds.all.jsonl")}
        for pool in ["*"] + sorted({r.get("frame_pool", "") for r in rows}):
            sub = [r for r in rows if pool == "*" or r.get("frame_pool", "") == pool]
            ok = [r for r in sub if r["status"] == "pass"]
            heads = [r["header"] for r in ok]
            overlap = [bool(word_runs(r["header"]) & set().union(*(word_runs(render_listing(1, src[e]))
                                                                   for e in r.get("exemplars", []) if e in src)))
                       for r in ok]
            judged = [verdicts[r["id"]] for r in ok if r["id"] in verdicts]
            rep[f"pass{p}" + ("" if pool == "*" else f"/{pool}")] = {
                "chains": len(sub), "gate_pass": len(ok),
                "copy_rejected": sum("copy" in r.get("rejections", {}).get("page", []) for r in sub),
                "filter_keep": sum(v == "keep" for v in judged), "filter_judged": len(judged),
                "keep_rate_end_to_end": round(sum(v == "keep" for v in judged) / len(sub), 3) if sub else None,
                "median_header_chars": statistics.median(len(h) for h in heads) if heads else None,
                "genre_stated_share": round(sum(r.get("genre", "none") != "none" for r in ok) / len(ok), 3)
                if ok else None,
                "six_gram_overlap_share": round(sum(overlap) / len(overlap), 3) if overlap else None,
                "theatre_meta_share": round(sum(bool(THEATRE_META.search(h)) for h in heads) / len(heads), 3)
                if heads else None}
    return rep


def boot_pick(out: Path, passes: Sequence[int], verdicts: dict[str, str], nll_cut: float = 0.05,
              cast_share: float = 0.3, seed: int = 0) -> dict[str, Any]:
    """Kept seeds of ``passes`` only, minus the worst ``nll_cut`` NLL tail, stratified by cast arm at
    ``cast_share`` casts (the largest pick both cells can fill); writes ``<out>/seeds.jsonl`` and
    ``<out>/pick.stats.json``."""
    rows = [r for p in passes for r in sp.read_jsonl(out / f"pass{p}" / "seeds.all.jsonl")
            if verdicts.get(r["id"]) == "keep"]
    nlls = sorted(r["nll"] for r in rows if r.get("nll") is not None)
    cut = nlls[max(0, math.ceil(len(nlls) * (1 - nll_cut)) - 1)] if nlls and nll_cut > 0 else None
    rows = [r for r in rows if cut is None or r.get("nll") is None or r["nll"] <= cut]
    cells = {a: [r for r in rows if r["cast_arm"] == a] for a in ("pair", "cast")}
    shares = {"pair": 1 - cast_share, "cast": cast_share}
    total = min(len(cells[a]) / shares[a] for a in cells) if all(cells.values()) else 0
    rng = random.Random(seed)
    picked = []
    for a, rs in cells.items():
        rs = rs[:]
        rng.shuffle(rs)
        picked += rs[:math.floor(total * shares[a])]
    sp.write_jsonl(out / "seeds.jsonl", picked)
    stats = {"passes": list(passes), "kept": len(rows), "nll_cut_value": cut,
             "available": {a: len(v) for a, v in cells.items()},
             "picked": dict(collections.Counter(r["cast_arm"] for r in picked)), "total": len(picked),
             "by_pass": dict(collections.Counter(r["pass"] for r in picked)),
             "by_frame_pool": dict(collections.Counter(r.get("frame_pool", "") for r in picked))}
    (out / "pick.stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def _verdict_map(files: Sequence[str]) -> dict[str, str]:
    """{id: verdict}, the last row per id winning (a rerun appends)."""
    return {v["id"]: v["verdict"] for f in files for v in sp.read_jsonl(Path(f))}


def main_drift(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="per-pass drift (python -m kotodama.posttrain.seeds boot-drift)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", default="0,1,2")
    ap.add_argument("--verdicts", action="append", default=[])
    ap.add_argument("--exemplar_file", action="append", default=[], help="PASS=FILE: that pass's exemplar source")
    a = ap.parse_args(argv)
    ex = {int(k): Path(v) for k, v in (x.split("=", 1) for x in a.exemplar_file)}
    rep = drift(Path(a.out), [int(x) for x in a.passes.split(",")], _verdict_map(a.verdicts), ex)
    (Path(a.out) / "drift.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


def main_pick(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="stratified pick over chosen passes (python -m kotodama.posttrain.seeds "
                                             "boot-pick)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", default="1,2")
    ap.add_argument("--verdicts", action="append", default=[])
    ap.add_argument("--nll_cut", type=float, default=0.05)
    a = ap.parse_args(argv)
    print(json.dumps(boot_pick(Path(a.out), [int(x) for x in a.passes.split(",")], _verdict_map(a.verdicts),
                               a.nll_cut), indent=1))
    return 0


COMMANDS = {"boot-frames": main_frames, "boot-frames-pick": main_frames_pick, "boot-organon": main_organon,
            "boot": main_boot, "boot-fan": main_fan, "boot-variety": main_variety, "boot-drift": main_drift,
            "boot-pick": main_pick}
