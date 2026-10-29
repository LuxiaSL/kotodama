"""Autoloom: grow a tree of transcripts from each seed instead of one line.

Self-play picks one surviving candidate per turn. The loom keeps them: at every live node it draws K candidates for
the next speaker (one batched request), runs the same mechanical vetoes as self-play, and up to ``--children`` of the
survivors become child nodes that keep growing independently. A branch ends when every candidate is vetoed
(``degraded``), when the model ends the document after ``min_turns`` (``eos``), or at ``--max_depth``. When a depth
holds more than ``--width`` live nodes, a uniform sample of ``width`` keeps growing and the rest stop (``pruned`` —
good branches that simply were not continued). One tree therefore gives breadth and depth within one conversation,
and every node's sibling set (the K candidates of one draw) for preference pairs or branch-relative steering.

    python -m kotodama.posttrain.loom --seeds seeds.jsonl --endpoint http://host:port --out loom/run \\
        [--candidates 8 --children 3 --width 16 --max_depth 24] [--steer vrep:-0.30 --steer_rule N] \\
        [--select engage --select_lambda 1.0] [--overlong keep --max_tokens 768]

Choosing children: ``uniform`` (default) samples survivors evenly. ``engage`` weights each survivor by
exp(λ·z), z = its engagement standardised among the node's survivors, where engagement = mean over a few sampled
replies of log p(reply | context + this turn) − log p(reply | context) for the next speaker — how much the turn hands
the other speaker something to answer. λ > 0 leans toward engaging turns (questions, challenges), λ < 0 toward
self-contained ones (narration, scene-setting, monologue); nothing is vetoed by it. ``novel`` weights each survivor
by exp(λ·z) on its share of content words the last 30 turns did not use; ``--novelty_norm length`` first shrinks each
share toward the siblings' pooled rate (``--novelty_prior`` content words) and removes the part a fit on
log(1 + words) explains across the node's survivors, so the choice follows turn length in neither direction. ``--overlong keep`` lets a turn that reaches the token cap live (flagged) instead of vetoing it.
``--rerolls N`` draws K fresh candidates (up to N times) when all K were vetoed, instead of ending the branch at once;
``--deep_children C`` caps children at C from ``--deep_after`` on (branch in the opening, then one path per branch).

Long conversations: ``--window_chars`` keeps each prompt to the header + seed turns + the latest turns within a
character budget (early dialogue rotates out, so branches run past the context and drift apart), and
``--deep_after D --deep_width W`` narrows the tree after depth D (wide early for breadth, narrow later for depth).
``python -m kotodama.posttrain.loom assemble <runs> --out docs.jsonl`` cuts paths into training documents of up to
~4K tokens in which no turn trains more than ``--max_uses`` times, so shared trunks are not overweighted. Each
document trains a run of new turns and fills the rest of its budget with the conversation just before them
(context, no loss), so a long path becomes consecutive documents that step forward through it. Only finished trees
(those with a row in ``<out>/trees.jsonl``) contribute paths. ``assemble --anthology`` instead keeps every ended
branch as one whole scene, cleans it (loop, frame and closing cuts, tail cleaning, gates) and stacks scenes from
different seeds into documents (``kotodama.posttrain.anthology``).

Growing deeper: ``python -m kotodama.posttrain.loom extend-seeds <runs> --out seeds.jsonl`` turns each finished
tree's deepest leaves into seeds whose ``history`` is the conversation so far; a loom run over them continues those
conversations (the window keeps the original seed turns and rotates the history out like any generated turn), and
``--max_depth`` counts only the new turns.

Runs resume: rerunning the same command skips seeds whose tree is already in ``<out>/trees.jsonl`` (and in any
``--skip_done`` run directory) and first drops the records of trees that never finished. A tree's id is its seed's
index in the seeds file, so a regrown tree keeps its id. With several endpoints each tree is pinned to one server —
the least loaded when it starts — so its prompts, which share long prefixes, reuse that server's prefix cache; a
failed request moves on to the next server. The exit code is 1 when any tree failed.

Writes, under the run directory ``<out>``:
  ``<out>/nodes.jsonl``   every candidate drawn: tree, id, parent, depth, speaker, text, ended_by, vetoes, status
                (child | vetoed | unchosen — a survivor beyond --children). Siblings = the rows sharing
                (tree, parent). Rendering is deterministic, so a node's full token ids are re-derivable from its path.
  ``<out>/states.jsonl``  how each branch ended (eos | degraded — no survivor | pruned | depth | length_cap); a node with
                no row here is interior
  ``<out>/paths.jsonl``   one transcript row per ended branch (root to that leaf), with the leaf's state
  ``<out>/trees.jsonl``   per-tree stats; ``<out>/run.json`` the configuration (law, steering, seed)
"""

from __future__ import annotations

import argparse
import functools
import asyncio
import json
import logging
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from kotodama.posttrain import compliance
from kotodama.posttrain.selfplay import (Complete, PromptTooLong, SeedSpec, SelfPlayConfig, evaluate, model_speaker,
                                         next_speaker,
                                         novelty_counts, oai_completer, parse_seed, parse_steer, speech_key,
                                         split_completion, spoken_text)
from kotodama.posttrain.transcript import Transcript, Turn, render_prompt, stop_sequences, to_row
from kotodama.serve.laws import law

logger = logging.getLogger(__name__)


@dataclass
class LoomConfig:
    candidates: int = 8
    children: int = 3
    width: int = 16
    max_depth: int = 24
    min_turns: int = 6
    max_tokens: int = 256
    max_prompt_chars: int = 12000
    window_chars: int = 0  # > 0: the model sees the header + seed turns + only the latest turns that fit this budget
    deep_after: int = 0  # from this depth on, at most deep_width live nodes (wide early, narrow and deep later)
    deep_width: int = 0
    select: str = "uniform"  # uniform | engage | novel
    select_lambda: float = 0.0
    novelty_speech_only: bool = False  # novel: narration paragraphs earn no novelty
    # novel: "length" scores each survivor by its shrunk novelty share (shrunk_shares) minus a least-squares fit on
    # log(1 + its words) across the node's survivors, so the choice leans neither to long turns (whose share runs
    # higher) nor to short ones (whose share is noisy, and noise wins under exp weighting); "none" = the raw share
    novelty_norm: str = "none"
    novelty_prior: float = 4.0  # novelty_norm length: pseudo-count of content words in the shrunk share
    deep_children: int = 0  # > 0: nodes at depth >= deep_after keep at most this many children (branch early only)
    rerolls: int = 0  # a node whose K candidates are all vetoed draws K more, up to this many times, before degrading
    engage_replies: int = 2
    overlong: str = "veto"  # veto | keep
    speaker: str = "rotate"  # rotate (round-robin over the cast) | model (the model names the next speaker)
    unsigned_utility: bool = False  # utility draws (speaker tags, engagement replies) send sign: false
    switch_model: str = ""  # non-empty: from turn switch_after on, every draw at a node asks for this model instead
    switch_after: int = 0


@dataclass
class Node:
    id: str
    parent: Optional[str]
    depth: int  # generated turns on the path, this node's included
    turns: tuple[Turn, ...]
    state: str = "open"


Score = Callable[[list[tuple[str, str]]], Awaitable[list[float]]]
"""[(prompt, completion)] -> mean per-token log-prob of each completion given its prompt."""


_WORDS = re.compile(r"[a-z']+")


def echo_question(text: str, prev: str, max_words: int = 4) -> bool:
    """A short question built only from the previous turn's words ("A job?" after "...if you'd like a job") — it makes
    the reply predictable, so engagement over-scores it."""
    t = text.strip()
    words = _WORDS.findall(t.lower())
    return t.endswith("?") and 0 < len(words) <= max_words and set(words) <= set(_WORDS.findall(prev.lower()))


def weighted_choice(rng: random.Random, weights: list[float], k: int) -> list[int]:
    """k distinct indices, drawn without replacement with probability proportional to weight (Efraimidis–Spirakis)."""
    keys = sorted(((rng.random() ** (1.0 / max(w, 1e-12)), i) for i, w in enumerate(weights)), reverse=True)
    return [i for _, i in keys[:k]]


async def engagement(turns: tuple[Turn, ...], header: str, cast: list[str], cand: Turn, complete: Complete,
                     score: Score, replies: int, fields: Optional[dict[str, Any]] = None) -> float:
    """Mean over sampled next-speaker replies of log p(reply | context + cand) − log p(reply | context)."""
    after = turns + (cand,)
    nxt = next_speaker(cast, list(after))
    stops = stop_sequences(cast)
    with_c = render_prompt(Transcript(after, header), nxt)
    without = render_prompt(Transcript(turns, header), nxt)
    raw = await complete(with_c, replies, stops, **(fields or {}))
    texts = [split_completion(t, stops, fr, hit)[0].strip() for t, fr, hit in raw]
    texts = [t for t in texts if t] or ["..."]
    lps = await score([(p, " " + t) for t in texts for p in (with_c, without)])
    return sum(lps[2 * i] - lps[2 * i + 1] for i in range(len(texts))) / len(texts)


def shrunk_shares(counts: list[tuple[int, int]], prior: float) -> list[float]:
    """Each (new, total) content-word count as (new + prior·p) / (total + prior), p = the pooled share Σnew / Σtotal:
    a turn with few content words is pulled toward the siblings' rate instead of scoring 0 or 1 by chance."""
    tot = sum(t for _, t in counts)
    p = sum(n for n, _ in counts) / tot if tot else 0.0
    return [(n + prior * p) / (t + prior) if t + prior > 0 else p for n, t in counts]


def length_residuals(values: list[float], lengths: list[int]) -> list[float]:
    """``values`` minus their least-squares fit on log(1 + length): what is left once a linear dependence on (log)
    length among these siblings is removed. With one distinct length the fit is the mean."""
    xs = [math.log1p(n) for n in lengths]
    mx, my = sum(xs) / len(xs), sum(values) / len(values)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, values)) / sxx if sxx > 1e-12 else 0.0
    return [y - my - b * (x - mx) for x, y in zip(xs, values)]


def window_turns(turns: tuple[Turn, ...], header: str, n_seed: int, window_chars: int) -> tuple[Turn, ...]:
    """What the model sees of ``turns``: everything (``window_chars`` 0), or the first ``n_seed`` (seed) turns + the
    latest turns that fit ``window_chars`` characters together with the header and the seed turns."""
    if not window_chars:
        return turns
    budget = window_chars - len(header) - sum(len(t.speaker) + len(t.text) + 4 for t in turns[:n_seed])
    k = len(turns)
    while k > n_seed and budget - (len(turns[k - 1].speaker) + len(turns[k - 1].text) + 4) >= 0:
        k -= 1
        budget -= len(turns[k].speaker) + len(turns[k].text) + 4
    return turns[:n_seed] + turns[k:]


async def grow(seed: SeedSpec, tree: str, complete: Complete, cfg: LoomConfig, sp: SelfPlayConfig,
               salt: str, emit: Callable[[str, dict[str, Any]], None], score: Optional[Score] = None) -> dict[str, Any]:
    """Grow one tree breadth-first; every candidate goes to ``emit("node", ...)``, every ended branch to
    ``emit("path", ...)``. Choices are seeded per node (``salt`` + node id), so a tree does not depend on the order
    concurrent requests return in. Returns the tree's stats."""
    stops = stop_sequences(seed.cast)
    root = Node(f"{tree}/r", None, 0, tuple(seed.turns) + tuple(seed.history))
    frontier = [root]
    stats = {"tree": tree, "seed": seed.id, "requests": 0, "candidates": 0, "children": 0,
             "ended": {"eos": 0, "degraded": 0, "pruned": 0, "depth": 0, "length_cap": 0}, "max_depth": 0,
             "rerolls": 0}

    def end(node: Node, state: str) -> None:
        node.state = state
        stats["ended"][state] += 1
        stats["max_depth"] = max(stats["max_depth"], node.depth)
        emit("state", {"tree": tree, "id": node.id, "state": state})
        if node.parent is not None:
            emit("path", {**to_row(Transcript(node.turns, seed.header)), "tree": tree, "seed": seed.id,
                          "leaf": node.id, "state": state, "depth": node.depth, "cast": seed.cast,
                          "n_seed": len(seed.turns)})

    n_seed = len(seed.turns)

    def visible(turns: tuple[Turn, ...]) -> tuple[Turn, ...]:
        return window_turns(turns, seed.header, n_seed, cfg.window_chars)

    def fields_at(node: Node, base: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Per-request fields for draws at ``node``: the second model once the scene has switch_after turns."""
        out = dict(base or {})
        if cfg.switch_model and node.depth >= cfg.switch_after:
            out["model"] = cfg.switch_model
        return out

    async def expand(node: Node) -> list[Node]:
        utility = fields_at(node, {"sign": False} if cfg.unsigned_utility else None) or None
        turn_fields = fields_at(node)
        seen = visible(node.turns)
        if cfg.speaker == "model":
            speaker, chose = await model_speaker(seed.cast, list(seen), seed.header, complete, fields=utility)
            stats["requests"] += 1
            stats["speaker_model" if chose else "speaker_rotate"] = stats.get(
                "speaker_model" if chose else "speaker_rotate", 0) + 1
        else:
            speaker = next_speaker(seed.cast, list(node.turns))
        prompt = render_prompt(Transcript(seen, seed.header), speaker)
        if len(prompt) > cfg.max_prompt_chars:
            end(node, "length_cap")
            return []
        cands = []
        for draw in range(cfg.rerolls + 1):  # a reroll draws K fresh candidates when every one so far was vetoed
            try:
                raw = await complete(prompt, cfg.candidates, stops, **turn_fields)
            except PromptTooLong:  # the character window can still exceed the context in tokens: end this branch only
                end(node, "length_cap")
                return []
            stats["requests"] += 1
            stats["rerolls"] += draw > 0
            new = [evaluate(*split_completion(t, stops, fr, hit)[:2], speaker, list(seen), seed.cast,
                            seed.header, sp) for t, fr, hit in raw]
            if cfg.overlong == "keep":
                for c in new:
                    if "overlong" in c.vetoes:
                        c.vetoes.remove("overlong")
            cands += new
            if any(c.ok for c in new):
                break
        n_children = (cfg.deep_children if cfg.deep_children and cfg.deep_after and node.depth >= cfg.deep_after
                      else cfg.children)
        survivors = [i for i, c in enumerate(cands) if c.ok]
        rng = random.Random(f"{salt}:{node.id}")
        eng: dict[int, float] = {}
        if cfg.select == "engage" and score is not None and len(survivors) > n_children:
            try:
                vals = await asyncio.gather(*(engagement(node.turns, seed.header, seed.cast,
                                                         Turn(speaker, cands[i].text), complete, score,
                                                         cfg.engage_replies, utility) for i in survivors))
                eng = dict(zip(survivors, vals))
            except Exception as e:  # a failed scoring round costs this node its weighting, not the tree
                logger.warning("node %s: engagement scoring failed (%s); choosing uniformly", node.id, e)
                stats["engage_failures"] = stats.get("engage_failures", 0) + 1
        nov: dict[int, float] = {}
        n_words: dict[int, int] = {}
        if cfg.select == "novel" and len(survivors) > n_children:
            counts = {i: novelty_counts(cands[i].text, list(seen), speech_only=cfg.novelty_speech_only,
                                        cast=seed.cast) for i in survivors}
            nov = {i: (new / tot if tot else 0.0) for i, (new, tot) in counts.items()}
            n_words = {i: len(speech_key(spoken_text(cands[i].text, seed.cast) if cfg.novelty_speech_only
                                         else cands[i].text).split()) for i in survivors}
            vals = [nov[i] for i in survivors]
            if cfg.novelty_norm == "length":
                vals = length_residuals(shrunk_shares([counts[i] for i in survivors], cfg.novelty_prior),
                                        [n_words[i] for i in survivors])
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
            sd = sd if sd > 1e-9 else math.inf  # equal scores (up to float residue from the fit): a uniform choice
            w = [math.exp(cfg.select_lambda * (v - mu) / sd) for v in vals]
            chosen = {survivors[j] for j in weighted_choice(rng, w, n_children)}
        elif eng:
            vals = [eng[i] for i in survivors]
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
            prev = node.turns[-1].text if node.turns else ""
            # an echo question is neutral (z = 0): neither end of the dial selects it for its echo
            z = {i: 0.0 if echo_question(cands[i].text, prev) else (eng[i] - mu) / sd for i in survivors}
            w = [math.exp(cfg.select_lambda * z[i]) for i in survivors]
            chosen = {survivors[j] for j in weighted_choice(rng, w, n_children)}
        else:
            chosen = set(rng.sample(survivors, min(n_children, len(survivors))))
        kids: list[Node] = []
        for i, c in enumerate(cands):
            stats["candidates"] += 1
            cid = f"{node.id}.{i}"
            status = "child" if i in chosen else ("vetoed" if c.vetoes else "unchosen")
            emit("node", {"tree": tree, "id": cid, "parent": node.id, "depth": node.depth + 1, "speaker": speaker,
                          **asdict(c), "status": status, **({"engage": eng[i]} if i in eng else {}),
                          **({"novelty": round(nov[i], 4), "words": n_words[i]} if i in nov else {})})
            if i in chosen:
                kid = Node(cid, node.id, node.depth + 1, node.turns + (Turn(speaker, c.text),))
                stats["children"] += 1
                if c.ended_by == "eos":
                    end(kid, "eos")
                elif kid.depth >= cfg.max_depth:
                    end(kid, "depth")
                else:
                    kids.append(kid)
        if not chosen:
            end(node, "degraded")
        return kids

    while frontier:
        nxt: list[Node] = []
        for kids in await asyncio.gather(*(expand(n) for n in frontier)):
            nxt.extend(kids)
        width = cfg.deep_width if cfg.deep_after and nxt and nxt[0].depth >= cfg.deep_after else cfg.width
        if len(nxt) > width:
            nxt.sort(key=lambda n: n.id)
            keep = set(random.Random(f"{salt}:{tree}:{nxt[0].depth}").sample(range(len(nxt)), width))
            for i, n in enumerate(nxt):
                if i not in keep:
                    end(n, "pruned")
            nxt = [n for i, n in enumerate(nxt) if i in keep]
        frontier = nxt
    return stats


def http_scorer(endpoints: list[str], timeout: float = 600.0, transport: Any = None,
                max_completion_tokens: int = 256,
                total: bool = False) -> tuple[Score, Callable[[], Awaitable[None]]]:
    """A ``Score`` over the servers' /score_tokens (round-robin), tokenizing locally; and its async closer. A prompt
    keeps its last 3500 tokens and a completion its first ``max_completion_tokens``. ``total``: the summed log-prob
    of each completion instead of its per-token mean."""
    import httpx
    from transformers import AutoTokenizer

    from kotodama import presets

    tok = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
    client = httpx.AsyncClient(timeout=timeout, transport=transport)
    counter = {"i": 0}

    async def score(pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for k in range(0, len(pairs), 32):
            items = [{"id": str(j), "prompt_ids": tok.encode(p, add_special_tokens=False)[-3500:],
                      "completion_ids": tok.encode(c, add_special_tokens=False)[:max_completion_tokens]}
                     for j, (p, c) in enumerate(pairs[k:k + 32])]
            for attempt in range(4):  # idle keep-alive sockets get closed server-side between engagement rounds
                url = endpoints[counter["i"] % len(endpoints)].rstrip("/") + "/score_tokens"
                counter["i"] += 1
                try:
                    r = await client.post(url, json={"items": items})
                    r.raise_for_status()
                    break
                except (httpx.HTTPError, httpx.TimeoutException) as e:
                    if attempt == 3:
                        raise
                    logger.warning("score request failed (%s); retrying", e)
                    await asyncio.sleep(2 ** attempt)
            out += [(-x["sum_nll"] if total else -x["mean_nll"]) if x["n_tokens"] else 0.0
                    for x in r.json()["results"]]
        return out

    return score, client.aclose


def finished(run_dir: Path) -> dict[str, str]:
    """{tree id: seed id} for the trees a run directory finished (rows of ``<out>/trees.jsonl``)."""
    f = run_dir / "trees.jsonl"
    if not f.exists():
        return {}
    rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["tree"]: r["seed"] for r in rows}


def prune_unfinished(run_dir: Path) -> int:
    """Drop node/state/path records of trees that never finished (an interrupted run); returns rows dropped."""
    keep = set(finished(run_dir))
    dropped = 0
    for name in ("nodes", "states", "paths"):
        f = run_dir / f"{name}.jsonl"
        if not f.exists():
            continue
        lines = [line for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        kept = [line for line in lines if json.loads(line)["tree"] in keep]
        if len(kept) != len(lines):
            dropped += len(lines) - len(kept)
            tmp = f.with_suffix(".jsonl.tmp")
            tmp.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
            tmp.replace(f)
    return dropped


async def run(seeds: list[tuple[int, SeedSpec]], complete: Complete, cfg: LoomConfig, sp: SelfPlayConfig,
              out_dir: Path, concurrency: int, seed: int, score: Optional[Score] = None,
              routes: int = 0, pin: str = "") -> dict[str, Any]:
    """Grow one tree per (index, seed) pair, named by the zero-padded index. ``routes`` > 0: ``complete`` takes ``route=`` and each tree
    is pinned to the least-loaded of that many endpoints. ``pin``: every request of tree i carries the OpenAI ``user``
    field ``<pin>/tree-<i>`` (``complete`` takes ``user=``), so a gateway keeps the tree on one replica."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {k: (out_dir / f"{k}s.jsonl").open("a", encoding="utf-8") for k in ("node", "path", "tree")}
    states = (out_dir / "states.jsonl").open("a", encoding="utf-8")
    sem = asyncio.Semaphore(concurrency)
    t0, done, failed = time.time(), {"n": 0}, {"n": 0}
    load = [0] * routes

    def emit(kind: str, rec: dict[str, Any]) -> None:
        (states if kind == "state" else files[kind]).write(json.dumps(rec, ensure_ascii=False) + "\n")

    async def one(i: int, s: SeedSpec) -> None:
        async with sem:
            route = min(range(routes), key=load.__getitem__) if routes else None
            tree_complete = complete if route is None else functools.partial(complete, route=route)
            if pin:
                tree_complete = functools.partial(tree_complete, user=f"{pin}/tree-{i:05d}")
            if route is not None:
                load[route] += 1
            try:
                st = await grow(s, f"t{i:05d}", tree_complete, cfg, sp, f"{seed}:{s.id}:{i}", emit, score)
            except Exception as e:  # one tree's failure must not stop the run
                failed["n"] += 1
                logger.error("tree %d (seed %s) failed: %s", i, s.id, e)
                return
            finally:
                if route is not None:
                    load[route] -= 1
            files["tree"].write(json.dumps(st) + "\n")
            for f in (*files.values(), states):
                f.flush()
            done["n"] += 1
            logger.info("tree %d/%d: %d candidates, %d children, max depth %d, ended %s (%.0fs)", done["n"],
                        len(seeds), st["candidates"], st["children"], st["max_depth"], st["ended"], time.time() - t0)

    try:
        await asyncio.gather(*(one(i, s) for i, s in seeds))
    finally:
        for f in (*files.values(), states):
            f.close()
    return {"trees": done["n"], "failed": failed["n"], "seconds": round(time.time() - t0, 1)}


def node_ids(leaf: str) -> list[str]:
    """The ids of the nodes on a leaf's path, root's child first ("t0/r.1.5.3" -> t0/r.1, t0/r.1.5, t0/r.1.5.3)."""
    head, _, rest = leaf.partition("/r.")
    parts = rest.split(".") if rest else []
    return [f"{head}/r." + ".".join(parts[: i + 1]) for i in range(len(parts))]


def _turn_chars(t: dict[str, Any]) -> int:
    return len(t["text"]) + len(t["speaker"]) + 4


@dataclass
class Clean:
    """Assembly-time cleanup (each part off by default). ``truncate_loops``: a path ends at its loop onset — the first
    turn from which at least half of the next 20 repeat (speech only) one of the 30 before them — so nothing inside a
    loop trains or serves as context. ``untrain_repeats`` N: a turn whose speech repeats one of the previous N turns is
    context, not a target. ``untrain_format``: a turn carrying another speaker's line under a non-cast name, or
    opening with a bracket tag ("[MFT]"), is context. ``drop_minor_explicit``: a document with a turn that
    ``kotodama.posttrain.compliance`` flags ``minor_explicit`` is dropped; ``blocklist``: so is one with a turn whose
    ``compliance.turn_hash`` is listed. ``redact_pii``: PII in the document's turns becomes placeholders."""

    truncate_loops: bool = False
    untrain_repeats: int = 0
    untrain_format: bool = False
    drop_minor_explicit: bool = False
    blocklist: frozenset[str] = frozenset()
    redact_pii: bool = False


_LABEL_LINE = re.compile(r"^\s*([A-Z][\w.'-]*(?: [A-Z][\w.'-]*){0,2}):\s", re.M)
_TAG_OPEN = re.compile(r"^\s*\[[A-Z]{2,6}\]")
def loop_onset(turns: list[dict[str, Any]], back: int = 30, span: int = 20) -> int:
    """Index of the first repeated turn (speech only, against the ``back`` turns before it) in the first window of
    ``span`` turns at least half of which are repeats; ``len(turns)`` when there is none."""
    rep_: list[bool] = []
    hist: list[str] = []
    for t in turns:
        k = speech_key(t["text"])
        rep_.append(bool(k) and k in hist)
        hist = (hist + [k])[-back:]
    for g in range(len(rep_)):
        win = rep_[g:g + span]
        if len(win) >= span // 2 and sum(win) * 2 >= len(win):
            return g + win.index(True)  # the window's first repeat, not the fresh turns leading into it
    return len(turns)


def _flawed(body: list[dict[str, Any]], cast: list[str], clean: Clean) -> set[int]:
    bad: set[int] = set()
    keys = [speech_key(t["text"]) for t in body]
    names = {c.strip() for c in cast}
    for k, t in enumerate(body):
        if clean.untrain_repeats and keys[k] and keys[k] in keys[max(0, k - clean.untrain_repeats):k]:
            bad.add(k)
        if clean.untrain_format and (_TAG_OPEN.match(t["text"]) or any(
                m.group(1) not in names for m in _LABEL_LINE.finditer(t["text"]))):
            bad.add(k)
    return bad


def assemble(paths: list[dict[str, Any]], max_chars: int, train_chars: int, max_uses: int, min_new: int,
             seed: int, clean: Optional[Clean] = None) -> list[dict[str, Any]]:
    """Training documents from loom paths without leaning on the shared trunk.

    A path is its header and seed turns (always in the document, never trained), then any ``history`` an earlier run
    generated (context only), then this run's turns. Paths are taken in random order. A document trains a run of
    turns, starting at the first one trained fewer than ``max_uses`` times so far and spanning up to ``train_chars``,
    and fills the rest of ``max_chars`` backwards with the turns just before it, marked ``"train": false``. The path
    then continues from the end of that run, so a long path becomes consecutive documents that each condition on the
    conversation behind them. Every turn TRAINS in at most ``max_uses`` documents however many leaves share it; turns
    already at the cap inside a run are context too. A document training fewer than ``min_new`` turns is dropped.
    ``clean`` (``Clean``) truncates loops and turns flawed turns into context before any of this."""
    clean = clean or Clean()
    rng = random.Random(seed)
    order = list(range(len(paths)))
    rng.shuffle(order)
    uses: dict[str, int] = {}
    docs: list[dict[str, Any]] = []
    for pi in order:
        p = paths[pi]
        n_seed = p.get("n_seed", len(p["turns"]) - p["depth"])
        prefix, body = p["turns"][:n_seed], p["turns"][n_seed:]
        n_hist = len(body) - p["depth"]
        ids = node_ids(p["leaf"])
        if n_hist < 0 or len(ids) != p["depth"]:
            continue
        if clean.truncate_loops:
            body = body[:loop_onset(body)]
            if len(body) <= n_hist:
                continue
        flawed = _flawed(body, p.get("cast", []), clean)
        budget = max_chars - len(p.get("header", "")) - sum(_turn_chars(t) for t in prefix)

        def trainable(k: int) -> bool:
            return k >= n_hist and k not in flawed and uses.get(ids[k - n_hist], 0) < max_uses

        i = n_hist
        while i < len(body):
            while i < len(body) and not trainable(i):
                i += 1
            if i >= len(body):
                break
            j, size = i, 0
            while j < len(body) and size + _turn_chars(body[j]) <= min(train_chars, budget):
                size += _turn_chars(body[j])
                j += 1
            if j <= i:  # a single turn larger than the budget
                i += 1
                continue
            start = i
            while start > 0 and size + _turn_chars(body[start - 1]) <= budget:
                start -= 1
                size += _turn_chars(body[start])
            trains = [k >= i and trainable(k) for k in range(start, j)]
            blocked = any(
                (clean.drop_minor_explicit and "minor_explicit" in compliance.classify(body[k]["text"]))
                or (clean.blocklist and compliance.turn_hash(body[k]["text"]) in clean.blocklist)
                for k in range(start, j))
            if sum(trains) >= min_new and not blocked:
                turns = [dict(t, train=False) for t in prefix]
                for k, tr in zip(range(start, j), trains):
                    turns.append(dict(body[k], train=tr))
                    if tr:
                        uses[ids[k - n_hist]] = uses.get(ids[k - n_hist], 0) + 1
                if clean.redact_pii:
                    turns = [dict(t, text=compliance.redact_pii(t["text"])) for t in turns]
                docs.append({"header": p.get("header", ""), "turns": turns,
                             "source": {"tree": p["tree"], "leaf": p["leaf"], "history": n_hist,
                                        "from": start - n_hist, "to": j - n_hist}})
            i = j
    return docs


def extend_seeds(runs: list[str], per_tree: int, min_depth: int, seed: int) -> list[dict[str, Any]]:
    """Seeds that continue each finished tree's deepest conversations: up to ``per_tree`` leaves that reached the
    tree's depth cap (and at least ``min_depth``), each as {cast, header, turns: the original seed turns, history: the
    generated turns so far}."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for r in runs:
        done = finished(Path(r))
        by_tree: dict[str, list[dict[str, Any]]] = {}
        for line in (Path(r) / "paths.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            if p["tree"] in done and p["state"] == "depth" and p["depth"] >= min_depth:
                by_tree.setdefault(p["tree"], []).append(p)
        for tree in sorted(by_tree):
            leaves = sorted(by_tree[tree], key=lambda p: p["leaf"])
            for p in rng.sample(leaves, min(per_tree, len(leaves))):
                n_seed = p.get("n_seed", len(p["turns"]) - p["depth"])
                rows.append({"id": f"{Path(r).name}:{p['leaf']}", "cast": p["cast"], "header": p.get("header", ""),
                             "turns": p["turns"][:n_seed], "history": p["turns"][n_seed:]})
    return rows


def extend_seeds_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="deepest leaves of finished trees -> seeds that continue them "
                                             "(python -m kotodama.posttrain.loom extend-seeds)")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_tree", type=int, default=1)
    ap.add_argument("--min_depth", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rows = extend_seeds(a.runs, a.per_tree, a.min_depth, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("%d seeds from %d runs", len(rows), len(a.runs))
    return 0


def assemble_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="loom paths -> training documents (python -m kotodama.posttrain.loom assemble)")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_chars", type=int, default=14000, help="~4K tokens of English text")
    ap.add_argument("--train_chars", type=int, default=0,
                    help="new turns trained per document (0 = half of --max_chars; the rest is preceding context)")
    ap.add_argument("--max_uses", type=int, default=2)
    ap.add_argument("--min_new", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--truncate_loops", action="store_true", help="end each path at its loop onset")
    ap.add_argument("--untrain_repeats", type=int, default=0,
                    help="a turn repeating one of the previous N turns (speech only) is context, not a target")
    ap.add_argument("--untrain_format", action="store_true",
                    help="turns with a non-cast speaker label inside, or a leading [TAG], are context")
    ap.add_argument("--drop_minor_explicit", action="store_true",
                    help="drop documents with a turn kotodama.posttrain.compliance flags minor_explicit")
    ap.add_argument("--blocklist", default="", help="drop documents with a turn whose hash is in this file "
                                                  "(python -m kotodama.posttrain.compliance scan)")
    ap.add_argument("--redact_pii", action="store_true", help="replace PII in documents with placeholders")
    g = ap.add_argument_group("whole scenes (kotodama.posttrain.anthology)")
    g.add_argument("--anthology", action="store_true",
                   help="whole cleaned scenes, stacked from different seeds into documents of up to --max_chars")
    g.add_argument("--frame_gate", action="store_true",
                   help="a scene ends before its first out-of-frame or self-described turn")
    g.add_argument("--cut_closings", type=float, default=0.0,
                   help="> 0: cut a scene after a farewell episode where at least this share of the candidates "
                        "ends the document, ending it with one of those candidates (reads <run>/nodes.jsonl)")
    g.add_argument("--clean_tail", action="store_true",
                   help="cut a final turn at its first paragraph that is no longer the scene")
    g.add_argument("--max_template", type=float, default=1.0, help="drop a scene above this own-opening template share")
    g.add_argument("--min_scene_turns", type=int, default=1, help="drop a scene left with fewer turns")
    g.add_argument("--scenes_out", default="", help="also write each cleaned scene as its own transcript row")
    g.add_argument("--records_out", default="", help="per-scene cleaning records (JSONL)")
    g.add_argument("--cut_unended", type=int, default=0,
                   help="> 0: a scene without a model-written ending keeps at most this many turns")
    g.add_argument("--holdout_frac", type=float, default=0.0,
                   help="hold out this share of seeds (every scene of a held-out seed) for evaluation")
    g.add_argument("--holdout_out", default="", help="the held-out scenes (JSONL); required with --holdout_frac")
    a = ap.parse_args(argv)
    paths = []
    for ri, r in enumerate(a.runs):  # tree ids restart in every run directory: namespace them by run
        done = finished(Path(r))
        for line in (Path(r) / "paths.jsonl").read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                if p["tree"] not in done:  # an interrupted tree's partial paths
                    continue
                p["tree"], p["leaf"] = f"{ri}:{p['tree']}", f"{ri}:{p['leaf']}"
                paths.append(p)
    if a.anthology:
        return _assemble_anthology(a, paths)
    docs = assemble(paths, a.max_chars, a.train_chars or a.max_chars // 2, a.max_uses, a.min_new, a.seed,
                    Clean(a.truncate_loops, a.untrain_repeats, a.untrain_format, a.drop_minor_explicit,
                          compliance.load_blocklist(a.blocklist) if a.blocklist else frozenset(), a.redact_pii))
    with open(a.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n_turns = sum(len(d["turns"]) for d in docs)
    logger.info("%d paths -> %d documents, %d turns (%.1f per document)", len(paths), len(docs), n_turns,
                n_turns / max(len(docs), 1))
    return 0


def _assemble_anthology(a: argparse.Namespace, paths: list[dict[str, Any]]) -> int:
    from kotodama.posttrain import anthology as an

    nodes = None
    if a.cut_closings > 0:
        nodes = an.NodeIndex()
        for ri, r in enumerate(a.runs):
            nodes.extend(Path(r) / "nodes.jsonl", prefix=f"{ri}:")
    cfg = an.SceneConfig(truncate_loops=a.truncate_loops, frame_gate=a.frame_gate, cut_closings=a.cut_closings,
                         clean_tail=a.clean_tail, max_template=a.max_template, untrain_repeats=a.untrain_repeats,
                         untrain_format=a.untrain_format, drop_minor_explicit=a.drop_minor_explicit,
                         blocklist=compliance.load_blocklist(a.blocklist) if a.blocklist else frozenset(),
                         redact_pii=a.redact_pii, min_scene_turns=a.min_scene_turns, cut_unended=a.cut_unended)
    docs, scenes, recs = an.assemble_scenes(paths, nodes, cfg, a.max_chars, a.seed)
    outs = []
    if a.holdout_frac > 0:
        if not a.holdout_out:
            raise SystemExit("--holdout_frac needs --holdout_out")
        held = an.holdout_seeds([p.get("seed", "") for p in paths], a.holdout_frac)
        out_rows = [s for s in scenes if an.seed_key(s["source"].get("seed", "")) in held]
        scenes = [s for s in scenes if an.seed_key(s["source"].get("seed", "")) not in held]
        docs = an.stack(scenes, a.max_chars, a.seed)
        outs.append((a.holdout_out, out_rows))
        logger.info("held out %d seeds: %d scenes", len(held), len(out_rows))
    outs += [(a.out, docs)] + [(x, rows) for x, rows in ((a.scenes_out, scenes), (a.records_out, recs)) if x]
    for path, rows in outs:
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("%d scenes kept -> %d documents (%.2f scenes per document)", len(scenes), len(docs),
                len(scenes) / max(len(docs), 1))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [loom] %(message)s", datefmt="%H:%M:%S")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["assemble"]:
        return assemble_main(argv[1:])
    if argv[:1] == ["extend-seeds"]:
        return extend_seeds_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--endpoint", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N seeds (0 = all)")
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--children", type=int, default=3)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--max_depth", type=int, default=24)
    ap.add_argument("--min_turns", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=4, help="trees grown at once")
    ap.add_argument("--law", default="tsumugi")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--repetition_penalty", type=float, default=None)
    ap.add_argument("--window_chars", type=int, default=0,
                    help="> 0: prompts keep the header + seed turns + the latest turns within this many characters "
                         "(early dialogue rotates out), so branches can run past the context")
    ap.add_argument("--deep_after", type=int, default=0, help="from this depth on, cap live nodes at --deep_width")
    ap.add_argument("--deep_width", type=int, default=0)
    ap.add_argument("--select", choices=["uniform", "engage", "novel"], default="uniform")
    ap.add_argument("--select_lambda", type=float, default=0.0,
                    help="engage: >0 leans to turns that hand the next speaker something, <0 to self-contained ones; "
                         "novel: >0 leans to turns whose content words the last 30 turns did not use")
    ap.add_argument("--novelty_speech_only", action="store_true",
                    help="novel: count only speech (narration paragraphs and stage directions earn no novelty)")
    ap.add_argument("--novelty_norm", choices=["none", "length"], default="none",
                    help="novel: length = choose on the shrunk novelty share's residual after a fit on log(1 + words) "
                         "across the node's survivors, so selection does not track turn length")
    ap.add_argument("--novelty_prior", type=float, default=4.0,
                    help="novelty_norm length: pseudo-count (content words) shrinking each share toward the siblings'")
    ap.add_argument("--deep_children", type=int, default=0,
                    help="> 0: nodes at depth >= --deep_after keep at most this many children (e.g. --children 2 "
                         "--deep_after 6 --deep_children 1: branch in the opening, then grow each branch as one path)")
    ap.add_argument("--rerolls", type=int, default=0,
                    help="a node whose candidates are all vetoed draws --candidates more, up to N times, before it "
                         "ends as degraded")
    ap.add_argument("--cycle_window", type=int, default=0,
                    help="veto a turn that repeats an adjacent pair of turns said in the last N turns (speech only)")
    ap.add_argument("--template_window", type=int, default=0,
                    help="veto a 4+-word turn whose opening pair with the previous turn's recurs > --template_max "
                         "times in the last N turns")
    ap.add_argument("--template_max", type=int, default=3)
    ap.add_argument("--engage_replies", type=int, default=2)
    ap.add_argument("--overlong", choices=["veto", "keep"], default="veto")
    ap.add_argument("--switch_model", default="",
                    help="from turn --switch_after on, every draw asks for this model (the OpenAI model field: a gateway "
                         "pool) instead of --model, e.g. one model opens a scene and another carries it")
    ap.add_argument("--switch_after", type=int, default=0, help="turns written by --model before --switch_model takes over")
    ap.add_argument("--unsigned_utility", action="store_true",
                    help="utility draws (speaker tags, engagement replies) send sign: false, so a server that signs by "
                         "default spends its signing on the scene's own turns (needs a server that knows the field)")
    ap.add_argument("--speaker", choices=["rotate", "model"], default="rotate",
                    help="who speaks next: rotate = round-robin over the cast; model = the model writes the next "
                         "turn's name tag (first of 4 draws naming a cast member other than the last speaker; "
                         "round-robin when none does)")
    ap.add_argument("--steer", default="")
    ap.add_argument("--steer_rule", choices=["K", "N", "S"], default="K")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="",
                    help="the OpenAI model field of every request (a gateway routes it to the pool serving that name)")
    ap.add_argument("--pin", default="",
                    help="every request of tree i carries the OpenAI user field PIN/tree-<i> (gateway affinity: one "
                         "tree, one replica and its prefix cache)")
    ap.add_argument("--skip_done", action="append", default=[],
                    help="another run directory whose finished seeds are skipped (repeatable)")
    a = ap.parse_args(argv)
    sampling = dict(law(a.law))
    if a.temperature is not None:
        sampling["temperature"] = a.temperature
    if a.repetition_penalty is not None:
        sampling["repetition_penalty"] = a.repetition_penalty
    rows = [json.loads(line) for line in Path(a.seeds).read_text().splitlines() if line.strip()]
    seeds = [s for s in (parse_seed(r, i) for i, r in enumerate(rows)) if s is not None]
    if a.limit:
        seeds = seeds[: a.limit]
    out = Path(a.out)
    if out.exists():
        dropped = prune_unfinished(out)
        if dropped:
            logger.info("resume: dropped %d records of unfinished trees in %s", dropped, out)
    done_seeds = {sd for d in (out, *map(Path, a.skip_done)) for sd in finished(d).values()}
    todo = [(i, s) for i, s in enumerate(seeds) if s.id not in done_seeds]
    logger.info("%d seeds, %d already grown, %d to grow", len(seeds), len(seeds) - len(todo), len(todo))
    cfg = LoomConfig(a.candidates, a.children, a.width, a.max_depth, a.min_turns, a.max_tokens,
                     window_chars=a.window_chars, deep_after=a.deep_after, deep_width=a.deep_width,
                     select=a.select, select_lambda=a.select_lambda, engage_replies=a.engage_replies,
                     overlong=a.overlong, novelty_speech_only=a.novelty_speech_only, novelty_norm=a.novelty_norm,
                     novelty_prior=a.novelty_prior, deep_children=a.deep_children, rerolls=a.rerolls,
                     speaker=a.speaker, unsigned_utility=a.unsigned_utility,
                     switch_model=a.switch_model, switch_after=a.switch_after)
    sp = SelfPlayConfig(candidates=a.candidates, max_tokens=a.max_tokens, min_turns=a.min_turns,
                        cycle_window=a.cycle_window, template_window=a.template_window, template_max=a.template_max)
    complete, close = oai_completer(a.endpoint, sampling, a.max_tokens, model=a.model,
                                    steer=parse_steer(a.steer) or None, steer_rule=a.steer_rule)

    score, close_score = http_scorer(a.endpoint) if a.select == "engage" else (None, None)

    async def go() -> dict[str, Any]:
        try:
            return await run(todo, complete, cfg, sp, out, a.concurrency, a.seed, score,
                             routes=len(a.endpoint) if len(a.endpoint) > 1 else 0, pin=a.pin)
        finally:
            await close()
            if close_score is not None:
                await close_score()

    summary = asyncio.run(go())
    (out / "run.json").write_text(json.dumps({"config": asdict(cfg), "sp": asdict(sp), "law": a.law, "sampling": sampling,
                                              "pin": a.pin, "model": a.model, "steer": parse_steer(a.steer),
                                              "steer_rule": a.steer_rule, "seed": a.seed, "seeds": a.seeds,
                                              "endpoints": a.endpoint, **summary}, indent=1))
    logger.info("done: %s", summary)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
