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
self-contained ones (narration, scene-setting, monologue); nothing is vetoed by it. ``--overlong keep`` lets a turn
that reaches the token cap live (flagged) instead of vetoing it.

Long conversations: ``--window_chars`` keeps each prompt to the header + seed turns + the latest turns within a
character budget (early dialogue rotates out, so branches run past the context and drift apart), and
``--deep_after D --deep_width W`` narrows the tree after depth D (wide early for breadth, narrow later for depth).
``python -m kotodama.posttrain.loom assemble <runs> --out docs.jsonl`` cuts paths into training documents of up to
~4K tokens in which no turn appears more than ``--max_uses`` times, so shared trunks are not overweighted.

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

from kotodama.posttrain.selfplay import (Complete, SeedSpec, SelfPlayConfig, evaluate, next_speaker, oai_completer,
                                         parse_seed, parse_steer, split_completion)
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
    select: str = "uniform"  # uniform | engage
    select_lambda: float = 0.0
    engage_replies: int = 2
    overlong: str = "veto"  # veto | keep


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
                     score: Score, replies: int) -> float:
    """Mean over sampled next-speaker replies of log p(reply | context + cand) − log p(reply | context)."""
    after = turns + (cand,)
    nxt = next_speaker(cast, list(after))
    stops = stop_sequences(cast)
    with_c = render_prompt(Transcript(after, header), nxt)
    without = render_prompt(Transcript(turns, header), nxt)
    raw = await complete(with_c, replies, stops)
    texts = [split_completion(t, stops, fr, hit)[0].strip() for t, fr, hit in raw]
    texts = [t for t in texts if t] or ["..."]
    lps = await score([(p, " " + t) for t in texts for p in (with_c, without)])
    return sum(lps[2 * i] - lps[2 * i + 1] for i in range(len(texts))) / len(texts)


async def grow(seed: SeedSpec, tree: str, complete: Complete, cfg: LoomConfig, sp: SelfPlayConfig,
               salt: str, emit: Callable[[str, dict[str, Any]], None], score: Optional[Score] = None) -> dict[str, Any]:
    """Grow one tree breadth-first; every candidate goes to ``emit("node", ...)``, every ended branch to
    ``emit("path", ...)``. Choices are seeded per node (``salt`` + node id), so a tree does not depend on the order
    concurrent requests return in. Returns the tree's stats."""
    stops = stop_sequences(seed.cast)
    root = Node(f"{tree}/r", None, 0, tuple(seed.turns))
    frontier = [root]
    stats = {"tree": tree, "seed": seed.id, "requests": 0, "candidates": 0, "children": 0,
             "ended": {"eos": 0, "degraded": 0, "pruned": 0, "depth": 0, "length_cap": 0}, "max_depth": 0}

    def end(node: Node, state: str) -> None:
        node.state = state
        stats["ended"][state] += 1
        stats["max_depth"] = max(stats["max_depth"], node.depth)
        emit("state", {"tree": tree, "id": node.id, "state": state})
        if node.parent is not None:
            emit("path", {**to_row(Transcript(node.turns, seed.header)), "tree": tree, "seed": seed.id,
                          "leaf": node.id, "state": state, "depth": node.depth, "cast": seed.cast})

    n_seed = len(seed.turns)

    def visible(turns: tuple[Turn, ...]) -> tuple[Turn, ...]:
        """What the model sees: everything, or (with a window) the seed turns + the latest turns that fit."""
        if not cfg.window_chars:
            return turns
        budget = cfg.window_chars - len(seed.header) - sum(len(t.speaker) + len(t.text) + 4 for t in turns[:n_seed])
        k = len(turns)
        while k > n_seed and budget - (len(turns[k - 1].speaker) + len(turns[k - 1].text) + 4) >= 0:
            k -= 1
            budget -= len(turns[k].speaker) + len(turns[k].text) + 4
        return turns[:n_seed] + turns[k:]

    async def expand(node: Node) -> list[Node]:
        speaker = next_speaker(seed.cast, list(node.turns))
        seen = visible(node.turns)
        prompt = render_prompt(Transcript(seen, seed.header), speaker)
        if len(prompt) > cfg.max_prompt_chars:
            end(node, "length_cap")
            return []
        raw = await complete(prompt, cfg.candidates, stops)
        stats["requests"] += 1
        cands = [evaluate(*split_completion(t, stops, fr, hit)[:2], speaker, list(seen), seed.cast,
                          seed.header, sp) for t, fr, hit in raw]
        if cfg.overlong == "keep":
            for c in cands:
                if "overlong" in c.vetoes:
                    c.vetoes.remove("overlong")
        survivors = [i for i, c in enumerate(cands) if c.ok]
        rng = random.Random(f"{salt}:{node.id}")
        eng: dict[int, float] = {}
        if cfg.select == "engage" and score is not None and len(survivors) > cfg.children:
            try:
                vals = await asyncio.gather(*(engagement(node.turns, seed.header, seed.cast,
                                                         Turn(speaker, cands[i].text), complete, score,
                                                         cfg.engage_replies) for i in survivors))
                eng = dict(zip(survivors, vals))
            except Exception as e:  # a failed scoring round costs this node its weighting, not the tree
                logger.warning("node %s: engagement scoring failed (%s); choosing uniformly", node.id, e)
                stats["engage_failures"] = stats.get("engage_failures", 0) + 1
        if eng:
            vals = [eng[i] for i in survivors]
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
            prev = node.turns[-1].text if node.turns else ""
            # an echo question is neutral (z = 0): neither end of the dial selects it for its echo
            z = {i: 0.0 if echo_question(cands[i].text, prev) else (eng[i] - mu) / sd for i in survivors}
            w = [math.exp(cfg.select_lambda * z[i]) for i in survivors]
            chosen = {survivors[j] for j in weighted_choice(rng, w, cfg.children)}
        else:
            chosen = set(rng.sample(survivors, min(cfg.children, len(survivors))))
        kids: list[Node] = []
        for i, c in enumerate(cands):
            stats["candidates"] += 1
            cid = f"{node.id}.{i}"
            status = "child" if i in chosen else ("vetoed" if c.vetoes else "unchosen")
            emit("node", {"tree": tree, "id": cid, "parent": node.id, "depth": node.depth + 1, "speaker": speaker,
                          **asdict(c), "status": status, **({"engage": eng[i]} if i in eng else {})})
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


def http_scorer(endpoints: list[str], timeout: float = 600.0, transport: Any = None) -> tuple[Score, Callable[[], Awaitable[None]]]:
    """A ``Score`` over the servers' /score_tokens (round-robin), tokenizing locally; and its async closer."""
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
                      "completion_ids": tok.encode(c, add_special_tokens=False)[:256]}
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
            out += [-x["mean_nll"] if x["n_tokens"] else 0.0 for x in r.json()["results"]]
        return out

    return score, client.aclose


async def run(seeds: list[SeedSpec], complete: Complete, cfg: LoomConfig, sp: SelfPlayConfig, out_dir: Path,
              concurrency: int, seed: int, score: Optional[Score] = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {k: (out_dir / f"{k}s.jsonl").open("a", encoding="utf-8") for k in ("node", "path", "tree")}
    states = (out_dir / "states.jsonl").open("a", encoding="utf-8")
    sem = asyncio.Semaphore(concurrency)
    t0, done = time.time(), {"n": 0}

    def emit(kind: str, rec: dict[str, Any]) -> None:
        (states if kind == "state" else files[kind]).write(json.dumps(rec, ensure_ascii=False) + "\n")

    async def one(i: int, s: SeedSpec) -> None:
        async with sem:
            try:
                st = await grow(s, f"t{i:05d}", complete, cfg, sp, f"{seed}:{s.id}:{i}", emit, score)
            except Exception as e:  # one tree's failure must not stop the run
                logger.error("tree %d (seed %s) failed: %s", i, s.id, e)
                return
            files["tree"].write(json.dumps(st) + "\n")
            for f in (*files.values(), states):
                f.flush()
            done["n"] += 1
            logger.info("tree %d/%d: %d candidates, %d children, max depth %d, ended %s (%.0fs)", done["n"],
                        len(seeds), st["candidates"], st["children"], st["max_depth"], st["ended"], time.time() - t0)

    try:
        await asyncio.gather(*(one(i, s) for i, s in enumerate(seeds)))
    finally:
        for f in (*files.values(), states):
            f.close()
    return {"trees": done["n"], "seconds": round(time.time() - t0, 1)}


def node_ids(leaf: str) -> list[str]:
    """The ids of the nodes on a leaf's path, root's child first ("t0/r.1.5.3" -> t0/r.1, t0/r.1.5, t0/r.1.5.3)."""
    head, _, rest = leaf.partition("/r.")
    parts = rest.split(".") if rest else []
    return [f"{head}/r." + ".".join(parts[: i + 1]) for i in range(len(parts))]


def assemble(paths: list[dict[str, Any]], max_chars: int, lead_in: int, max_uses: int, min_new: int,
             seed: int) -> list[dict[str, Any]]:
    """Training documents from loom paths without leaning on the shared trunk.

    Each document is the path's header (and seed turns) + a contiguous run of its generated turns within
    ``max_chars``. Paths are taken in random order; a document starts at the first turn trained fewer than
    ``max_uses`` times so far, preceded by up to ``lead_in`` turns marked ``"train": false`` (context the model
    conditions on, no loss), and a path keeps yielding consecutive documents until its end. Every turn therefore
    TRAINS in at most ``max_uses`` documents however many leaves share it, and deep paths give several documents.
    Turns already at the cap inside a document are context too. A document training fewer than ``min_new`` turns is
    dropped."""
    rng = random.Random(seed)
    order = list(range(len(paths)))
    rng.shuffle(order)
    uses: dict[str, int] = {}
    docs: list[dict[str, Any]] = []
    for pi in order:
        p = paths[pi]
        n_seed = len(p["turns"]) - p["depth"]
        prefix, gen = p["turns"][:n_seed], p["turns"][n_seed:]
        ids = node_ids(p["leaf"])
        if len(ids) != len(gen):
            continue
        fixed = len(p.get("header", "")) + sum(len(t["text"]) + len(t["speaker"]) + 4 for t in prefix)
        i = 0
        while i < len(gen):
            while i < len(gen) and uses.get(ids[i], 0) >= max_uses:
                i += 1
            if i >= len(gen):
                break
            start = max(0, i - lead_in)
            j, size = start, fixed
            while j < len(gen) and size + len(gen[j]["text"]) + len(gen[j]["speaker"]) + 4 <= max_chars:
                size += len(gen[j]["text"]) + len(gen[j]["speaker"]) + 4
                j += 1
            if j <= i:  # a single turn larger than the budget
                i += 1
                continue
            trains = [k >= i and uses.get(ids[k], 0) < max_uses for k in range(start, j)]
            if sum(trains) >= min_new:
                turns = [dict(t, train=False) for t in prefix]
                for k, tr in zip(range(start, j), trains):
                    turns.append(dict(gen[k], train=tr))
                    if tr:
                        uses[ids[k]] = uses.get(ids[k], 0) + 1
                docs.append({"header": p.get("header", ""), "turns": turns,
                             "source": {"tree": p["tree"], "leaf": p["leaf"], "from": start, "to": j}})
            i = j
    return docs


def assemble_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="loom paths -> training documents (python -m kotodama.posttrain.loom assemble)")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_chars", type=int, default=14000, help="~4K tokens of English text")
    ap.add_argument("--lead_in", type=int, default=4)
    ap.add_argument("--max_uses", type=int, default=2)
    ap.add_argument("--min_new", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    paths = []
    for ri, r in enumerate(a.runs):  # tree ids restart in every run directory: namespace them by run
        for line in (Path(r) / "paths.jsonl").read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                p["tree"], p["leaf"] = f"{ri}:{p['tree']}", f"{ri}:{p['leaf']}"
                paths.append(p)
    docs = assemble(paths, a.max_chars, a.lead_in, a.max_uses, a.min_new, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n_turns = sum(len(d["turns"]) for d in docs)
    logger.info("%d paths -> %d documents, %d turns (%.1f per document)", len(paths), len(docs), n_turns,
                n_turns / max(len(docs), 1))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [loom] %(message)s", datefmt="%H:%M:%S")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["assemble"]:
        return assemble_main(argv[1:])
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
    ap.add_argument("--select", choices=["uniform", "engage"], default="uniform")
    ap.add_argument("--select_lambda", type=float, default=0.0,
                    help="engage: >0 leans to turns that hand the next speaker something, <0 to self-contained ones")
    ap.add_argument("--engage_replies", type=int, default=2)
    ap.add_argument("--overlong", choices=["veto", "keep"], default="veto")
    ap.add_argument("--steer", default="")
    ap.add_argument("--steer_rule", choices=["K", "N", "S"], default="K")
    ap.add_argument("--seed", type=int, default=0)
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
    cfg = LoomConfig(a.candidates, a.children, a.width, a.max_depth, a.min_turns, a.max_tokens,
                     window_chars=a.window_chars, deep_after=a.deep_after, deep_width=a.deep_width,
                     select=a.select, select_lambda=a.select_lambda, engage_replies=a.engage_replies,
                     overlong=a.overlong)
    sp = SelfPlayConfig(candidates=a.candidates, max_tokens=a.max_tokens, min_turns=a.min_turns)
    complete, close = oai_completer(a.endpoint, sampling, a.max_tokens, steer=parse_steer(a.steer) or None,
                                    steer_rule=a.steer_rule)
    out = Path(a.out)

    score, close_score = http_scorer(a.endpoint) if a.select == "engage" else (None, None)

    async def go() -> dict[str, Any]:
        try:
            return await run(seeds, complete, cfg, sp, out, a.concurrency, a.seed, score)
        finally:
            await close()
            if close_score is not None:
                await close_score()

    summary = asyncio.run(go())
    (out / "run.json").write_text(json.dumps({"config": asdict(cfg), "sampling": sampling, "steer": parse_steer(a.steer),
                                              "steer_rule": a.steer_rule, "seed": a.seed, "seeds": a.seeds,
                                              "endpoints": a.endpoint, **summary}, indent=1))
    logger.info("done: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
