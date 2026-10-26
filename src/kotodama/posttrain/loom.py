"""Autoloom: grow a tree of transcripts from each seed instead of one line.

Self-play picks one surviving candidate per turn. The loom keeps them: at every live node it draws K candidates for
the next speaker (one batched request), runs the same mechanical vetoes as self-play, and up to ``--children`` of the
survivors become child nodes that keep growing independently. A branch ends when every candidate is vetoed
(``degraded``), when the model ends the document after ``min_turns`` (``eos``), or at ``--max_depth``. When a depth
holds more than ``--width`` live nodes, a uniform sample of ``width`` keeps growing and the rest stop (``pruned`` —
good branches that simply were not continued). One tree therefore gives breadth and depth within one conversation,
and every node's sibling set (the K candidates of one draw) for preference pairs or branch-relative steering.

    python -m kotodama.posttrain.loom --seeds seeds.jsonl --endpoint http://host:port --out loom/run \\
        [--candidates 8 --children 3 --width 16 --max_depth 24] [--steer vrep:-0.30 --steer_rule N]

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
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

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


@dataclass
class Node:
    id: str
    parent: Optional[str]
    depth: int  # generated turns on the path, this node's included
    turns: tuple[Turn, ...]
    state: str = "open"


async def grow(seed: SeedSpec, tree: str, complete: Complete, cfg: LoomConfig, sp: SelfPlayConfig,
               salt: str, emit: Callable[[str, dict[str, Any]], None]) -> dict[str, Any]:
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

    async def expand(node: Node) -> list[Node]:
        speaker = next_speaker(seed.cast, list(node.turns))
        prompt = render_prompt(Transcript(node.turns, seed.header), speaker)
        if len(prompt) > cfg.max_prompt_chars:
            end(node, "length_cap")
            return []
        raw = await complete(prompt, cfg.candidates, stops)
        stats["requests"] += 1
        cands = [evaluate(*split_completion(t, stops, fr, hit)[:2], speaker, list(node.turns), seed.cast,
                          seed.header, sp) for t, fr, hit in raw]
        survivors = [i for i, c in enumerate(cands) if c.ok]
        chosen = set(random.Random(f"{salt}:{node.id}").sample(survivors, min(cfg.children, len(survivors))))
        kids: list[Node] = []
        for i, c in enumerate(cands):
            stats["candidates"] += 1
            cid = f"{node.id}.{i}"
            status = "child" if i in chosen else ("vetoed" if c.vetoes else "unchosen")
            emit("node", {"tree": tree, "id": cid, "parent": node.id, "depth": node.depth + 1, "speaker": speaker,
                          **asdict(c), "status": status})
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
        if len(nxt) > cfg.width:
            nxt.sort(key=lambda n: n.id)
            keep = set(random.Random(f"{salt}:{tree}:{nxt[0].depth}").sample(range(len(nxt)), cfg.width))
            for i, n in enumerate(nxt):
                if i not in keep:
                    end(n, "pruned")
            nxt = [n for i, n in enumerate(nxt) if i in keep]
        frontier = nxt
    return stats


async def run(seeds: list[SeedSpec], complete: Complete, cfg: LoomConfig, sp: SelfPlayConfig, out_dir: Path,
              concurrency: int, seed: int) -> dict[str, Any]:
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
                st = await grow(s, f"t{i:05d}", complete, cfg, sp, f"{seed}:{s.id}:{i}", emit)
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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [loom] %(message)s", datefmt="%H:%M:%S")
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
    cfg = LoomConfig(a.candidates, a.children, a.width, a.max_depth, a.min_turns, a.max_tokens)
    sp = SelfPlayConfig(candidates=a.candidates, max_tokens=a.max_tokens, min_turns=a.min_turns)
    complete, close = oai_completer(a.endpoint, sampling, a.max_tokens, steer=parse_steer(a.steer) or None,
                                    steer_rule=a.steer_rule)
    out = Path(a.out)

    async def go() -> dict[str, Any]:
        try:
            return await run(seeds, complete, cfg, sp, out, a.concurrency, a.seed)
        finally:
            await close()

    summary = asyncio.run(go())
    (out / "run.json").write_text(json.dumps({"config": asdict(cfg), "sampling": sampling, "steer": parse_steer(a.steer),
                                              "steer_rule": a.steer_rule, "seed": a.seed, "seeds": a.seeds,
                                              "endpoints": a.endpoint, **summary}, indent=1))
    logger.info("done: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
