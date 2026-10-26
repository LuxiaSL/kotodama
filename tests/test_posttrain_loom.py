"""posttrain.loom: tree growth over a fake completer — children, vetoes, width pruning, eos/degraded/depth ends,
sibling grouping, and order-independent determinism."""

from __future__ import annotations

import asyncio
import json
import random

from kotodama.posttrain import loom
from kotodama.posttrain.selfplay import SeedSpec, SelfPlayConfig

SEED = SeedSpec("s0", ["Mara", "Teo"], "Two siblings on a ferry.", [])


def fake(script=None, jitter: bool = False):
    """A completer returning distinct, plausible turns; ``script(prompt, i)`` may override a candidate."""
    calls = {"n": 0}

    async def complete(prompt: str, n: int, stops: list[str]):
        calls["n"] += 1
        if jitter:
            await asyncio.sleep(random.random() / 200)
        depth = prompt.count("\n\n")
        out = []
        for i in range(n):
            text = script(prompt, i) if script else None
            if text is None:
                text = f"line {depth}-{i} about the {['tide', 'map', 'rope'][i % 3]}"
            out.append((text + stops[0], "stop", False))
        return out

    return complete, calls


def grow(cfg, complete, salt="x"):
    recs = {"node": [], "path": [], "state": []}
    st = asyncio.run(loom.grow(SEED, "t0", complete, cfg, SelfPlayConfig(min_turns=cfg.min_turns), salt,
                               lambda k, r: recs[k].append(r)))
    return st, recs


def test_tree_grows_to_depth_with_width_cap() -> None:
    cfg = loom.LoomConfig(candidates=4, children=2, width=3, max_depth=5, min_turns=2)
    complete, calls = fake()
    st, recs = grow(cfg, complete)
    # width 3 open nodes reach depth 4; their 2 children each end at max depth 5 (not counted against the width)
    assert st["max_depth"] == 5 and st["ended"]["depth"] == 6
    assert st["ended"]["pruned"] > 0
    assert all(len(p["turns"]) == p["depth"] for p in recs["path"])
    kids = [r for r in recs["node"] if r["status"] == "child"]
    sibs = {}
    for r in recs["node"]:
        sibs.setdefault(r["parent"], []).append(r)
    assert all(len(v) == 4 for v in sibs.values())  # every draw = one sibling group of K candidates
    assert all(sum(r["status"] == "child" for r in v) <= 2 for v in sibs.values())
    assert len(kids) == st["children"] and calls["n"] == st["requests"]


def test_vetoes_eos_and_degraded() -> None:
    def script(prompt, i):
        d = prompt.count("\n\n")
        if d >= 3:
            return ""  # every candidate empty -> vetoed -> the branch degrades
        return None

    cfg = loom.LoomConfig(candidates=3, children=1, width=4, max_depth=10, min_turns=2)
    complete, _ = fake(script)
    st, recs = grow(cfg, complete)
    assert st["ended"]["degraded"] >= 1
    assert any(r["status"] == "vetoed" and "empty" in r["vetoes"] for r in recs["node"])

    async def eos_complete(prompt, n, stops):
        return [("the end of it", "stop" if prompt.count("\n\n") < 2 else "eos", False) for _ in range(n)]

    st2, recs2 = grow(loom.LoomConfig(candidates=2, children=1, width=2, max_depth=10, min_turns=1), eos_complete)
    assert st2["ended"]["eos"] == 1 and recs2["path"][0]["state"] == "eos"


def test_growth_is_order_independent() -> None:
    cfg = loom.LoomConfig(candidates=5, children=2, width=4, max_depth=4, min_turns=2)
    a = grow(cfg, fake(jitter=True)[0], salt="7")[1]
    b = grow(cfg, fake(jitter=True)[0], salt="7")[1]
    key = lambda recs: sorted(json.dumps(r, sort_keys=True) for r in recs["node"])  # noqa: E731
    assert key(a) == key(b)
    c = grow(cfg, fake()[0], salt="8")[1]
    assert key(a) != key(c)
