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


def test_engagement_selection_leans_with_lambda() -> None:
    """A fake scorer makes turns mentioning "?" engaging: λ > 0 should pick them, λ < 0 avoid them."""
    def script(prompt, i):
        d = prompt.count("\n\n")
        return f"what about the {d}-{i}?" if i % 2 == 0 else f"the tide is {d}-{i}."

    async def score(pairs):
        # (with candidate, without) pairs: engaging candidates raise the reply's log-prob
        return [(1.0 if "?" in p.rsplit("\n\n", 2)[-2] else 0.0) if k % 2 == 0 else 0.0 for k, (p, c) in enumerate(pairs)]

    def picked(lam):
        cfg = loom.LoomConfig(candidates=6, children=2, width=8, max_depth=4, min_turns=2, select="engage",
                              select_lambda=lam)
        recs = {"node": [], "path": [], "state": []}
        asyncio.run(loom.grow(SEED, "t0", fake(script)[0], cfg, SelfPlayConfig(min_turns=2), "s",
                              lambda k, r: recs[k].append(r), score))
        kids = [r for r in recs["node"] if r["status"] == "child"]
        assert all("engage" in r for r in kids)
        return sum("?" in r["text"] for r in kids) / len(kids)

    assert picked(4.0) > 0.85 and picked(-4.0) < 0.15


def test_overlong_keep() -> None:
    async def complete(prompt, n, stops):
        d = prompt.count("\n\n")
        words = ["harbour", "lantern", "orchard", "violin", "gravel", "sparrow", "kettle", "meadow"]
        return [(f"{words[d % 8]} {words[(d + 3 * i + 1) % 8]} keeps going {d}{i}", "length", True) for i in range(n)]

    cfg = loom.LoomConfig(candidates=2, children=1, width=2, max_depth=3, min_turns=1, overlong="keep")
    st, recs = grow(cfg, complete)
    assert st["ended"]["degraded"] == 0 and st["max_depth"] == 3
    st2, _ = grow(loom.LoomConfig(candidates=2, children=1, width=2, max_depth=3, min_turns=1), complete)
    assert st2["ended"]["degraded"] == 1


def test_engagement_failure_falls_back_to_uniform() -> None:
    async def broken(pairs):
        raise RuntimeError("server went away")

    cfg = loom.LoomConfig(candidates=4, children=2, width=4, max_depth=3, min_turns=2, select="engage",
                          select_lambda=1.0)
    recs = {"node": [], "path": [], "state": []}
    st = asyncio.run(loom.grow(SEED, "t0", fake()[0], cfg, SelfPlayConfig(min_turns=2), "s",
                               lambda k, r: recs[k].append(r), broken))
    assert st["max_depth"] == 3 and st.get("engage_failures", 0) > 0
    assert all("engage" not in r for r in recs["node"])


def test_echo_question() -> None:
    assert loom.echo_question("A job?", "He wanted to know if you'd be interested in a job?")
    assert loom.echo_question("a house?", "the guy's looking for a handyman for his house.")
    assert not loom.echo_question("What kind of house?", "for his house.")  # 'what', 'kind' are new words
    assert not loom.echo_question("A job.", "a job")  # not a question
    assert not loom.echo_question("So what would you pay me for the job?", "a job")  # too long


def test_window_and_deep_width() -> None:
    prompts: list[str] = []

    async def complete(prompt, n, stops):
        prompts.append(prompt)
        d = prompt.count("\n\n")
        words = ["harbour", "lantern", "orchard", "violin", "gravel", "sparrow", "kettle", "meadow", "copper", "thistle"]
        rng = random.Random(len(prompts) * 10 + 1)
        return [(" ".join(rng.choice(words) + str(rng.randint(0, 999)) for _ in range(12)) + stops[0], "stop", False)
                for i in range(n)]

    cfg = loom.LoomConfig(candidates=3, children=2, width=4, max_depth=30, min_turns=2, window_chars=600,
                          deep_after=5, deep_width=1)
    st, recs = grow(cfg, complete)
    assert st["max_depth"] == 30
    assert max(len(p) for p in prompts) <= 600 + 40  # header + latest turns + the open tag
    assert all(p.startswith("Two siblings on a ferry.") for p in prompts)  # the header always stays
    deep = [r for r in recs["node"] if r["status"] == "child" and r["depth"] > 6]
    assert len({r["depth"] for r in deep}) == len(deep) // 2 or all(
        sum(1 for x in deep if x["depth"] == d) <= 2 for d in {r["depth"] for r in deep})  # 1 live node -> <=2 kids


def test_assemble_bounds_trunk_reuse() -> None:
    def path(leaf, n):
        return {"tree": "t0", "leaf": leaf, "depth": n, "header": "H",
                "turns": [{"speaker": "A" if k % 2 == 0 else "B", "text": f"x{k}"} for k in range(n)]}

    trunk = "t0/r." + ".".join(["0"] * 10)
    paths = [path(trunk + f".{b}" + ".0" * 9, 20) for b in range(5)]  # five leaves sharing a 10-turn trunk
    docs = loom.assemble(paths, max_chars=10_000, lead_in=2, max_uses=2, min_new=3, seed=0)
    uses: dict[str, int] = {}
    for d in docs:
        src = d["source"]
        ids = loom.node_ids(src["leaf"])[src["from"]:src["to"]]
        for i, t in zip(ids, d["turns"]):
            uses[i] = uses.get(i, 0) + bool(t["train"])
    assert max(uses.values()) <= 2  # no turn TRAINS more than max_uses times; lead-in is context
    assert any(not t["train"] for d in docs for t in d["turns"])  # lead-in / capped turns are context
    assert all(d["header"] == "H" for d in docs) and len(docs) >= 5
    small = loom.assemble(paths[:1], max_chars=60, lead_in=0, max_uses=1, min_new=1, seed=0)
    assert len(small) > 1 and all(len(d["turns"]) * 6 <= 60 for d in small)  # a deep path yields several windows
