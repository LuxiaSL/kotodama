"""posttrain.loom: tree growth over a fake completer — children, vetoes, width pruning, eos/degraded/depth ends,
sibling grouping, and order-independent determinism."""

from __future__ import annotations

import asyncio
import json
import random

from kotodama.posttrain import loom
from kotodama.posttrain.selfplay import SeedSpec, SelfPlayConfig
from kotodama.posttrain.transcript import Turn

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
    docs = loom.assemble(paths, max_chars=10_000, train_chars=5_000, max_uses=2, min_new=3, seed=0)
    uses: dict[str, int] = {}
    for d in docs:
        src = d["source"]
        ids = loom.node_ids(src["leaf"])[src["from"]:src["to"]]
        for i, t in zip(ids, d["turns"]):
            uses[i] = uses.get(i, 0) + bool(t["train"])
    assert max(uses.values()) <= 2  # no turn TRAINS more than max_uses times; lead-in is context
    assert any(not t["train"] for d in docs for t in d["turns"])  # lead-in / capped turns are context
    assert all(d["header"] == "H" for d in docs) and len(docs) >= 5
    small = loom.assemble(paths[:1], max_chars=60, train_chars=30, max_uses=1, min_new=1, seed=0)
    assert len(small) > 1 and all(len(d["turns"]) * 6 <= 60 for d in small)  # a deep path yields several windows


def test_assemble_main_namespaces_runs(tmp_path) -> None:
    for name in ("a", "b"):  # two runs whose tree/leaf ids collide
        d = tmp_path / name
        d.mkdir()
        p = {"tree": "t00000", "leaf": "t00000/r.0.0.0.0.0.0.0.0", "depth": 8, "header": name, "state": "depth",
             "turns": [{"speaker": "A" if k % 2 == 0 else "B", "text": f"{name}{k}"} for k in range(8)]}
        (d / "paths.jsonl").write_text(json.dumps(p) + "\n")
        (d / "trees.jsonl").write_text(json.dumps({"tree": "t00000", "seed": name}) + "\n")
    out = tmp_path / "docs.jsonl"
    assert loom.main(["assemble", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(out), "--max_uses", "1",
                      "--min_new", "4"]) == 0
    docs = [json.loads(x) for x in out.read_text().splitlines()]
    assert {d["header"] for d in docs} == {"a", "b"}  # the second run's identical ids were not counted as used


def test_resume_skips_finished_seeds_and_drops_unfinished_trees(tmp_path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    # an interrupted run: tree t00000 finished, t00001 left partial records behind
    (out / "trees.jsonl").write_text(json.dumps({"tree": "t00000", "seed": "a"}) + "\n")
    for name in ("nodes", "states", "paths"):
        (out / f"{name}.jsonl").write_text("".join(json.dumps({"tree": t, "id": name}) + "\n"
                                                   for t in ("t00000", "t00001", "t00001")))
    assert loom.prune_unfinished(out) == 6
    assert all(json.loads(x)["tree"] == "t00000" for n in ("nodes", "states", "paths")
               for x in (out / f"{n}.jsonl").read_text().splitlines())
    other = tmp_path / "older"
    other.mkdir()
    (other / "trees.jsonl").write_text(json.dumps({"tree": "t00007", "seed": "b"}) + "\n")
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text("".join(json.dumps({"id": i, "cast": ["Mara", "Teo"], "header": "h"}) + "\n" for i in "abc"))
    grown = []

    async def fake_run(todo, *args, **kw):
        grown.extend(todo)
        return {"trees": len(todo), "failed": 0, "seconds": 0.0}

    real_run, loom.run = loom.run, fake_run
    try:
        assert loom.main(["--seeds", str(seeds), "--endpoint", "http://x:1", "--out", str(out),
                          "--skip_done", str(other)]) == 0
    finally:
        loom.run = real_run
    assert [(i, s.id) for i, s in grown] == [(2, "c")]  # ids follow the seed's index, so a regrown tree keeps its id


def test_run_pins_each_tree_to_the_least_loaded_route(tmp_path) -> None:
    routes: dict[str, set] = {}

    async def complete(prompt: str, n: int, stops: list[str], route=None):
        await asyncio.sleep(random.random() / 500)
        routes.setdefault(prompt.split("\n")[0], set()).add(route)
        return [(f"line {prompt.count(chr(10))}-{i}" + stops[0], "stop", False) for i in range(n)]

    seeds = [(i, SeedSpec(f"s{i}", ["Mara", "Teo"], f"header {i}", [])) for i in range(6)]
    cfg = loom.LoomConfig(candidates=3, children=2, width=3, max_depth=4, min_turns=2)
    summary = asyncio.run(loom.run(seeds, complete, cfg, SelfPlayConfig(min_turns=2), tmp_path, concurrency=3,
                                   seed=0, routes=3))
    assert summary == {**summary, "trees": 6, "failed": 0}
    assert all(len(r) == 1 for r in routes.values())  # every request of a tree went to one server
    assert set().union(*routes.values()) == {0, 1, 2}  # three trees at once spread over three servers


def test_assemble_main_ignores_unfinished_trees(tmp_path) -> None:
    d = tmp_path / "a"
    d.mkdir()
    rows = [{"tree": t, "leaf": f"{t}/r.0.0.0.0.0.0", "depth": 6, "header": t, "state": "depth",
             "turns": [{"speaker": "A" if k % 2 == 0 else "B", "text": f"{t}{k}"} for k in range(6)]}
            for t in ("t00000", "t00001")]
    (d / "paths.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "trees.jsonl").write_text(json.dumps({"tree": "t00000", "seed": "a"}) + "\n")
    out = tmp_path / "docs.jsonl"
    assert loom.main(["assemble", str(d), "--out", str(out), "--min_new", "2"]) == 0
    assert {json.loads(x)["header"] for x in out.read_text().splitlines()} == {"t00000"}


def test_history_seed_continues_a_conversation() -> None:
    hist = [Turn("Mara" if k % 2 == 0 else "Teo", f"old {k} " + "w" * 40) for k in range(10)]
    seed = SeedSpec("x", ["Mara", "Teo"], "Two siblings on a ferry.", [Turn("Mara", "seed line")], hist)
    prompts = []

    async def complete(prompt: str, n: int, stops: list[str]):
        prompts.append(prompt)
        return [(f"new {prompt.count(chr(10))}-{i}" + stops[0], "stop", False) for i in range(n)]

    cfg = loom.LoomConfig(candidates=2, children=1, width=1, max_depth=3, min_turns=1, window_chars=300)
    recs = {"node": [], "path": [], "state": []}
    st = asyncio.run(loom.grow(seed, "t0", complete, cfg, SelfPlayConfig(min_turns=1), "s",
                               lambda k, r: recs[k].append(r)))
    assert st["max_depth"] == 3  # depth counts only the new turns
    path = recs["path"][0]
    assert path["n_seed"] == 1 and len(path["turns"]) == 1 + 10 + 3
    assert all("seed line" in pr for pr in prompts)  # the window keeps the original seed turns
    assert "old 0 " not in prompts[-1] and "old 9 " in prompts[0]  # and rotates the history out


def test_assemble_backfills_context_and_steps_through_long_paths() -> None:
    n = 40
    path = {"tree": "t0", "leaf": "t0/r" + ".0" * n, "depth": n, "header": "H", "n_seed": 1,
            "turns": [{"speaker": "A", "text": "seed"}] + [{"speaker": "B", "text": f"h{k}"} for k in range(5)]
                     + [{"speaker": "A" if k % 2 else "B", "text": f"g{k:02d}"} for k in range(n)]}
    docs = loom.assemble([path], max_chars=120, train_chars=40, max_uses=1, min_new=1, seed=0)
    assert len(docs) > 3  # consecutive documents step forward through the path
    trained = [t["text"] for d in docs for t in d["turns"] if t["train"]]
    assert sorted(trained) == [f"g{k:02d}" for k in range(n)]  # every generated turn trains exactly once
    assert not any(t["text"].startswith("h") and t["train"] for d in docs for t in d["turns"])  # history = context
    assert all(d["turns"][0]["text"] == "seed" and not d["turns"][0]["train"] for d in docs)
    later = docs[1:]
    assert all(not d["turns"][1]["train"] for d in later)  # later documents open with preceding context
    assert all(sum(len(t["text"]) + len(t["speaker"]) + 4 for t in d["turns"]) + 1 <= 120 for d in docs)


def test_extend_seeds_takes_deepest_finished_leaves(tmp_path) -> None:
    d = tmp_path / "run"
    d.mkdir()
    def path(tree, leaf, state, depth):
        return {"tree": tree, "leaf": leaf, "state": state, "depth": depth, "cast": ["A", "B"], "header": "H",
                "turns": [{"speaker": "A", "text": "seed"}] + [{"speaker": "B", "text": f"g{k}"} for k in range(depth)]}
    rows = [path("t0", "t0/r.0.0.0", "depth", 3), path("t0", "t0/r.1", "pruned", 1), path("t1", "t1/r.0.0.0", "depth", 3)]
    (d / "paths.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "trees.jsonl").write_text(json.dumps({"tree": "t0", "seed": "s"}) + "\n")  # t1 never finished
    out = loom.extend_seeds([str(d)], per_tree=1, min_depth=2, seed=0)
    assert [r["id"] for r in out] == ["run:t0/r.0.0.0"]
    assert out[0]["turns"] == [{"speaker": "A", "text": "seed"}] and len(out[0]["history"]) == 3


def test_prompt_too_long_ends_the_branch_not_the_tree() -> None:
    from kotodama.posttrain.selfplay import PromptTooLong

    async def complete(prompt: str, n: int, stops: list[str]):
        if prompt.count("\n\n") >= 4:
            raise PromptTooLong("Prompt too long")
        return [(f"line {prompt.count(chr(10))}-{i}" + stops[0], "stop", False) for i in range(n)]

    cfg = loom.LoomConfig(candidates=2, children=1, width=2, max_depth=10, min_turns=1)
    st, recs = grow(cfg, complete)
    assert st["ended"]["length_cap"] >= 1 and st["max_depth"] < 10
    assert all(p["state"] in ("length_cap", "eos", "degraded", "pruned") for p in recs["path"])
