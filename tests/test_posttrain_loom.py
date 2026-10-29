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


def test_novel_selection_prefers_turns_with_new_content() -> None:
    fresh = ["lighthouse keeper rowed", "harbour master waited", "ferry captain shouted", "orchard thieves fled",
             "winter market opened", "copper kettle whistled"]

    def script(prompt, i):  # a different set of content words at every depth, so the new turn is always novel
        return "Okay." if i < 4 else f"The {fresh[prompt.count(chr(10)) // 2 % len(fresh)]} at dawn."

    cfg = loom.LoomConfig(candidates=6, children=1, width=1, max_depth=4, min_turns=1, select="novel",
                          select_lambda=4.0)
    st, recs = grow(cfg, fake(script)[0])
    chosen = [r for r in recs["node"] if r["status"] == "child"]
    assert chosen and all(r["text"].startswith("The ") for r in chosen)
    assert all("novelty" in r and "words" in r for r in chosen)  # the selector's inputs are recorded per candidate


def test_length_residuals() -> None:
    import math

    lengths = [2, 5, 9, 20]
    exact = [0.3 + 0.1 * math.log1p(n) for n in lengths]  # novelty fully explained by length
    assert all(abs(r) < 1e-12 for r in loom.length_residuals(exact, lengths))
    assert loom.length_residuals([1.0, 0.0], [4, 4]) == [0.5, -0.5]  # one length: centred on the mean
    res = loom.length_residuals([0.9, 0.1, 0.9, 0.1], [3, 3, 12, 12])  # within each length the novel turn stays ahead
    assert res[0] > res[1] and res[2] > res[3]


def test_shrunk_shares() -> None:
    got = loom.shrunk_shares([(1, 1), (0, 1), (5, 10)], 4.0)  # pooled share 6/12 = .5
    assert [round(x, 6) for x in got] == [0.6, 0.4, 0.5]  # one-word turns pulled toward .5, not 1 or 0
    assert loom.shrunk_shares([(1, 1), (0, 1)], 0.0) == [1.0, 0.0]  # no prior: the raw shares
    assert loom.shrunk_shares([(0, 0), (0, 0)], 4.0) == [0.0, 0.0]


def test_novelty_norm_length_stops_selection_tracking_length() -> None:
    """Long candidates are all-new, short ones half-new: raw novelty picks the long ones every time, while the
    length-normalised choice has nothing left to prefer and picks either."""

    def script(prompt, i):
        d = prompt.count("\n\n")
        return f"w{d}a{i} w{d}b{i} w{d}c{i} w{d}d{i}." if i % 2 == 0 else f"w{d}e{i} tide."

    def long_share(norm: str) -> float:
        picks = []
        for salt in "abc":
            cfg = loom.LoomConfig(candidates=6, children=1, width=1, max_depth=16, min_turns=1, select="novel",
                                  select_lambda=4.0, novelty_norm=norm)
            _, recs = grow(cfg, fake(script)[0], salt=salt)
            picks += [r["text"].count(" ") == 3 for r in recs["node"] if r["status"] == "child"]
        return sum(picks) / len(picks)

    assert long_share("none") > 0.9
    assert 0.2 < long_share("length") < 0.8


def test_rerolls_redraw_before_degrading() -> None:
    draws: dict[str, int] = {}

    async def complete(prompt: str, n: int, stops: list[str]):
        d = prompt.count("\n\n")
        draws[prompt] = draws.get(prompt, 0) + 1
        if d == 2 and draws[prompt] == 1:  # the first draw after one generated turn: every candidate empty (vetoed)
            return [("" + stops[0], "stop", False) for _ in range(n)]
        return [(f"line {d}-{i} about the {['tide', 'map', 'rope'][i % 3]}" + stops[0], "stop", False)
                for i in range(n)]

    st0, _ = grow(loom.LoomConfig(candidates=3, children=1, width=1, max_depth=5, min_turns=1), complete)
    assert st0["ended"]["degraded"] == 1 and st0["max_depth"] == 1
    draws.clear()
    st1, recs = grow(loom.LoomConfig(candidates=3, children=1, width=1, max_depth=5, min_turns=1, rerolls=2), complete)
    assert st1["ended"] == {**st1["ended"], "degraded": 0, "depth": 1} and st1["rerolls"] == 1
    sibs: dict[str, list] = {}
    for r in recs["node"]:
        sibs.setdefault(r["parent"], []).append(r)
    assert sorted(len(v) for v in sibs.values()) == [3, 3, 3, 3, 6]  # the rerolled node keeps both draws' candidates
    assert len({r["id"] for r in recs["node"]}) == len(recs["node"])


def test_deep_children_branches_only_in_the_opening() -> None:
    cfg = loom.LoomConfig(candidates=4, children=2, width=16, max_depth=6, min_turns=1, deep_after=2,
                          deep_width=16, deep_children=1)
    st, recs = grow(cfg, fake()[0])
    assert st["ended"]["depth"] == 4 and st["ended"]["pruned"] == 0  # 2 at the root, 2 each at depth 1, then 1
    kids: dict[str, int] = {}
    for r in recs["node"]:
        kids[r["parent"]] = kids.get(r["parent"], 0) + (r["status"] == "child")
    depth_of = {r["id"]: r["depth"] for r in recs["node"]}
    assert all(n == 1 for p, n in kids.items() if depth_of.get(p, 0) >= 2)


def _path(texts, tree="t0", cast=("A", "B")):
    n = len(texts)
    return {"tree": tree, "leaf": tree + "/r" + ".0" * n, "depth": n, "header": "H", "n_seed": 0, "cast": list(cast),
            "turns": [{"speaker": cast[k % 2], "text": x} for k, x in enumerate(texts)]}


def test_loop_onset_and_truncation() -> None:
    fresh = [f"line number {k} about the harbour" for k in range(30)]
    loop = ["You do.", "I don't.", "You do, Paulo.", "You don't, Richard.", "No.", "No. You don't."] * 8
    turns = [{"speaker": "A", "text": x} for x in fresh + loop]
    o = loom.loop_onset(turns)
    assert o == 36  # the first cycle (turns 30-35) is new speech; the cut falls at its first repeat
    assert loom.loop_onset(turns[:30]) == 30
    p = _path(fresh + loop)
    plain = loom.assemble([p], 100_000, 50_000, 1, 1, 0)
    cut = loom.assemble([p], 100_000, 50_000, 1, 1, 0, loom.Clean(truncate_loops=True))
    trained = lambda docs: [t["text"] for d in docs for t in d["turns"] if t["train"]]  # noqa: E731
    assert len(trained(plain)) == len(fresh + loop)
    assert len(trained(cut)) == o and trained(cut)[30:] == loop[:6]  # one pass of the cycle, then the cut


def test_untrain_repeats_and_format() -> None:
    texts = ["Where were you last night?", "Out by the docks, with Jac.", "Yes.", "Why the docks?",
             "Jac: Because the boats were in.\nAnd we watched them.", "[MFT] So you say.", "Yes.", "I do say."]
    p = _path(texts)
    docs = loom.assemble([p], 100_000, 50_000, 1, 1, 0, loom.Clean(untrain_repeats=10, untrain_format=True))
    flags = {t["text"]: t["train"] for t in docs[0]["turns"]}
    assert flags["Where were you last night?"] and flags["Why the docks?"] and flags["I do say."]
    assert not flags["[MFT] So you say."]  # bracket tag
    assert not [t for t in docs[0]["turns"] if t["text"] == "Yes." and t["train"]][1:]  # the second "Yes." is context
    assert flags["Out by the docks, with Jac."]  # a name inside a sentence is not a label line
    q = _path(["Hello there, friend.", "Hello. I have news.\nJac: and so do I", "Tell me."], tree="t1")
    d2 = loom.assemble([q], 100_000, 50_000, 1, 1, 0, loom.Clean(untrain_format=True))
    assert not {t["text"]: t["train"] for t in d2[0]["turns"]}["Hello. I have news.\nJac: and so do I"]
    cast_ok = _path(["Hello there, friend.", "Fine.\nB: and you?", "Tell me."], tree="t2")
    d3 = loom.assemble([cast_ok], 100_000, 50_000, 1, 1, 0, loom.Clean(untrain_format=True))
    assert all(t["train"] for t in d3[0]["turns"])  # a cast name's label is cut_stray_label's job, not this rule's


def test_run_pin_sends_one_user_per_tree(tmp_path) -> None:
    users: dict[str, set] = {}

    async def complete(prompt: str, n: int, stops: list[str], user=None):
        users.setdefault(prompt.split("\n")[0], set()).add(user)
        return [(f"line {prompt.count(chr(10))}-{i}" + stops[0], "stop", False) for i in range(n)]

    seeds = [(i, SeedSpec(f"s{i}", ["Mara", "Teo"], f"header {i}", [])) for i in range(3)]
    cfg = loom.LoomConfig(candidates=2, children=1, width=1, max_depth=3, min_turns=2)
    asyncio.run(loom.run(seeds, complete, cfg, SelfPlayConfig(min_turns=2), tmp_path, concurrency=3, seed=0,
                         pin="probe0"))
    assert users == {f"header {i}": {f"probe0/tree-{i:05d}"} for i in range(3)}


def test_main_sends_the_model_field_and_records_it(tmp_path, monkeypatch) -> None:
    seen = {}

    def fake_completer(endpoints, sampling, max_tokens, **kw):
        seen.update(kw)

        async def complete(*a, **k):
            return []

        async def close():
            return None

        return complete, close

    monkeypatch.setattr(loom, "oai_completer", fake_completer)
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text("")
    out = tmp_path / "run"
    assert loom.main(["--seeds", str(seeds), "--endpoint", "http://x", "--out", str(out), "--model", "pool-b"]) == 0
    assert seen["model"] == "pool-b"
    assert json.loads((out / "run.json").read_text())["model"] == "pool-b"


def test_model_speaker_mode_lets_the_model_name_the_next_speaker() -> None:
    seed = SeedSpec("s3", ["Mara", "Teo", "Ines"], "Three on a ferry.", [])
    tags = []

    async def complete(prompt: str, n: int, stops: list[str], **kw):
        if stops == [":", "\n"]:  # the name-tag draw: always pick Ines unless she just spoke, then Mara
            last = prompt.rstrip("\n").rsplit("\n\n", 1)[-1].split(":", 1)[0]
            tags.append(last)
            return [("Mara" if last == "Ines" else "Ines", "stop", False)] * n
        depth = prompt.count("\n\n")
        return [(f"line {depth}-{i} about the tide{stops[0]}", "stop", False) for i in range(n)]

    cfg = loom.LoomConfig(candidates=2, children=1, width=1, max_depth=4, min_turns=2, speaker="model")
    recs = {"node": [], "path": [], "state": []}
    st = asyncio.run(loom.grow(seed, "t3", complete, cfg, SelfPlayConfig(min_turns=2), "x",
                               lambda k, r: recs[k].append(r)))
    speakers = [t["speaker"] for t in max(recs["path"], key=lambda p: p["depth"])["turns"]]
    assert speakers[:4] == ["Ines", "Mara", "Ines", "Mara"]  # never the strict rotation Mara, Teo, Ines
    assert st["speaker_model"] == len(tags) and st.get("speaker_rotate", 0) == 0
