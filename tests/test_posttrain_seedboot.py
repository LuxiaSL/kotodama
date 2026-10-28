"""Seeds with no hand-written examples: frames, organon pools, bootstrap passes, readouts (fake completer, CPU)."""
from __future__ import annotations

import asyncio
import collections
import json
import random
import re
from pathlib import Path

import pytest

from kotodama.posttrain import seedboot as sb
from kotodama.posttrain import seedpools as sp
from kotodama.posttrain import seeds
from kotodama.posttrain.selfplay import parse_seed

pytest.importorskip("scipy")

_W = ("amber basil cedar delta ember fjord garnet harbor indigo juniper kestrel lantern meadow nettle orchard pewter "
      "quarry russet saffron thistle umber velvet willow yarrow zephyr cobalt marble linen walnut hollow").split()
LISTS = {"antisyzygies": ["order and chaos", "teacher and pupil", "friend and enemy", "old and new"],
         "moments": ["Reunion", "Moving house", "First date", "Retirement"],
         "illusion_examples": [f"Absolute {w}" for w in _W[:12]],
         "illusion_causes": [f"{w.title()} of perception" for w in _W[12:24]]}


def fake(seed: int = 0):
    rng = random.Random(seed)
    calls = collections.Counter()

    def phrase(n: int = 6) -> str:
        return " ".join(rng.sample(_W, n))

    async def complete(prompt: str, n: int, stops: list[str], route=None, user=None):
        calls[user] += 1
        outs = []
        for _ in range(n):
            if prompt.endswith("The play opens:\n"):
                entry = prompt.rsplit("\n\n", 1)[1]
                first = entry.split(" — ")[0].split(". ", 1)[1].split()[0]
                text = f"A kitchen at dawn in May. {first} waits by the {phrase(3)} door. The bus leaves at six."
            elif prompt.endswith(","):
                text = f" forty, a {phrase(4)} keeper"
            elif prompt.endswith("\n") and " — " in prompt.rsplit("\n\n", 1)[1]:
                text = f"{phrase(8).capitalize()}. Then {phrase(7)} happens at night. A comedy about {phrase(3)}."
            elif re.search(r"\d+\.$", prompt) or re.search(r"\d+\. \w+$", prompt):
                text = f" {phrase(5)}\n{phrase(1)}. second {phrase(5)}"
                if "Guest List" in prompt:
                    text = f" {rng.choice(_W).title()} {rng.choice(_W).title()}son\n9. {rng.choice(_W).title()} Lund"
            else:  # a frame title
                text = f"This volume collects {phrase(6)}. It is arranged by {phrase(4)}, and each entry {phrase(5)}."
            outs.append((text, "stop", False))
        return outs

    complete.calls = calls
    return complete


async def _score(pairs):
    return [-(2.0 + len(c) % 7 / 10) for _, c in pairs]


def test_pieces() -> None:
    assert sb.numbered_list("68 Examples of Illusions", ["A", "B"], "Courage") == \
        "68 Examples of Illusions\n\n1. A\n2. B\n3. Courage"
    assert sb.numbered_list("Guest List", ["Ines Moreau"]).endswith("1. Ines Moreau\n2.")
    assert sb.parse_numbered(" and the illusion of enough\n9. Envy and the absent\nNot numbered", False,
                             "Courage") == ["Courage and the illusion of enough", "Envy and the absent"]
    assert sb.genre_tag("Two sisters fight. A bitter little comedy about money.") == "bitter little comedy"
    assert sb.genre_tag("Nothing named here.") == "none"
    g = sb.repetition_guards("one two three four five six seven eight one two three four five six seven eight")
    assert g["loop8"] and g["rep4"] > 0.3
    assert not sb.repetition_guards("a quiet room at dawn with nobody in it")["loop8"]


def test_trim_intro() -> None:
    assert sb.trim_intro("One para.\n\nTwo para cut mid", True) == "One para."
    assert sb.trim_intro("First sentence. Second cut mid", True) == "First sentence."
    assert sb.trim_intro("Whole thing, not capped", False) == "Whole thing, not capped"


def test_frames_and_pick(tmp_path: Path) -> None:
    rows = asyncio.run(sb.write_frames(fake(), tmp_path, ["Catalogue", "Index"], 3, _score))
    assert len(rows) == 6 and all(r["mechanical_ok"] and "nll" in r for r in rows)
    v = tmp_path / "v.jsonl"
    verdict = {"index-0": ("reject", "not-a-collection"), "index-1": ("reject", "incoherent")}
    v.write_text("".join(json.dumps({"id": r["id"], "verdict": verdict.get(r["id"], ("keep", "none"))[0],
                                     "category": verdict.get(r["id"], ("keep", "none"))[1], "reason": "r"}) + "\n"
                         for r in rows))
    frames = sb.pick_frames(tmp_path, v, 8, control=1)
    assert [f["frame"] for f in frames] == ["F0", "F1", "F2", "F3", "FC0"] and frames[-1]["source"] == "index-0"
    assert [f["title"] for f in frames[:4]] == ["Catalogue"] * 3 + ["Index"]
    assert len(sb.pick_frames(tmp_path, v, 1)) == 2  # at most k per title
    assert frames[-1]["control"] and not frames[0]["control"]
    rejected = sp.read_jsonl(tmp_path / "frames.rejected.jsonl")
    assert {r["id"]: r["category"] for r in rejected} == {"index-0": "not-a-collection", "index-1": "incoherent"}
    frames = [f for f in frames if not f["control"]]  # noqa: F841
    assert all(f["text"].startswith(f["title"] + "\n\n") for f in frames)


def test_organon_pools() -> None:
    res = asyncio.run(sb.grow_organon(fake(), "theme", LISTS, 20, random.Random(0), parallel=4))
    assert res["counts"]["final"] == 20
    assert all(r["text"][:1].isupper() for r in res["kept"])
    crossed = [r for r in res["kept"] if r["crossing"]]
    assert crossed and all(r["crossing"][0] in ("illusion_examples", "illusion_causes") for r in crossed)
    names = asyncio.run(sb.grow_organon(fake(), "names", LISTS, 10, random.Random(0),
                                        seed_items=["Ines Moreau", "Kwame Asante"], parallel=4))
    assert names["counts"]["final"] == 10 and all(sp.name_label(r["text"]) for r in names["kept"])


def _setup(tmp_path: Path) -> tuple[dict, list[dict]]:
    pools = {"theme": [{"id": f"theme:{i:04d}", "text": f"Absolute {w} and the longing for rest"}
                       for i, w in enumerate(_W[:10])],
             "names": [{"id": f"names:{i:04d}", "text": f"{c}{c.lower()}na Lund"} for i, c in
                       enumerate("ABCDEFGHIJKLMNOP")]}
    frames = [{"frame": "F0", "title": "Catalogue", "text": "Catalogue\n\nThis volume collects plays."},
              {"frame": "F1", "title": "Index", "text": "Index\n\nEntries in order of performance."}]
    return pools, frames


def test_boot_passes(tmp_path: Path) -> None:
    pools, frames = _setup(tmp_path)
    starts = sb.draw_boot_starts(pools, LISTS, frames, 20, seed=0, pass_no=0)
    assert collections.Counter(s["arm"] for s in starts) == {"note": 10, "bare": 10}
    assert {s["frame"] for s in starts} == {"F0", "F1"} and all(s["frame_title"] for s in starts)
    assert collections.Counter(s["cast_arm"] for s in starts) == {"pair": 14, "cast": 6}
    p = sb.boot_prompt(frames[0]["text"], [], starts[0], "listing")
    assert p.startswith("Catalogue\n\nThis volume collects plays.\n\n1. ") and p.endswith("\n")
    complete = fake()
    d0 = tmp_path / "pass0"
    d0.mkdir()
    fm = {f["frame"]: f["text"] for f in frames}
    stats = asyncio.run(sb.run_boot(complete, _score, starts, fm, [], d0, (2, 4), 0))
    assert stats["pass"] == 20
    assert all(u and u.startswith("p0-") for u in complete.calls)  # every request pinned by its seed
    seeds0 = sp.read_jsonl(d0 / "seeds.all.jsonl")
    for r in seeds0:
        assert parse_seed(r, 0).cast == r["cast"] and r["guards"] and r["nll"] > 0
        assert r["genre"].endswith("comedy")
        assert r["header"].startswith(r["listing"]) == (r["arm"] == "note")
    starts1 = sb.draw_boot_starts(pools, LISTS, frames, 10, seed=0, pass_no=1)
    ex = sb.exemplar_pool(seeds0, "kmedoids", 5, 0)
    assert len(ex) == 5 and len({e["id"] for e in ex}) == 5
    d1 = tmp_path / "pass1"
    d1.mkdir()
    asyncio.run(sb.run_boot(fake(1), None, starts1, fm, ex, d1, (2, 4), 0))
    rows1 = sp.read_jsonl(d1 / "chains.jsonl")
    assert all(2 <= len(r["exemplars"]) <= 4 and set(r["exemplars"]) <= {e["id"] for e in ex} for r in rows1)
    assert len({tuple(r["exemplars"]) for r in rows1}) > 1  # rotated per seed
    prompt = sb.boot_prompt(fm["F0"], ex[:2], starts1[0], "listing")
    assert prompt.count(sb.SCENE_TAIL) == 2 and "\n\n3. " in prompt


def test_variety_and_npz(tmp_path: Path) -> None:
    rng = random.Random(0)
    rows = []
    for i in range(24):
        frame = "F0" if i < 12 else "F1"
        words = _W[:15] if frame == "F0" else _W[15:]
        rows.append({"id": f"s{i}", "frame": frame, "pass": i % 2, "arm": "note", "genre": "comedy",
                     "header": " ".join(rng.choice(words) for _ in range(30)), "nll": 2.0,
                     "guards": {"rep4": 0.0, "loop8": False}})
    rep = sb.variety(rows, "tfidf", "energy", perms=50, verdicts={"s0": "keep", "s1": "reject"})
    c = rep["contrasts"]["frame:F0|F1"]
    assert c["value"] > c["null_q95"] and c["p"] < 0.05 and c["auc"] > 0.9
    assert rep["contrasts"]["pass:0|1"]["auc"] < 0.8 and rep["groups"]["frame=F0"]["mean_cos"] > 0
    assert rep["contrasts"]["pass:0|1"]["p"] > 0.05
    assert rep["groups"]["frame=F0"]["filter_keep"] == 0.5 and rep["groups"]["frame=F0"]["dispersion"] > 0
    assert sb.variety(rows, "tfidf", "mmd", perms=20)["contrasts"]["frame:F0|F1"]["value"] > 0
    import numpy as np

    np.savez(tmp_path / "v.npz", ids=np.array([r["id"] for r in rows]),
             vectors=np.array([[1.0, 0.0] if r["frame"] == "F0" else [0.0, 1.0] for r in rows]))
    x = sb.embed([r["header"] for r in rows], f"npz:{tmp_path / 'v.npz'}", [r["id"] for r in rows])
    assert x.shape == (24, 2)
    with pytest.raises(KeyError):
        sb.embed(["x"], f"npz:{tmp_path / 'v.npz'}", ["nope"])


def test_fan_and_cli(tmp_path: Path) -> None:
    _pools, frames = _setup(tmp_path)
    asyncio.run(sb.frame_fan(fake(), frames, 4, tmp_path))
    fan = sp.read_jsonl(tmp_path / "fan.jsonl")
    assert len(fan) == 8 and {r["frame"] for r in fan} == {"F0", "F1"}
    with pytest.raises(SystemExit):
        seeds.main(["boot", "--endpoint", "http://x", "--out", str(tmp_path), "--lists", str(tmp_path),
                    "--pass", "0"])
