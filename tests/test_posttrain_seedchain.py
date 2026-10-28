"""Conditional-chain seeds: step prompts, step checks, a full chain on a fake completer, the stratified pick."""
from __future__ import annotations

import asyncio
import collections
import json
import random
from pathlib import Path

import pytest

from kotodama.posttrain import seedchain as sc
from kotodama.posttrain import seedpools as sp
from kotodama.posttrain import seeds
from kotodama.posttrain.selfplay import parse_seed

pytest.importorskip("scipy")

_W = ("amber basil cedar delta ember fjord garnet harbor indigo juniper kestrel lantern meadow nettle orchard pewter "
      "quarry russet saffron thistle umber velvet willow yarrow zephyr").split()


def _start(k: int = 2, cast_arm: str = "pair", secret_arm: str = "plain", i: int = 0) -> dict:
    names = ["Ines Moreau", "Kwame Asante", "Dolores Vega", "Bram Visser"][:k]
    return {"id": f"ch{i:05d}", "friction": "collision", "secret_arm": secret_arm, "cast_arm": cast_arm,
            "theme": "owing a debt and the illusion of standing alone", "relationship": "a teacher and her pupil",
            "register": "a dry comedy", "roots": {"theme": "theme:0000", "relationship": "relationship_pair:0000",
                                                  "register": "register:0000", "names": []},
            "names": [{"text": n, "label": sp.name_label(n)} for n in names]}


def fake_chain_completer(seed: int = 0, bad_people: int = 0):
    rng = random.Random(seed)
    calls = collections.Counter()

    def phrase(n: int = 5) -> str:
        return " ".join(rng.sample(_W, n))

    async def complete(prompt: str, n: int, stops: list[str]):
        entry = prompt.rsplit("\nNo. ", 1)[1]
        last = "Page:" if prompt.endswith("\nPage:\n") else prompt.rsplit("\n", 1)[1]
        outs = []
        for _ in range(n):
            if last.endswith("—"):
                calls["person"] += 1
                text = " a couple who share a flat" if calls["person"] <= bad_people else f" a {phrase()} keeper"
            elif last.endswith(":") and last.startswith("- "):
                text = f" to {phrase()}"
            elif last == "Secret:":
                labels = [ln[2:].split(" — ")[0].split()[0] for ln in entry.split("\n") if " — " in ln]
                text = f" {labels[-1]} keeps it from {labels[0]}: the {phrase()} was sold"
            elif last in ("Place:", "Happening:", "Clock:"):
                text = f" the {phrase()} at noon"
            elif last == "Page:":
                people = [ln[2:].split(" — ") for ln in entry.split("\n") if " — " in ln]
                wants = [ln.split(": ", 1)[1] for ln in entry.split("Wants today:\n")[1].split("\nSecret:")[0]
                         .split("\n")]
                secret = entry.split("Secret: ")[1].split(": ", 1)[1].split("\n")[0]
                page = "\n".join(f"{nm}, forty, {who}. {nm.split()[0]} wants {w}." for (nm, who), w
                                 in zip(people, wants))
                text = f"{page}\nNobody else knows {secret}.\n\nA room at noon in May. The bus leaves at one.\n* * *"
                outs.append((text, "stop", False))
                continue
            else:
                raise AssertionError(f"unexpected prompt tail {last!r}")
            outs.append((text + "\n", "stop", False))
        return outs

    return complete


def test_step_prompts_show_the_entry_so_far() -> None:
    st = _start()
    p = sc.step_prompt(st, "person", 0)
    assert p.startswith(sc.CHAIN_HEAD) and p.endswith("\nPeople:\n- Ines Moreau —")
    assert p.count(sp.END_MARK) == len(sp.HEADER_EXAMPLES)
    assert "Drawn: two people; their wants collide; the secret is stated plainly on the page" in p
    st["people"] = ["a baker"]
    assert sc.step_prompt(st, "person", 1).endswith("- Ines Moreau — a baker\n- Kwame Asante —")
    st["people"].append("a pilot")
    assert sc.step_prompt(st, "need", 0).endswith("- Kwame Asante — a pilot\nNeeds:\n- Ines:")
    st["needs"] = ["to rest", "to fly"]
    st["wants"] = ["the cake", "the plane"]
    assert sc.step_prompt(st, "secret").endswith("- Kwame: the plane\nSecret:")
    st["secret"] = {"holder": "Kwame", "target": "Ines", "text": "he sold it"}
    assert sc.step_prompt(st, "place").endswith("Secret: Kwame keeps it from Ines: he sold it\nPlace:")
    st.update(place="a shop", happening="a fire", clock="the bus at one")
    assert sc.step_prompt(st, "page").endswith("Clock: the bus at one\nPage:\n")


def test_check_short() -> None:
    st = _start()
    assert sc.check_short("person", " a night baker who never sleeps", st) == ("a night baker who never sleeps", "")
    assert sc.check_short("person", " a couple who run a bakery", st)[1] == "group_as_person"
    assert sc.check_short("person", " two nurses on the night shift", st)[1] == "group_as_person"
    assert sc.check_short("need", " ok", st)[1] == "too_short"
    assert sc.check_short("want", " the cake. Secret: none", st)[1] == "field_leak"
    v, why = sc.check_short("secret", " Kwame keeps it from Ines: he sold the piano", st)
    assert why == "" and v == {"holder": "Kwame", "target": "Ines", "text": "he sold the piano"}
    assert sc.check_short("secret", " Kwame keeps it from Kwame: he sold the piano", st)[1] == "secret_people"
    assert sc.check_short("secret", " he sold the piano", st)[1] == "secret_format"


def test_chain_end_to_end(tmp_path: Path) -> None:
    complete = fake_chain_completer(bad_people=2)  # the first person draw is two groups -> redraw

    async def score(pairs):
        return [-(1.0 + len(c) / 1000) for _, c in pairs]

    starts = [_start(2, "pair", "plain", 0), _start(3, "cast", "hinted", 1), _start(4, "cast", "plain", 2)]
    stats = asyncio.run(sc.run_chains(complete, complete, score, starts, tmp_path, concurrency=2))
    assert stats == {"chains": 3, "pass": 3, "failed_step": {}}
    rows = sp.read_jsonl(tmp_path / "chains.jsonl")
    first = next(r for r in rows if r["id"] == "ch00000")
    assert first["rejections"]["person0"] == ["group_as_person", "group_as_person"]
    for r in rows:
        assert len(r["people"]) == len(r["needs"]) == len(r["wants"]) == len(r["names"])
        seed = sc.chain_seed(r)
        s = parse_seed(seed, 0)
        assert s is not None and s.cast == seed["cast"] and s.header == r["page"] and seed["nll"] > 1
    assert asyncio.run(sc.run_chains(None, None, None, starts, tmp_path))["chains"] == 3  # resume: nothing redrawn


def test_chain_fails_at_a_step(tmp_path: Path) -> None:
    async def complete(prompt: str, n: int, stops: list[str]):
        return [(" x", "stop", False)] * n

    stats = asyncio.run(sc.run_chains(complete, complete, None, [_start()], tmp_path))
    row = sp.read_jsonl(tmp_path / "chains.jsonl")[0]
    assert stats["failed_step"] == {"person": 1} and row["rejections"]["person0"] == ["too_short"] * 4


def test_draw_roots_and_stratified_pick(tmp_path: Path) -> None:
    def pool(axis: str, n: int) -> list[dict]:
        return [{"id": f"{axis}:{i:04d}", "text": f"{axis} entry {i}"} for i in range(n)]

    pools = {a: pool(a, 20) for a in ("theme", "relationship_pair", "relationship_cast", "register")}
    pools["names"] = [{"id": f"names:{i:04d}", "text": f"{c}{c.lower()}na Lund"} for i, c in
                      enumerate("ABCDEFGHIJKLMNOPQRST")]
    starts = sc.draw_roots(pools, 40, seed=3)
    assert collections.Counter(s["cast_arm"] for s in starts) == {"pair": 28, "cast": 12}
    assert len({(s["roots"]["theme"], s["roots"]["relationship"], s["roots"]["register"]) for s in starts}) == 40
    assert all(s["roots"]["relationship"].startswith("relationship_" + ("pair" if s["cast_arm"] == "pair"
                                                                         else "cast")) for s in starts)
    rows = []
    for i in range(100):  # pass rates skewed by arm: plain casts are scarce
        sa, ca = [("plain", "pair"), ("hinted", "pair"), ("hinted", "cast"), ("plain", "cast")][
            0 if i < 40 else 1 if i < 80 else 2 if i < 95 else 3]
        rows.append({"id": f"r{i}", "secret_arm": sa, "cast_arm": ca, "friction": sp.FRICTIONS[i % 4]})
    picked, stats = sc.stratified(rows)
    assert stats["available"]["plain/cast"] == 5 and stats["total"] == len(picked)
    assert stats["picked"] == {"plain/pair": 11, "hinted/pair": 11, "plain/cast": 5, "hinted/cast": 5}


def test_pick_applies_verdicts_and_cli(tmp_path: Path) -> None:
    complete = fake_chain_completer()
    starts = [_start(2, "pair", "plain", i) for i in range(4)]
    asyncio.run(sc.run_chains(complete, complete, None, starts, tmp_path))
    (tmp_path / "verdicts.jsonl").write_text(json.dumps({"id": "sc-ch00001", "verdict": "reject",
                                                         "category": "mashup", "reason": "x"}) + "\n")
    assert seeds.main(["pick", "--out", str(tmp_path), "--verdicts", str(tmp_path / "verdicts.jsonl")]) == 0
    kept = sp.read_jsonl(tmp_path / "seeds.all.jsonl")
    assert [r["id"] for r in kept] == sorted(f"sc-ch{i:05d}" for i in (0, 2, 3))
    rep = sc.chain_report(tmp_path, sample=2, reject_sample=1, verdicts=tmp_path / "verdicts.jsonl")
    assert rep["filter"]["reject_categories"] == {"mashup": 1}
    assert rep["by_arm"]["cast_arm=pair"]["pass_rate"] == 1.0
    assert "filter: reject / mashup" in (tmp_path / "filter-rejects-sample.md").read_text()


def test_native_chain(tmp_path: Path) -> None:
    async def complete(prompt: str, n: int, stops: list[str]):
        if prompt.endswith("\nThe play opens:\n"):
            text = "A kitchen at dawn in May. Ines stands by the kettle; Kwame waits at the door. The bus is due at six."
        elif prompt.endswith(","):
            text = " forty, a baker who never sleeps"
        else:
            text = ("Two bakers have shared a kitchen and a grudge for twenty years. One of them is leaving at six "
                    "with the recipe the other never wrote down. A comedy about flour and forgiveness.")
        return [(text + "\n", "stop", False)] * n

    starts = [_start(2, "pair", "plain", 0), _start(3, "cast", "plain", 1)]
    sc.assign_genres(starts, 0)
    assert all(s["genre"] in sc.GENRES for s in starts)
    p = sc.native_prompt(starts[0], "blurb")
    assert p.startswith(sc.NATIVE_HEAD) and p.endswith(": Ines Moreau and Kwame Asante.\n")
    assert sc.native_prompt({**starts[0], "blurb": "x. y."}, "character", 0).endswith("x. y.\nCharacters:\nInes Moreau,")
    assert sc.check_native("character", " forty, a couple who run a bakery", starts[0])[1] == "group_as_person"
    assert sc.check_native("scene", " " + "A kitchen with nobody named in it at all, at dawn in May, quiet. " * 2,
                           starts[0])[1] == "nobody_on_stage"
    stats = asyncio.run(sc.run_chains(complete, complete, None, starts, tmp_path, form="native"))
    assert stats["pass"] == 2
    rows = sp.read_jsonl(tmp_path / "chains.jsonl")
    seed = sc.chain_seed(next(r for r in rows if r["id"] == "ch00000"))
    assert seed["form"] == "native" and seed["header"].startswith("Ines Moreau, forty, a baker who never sleeps\n")
    assert seed["header"].endswith("The bus is due at six.") and seed["tags"]["clock_on_page"]
    assert parse_seed(seed, 0).cast == seed["cast"]
    stats = sc.pick(tmp_path)
    assert stats["stratified"]["available"] == {"none/pair": 1, "none/cast": 1}
    rep = sc.chain_report(tmp_path, sample=2)
    assert set(rep["by_arm"]["cast_arm=pair"]["stage_pass_rate"]) == {"blurb", "character", "scene", "page"}
