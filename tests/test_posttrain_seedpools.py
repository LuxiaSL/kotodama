"""Seed pools -> combinations -> headers -> report, on a fake completer and a fake scorer (CPU only)."""
from __future__ import annotations

import asyncio
import collections
import json
import random
from pathlib import Path

import pytest

from kotodama.posttrain import seedpools as sp
from kotodama.posttrain import seeds
from kotodama.posttrain.selfplay import parse_seed

pytest.importorskip("scipy")

_WORDS = ("amber basil cedar delta ember fjord garnet harbor indigo juniper kestrel lantern meadow nettle orchard "
          "pewter quarry russet saffron thistle umber velvet willow yarrow zephyr anchor bramble cobalt dune "
          "falcon glacier hollow ivory jasper kelp linen marble nectar onyx pollen quill raven slate tundra "
          "lumen vessel walnut oxbow yew zinc copper maple birch alder").split()
_FIRST = ("Ada Bea Cy Dov Eli Fen Gus Hal Ivo Jun Kit Lev Max Ned Ola Pia Quin Rex Sol Tam Uma Vic Wes Xan Yul "
          "Zev Abe Bo Cal Dee").split()


def _phrase(rng: random.Random, n: int = 6) -> str:
    return "a " + " ".join(rng.sample(_WORDS, n))


def fake_pool_completer(seed: int = 0):
    rng = random.Random(seed)

    async def complete(prompt: str, n: int, stops: list[str]):
        names = prompt.startswith(sp.AXES["names"].head)
        outs = []
        for _ in range(n):
            if names:
                lines = [f" {rng.choice(_FIRST)} {rng.choice(_WORDS).title()}son" for _ in range(4)]
            else:
                lines = [" " + _phrase(rng)] + [f"- {_phrase(rng)}" for _ in range(3)]
                lines.append("- \u00c3\u00a9 broken text here")  # mojibake
                lines.append("- tiny")  # too short
            outs.append(("\n".join(lines) + "\n- cut off mid", "length", True))
        return outs

    return complete


def test_entry_hygiene() -> None:
    ax = sp.AXES["setting"]
    assert sp.clean_entry("- a quiet harbour at dusk; a red buoy.") == "a quiet harbour at dusk; a red buoy"
    assert sp.check_entry("a night desk at 3:00 a.m.; a missing key card", ax) is None
    assert sp.check_entry("Setting: a night desk; a missing key card", ax) == "malformed"
    assert sp.check_entry("a caf\u00c3\u00a9 by the docks at midnight", ax) == "mojibake"
    assert sp.check_entry("a room", ax) == "too_short"
    assert sp.check_entry("x " * 100, ax) == "too_long"
    assert sp.check_entry("the the the the the the the the the the", ax) == "repetitive"
    assert sp.name_label("Ines Moreau") == "Ines"
    assert sp.name_label("Mrs. Ada Okafor") == "Mrs. Okafor"
    assert sp.name_label("ines moreau") is None and sp.name_label("Cher") is None
    assert sp.check_entry("Nguy\u1ec5n Th\u1ecb Lan", sp.AXES["names"]) is None


def test_pool_prompt_and_parse() -> None:
    ax = sp.AXES["character"]
    p = sp.pool_prompt(ax, ["a baker who never sleeps"], ("Fickle", "Dutiful"))
    assert p.startswith(ax.head) and p.endswith("\u201cfickle\u201d and \u201cdutiful\u201d:\n-")
    assert sp.pool_prompt(ax, ["a baker who never sleeps"], None).endswith("- a baker who never sleeps\n-")
    got = sp.parse_list_completion(" a pilot who hums\n- a tailor who lies.\nNot a bullet\n- after the break", False)
    assert got == ["a pilot who hums", "a tailor who lies"]
    assert sp.parse_list_completion(" a pilot who hums\n- a tailor wh", True) == ["a pilot who hums"]


def test_dissimilarity_drops_paraphrase() -> None:
    texts = ["a lighthouse keeper who counts every ship twice",
             "a lighthouse keeper who counts every ship two times",
             "a pastry chef who hides letters in the bread"]
    kept, dropped = sp.greedy_dissimilar(texts, 0.5)
    assert kept == [0, 2] and dropped[1][0] == 0 and dropped[1][1] >= 0.5
    nn = sp.nearest_neighbour_sims(texts)
    assert len(nn) == 3 and nn[2] < nn[1]


def _build_pools(tmp_path: Path, per_pool: int = 30) -> dict:
    lists = {"traits": ["Fickle", "Dutiful", "Gruff"], "emotions": ["Awe", "Envy"], "bads": ["Cheating", "Haste"],
             "moments": ["Reunion", "Moving"], "antisyzygies": ["old and new"], "illusion_examples": ["Enough"]}
    return asyncio.run(sp.run_pools(fake_pool_completer(), tmp_path, list(sp.AXES), per_pool, lists, {}, 0, {},
                                    names_extra=["Ines Moreau", "Ines Other"], parallel=4))


def test_run_pools(tmp_path: Path) -> None:
    stats = _build_pools(tmp_path)
    for name in sp.AXES:
        rows = sp.read_jsonl(tmp_path / "pools" / f"{name}.jsonl")
        assert len(rows) == 30 and len({r["text"] for r in rows}) == 30
        assert stats[name]["final"] == 30
    assert stats["character"]["reasons"]["mojibake"] > 0 and stats["character"]["reasons"]["too_short"] > 0
    names = sp.read_jsonl(tmp_path / "pools" / "names.jsonl")
    assert names[0]["text"] == "Ines Moreau"  # the provided list comes first
    labels = [sp.name_label(r["text"]) for r in names]
    assert len(set(labels)) == len(labels)  # "Ines Other" was a duplicate label
    crossed = sp.read_jsonl(tmp_path / "pools" / "character.jsonl")
    assert all(r["crossing"] and len(r["crossing"]) == 2 for r in crossed)
    again = asyncio.run(sp.run_pools(None, tmp_path, ["character"], 30, {}, {}, 0, {}))  # resumes: no requests
    assert again["character"] == stats["character"]


def test_combine_balance_and_spread(tmp_path: Path) -> None:
    _build_pools(tmp_path)
    combos, stats = sp.combine(sp.load_pools(tmp_path), 80, seed=1, cast_share=0.3)
    assert len(combos) == 80
    cells = collections.Counter((c["friction"], c["secret_arm"]) for c in combos)
    assert len(cells) == 8 and max(cells.values()) - min(cells.values()) <= 1
    assert stats["arms"]["cast_arm"] == {"pair": 56, "cast": 24}
    for c in combos:
        labels = [ch["label"] for ch in c["characters"]]
        assert len(labels) == (2 if c["cast_arm"] == "pair" else len(labels)) and 2 <= len(labels) <= 4
        assert len(set(labels)) == len(labels)
        assert c["secret_holder"] != c["secret_target"]
    uses = collections.Counter(ch["character"]["id"] for c in combos for ch in c["characters"])
    assert max(uses.values()) - min(uses.values()) <= 6  # least-used first: no entry hogs the draws


def _combo() -> dict:
    def e(axis: str, i: int, text: str) -> dict:
        return {"id": f"{axis}:{i:04d}", "text": text}

    return {"id": "c00000", "friction": "collision", "secret_arm": "plain", "cast_arm": "pair",
            "relationship": e("relationship_pair", 0, "a ferry captain and her estranged brother"),
            "scene": {"theme": e("theme", 0, "maps and the longing for a fixed place"),
                      "setting": e("setting", 0, "a ferry terminal at midnight; a broken vending machine"),
                      "inciting": e("inciting", 0, "the brother has turned up with a suitcase"),
                      "stakes": e("stakes", 0, "the last crossing leaves at half past twelve"),
                      "register": e("register", 0, "a dry comedy that turns")},
            "characters": [{"name": "Mara Okafor", "name_id": "names:0000", "label": "Mara",
                            "character": e("character", 0, "a ferry captain who never misses a departure"),
                            "need": e("need", 0, "to be needed by someone")},
                           {"name": "Teo Okafor", "name_id": "names:0001", "label": "Teo",
                            "character": e("character", 1, "a drummer who lies to be liked"),
                            "need": e("need", 1, "to stop being forgiven in advance")}],
            "secret_holder": 1, "secret_target": 0}


GOOD = (" Teo on the last crossing home tonight, before their mother's house is sold\n"
        "Teo wants: the five hundred pounds Mara keeps in the tin under the counter\n"
        "Teo's secret: the house sold last month and he signed for it\n"
        "Page:\n"
        "Mara Okafor, fifty, ferry captain. She wants Teo on the last crossing home tonight, before their mother's "
        "house is sold.\n"
        "Teo Okafor, thirty-one, a drummer. He wants the five hundred pounds Mara keeps in the tin under the "
        "counter. The house sold last month; he signed for it.\n"
        "Mara is Teo's elder sister.\n\n"
        "A ferry terminal at midnight. The vending machine is broken. The last crossing leaves at half past twelve.\n"
        + sp.END_MARK)


def test_header_prompt_and_parse() -> None:
    c = _combo()
    p = sp.header_prompt(c, 4)
    assert p.startswith(sp.NOTEBOOK_HEAD) and p.endswith("Notes:\nMara wants:")
    assert p.count(sp.END_MARK) == len(sp.HEADER_EXAMPLES)
    assert "Secret: kept by Teo, bearing on what Mara wants" in p
    parsed, why = sp.parse_header_completion(GOOD, ["Mara", "Teo"], 1)
    assert why == "" and parsed["wants"][1].startswith("the five hundred")
    assert parsed["secret"].startswith("the house sold") and parsed["page"].startswith("Mara Okafor, fifty")
    assert sp.gate_page(parsed, c) is None
    assert sp.parse_header_completion(GOOD.replace(sp.END_MARK, ""), ["Mara", "Teo"], 1)[1] == "unterminated"
    assert sp.parse_header_completion(GOOD.replace("Teo wants:", "Teo hopes:"), ["Mara", "Teo"], 1)[1] == \
        "want_missing"


@pytest.mark.parametrize("edit,reason", [
    (lambda p: p + "\n\nMara: You came back.", "speaker_tag"),
    (lambda p: p + "\nSecret: he signed for it.", "label_leak"),
    (lambda p: p.replace("Teo Okafor, thirty-one", "Teo Okafor, two hundred"), "name_duplicated"),
    (lambda p: p.replace("Mara Okafor, fifty", "Mara Okafor, 140"), "impossible_age"),
    (lambda p: p.replace("five hundred pounds Mara keeps in the tin under the counter", "nothing at all"),
     "want_absent"),
    (lambda p: p.replace("Teo", "Theo"), "name_missing"),
])
def test_gate_rules(edit, reason) -> None:
    c = _combo()
    parsed, _ = sp.parse_header_completion(GOOD, ["Mara", "Teo"], 1)
    page = edit(parsed["page"])
    if reason == "name_duplicated":
        page = page + "\nTeo Okafor, again."
    got = sp.gate_page({**parsed, "page": page}, c)
    assert got == reason


def test_gate_catches_copied_example() -> None:
    c = _combo()
    parsed, _ = sp.parse_header_completion(GOOD, ["Mara", "Teo"], 1)
    page = parsed["page"] + "\n" + sp.HEADER_EXAMPLES[0].page
    assert sp.gate_page({**parsed, "page": page}, c) in ("copied_example", "name_duplicated", "page_length")


def test_headers_end_to_end(tmp_path: Path) -> None:
    _build_pools(tmp_path)
    combos, _ = sp.combine(sp.load_pools(tmp_path), 24, seed=2)
    sp.write_jsonl(tmp_path / "combos.jsonl", combos)

    calls = {"n": 0}

    async def complete(prompt: str, n: int, stops: list[str]):
        calls["n"] += 1
        no = prompt.rsplit("No. ", 1)[1]
        people = [ln[len("Person: "):] for ln in no.split("\n") if ln.startswith("Person: ")]
        labels = [sp.name_label(p.split(" \u2014 ")[0]) for p in people]
        holder = no.split("Secret: kept by ")[1].split(",")[0]
        outs = []
        for k in range(n):
            wants = [f"the {w} key from the {w} drawer" for w in ("brass", "iron", "glass", "tin")[:len(people)]]
            page = "\n".join(f"{p.split(' — ')[0]}, forty, {p.split(' — ')[1].split(';')[0]}. "
                             f"{lb} wants {w}." for p, lb, w in zip(people, labels, wants))
            if k == 0 and calls["n"] == 4:
                page += f"\n\n{labels[0]}: hello"  # rejected: speaker tag
            text = (f" {wants[0]}\n" + "".join(f"{lb} wants: {w}\n" for lb, w in zip(labels[1:], wants[1:]))
                    + f"{holder}'s secret: the drawer was never locked\nPage:\n{page}\n\nA quiet room at noon. "
                      f"The bus leaves at one.\n")
            outs.append((text, "stop", False))
        return outs

    async def score(pairs):
        assert all(p == sp.DOC_START for p, _ in pairs)
        return [-(1.0 + len(c) / 1000) for _, c in pairs]

    stats = asyncio.run(sp.run_headers(complete, score, combos, tmp_path, candidates=2, nll_cut=0.25))
    assert stats["combos_with_pass"] == 24 and stats["candidate_verdicts"]["speaker_tag"] == 1
    rows = sp.read_jsonl(tmp_path / "seeds.jsonl")
    assert len(rows) == stats["kept"] == 24 - stats["cut_by_nll"] and stats["cut_by_nll"] >= 1
    for i, r in enumerate(rows):
        s = parse_seed(r, i)
        assert s is not None and s.cast == r["cast"] and s.header == r["header"]
        assert {ch["label"] for ch in r["characters"]} == set(r["cast"])
        assert r["secret"]["holder"] in r["cast"] and r["nll"] is not None
    again = asyncio.run(sp.run_headers(None, score, combos, tmp_path, nll_cut=0.25))  # resume: nothing redrawn
    assert again["candidates"] == stats["candidates"]

    rep = sp.report(tmp_path, sample=5)
    assert rep["seeds"] == len(rows) and rep["entry_reuse"]["character"]["used"] > 0
    assert set(rep["arms"]["secret_arm"]) == {"plain", "hinted"}
    md = (tmp_path / "reading-sample.md").read_text()
    assert md.count("\n## sp-c") == 5
    assert json.loads((tmp_path / "report.json").read_text())["seeds"] == len(rows)


def test_cli_dispatch(tmp_path: Path) -> None:
    _build_pools(tmp_path)
    assert seeds.main(["combine", "--out", str(tmp_path), "--n", "16"]) == 0
    assert len(sp.read_jsonl(tmp_path / "combos.jsonl")) == 16
    with pytest.raises(SystemExit):
        seeds.main(["pools", "--endpoint", "http://x", "--out", str(tmp_path), "--axes", "nonsense"])
