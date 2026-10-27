"""Pinned bindprobes files render byte-identical prompts.

A bindprobes item file stores specs, and the prompts are rendered at scoring time, so the file's sha256 pin
holds only together with a render pin: ``tests/data/bindprobes_render_golden.json.gz`` keeps, per item file,
item and condition, the sha256 of the prompt rendered by the code that generated the file
(``tests/data/make_bindprobes_render_golden.py``). Its files are every registry bindprobes battery plus scored
item files no battery pins (``registered: false``), so every generation of every family is in it. Every family
renders an item with the templates of the generation that built it (``base.version_of``), so a new wording
never rewrites the strings of a published grid.

- the golden covers every bindprobes battery in the registry, with the registry's file and render pins;
- a stratified sample of every file (``bindprobes_render_sample.jsonl.gz``: every family, generation, cell
  and variant) renders to its golden digest in every condition, on any machine;
- with ``$KOTODAMA_BINDING_PROMPTS`` set, every item of every golden file present there (at the golden's
  relative path and sha256) does, and ``batteries.check_render`` accepts each registry file.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from kotodama.eval.binding import batteries as bt
from kotodama.eval.bindprobes.base import Item
from kotodama.eval.bindprobes.items import CONDITIONS, build_prompt, load_items, render_sha256

DATA = Path(__file__).parent / "data"
GOLDEN: dict[str, Any] = json.loads(gzip.decompress((DATA / "bindprobes_render_golden.json.gz").read_bytes()))
HEX = int(GOLDEN["digest_hex"])
SAMPLE = [json.loads(ln) for ln in
          gzip.decompress((DATA / "bindprobes_render_sample.jsonl.gz").read_bytes()).decode().splitlines()]
PINNED = [b for b in bt.BATTERIES.values() if b.kind == "bindprobes"]


def _golden_digest(name: str, cond: str, index: int) -> str:
    s = GOLDEN["batteries"][name]["conditions"][cond]["items"]
    return s[index * HEX:(index + 1) * HEX]


def _digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:HEX]


def test_golden_covers_every_pinned_battery():
    registered = {n for n, g in GOLDEN["batteries"].items() if g["registered"]}
    assert registered == {b.name for b in PINNED}
    for b in PINNED:
        g = GOLDEN["batteries"][b.name]
        assert (g["file"], g["sha256"]) == (b.file, b.sha256), b.name
        assert b.render_sha256 == g["render_sha256"], b.name
    for name, g in GOLDEN["batteries"].items():
        for c in CONDITIONS:
            assert len(g["conditions"][c]["items"]) == g["n_items"] * HEX, (name, c)


def test_sample_covers_every_file_family_and_generation():
    by_file: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for r in SAMPLE:
        by_file[r["battery"]].add((r["item"]["family"], r["item"]["item_id"].split("_")[0]))
    assert set(by_file) == set(GOLDEN["batteries"])
    tags = {t for v in by_file.values() for _, t in v}
    # every generation each family has written (codevar and codevarmut have one wording)
    for tag in ("attr", "attr-v2", "dialogue", "dialogue-v2", "deixis-v2", "deixis-v3", "setbind-v1",
                "setbind-v2", "setbind-v3", "answer-v2", "answer-v3", "adjacency-v1", "adjacency-v2",
                "adjacency-v3", "codevar", "codevarmut-v1"):
        assert tag in tags, tag


@pytest.mark.parametrize("cond", CONDITIONS)
def test_sample_renders_as_golden(cond):
    bad = []
    for r in SAMPLE:
        it = Item.from_dict(r["item"])
        try:
            got = _digest(build_prompt(it, cond))
        except Exception as exc:   # an old item the current code cannot render is a failure, not an error
            got = f"{type(exc).__name__}: {exc}"
        if got != _golden_digest(r["battery"], cond, r["index"]):
            bad.append(f"{r['battery']}:{it.item_id}: {got}")
    assert not bad, f"{len(bad)}/{len(SAMPLE)} sampled prompts differ from the golden: {bad[:5]}"


@pytest.mark.parametrize("name", sorted(GOLDEN["batteries"]))
def test_file_renders_as_golden(name):
    root = os.environ.get(bt.PROMPTS_ENV)
    g = GOLDEN["batteries"][name]
    path = Path(root) / g["file"] if root else None
    if path is None or not path.exists():
        pytest.skip(f"${bt.PROMPTS_ENV} does not hold {g['file']}")
    if bt.sha256(path) != g["sha256"]:
        if g["registered"]:
            pytest.fail(f"{path}: sha256 differs from {name}'s pin")
        pytest.skip(f"{path} is another file than {name}'s")
    its = load_items(path)
    assert len(its) == g["n_items"]
    for c in CONDITIONS:
        whole = hashlib.sha256()
        bad = []
        for i, it in enumerate(its):
            try:
                d = hashlib.sha256(build_prompt(it, c).encode("utf-8"))
            except Exception as exc:
                bad.append(f"{it.item_id}: {type(exc).__name__}: {exc}")
                continue
            whole.update(d.digest())
            if d.hexdigest()[:HEX] != _golden_digest(name, c, i):
                bad.append(it.item_id)
        assert not bad, f"{name}/{c}: {len(bad)}/{len(its)} prompts differ: {bad[:5]}"
        assert whole.hexdigest() == g["conditions"][c]["all"], (name, c)
    assert render_sha256(its) == g["render_sha256"]
    if g["registered"]:
        bt.check_render(bt.BATTERIES[name], path)


def test_check_render_refuses_changed_strings(tmp_path):
    """A file whose strings are not its battery's is refused, whatever its file pin says."""
    b = bt.BATTERIES["bp-v3.2"]
    rows = [r["item"] for r in SAMPLE if r["battery"] == b.name][:4]
    path = tmp_path / "items.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(bt.PinError, match="render pin"):
        bt.check_render(b, path)
    rows[0]["spec"] = {}                              # a spec the renderer cannot read
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(bt.PinError, match="does not render"):
        bt.check_render(b, path)
    bt.check_render(bt.BATTERIES["joinlite-v1"], path)   # prompt-row batteries store their strings
