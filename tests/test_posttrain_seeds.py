"""Self-generated seeds: the catalogue prompt, entry parsing, de-duplication."""
from __future__ import annotations

import asyncio

from kotodama.posttrain import seeds


def test_catalogue_prompt_ends_open() -> None:
    p = seeds.catalogue_prompt(["Participants: A, x; B, y\nSetting: a room somewhere"], 0)
    assert p.startswith(seeds.FRAME_HEAD) and p.endswith("Transcript 2\nParticipants:")


def test_parse_entry() -> None:
    e = seeds.parse_entry(" June, a ferry captain; Old Tom, her deckhand\nSetting: the wheelhouse in fog")
    assert e == {"cast": ["June", "Old Tom"],
                 "header": "Participants: June, a ferry captain; Old Tom, her deckhand\nSetting: the wheelhouse in fog"}
    assert seeds.parse_entry(" June, a captain\nSetting: the wheelhouse in fog") is None  # one person
    assert seeds.parse_entry(" June, a; june lower, b\nSetting: the wheelhouse in fog") is None  # bad name
    assert seeds.parse_entry(" June, a; Tom, b") is None  # no setting
    assert seeds.parse_entry(" A, x; B, y; C, z; D, w\nSetting: a crowded room at noon") is None  # too many


def test_generate_dedupes() -> None:
    dup = (" June, a captain; Tom, a deckhand\nSetting: the wheelhouse in fog\n\nTranscript 9", "stop", False)
    other = (" Ada, a baker; Lee, a courier\nSetting: a bakery before dawn opens", "stop", False)

    async def complete(prompt, n, stops):
        return [dup, dup, other, (" garbage", "stop", False)]

    out = asyncio.run(seeds.generate(complete, 5, list(seeds.DEFAULT_EXAMPLES), 0))
    assert [s["cast"] for s in out] == [["June", "Tom"], ["Ada", "Lee"]]
    assert out[0]["id"] == "seed000000"
