"""posttrain.narration: narrator paragraphs vs a speaker still talking across a blank line."""

from __future__ import annotations

import pytest

from kotodama.posttrain.narration import classify_paragraph, split_turn

CAST = ["Mara", "Teo"]


@pytest.mark.parametrize("para", [
    "(The wind gets worse. She pulls her coat tighter.)",
    "[They enter the pub.]",
    "*sits down at the table*",
    "As he leaves, the sky turns clouded and the usual foreboding settles over the town.",
    "Years later, Mara still kept the watch in a drawer.",
    "Meanwhile the tide crept up the beach.",
    "Teo walked to the window and looked out at the harbour.",
])
def test_narration(para: str) -> None:
    assert classify_paragraph(para, "I'm not going.", CAST)[0] == "narr"


@pytest.mark.parametrize("para,prev", [
    ("“And another thing — you never asked me.”", "I'm leaving."),
    ("I don't know why I bother with you.", "Fine."),
    ("Teo, listen to me.", "Stop."),
    ("2. bring the lantern", "1. check the rope"),
    ("and then we walked all the way home", "We left at six,"),
    ("Do you even remember the lighthouse?", "It was years ago."),
])
def test_speech_continuation(para: str, prev: str) -> None:
    assert classify_paragraph(para, prev, CAST)[0] == "speech"


def test_unclear_and_split() -> None:
    assert classify_paragraph("", "x")[0] == "unclear"
    assert classify_paragraph("(I think) you should go.", "", CAST)[0] in ("unclear", "speech")
    assert split_turn("one\n\n\n two \n \n three") == ["one", "two", "three"]
