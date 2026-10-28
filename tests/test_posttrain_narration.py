"""posttrain.narration: narrator paragraphs vs a speaker still talking across a blank line."""

from __future__ import annotations

import pytest

from kotodama.posttrain.narration import classify_paragraph, out_of_frame, self_described, split_turn

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


@pytest.mark.parametrize("text,off", [
    ("I need to change my shirt.", False),
    ("She sets the cup down. The kettle is still warm.", False),
    ("I loved the novel, honestly. You should read it.", False),  # a character talking about a book is speech
    ("The movie starts at nine.", False),  # one weak cue
    ("I was afraid of it. The most interesting character is a young woman. In a scene that punctuates the novel's "
     "beginning, she laments a man in a white labcoat.", True),
    ("The story drags in the middle, and the characters never change.", True),  # two weak cues
    ("Posted by admin on March 3.", True),
    ("Rating: 4/5. Spoilers below.", True),
])
def test_out_of_frame(text: str, off: bool) -> None:
    assert out_of_frame(text) is off


def test_self_described() -> None:
    bio = ("Nick Olsson, as the leader of the left faction, calls for all three to be expelled from the party. He is a "
           "fifty-five-year-old member of parliament.")
    assert self_described(bio, "Nick")
    assert self_described("Bonnie is a twenty-nine-year-old member of parliament for an area in the Northwest, and "
                          "she believes Nick should resign.", "Bonnie")
    assert not self_described(bio, "Bonnie")  # another character's name: narration about someone else
    assert not self_described("Nick, I am telling you, the party will split if we expel all three of them now.", "Nick")
    assert not self_described('"Nick is a fool," he says, and the whole room laughs at the long old joke again.', "Nick")
    assert not self_described("Nick is late.", "Nick")  # too short to be a description
    assert not self_described("Fine.\n\nNick Olsson, the party leader, walks out of the room without another word "
                              "to any of them.", "Nick")  # narration after speech is not read
