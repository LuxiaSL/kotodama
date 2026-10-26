"""Self-copy detector: loops and template loops are flagged; binding (reusing names, objects, short phrases) and code
identifiers are not; terse answers are never 'desirable'."""
from __future__ import annotations

import json

from kotodama.posttrain import selfcopy
from kotodama.posttrain.selfcopy import SelfCopyConfig, label, score

BINDING = (
    "Mara gave Teo the brass key. Teo put the brass key in his coat and walked to the station. At the station, "
    "Mara asked Teo whether he still had the key. Teo checked his coat: the brass key was there. He handed the key "
    "back to Mara, and Mara put it in her bag next to the letter. Later, at the café, Teo asked for the letter, "
    "but Mara kept the letter and the key in her bag until the train left."
)

LOOP = (
    "She went out and looked at the window from the porch. The violin was gone, but the telescope was still there. "
    "She went out and looked at the window from the porch. The violin was gone, but the telescope was still there. "
    "She went out and looked at the window from the porch. The violin was gone, but the telescope was still there."
)

TEMPLATE = (
    "Day one. I woke up early and walked down to the river to look for the boat. Nothing there. "
    "Day two. I woke up early and walked down to the river to look for the dog. Nothing there. "
    "Day three. I woke up early and walked down to the river to look for the girl. Nothing there. "
    "Day four. I woke up early and walked down to the river to look for the man. Nothing there."
)


def test_binding_text_is_clean_and_desirable() -> None:
    r = score(BINDING)
    assert r.copied_frac == 0.0 and r.longest_span == 0
    assert label(r) is True


def test_verbatim_loop_is_undesirable() -> None:
    r = score(LOOP)
    assert r.copied_frac > 0.6 and r.longest_span >= 24
    assert label(r) is False


def test_template_loop_is_flagged_despite_swapped_nouns() -> None:
    r = score(TEMPLATE)
    assert r.template_repeats >= 2
    assert label(r) is False


def test_self_referencing_run_counts() -> None:
    # "a b c d a b c d a b c d ..." copies from an overlapping source
    words = " ".join(["one two three four"] * 8)
    assert score(words, cfg=SelfCopyConfig(min_span=6)).longest_span >= 24


def test_context_is_own_text_but_prompt_is_not() -> None:
    para = "The keeper climbed the stairs every evening to light the lamp above the black rocks of the bay."
    fresh = ("Nobody in the village had seen him since the storm, though his boots still stood by the door and "
             "the kettle was warm when the girl looked in on Sunday morning.")
    as_context = score(para + " " + fresh, context=para)
    as_prompt = score(para + " " + fresh, prompt=para)
    assert as_context.copied_frac > 0.3
    assert as_prompt.copied_frac == 0.0 and as_prompt.prompt_copied_frac > 0.3


def test_code_fences_are_ignored() -> None:
    code = "```python\n" + "\n".join(f"x = compute_value(alpha, beta, gamma, delta)  # {i}" for i in range(10)) + "\n```"
    r = score(BINDING + "\n" + code)
    assert r.copied_frac == 0.0


def test_terse_clean_answer_is_not_desirable() -> None:
    r = score("Yes, I think so.")
    assert r.copied_frac == 0.0 and label(r) is None


def test_cli_label(tmp_path) -> None:
    src = tmp_path / "s.jsonl"
    rows = [{"prompt": "Once upon a time", "completion": BINDING}, {"prompt": "x", "completion": LOOP},
            {"prompt": "y", "completion": "Fine."}]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "kto.jsonl"
    assert selfcopy.main(["label", "--in", str(src), "--out", str(out)]) == 0
    got = [json.loads(x) for x in out.read_text().splitlines()]
    assert [g["label"] for g in got] == [True, False]
    assert "copied_frac" in got[0]["selfcopy"]
