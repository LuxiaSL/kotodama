"""Whole-scene assembly: closing cuts end on a model-written candidate, tails stop at the scene's edge, gates drop or
truncate, scenes stack without repeating a seed, and anthology rows encode with end-of-text between scenes."""
from __future__ import annotations

import json

import pytest

from kotodama.posttrain import anthology as an
from kotodama.posttrain import compliance, loom
from kotodama.posttrain import transcript as tx
from kotodama.posttrain.chat_data import IGNORE_INDEX
from kotodama.serve.chatml import EOS_TOKEN_ID

TALK = ["The shipment never came.", "Then who signed for it?", "Your brother did, at six.", "He was in Lisbon.",
        "Somebody used his name.", "We should call the depot.", "I already did. Nobody answers.",
        "Then we drive there tonight.", "It's late. I'm tired.", "Fine, tomorrow at dawn.", "Good night, Teo.",
        "Good night.", "Bye.", "Love you.", "Love you too.", "Bye then.", "Goodnight.", "Night."]


def _path(texts, leaf_prefix="t0/r", state="depth", seed="p1-00001~t0", cast=("A", "B"), header="Two siblings."):
    turns = [{"speaker": cast[i % len(cast)], "text": x} for i, x in enumerate(texts)]
    leaf = leaf_prefix + "".join(".0" for _ in texts)
    return {"turns": turns, "header": header, "tree": leaf_prefix.split("/")[0], "seed": seed, "leaf": leaf,
            "state": state, "depth": len(texts), "cast": list(cast), "n_seed": 0}


def _nodes(path, eos_after=None, ending="Sleep well, A."):
    """A NodeIndex with 8 candidates after every turn; ``eos_after`` {turn index: eos count} draws that many
    end-of-text candidates there (for the speaker after that turn)."""
    idx = an.NodeIndex()
    ids = loom.node_ids(path["leaf"])
    cast = path["cast"]
    for k, nid in enumerate(ids):
        e = (eos_after or {}).get(k, 0)
        idx.counts[nid] = (8, e)
        if e:
            nxt = cast[(k + 1) % len(cast)]
            idx.eos[nid] = [{"id": f"{nid}.e{i}", "speaker": nxt, "text": ending, "vetoes": [], "status": "unchosen"}
                            for i in range(e)]
    return idx


def test_closing_episode_cut_ends_on_a_candidate_ending() -> None:
    p = _path(TALK)
    nodes = _nodes(p, {11: 3, 12: 1})
    s, rec = an.clean_scene(p, nodes, an.SceneConfig(cut_closings=0.125), {})
    assert rec["closing"]["kind"] == "episode" and rec["closing"]["after"] == 11
    assert [t["text"] for t in s["turns"]][-2:] == ["Good night.", "Sleep well, A."]
    assert s["ended"] is True and s["turns"][-1]["speaker"] == "A"


def test_no_cut_without_end_of_text_evidence_or_when_little_would_go() -> None:
    p = _path(TALK)
    s, rec = an.clean_scene(p, _nodes(p), an.SceneConfig(cut_closings=0.125), {})
    assert "closing" not in rec and len(s["turns"]) == len(TALK) and s["ended"] is False
    short = _path(TALK[:14])  # the peak at turn 11 would remove only one turn
    _, rec = an.clean_scene(short, _nodes(short, {11: 3}), an.SceneConfig(cut_closings=0.125), {})
    assert "closing" not in rec


def test_sustained_orbit_is_cut_at_its_onset() -> None:
    texts = TALK[:8] + ["Bye.", "Bye, bye.", "Goodnight.", "Love you.", "Love you too.", "Bye.", "Night.", "Bye.",
                        "Goodnight.", "Love you.", "Bye."]
    p = _path(texts, cast=("A", "B", "C", "D"))
    s, rec = an.clean_scene(p, _nodes(p), an.SceneConfig(cut_closings=0.25), {})
    assert rec["closing"]["kind"] == "orbit" and rec["closing"]["ending"] is False
    assert s["ended"] is False and len(s["turns"]) == 9
    # an end-of-text candidate drawn at the onset that is no farewell is not spliced in
    s, rec = an.clean_scene(p, _nodes(p, {8: 1}, ending="My mother will take care of it."),
                            an.SceneConfig(cut_closings=0.25), {})
    assert rec["closing"]["ending"] is False and s["ended"] is False and len(s["turns"]) == 9


def test_love_you_without_a_parting_is_not_a_closing() -> None:
    texts = TALK[:10] + ["I love you.", "I love you too.", "I have to go.", "Then go to Lisbon.", "The depot?",
                         "Yes, the depot.", "Fine."]
    p = _path(texts)
    _, rec = an.clean_scene(p, _nodes(p, {11: 4}), an.SceneConfig(cut_closings=0.125), {})
    assert "closing" not in rec
    assert an.is_farewell("I have to go.", ["A"]) and not an.is_farewell("We have to go in there.", ["A"])


def test_a_cut_before_the_episode_needs_a_farewell_ending() -> None:
    p = _path(TALK)
    _, rec = an.clean_scene(p, _nodes(p, {9: 4}, ending="Why would I?"), an.SceneConfig(cut_closings=0.125), {})
    assert "closing" not in rec  # turn 9 is before the first farewell (10) and the ending is no farewell
    s, rec = an.clean_scene(p, _nodes(p, {9: 4}, ending="Good night, then."), an.SceneConfig(cut_closings=0.125), {})
    assert rec["closing"]["after"] == 9 and s["turns"][-1]["text"] == "Good night, then."


def test_farewell_does_not_match_take_care_of() -> None:
    assert not an.is_farewell("I'm so glad to see you.", ["A"]) and an.is_farewell("See you.", ["A"])
    assert not an.is_farewell("I have to take care of you, Marie.", ["A"])
    assert an.is_farewell("Take care.", ["A"]) and an.is_farewell("Good night, Teo.", ["A"])


@pytest.mark.parametrize("text,kept,why", [
    ("So long.\n\nShe walks to the car and drives away.", "So long.\n\nShe walks to the car and drives away.", None),
    ("So long.\n\nThe end.\n\nThe film was not well received.", "So long.", "doc_marker"),
    ("So long.\n\n--\nThis email has been scanned.", "So long.", "doc_marker"),
    ("So long.\n\nRead the novel's sequel at https://example.org", "So long.", "out_of_frame"),
    ("So long.\n\nMara Lind, thirty-two, a pilot\nTeo Lind, twenty", "So long.", "cast_list"),
    ("So long.\n\nKai: who are you?", "So long.", "foreign_label"),
    ("So long.\n\n[Footnote: from the case files.]", "So long.", "doc_marker"),
    ("So long.\n\n[*] What is going to happen next?", "So long.", "doc_marker"),
    ("So long.\n\n[She leaves the room.]", "So long.\n\n[She leaves the room.]", None),
    ("So long.\n\n" + "He walks. " * 30 + "\n\n" + "She waits. " * 40, "So long.\n\n" + ("He walks. " * 30).strip(),
     "long_narration"),
])
def test_clean_tail(text, kept, why) -> None:
    assert an.clean_tail(text, ["Mara", "Teo"]) == (kept, why)


def test_frame_gate_truncates_and_short_scenes_drop() -> None:
    texts = TALK[:9] + ["The novel's author wrote this chapter in 1990."] + TALK[9:12]
    p = _path(texts)
    s, rec = an.clean_scene(p, None, an.SceneConfig(frame_gate=True), {})
    assert rec["frame_cut"] == 9 and len(s["turns"]) == 9 and s["ended"] is False
    s, rec = an.clean_scene(p, None, an.SceneConfig(frame_gate=True, min_scene_turns=10), {})
    assert s is None and rec["drop"] == "short"


def test_natural_ending_keeps_its_speech_when_only_the_tail_leaves_the_scene() -> None:
    p = _path(TALK[:9] + ["So long.\n\nRead the novel's sequel at https://example.org"], state="eos")
    s, rec = an.clean_scene(p, None, an.SceneConfig(frame_gate=True, clean_tail=True), {})
    assert "frame_cut" not in rec and rec["tail_cut"] == "out_of_frame"
    assert s["ended"] is True and s["turns"][-1]["text"] == "So long."
    p = _path(TALK[:9] + ["The novel's author wrote this chapter in 1990."], state="eos")
    s, rec = an.clean_scene(p, None, an.SceneConfig(frame_gate=True, clean_tail=True), {})
    assert rec["frame_cut"] == 9 and s["ended"] is False and len(s["turns"]) == 9


def test_a_repeated_farewell_that_ends_the_scene_still_trains() -> None:
    p = _path(TALK[:12] + ["Good night."], state="eos")  # repeats turn 11
    s, _ = an.clean_scene(p, None, an.SceneConfig(untrain_repeats=20), {})
    assert s["turns"][-1]["train"] is True
    s, _ = an.clean_scene(dict(p, state="depth"), None, an.SceneConfig(untrain_repeats=20), {})
    assert s["turns"][-1]["train"] is False


def test_natural_ending_trains_and_shared_opening_trains_once() -> None:
    a = _path(TALK[:10], state="eos")
    b = dict(_path(TALK[:3] + ["Different now.", "Yes.", "Then go.", "I will.", "Good.", "Fine.", "Done."]),
             leaf="t0/r.0.0.0.1.0.0.0.0.0.0")
    uses: dict[str, int] = {}
    sa, _ = an.clean_scene(a, None, an.SceneConfig(), uses)
    sb, rec = an.clean_scene(b, None, an.SceneConfig(), uses)
    assert sa["ended"] is True and all(t["train"] for t in sa["turns"])
    assert [t["train"] for t in sb["turns"][:4]] == [False, False, False, True] and rec["untrained_shared"] == 3


def test_compliance_blocklist_drops_and_pii_is_redacted() -> None:
    p = _path(TALK[:9] + ["Mail me at someone@example.com"])
    cfg = an.SceneConfig(blocklist=frozenset({compliance.turn_hash(TALK[2])}))
    assert an.clean_scene(p, None, cfg, {})[1]["drop"] == "blocklist"
    s, rec = an.clean_scene(p, None, an.SceneConfig(redact_pii=True), {})
    assert s["turns"][-1]["text"] == "Mail me at <email>" and rec["pii_redacted"] == 1


def test_stack_never_repeats_a_seed_and_respects_the_budget() -> None:
    scenes = []
    for i in range(30):
        s, _ = an.clean_scene(_path(TALK[:10], leaf_prefix=f"t{i}/r", seed=f"p1-{i % 5:05d}~t{i // 5}"), None,
                              an.SceneConfig(), {})
        scenes.append(s)
    docs = an.stack(scenes, max_chars=1000, seed=0)
    assert sum(len(d["scenes"]) for d in docs) == 30
    for d in docs:
        keys = [an.seed_key(s["source"]["seed"]) for s in d["scenes"]]
        assert len(keys) == len(set(keys))
        assert len(d["scenes"]) == 1 or sum(an.scene_chars(s) for s in d["scenes"]) <= 1000
    assert max(len(d["scenes"]) for d in docs) > 1


def test_assemble_main_anthology_end_to_end(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    paths = [_path(TALK, leaf_prefix="t0/r", seed="p1-00001~t0"),
             _path(TALK[:10], leaf_prefix="t1/r", state="eos", seed="p1-00002~t0")]
    nodes = []
    for p in paths:
        idx = _nodes(p, {11: 2} if p["tree"] == "t0" else {})
        for par, (n, e) in idx.counts.items():
            cands = idx.eos.get(par, [])
            nodes += [{"tree": p["tree"], "parent": par, "ended_by": "eos", **c} for c in cands]
            nodes += [{"tree": p["tree"], "parent": par, "id": f"{par}.s{i}", "ended_by": "stop", "text": "x"}
                      for i in range(n - e)]
    (run / "paths.jsonl").write_text("".join(json.dumps(p) + "\n" for p in paths))
    (run / "nodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in nodes))
    (run / "trees.jsonl").write_text("".join(json.dumps({"tree": p["tree"], "seed": p["seed"]}) + "\n" for p in paths))
    out, scenes = tmp_path / "docs.jsonl", tmp_path / "scenes.jsonl"
    assert loom.main(["assemble", str(run), "--out", str(out), "--anthology", "--cut_closings", "0.125",
                      "--clean_tail", "--min_scene_turns", "8", "--scenes_out", str(scenes)]) == 0
    docs = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(docs) == 1 and len(docs[0]["scenes"]) == 2
    assert all(s["ended"] for s in docs[0]["scenes"])
    assert all(tx.parse_row(json.loads(x)) is not None for x in scenes.read_text().splitlines())


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    from kotodama.presets import TOKENIZER_NAME

    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except OSError as e:
        pytest.skip(f"tokenizer unavailable: {e}")


def test_cut_scene_keeps_end_of_text_as_context(tok) -> None:
    tr = tx.Transcript((tx.Turn("A", "one"), tx.Turn("B", "two")), "Header.", ended=False)
    ex = tx.encode([tr], tok, 512, EOS_TOKEN_ID)[0]
    assert ex.input_ids[-1] == EOS_TOKEN_ID and ex.labels[-1] == IGNORE_INDEX
    assert tok.decode([t for t, l in zip(ex.input_ids, ex.labels) if l != IGNORE_INDEX]).endswith("B: two")


def test_anthology_row_is_its_scenes_back_to_back(tok) -> None:
    s1 = {"header": "First.", "turns": [{"speaker": "A", "text": "hi"}, {"speaker": "B", "text": "bye"}]}
    s2 = {"header": "Second.", "turns": [{"speaker": "C", "text": "go"}, {"speaker": "D", "text": "no"}],
          "ended": False}
    doc = tx.parse_document({"scenes": [s1, s2]})
    assert isinstance(doc, tx.Anthology)
    ex = tx.encode([doc], tok, 512, EOS_TOKEN_ID)[0]
    one = tx.encode([tx.parse_row(s1)], tok, 512, EOS_TOKEN_ID)[0]
    assert ex.input_ids[:len(one.input_ids)] == one.input_ids and ex.labels[:len(one.labels)] == one.labels
    assert ex.input_ids.count(EOS_TOKEN_ID) == 2 and ex.labels[len(one.input_ids) - 1] == EOS_TOKEN_ID
    assert ex.labels[-1] == IGNORE_INDEX
    trained = tok.decode([t for t, l in zip(ex.input_ids, ex.labels) if l != IGNORE_INDEX])
    assert "First." not in trained and "Second." not in trained and "D: no" in trained
    assert tx.parse_document({"scenes": [s1, {"turns": []}]}) is None
