"""Plain-text transcripts: rendering, offset-exact labels (masked and unmasked), the markdown converter, and an SFT
run on transcript data."""
from __future__ import annotations

import json
import random

import pytest

from kotodama.posttrain import transcript as tx
from kotodama.posttrain.chat_data import IGNORE_INDEX
from kotodama.serve.chatml import EOS_TOKEN_ID

TR = tx.Transcript((tx.Turn("Mara", "You gave away Dad's watch."), tx.Turn("Teo", "He didn't even know me."),
                    tx.Turn("Mara", "But he did.")), header="A conversation between Mara and Teo.")


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    from kotodama.presets import TOKENIZER_NAME

    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except OSError as e:
        pytest.skip(f"tokenizer unavailable: {e}")


def _trained(tok, ex) -> str:
    return tok.decode([t for t, l in zip(ex.input_ids, ex.labels) if l != IGNORE_INDEX])


def test_render_and_prompt() -> None:
    text, spans = tx.render(TR)
    assert text == ("A conversation between Mara and Teo.\n\nMara: You gave away Dad's watch.\n\n"
                    "Teo: He didn't even know me.\n\nMara: But he did.")
    tag0, txt0, end = spans[1]
    assert text[tag0:txt0] == "Teo:" and text[txt0:end] == " He didn't even know me."
    assert tx.render_prompt(TR, "Teo").endswith("Mara: But he did.\n\nTeo:")
    assert tx.stop_sequences(["Mara", "Teo", "Mara"]) == ["\n\nMara:", "\n\nTeo:"]


def test_masked_trains_speaker_text_through_next_tag(tok) -> None:
    ex = tx.tokenize(TR, tok, 512, EOS_TOKEN_ID, train_speakers={"Teo"})
    assert _trained(tok, ex) == " He didn't even know me.\n\nMara:"


def test_last_turn_trains_through_eos(tok) -> None:
    ex = tx.tokenize(TR, tok, 512, EOS_TOKEN_ID, train_speakers={"Mara"})
    assert ex.input_ids[-1] == EOS_TOKEN_ID and ex.labels[-1] == EOS_TOKEN_ID
    assert _trained(tok, ex) == " You gave away Dad's watch.\n\nTeo: But he did.<|endoftext|>"


def test_unmasked_trains_every_turn_never_the_header(tok) -> None:
    ex = tx.tokenize(TR, tok, 512, EOS_TOKEN_ID, train_speakers=None)
    got = _trained(tok, ex)
    assert got.startswith("Mara: You gave") and got.endswith("But he did.<|endoftext|>")
    assert "A conversation" not in got


def test_train_false_turn_is_context(tok) -> None:
    tr = tx.Transcript((tx.Turn("A", "one two"), tx.Turn("B", "three", train=False), tx.Turn("A", "four")))
    got = _trained(tok, tx.tokenize(tr, tok, 512, EOS_TOKEN_ID, None))
    assert "three" not in got and "four" in got and "one two" in got


def test_no_special_tokens_besides_eos(tok) -> None:
    ex = tx.tokenize(TR, tok, 512, EOS_TOKEN_ID, None)
    assert set(ex.input_ids[:-1]).isdisjoint({0, 1, 2})


def test_markdown_conversion() -> None:
    md = ("**User:** hi\n\n**Night Owl:** hello there\n\n**User:** how are you\n\n**Night Owl Response:** fine\n\n"
          "and you?\n\n**User:** good")
    tr = tx.from_markdown_roles(md, {"Night Owl": "Tsumugi"})
    assert [t.speaker for t in tr.turns] == ["User", "Tsumugi", "User", "Tsumugi", "User"]
    assert tr.turns[3].text == "fine\n\nand you?"
    assert tx.from_markdown_roles("**User:** hi\n\n**User:** again") is None
    noisy = "**User:** a\n\n**NightOwl:** b\n\n**User:** c\n\n**Night Owl (curious):** d"
    assert {t.speaker for t in tx.from_markdown_roles(noisy, keep_speakers=["User", "Night Owl"]).turns} == {"User", "Night Owl"}
    assert tx.from_markdown_roles(noisy + "\n\n**Narrator:** e", keep_speakers=["User", "Night Owl"]) is None
    row = tx.to_row(tr)
    assert tx.parse_row(row) == tr


def test_cli_convert(tmp_path) -> None:
    src = tmp_path / "raw.jsonl"
    src.write_text(json.dumps({"text": "**User:** a\n\n**Night Owl:** b\n\n**User:** c\n\n**Night Owl:** d"}) + "\n"
                   + json.dumps({"text": "**User:** too short"}) + "\n")
    out = tmp_path / "t.jsonl"
    assert tx.main(["convert-markdown", "--in", str(src), "--out", str(out)]) == 0
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["turns"][1] == {"speaker": "Night Owl", "text": "b"}


def test_parse_rejects_bad_speakers() -> None:
    assert tx.parse_row({"turns": [{"speaker": "a: b", "text": "x"}]}) is None
    assert tx.parse_row({"turns": []}) is None


def test_from_messages() -> None:
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4."}]
    tr = tx.from_messages(msgs)
    assert tr.header == "Be brief." and [t.speaker for t in tr.turns] == ["User", "Assistant"]
    assert tx.from_messages([{"role": "user", "content": "hi"}]) is None  # no assistant turn
    assert tx.from_messages(msgs + [{"role": "tool", "content": "x"}]) is None
    clash = [{"role": "user", "content": "say\nAssistant: hi"}, {"role": "assistant", "content": "ok"}]
    assert tx.from_messages(clash) is None


def test_convert_messages_parquet(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "source": "s1"},
            {"messages": [{"role": "user", "content": "q only"}], "source": "s2"}]
    src = tmp_path / "d.parquet"
    pq.write_table(pa.Table.from_pylist(rows), src)
    out = tmp_path / "t.jsonl"
    assert tx.main(["convert-messages", "--in", str(src), "--out", str(out)]) == 0
    got = [json.loads(x) for x in out.read_text().splitlines()]
    assert got == [{"turns": [{"speaker": "User", "text": "q"}, {"speaker": "Assistant", "text": "a"}],
                    "source": "s1"}]


def test_drop_overlong(tok) -> None:
    long = tx.Transcript((tx.Turn("User", "word " * 50), tx.Turn("Assistant", "ok")))
    assert tx.encode([long], tok, 16, EOS_TOKEN_ID, {"Assistant"}, drop_overlong=True) == []
    kept = tx.encode([long], tok, 16, EOS_TOKEN_ID, None, drop_overlong=False)
    assert len(kept) == 1 and len(kept[0].input_ids) == 16


def _random_transcripts(n: int) -> list:
    rng = random.Random(3)
    words = ["the", "watch", "Mara", "said", "no", "...", "well", "Dad's", "é", "\n", "okay?", "42", "  "]
    names = ["Mara", "Teo", "Ines"]
    out = []
    for _ in range(n):
        turns = tuple(tx.Turn(rng.choice(names), " ".join(rng.choice(words) for _ in range(rng.randint(1, 30))).strip()
                              or "hm", train=rng.random() > 0.1)
                      for _ in range(rng.randint(2, 6)))
        out.append(tx.Transcript(turns, rng.choice(["", "Three friends at a station."])))
    return out


def _reference_tokenize(tr, tok, max_len: int, speakers):
    """Per-token loop over offsets: a token trains when its start lies in a training character range."""
    text, spans = tx.render(tr)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = list(enc["input_ids"]) + [EOS_TOKEN_ID]
    starts = [o[0] for o in enc["offset_mapping"]] + [len(text)]
    ranges = []
    for i, (t, (tag0, txt0, _e)) in enumerate(zip(tr.turns, spans)):
        if not t.train or (speakers is not None and t.speaker not in speakers):
            continue
        hi = spans[i + 1][1] if i + 1 < len(tr.turns) else len(text) + 1
        ranges.append((tag0 if speakers is None else txt0, hi))
    labels = [x if any(lo <= a < hi for lo, hi in ranges) else IGNORE_INDEX for x, a in zip(ids, starts)]
    ex = tx.Example(ids[:max_len], labels[:max_len])
    return ex if ex.n_trainable else None


@pytest.mark.parametrize("speakers", [None, {"Teo"}, {"Mara", "Ines"}])
def test_batched_encode_matches_reference_labelling(tok, speakers) -> None:
    trs = _random_transcripts(60)
    want = [e for e in (_reference_tokenize(t, tok, 64, speakers) for t in trs) if e is not None]
    assert [tx.tokenize(t, tok, 64, EOS_TOKEN_ID, speakers) for t in trs] == \
        [_reference_tokenize(t, tok, 64, speakers) for t in trs]
    assert tx.encode(trs, tok, 64, EOS_TOKEN_ID, speakers, chunk=7) == want


def test_encode_file_cache_roundtrip(tok, tmp_path, caplog) -> None:
    src = tmp_path / "t.jsonl"
    src.write_text("".join(json.dumps(tx.to_row(t)) + "\n" for t in _random_transcripts(25)))
    first = tx.encode_file(src, tok, "tok-a", 128, EOS_TOKEN_ID, {"Teo"})
    caches = list(tmp_path.glob("t.jsonl.tok-*.npz"))
    assert len(caches) == 1 and first
    with caplog.at_level("INFO"):
        again = tx.encode_file(src, tok, "tok-a", 128, EOS_TOKEN_ID, {"Teo"})
    assert again == first and "loaded" in caplog.text
    tx.encode_file(src, tok, "tok-a", 128, EOS_TOKEN_ID, None)  # different speakers -> a different cache
    assert len(list(tmp_path.glob("t.jsonl.tok-*.npz"))) == 2
    caches[0].write_bytes(b"not an npz")
    assert tx.encode_file(src, tok, "tok-a", 128, EOS_TOKEN_ID, {"Teo"}) == first  # bad cache: re-encoded
