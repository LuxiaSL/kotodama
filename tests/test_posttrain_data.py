"""Chat SFT data: label masking matches the 3B SFT semantics; packing tiles rows exactly; segments never leak labels
across documents; the collator emits the trunk's flat cu_seqlens layout."""
from __future__ import annotations

import pytest
import torch

from kotodama.posttrain import chat_data as cd
from kotodama.serve.chatml import IM_END_TOKEN_ID, IM_START_TOKEN_ID


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    from kotodama.presets import TOKENIZER_NAME

    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except OSError as e:  # offline without a cache
        pytest.skip(f"tokenizer unavailable: {e}")


CONV = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello!"}, {"role": "user", "content": "and?"},
        {"role": "assistant", "content": "that is all."}]


def _decode_trained(tok, ex: cd.Example) -> str:
    return tok.decode([t for t, l in zip(ex.input_ids, ex.labels) if l != cd.IGNORE_INDEX])


def test_special_ids_match_tokenizer(tok) -> None:
    assert tok.convert_tokens_to_ids("<|im_start|>") == IM_START_TOKEN_ID
    assert tok.convert_tokens_to_ids("<|im_end|>") == IM_END_TOKEN_ID


def test_render_matches_frozen_template(tok) -> None:
    from kotodama.serve.chatml import CHATML_TEMPLATE

    tok.chat_template = CHATML_TEMPLATE
    assert cd.render_conversation(CONV) == tok.apply_chat_template(CONV, tokenize=False, add_generation_prompt=False)


def test_only_assistant_content_and_im_end_train(tok) -> None:
    ex = cd.tokenize_and_mask(CONV, tok, 512)
    assert ex is not None
    assert _decode_trained(tok, ex) == "hello!<|im_end|>that is all.<|im_end|>"


def test_train_false_turn_stays_masked(tok) -> None:
    conv = [dict(m) for m in CONV]
    conv[2]["train"] = False
    ex = cd.tokenize_and_mask(conv, tok, 512)
    assert _decode_trained(tok, ex) == "that is all.<|im_end|>"


def test_nothing_trainable_is_dropped(tok) -> None:
    assert cd.tokenize_and_mask(CONV[:2], tok, 512) is None


def test_chunking_ends_chunks_on_assistant_turns(tok) -> None:
    long = [{"role": "user", "content": "word " * 60}, {"role": "assistant", "content": "ok " * 60}] * 4
    chunks = cd.chunk_conversation(long, tok, 200)
    assert len(chunks) > 1
    assert all(any(m["role"] == "assistant" for m in c) for c in chunks)


def test_pack_respects_capacity() -> None:
    lengths = [7, 3, 5, 9, 1, 4, 10, 2, 6]
    bins = cd.pack(lengths, 10)
    assert sorted(i for b in bins for i in b) == list(range(len(lengths)))
    assert all(sum(lengths[i] for i in b) <= 10 for b in bins)
    assert len(bins) == 5  # 47 tokens, best-fit-decreasing reaches the bound


def test_rows_and_collate(tok) -> None:
    exs = cd.encode_conversations([CONV, CONV[:3], CONV], tok, 128)
    rows = cd.build_rows(exs, 128, pad_id=0)
    for r in rows:
        assert len(r.input_ids) == len(r.labels) == len(r.position_ids) == 128 == sum(r.seg_lens)
        start = 0
        for n in r.seg_lens:  # positions restart per segment; a segment's first label never trains
            assert r.position_ids[start:start + n] == list(range(n))
            assert r.labels[start] == cd.IGNORE_INDEX
            start += n
    trained = sum(sum(l != cd.IGNORE_INDEX for l in r.labels) for r in rows)
    assert trained == sum(e.n_trainable for e in exs)
    b = cd.collate(rows)
    assert b["input_ids"].shape == (len(rows), 128)
    assert b["cu_seqlens"].dtype == torch.int32 and int(b["cu_seqlens"][-1]) == 128 * len(rows)
    assert b["max_seqlen"] == max(s for r in rows for s in r.seg_lens)


def test_prompt_text_forms() -> None:
    assert cd.prompt_text({"prompt": "raw"}) == ("raw", False)
    text, chat = cd.prompt_text({"messages": [{"role": "user", "content": "q"}]})
    assert chat and text.endswith("<|im_start|>assistant\n")
    with pytest.raises(KeyError):
        cd.prompt_text({"completion": "x"})
