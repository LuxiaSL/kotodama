"""Chat SFT data: ChatML + assistant-only labels, turn-boundary chunking, packing, and the packed collator.

Labels are -100 everywhere except the
content of assistant turns and each assistant turn's ``<|im_end|>``; an assistant message carrying ``"train": false``
stays fully masked (context the model conditions on, no gradient). The ``\\n`` after ``<|im_end|>`` and the
``<|im_start|>assistant\\n`` header are never supervised.

Packing puts several conversations in one row; each conversation is its own attention segment (no cross-document
attention), position ids restart per segment, and the padded tail is one more segment with labels -100. The collator
emits the trunk's doc-masking layout: ``cu_seqlens`` flat across the whole batch (``kotodama.data.dataset
.collate_packed``), so SFT runs the same varlen attention path pretraining did.
"""

from __future__ import annotations

import bisect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import torch

from kotodama.serve.chatml import IM_END_TOKEN_ID, IM_START_TOKEN_ID, render_chatml

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

Message = dict[str, Any]


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


@dataclass(frozen=True)
class Example:
    """One tokenized conversation (or chunk of one)."""

    input_ids: list[int]
    labels: list[int]

    @property
    def n_trainable(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)


@dataclass(frozen=True)
class PackedRow:
    input_ids: list[int]
    labels: list[int]
    position_ids: list[int]
    seg_lens: list[int]  # segment lengths tiling the row exactly (the pad tail is the last segment)


def render_conversation(messages: list[Message]) -> str:
    """The frozen ChatML template without a generation prompt (``CHATML_TEMPLATE``, add_generation_prompt=False)."""
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)


def load_chat_jsonl(path: str | Path) -> list[list[Message]]:
    """Conversations from a JSONL of ``{"messages": [{"role", "content"[, "train"]}, ...]}`` rows."""
    convs: list[list[Message]] = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                bad += 1
                logger.warning("%s:%d: unparseable JSON (%s) — skipped", path, ln, e)
                continue
            msgs = row.get("messages") if isinstance(row, dict) else None
            if not msgs or not all(isinstance(m, dict) and "role" in m and "content" in m for m in msgs):
                bad += 1
                continue
            convs.append(msgs)
    if bad:
        logger.warning("%s: %d rows skipped (no usable messages)", path, bad)
    return convs


def tokenize_and_mask(messages: list[Message], tokenizer: Tokenizer, max_len: int) -> Optional[Example]:
    """ChatML-render, tokenize (truncating to ``max_len``) and label one conversation; None if nothing trains."""
    text = render_conversation(messages)
    if not text.strip():
        return None
    input_ids = list(tokenizer.encode(text, add_special_tokens=False))[:max_len]
    if not input_ids:
        return None
    labels = [IGNORE_INDEX] * len(input_ids)
    header = list(tokenizer.encode("assistant\n", add_special_tokens=False))
    # The k-th assistant span in the stream is the k-th assistant message (order is preserved; truncation only cuts
    # the tail).
    trainable = [bool(m.get("train", True)) for m in messages if m.get("role") == "assistant"]
    a_idx = -1
    in_assistant = False
    i = 0
    while i < len(input_ids):
        tok = input_ids[i]
        if tok == IM_START_TOKEN_ID:
            end = i + 1 + len(header)
            if end <= len(input_ids) and input_ids[i + 1:end] == header:
                in_assistant = True
                a_idx += 1
                i = end
            else:
                in_assistant = False
                i += 1
            continue
        turn_trains = trainable[a_idx] if 0 <= a_idx < len(trainable) else True
        if in_assistant and turn_trains:
            labels[i] = tok
        if tok == IM_END_TOKEN_ID:
            in_assistant = False
        i += 1
    ex = Example(input_ids, labels)
    return ex if ex.n_trainable else None


def chunk_conversation(messages: list[Message], tokenizer: Tokenizer, max_len: int) -> list[list[Message]]:
    """Split a conversation that does not fit at turn boundaries; every chunk ends up holding an assistant turn.

    Greedy: accumulate turns until the next one would exceed ``max_len - 10`` tokens. A chunk without an assistant
    turn is dropped (nothing to train on). A single turn longer than the budget becomes its own (truncated) chunk."""
    def n_tok(msgs: list[Message]) -> int:
        return len(tokenizer.encode(render_conversation(msgs), add_special_tokens=False))

    if n_tok(messages) <= max_len:
        return [messages]
    budget = max_len - 10
    chunks: list[list[Message]] = []
    cur: list[Message] = []
    cur_len = 0
    for m in messages:
        t = n_tok([m])
        if cur and cur_len + t > budget:
            if any(x.get("role") == "assistant" for x in cur):
                chunks.append(cur)
            cur, cur_len = [m], t
        else:
            cur.append(m)
            cur_len += t
    if cur and any(x.get("role") == "assistant" for x in cur):
        chunks.append(cur)
    return chunks or [messages]


def encode_conversations(convs: Iterable[list[Message]], tokenizer: Tokenizer, max_len: int,
                         chunk: bool = True) -> list[Example]:
    out: list[Example] = []
    n_conv = n_drop = 0
    for msgs in convs:
        n_conv += 1
        for part in (chunk_conversation(msgs, tokenizer, max_len) if chunk else [msgs]):
            ex = tokenize_and_mask(part, tokenizer, max_len)
            if ex is None:
                n_drop += 1
            else:
                out.append(ex)
    logger.info("encoded %d conversations -> %d examples (%d chunks with nothing to train dropped); "
                "%d trainable tokens", n_conv, len(out), n_drop, sum(e.n_trainable for e in out))
    return out


def pack(lengths: list[int], max_len: int) -> list[list[int]]:
    """Best-fit-decreasing bins of example indices, each bin's total length <= ``max_len``.

    Longest first; each example goes in the fullest bin that still fits it (O(n log n) with a sorted list of
    remaining capacities). Examples longer than ``max_len`` are skipped (tokenize_and_mask truncates, so there are
    none from encode_conversations). Bins come back in creation order."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    bins: list[list[int]] = []
    # sorted (remaining, bin_id) pairs
    free: list[tuple[int, int]] = []
    for idx in order:
        n = lengths[idx]
        if n > max_len or n <= 0:
            continue
        k = bisect.bisect_left(free, (n, -1))
        if k < len(free):
            rem, b = free.pop(k)
            bins[b].append(idx)
            bisect.insort(free, (rem - n, b))
        else:
            bins.append([idx])
            bisect.insort(free, (max_len - n, len(bins) - 1))
    return bins


def build_rows(examples: list[Example], max_len: int, pad_id: int) -> list[PackedRow]:
    """Packed rows of exactly ``max_len`` tokens."""
    bins = pack([len(e.input_ids) for e in examples], max_len)
    rows: list[PackedRow] = []
    for b in bins:
        ids: list[int] = []
        labs: list[int] = []
        pos: list[int] = []
        segs: list[int] = []
        for idx in b:
            e = examples[idx]
            n = len(e.input_ids)
            ids.extend(e.input_ids)
            # A segment's first label would be predicted from the previous segment's last position: never train it.
            labs.append(IGNORE_INDEX)
            labs.extend(e.labels[1:])
            pos.extend(range(n))
            segs.append(n)
        pad = max_len - len(ids)
        if pad > 0:
            ids.extend([pad_id] * pad)
            labs.extend([IGNORE_INDEX] * pad)
            pos.extend(range(pad))
            segs.append(pad)
        rows.append(PackedRow(ids, labs, pos, segs))
    if rows:
        fill = sum(sum(s for s in r.seg_lens[:len(b)]) for r, b in zip(rows, bins)) / (len(rows) * max_len)
        logger.info("packed %d examples into %d rows of %d (fill %.1f%%)", len(examples), len(rows), max_len,
                    100 * fill)
    return rows


def collate(rows: list[PackedRow]) -> dict[str, Any]:
    """Batch tensors: input_ids/labels/position_ids (B, S) and flat ``cu_seqlens`` (n_segments + 1,) int32."""
    if not rows:
        raise ValueError("collate: empty batch")
    seq = len(rows[0].input_ids)
    if any(len(r.input_ids) != seq for r in rows):
        raise ValueError("collate: rows must share one length (pack with one max_len)")
    lens = [s for r in rows for s in r.seg_lens]
    cu = torch.zeros(len(lens) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
    if int(cu[-1]) != seq * len(rows):
        raise ValueError(f"segments sum to {int(cu[-1])}, batch has {seq * len(rows)} tokens")
    return {
        "input_ids": torch.tensor([r.input_ids for r in rows], dtype=torch.long),
        "labels": torch.tensor([r.labels for r in rows], dtype=torch.long),
        "position_ids": torch.tensor([r.position_ids for r in rows], dtype=torch.long),
        "cu_seqlens": cu,
        "max_seqlen": max(lens),
    }


def prompt_text(row: dict[str, Any]) -> tuple[str, bool]:
    """(prompt text, is_chat) for a preference / KTO row.

    ``prompt_chatml``: an already-rendered ChatML prompt ending in the open assistant tag. ``messages``: rendered
    here (the generation prompt is appended). ``prompt``: raw base-model text (no chat format)."""
    if "prompt_chatml" in row:
        return str(row["prompt_chatml"]), True
    if "messages" in row:
        return render_chatml(row["messages"]), True
    if "prompt" in row:
        return str(row["prompt"]), False
    raise KeyError("row needs one of prompt_chatml / messages / prompt")
