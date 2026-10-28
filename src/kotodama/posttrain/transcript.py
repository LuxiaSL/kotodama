"""Plain-text transcripts: conversations as documents, with no special chat tokens.

A transcript renders as an optional header (plain prose: the setting, the participants), a blank line, then one
``Name: text`` turn per paragraph, turns separated by a blank line, and the end-of-text token after the last turn::

    A conversation between Mara and Teo, recorded at the station.

    Mara: You gave away Dad's watch.

    Teo: He didn't even know me.<|endoftext|>

Every piece of this is ordinary text the base model has read in pretraining (transcripts, screenplays, forum
threads), so post-training teaches a document genre rather than a new protocol. A turn ends where the next speaker's
``Name:`` begins, which is also the stop sequence at generation time.

Labels come from character offsets on the whole-document tokenization, so token boundaries are exactly the ones the
model sees at inference:

* ``train_speakers=None`` (unmasked): every turn trains, from the first speaker tag to the end-of-text token. The
  model learns the whole document — every participant's side — which is what a multi-voice transcript needs.
* ``train_speakers={...}`` (masked): a listed speaker's turn trains from its text through the NEXT speaker's ``Name:``
  tag (that tag is how the turn learns to stop); the last turn trains through the end-of-text token. Other turns are
  context. A turn with ``"train": false`` is context whoever speaks it.

The header never trains: it is the frame a run supplies. A row with ``"ended": false`` keeps its end-of-text as
context: the conversation was cut there, so the model must not learn to stop there.

An anthology row ``{"scenes": [transcript row, ...]}`` is one document holding several transcripts, each followed
by its own end-of-text token (``kotodama.posttrain.anthology`` builds them from loom runs).

CLI:  python -m kotodama.posttrain.transcript convert-markdown --in raw.jsonl --out turns.jsonl [--rename "A=B"]
      (rows ``{"text": "**User:** ... **Night Owl:** ..."}`` -> rows ``{"turns": [{"speaker", "text"}, ...]}``)
      python -m kotodama.posttrain.transcript convert-messages --in data.parquet|.jsonl --out turns.jsonl
      (instruction data ``{"messages": [{"role", "content"}]}`` -> ``User:`` / ``Assistant:`` transcripts; a system
      message becomes the header; rows whose text contains a turn tag are dropped)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from kotodama.posttrain.chat_data import IGNORE_INDEX, Example

logger = logging.getLogger(__name__)

SEP = "\n\n"
_NAME_OK = re.compile(r"^[^\n:]{1,40}$")
_MD_ROLE = re.compile(r"\*\*([^*\n]{1,40}?):\*\*")


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str
    train: bool = True


@dataclass(frozen=True)
class Transcript:
    turns: tuple[Turn, ...]
    header: str = ""
    ended: bool = True  # False: the end-of-text after the last turn is context (the conversation was cut, not ended)


@dataclass(frozen=True)
class Anthology:
    """Several transcripts in one document, one after another, each followed by its end-of-text token, in one
    attention segment (a later scene sees the earlier ones)."""

    scenes: tuple[Transcript, ...]


def parse_row(row: dict[str, Any]) -> Optional[Transcript]:
    """A transcript from ``{"turns": [{"speaker", "text"[, "train"]}], "header"?, "ended"?}``; None if malformed."""
    turns = row.get("turns") if isinstance(row, dict) else None
    if not isinstance(turns, list) or not turns:
        return None
    out = []
    for t in turns:
        if not isinstance(t, dict) or not isinstance(t.get("speaker"), str) or not isinstance(t.get("text"), str):
            return None
        sp = t["speaker"].strip()
        if not _NAME_OK.match(sp):
            return None
        out.append(Turn(sp, t["text"].strip(), bool(t.get("train", True))))
    header = row.get("header", "")
    return Transcript(tuple(out), header.strip() if isinstance(header, str) else "", bool(row.get("ended", True)))


def parse_document(row: dict[str, Any]) -> Optional[Transcript | Anthology]:
    """A transcript row, or an anthology row ``{"scenes": [transcript row, ...]}``; None if malformed (an anthology
    with any malformed scene is malformed)."""
    if isinstance(row, dict) and "scenes" in row:
        scenes = row["scenes"]
        if not isinstance(scenes, list) or not scenes:
            return None
        parsed = [parse_row(s) for s in scenes]
        return None if any(s is None for s in parsed) else Anthology(tuple(parsed))  # type: ignore[arg-type]
    return parse_row(row)


def load_jsonl(path: str | Path) -> list[Transcript | Anthology]:
    res: list[Transcript | Anthology] = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                tr = parse_document(json.loads(line))
            except json.JSONDecodeError:
                tr = None
            if tr is None:
                bad += 1
            else:
                res.append(tr)
    if bad:
        logger.warning("%s: %d rows skipped (need turns of {speaker, text}; speaker without ':' or newline)", path, bad)
    return res


def render(tr: Transcript) -> tuple[str, list[tuple[int, int, int]]]:
    """(document text without the end-of-text token, per-turn (tag_start, text_start, end) character offsets)."""
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    pos = 0
    if tr.header:
        parts.append(tr.header + SEP)
        pos += len(tr.header) + len(SEP)
    for i, t in enumerate(tr.turns):
        tag = f"{t.speaker}:"
        body = f" {t.text}" if t.text else ""
        spans.append((pos, pos + len(tag), pos + len(tag) + len(body)))
        piece = tag + body + (SEP if i < len(tr.turns) - 1 else "")
        parts.append(piece)
        pos += len(piece)
    return "".join(parts), spans


def render_prompt(tr: Transcript, next_speaker: str) -> str:
    """The document up to an open turn for ``next_speaker`` — what a generator continues. Stop on ``"\\n\\n<Name>:"``."""
    text, _ = render(tr)
    lead = text + SEP if tr.turns else (tr.header + SEP if tr.header else "")
    return f"{lead}{next_speaker}:"


def stop_sequences(speakers: Iterable[str]) -> list[str]:
    """Stops that end a generated turn: the next turn starting under any participant's name."""
    return [f"{SEP}{s}:" for s in dict.fromkeys(speakers)]


def _train_ranges(tr: Transcript, spans: list[tuple[int, int, int]], n_text: int,
                  train_speakers: Optional[set[str]]) -> list[tuple[int, int]]:
    """Character ranges that train (sorted, disjoint); ``n_text + 1`` stands for the end-of-text token."""
    ranges: list[tuple[int, int]] = []
    n = len(tr.turns)
    for i, (t, (tag0, txt0, _end)) in enumerate(zip(tr.turns, spans)):
        if not t.train:
            continue
        if train_speakers is None:
            lo = tag0  # unmasked: the speaker tag trains too (who speaks next is part of the document)
        elif t.speaker in train_speakers:
            lo = txt0
        else:
            continue
        # through the next tag; the last turn through end-of-text when the conversation ended there
        hi = spans[i + 1][1] if i + 1 < n else (n_text + 1 if tr.ended else n_text)
        ranges.append((lo, hi))
    return ranges


def _example(ids: list[int], starts: list[int], ranges: list[tuple[int, int]], n_text: int, max_len: int,
             eos_id: int, keep_empty: bool = False) -> Optional[Example]:
    """Labels for one tokenized document: a token trains when its start offset falls in a training range. None when
    nothing trains, unless ``keep_empty`` (an anthology scene may be all context)."""
    ids = ids + [eos_id]
    a = np.asarray(list(starts) + [n_text], dtype=np.int64)
    if ranges:
        lo = np.asarray([r[0] for r in ranges], dtype=np.int64)
        hi = np.asarray([r[1] for r in ranges], dtype=np.int64)
        k = np.searchsorted(lo, a, side="right") - 1
        inside = (k >= 0) & (a < hi[np.clip(k, 0, None)])
    else:
        inside = np.zeros(len(a), dtype=bool)
    labels = np.where(inside, np.asarray(ids, dtype=np.int64), IGNORE_INDEX).tolist()
    if len(ids) > max_len:
        ids, labels = ids[:max_len], labels[:max_len]
    ex = Example(ids, labels)
    return ex if ex.n_trainable or keep_empty else None


def tokenize(tr: Transcript, tokenizer: Any, max_len: int, eos_id: int,
             train_speakers: Optional[set[str]] = None) -> Optional[Example]:
    """Tokenize + label one transcript (see the module docstring); None if nothing trains after truncation."""
    text, spans = render(tr)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return _example(list(enc["input_ids"]), [o[0] for o in enc["offset_mapping"]],
                    _train_ranges(tr, spans, len(text), train_speakers), len(text), max_len, eos_id)


def encode(transcripts: Iterable[Transcript | Anthology], tokenizer: Any, max_len: int, eos_id: int,
           train_speakers: Optional[set[str]] = None, drop_overlong: bool = False,
           chunk: int = 1024) -> list[Example]:
    """Tokenized examples (the tokenizer is called on ``chunk`` documents at a time — a fast tokenizer encodes a batch
    in parallel). ``drop_overlong`` drops a transcript longer than ``max_len`` instead of truncating it (instruction
    data: a truncated answer teaches answers that stop mid-thought). An ``Anthology`` is one example: each scene is
    tokenized and labelled on its own (so token boundaries match a scene generated from its header) and the scenes'
    tokens, each ending in end-of-text, are concatenated before truncation."""
    out: list[Example] = []
    n = dropped = 0
    cap = max_len + (1 if drop_overlong else 0)
    unbounded = 1 << 62
    it = iter(transcripts)
    while True:
        batch = [tr for _, tr in zip(range(chunk), it)]
        if not batch:
            break
        units = [s for tr in batch for s in (tr.scenes if isinstance(tr, Anthology) else (tr,))]
        rendered = [render(tr) for tr in units]
        enc = tokenizer([t for t, _ in rendered], return_offsets_mapping=True, add_special_tokens=False)
        parts = iter(zip(units, rendered, enc["input_ids"], enc["offset_mapping"]))
        for doc in batch:
            n += 1
            pieces = []
            for _ in (doc.scenes if isinstance(doc, Anthology) else (doc,)):
                tr, (text, spans), ids, offs = next(parts)
                pieces.append(_example(list(ids), [o[0] for o in offs],
                                       _train_ranges(tr, spans, len(text), train_speakers), len(text), unbounded,
                                       eos_id, keep_empty=True))
            all_ids = [t for p in pieces if p is not None for t in p.input_ids]
            all_labels = [t for p in pieces if p is not None for t in p.labels]
            ex: Optional[Example] = Example(all_ids[:cap], all_labels[:cap])
            if not ex.n_trainable or (drop_overlong and len(ex.input_ids) > max_len):
                ex = None
            if ex is None:
                dropped += 1
            else:
                out.append(ex)
    logger.info("encoded %d transcripts -> %d examples (%d with nothing to train dropped); %d trainable tokens",
                n, len(out), dropped, sum(e.n_trainable for e in out))
    return out


CACHE_VERSION = 1


def encode_file(path: str | Path, tokenizer: Any, tokenizer_name: str, max_len: int, eos_id: int,
                train_speakers: Optional[set[str]] = None, drop_overlong: bool = False,
                cache: bool = True) -> list[Example]:
    """``encode(load_jsonl(path))``, cached beside the data as ``<path>.tok-<key>.npz``. The key covers the file
    (size, mtime), the tokenizer name, max_len, the speakers, drop_overlong and the cache format; a miss encodes and
    writes the cache atomically, so concurrent runs on one dataset tokenize it at most once each and later runs load it."""
    path = Path(path)
    st = path.stat()
    spk = "all" if train_speakers is None else ",".join(sorted(train_speakers))
    key = hashlib.sha256(json.dumps([CACHE_VERSION, st.st_size, int(st.st_mtime), tokenizer_name, max_len, eos_id,
                                     spk, drop_overlong]).encode()).hexdigest()[:12]
    cpath = path.with_name(f"{path.name}.tok-{key}.npz")
    if cache and cpath.exists():
        try:
            z = np.load(cpath)
            ids, train, lens = z["ids"], z["train"], z["lens"]
            cuts = np.cumsum(lens)[:-1]
            labels = np.where(train, ids, IGNORE_INDEX)
            out = [Example(i.tolist(), l.tolist()) for i, l in zip(np.split(ids, cuts), np.split(labels, cuts))]
            logger.info("loaded %d examples from %s", len(out), cpath)
            return out
        except Exception as e:  # noqa: BLE001 — a bad cache is re-encoded, never fatal
            logger.warning("ignoring unreadable token cache %s (%s: %s)", cpath, type(e).__name__, e)
    out = encode(load_jsonl(path), tokenizer, max_len, eos_id, train_speakers, drop_overlong)
    if cache and out:
        ids = np.concatenate([np.asarray(e.input_ids, dtype=np.int32) for e in out])
        train = np.concatenate([np.asarray(e.labels) != IGNORE_INDEX for e in out])
        lens = np.asarray([len(e.input_ids) for e in out], dtype=np.int64)
        tmp = cpath.with_name(f"{cpath.name}.{os.getpid()}.tmp.npz")
        try:
            np.savez(tmp, ids=ids, train=train, lens=lens)
            os.replace(tmp, cpath)
            logger.info("wrote token cache %s", cpath)
        except OSError as e:
            logger.warning("could not write token cache %s (%s)", cpath, e)
            tmp.unlink(missing_ok=True)
    return out


# ── conversion ───────────────────────────────────────────────────────────────


def _name_key(name: str) -> str:
    """Spacing, case and a parenthetical descriptor do not make a different speaker: "NightOwl", "night owl" and
    "Night Owl (curious)" share a key."""
    return re.sub(r"\s+|\(.*?\)", "", name).lower()


def from_markdown_roles(text: str, rename: dict[str, str] | None = None, min_turns: int = 4,
                        keep_speakers: Optional[list[str]] = None) -> Optional[Transcript]:
    """``**Name:** text`` markdown dialogue -> a transcript (None below ``min_turns`` or with fewer than 2 speakers).
    A trailing "Response" on a label is dropped. With ``keep_speakers``, labels fold onto those canonical names by
    ``_name_key`` and a conversation with any other speaker is rejected (label noise is usually a broken turn)."""
    rename = rename or {}
    canon = {_name_key(s): s for s in keep_speakers} if keep_speakers else None
    marks = list(_MD_ROLE.finditer(text))
    turns: list[Turn] = []
    for i, m in enumerate(marks):
        body = text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)].strip()
        raw = re.sub(r"\s+Response$", "", m.group(1).strip())
        if canon is not None:
            if _name_key(raw) not in canon:
                return None
            raw = canon[_name_key(raw)]
        sp = rename.get(raw, raw)
        if not body or not _NAME_OK.match(sp):
            continue
        if turns and turns[-1].speaker == sp:  # a split turn: merge
            turns[-1] = Turn(sp, turns[-1].text + SEP + body)
        else:
            turns.append(Turn(sp, body))
    if len(turns) < min_turns or len({t.speaker for t in turns}) < 2:
        return None
    return Transcript(tuple(turns))


ROLE_SPEAKERS = {"user": "User", "assistant": "Assistant"}


def from_messages(messages: Any) -> Optional[Transcript]:
    """``[{"role", "content"}]`` -> a ``User:`` / ``Assistant:`` transcript (a leading system message becomes the
    header). None for other roles (tools), an empty turn, no assistant turn, or text containing a turn tag."""
    if not isinstance(messages, (list, tuple)) or not messages:
        return None
    header = ""
    turns: list[Turn] = []
    for i, m in enumerate(messages):
        role, content = (m.get("role"), m.get("content")) if isinstance(m, dict) else (None, None)
        if not isinstance(content, str):
            return None
        if role == "system" and i == 0:
            header = content.strip()
            continue
        sp = ROLE_SPEAKERS.get(role)
        if sp is None or not content.strip():
            return None
        turns.append(Turn(sp, content.strip()))
    if not any(t.speaker == "Assistant" for t in turns):
        return None
    tags = [f"{SEP}{s}:" for s in ROLE_SPEAKERS.values()] + [f"\n{s}:" for s in ROLE_SPEAKERS.values()]
    if any(tag in t.text for t in turns for tag in tags) or any(t.text.startswith(f"{s}:") for t in turns
                                                                  for s in ROLE_SPEAKERS.values()):
        return None
    return Transcript(tuple(turns), header)


def _iter_records(path: str) -> Iterable[dict[str, Any]]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            yield from batch.to_pylist()
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def to_row(tr: Transcript) -> dict[str, Any]:
    row: dict[str, Any] = {"turns": [{"speaker": t.speaker, "text": t.text} | ({} if t.train else {"train": False})
                                     for t in tr.turns]}
    if tr.header:
        row["header"] = tr.header
    return row


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="plain-text transcript tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert-markdown", help="**Name:** markdown dialogues -> transcript rows")
    c.add_argument("--in", dest="inp", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--field", default="text")
    c.add_argument("--rename", action="append", default=[], help="OLD=NEW speaker rename (repeatable)")
    c.add_argument("--min_turns", type=int, default=4)
    c.add_argument("--keep_speakers", default=None,
                   help="comma list of canonical speaker labels (before --rename); other speakers reject the row")
    cm = sub.add_parser("convert-messages", help="{messages: [{role, content}]} -> User:/Assistant: transcript rows")
    cm.add_argument("--in", dest="inp", required=True, nargs="+", help=".parquet or .jsonl files")
    cm.add_argument("--out", required=True)
    cm.add_argument("--field", default="messages")
    cm.add_argument("--source_field", default="source", help="per-row provenance column carried into the output")
    a = ap.parse_args(argv)
    if a.cmd == "convert-messages":
        kept = skipped = 0
        with open(a.out, "w", encoding="utf-8") as fo:
            for path in a.inp:
                for rec in _iter_records(path):
                    tr = from_messages(rec.get(a.field))
                    if tr is None:
                        skipped += 1
                        continue
                    row = to_row(tr)
                    if rec.get(a.source_field) is not None:
                        row["source"] = str(rec[a.source_field])
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
        logger.info("converted %d, skipped %d (tool roles, empty turns, turn-tag collisions) -> %s", kept, skipped,
                    a.out)
        return 0
    rename = dict(r.split("=", 1) for r in a.rename)
    keep = [s.strip() for s in a.keep_speakers.split(",")] if a.keep_speakers else None
    kept = skipped = 0
    with open(a.inp, encoding="utf-8") as fi, open(a.out, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            try:
                text = json.loads(line).get(a.field)
            except (json.JSONDecodeError, AttributeError):
                text = None
            tr = from_markdown_roles(text, rename, a.min_turns, keep) if isinstance(text, str) else None
            if tr is None:
                skipped += 1
                continue
            fo.write(json.dumps(to_row(tr), ensure_ascii=False) + "\n")
            kept += 1
    logger.info("converted %d, skipped %d -> %s", kept, skipped, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
