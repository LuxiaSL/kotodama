"""Score batteries with one loaded model. Two row shapes, one scorer (``riders.score.score_logprob_items``).

Prompt rows in: ``{id, prompt, candidates, …}``. Out: the input minus ``prompt``, plus
    model, prompt_tokens,
    cand_lps     {stripped candidate: logprob of its FIRST token at the next position}
    rank_full    {stripped candidate: rank of that token in the full vocabulary (1 = argmax)}
    multi_token  [candidates whose text is more than one token]
— the stage-gauge row contract plus ``rank_full``, so readers of recorded gauge rows apply unchanged.
Tokenization is the same as ``/logprobs``: ``tok.encode(prompt)`` with the tokenizer's
default special tokens, candidates encoded without them. Batching is length-sorted with right padding (exact under
causal attention; ``score_logprob_batch`` documents why).

Bindprobes items are rendered per condition (``bindprobes.grid.build_payload_item``) and written as bindprobes grid
rows (``bindprobes.grid.score_row``), so ``python -m kotodama.eval.bindprobes aggregate`` reads them unchanged.

Both writers append and flush per chunk, and skip ids already in the output: resumable, and tolerant of a torn last
line from a cancelled job.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

from kotodama.eval.binding.models import LoadedModel
from kotodama.eval.riders.score import LOGPROB_MAX_TOTAL_TOKENS, score_logprob_items

log = logging.getLogger(__name__)

CHUNK = 512   # rows per write/flush


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _clean_existing(out: Path, key: "Any") -> set[Any]:
    """Keys already scored in ``out``; rewrites the file without a torn trailing line."""
    if not out.exists():
        return set()
    good: list[str] = []
    done: set[Any] = set()
    for line in out.read_text().splitlines():
        try:
            done.add(key(json.loads(line)))
            good.append(line)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    out.write_text("".join(x + "\n" for x in good))
    return done


def _chunks(xs: Sequence[Any], n: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def prompt_row(row: Mapping[str, Any], res: Mapping[str, Any], model: str) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k != "prompt"}
    cands = res["candidates"]
    out.update(model=model, prompt_tokens=res["prompt_tokens"],
               multi_token=[c["text"] for c in cands if c["n_tokens"] > 1],
               cand_lps={c["text"].strip(): c["logprob"] for c in cands},
               rank_full={c["text"].strip(): c["rank_full"] for c in cands})
    return out


def _validate(rows: Iterable[Mapping[str, Any]]) -> None:
    for r in rows:
        for k in ("id", "prompt", "candidates"):
            if k not in r:
                raise ValueError(f"prompt row missing {k!r}: {str(r)[:120]}")
        if r.get("gold") is not None and r["gold"] not in {c.strip() for c in r["candidates"]}:
            raise ValueError(f"row {r['id']}: gold {r['gold']!r} is not a candidate")


def score_prompt_rows(model: LoadedModel, rows: Sequence[Mapping[str, Any]], out: Path, name: str,
                      max_batch: int = 32) -> int:
    """Score prompt rows into ``out``; returns the number of rows written this call."""
    _validate(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _clean_existing(out, lambda r: r["id"])
    todo = [r for r in rows if r["id"] not in done]
    log.info("%s → %s: %d of %d rows pending", name, out.name, len(todo), len(rows))
    t0, n = time.time(), 0
    with torch.inference_mode(), out.open("a") as f:
        for chunk in _chunks(todo, CHUNK):
            res = score_logprob_items(model.logits_fn, model.tokenizer, chunk, top_n=0, device=model.device,
                                      max_seq_len=model.max_seq_len, max_batch=max_batch,
                                      max_total_tokens=LOGPROB_MAX_TOTAL_TOKENS)
            for r, x in zip(chunk, res):
                f.write(json.dumps(prompt_row(r, x, name)) + "\n")
            f.flush()
            n += len(chunk)
            log.info("%s: %d/%d (%.0fs)", out.name, n, len(todo), time.time() - t0)
    return n


def score_bindprobe_items(model: LoadedModel, items: Sequence[Any], conditions: Sequence[str], out: Path,
                          name: str, max_batch: int = 32, top_n: int = 5) -> int:
    """Score bindprobes items × conditions into bindprobes grid rows; returns rows written this call."""
    from kotodama.eval.bindprobes.grid import build_payload_item, score_row

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _clean_existing(out, lambda r: (r["item_id"], r["condition"]))
    jobs = [(it, c) for it in items for c in conditions if (it.item_id, c) not in done]
    log.info("%s → %s: %d of %d (item, condition) pending", name, out.name, len(jobs),
             len(items) * len(conditions))
    t0, n = time.time(), 0
    with torch.inference_mode(), out.open("a") as f:
        for chunk in _chunks(jobs, CHUNK):
            payload = [build_payload_item(it, c) for it, c in chunk]
            res = score_logprob_items(model.logits_fn, model.tokenizer, payload, top_n=top_n, device=model.device,
                                      max_seq_len=model.max_seq_len, max_batch=max_batch,
                                      max_total_tokens=LOGPROB_MAX_TOTAL_TOKENS)
            for (it, c), x in zip(chunk, res):
                f.write(json.dumps(score_row(it, x, c, name)) + "\n")
            f.flush()
            n += len(chunk)
            log.info("%s: %d/%d (%.0fs)", out.name, n, len(jobs), time.time() - t0)
    return n
