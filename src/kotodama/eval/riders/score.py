"""Closed-candidate logprob scoring — the one implementation behind ``/logprobs``.

``kotodama.serve.server`` exposes it over HTTP (``_score_logprobs``) and the
training loop calls it in-process on the live weights (``kotodama.training.inloop``);
both go through :func:`score_logprob_batch`, so a rider row means the same thing
from either path (parity: ``tests/test_inloop.py``).

One forward per batch, no decode loop: prompts are right-padded with id 0 and
logits are gathered at each sequence's last REAL position — exactly correct
under causal attention (pad tokens sit after the read position and can never
attend into it), so no attention mask is needed.

Result dict per item (the ``/logprobs`` contract)::

    {"id", "prompt_tokens", "entropy",
     "candidates": [{"text", "id", "n_tokens", "logprob", "rank_full"}],
     "top": [{"token", "id", "logprob"}]}

A candidate is scored by its FIRST token; ``n_tokens > 1`` is reported, and the
rider runner treats it as a name-verification failure.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import torch

LOGPROB_MAX_BATCH = 64
LOGPROB_MAX_TOTAL_TOKENS = 96_000  # sum over the batch; logits are B x T x 49152 bf16

# (B, T) ids -> (B, T, V) logits
LogitsFn = Callable[[torch.Tensor], torch.Tensor]


class LogprobLimitError(ValueError):
    """A batch breaks a size limit (prompt too long / too many total tokens)."""


def _get(item: Any, key: str) -> Any:
    return item[key] if isinstance(item, Mapping) else getattr(item, key)


def score_logprob_batch(logits_fn: LogitsFn, tokenizer: Any, items: Sequence[Any],
                        top_n: int, device: torch.device | str,
                        max_seq_len: Optional[int] = None,
                        max_total_tokens: Optional[int] = LOGPROB_MAX_TOTAL_TOKENS,
                        ) -> list[dict[str, Any]]:
    """Score one batch of ``{id, prompt, candidates}`` items (dicts or objects).

    ``max_seq_len``: a prompt of ``>= max_seq_len`` tokens raises
    :class:`LogprobLimitError` (as does a batch above ``max_total_tokens``).
    Gradient mode is the caller's choice (the server wraps this in
    ``inference_mode``, the training loop in ``no_grad``).
    """
    if not items:
        return []
    tok = tokenizer
    enc = [tok.encode(_get(it, "prompt")) for it in items]
    lens = [len(e) for e in enc]
    if min(lens) < 1:
        raise LogprobLimitError("empty prompt (0 tokens)")
    if max_seq_len is not None and max(lens) >= max_seq_len:
        raise LogprobLimitError(f"prompt too long ({max(lens)} tokens, max {max_seq_len})")
    if max_total_tokens is not None and sum(lens) > max_total_tokens:
        raise LogprobLimitError(f"batch too large ({sum(lens)} total tokens, "
                                f"max {max_total_tokens})")
    batch = torch.zeros((len(enc), max(lens)), dtype=torch.long)
    for r, e in enumerate(enc):
        batch[r, : len(e)] = torch.tensor(e, dtype=torch.long)
    batch = batch.to(device)
    logits = logits_fn(batch)

    results: list[dict[str, Any]] = []
    for r, it in enumerate(items):
        last = logits[r, lens[r] - 1].float()
        logsm = torch.log_softmax(last, dim=-1)
        probs = logsm.exp()
        entropy = float(-(probs * logsm).sum())
        cands = []
        for c in _get(it, "candidates"):
            cids = tok.encode(c, add_special_tokens=False)
            first = cids[0]
            cands.append({
                "text": c,
                "id": int(first),
                "n_tokens": len(cids),  # runner treats >1 as a verification failure
                "logprob": float(logsm[first]),
                "rank_full": int((logsm > logsm[first]).sum()) + 1,
            })
        top = []
        if top_n:
            tv, ti = logsm.topk(top_n)
            top = [{"token": tok.decode([int(i)]), "id": int(i), "logprob": float(v)}
                   for v, i in zip(tv, ti)]
        results.append({"id": _get(it, "id"), "prompt_tokens": lens[r],
                        "entropy": entropy, "candidates": cands, "top": top})
    return results


def plan_batches(lengths: Sequence[int], max_batch: int = LOGPROB_MAX_BATCH,
                 max_total_tokens: int = LOGPROB_MAX_TOTAL_TOKENS) -> list[list[int]]:
    """Index batches over ``lengths``: sorted by length (less padding), each batch
    at most ``max_batch`` items and ``max_total_tokens`` summed prompt tokens
    (a single over-budget prompt still gets a batch of its own)."""
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    out: list[list[int]] = []
    cur: list[int] = []
    cur_tokens = 0
    for i in order:
        if cur and (len(cur) >= max_batch or cur_tokens + lengths[i] > max_total_tokens):
            out.append(cur)
            cur, cur_tokens = [], 0
        cur.append(i)
        cur_tokens += lengths[i]
    if cur:
        out.append(cur)
    return out


def score_logprob_items(logits_fn: LogitsFn, tokenizer: Any, items: Sequence[Any],
                        top_n: int, device: torch.device | str,
                        max_seq_len: Optional[int] = None,
                        max_batch: int = LOGPROB_MAX_BATCH,
                        max_total_tokens: int = LOGPROB_MAX_TOTAL_TOKENS,
                        lengths: Optional[Sequence[int]] = None,
                        ) -> list[dict[str, Any]]:
    """Score any number of items in length-sorted batches; results in INPUT order.
    ``lengths`` (prompt token counts, if already known) only steers batching."""
    if lengths is None:
        lengths = [len(tokenizer.encode(_get(it, "prompt"))) for it in items]
    elif len(lengths) != len(items):
        raise ValueError(f"{len(lengths)} lengths for {len(items)} items")
    out: list[Optional[dict[str, Any]]] = [None] * len(items)
    for idx in plan_batches(lengths, max_batch, max_total_tokens):
        res = score_logprob_batch(logits_fn, tokenizer, [items[i] for i in idx], top_n,
                                  device, max_seq_len=max_seq_len, max_total_tokens=None)
        for i, r in zip(idx, res):
            out[i] = r
    assert all(r is not None for r in out)
    return out  # type: ignore[return-value]
