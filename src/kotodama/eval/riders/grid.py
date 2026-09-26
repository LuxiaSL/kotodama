"""Rider-grid runner: score every (item, renderer, shots) against ``/logprobs``.

One run = one model behind one server. The endpoint contract is
``kotodama.serve.server`` (``LogprobRequest``/``_score_logprobs``): POST
``{"items": [{"id", "prompt", "candidates"}], "top_n"}`` -> ``{"results": [{"id",
"prompt_tokens", "entropy", "candidates": [{"text", "id", "n_tokens", "logprob",
"rank_full"}], "top"}]}``; at most ``LOGPROB_MAX_BATCH`` (64) items per request.

Model kinds (how the chat renderer is delivered):
  koto     raw prompts everywhere; chat renderer = ChatML string built here
           (base and tuned koto are BOTH driven raw — wave-1 convention)
  hf-base  raw prompts everywhere; chat renderer = the same ChatML string as a
           LITERAL (multi-token delimiters for non-ChatML vocabs — flagged as
           chat_format=chatml-literal, read with that caveat)
  hf-chat  raw prompts for code/narrative/screenplay; chat renderer = messages +
           answer_prefix, templated server-side by an HF shim. The kotodama
           server requires ``prompt`` and rejects these items.

Scoring is closed-candidate-set: gold rank among in-item candidates, margin =
logp(gold) - max(logp(distractor)). Chance correction happens in aggregate.

Crash-safe: appends to the output JSONL; already-scored (item, renderer, shots)
keys are skipped on relaunch.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import requests

from kotodama.eval.riders.items import RENDERERS, Item, build_prompt, load_items

Kind = Literal["koto", "hf-base", "hf-chat"]
KINDS: tuple[Kind, ...] = ("koto", "hf-base", "hf-chat")

SHOT_CONDS = (0, 3)
BATCH = 32
TIMEOUT = 300.0
RETRIES = 4
DEFAULT_ENDPOINT = "http://localhost:2222"


class LogprobsError(RuntimeError):
    """The server failed or answered outside the /logprobs contract."""


def default_endpoint() -> str:
    """``$KOTODAMA_ENDPOINT``, else the local gateway."""
    return os.environ.get("KOTODAMA_ENDPOINT") or DEFAULT_ENDPOINT


def chat_family(kind: str) -> tuple[str, str]:
    """(prompt family, chat_format tag) for the chat renderer under ``kind``."""
    family = {"koto": "chatml", "hf-base": "chatml", "hf-chat": "hf-chat"}.get(kind)
    if family is None:
        raise ValueError(f"unknown model kind {kind!r}; choose from {KINDS}")
    return family, ("chatml-literal" if kind == "hf-base" else family)


def job_id(item: Item, renderer: str, nshots: int) -> str:
    return f"{item.item_id}|{renderer}|{nshots}"


def build_payload_item(item: Item, renderer: str, nshots: int,
                       shots_all: list[Item], kind: str
                       ) -> tuple[dict[str, Any], Optional[str]]:
    """One ``/logprobs`` request item + its chat_format tag (None off-chat)."""
    if nshots > len(shots_all):
        raise ValueError(f"{nshots}-shot needs {nshots} shots, have {len(shots_all)}")
    if renderer == "chat":
        family, chat_format = chat_family(kind)
    else:
        family, chat_format = "raw", None
    built = build_prompt(item, renderer, shots_all[:nshots], family)
    return ({"id": job_id(item, renderer, nshots), **built,
             "candidates": [" " + c for c in item.candidates]}, chat_format)


def request_batch(endpoint: str, payload_items: list[dict[str, Any]],
                  top_n: int) -> list[dict[str, Any]]:
    """POST one batch; retries transport/5xx errors, then fails loud."""
    body = {"items": payload_items, "top_n": top_n}
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = requests.post(f"{endpoint.rstrip('/')}/logprobs", json=body,
                              timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()["results"]
        except Exception as e:  # noqa: BLE001 — retry transport/5xx, then fail loud
            last = e
            time.sleep(3.0 * (attempt + 1))
    raise LogprobsError(f"/logprobs failed after {RETRIES} attempts: {last}")


def score_row(item: Item, res: dict[str, Any], renderer: str, nshots: int,
              model: str, chat_format: str | None) -> dict[str, Any]:
    """One grid row from one ``/logprobs`` result. Ties rank against gold."""
    cands = res["candidates"]
    bad = [c["text"] for c in cands if c["n_tokens"] != 1]
    by_text = {c["text"]: c for c in cands}
    gold_key = " " + item.gold
    if gold_key not in by_text:
        raise LogprobsError(f"{item.item_id}: gold {gold_key!r} missing from result")
    gold_lp = by_text[gold_key]["logprob"]
    distractor_lps = [c["logprob"] for c in cands if c["text"] != gold_key]
    if not distractor_lps:
        raise LogprobsError(f"{item.item_id}: no distractors in result")
    margin = gold_lp - max(distractor_lps)
    rank = 1 + sum(1 for lp in distractor_lps if lp >= gold_lp)
    return {
        "model": model, "item_id": item.item_id, "renderer": renderer,
        "nshots": nshots, "tier": item.tier, "n": item.n, "m": item.m,
        "recency_control": item.recency_control, "gold": item.gold,
        "n_candidates": len(cands), "gold_lp": gold_lp, "margin": margin,
        "gold_rank_cand": rank, "gold_rank_full": by_text[gold_key]["rank_full"],
        "entropy": res["entropy"], "prompt_tokens": res["prompt_tokens"],
        "cand_lps": {c["text"].strip(): c["logprob"] for c in cands},
        "top": res.get("top", []),
        "multi_token_candidates": bad,  # must stay empty; aggregate hard-fails otherwise
        "chat_format": chat_format,
    }


def _done_keys(out_path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["item_id"], r["renderer"], r["nshots"]))
        print(f"resume: {len(done)} rows already scored", flush=True)
    return done


def run_grid(model: str, kind: str, items: list[Item], shots_all: list[Item],
             out_path: Path, endpoint: Optional[str] = None,
             renderers: tuple[str, ...] | list[str] = RENDERERS,
             tiers: tuple[str, ...] | list[str] | set[str] = ("d0", "std"),
             batch: int = BATCH, top_n: int = 5,
             log: Callable[[str], None] = lambda s: print(s, flush=True)) -> int:
    """Score the grid for one model, appending rows to ``out_path``; returns rows written."""
    endpoint = endpoint or default_endpoint()
    chat_family(kind)  # validate early
    for rd in renderers:
        if rd not in RENDERERS:
            raise ValueError(f"unknown renderer {rd!r}; choose from {RENDERERS}")
    if not 1 <= batch <= 64:
        raise ValueError(f"batch must be in [1, 64] (server LOGPROB_MAX_BATCH), got {batch}")
    tier_set = set(tiers)
    items = [it for it in items if it.tier in tier_set]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path)

    # health check — fail fast on a wrong/absent server
    try:
        h = requests.get(f"{endpoint.rstrip('/')}/health", timeout=30).json()
    except Exception as exc:  # noqa: BLE001
        raise LogprobsError(f"no healthy server at {endpoint}: {exc}") from exc
    log(f"server: {h}")

    jobs: list[tuple[Item, str, int]] = [
        (it, rd, ns) for rd in renderers for ns in SHOT_CONDS for it in items
        if (it.item_id, rd, ns) not in done]
    log(f"{len(jobs)} scoring jobs (batch {batch})")

    n_done, t0 = 0, time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for k in range(0, len(jobs), batch):
            chunk = jobs[k: k + batch]
            payload: list[dict[str, Any]] = []
            meta: list[tuple[Item, str, int, Optional[str]]] = []
            for it, rd, ns in chunk:
                p, chat_format = build_payload_item(it, rd, ns, shots_all, kind)
                payload.append(p)
                meta.append((it, rd, ns, chat_format))
            results = request_batch(endpoint, payload, top_n)
            by_id = {r["id"]: r for r in results}
            for it, rd, ns, chat_format in meta:
                jid = job_id(it, rd, ns)
                if jid not in by_id:
                    raise LogprobsError(f"server returned no result for {jid}")
                row = score_row(it, by_id[jid], rd, ns, model, chat_format)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += len(chunk)
            if (k // batch) % 20 == 0:
                rate = n_done / max(time.time() - t0, 1e-6)
                log(f"  {n_done}/{len(jobs)} ({rate:.0f}/s, "
                    f"eta {(len(jobs) - n_done) / max(rate, 1e-6) / 60:.1f}m)")

    log(f"done: {n_done} rows -> {out_path}")
    return n_done


def run_grid_files(model: str, kind: str, items_path: Path, shots_path: Path,
                   out_path: Path, **kw: Any) -> int:
    """``run_grid`` from an emitted items.jsonl + shots.json."""
    return run_grid(model, kind, load_items(items_path), load_items(shots_path),
                    out_path, **kw)
