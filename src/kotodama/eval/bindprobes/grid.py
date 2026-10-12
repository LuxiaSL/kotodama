"""Bindprobe runner: score every (item, condition) against ``/logprobs``.

The HTTP path is the riders' (``riders.grid.request_batch`` / ``check_health`` /
``default_endpoint``) and the row's rank/margin come from the riders'
closed-candidate read (``riders.grid.read_result``), so a bindprobe row means
what a rider row means. All three families are plain-text prompts, so there is
no model "kind": koto and HF models get the same strings.

Row = ``{model, item_id, family, cell, cell_key, condition, nshots, <controls>,
gold, <read_result fields>, pick, error_type, error_types}``; ``error_type`` is
``None`` on a correct row. Crash-safe: appends to the output JSONL and skips
(item, condition) keys already scored.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from kotodama.eval.bindprobes.base import Item
from kotodama.eval.bindprobes.errors import error_types, picked
from kotodama.eval.bindprobes.items import (CONDITIONS, COND_NSHOTS, FAMILY_NAMES, build_prompt,
                                            load_items)
from kotodama.eval.riders.grid import (BATCH, DEFAULT_ENDPOINT, LogprobsError, check_health,
                                       default_endpoint, read_result, request_batch)

__all__ = ["BATCH", "DEFAULT_ENDPOINT", "LogprobsError", "build_payload_item", "job_id",
           "run_grid", "run_grid_files", "score_row"]

# per-family controls copied from item.meta onto every row (aggregate reads them)
ROW_CONTROLS: dict[str, tuple[str, ...]] = {
    "attr": ("qpos", "recency_control", "n_distractors"),
    "codevar": ("gold_is_last_literal", "n_stale", "n_reference"),
    "dialogue": ("qkind", "n_pronoun_ops", "gold_is_last_mention"),
}


def job_id(item: Item, condition: str) -> str:
    return f"{item.item_id}|{condition}"


def build_payload_item(item: Item, condition: str) -> dict[str, Any]:
    """One ``/logprobs`` request item: candidates in the scored form ``" " + c``."""
    return {"id": job_id(item, condition), "prompt": build_prompt(item, condition),
            "candidates": [" " + c for c in item.candidates]}


def score_row(item: Item, res: Mapping[str, Any], condition: str, model: str
              ) -> dict[str, Any]:
    """One grid row from one ``/logprobs`` result (ties rank against gold)."""
    read = read_result(dict(res), " " + item.gold, item.item_id)
    row: dict[str, Any] = {
        "model": model, "item_id": item.item_id, "family": item.family,
        "cell": list(item.cell), "cell_key": item.cell_key, "condition": condition,
        "nshots": COND_NSHOTS[condition],
        **{k: item.meta.get(k) for k in ROW_CONTROLS.get(item.family, ())},
        "gold": item.gold, **read,
    }
    pick = picked(row)
    wrong = read["gold_rank_cand"] != 1
    types = error_types(item, pick) if wrong and pick != item.gold else []
    # a tie with gold ranks against gold but can argmax to gold: type it "tie"
    row.update(pick=pick, error_type=(types[0] if types else ("tie" if wrong else None)),
               error_types=types)
    return row


def _done_keys(out_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["item_id"], r["condition"]))
        print(f"resume: {len(done)} rows already scored", flush=True)
    return done


def run_grid(model: str, items: Sequence[Item], out_path: Path,
             endpoint: Optional[str] = None,
             conditions: Sequence[str] = CONDITIONS,
             families: Sequence[str] = FAMILY_NAMES,
             batch: int = BATCH, top_n: int = 5,
             log: Callable[[str], None] = lambda s: print(s, flush=True)) -> int:
    """Score items x conditions for one model, appending rows; returns rows written."""
    endpoint = endpoint or default_endpoint()
    for c in conditions:
        if c not in CONDITIONS:
            raise ValueError(f"unknown condition {c!r}; choose from {CONDITIONS}")
    for f in families:
        if f not in FAMILY_NAMES:
            raise ValueError(f"unknown family {f!r}; choose from {FAMILY_NAMES}")
    if not 1 <= batch <= 64:
        raise ValueError(f"batch must be in [1, 64] (server LOGPROB_MAX_BATCH), got {batch}")
    fams = set(families)
    items = [it for it in items if it.family in fams]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out_path)
    check_health(endpoint, log)

    jobs = [(it, c) for c in conditions for it in items if (it.item_id, c) not in done]
    log(f"{len(jobs)} scoring jobs (batch {batch})")
    n_done, t0 = 0, time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for k in range(0, len(jobs), batch):
            chunk = jobs[k: k + batch]
            payload = [build_payload_item(it, c) for it, c in chunk]
            by_id = {r["id"]: r for r in request_batch(endpoint, payload, top_n)}
            for it, c in chunk:
                jid = job_id(it, c)
                if jid not in by_id:
                    raise LogprobsError(f"server returned no result for {jid}")
                fh.write(json.dumps(score_row(it, by_id[jid], c, model),
                                    ensure_ascii=False) + "\n")
            fh.flush()
            n_done += len(chunk)
            if (k // batch) % 20 == 0:
                rate = n_done / max(time.time() - t0, 1e-6)
                log(f"  {n_done}/{len(jobs)} ({rate:.0f}/s, "
                    f"eta {(len(jobs) - n_done) / max(rate, 1e-6) / 60:.1f}m)")
    log(f"done: {n_done} rows -> {out_path}")
    return n_done


def run_grid_files(model: str, items_path: Path, out_path: Path, **kw: Any) -> int:
    return run_grid(model, load_items(items_path), out_path, **kw)
