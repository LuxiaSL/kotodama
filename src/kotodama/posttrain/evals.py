"""Generative instruction evals for transcript-format assistants, through a server's /v1/completions.

Every prompt is a plain ``User:`` / ``Assistant:`` transcript (``transcript.render_prompt``), the format the
assistant track trains on; decoding is greedy with no repetition penalty, stopped at the next ``User:`` tag or
end-of-text. Tasks:

* ``ifeval`` — the 541 IFEval prompts as a single user turn; scored with IFEval's own instruction checkers (from
  lm-eval, ``pip install lm-eval[ifeval]``): prompt- and instruction-level accuracy, strict and loose.
* ``gsm8k`` — the 1,319 GSM8K test problems, 8-shot as a multi-turn transcript: eight training problems with their
  reference solutions (calculator annotations removed, ending "The answer is N.") as prior exchanges. Exact match
  on the number after the first "The answer is" (else the last number in the reply).

    python -m kotodama.posttrain.evals --endpoint http://host:port --tasks ifeval,gsm8k --out_dir <dir> \\
        [--expect_name <served name>] [--limit 100]

Writes ``<out_dir>/<task>.jsonl`` (prompt, reply, per-item scores) and ``<out_dir>/summary.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from kotodama.posttrain.transcript import SEP, Transcript, Turn, render_prompt

logger = logging.getLogger(__name__)

USER, ASSISTANT = "User", "Assistant"
STOPS = [f"{SEP}{USER}:"]
GSM8K_SHOTS = 8
_CALC = re.compile(r"<<[^>]*>>")
_ANSWER = re.compile(r"[Tt]he (?:final )?answer is[:\s]*\$?\s*(-?[\d,]*\.?\d+)")
_NUMBER = re.compile(r"-?[\d,]*\.?\d+")

Complete = Callable[[str, int], Awaitable[str]]


def prompt_for(exchanges: list[tuple[str, str]], question: str) -> str:
    """The transcript up to an open Assistant turn: prior (user, assistant) exchanges, then the question."""
    turns: list[Turn] = []
    for q, a in exchanges:
        turns += [Turn(USER, q), Turn(ASSISTANT, a)]
    turns.append(Turn(USER, question))
    return render_prompt(Transcript(tuple(turns)), ASSISTANT)


def cut_reply(text: str) -> str:
    """The server returns text through the stop string; keep the assistant's turn only (a tag after a single
    newline also ends it)."""
    for s in STOPS + [f"\n{USER}:"]:
        i = text.find(s)
        if i >= 0:
            text = text[:i]
    return text.strip()


# ── GSM8K ────────────────────────────────────────────────────────────────────


def gsm8k_solution(answer: str) -> tuple[str, str]:
    """A reference solution ("... #### 72") -> (worked solution ending "The answer is 72.", "72")."""
    body, _, final = answer.partition("####")
    final = final.strip().replace(",", "")
    steps = _CALC.sub("", body).strip()
    return f"{steps}\nThe answer is {final}.", final


def gsm8k_extract(reply: str) -> Optional[str]:
    """The first stated answer ("The answer is N" — a model that keeps writing past it often starts a new problem),
    else the last number in the reply."""
    m = _ANSWER.search(reply)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = _NUMBER.findall(reply)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def _same_number(a: Optional[str], b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


async def run_gsm8k(complete: Complete, test: list[dict[str, Any]], train: list[dict[str, Any]],
                    concurrency: int, max_tokens: int = 512) -> tuple[list[dict[str, Any]], dict[str, float]]:
    shots = [(r["question"], gsm8k_solution(r["answer"])[0]) for r in train[:GSM8K_SHOTS]]
    sem = asyncio.Semaphore(concurrency)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        prompt = prompt_for(shots, row["question"])
        async with sem:
            reply = cut_reply(await complete(prompt, max_tokens))
        target = gsm8k_solution(row["answer"])[1]
        got = gsm8k_extract(reply)
        return {"question": row["question"], "reply": reply, "target": target, "extracted": got,
                "correct": _same_number(got, target)}

    items = await asyncio.gather(*(one(r) for r in test))
    return items, {"gsm8k_exact_match": sum(i["correct"] for i in items) / max(len(items), 1)}


# ── IFEval ───────────────────────────────────────────────────────────────────


async def run_ifeval(complete: Complete, docs: list[dict[str, Any]], concurrency: int,
                     max_tokens: int = 1280) -> tuple[list[dict[str, Any]], dict[str, float]]:
    from lm_eval.tasks.ifeval.utils import process_results  # lm-eval[ifeval]

    sem = asyncio.Semaphore(concurrency)

    async def one(doc: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            reply = cut_reply(await complete(prompt_for([], doc["prompt"]), max_tokens))
        return {"key": doc["key"], "prompt": doc["prompt"], "reply": reply, **process_results(doc, [reply])}

    items = await asyncio.gather(*(one(d) for d in docs))
    n = max(len(items), 1)
    inst = lambda k: [x for i in items for x in i[k]]  # noqa: E731
    summary = {
        "ifeval_prompt_strict": sum(i["prompt_level_strict_acc"] for i in items) / n,
        "ifeval_prompt_loose": sum(i["prompt_level_loose_acc"] for i in items) / n,
        "ifeval_inst_strict": sum(inst("inst_level_strict_acc")) / max(len(inst("inst_level_strict_acc")), 1),
        "ifeval_inst_loose": sum(inst("inst_level_loose_acc")) / max(len(inst("inst_level_loose_acc")), 1),
    }
    return items, summary


# ── client + cli ─────────────────────────────────────────────────────────────


def make_client(endpoint: str, timeout: float = 600.0, transport: Any = None) -> tuple[Complete, Callable[[], Awaitable[None]]]:
    import httpx

    client = httpx.AsyncClient(timeout=timeout, transport=transport)
    url = endpoint.rstrip("/") + "/v1/completions"

    async def complete(prompt: str, max_tokens: int) -> str:
        body = {"prompt": prompt, "n": 1, "max_tokens": max_tokens, "temperature": 0.0, "top_p": 0.0,
                "frequency_penalty": 0.0, "stop": STOPS}
        for attempt in range(4):
            try:
                r = await client.post(url, json=body)
                r.raise_for_status()
                return r.json()["choices"][0]["text"]
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                if attempt == 3:
                    raise
                logger.warning("completion request failed (%s); retrying", e)
                await asyncio.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    return complete, client.aclose


def load_task_data(task: str, limit: int) -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    if task == "ifeval":
        docs = list(load_dataset("google/IFEval", split="train"))
        return {"docs": docs[:limit] if limit else docs}
    if task == "gsm8k":
        test = list(load_dataset("openai/gsm8k", "main", split="test"))
        train = list(load_dataset("openai/gsm8k", "main", split="train"))
        return {"test": test[:limit] if limit else test, "train": train}
    raise ValueError(f"unknown task {task!r} (ifeval, gsm8k)")


async def run(endpoint: str, tasks: list[str], out_dir: Path, limit: int, concurrency: int,
              transport: Any = None, data: Optional[dict[str, dict[str, list[dict[str, Any]]]]] = None) -> dict[str, float]:
    complete, close = make_client(endpoint, transport=transport)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, float] = {}
    try:
        for task in tasks:
            d = (data or {}).get(task) or load_task_data(task, limit)
            if task == "ifeval":
                items, s = await run_ifeval(complete, d["docs"], concurrency)
            else:
                items, s = await run_gsm8k(complete, d["test"], d["train"], concurrency)
            with (out_dir / f"{task}.jsonl").open("w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            summary.update(s)
            summary[f"{task}_n"] = len(items)
            logger.info("%s: %s", task, {k: round(v, 4) for k, v in s.items()})
    finally:
        await close()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [evals] %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--tasks", default="ifeval,gsm8k")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N items per task (0 = all)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--expect_name", default=None,
                    help="refuse unless the server's /info name equals this (never score the wrong model)")
    a = ap.parse_args(argv)
    if a.expect_name:
        import httpx

        try:
            name = httpx.get(a.endpoint.rstrip("/") + "/info", timeout=10).json().get("name")
        except (httpx.HTTPError, ValueError) as e:
            print(f"cannot read {a.endpoint}/info: {e}", file=sys.stderr)
            return 3
        if name != a.expect_name:
            print(f"REFUSING: {a.endpoint} serves {name!r}, not {a.expect_name!r}", file=sys.stderr)
            return 3
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    summary = asyncio.run(run(a.endpoint, tasks, Path(a.out_dir), a.limit, a.concurrency))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
