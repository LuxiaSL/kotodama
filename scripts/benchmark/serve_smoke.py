#!/usr/bin/env python3
"""End-to-end check of a running server or gateway (one endpoint): the request surface a generation client relies
on, then a concurrent burst for aggregate throughput.

Checks (each reported pass/fail, with the measured values):
  * ``/info`` answers and names what is served;
  * a seeded fan (``n`` candidates, ``seed``) reproduces exactly when repeated, and a different seed differs;
  * a prompt sent as token ids returns the same completion as the same prompt sent as text (greedy);
  * ``/score_tokens`` scores a completion;
  * a raw ``injection`` (steering) changes a greedy completion, and the unsteered request after it is unchanged;
  * an over-long prompt is a 4xx, not a 5xx;
  * a burst of ``--burst`` concurrent requests of ``--burst_n`` candidates: aggregate completion tokens/s and latency
    percentiles (each request pinned by an OpenAI ``user`` key, as a tree-search client pins its trees).

    tools/run_py.sh scripts/benchmark/serve_smoke.py --endpoint http://127.0.0.1:2390 [--out <json>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

PROMPT = ("The harbor was quiet that evening. Mara walked down to the water, where the boats knocked against the "
          "pier, and she thought about the letter in her coat pocket.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--burst", type=int, default=32, help="concurrent requests in the throughput burst")
    ap.add_argument("--burst_n", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=1536, help="hidden size, for the steering injection")
    ap.add_argument("--site", type=int, default=46, help="effective layer of the steering injection")
    ap.add_argument("--dose", type=float, default=800.0, help="norm of the steering injection's random row")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    url = a.endpoint.rstrip("/")
    res: dict = {"endpoint": url, "checks": {}}
    ok = True

    def check(name: str, passed: bool, **info) -> None:
        nonlocal ok
        ok &= bool(passed)
        res["checks"][name] = {"pass": bool(passed), **info}
        print(f"{'PASS' if passed else 'FAIL'} {name} {json.dumps(info)[:300]}", flush=True)

    with httpx.Client(timeout=600) as http:
        info = http.get(f"{url}/info").json()
        check("info", bool(info.get("name")), name=info.get("name"))

        def complete(**kw) -> dict:
            body = {"prompt": PROMPT, "max_tokens": 32, "temperature": 0.75, "top_p": 0, "frequency_penalty": 0.2,
                    "return_token_ids": True, **kw}
            r = http.post(f"{url}/v1/completions", json=body)
            return {"status": r.status_code, "json": r.json() if r.headers.get("content-type", "").startswith(
                "application/json") else r.text}

        def ids_of(r: dict) -> list[list[int]]:
            return [c["token_ids"]["completion"] for c in r["json"]["choices"]]

        a1, a2, b1 = complete(n=8, seed=11), complete(n=8, seed=11), complete(n=8, seed=12)
        if all(x["status"] == 200 for x in (a1, a2, b1)):
            same, diff = ids_of(a1) == ids_of(a2), ids_of(a1) != ids_of(b1)
            distinct = len({tuple(x) for x in ids_of(a1)})
            check("seeded_fan", same and diff and distinct > 1, repeat_equal=same, other_seed_differs=diff,
                  distinct_candidates=distinct)
        else:
            check("seeded_fan", False, status=[a1["status"], a2["status"], b1["status"]], body=str(a1["json"])[:200])

        g_text = complete(temperature=0)
        prompt_ids = g_text["json"]["choices"][0]["token_ids"]["prompt"]
        g_ids = complete(temperature=0, prompt=prompt_ids)
        check("token_id_prompt", g_ids["status"] == 200 and ids_of(g_ids) == ids_of(g_text),
              prompt_tokens=len(prompt_ids))

        comp = ids_of(g_text)[0]
        r = http.post(f"{url}/score_tokens", json={"items": [{"id": "0", "prompt_ids": prompt_ids,
                                                              "completion_ids": comp}]})
        scored = r.status_code == 200 and r.json()["results"][0]["n_tokens"] == len(comp)
        check("score_tokens", scored, status=r.status_code,
              mean_nll=r.json()["results"][0].get("mean_nll") if r.status_code == 200 else None)

        import random
        rng = random.Random(0)
        v = [rng.gauss(0, 1) for _ in range(a.hidden)]
        norm = sum(x * x for x in v) ** 0.5
        v = [a.dose * x / norm for x in v]
        steered = complete(temperature=0, injection={"sites": [a.site], "vectors": [v], "rule": "S"})
        after = complete(temperature=0)
        check("injection", steered["status"] == 200 and ids_of(steered) != ids_of(g_text)
              and ids_of(after) == ids_of(g_text), status=steered["status"],
              unsteered_after_unchanged=ids_of(after) == ids_of(g_text) if after["status"] == 200 else None)

        long = complete(prompt="word " * 9000, max_tokens=4)
        check("too_long_is_4xx", 400 <= long["status"] < 500, status=long["status"])

    async def burst() -> dict:
        lat, toks = [], 0
        async with httpx.AsyncClient(timeout=1200) as cli:
            async def one(i: int) -> None:
                nonlocal toks
                t0 = time.perf_counter()
                r = await cli.post(f"{url}/v1/completions", json={
                    "prompt": f"Story {i}. " + PROMPT, "n": a.burst_n, "max_tokens": a.max_tokens,
                    "temperature": 0.75, "top_p": 0, "user": f"smoke/tree-{i % 8}"})
                lat.append(time.perf_counter() - t0)
                if r.status_code == 200:
                    toks += r.json()["usage"]["completion_tokens"]
            t0 = time.perf_counter()
            await asyncio.gather(*(one(i) for i in range(a.burst)))
            wall = time.perf_counter() - t0
        lat.sort()
        return {"requests": a.burst, "n": a.burst_n, "wall_s": round(wall, 2), "completion_tokens": toks,
                "tok_per_s": round(toks / wall, 1), "p50_s": round(statistics.median(lat), 2),
                "p90_s": round(lat[int(0.9 * (len(lat) - 1))], 2)}

    res["burst"] = asyncio.run(burst())
    print("burst", json.dumps(res["burst"]), flush=True)
    res["ok"] = ok
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
