#!/usr/bin/env python3
"""Batched row decode on a real checkpoint (GPU): throughput, agreement with the single-stream engine, and the
cross-request batcher under loom-like load.

Builds the model, engine and row decoders through the server's own load_model (the serving build and warmup), then
runs the ``--sections`` asked for:
  * ``width``: single-stream decode ms/token (the engine's fused sampled step) and batched ms/step at each bucket
    width, as tokens/s;
  * ``agreement``: one forced continuation fed to every row (eager batched forward) vs the engine's step, per-step
    KL(engine || row) and top-1 agreement (bf16 kernels differ between a B-row GEMM and a GEMV, so the bar is
    distributional, not bitwise), and the compiled batched step at a near-zero temperature vs the engine's greedy
    continuation (matching prefix);
  * ``xparity``: the same two checks with the rows holding DIFFERENT prompts of different lengths (every row at its
    own position), each row against the single-stream engine on its own prompt;
  * ``load``: concurrent ``/v1/completions`` requests through the server app (in-process ASGI, so every request
    takes the real handler, scheduler, engine lock and generation thread) with distinct long prompts, n candidates
    and stop sequences, per serving mode: ``current`` (the n-row decoder, one request at a time), ``xbatch``
    (cross-request batches) and ``xbatch_refill`` (rows refilled as requests finish). Reports aggregate completion
    tokens/s and request latency percentiles per mode and concurrency.

``--rows`` takes a list (one compiled decoder per bucket) and ``--context_tokens`` a list of prompt lengths (the
dialogue repeated to that many tokens), so batch width is measured where the loom decodes: deep in a long context.
``--prompts_jsonl`` takes transcript rows (``{"header", "turns": [{"speaker", "text"}], "cast"?}``, the format of
kotodama.posttrain.transcript and of loom paths); each becomes the prompt for the next turn, cut from the front to
fit ``--prompt_tokens``, with the cast's stop sequences. Without it, prompts are built from a fixed dialogue.

Usage (node, GPU):
  tools/run_py.sh scripts/benchmark/multi_decode_bench.py --checkpoint <step.pt> --model_size 1b-moonshot-loop \\
      --rope_fp32 --rows 8 [--context_tokens 0,3000] [--out results/multi_decode.json]
  tools/run_py.sh scripts/benchmark/multi_decode_bench.py ... --rows 8,16,32,64 --sections xparity,load \\
      --prompts_jsonl <paths.jsonl> --steer_npz <bank.npz>,<norms.npz> --load_concurrency 4,8 --load_requests 96
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kotodama.model.decode_engine import SamplingParams  # noqa: E402
from kotodama.posttrain.transcript import Transcript, parse_row, render_prompt, stop_sequences  # noqa: E402
from kotodama.serve import server  # noqa: E402
from kotodama.serve.xbatch import XBatchRunner, XBatchScheduler  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("multi_decode_bench")
PROMPT = ("Participants: Mara, a ferry engineer; Teo, her younger brother.\nSetting: the night crossing, 1987.\n\n"
          "Mara: You gave away Dad's watch.\n\nTeo: He didn't even know me.\n\nMara:")


def timed(fn, n_iter: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter


def kl_top1(ref: list[torch.Tensor], got: list[torch.Tensor]) -> tuple[list[float], list[float]]:
    """Per step: max over rows of KL(ref || row), and the fraction of rows whose argmax matches the reference.
    ``ref[t]`` is (V,) or (B, V) (one reference per row); ``got[t]`` is (B, V)."""
    kl, top1 = [], []
    for r, b in zip(ref, got):
        r = r.expand_as(b)
        p = F.log_softmax(r, -1)
        q = F.log_softmax(b, -1)
        kl.append(float((p.exp() * (p - q)).sum(-1).max()))
        top1.append(float((b.argmax(-1) == r.argmax(-1)).float().mean()))
    return kl, top1


def load_prompts(tok, path: str | None, count: int, lo: int, hi: int, seed: int) -> list[tuple[str, list[str]]]:
    """``count`` distinct (prompt, stops) pairs of lo..hi tokens: transcript rows from ``path`` (leading turns
    dropped until the prompt fits under hi; rows that cannot reach lo are skipped), else the fixed dialogue padded
    with numbered exchanges."""
    rng = random.Random(seed)
    out: list[tuple[str, list[str]]] = []
    if path:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if len(out) >= count:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tr = parse_row(row)
                if tr is None or len(tr.turns) < 2:
                    continue
                cast = row.get("cast") or list(dict.fromkeys(t.speaker for t in tr.turns))
                turns = list(tr.turns)
                nxt = turns[-2].speaker
                target = rng.randint(lo, hi)
                while turns:
                    text = render_prompt(Transcript(tuple(turns), tr.header), nxt)
                    n_tok = len(tok.encode(text))
                    if n_tok <= target:
                        break
                    turns = turns[1:]
                if turns and n_tok >= lo:
                    out.append((text, stop_sequences(cast)))
        if len(out) < count:
            raise SystemExit(f"{path}: only {len(out)} transcripts reach {lo}..{hi} tokens (need {count})")
        return out
    lines = ["Mara: You gave away Dad's watch.", "Teo: He didn't even know me.", "Mara: That isn't the point.",
             "Teo: Then what is?", "Mara: It was the last thing of his on this boat.", "Teo: The boat is his too."]
    for i in range(count):
        target = rng.randint(lo, hi)
        body = [PROMPT.split("\n\n")[0]]
        k = i
        while len(tok.encode("\n\n".join(body))) < target:
            body.append(lines[k % len(lines)] + f" ({k})")
            k += 7
        out.append(("\n\n".join(body) + "\n\nMara:", ["\n\nMara:", "\n\nTeo:"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_size", default="1b-moonshot-loop")
    ap.add_argument("--rope_fp32", action="store_true")
    ap.add_argument("--rows", default="8", help="comma list of batch widths, e.g. 8,16,32,64")
    ap.add_argument("--context_tokens", default="0", help="comma list of prompt lengths (0 = the short prompt)")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--sections", default="width,agreement", help="comma list: width,agreement,xparity,load")
    ap.add_argument("--steer_npz", default=None, help="the server's --steer-npz; load requests then steer")
    ap.add_argument("--steer", default="vrep:-0.30", help="name:alpha[,name:alpha] sent with every load request")
    ap.add_argument("--steer_rule", default="N")
    ap.add_argument("--prompts_jsonl", default=None)
    ap.add_argument("--prompt_tokens", default="1500,3000", help="lo,hi prompt length for xparity and load")
    ap.add_argument("--xparity_prompts", type=int, default=16, help="distinct prompts across the widest bucket")
    ap.add_argument("--load_requests", type=int, default=96, help="requests per (mode, concurrency) cell")
    ap.add_argument("--load_concurrency", default="4,8", help="comma list of requests in flight")
    ap.add_argument("--load_modes", default="current,xbatch,xbatch_refill")
    ap.add_argument("--load_n", type=int, default=8)
    ap.add_argument("--load_max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    widths = sorted({int(x) for x in a.rows.split(",") if x.strip()})
    contexts = [int(x) for x in a.context_tokens.split(",") if x.strip()]
    sections = {x.strip() for x in a.sections.split(",") if x.strip()}
    lo, hi = (int(x) for x in a.prompt_tokens.split(","))
    server._rope_fp32_override = a.rope_fp32
    _model, tok = server.load_model(a.checkpoint, "cuda", mode="base", model_size=a.model_size, engine="fast",
                                    prefix_cache=True, max_seq_len=4096, multi_rows=tuple(widths),
                                    steer_npz=a.steer_npz)
    server._model, server._tokenizer = _model, tok
    eng = server._engine
    assert eng is not None and all(server._multi.get(w) is not None for w in widths)
    dec = server._multi[widths[-1]]
    base_ids = tok.encode(PROMPT)
    body = tok.encode(PROMPT.split("\n\n", 1)[1] + " Fine.\n\nTeo: Fine.\n\n")

    def prompt_ids(n: int) -> torch.Tensor:
        out = list(base_ids)
        while n and len(out) < n:
            out = body + out
        return torch.tensor([out[-n:] if n else out], device="cuda")

    ids = prompt_ids(0)
    params = SamplingParams(temperature=a.temperature, repetition_penalty=a.repetition_penalty)
    res: dict = {"rows": widths, "context_tokens": contexts, "steps": a.steps, "sections": sorted(sections)}

    def run_on_worker(fn):
        return server._generate_executor.submit(fn).result()  # compiled graphs live on the serving thread

    def bench() -> dict:
        out: dict = {}
        with torch.inference_mode():
            for ctx in contexts:
                cids = prompt_ids(ctx)
                tag = f"ctx{cids.shape[1]}"
                logits = eng.prefill(cids)
                t = eng.sample_first(logits, params)
                for _ in range(8):
                    t = eng.step(t, params)
                single = 1000 * timed(lambda: eng.step(eng.cur_token, params), a.steps)
                out[f"{tag}_single_ms_per_token"] = single
                for w in widths:
                    d = server._multi[w]
                    logits = eng.prefill(cids)
                    d.start(logits, params)
                    for _ in range(8):
                        d.step(params)
                    ms = 1000 * timed(lambda: d.step(params), a.steps)
                    out[f"{tag}_rows{w}_ms_per_step"] = ms
                    out[f"{tag}_rows{w}_tok_s"] = w * 1000 / ms
                    out[f"{tag}_rows{w}_step_vs_single"] = ms / single
                logger.info("%s: %s", tag, {k: round(v, 2) for k, v in out.items() if k.startswith(tag)})
        return out

    def greedy_of(cids: torch.Tensor, n: int) -> list[int]:
        logits = eng.prefill(cids)
        greedy = [int(logits.argmax())]
        for _ in range(n - 1):
            greedy.append(int(eng.step_logits(torch.tensor([[greedy[-1]]], device="cuda"))[0, -1].argmax()))
        return greedy

    def agreement() -> dict:
        with torch.inference_mode():
            # a model-written continuation (the engine's greedy) is a fairer probe than random ids
            greedy = greedy_of(ids, 64)
            forced = torch.tensor(greedy, device="cuda")
            l0 = eng.prefill(ids)
            ref = [l0[0].float()]
            for t in forced[:-1]:
                ref.append(eng.step_logits(t.view(1, 1))[0, -1].float().clone())
            l0 = eng.prefill(ids)
            dec.start(l0, params)
            rows = [l0[0].float().expand(dec.rows, -1)]
            for t in forced[:-1]:
                rows.append(dec.step_logits(t.view(1, 1).expand(dec.rows, 1).contiguous()).float())
            kl, top1 = kl_top1(ref, rows)
            # compiled batched step, near-greedy, vs the engine's greedy continuation
            cold = SamplingParams(temperature=1e-4, repetition_penalty=1.0)
            l0 = eng.prefill(ids)
            toks = [dec.start(l0, cold).view(-1).tolist()]
            for _ in range(31):
                toks.append(dec.step(cold).view(-1).tolist())
            match = []
            for r in range(dec.rows):
                seq = [s[r] for s in toks]
                k = next((i for i, (x, y) in enumerate(zip(seq, greedy)) if x != y), len(seq))
                match.append(k)
        return {"kl_max_per_step_mean": sum(kl) / len(kl), "kl_max_overall": max(kl),
                "top1_agreement": sum(top1) / len(top1), "compiled_greedy_match_prefix": match}

    def xparity() -> dict:
        """Rows hold different prompts (row r: prompt r mod P), each forced along its own prompt's greedy."""
        steps = 32
        texts = load_prompts(tok, a.prompts_jsonl, a.xparity_prompts, lo, hi, a.seed)
        pids = [torch.tensor([tok.encode(t)], device="cuda") for t, _ in texts]
        P, B = len(pids), dec.rows
        with torch.inference_mode():
            greedy = [greedy_of(p, steps) for p in pids]
            refs = []  # refs[p][t]: the engine's logits for prompt p at step t
            for p, g in zip(pids, greedy):
                l0 = eng.prefill(p)
                seq = [l0[0].float()]
                for t in g[:-1]:
                    seq.append(eng.step_logits(torch.tensor([[t]], device="cuda"))[0, -1].float().clone())
                refs.append(seq)
            cold = SamplingParams(temperature=1e-4, repetition_penalty=1.0)
            first: dict[int, torch.Tensor] = {}
            for p in range(P):
                rows = list(range(p, B, P))
                l0 = eng.prefill(pids[p])
                dec.admit(rows, l0, cold)
                for r in rows:
                    first[r] = l0[0].float()
            got = [torch.stack([first[r] for r in range(B)])]
            forced = torch.tensor([greedy[r % P] for r in range(B)], device="cuda")  # (B, steps)
            for t in range(steps - 1):
                got.append(dec.step_logits(forced[:, t:t + 1].contiguous()).float())
            ref = [torch.stack([refs[r % P][t] for r in range(B)]) for t in range(steps)]
            kl, top1 = kl_top1(ref, got)
            # compiled near-greedy with every row at its own prompt
            toks = []
            for p in range(P):
                rows = list(range(p, B, P))
                dec.admit(rows, eng.prefill(pids[p]), cold)
            toks.append(dec.cur_tokens.view(-1).tolist())
            for _ in range(steps - 1):
                toks.append(dec.step(cold).view(-1).tolist())
            match = []
            for r in range(B):
                seq = [s[r] for s in toks]
                match.append(next((i for i, (x, y) in enumerate(zip(seq, greedy[r % P])) if x != y), len(seq)))
        return {"xparity_prompts": P, "xparity_prompt_tokens": [int(p.shape[1]) for p in pids],
                "xparity_rows": B, "xparity_kl_max_per_step_mean": sum(kl) / len(kl),
                "xparity_kl_max_overall": max(kl), "xparity_top1_agreement": sum(top1) / len(top1),
                "xparity_compiled_greedy_match_prefix": match}

    def load() -> dict:
        import httpx

        texts = load_prompts(tok, a.prompts_jsonl, a.load_requests, lo, hi, a.seed + 1)
        prompt_lens = [len(tok.encode(t)) for t, _ in texts]
        steer = []
        for item in a.steer.split(","):
            if a.steer_npz and item.strip():
                name, alpha = item.split(":")
                steer.append({"name": name, "alpha": float(alpha)})
        all_dec = dict(server._multi)
        xrows = tuple(w for w in widths if w >= 8)

        def set_mode(mode: str) -> None:
            server._multi = {8: all_dec[8]} if 8 in all_dec else {}
            server._xbatch = None
            if mode.startswith("xbatch"):
                runner = XBatchRunner({w: all_dec[w] for w in xrows}, prefill=server._engine_prefill,
                                      set_steer=eng.set_steer,
                                      decode=lambda i: tok.decode(i, skip_special_tokens=True),
                                      stop_ids=server._stop_token_ids, max_seq_len=server._max_seq_len,
                                      device=torch.device("cuda"), refill=mode == "xbatch_refill")
                server._xbatch = XBatchScheduler(runner, server._engine_lock, server._run_generation)

        async def cell(conc: int, reqs: list[int]) -> dict:
            # an asyncio lock binds to the loop it is first contended on, and every cell runs its own loop
            server._engine_lock = asyncio.Lock()
            if server._xbatch is not None:
                server._xbatch.lock = server._engine_lock
            transport = httpx.ASGITransport(app=server.app)
            lat: list[float] = []
            comp: list[int] = []
            qi = list(reqs)
            async with httpx.AsyncClient(transport=transport, base_url="http://bench", timeout=1800) as client:
                async def worker() -> None:
                    while qi:
                        i = qi.pop(0)
                        text, stops = texts[i]
                        body = {"prompt": text, "n": a.load_n, "max_tokens": a.load_max_tokens,
                                "temperature": a.temperature, "top_p": 0.0,
                                "frequency_penalty": max(0.0, a.repetition_penalty - 1.0), "stop": stops,
                                "return_token_ids": True}
                        if steer:
                            body["vectors"], body["steer_rule"] = steer, a.steer_rule
                        t0 = time.perf_counter()
                        r = await client.post("/v1/completions", json=body)
                        r.raise_for_status()
                        lat.append(time.perf_counter() - t0)
                        comp.extend(len(c["token_ids"]["completion"]) for c in r.json()["choices"])

                t0 = time.perf_counter()
                await asyncio.gather(*(worker() for _ in range(conc)))
                wall = time.perf_counter() - t0
            lat.sort()
            q = statistics.quantiles(lat, n=10) if len(lat) >= 2 else lat * 9
            return {"requests": len(lat), "wall_s": round(wall, 2), "completion_tokens": sum(comp),
                    "tok_s": round(sum(comp) / wall, 1), "req_s": round(len(lat) / wall, 2),
                    "lat_p50_s": round(statistics.median(lat), 3), "lat_p90_s": round(q[8], 3),
                    "completion_median": statistics.median(comp), "completion_mean": round(sum(comp) / len(comp), 1)}

        out: dict = {"load_prompt_tokens": {"min": min(prompt_lens), "median": statistics.median(prompt_lens),
                                            "max": max(prompt_lens)},
                     "load_n": a.load_n, "load_steer": steer, "xbatch_rows": list(xrows)}
        idx = list(range(len(texts)))
        for mode in [m.strip() for m in a.load_modes.split(",") if m.strip()]:
            set_mode(mode)
            asyncio.run(cell(2, idx[:4]))  # warm the path (first-request lazy work) outside the measurement
            for conc in [int(x) for x in a.load_concurrency.split(",") if x.strip()]:
                if server._xbatch is not None:
                    server._xbatch.runner.stats.update({k: 0 for k in server._xbatch.runner.stats})
                r = asyncio.run(cell(conc, idx))
                if server._xbatch is not None:
                    st = server._xbatch.runner.stats
                    r["batches"] = st["batches"]
                    r["refilled"] = st["refilled"]
                    r["row_steps"] = st["row_steps"]
                    r["slot_efficiency"] = round(st["useful_row_steps"] / max(st["row_steps"], 1), 3)
                out[f"load_{mode}_c{conc}"] = r
                logger.info("load %s c=%d: %s", mode, conc, r)
        set_mode("current")
        return out

    if "width" in sections:
        res.update(run_on_worker(bench))
    if "agreement" in sections:
        res.update(run_on_worker(agreement))
    if "xparity" in sections:
        res.update(run_on_worker(xparity))
    if "load" in sections:
        res.update(load())
    logger.info("RESULT %s", json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
