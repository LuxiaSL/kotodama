#!/usr/bin/env python3
"""Batched row decode on a real checkpoint (GPU): throughput, agreement with the single-stream engine, and the
cross-request batcher under loom-like load.

Builds the model, engine and row decoders through the server's own load_model (the serving build and warmup), then
runs the ``--sections`` asked for (``xgreedy``: the batcher end to end at a near-zero temperature, every row
against the engine's greedy continuation of its own prompt):
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
    (cross-request batches), ``xbatch_refill`` (rows refilled as requests finish) and ``xbatch_refill_pool`` (the
    same with the multi-entry prefix KV pool, ``--prefix_pool_tokens``, in front of every prefill; the pool starts
    empty in each cell). Reports aggregate completion tokens/s, request latency percentiles, prefill seconds and the
    fraction of prompt tokens whose KV was reused (and, with the pool, taken from it) per mode and concurrency.
    With ``--load_trees_jsonl`` (a loom run's ``<out>/paths.jsonl``, kotodama.posttrain.loom) the requests are
    tree-shaped, as the loom sends them: every distinct node of ``--load_trees`` trees over ``--load_tree_depths``
    consecutive depths, rendered through the loom's own window (``--load_window_chars``), issued breadth-first and
    interleaved across trees, so siblings share all but their last turn and several conversations compete for the
    cache;
  * ``poolparity``: a prompt prefilled from a pool hit (a sibling's blocks pooled, the engine holding an unrelated
    prompt, only the suffix prefilled) vs the engine's full prefill of the same prompt, along the full prefill's
    greedy continuation (per-step KL and top-1), beside both against one no-cache forward (the noise floor);
  * ``poolcost``: on an idle GPU, the time of a pool hit (load + suffix extend + store) against a full prefill of
    the same prompt, by suffix length, with the engine's one-chunk extends (as the pool runs it) and with the greedy
    chunk cover (the time a hit saves, apart from any batch decoding around it);
  * ``xgreedy_pool``: ``xgreedy`` on prompt families that share long prefixes, interleaved, with the pool in front
    of the batcher's prefill (every family member after the first extends from pooled blocks), replayed tie-proof
    through the engine's full prefill;
  * ``cascade_kernel``: the cascade part kernel the decoder runs (kotodama.model.cascade_attn, flash-attn on the GPU)
    against the fp32 reference at the decode shapes (a group's packed queries over a long prompt slot, one query
    per row over a tail), on random data: output and log-sum-exp error.

``--cascade`` picks the row decoders: ``off`` per-row KV slots (kotodama.model.multi_decode.MultiSampleDecoder),
``on`` cascade decoders (``CascadeDecoder``: one KV copy per prompt group, cascade attention), ``both`` the two
sets side by side (``--cascade_rows`` widths for the cascade set), every section run on each, the cascade set's keys
suffixed ``_cascade``. ``width`` admits rows ``--width_n`` at a time from one prefill (one group per admit), as a
request's n candidates arrive (a list measures each), and reports each set's KV memory.

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
  tools/run_py.sh scripts/benchmark/multi_decode_bench.py ... --rows 8,16,32,64 --prefix_pool_tokens 65536 \\
      --sections poolparity,xgreedy_pool,load --load_trees_jsonl <loom run>/paths.jsonl \\
      --load_modes xbatch_refill,xbatch_refill_pool --load_concurrency 16 --steer_npz <bank.npz>,<norms.npz>
  tools/run_py.sh scripts/benchmark/multi_decode_bench.py ... --rows 8,16,32,64 --cascade both \\
      --cascade_rows 8,16,32,64,128 --sections cascade_kernel,width,xparity,xgreedy --context_tokens 0,1500,3000
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
import torch._dynamo
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kotodama.model.decode_engine import SamplingParams  # noqa: E402
from kotodama.posttrain.loom import window_turns  # noqa: E402
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


def load_tree_prompts(tok, path: str, n_trees: int, depths: int, window_chars: int, lo: int, hi: int
                      ) -> tuple[list[tuple[str, list[str]]], list[tuple[str, int, str]]]:
    """Loom requests rebuilt from a loom run's ``<out>/paths.jsonl``: for the first ``n_trees`` trees (in file order) whose
    prompts reach ``lo`` tokens, every distinct node at ``depths`` consecutive depths (from the first depth whose
    prompt reaches ``lo`` tokens; prompts over ``hi`` are left out), rendered as the loom renders the prompt that
    expands the node (header + the loom's window + the next speaker's open turn). Returns (prompt, stops) pairs in
    issue order (breadth-first per tree, round-robin across trees) and (tree, depth, node id) per prompt."""
    import re

    tree_re = re.compile(r'"tree": "([^"]+)"')
    rows: dict[str, list[dict]] = {}
    order: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = tree_re.search(line)
            if m is None:
                continue
            t = m.group(1)
            if t not in rows:
                if len(order) >= 8 * n_trees:  # candidates: some trees never reach lo tokens
                    continue
                rows[t] = []
                order.append(t)
            try:
                rows[t].append(json.loads(line))
            except json.JSONDecodeError:
                continue
    per_tree: list[list[tuple[str, list[str], int, str]]] = []
    for t in order:
        nodes: dict[str, tuple[int, str, list[str]]] = {}
        for row in rows[t]:
            tr = parse_row(row)
            depth = row.get("depth")
            if tr is None or not isinstance(depth, int):
                continue
            n0 = len(tr.turns) - depth  # turns before the loom's first generated one (seed turns + history)
            parts = str(row.get("leaf", "")).split(".")
            cast = row.get("cast") or list(dict.fromkeys(x.speaker for x in tr.turns))
            n_seed = int(row.get("n_seed", 0))
            for d in range(depth):  # the request that expanded the node at depth d (it drew turn n0 + d)
                node = ".".join(parts[:d + 1])
                if node in nodes:
                    continue
                seen = window_turns(tr.turns[:n0 + d], tr.header, n_seed, window_chars)
                text = render_prompt(Transcript(seen, tr.header), tr.turns[n0 + d].speaker)
                nodes[node] = (d, text, stop_sequences(cast))
        if not nodes:
            continue
        by_depth: dict[int, list[tuple[str, str, list[str]]]] = {}
        for node, (d, text, stops) in nodes.items():
            by_depth.setdefault(d, []).append((node, text, stops))
        lens = {d: len(tok.encode(v[0][1])) for d, v in by_depth.items()}
        start = next((d for d in sorted(by_depth) if lens[d] >= lo), None)
        if start is None:
            continue
        reqs = []
        for d in range(start, start + depths):
            for node, text, stops in sorted(by_depth.get(d, [])):
                if len(tok.encode(text)) <= hi:
                    reqs.append((text, stops, d, node))
        if reqs:
            per_tree.append(reqs)
        if len(per_tree) >= n_trees:
            break
    if len(per_tree) < n_trees:
        raise SystemExit(f"{path}: only {len(per_tree)} trees have prompts of {lo}..{hi} tokens (need {n_trees})")
    texts, meta = [], []
    for k in range(max(len(r) for r in per_tree)):
        for i, reqs in enumerate(per_tree):
            if k < len(reqs):
                text, stops, d, node = reqs[k]
                texts.append((text, stops))
                meta.append((f"tree{i}", d, node))
    return texts, meta


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
    ap.add_argument("--xparity_prompts", default="16",
                    help="comma list: distinct prompts across the widest bucket, one xparity run each (1 = every "
                         "row on one long prompt)")
    ap.add_argument("--load_requests", type=int, default=96, help="requests per (mode, concurrency) cell")
    ap.add_argument("--load_concurrency", default="4,8", help="comma list of requests in flight")
    ap.add_argument("--load_modes", default="current,xbatch,xbatch_refill",
                    help="comma list: current, xbatch, xbatch_refill, xbatch_refill_pool (needs --prefix_pool_tokens)")
    ap.add_argument("--load_trees_jsonl", default=None,
                    help="a loom run's paths.jsonl: tree-shaped load (the loom's own requests) instead of distinct "
                         "prompts; also the source of poolparity's sibling pairs")
    ap.add_argument("--load_trees", type=int, default=8, help="trees in the tree-shaped load")
    ap.add_argument("--load_tree_depths", type=int, default=4, help="consecutive depths per tree")
    ap.add_argument("--load_window_chars", type=int, default=11000, help="the loom's --window_chars")
    ap.add_argument("--prefix_pool_tokens", type=int, default=0,
                    help="the server's --prefix_pool_tokens (for xbatch_refill_pool, poolparity, xgreedy_pool)")
    ap.add_argument("--prefix_pool_block", type=int, default=16)
    ap.add_argument("--poolparity_pairs", type=int, default=8)
    ap.add_argument("--load_n", type=int, default=8)
    ap.add_argument("--load_max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cascade", choices=["off", "on", "both"], default="off",
                    help="row decoders: per-row KV (off), cascade (on), or both sets side by side")
    ap.add_argument("--cascade_rows", default="", help="widths of the cascade set with --cascade both (default --rows)")
    ap.add_argument("--cascade_groups", type=int, default=0, help="the server's --cascade_groups")
    ap.add_argument("--cascade_tail_tokens", type=int, default=1024, help="the server's --cascade_tail_tokens")
    ap.add_argument("--width_n", default="8",
                    help="comma list: rows per prompt in the width section (keys of the first plain, of the others "
                         "suffixed _n<n>)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    widths = sorted({int(x) for x in a.rows.split(",") if x.strip()})
    contexts = [int(x) for x in a.context_tokens.split(",") if x.strip()]
    sections = {x.strip() for x in a.sections.split(",") if x.strip()}
    width_ns = [int(x) for x in a.width_n.split(",") if x.strip()]
    lo, hi = (int(x) for x in a.prompt_tokens.split(","))
    server._rope_fp32_override = a.rope_fp32
    if a.cascade == "both":
        # every decoder compiles the same step function once; two sets of buckets pass dynamo's default limit (8)
        torch._dynamo.config.recompile_limit = 64
    _model, tok = server.load_model(a.checkpoint, "cuda", mode="base", model_size=a.model_size, engine="fast",
                                    prefix_cache=True, max_seq_len=4096, multi_rows=tuple(widths),
                                    steer_npz=a.steer_npz, prefix_pool_tokens=a.prefix_pool_tokens,
                                    prefix_pool_block=a.prefix_pool_block, cascade_attn=a.cascade == "on",
                                    cascade_groups=a.cascade_groups, cascade_tail_tokens=a.cascade_tail_tokens)
    server._model, server._tokenizer = _model, tok
    eng = server._engine
    pool = server._prefix_pool
    server._prefix_pool = None  # each section and load mode decides whether the pool is in front of the prefill
    needs_pool = bool({"poolparity", "poolcost", "xgreedy_pool"} & sections) or ("load" in sections and "pool" in a.load_modes)
    if needs_pool and pool is None:
        raise SystemExit("poolparity / xgreedy_pool / xbatch_refill_pool need --prefix_pool_tokens > 0")
    assert eng is not None and all(server._multi.get(w) is not None for w in widths)
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

    # decoder sets: key suffix -> {width: decoder}
    sets: dict[str, dict] = {"_cascade" if a.cascade == "on" else "": dict(server._multi)}
    if a.cascade == "both":
        cwidths = sorted({int(x) for x in (a.cascade_rows or a.rows).split(",") if x.strip()})

        def build_cascade_set() -> dict:
            out: dict = {}
            for w in sorted(cwidths, reverse=True):
                out[w] = server.build_and_warm_row_decoder(eng, w, out[max(out)] if out else None,
                                                           (a.cascade_groups, a.cascade_tail_tokens))
            return dict(sorted(out.items()))

        sets["_cascade"] = run_on_worker(build_cascade_set)

    def use_set(suffix: str) -> None:
        """Route the server (and every section) through one decoder set."""
        decs = sets[suffix]
        server._multi = dict(decs)
        server._row_token_cap = getattr(decs[max(decs)], "tail_tokens", 0)

    for suffix, decs in sets.items():
        widest = decs[max(decs)]
        res[f"kv_gb{suffix}"] = round(widest.kv_bytes() / 1e9, 2)
        res[f"widths{suffix}"] = sorted(decs)
        if hasattr(widest, "groups"):
            res[f"groups{suffix}"] = {w: d.groups for w, d in decs.items()}
            res[f"tail_tokens{suffix}"] = widest.tail_tokens
            res[f"attention{suffix}"] = widest.attend_backend

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
                for w in sorted(server._multi):
                    d = server._multi[w]
                    for i, n_req in enumerate(width_ns):
                        logits = eng.prefill(cids)
                        n = min(n_req, w)
                        d.release(range(w))  # groups an earlier measurement left held
                        for lo_row in range(0, w, n):  # one admit per request's n candidates (a group each)
                            d.admit(range(lo_row, min(lo_row + n, w)), logits, params)
                        for _ in range(8):
                            d.step(params)
                        ms = 1000 * timed(lambda: d.step(params), a.steps)
                        key = f"{tag}_rows{w}" + (f"_n{n_req}" if i else "")
                        out[f"{key}_ms_per_step"] = ms
                        out[f"{key}_tok_s"] = w * 1000 / ms
                        out[f"{key}_step_vs_single"] = ms / single
                logger.info("%s: %s", tag, {k: round(v, 2) for k, v in out.items() if k.startswith(tag)})
        return out

    def greedy_of(cids: torch.Tensor, n: int) -> list[int]:
        logits = eng.prefill(cids)
        greedy = [int(logits.argmax())]
        for _ in range(n - 1):
            greedy.append(int(eng.step_logits(torch.tensor([[greedy[-1]]], device="cuda"))[0, -1].argmax()))
        return greedy

    def agreement() -> dict:
        dec = server._multi[max(server._multi)]
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

    def xparity(count: int) -> dict:
        """Rows hold different prompts (row r: prompt r mod P), each forced along its own prompt's greedy."""
        steps = 32
        texts = load_prompts(tok, a.prompts_jsonl, count, lo, hi, a.seed)
        pids = [torch.tensor([tok.encode(t)], device="cuda") for t, _ in texts]
        dec = server._multi[max(server._multi)]
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
            # the noise floor: one full forward of prompt + continuation (no cache), the parity oracle both the
            # engine and the rows are measured against
            truth = []
            for p, g in zip(pids, greedy):
                full = _model(torch.cat([p, torch.tensor([g[:-1]], device="cuda")], dim=1))["logits"][0].float()
                truth.append(list(full[p.shape[1] - 1:]))
            cold = SamplingParams(temperature=1e-4, repetition_penalty=1.0)
            first: dict[int, torch.Tensor] = {}
            dec.release(range(B))  # groups an earlier section left held
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
            tru = [torch.stack([truth[r % P][t] for r in range(B)]) for t in range(steps)]
            kl_t_rows, top1_t_rows = kl_top1(tru, got)
            kl_t_eng, top1_t_eng = kl_top1(tru, ref)
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
                "xparity_compiled_greedy_match_prefix": match,
                "xparity_truth_vs_rows_kl_max_per_step_mean": sum(kl_t_rows) / len(kl_t_rows),
                "xparity_truth_vs_rows_top1": sum(top1_t_rows) / len(top1_t_rows),
                "xparity_truth_vs_engine_kl_max_per_step_mean": sum(kl_t_eng) / len(kl_t_eng),
                "xparity_truth_vs_engine_top1": sum(top1_t_eng) / len(top1_t_eng)}

    def xgreedy(use_pool: bool = False) -> dict:
        """The batcher end to end on the GPU (refill, candidate restarts and the pipelined step loop included): 24
        requests of n=3 on short prompts of different lengths (where the batched and single-stream greedy agree),
        decoded at a near-zero temperature through a refilling runner on the narrowest bucket, each candidate
        against the engine's greedy continuation of its own prompt (matching prefix; 40 = the whole continuation)
        and, tie-proof, replayed through the engine token by token (each token's distance below the top logit).
        ``use_pool``: 6 families of 4 prompts (a shared 300-600-token trunk, each member its own closing turns),
        issued round-robin across families, with the prefix KV pool in front of the batcher's prefill."""
        from kotodama.serve.xbatch import XBatchJob

        steps, n_req = 40, 24
        if use_pool:
            trunks = [tok.encode(t) for t, _ in load_prompts(tok, None, 6, 300, 600, a.seed + 3)]
            tails = ["Teo: Fine. {k} times fine.", "Mara: Say that again, number {k}.", "Teo: Ask the captain ({k}).",
                     "Mara: Then go. Go now, {k}."]
            pids = [trunks[f] + tok.encode("\n\n" + tails[m].format(k=f * 4 + m) + "\n\nTeo:")
                    for m in range(4) for f in range(6)]
            assert pool is not None
            pool.clear()
            pool.stats.update({k: 0 for k in pool.stats})
            prefill = pool.prefill
        else:
            texts = load_prompts(tok, None, n_req, 64, 700, a.seed + 2)
            pids = [tok.encode(t) for t, _ in texts]
            prefill = None
        cold = SamplingParams(temperature=1e-4, repetition_penalty=1.0)
        with torch.inference_mode():
            want = [greedy_of(torch.tensor([p], device="cuda"), steps) for p in pids]
            if use_pool:
                eng.reset()  # greedy_of left the last prompt in the engine: the pool, not the engine, holds trunks
            # full prefills, as greedy_of makes (a prefix-cache extend rounds differently and can flip a near-tie)
            narrow = min(server._multi)
            runner = XBatchRunner({narrow: server._multi[narrow]},
                                  prefill=prefill or (lambda i: (eng.prefill(i), None)),
                                  set_steer=eng.set_steer, decode=lambda i: tok.decode(i, skip_special_tokens=True),
                                  stop_ids=frozenset(), max_seq_len=server._max_seq_len,
                                  device=torch.device("cuda"), refill=True)
            got: dict[int, list[list[int]]] = {}
            for i, p in enumerate(pids):
                runner.queue.put(XBatchJob(key="g", prompt_ids=p, n=3, max_new_tokens=steps, params=cold,
                                           done=lambda o, i=i: got.__setitem__(i, o.rows)))
            served = runner.run_batch()
        match = []
        for i in range(n_req):
            for row in got[i]:
                match.append(next((k for k, (x, y) in enumerate(zip(row, want[i])) if x != y), len(row)))
        # bf16 logits tie often, and a near-zero temperature breaks a tie at random where argmax takes the lower
        # id, so a diverging prefix is not by itself an error. The tie-proof check: replay every candidate through
        # the engine and measure, at each of its tokens, how far below the engine's top logit the chosen token
        # sits (0 = an argmax).
        gaps: list[float] = []
        with torch.inference_mode():
            for i in range(n_req):
                for row in got[i]:
                    logits = eng.prefill(torch.tensor([pids[i]], device="cuda"))[0]
                    for t, tok_id in enumerate(row):
                        gaps.append(float(logits.max() - logits[tok_id]))
                        if t + 1 < len(row):
                            logits = eng.step_logits(torch.tensor([[tok_id]], device="cuda"))[0, -1].float()
        out = {"xgreedy_tokens": len(gaps), "xgreedy_argmax_rate": sum(g == 0 for g in gaps) / len(gaps),
               "xgreedy_gap_max": max(gaps), "xgreedy_gap_over_0p25": sum(g > 0.25 for g in gaps),
               "xgreedy_served": served, "xgreedy_refilled": runner.stats["refilled"],
               "xgreedy_restarts": runner.stats["restarts"],
               "xgreedy_bucket": narrow, "xgreedy_prompt_tokens": [len(p) for p in pids],
               "xgreedy_match_prefix": match}
        if use_pool:
            out = {k.replace("xgreedy_", "xgreedy_pool_"): v for k, v in out.items()}
            out["xgreedy_pool_stats"] = dict(pool.stats)
        return out

    def poolcost() -> dict:
        """Median ms over 5 repeats of: a full prefill of a ``hi``-token prompt, and pool hits on it whose suffix past
        the pooled blocks is 1..300 tokens (the engine holding an unrelated prompt), each timed between device syncs,
        with one-chunk extends and with the greedy chunk cover."""
        assert pool is not None
        b = pool.block_size
        base = load_prompts(tok, None, 1, hi, hi, a.seed + 5)[0][0]
        ids = tok.encode(base)
        g = torch.Generator().manual_seed(a.seed)
        other = torch.randint(4, eng.config.vocab_size, (1, 64), generator=g).cuda()  # shares nothing with ids
        full_ids = torch.tensor([ids], device="cuda")

        def ms(fn, setup=None) -> float:
            vals = []
            for _ in range(5):
                if setup is not None:
                    setup()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                vals.append((time.perf_counter() - t0) * 1000)
            return statistics.median(vals)

        out: dict = {"poolcost_prompt_tokens": len(ids), "poolcost_buckets": list(eng.block_buckets)}
        with torch.inference_mode():
            out["poolcost_full_prefill_ms"] = ms(lambda: eng.prefill(full_ids))
            built = eng.extend_single_chunk
            for name, single in (("onechunk", True), ("greedy", False)):
                eng.extend_single_chunk = single
                for suffix in (1, 8, 20, 40, 100, 300):
                    # the pooled prefix ends on a block boundary, ``suffix`` tokens before the prompt's end
                    cut = len(ids) - suffix
                    pooled = ids[:cut - cut % b]
                    primer = torch.tensor([pooled + [5] * b], device="cuda")  # pools ``pooled`` (and a decoy block)

                    def setup() -> None:
                        pool.clear()
                        eng.reset()
                        pool.prefill(primer)
                        eng.prefill(other)  # the engine holds nothing of the prompt: the hit comes from the pool

                    out[f"poolcost_{name}_suffix{len(ids) - len(pooled)}_hit_ms"] = round(
                        ms(lambda: pool.prefill(full_ids), setup), 3)
            eng.extend_single_chunk = built
            pool.clear()
            eng.reset()
        return out

    def poolparity() -> dict:
        """Pool hit vs full prefill of the same prompt. Pairs (A, B) are siblings from the tree-shaped load (or,
        without ``--load_trees_jsonl``, a trunk with two different closing turns). Per pair: A goes through the pool
        (its blocks pooled), an unrelated prompt is prefilled on the engine, then B through the pool (a hit: only
        B's suffix past the pooled blocks is prefilled). B's continuation is the full prefill's greedy, forced on
        both; the no-cache forward of B + continuation is the floor both are measured against."""
        assert pool is not None
        steps = 32
        if a.load_trees_jsonl:
            texts, meta = load_tree_prompts(tok, a.load_trees_jsonl, a.poolparity_pairs, 2, a.load_window_chars,
                                            lo, hi)
            by_parent: dict[tuple[str, int, str], list[str]] = {}
            for (text, _), (tree, d, node) in zip(texts, meta):
                by_parent.setdefault((tree, d, node.rsplit(".", 1)[0]), []).append(text)
            pairs = [(v[0], v[1]) for v in by_parent.values() if len(v) >= 2][:a.poolparity_pairs]
        else:
            trunks = load_prompts(tok, None, a.poolparity_pairs, lo, hi, a.seed + 4)
            pairs = [(t + " No.\n\nTeo: Then what?\n\nMara:", t + " Fine.\n\nTeo: Is it?\n\nMara:")
                     for t, _ in trunks]
        other = torch.tensor([tok.encode(PROMPT)], device="cuda")
        kl_hf, t1_hf, kl_tf, t1_tf, kl_th, t1_th, pooled, plens = [], [], [], [], [], [], [], []
        with torch.inference_mode():
            for ta, tb in pairs:
                ia = torch.tensor([tok.encode(ta)], device="cuda")
                ib = torch.tensor([tok.encode(tb)], device="cuda")
                g = greedy_of(ib, steps)
                forced = [torch.tensor([[t]], device="cuda") for t in g[:-1]]
                full = [eng.prefill(ib)[0].float()]
                full += [eng.step_logits(t)[0, -1].float().clone() for t in forced]
                pool.clear()
                eng.reset()
                pool.prefill(ia)
                eng.prefill(other)
                l0, info = pool.prefill(ib)
                hit = [l0[0].float()] + [eng.step_logits(t)[0, -1].float().clone() for t in forced]
                truth = _model(torch.cat([ib, torch.tensor([g[:-1]], device="cuda")], dim=1))["logits"][0].float()
                truth = list(truth[ib.shape[1] - 1:])
                for ref, got, kl_l, t1_l in ((full, hit, kl_hf, t1_hf), (truth, full, kl_tf, t1_tf),
                                             (truth, hit, kl_th, t1_th)):
                    kl, t1 = kl_top1([r.view(1, -1) for r in ref], [x.view(1, -1) for x in got])
                    kl_l.append(sum(kl) / len(kl))
                    t1_l.append(sum(t1) / len(t1))
                pooled.append(int(info["pool_tokens"]))
                plens.append(int(ib.shape[1]))

        def mean(x: list[float]) -> float:
            return sum(x) / len(x)

        return {"poolparity_pairs": len(pairs), "poolparity_prompt_tokens": plens, "poolparity_pooled_tokens": pooled,
                "poolparity_full_vs_hit_kl_per_step_mean": mean(kl_hf), "poolparity_full_vs_hit_kl_max": max(kl_hf),
                "poolparity_full_vs_hit_top1": mean(t1_hf),
                "poolparity_truth_vs_full_kl_per_step_mean": mean(kl_tf), "poolparity_truth_vs_full_top1": mean(t1_tf),
                "poolparity_truth_vs_hit_kl_per_step_mean": mean(kl_th), "poolparity_truth_vs_hit_top1": mean(t1_th)}

    def load() -> dict:
        import httpx

        if a.load_trees_jsonl:
            texts, meta = load_tree_prompts(tok, a.load_trees_jsonl, a.load_trees, a.load_tree_depths,
                                            a.load_window_chars, lo, hi)
        else:
            texts, meta = load_prompts(tok, a.prompts_jsonl, a.load_requests, lo, hi, a.seed + 1), []
        prompt_lens = [len(tok.encode(t)) for t, _ in texts]
        tally = {"prefills": 0, "prompt_tokens": 0, "reused_tokens": 0, "pool_tokens": 0, "prefill_s": 0.0}

        def counted_prefill(ids: torch.Tensor):
            # every admit's prefill, in every mode: how much of the prompt's KV was reused, and from where
            t0 = time.perf_counter()
            logits, info = server._engine_prefill(ids)
            torch.cuda.synchronize()
            tally["prefill_s"] += time.perf_counter() - t0
            tally["prefills"] += 1
            tally["prompt_tokens"] += int(ids.shape[1])
            if info is not None:
                tally["reused_tokens"] += int(info.get("common_prefix", 0))
                tally["pool_tokens"] += int(info.get("pool_tokens", 0))
            return logits, info

        steer = []
        for item in a.steer.split(","):
            if a.steer_npz and item.strip():
                name, alpha = item.split(":")
                steer.append({"name": name, "alpha": float(alpha)})
        all_dec = dict(server._multi)
        xrows = tuple(w for w in sorted(all_dec) if w >= 8)

        def set_mode(mode: str) -> None:
            server._multi = {8: all_dec[8]} if 8 in all_dec else {}
            server._xbatch = None
            server._prefix_pool = pool if mode.endswith("_pool") else None
            if mode.startswith("xbatch"):
                runner = XBatchRunner({w: all_dec[w] for w in xrows}, prefill=counted_prefill,
                                      set_steer=eng.set_steer,
                                      decode=lambda i: tok.decode(i, skip_special_tokens=True),
                                      stop_ids=server._stop_token_ids, max_seq_len=server._max_seq_len,
                                      device=torch.device("cuda"), refill=mode.startswith("xbatch_refill"))
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
                     "load_requests": len(texts), "load_shape": "trees" if meta else "distinct",
                     "load_n": a.load_n, "load_steer": steer, "xbatch_rows": list(xrows)}
        if meta:
            out["load_trees"] = len({m[0] for m in meta})
            out["load_tree_depths"] = sorted({m[1] for m in meta})
        idx = list(range(len(texts)))

        def warm_prefills() -> None:
            # the first prefill at a new prompt length pays a one-time cost (about a second, measured here at
            # 1.5-2.6K tokens); paying it once for every prompt keeps it out of whichever mode runs first
            with torch.inference_mode():
                for text, _ in texts:
                    eng.prefill(torch.tensor([tok.encode(text)], device="cuda"))
                eng.reset()

        t0 = time.perf_counter()
        run_on_worker(warm_prefills)
        out["load_warm_prefill_s"] = round(time.perf_counter() - t0, 1)
        for mode in [m.strip() for m in a.load_modes.split(",") if m.strip()]:
            set_mode(mode)
            asyncio.run(cell(2, idx[:4]))  # warm the path (first-request lazy work) outside the measurement
            for conc in [int(x) for x in a.load_concurrency.split(",") if x.strip()]:
                if server._xbatch is not None:
                    server._xbatch.runner.stats.update({k: 0 for k in server._xbatch.runner.stats})
                # every cell starts cold: an empty pool and an engine holding nothing
                run_on_worker(eng.reset)
                if pool is not None:
                    pool.clear()
                    pool.stats.update({k: 0 for k in pool.stats})
                tally.update({k: 0 for k in tally})
                r = asyncio.run(cell(conc, idx))
                r["prefill_tally"] = {k: round(v, 2) for k, v in tally.items()}
                r["reused_frac"] = round(tally["reused_tokens"] / max(tally["prompt_tokens"], 1), 4)
                r["pool_frac"] = round(tally["pool_tokens"] / max(tally["prompt_tokens"], 1), 4)
                if server._prefix_pool is not None:
                    r["pool_stats"] = {k: round(v, 3) for k, v in pool.stats.items()}
                if server._xbatch is not None:
                    st = server._xbatch.runner.stats
                    r["batches"] = st["batches"]
                    r["refilled"] = st["refilled"]
                    r["row_steps"] = st["row_steps"]
                    r["slot_efficiency"] = round(st["useful_row_steps"] / max(st["row_steps"], 1), 3)
                    for k in ("steps", "restarts", "prefill_s", "step_s", "host_s"):
                        r[k] = round(st[k], 2)
                out[f"load_{mode}_c{conc}"] = r
                logger.info("load %s c=%d: %s", mode, conc, r)
        set_mode("current")
        server._prefix_pool = None
        server._multi = all_dec
        return out

    def cascade_kernel() -> dict:
        """flash-attn's part kernel vs the fp32 reference at the decode shapes, on random bf16 data."""
        from kotodama.model.cascade_attn import (attend_for, attend_reference, merge, pack_group_queries,
                                                 unpack_group_outputs)

        name, fn = attend_for(torch.device("cuda"), torch.bfloat16)
        cfg = eng.config
        hq, hk, d, S = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim, eng.max_seq_len
        g = torch.Generator(device="cuda").manual_seed(a.seed)
        out: dict = {"cascade_kernel_backend": name}

        def err(x: torch.Tensor, y: torch.Tensor) -> float:
            return float((x.float() - y.float()).norm() / y.float().norm())

        with torch.inference_mode():
            for groups, m, rows, tail in ((8, 8, 64, 200), (2, 32, 64, 17), (16, 8, 128, 900)):
                q = torch.randn((rows, hq, d), device="cuda", dtype=torch.bfloat16, generator=g)
                pk, pv = (torch.randn((groups, S, hk, d), device="cuda", dtype=torch.bfloat16, generator=g)
                          for _ in range(2))
                tk, tv = (torch.randn((rows, tail + 8, hk, d), device="cuda", dtype=torch.bfloat16, generator=g)
                          for _ in range(2))
                plens = torch.randint(1500, 3000, (groups,), device="cuda", generator=g).to(torch.int32)
                plens[-1] = 0  # an empty group
                tlens = torch.randint(1, tail, (rows,), device="cuda", generator=g).to(torch.int32)
                members = torch.arange(groups * m, device="cuda").remainder(rows).view(groups, m)
                slot = torch.arange(rows, device="cuda")  # row r sits at slot r (group r // m)
                qg = pack_group_queries(q, members, hk)
                got = {}
                for tag, f in (("got", fn), ("ref", attend_reference)):
                    o1, l1 = f(qg, pk, pv, plens)
                    o1, l1 = unpack_group_outputs(o1, l1, m)
                    o2, l2 = f(q.unsqueeze(1), tk, tv, tlens)
                    got[tag] = (o1[slot], l1[slot], o2[:, 0], l2[:, :, 0])
                go1, gl1, go2, gl2 = got["got"]
                ro1, rl1, ro2, rl2 = got["ref"]
                live = torch.isfinite(rl1).all(-1)  # rows of non-empty groups
                key = f"cascade_kernel_g{groups}_m{m}_rows{rows}"
                out[f"{key}_shared_rel_err"] = err(go1[live], ro1[live])
                out[f"{key}_shared_lse_max_abs"] = float((gl1[live] - rl1[live]).abs().max())
                out[f"{key}_empty_group_lse_is_neg_inf"] = bool(torch.isneginf(gl1[~live]).all())
                out[f"{key}_tail_rel_err"] = err(go2, ro2)
                out[f"{key}_tail_lse_max_abs"] = float((gl2 - rl2).abs().max())
                got_m, ref_m = merge(go1, gl1, go2, gl2), merge(ro1, rl1, ro2, rl2)
                out[f"{key}_merged_rel_err"] = err(got_m[live], ref_m[live])
                out[f"{key}_merged_rel_err_empty_group_rows"] = err(got_m[~live], ref_m[~live])
        logger.info("cascade_kernel: %s", out)
        return out

    def per_set(fn) -> dict:
        """Run ``fn`` once per decoder set, its keys suffixed with the set's."""
        merged: dict = {}
        for suffix in sets:
            use_set(suffix)
            merged.update({f"{k}{suffix}": v for k, v in fn().items()})
        use_set(next(iter(sets)))
        return merged

    if "cascade_kernel" in sections:
        res.update(run_on_worker(cascade_kernel))
    if "width" in sections:
        res.update(per_set(lambda: run_on_worker(bench)))
    if "agreement" in sections:
        res.update(per_set(lambda: run_on_worker(agreement)))
    if "xparity" in sections:
        for count in [int(x) for x in a.xparity_prompts.split(",") if x.strip()]:
            res.update(per_set(lambda: {f"{k}_p{count}": v for k, v in run_on_worker(lambda: xparity(count)).items()}))
    if "xgreedy" in sections:
        res.update(per_set(lambda: run_on_worker(xgreedy)))
    if "poolcost" in sections:
        res.update(run_on_worker(poolcost))
    if "xgreedy_pool" in sections:
        res.update(per_set(lambda: run_on_worker(lambda: xgreedy(use_pool=True))))
    if "poolparity" in sections:
        res.update(run_on_worker(poolparity))
    if "load" in sections:
        res.update(per_set(load))
    logger.info("RESULT %s", json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
