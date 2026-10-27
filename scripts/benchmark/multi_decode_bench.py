#!/usr/bin/env python3
"""Batched multi-sample decode on a real checkpoint (GPU): throughput against the single-stream engine, and agreement.

Builds the model, engine and MultiSampleDecoder through the server's own load_model (the serving build and warmup),
then reports:
  * single-stream decode ms/token (the engine's fused sampled step) and batched ms/step at n = 2..rows, as tokens/s;
  * forward agreement: one forced continuation fed to every row (eager batched forward) vs the engine's step,
    per-step KL(engine || row) and top-1 agreement (bf16 kernels differ between a B-row GEMM and a GEMV, so the bar
    is distributional, not bitwise);
  * the compiled batched step at a near-zero temperature vs the engine's greedy continuation (matching prefix).

``--rows`` takes a list (one compiled decoder per bucket, timed at its full width) and ``--context_tokens`` a list of
prompt lengths (the dialogue repeated to that many tokens), so batch width is measured where the loom decodes: deep
in a long context, where attention reads every row's KV each step.

Usage (node, GPU):
  tools/run_py.sh scripts/benchmark/multi_decode_bench.py --checkpoint <step.pt> --model_size 1b-moonshot-loop \
      --rope_fp32 --rows 8 [--context_tokens 0,3000] [--out results/multi_decode.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kotodama.model.decode_engine import SamplingParams  # noqa: E402
from kotodama.serve import server  # noqa: E402

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_size", default="1b-moonshot-loop")
    ap.add_argument("--rope_fp32", action="store_true")
    ap.add_argument("--rows", default="8", help="comma list of batch widths, e.g. 8,16,32,64")
    ap.add_argument("--context_tokens", default="0", help="comma list of prompt lengths (0 = the short prompt)")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    widths = sorted({int(x) for x in a.rows.split(",") if x.strip()})
    contexts = [int(x) for x in a.context_tokens.split(",") if x.strip()]
    server._rope_fp32_override = a.rope_fp32
    _model, tok = server.load_model(a.checkpoint, "cuda", mode="base", model_size=a.model_size, engine="fast",
                                    prefix_cache=True, max_seq_len=4096, multi_rows=tuple(widths))
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
    params = SamplingParams(temperature=0.75, repetition_penalty=1.2)
    res: dict = {"rows": widths, "context_tokens": contexts, "steps": a.steps}

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

    def agreement() -> dict:
        with torch.inference_mode():
            # a model-written continuation (the engine's greedy) is a fairer probe than random ids
            logits = eng.prefill(ids)
            greedy = [int(logits.argmax())]
            for _ in range(63):
                greedy.append(int(eng.step_logits(torch.tensor([[greedy[-1]]], device="cuda"))[0, -1].argmax()))
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
            kl, top1 = [], []
            for r, b in zip(ref, rows):
                p = F.log_softmax(r, -1)
                q = F.log_softmax(b, -1)
                kl.append(float((p.exp() * (p - q)).sum(-1).max()))
                top1.append(float((b.argmax(-1) == r.argmax()).float().mean()))
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

    res.update(run_on_worker(bench))
    res.update(run_on_worker(agreement))
    logger.info("RESULT %s", json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
