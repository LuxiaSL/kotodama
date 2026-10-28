#!/usr/bin/env python3
"""Regional compile of the single-stream DecodeEngine on a real checkpoint (GPU): what it costs to compile, what it
costs per step and per prefill block, how far its numbers are from the eager and whole-step paths, and whether its
captured graphs still draw fresh noise.

Builds one engine compiled regionally (``compile_step(mode, regional=True)``) and, with ``--whole``, one compiled the
way the server compiles it today (``compile_step(mode)``). Reports:
  * ``compile_s``: wall seconds of compile_step + warm_blocks, and of the first three calls of each step variant
    (sampled, truncated, forward), which is where the regional path compiles, warms up and captures; ``--cold`` turns
    inductor's caches off so these are cold numbers;
  * ``ms``: steady-state ms per sampled step, and per block replay at each bucket;
  * ``prefill``: KL and top-1 agreement of each path's prefill logits against the reference (eager) prefill, at
    prompt lengths that take an exact bucket, an overlap chunk and a multi-bucket cover;
  * ``forward``: along a forced continuation under a decode-span lever, per-step KL and top-1 agreement of the
    regional forward (and the whole-step forward) against the eager forward;
  * ``rng``: unseeded sampling through the captured graphs: distinct continuations among repeats of one prompt on the
    engine, and among the rows of an unseeded regional cascade decoder (rows of one prompt) and across two of its
    starts. A captured graph that replayed one fixed noise draw would give 1 distinct continuation;
  * ``rows`` (``--sections rows``): each row decoder class (per-row KV and cascade) compiled regionally and captured,
    through every path a client takes (admit, the sampled / greedy / truncated steps, ``force`` and ``score``),
    against the same decoder uneager: per-step KL and top-1 of the forced continuation, the largest score
    difference, and whether the greedy rows agree.

Usage (node, GPU):
  tools/run_py.sh scripts/benchmark/check_regional_engine.py --checkpoint <model.pt> --model_size 1b-tsumugi \\
      --cold --out <dir>/check_regional_engine.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model.decode_engine import DecodeEngine, SamplingParams  # noqa: E402
from kotodama.model.levers import LeverPlan  # noqa: E402
from kotodama.model.multi_decode import CascadeDecoder  # noqa: E402
from kotodama.serve import server  # noqa: E402

logger = logging.getLogger("check_regional_engine")

PROMPT = "Chapter 4\n\nThe council met at dawn. Nobody wanted to be the first to say what everyone already knew:"
FORCED = (" The water was very still that morning, and the gulls had gone quiet over the harbor wall, where the old"
          " nets still hung drying in the wind.")
PREFILL_LENGTHS = (16, 100, 256, 700)


def timed(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def kl_top1(lp_a: torch.Tensor, lp_b: torch.Tensor) -> tuple[float, float]:
    """KL(a || b) and whether the argmaxes agree, from log-probs (V,)."""
    return float((lp_a.exp() * (lp_a - lp_b)).sum()), float(lp_a.argmax() == lp_b.argmax())


@torch.inference_mode()
def compile_cost(a, eng: DecodeEngine, regional: bool, ids: torch.Tensor) -> dict:
    out: dict = {"compile_s": {}, "ms": {}}
    out["compile_s"]["compile_step"] = round(timed(lambda: eng.compile_step(mode=a.mode, regional=regional)), 2)
    out["compile_s"]["warm_blocks"] = round(timed(eng.warm_blocks), 2)
    laws = {"sampled": SamplingParams(temperature=a.temperature, repetition_penalty=a.repetition_penalty),
            "truncated": SamplingParams(temperature=a.temperature, repetition_penalty=a.repetition_penalty,
                                        top_p=0.9, min_p=0.05)}
    for name, params in laws.items():
        tok = eng.sample_first(eng.prefill(ids), params)
        out["compile_s"][name] = round(sum(timed(lambda: eng.step(eng.cur_token, params)) for _ in range(3)), 2)
    eng.prefill(ids)
    out["compile_s"]["forward"] = round(sum(timed(lambda: eng.step_logits(eng.cur_token)) for _ in range(3)), 2)
    out["compile_s"]["total"] = round(sum(out["compile_s"].values()), 2)
    params = laws["sampled"]
    tok = eng.sample_first(eng.prefill(ids), params)
    for _ in range(8):
        tok = eng.step(tok, params)
    out["ms"]["step_sampled"] = round(timed(lambda: [eng.step(eng.cur_token, params) for _ in range(a.steps)])
                                      / a.steps * 1000, 3)
    g = torch.Generator().manual_seed(11)
    for b in eng.block_buckets:
        chunk = torch.randint(4, eng.config.vocab_size, (1, b), generator=g).to(eng.device)
        eng.prefill(chunk)  # settles the bucket's graph (capture on a regional engine)
        out["ms"][f"block_{b}"] = round(timed(lambda: [eng.prefill(chunk) for _ in range(10)]) / 10 * 1000, 3)
    return out


@torch.inference_mode()
def prefill_gap(eng: DecodeEngine, ref: DecodeEngine, vocab: int) -> dict:
    g = torch.Generator().manual_seed(7)
    out = {}
    for n in PREFILL_LENGTHS:
        ids = torch.randint(4, vocab, (1, n), generator=g).to(eng.device)
        want = F.log_softmax(ref.prefill_reference(ids).float().view(-1), -1)
        got = F.log_softmax(eng.prefill(ids).float().view(-1), -1)
        kl, top1 = kl_top1(want, got)
        out[str(n)] = {"kl": kl, "top1": top1}
    return out


@torch.inference_mode()
def forward_gap(eng: DecodeEngine, ref: DecodeEngine, ids: torch.Tensor, forced: torch.Tensor, slot: int) -> dict:
    """Per-step logits of ``eng``'s compiled forward against the eager forward of ``ref`` along a forced
    continuation, both from the same reference prefill, at lever ``slot``."""
    runs = []
    for e, fwd in ((ref, ref._forward_step), (eng, eng.step_logits)):
        e.prefill_reference(ids)
        if e.levers is not None:
            e.lever_slot.fill_(slot)
        steps = []
        for t in range(forced.shape[1]):
            x = forced[:, t:t + 1].contiguous()
            steps.append(F.log_softmax(fwd(x).float().view(-1), -1).clone())
            if fwd is ref._forward_step:
                ref._pos_cpu += 1
        runs.append(steps)
    kls, top1 = zip(*(kl_top1(x, y) for x, y in zip(*runs)))
    return {"kl_mean": float(np.mean(kls)), "kl_max": float(np.max(kls)), "top1": float(np.mean(top1))}


@torch.inference_mode()
def rng_check(a, eng: DecodeEngine, ids: torch.Tensor) -> dict:
    params = SamplingParams(temperature=1.0, repetition_penalty=1.0)
    seqs = []
    for _ in range(4):
        tok = eng.sample_first(eng.prefill(ids), params)
        seq = [int(tok.item())]
        for _ in range(24):
            seq.append(int(eng.step(eng.cur_token, params).item()))
        seqs.append(tuple(seq))
    out = {"engine_distinct_of_4": len(set(seqs))}
    dec = CascadeDecoder(eng, 8, groups=1, group_rows=8, tail_tokens=256)
    dec.compile_step(mode=a.mode, regional=True)
    starts = []
    for _ in range(2):
        dec.release(range(8))
        dec.start(eng.prefill(ids), params)
        rows = [dec.cur_tokens.view(-1).tolist()]
        for _ in range(24):
            rows.append(dec.step(params).view(-1).tolist())
        starts.append([tuple(r) for r in np.array(rows).T.tolist()])
    out["cascade_rows_distinct_of_8"] = [len(set(s)) for s in starts]
    out["cascade_starts_identical"] = starts[0] == starts[1]
    return out


@torch.inference_mode()
def rows_check(a, eng: DecodeEngine, ids: torch.Tensor, forced: torch.Tensor) -> dict:
    from kotodama.model.multi_decode import MultiSampleDecoder

    out = {}
    laws = [SamplingParams(temperature=a.temperature, repetition_penalty=a.repetition_penalty),
            SamplingParams(temperature=a.temperature, repetition_penalty=a.repetition_penalty, top_p=0.9, min_p=0.05),
            SamplingParams(temperature=0.0, repetition_penalty=a.repetition_penalty)]
    rows = 8
    for name, make in (("per_row", lambda: MultiSampleDecoder(eng, rows, seeded=True)),
                       ("cascade", lambda: CascadeDecoder(eng, rows, groups=2, group_rows=4, tail_tokens=256,
                                                          seeded=True))):
        ref, reg = make(), make()
        t0 = time.perf_counter()
        reg.compile_step(mode=a.mode, regional=True)
        r = {}
        for law in laws:  # every step variant, past its capture
            reg.start(eng.prefill(ids), law)
            for _ in range(6):
                reg.step(law)
        r["compile_and_warm_s"] = round(time.perf_counter() - t0, 1)
        fz = forced.expand(rows, -1).contiguous()
        sc = []
        for dec in (ref, reg):
            dec.start(eng.prefill(ids), laws[0])
            sc.append(dec.score(fz).float().clone())
        r["score_max_abs_diff"] = float((sc[0] - sc[1]).abs().max())
        lps = []
        for dec, fwd in ((ref, ref._forward_step), (reg, reg._regional_forward)):
            dec.start(eng.prefill(ids), laws[0])
            steps = []
            for t in range(forced.shape[1]):
                steps.append(F.log_softmax(fwd(fz[:, t:t + 1].clone()).float(), -1).clone())
                dec._advance_host()
            lps.append(steps)
        kls, top1 = [], []
        for x, y in zip(*lps):
            kls.append(float((x.exp() * (x - y)).sum(-1).max()))
            top1.append(float((x.argmax(-1) == y.argmax(-1)).float().mean()))
        r["forward_kl_max"], r["forward_top1"] = max(kls), float(np.mean(top1))
        seqs = []
        for dec in (ref, reg):
            dec.start(eng.prefill(ids), laws[2])
            seqs.append([dec.step(laws[2]).view(-1).tolist() for _ in range(24)])
        r["greedy_rows_agree_steps"] = next((i for i, (x, y) in enumerate(zip(*seqs)) if x != y), 24)
        firsts = []
        for dec in (ref, reg):  # the prompt fed through the decode step, then a greedy continuation
            dec.admit_empty()
            firsts.append(dec.force(ids, laws[2]))
        r["force_first_tokens_agree"] = firsts[0] == firsts[1]
        r["force_then_steps"] = len([reg.step(laws[0]) for _ in range(4)])
        out[name] = r
        del ref, reg
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_size", default="1b-tsumugi")
    ap.add_argument("--rope_fp32", action="store_true")
    ap.add_argument("--levers", type=int, default=4, help="lever slots (0: no tables); the forward check runs under a "
                                                          "bound decode-span lever")
    ap.add_argument("--mode", default="max-autotune", help="the whole-step mode; the regions use its no-cudagraphs "
                                                           "form")
    ap.add_argument("--whole", action="store_true", help="also build and measure the whole-step engine")
    ap.add_argument("--sections", default="engine", help="comma list: engine (the engine checks), rows (the row "
                                                         "decoders over a regional engine)")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--cold", action="store_true", help="inductor caches off: cold compile seconds")
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    torch._dynamo.config.recompile_limit = 64
    if a.cold:
        import torch._inductor.config as inductor_config

        inductor_config.force_disable_caches = True
    server._rope_fp32_override = a.rope_fp32
    model, tok = server.load_model(a.checkpoint, "cuda", mode="base", model_size=a.model_size, engine="reference",
                                   max_seq_len=a.max_seq_len)
    cfg = model.config
    ids = torch.tensor([tok.encode(PROMPT)], device="cuda")
    forced = torch.tensor([tok.encode(FORCED)], device="cuda")
    ref = DecodeEngine(model, max_seq_len=a.max_seq_len, levers=a.levers)
    slot = 0
    plan = None
    if a.levers > 0:
        v = np.random.default_rng(a.seed).standard_normal(cfg.hidden_size).astype(np.float32)
        v *= 8.0 / np.linalg.norm(v)
        plan = LeverPlan(attn={cfg.num_layers // 2: v}, span="decode")
        slot = ref.levers.bind(plan)
    res: dict = {"checkpoint": a.checkpoint, "levers": a.levers, "mode": a.mode, "cold": a.cold,
                 "layers": cfg.num_layers}
    sections = {x.strip() for x in a.sections.split(",") if x.strip()}
    if "rows" in sections:
        eng = DecodeEngine(model, max_seq_len=a.max_seq_len, levers=a.levers)
        eng.compile_step(mode=a.mode, regional=True)
        res["rows"] = rows_check(a, eng, ids, forced)
        logger.info("rows: %s", json.dumps(res["rows"]))
        del eng
        torch.cuda.empty_cache()
    for name, regional in ((("regional", True), ("whole", False)) if a.whole else (("regional", True),)) \
            if "engine" in sections else ():
        eng = DecodeEngine(model, max_seq_len=a.max_seq_len, levers=a.levers)
        if plan is not None:
            assert eng.levers.bind(plan) == slot
        torch.manual_seed(a.seed)
        r = compile_cost(a, eng, regional, ids)
        if regional:
            r["layer_signatures"] = eng.regional_variants
        logger.info("%s cost: %s", name, json.dumps(r))
        r["prefill"] = prefill_gap(eng, ref, cfg.vocab_size)
        logger.info("%s prefill vs reference: %s", name, json.dumps(r["prefill"]))
        r["forward"] = forward_gap(eng, ref, ids, forced, slot)
        logger.info("%s forward vs eager: %s", name, json.dumps(r["forward"]))
        if regional:
            r["rng"] = rng_check(a, eng, ids)
            logger.info("regional rng: %s", json.dumps(r["rng"]))
        res[name] = r
        del eng
        torch.cuda.empty_cache()
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
