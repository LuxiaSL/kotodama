#!/usr/bin/env python3
"""Regional decode compile on a real checkpoint (GPU): what it costs to compile, what it costs per step, and how far
its numbers are from the whole-step graph.

Builds one engine and two row decoders of the same width over it (``--cascade`` picks per-row KV or cascade
decoders): one compiled the way the server compiles it (``compile_step(mode)``: each step variant one whole-step
graph, CUDA graphs from the mode), one compiled regionally (``compile_step(mode, regional=True)``: one graph per
decode-layer signature, without CUDA graphs, and the whole step captured as one CUDA graph on its third call).
Reports:
  * ``compile_s``: wall seconds of the first three calls of each step variant (sampled, greedy, scored), which is
    where both paths compile, warm up and capture; ``--cold`` turns inductor's caches off so these are cold numbers;
  * ``ms_per_step``: steady-state ms per sampled and greedy step at the width, every row on a bound lever when
    ``--levers`` > 0;
  * ``forward``: along a forced continuation, per-step KL and top-1 agreement of the logits between the whole-step
    compiled forward and the regional forward, each also against the eager forward (the noise floor);
  * ``score``: the largest |logprob difference| of the forced continuation scored through each path's scored step;
  * ``greedy``: how many tokens the two greedy steps agree before they first differ.

Usage (node, GPU):
  tools/run_py.sh scripts/benchmark/check_regional_compile.py --checkpoint <model.pt> --model_size 1b-tsumugi \\
      --rope_fp32 --rows 8 --cascade on --out <dir>/check_regional_compile.json
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
from kotodama.model.multi_decode import CascadeDecoder, MultiSampleDecoder  # noqa: E402
from kotodama.serve import server  # noqa: E402

logger = logging.getLogger("check_regional_compile")

PROMPT = "Chapter 4\n\nThe council met at dawn. Nobody wanted to be the first to say what everyone already knew:"
FORCED = (" The water was very still that morning, and the gulls had gone quiet over the harbor wall, where the old"
          " nets still hung drying in the wind.")


def make_decoder(a: argparse.Namespace, eng: DecodeEngine) -> MultiSampleDecoder:
    if a.cascade == "on":
        return CascadeDecoder(eng, a.rows, groups=max(1, a.rows // 8), group_rows=min(a.rows, 8),
                              tail_tokens=a.cascade_tail_tokens)
    return MultiSampleDecoder(eng, a.rows)


def admit(eng: DecodeEngine, dec: MultiSampleDecoder, ids: torch.Tensor, params: SamplingParams, slots) -> None:
    logits = eng.prefill(ids)
    dec.release(range(dec.rows))
    dec.admit(list(range(dec.rows)), logits, params, slots=slots)


def timed(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


@torch.inference_mode()
def compile_cost(a, eng, dec, ids, slots, sampled, greedy) -> dict:
    """Seconds of the first three calls of each variant (compile, warm-up, capture), then steady ms per step."""
    out: dict = {"compile_s": {}, "ms_per_step": {}}
    for name, params in (("sampled", sampled), ("greedy", greedy)):
        admit(eng, dec, ids, params, slots)
        out["compile_s"][name] = round(sum(timed(lambda: dec.step(params)) for _ in range(3)), 2)
        for _ in range(8):
            dec.step(params)
        out["ms_per_step"][name] = round(timed(lambda: [dec.step(params) for _ in range(a.steps)]) / a.steps * 1000,
                                         3)
    admit(eng, dec, ids, sampled, slots)
    scored = ids[:, :4].expand(dec.rows, 4).contiguous()  # three scored steps: three calls of the scored variant
    out["compile_s"]["scored"] = round(timed(lambda: dec.score(scored)), 2)
    return out


def kl_top1(lp_a: torch.Tensor, lp_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row KL(a || b) and whether the argmaxes agree, from log-probs (rows, V)."""
    return (lp_a.exp() * (lp_a - lp_b)).sum(-1), (lp_a.argmax(-1) == lp_b.argmax(-1)).float()


@torch.inference_mode()
def forward_gap(a, eng, whole, regional, ids, forced, slots, params) -> dict:
    """Per-step logits of the whole-step compiled forward, the regional forward and the eager forward along one
    forced continuation, every row fed the same tokens under its own lever."""
    whole_fwd = torch.compile(whole._forward_step, mode=a.mode, fullgraph=True, dynamic=False)
    runs = {}
    for name, dec, fwd in (("eager", whole, whole._forward_step), ("whole", whole, whole_fwd),
                           ("regional", regional, regional._regional_forward)):
        admit(eng, dec, ids, params, slots)
        steps = []
        for t in range(forced.shape[1]):
            x = forced[:, t:t + 1].expand(dec.rows, 1).contiguous()
            steps.append(F.log_softmax(fwd(x).float(), -1).clone())
            dec._advance_host()
        runs[name] = steps
    out = {}
    for pa, pb in (("whole", "regional"), ("eager", "whole"), ("eager", "regional")):
        kls, top1 = zip(*(kl_top1(x, y) for x, y in zip(runs[pa], runs[pb])))
        kl_t, top1_t = torch.stack(kls), torch.stack(top1)
        out[f"{pa}_vs_{pb}"] = {"kl_mean": float(kl_t.mean()), "kl_max": float(kl_t.max()),
                                "top1": float(top1_t.mean())}
    return out


@torch.inference_mode()
def score_gap(eng, whole, regional, ids, forced, slots, params) -> dict:
    lps = []
    for dec in (whole, regional):
        admit(eng, dec, ids, params, slots)
        lps.append(dec.score(forced.expand(dec.rows, -1).contiguous()).float())
    d = (lps[0] - lps[1]).abs()
    return {"max_abs_logprob_diff": float(d.max()), "mean_abs_logprob_diff": float(d.mean())}


@torch.inference_mode()
def greedy_gap(a, eng, whole, regional, ids, slots, params) -> dict:
    seqs = []
    for dec in (whole, regional):
        admit(eng, dec, ids, params, slots)
        toks = [dec.cur_tokens.view(-1).tolist()]
        for _ in range(a.steps):
            toks.append(dec.step(params).view(-1).tolist())
        seqs.append(np.array(toks))  # (steps + 1, rows)
    same = seqs[0] == seqs[1]
    depth = [int(np.argmin(same[:, r])) if not same[:, r].all() else same.shape[0] for r in range(same.shape[1])]
    return {"divergence_depth_per_row": depth, "tokens": same.shape[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_size", default="1b-tsumugi")
    ap.add_argument("--rope_fp32", action="store_true")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cascade", choices=["off", "on"], default="on")
    ap.add_argument("--cascade_tail_tokens", type=int, default=512)
    ap.add_argument("--levers", type=int, default=4, help="lever slots (0: no tables); rows alternate a bound lever "
                                                          "and the identity")
    ap.add_argument("--mode", default="max-autotune", help="the whole-step mode; the regions use its no-cudagraphs "
                                                           "form")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=64)
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
    eng = DecodeEngine(model, max_seq_len=a.max_seq_len, levers=a.levers)
    slots = None
    if a.levers > 0:
        rng = np.random.default_rng(a.seed)
        v = rng.standard_normal(cfg.hidden_size).astype(np.float32)
        v *= 8.0 / np.linalg.norm(v)
        s = eng.levers.bind(LeverPlan(attn={cfg.num_layers // 2: v}, span="decode"))
        slots = [s if r % 2 == 0 else 0 for r in range(a.rows)]
    whole, regional = make_decoder(a, eng), make_decoder(a, eng)
    whole.compile_step(mode=a.mode)
    regional.compile_step(mode=a.mode, regional=True)
    sampled = SamplingParams(temperature=a.temperature, repetition_penalty=a.repetition_penalty)
    greedy = SamplingParams(temperature=0.0, repetition_penalty=a.repetition_penalty)
    ids = torch.tensor([tok.encode(PROMPT)], device="cuda")
    forced = torch.tensor([tok.encode(FORCED)], device="cuda")
    res: dict = {"checkpoint": a.checkpoint, "rows": a.rows, "cascade": a.cascade, "levers": a.levers,
                 "mode": a.mode, "cold": a.cold, "layers": regional.n_layers,
                 "layer_signatures": regional.regional_variants}
    torch.manual_seed(a.seed)
    res["whole"] = compile_cost(a, eng, whole, ids, slots, sampled, greedy)
    logger.info("whole: %s", json.dumps(res["whole"]))
    torch.manual_seed(a.seed)
    res["regional"] = compile_cost(a, eng, regional, ids, slots, sampled, greedy)
    logger.info("regional: %s", json.dumps(res["regional"]))
    res["forward"] = forward_gap(a, eng, whole, regional, ids, forced, slots, sampled)
    logger.info("forward: %s", json.dumps(res["forward"]))
    res["score"] = score_gap(eng, whole, regional, ids, forced, slots, sampled)
    logger.info("score: %s", json.dumps(res["score"]))
    res["greedy"] = greedy_gap(a, eng, whole, regional, ids, slots, greedy)
    logger.info("greedy: %s", json.dumps(res["greedy"]))
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
