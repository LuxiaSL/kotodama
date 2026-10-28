#!/usr/bin/env python3
"""Lever tables on a real checkpoint (GPU, compiled): what the tables change on unsteered requests, and what they cost.

Builds two engines over one model, without lever tables and with them (``--levers`` slots, a few bound), each
compiled as the server compiles it (max-autotune, block buckets warmed), plus a row decoder per ``--rows`` width on
each (``--cascade`` picks per-row KV or cascade decoders). Sections (``--sections``):
  * ``identity``: on every prompt, the engines' prefill logits, compiled forward logits along a forced continuation
    and fused sampled tokens (same RNG seed) are compared bit for bit; then each decoder width decodes sampled tokens
    with every row on the identity slot, and again with the rows alternating bound levers and the identity, against
    the table-free decoder on the same seed. Reported, not required: the identity slot is exact per operation, but a
    compiled graph with the lever adds fuses differently from one without them, so the two engines agree to within
    kernel rounding, not bit for bit (``gap`` measures how far).
  * ``gap``: per prompt, KL(table-free || levers) and top-1 agreement of the two engines' compiled forwards along a
    forced continuation, and the greedy divergence depth (tokens until the two greedy continuations first differ).
  * ``cost``: single-stream fused step ms and batched ms/step per width, without tables and with them (every row
    on a bound lever), as the tables' per-step price.
  * ``steered``: per width, the compiled batched step with rows on bound levers vs the eager batched forward of the
    same rows (per-step KL and top-1 agreement along a forced continuation): the compiled lever add reads the slot
    current at replay.

Usage (node, GPU):
  tools/run_py.sh scripts/benchmark/levers_check.py --checkpoint <model.pt> --model_size 1b-tsumugi --rope_fp32 \\
      --rows 8,64 --cascade on --out <dir>/levers_check.json
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
from kotodama.model.levers import LeverPlan, neuron_row, write_plan  # noqa: E402
from kotodama.model.multi_decode import CascadeDecoder, MultiSampleDecoder  # noqa: E402
from kotodama.serve import server  # noqa: E402

logger = logging.getLogger("levers_check")

PROMPTS = [
    "The lighthouse keeper had not spoken to anyone in three weeks when the boat finally came.",
    "Chapter 4\n\nThe council met at dawn. Nobody wanted to be the first to say what everyone already knew:",
    "def merge_intervals(intervals):\n    \"\"\"Merge overlapping intervals.\"\"\"\n",
    "Q: Why does ice float on water?\nA:",
]


def build(model, levers: int, max_seq_len: int, widths: list[int], cascade: bool, tail: int):
    eng = DecodeEngine(model, max_seq_len=max_seq_len, levers=levers)
    eng.compile_step(mode="max-autotune")
    eng.warm_blocks()
    decs: dict[int, MultiSampleDecoder] = {}
    for w in sorted(widths, reverse=True):
        widest = decs[max(decs)] if decs else None
        if cascade:
            dec = CascadeDecoder(eng, w, groups=max(1, w // 8), group_rows=8, tail_tokens=tail, kv_from=widest)
        else:
            dec = MultiSampleDecoder(eng, w, kv_from=widest)
        dec.compile_step(mode="max-autotune")
        decs[w] = dec
    return eng, decs


def bound_plans(model, n_layers: int, boundaries, hidden: int, seed: int) -> list[LeverPlan]:
    """A few levers of every kind: an attn K write, an attn S write in the loop's second pass, a neuron write."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(hidden).astype(np.float32)
    v *= 8.0 / np.linalg.norm(v)
    return [
        LeverPlan(attn=write_plan([29], v[None], boundaries, n_layers, "K"), span="decode"),
        LeverPlan(attn={46: -v}, span="decode"),
        LeverPlan(mlp={45: neuron_row(model, 45, 7, 6.0)}, span="decode"),
    ]


@torch.inference_mode()
def section_identity(a: argparse.Namespace, tok, plain, plain_decs, lev, lev_decs, slots, params) -> dict:
    out: dict = {"single": [], "rows": {}}
    for text in PROMPTS:
        ids = torch.tensor([tok.encode(text)], device="cuda")
        forced = torch.tensor([tok.encode(" and then the rest of it followed, slowly, one word at a time.")],
                              device="cuda")
        la, lb = plain.prefill(ids), lev.prefill(ids)
        same_prefill = bool(torch.equal(la, lb))
        same_fwd = True
        for t in range(forced.shape[1]):
            x = forced[:, t:t + 1].clone()  # a fresh offset-0 tensor: a slice's offset would recompile per step
            fa = plain.step_logits(x).clone()
            fb = lev.step_logits(x).clone()
            same_fwd &= bool(torch.equal(fa, fb))
        plain.prefill(ids), lev.prefill(ids)
        torch.manual_seed(a.seed)
        ta = plain.sample_first(la, params).clone()
        seq_a = [int(ta)]
        for _ in range(a.steps):
            ta = plain.step(ta, params)
            seq_a.append(int(ta))
        torch.manual_seed(a.seed)
        tb = lev.sample_first(lb, params).clone()
        seq_b = [int(tb)]
        for _ in range(a.steps):
            tb = lev.step(tb, params)
            seq_b.append(int(tb))
        out["single"].append({"prompt_tokens": ids.shape[1], "prefill_bitwise": same_prefill,
                              "forward_bitwise": same_fwd, "sampled_equal": seq_a == seq_b})
    ids = torch.tensor([tok.encode(PROMPTS[1])], device="cuda")
    for w in sorted(plain_decs):
        res = {}
        for layout in ("all_identity", "mixed"):
            row_slots = [0] * w if layout == "all_identity" else [slots[r % len(slots)] if r % 2 == 0 else 0
                                                                  for r in range(w)]
            seqs = []
            for eng, dec, ss in ((plain, plain_decs[w], None), (lev, lev_decs[w], row_slots)):
                logits = eng.prefill(ids)
                dec.release(range(w))
                torch.manual_seed(a.seed)
                dec.admit(list(range(w)), logits, params, slots=ss)
                toks = [dec.cur_tokens.view(-1).tolist()]
                for _ in range(a.steps):
                    toks.append(dec.step(params).view(-1).tolist())
                seqs.append(np.array(toks))  # (steps + 1, w)
            ident = [r for r in range(w) if row_slots[r] == 0]
            steered = [r for r in range(w) if row_slots[r] != 0]
            res[layout] = {
                "identity_rows_equal": bool((seqs[0][:, ident] == seqs[1][:, ident]).all()),
                "identity_rows": len(ident),
                "steered_rows_differ": (bool((seqs[0][:, steered] != seqs[1][:, steered]).any())
                                        if steered else None),
            }
        out["rows"][w] = res
    return out


@torch.inference_mode()
def section_cost(a: argparse.Namespace, tok, plain, plain_decs, lev, lev_decs, slots, params) -> dict:
    ids = torch.tensor([tok.encode(PROMPTS[0])], device="cuda")

    def single(eng) -> float:
        logits = eng.prefill(ids)
        t = eng.sample_first(logits, params)
        for _ in range(16):
            t = eng.step(t, params)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            t = eng.step(t, params)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / a.steps * 1000

    def batched(eng, dec, ss) -> float:
        logits = eng.prefill(ids)
        dec.release(range(dec.rows))
        dec.admit(list(range(dec.rows)), logits, params, slots=ss)
        for _ in range(16):
            dec.step(params)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            dec.step(params)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / a.steps * 1000

    out = {"single_ms": {"plain": single(plain), "levers": single(lev)}, "rows_ms": {}}
    for w in sorted(plain_decs):
        ss = [slots[r % len(slots)] for r in range(w)]
        out["rows_ms"][w] = {"plain": batched(plain, plain_decs[w], None), "levers": batched(lev, lev_decs[w], ss)}
    return out


@torch.inference_mode()
def section_gap(a: argparse.Namespace, tok, plain, lev, params) -> dict:
    """The serving gap the tables open on UNSTEERED requests: per prompt, the compiled forward of the lever engine
    (slot 0) vs the table-free engine along one forced continuation (per-step KL(plain || levers), top-1 agreement),
    and each engine's own greedy continuation (how many tokens before they first differ)."""
    forced = torch.tensor([tok.encode(" The water was very still that morning, and the gulls had gone quiet over"
                                      " the harbor wall, where the old nets still hung drying in the wind.")],
                          device="cuda")
    out = []
    for text in PROMPTS:
        ids = torch.tensor([tok.encode(text)], device="cuda")
        plain.prefill(ids)
        lev.prefill(ids)
        kl, top1 = [], []
        for t in range(forced.shape[1]):
            x = forced[:, t:t + 1].clone()
            lp_a = F.log_softmax(plain.step_logits(x)[0, -1].float(), -1)
            lp_b = F.log_softmax(lev.step_logits(x)[0, -1].float(), -1)
            kl.append(float((lp_a.exp() * (lp_a - lp_b)).sum()))
            top1.append(bool(lp_a.argmax() == lp_b.argmax()))
        seqs = []
        for eng in (plain, lev):
            t = eng.prefill(ids).argmax(-1).view(1, 1)
            seq = [int(t)]
            for _ in range(a.steps - 1):
                t = eng.step_logits(t.clone())[0, -1].argmax().view(1, 1)
                seq.append(int(t))
            seqs.append(seq)
        depth = next((i for i, (x, y) in enumerate(zip(*seqs)) if x != y), len(seqs[0]))
        out.append({"prompt_tokens": ids.shape[1], "kl_mean": float(np.mean(kl)), "kl_max": float(np.max(kl)),
                    "top1": float(np.mean(top1)), "greedy_divergence_depth": depth, "greedy_tokens": len(seqs[0])})
    return {"prompts": out, "kl_mean": float(np.mean([r["kl_mean"] for r in out])),
            "top1": float(np.mean([r["top1"] for r in out])),
            "greedy_divergence_depth_median": float(np.median([r["greedy_divergence_depth"] for r in out]))}


@torch.inference_mode()
def section_steered(a: argparse.Namespace, tok, lev, lev_decs, slots, params) -> dict:
    ids = torch.tensor([tok.encode(PROMPTS[0])], device="cuda")
    forced = torch.tensor(tok.encode(" The water was very still that morning, and the gulls had gone quiet."),
                          device="cuda")
    out = {}
    for w, dec in sorted(lev_decs.items()):
        ss = [slots[r % len(slots)] if r % 2 == 0 else 0 for r in range(w)]
        logits = lev.prefill(ids)
        dec.release(range(w))
        dec.admit(list(range(w)), logits, params, slots=ss)
        # eager forward of the forced continuation, then the same rows through the compiled step's forward
        eager = []
        for t in range(forced.shape[0]):
            eager.append(dec.step_logits(forced[t].view(1, 1).expand(w, 1).contiguous()).float())
        logits = lev.prefill(ids)
        dec.release(range(w))
        dec.admit(list(range(w)), logits, params, slots=ss)
        compiled_fwd = torch.compile(dec._forward_step, mode="max-autotune", fullgraph=True, dynamic=False)
        comp = []
        for t in range(forced.shape[0]):
            comp.append(compiled_fwd(forced[t].view(1, 1).expand(w, 1).contiguous()).float().clone())
            dec._advance_host()
        kl, top1 = [], []
        for e, c in zip(eager, comp):
            lp_e, lp_c = F.log_softmax(e, -1), F.log_softmax(c, -1)
            kl.append((lp_e.exp() * (lp_e - lp_c)).sum(-1))
            top1.append((e.argmax(-1) == c.argmax(-1)).float())
        kl_t, top1_t = torch.stack(kl, 1), torch.stack(top1, 1)  # (w, steps)
        st = [r for r in range(w) if ss[r]]
        out[w] = {"kl_mean_steered": float(kl_t[st].mean()), "kl_max_steered": float(kl_t[st].max()),
                  "top1_steered": float(top1_t[st].mean()),
                  "kl_mean_identity": float(kl_t[[r for r in range(w) if not ss[r]]].mean())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_size", default="1b-tsumugi")
    ap.add_argument("--rope_fp32", action="store_true")
    ap.add_argument("--rows", default="8,64")
    ap.add_argument("--cascade", choices=["off", "on"], default="on")
    ap.add_argument("--cascade_tail_tokens", type=int, default=512)
    ap.add_argument("--levers", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--sections", default="identity,gap,cost,steered")
    ap.add_argument("--gap_only", action="store_true", help="build no row decoders; run the gap section only")
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    widths = sorted({int(x) for x in a.rows.split(",") if x.strip()})
    sections = {x.strip() for x in a.sections.split(",") if x.strip()}
    if a.gap_only:
        sections, widths = {"gap"}, []
    torch._dynamo.config.recompile_limit = 64
    server._rope_fp32_override = a.rope_fp32
    model, tok = server.load_model(a.checkpoint, "cuda", mode="base", model_size=a.model_size, engine="reference",
                                   max_seq_len=a.max_seq_len)
    cfg = model.config
    params = SamplingParams(temperature=a.temperature, repetition_penalty=a.repetition_penalty)
    t0 = time.time()
    plain, plain_decs = build(model, 0, a.max_seq_len, widths, a.cascade == "on", a.cascade_tail_tokens)
    lev, lev_decs = build(model, a.levers, a.max_seq_len, widths, a.cascade == "on", a.cascade_tail_tokens)
    plans = bound_plans(model, cfg.num_layers, list(cfg.attn_res_boundaries), cfg.hidden_size, a.seed)
    slots = [lev.levers.bind(p) for p in plans]
    res: dict = {"checkpoint": a.checkpoint, "rows": widths, "cascade": a.cascade, "levers": a.levers,
                 "lever_table_mb": round(lev.levers.nbytes() / 1e6, 1), "build_s": round(time.time() - t0, 1)}
    logger.info("built both engines in %.0fs; lever tables %.1f MB", res["build_s"], res["lever_table_mb"])
    if "identity" in sections:
        res["identity"] = section_identity(a, tok, plain, plain_decs, lev, lev_decs, slots, params)
        logger.info("identity: %s", json.dumps(res["identity"]))
    if "cost" in sections:
        res["cost"] = section_cost(a, tok, plain, plain_decs, lev, lev_decs, slots, params)
        logger.info("cost: %s", json.dumps(res["cost"]))
    if "gap" in sections:
        res["gap"] = section_gap(a, tok, plain, lev, params)
        logger.info("gap: %s", json.dumps(res["gap"]))
    if "steered" in sections:
        res["steered"] = section_steered(a, tok, lev, lev_decs, slots, params)
        logger.info("steered: %s", json.dumps(res["steered"]))
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
