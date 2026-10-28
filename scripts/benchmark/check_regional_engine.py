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
    difference, and whether the greedy rows agree;
  * ``sign`` (``--sections sign``): decode signing (kotodama.model.signing) on a cascade decoder of ``--sign_rows``
    rows, regional and captured: (1) signing on vs off, mixed laws and a decode-span lever on half the rows, tokens
    and per-step readouts compared bit for bit, and the step time of each; (2) the signed surfaces of two rows
    against the replay capture (kotodama.model.capture.cached_forward over the decoded sequence), relative L2 per
    surface; (3) batchmate independence at a fixed width: target rows decoded beside their own prompt's rows and
    beside another prompt's rows (another cascade grouping), differing coordinates per surface; (4) the cross-width
    floor: the same rows decoded at narrower decoder widths, relative L2 per surface against the widest.

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


def _rel(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    return float((x - y).norm() / y.norm().clamp_min(1e-30))


@torch.inference_mode()
def sign_check(a, eng: DecodeEngine, tok, slot: int) -> dict:
    """Decode signing on a regional cascade decoder; see the module docstring (``sign``)."""
    from kotodama.model.capture import cached_forward, hidden_rows
    from kotodama.model.signing import SignSpec

    B, steps = a.sign_rows, a.sign_steps
    spec = SignSpec(tuple(int(x) for x in a.sign_layers.split(",")), tail=steps, depth=True)
    laws = [SamplingParams(temperature=a.temperature, repetition_penalty=a.repetition_penalty),
            SamplingParams(temperature=a.temperature, repetition_penalty=a.repetition_penalty, top_p=0.9, min_p=0.05)]
    prompts = [torch.tensor([tok.encode(PROMPT)], device="cuda"),
               torch.tensor([tok.encode(FORCED.strip() + " " + PROMPT)], device="cuda")]
    seeds = [1000 + r for r in range(B)]
    slots = [slot if r % 2 else 0 for r in range(B)]

    def make(rows: int, sign):
        dec = CascadeDecoder(eng, rows, groups=rows // 32 + 1, group_rows=min(rows, 32),
                             tail_tokens=steps + 8, readout_k=4, seeded=True, sign=sign)
        dec.compile_step(mode=a.mode, regional=True)
        return dec

    def run(dec, law, layout: list[int]) -> tuple[list, list, float, dict]:
        """Rows admitted from ``prompts[layout[r]]`` (each prompt's rows in one admit), ``steps`` steps; per step the
        tokens and readouts; ms per step past the first 8; each row's first token (sampled at admission)."""
        dec.release(range(dec.rows))
        firsts = {}
        for p in sorted(set(layout)):
            rows = [r for r in range(dec.rows) if layout[r] == p]
            firsts.update(zip(rows, dec.admit(rows, eng.prefill(prompts[p]), law, slots=[slots[r] for r in rows],
                                              seeds=[seeds[r] for r in rows])))
        toks, reads = [], []
        t_steady = 0.0
        for i in range(steps):
            t0 = time.perf_counter()
            toks.append(dec.step(law).view(-1).clone())
            reads.append((dec.step_logprob.clone(), dec.step_top_lps.clone(), dec.step_top_ids.clone()))
            if i >= 8:
                torch.cuda.synchronize()
                t_steady += time.perf_counter() - t0
        return toks, reads, t_steady / max(1, steps - 8) * 1000, firsts

    out: dict = {"rows": B, "steps": steps, "layers": list(spec.layers), "depth": spec.depth}
    # (1) on vs off
    base = [0] * B
    ident = {}
    records = {}
    for name, sign in (("off", None), ("on", spec)):
        dec = make(B, sign)
        if sign is not None:
            out["sign_mb"] = round(sum(getattr(dec, f"sign_{n}").numel() * getattr(dec, f"sign_{n}").element_size()
                                       for n in dec.sign_names) / 1e6, 1)
        per_law = []
        for law in laws:
            per_law.append(run(dec, law, base))
        records[name] = per_law
        if sign is not None:
            signed_dec = dec
        else:
            del dec
            torch.cuda.empty_cache()
    for li, law in enumerate(("plain", "truncated")):
        (t0, r0, ms0, _), (t1, r1, ms1, _) = records["off"][li], records["on"][li]
        first_tok = next((i for i, (x, y) in enumerate(zip(t0, t1)) if not torch.equal(x, y)), None)
        first_read = next((i for i, (x, y) in enumerate(zip(r0, r1)) if not all(torch.equal(p, q)
                                                                              for p, q in zip(x, y))), None)
        ident[law] = {"tokens_bitwise": first_tok is None, "first_token_diff_step": first_tok,
                      "readouts_bitwise": first_read is None, "first_readout_diff_step": first_read,
                      "ms_off": round(ms0, 3), "ms_on": round(ms1, 3), "overhead": round(ms1 / ms0 - 1, 4)}
    out["on_off"] = ident
    logger.info("sign on/off: %s", json.dumps(ident))
    # (2) the last law's signed rows vs the replay capture (rows 0 and 1: no lever and the lever)
    model = eng.model
    toks = torch.stack(records["on"][-1][0], dim=1)  # (B, steps)
    firsts = records["on"][-1][3]
    P = prompts[0].shape[1]
    replay = {}
    for r in (0, 1):
        row = signed_dec.signed(r, device=torch.device("cuda"))
        if slots[r]:
            replay[str(r)] = {"lever_slot": row.lever_slot, "skipped": "the replay forward runs no levers"}
            continue
        # the sequence: the prompt, the first token (sampled at admission), then every step's token but the last
        # (sampled, never fed)
        replay[str(r)] = {"positions": row.positions}
        seq = torch.cat([prompts[0][0], torch.tensor([firsts[r]], device="cuda"), toks[r, :-1]]).view(1, -1)
        _, cap, _ = cached_forward(model, seq, past=None, position_offset=0, light=False, capture_hidden=True)
        hid = hidden_rows(cap, model.config.num_layers)
        span = slice(P, P + row.positions)
        want_depth = torch.stack([hid[li + 1][0, span] for li in range(model.config.num_layers)], dim=1)
        rel = {"depth": _rel(row.surfaces["depth"], want_depth)}
        for name, key in (("q", "queries"), ("k", "keys"), ("v", "values")):
            want = torch.stack([cap[key][li][0, :, span].transpose(0, 1) for li in spec.layers], dim=1)
            rel[name] = _rel(row.surfaces[name], want)
        rel["gate"] = _rel(row.surfaces["gate"], torch.stack([cap["gate"][li][0, span] for li in spec.layers], 1))
        replay[str(r)]["rel_l2"] = rel
        del cap
    out["vs_replay"] = replay
    logger.info("sign vs replay: %s", json.dumps(replay))

    def surfaces(dec, rows) -> dict:
        return {r: dec.signed(r, device=torch.device("cuda")).surfaces for r in rows}

    # (3) batchmate independence at the fixed width: targets beside their own prompt vs beside another prompt
    targets = list(range(8))
    run(signed_dec, laws[0], base)
    ref = surfaces(signed_dec, targets)
    mixed = [0 if r < 8 else 1 for r in range(B)]
    run(signed_dec, laws[0], mixed)
    got = surfaces(signed_dec, targets)
    bm = {}
    for name in ref[0]:
        differ = sum(int((ref[r][name] != got[r][name]).sum()) for r in targets)
        bm[name] = {"coords": sum(ref[r][name].numel() for r in targets), "differ": differ}
    out["batchmates_fixed_width"] = bm
    logger.info("sign batchmates: %s", json.dumps(bm))
    del signed_dec
    torch.cuda.empty_cache()
    # (4) the cross-width floor: the same rows at narrower widths against the widest
    floor = {}
    for w in (16, 32, 64):
        if w >= B:
            continue
        dec = make(w, spec)
        run(dec, laws[0], [0] * w)
        sub = surfaces(dec, targets)
        floor[str(w)] = {name: {"differ_frac": round(sum(int((ref[r][name] != sub[r][name]).sum()) for r in targets)
                                                    / sum(ref[r][name].numel() for r in targets), 4),
                                "rel_l2_max": max(_rel(sub[r][name], ref[r][name]) for r in targets)}
                         for name in ref[0]}
        del dec
        torch.cuda.empty_cache()
    out["cross_width_floor"] = floor
    logger.info("sign cross-width floor: %s", json.dumps(floor))
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
                                                         "decoders over a regional engine), sign (decode signing)")
    ap.add_argument("--sign_rows", type=int, default=128)
    ap.add_argument("--sign_steps", type=int, default=256)
    ap.add_argument("--sign_layers", default="0,18,25,34,36,45,54,56,63,71",
                    help="the sampled effective layers (default: the tsumugi lane's)")
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
    if "sign" in sections:
        eng = DecodeEngine(model, max_seq_len=a.max_seq_len, levers=a.levers)
        sslot = eng.levers.bind(plan) if plan is not None else 0
        eng.compile_step(mode=a.mode, regional=True)
        res["sign"] = sign_check(a, eng, tok, sslot)
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
