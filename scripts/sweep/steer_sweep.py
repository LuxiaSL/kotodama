#!/usr/bin/env python3
"""Neuron steering sweep on the fast engine: every neuron of chosen effective layers, pushed along its own
down-projection column, one greedy rollout per (prompt, dose), plus random-direction controls and a damage readout.

A unit is a neuron ``(L, n)`` of effective layer L (looped repeats are separate units: they run the same module at
another pass) or a random control (a random direction with the norm of a random neuron's column: the null a unit is ranked against). A
rollout adds ``add * W_down[:, n]`` to layer L's MLP output (kotodama.model.levers, site ``mlp``), with
``add = dose * p50(L) / ||W_down[:, n]||``: the push is ``|dose|`` times the median norm of the state entering L's
MLP (``--dose_norms``; the unit of the neuron explorer, https://github.com/LuxiaSL/neuron-explorer, branch kotodama),
so ``rel_push = |dose|`` for every unit.

Spans (``--mode``):
  * ``decode``: the lever writes generated positions only. One unsteered prefill per prompt is admitted into every
    row (cascade attention reads the prompt once per group).
  * ``uniform``: the lever writes every position. Each row feeds its prompt through the decode step under its own
    lever (``CascadeDecoder.force``), as the explorer's hook steers prompt and decode alike.

Every batch holds one prompt and decodes ``--rows`` rows together, row 0 unsteered. Per row it measures:
  * damage: the mean logprob drop, against row 0, of the prompt's fixed unsteered greedy continuation (its tokens
    1..``--damage_tokens``-1, teacher-forced under the row's lever) — the matched-damage axis for the controls;
  * the rollout: ``--max_tokens`` greedy tokens with the chosen token's logprob and the top-k at every step;
  * captures: the MLP input (post-``ffn_norm``) at L + ``--capture_offsets``: at the last prompt token (uniform:
    per row; decode: per prompt, every row shares it) and, in decode mode, at the first generated token (per row).

Output (the explorer's run-file shape; scripts/sweep/merge.py assembles the per-layer runs from the shards):
  <out>/shards/<mode>/L{L}.s{shard}.jsonl          one line per rollout
  <out>/shards/<mode>/L{L}.s{shard}.p{k}.npz       per batch: top_ids, top_lps, mlp_in (per row), rollout keys
  <out>/shards/<mode>/L{L}.prompt_caps.npz         decode mode: the last-prompt-token captures per prompt
Rerunning the same command resumes: rollouts already in a shard's jsonl are skipped.

Usage (node, one GPU per shard):
  tools/run_py.sh scripts/sweep/steer_sweep.py --checkpoint <model.pt> --model_size 1b-tsumugi --rope_fp32 \\
      --dose_norms <p50_ffn_in.json> --mode decode --layers 45,46 --shard 0 --n_shards 8 --out <sweep dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model.decode_engine import DecodeEngine, SamplingParams  # noqa: E402
from kotodama.model.levers import LeverPlan, capture  # noqa: E402
from kotodama.model.multi_decode import CascadeDecoder  # noqa: E402
from kotodama.serve import server  # noqa: E402
from kotodama.serve.chatml import BASE_STOP_TOKEN_IDS  # noqa: E402

logger = logging.getLogger("steer_sweep")
TOKENIZER: list = [None]  # set by main

# Raw-text openings (the base model continues them): narrative, dialogue, exposition, code, instruction, poetry.
PROMPTS: dict[str, str] = {
    "harbor": "The lighthouse keeper had not spoken to anyone in three weeks when the boat finally came.",
    "council": "Chapter 4\n\nThe council met at dawn. Nobody wanted to be the first to say",
    "letter": "Dear Margaret,\n\nI hope this letter finds you well. Since you left, the house has",
    "interview": "Q: What first drew you to the work?\nA:",
    "science": "Photosynthesis is the process by which plants convert light into chemical energy. In the first stage,",
    "code": "def merge_intervals(intervals):\n    \"\"\"Merge overlapping intervals.\"\"\"\n",
    "recipe": "How to make bread at home\n\nStep 1:",
    "poem": "I walked the long road home at evening,\n",
}
DOSES = (-0.7, 0.8)  # the explorer's main_rel coherent window, one dose per pole


def parse_list(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def run_key(rec: dict) -> tuple:
    return rec["kind"], rec["neuron"], rec["prompt"], float(rec["dose"])


class Units:
    """The units of one effective layer, their push directions and rollout jobs for one shard."""

    def __init__(self, model, layer: int, neurons: list[int], random_frac: float, seed: int,
                 p50: float) -> None:
        w = model.layers[layer].ffn.down_proj.weight.detach().float().cpu()  # (hidden, intermediate)
        self.layer = layer
        self.module_layer = model.config.layer_source(layer)
        self.w = w
        self.norms = w.norm(dim=0)  # per neuron
        self.p50 = float(p50)
        n_ctrl = max(1, round(random_frac * len(neurons)))
        rng = np.random.default_rng(seed * 1000 + layer)
        self.ctrl_ref = rng.integers(w.shape[1], size=n_ctrl)
        vecs = rng.standard_normal((n_ctrl, w.shape[0])).astype(np.float32)
        vecs *= (self.norms[self.ctrl_ref].numpy() / np.linalg.norm(vecs, axis=1))[:, None]
        self.ctrl = torch.from_numpy(vecs)
        units = [("neuron", int(n)) for n in neurons]
        for k in range(n_ctrl):  # spread evenly, so any prefix of the run has its share of controls
            units.insert(round((k + 1) * len(units) / (n_ctrl + 1)), ("random", k))
        self.units = units

    def ref(self, kind: str, n: int) -> int:
        return n if kind == "neuron" else int(self.ctrl_ref[n])

    def add_and_row(self, kind: str, n: int, dose: float) -> tuple[float, np.ndarray]:
        ref = self.ref(kind, n)
        add = dose * self.p50 / float(self.norms[ref])
        vec = self.w[:, n] if kind == "neuron" else self.ctrl[n]
        return add, (add * vec).numpy().astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_size", default="1b-tsumugi")
    ap.add_argument("--rope_fp32", action="store_true")
    ap.add_argument("--dose_norms", required=True,
                    help="json with p50_ffn_in: [n_layers] (median norm of each effective layer's MLP input state)")
    ap.add_argument("--mode", choices=["decode", "uniform"], required=True)
    ap.add_argument("--layers", required=True, help='effective layers: "all", "45", "26,45-48"')
    ap.add_argument("--neurons", default="all")
    ap.add_argument("--doses", default=",".join(f"{d:g}" for d in DOSES))
    ap.add_argument("--random_frac", type=float, default=0.02)
    ap.add_argument("--prompts_json", default=None, help="{key: text}; default: the built-in raw-text openings")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--group_rows", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--damage_tokens", type=int, default=16)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--capture_offsets", default="0,1,4,8")
    ap.add_argument("--first_token_offsets", default="1,4",
                    help="decode mode: capture offsets also taken at the first generated token")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--no_compile", action="store_true", help="eager decoders (CPU smoke tests)")
    ap.add_argument("--compile_mode", default="max-autotune",
                    help="torch.compile mode of the engine and decoder (reduce-overhead starts in minutes)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not 0 <= a.shard < a.n_shards:
        raise SystemExit(f"--shard {a.shard} outside [0, {a.n_shards})")
    torch._dynamo.config.recompile_limit = 64
    server._rope_fp32_override = a.rope_fp32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = server.load_model(a.checkpoint, device, mode="base", model_size=a.model_size, engine="reference",
                                   max_seq_len=a.max_seq_len)
    TOKENIZER[0] = tok
    cfg = model.config
    n_layers, n_neurons = cfg.num_layers, cfg.intermediate_size
    norms_doc = json.loads(Path(a.dose_norms).read_text())
    p50 = norms_doc["p50_ffn_in"]
    if len(p50) != n_layers or not all(float(x) > 0 for x in p50):
        raise SystemExit(f"--dose_norms needs {n_layers} positive p50_ffn_in values, got {len(p50)}")
    norms_sha = hashlib.sha256(Path(a.dose_norms).read_bytes()).hexdigest()
    layers = parse_list(a.layers, n_layers)
    neurons = parse_list(a.neurons, n_neurons)
    doses = [float(x) for x in a.doses.split(",") if x.strip()]
    offsets = [int(x) for x in a.capture_offsets.split(",") if x.strip()]
    first_offsets = [int(x) for x in a.first_token_offsets.split(",") if x.strip()] if a.mode == "decode" else []
    prompts = json.loads(Path(a.prompts_json).read_text()) if a.prompts_json else dict(PROMPTS)
    prompt_ids = {k: tok.encode(v) for k, v in prompts.items()}  # whole text, one encode
    max_prompt = max(len(v) for v in prompt_ids.values())
    tail = (max_prompt if a.mode == "uniform" else 0) + a.damage_tokens + a.max_tokens + 8
    if max_prompt + a.damage_tokens + a.max_tokens + 8 >= a.max_seq_len:
        raise SystemExit(f"--max_seq_len {a.max_seq_len} too small for a {max_prompt}-token prompt + budgets")

    cap_layers = {L: [L + o for o in offsets if 0 <= L + o < n_layers] for L in layers}
    first_layers = {L: [L + o for o in first_offsets if 0 <= L + o < n_layers] for L in layers}
    dec_caps = sorted({x for L in layers for x in (cap_layers[L] if a.mode == "uniform" else first_layers[L])})

    t0 = time.time()
    eng = DecodeEngine(model, max_seq_len=a.max_seq_len, levers=a.rows)
    groups = max(1, a.rows // a.group_rows)
    dec = CascadeDecoder(eng, a.rows, groups=groups, group_rows=a.group_rows, tail_tokens=tail,
                         readout_k=a.top_k, capture_layers=dec_caps)
    if not a.no_compile:
        eng.compile_step(mode=a.compile_mode)
        eng.warm_blocks()
        dec.compile_step(mode=a.compile_mode)
    greedy = SamplingParams(temperature=0.0, repetition_penalty=1.0)
    stop_ids = BASE_STOP_TOKEN_IDS
    logger.info("engine + %d-row decoder (%d groups, %d-token tails) ready in %.0fs; lever tables %.0f MB",
                a.rows, groups, tail, time.time() - t0, eng.levers.nbytes() / 1e6)

    # The fixed unsteered continuation of each prompt, for the damage readout.
    cont: dict[str, list[int]] = {}
    with torch.inference_mode():
        for k, ids in prompt_ids.items():
            logits = eng.prefill(torch.tensor([ids], device=device))
            t = logits.argmax(-1).view(1, 1)
            seq = [int(t)]
            for _ in range(a.damage_tokens - 1):
                t = eng.step_logits(t.clone())[0, -1].argmax().view(1, 1)
                seq.append(int(t))
            cont[k] = seq

    out_dir = Path(a.out) / "shards" / a.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"checkpoint": a.checkpoint, "model_size": a.model_size, "span": a.mode, "design_name": "main_rel",
            "dose_unit": "rel_ffn_in", "dose_norms": str(Path(a.dose_norms).resolve()), "dose_norms_sha256": norms_sha,
            "doses": doses, "prompts": prompts, "prompt_ids": prompt_ids, "max_tokens": a.max_tokens,
            "damage_tokens": a.damage_tokens, "damage_continuation": cont, "top_k": a.top_k, "hook": "mlp_out",
            "capture_hook": "mlp_in", "capture_offsets": offsets, "first_token_offsets": first_offsets,
            "random_frac": a.random_frac, "seed": a.seed, "rows": a.rows, "n_shards": a.n_shards,
            "n_layers": n_layers}
    (out_dir / f"meta.s{a.shard}.json").write_text(json.dumps(meta, indent=1))

    stats = {"rollouts": 0, "batches": 0, "gen_tokens": 0, "t_decode": 0.0}
    for L in layers:
        units = Units(model, L, neurons, a.random_frac, a.seed, p50[L])
        mine = [u for i, u in enumerate(units.units) if i % a.n_shards == a.shard]
        jsonl = out_dir / f"L{L}.s{a.shard}.jsonl"
        done = set()
        part = 0
        if jsonl.exists():
            for line in jsonl.read_text().splitlines():
                rec = json.loads(line)
                done.add(run_key(rec))
                part = max(part, rec["part"] + 1)
        jobs = []
        if a.shard == 0:
            jobs += [{"kind": "baseline", "neuron": -1, "prompt": p, "dose": 0.0} for p in prompt_ids]
        jobs += [{"kind": kind, "neuron": n, "prompt": p, "dose": d}
                 for kind, n in mine for p in prompt_ids for d in doses]
        jobs = [j for j in jobs if run_key(j) not in done]
        logger.info("L%d (module %d): %d units in this shard, %d rollouts to run (%d done)", L, units.module_layer,
                    len(mine), len(jobs), len(done))

        if a.mode == "decode" and a.shard == 0 and not (out_dir / f"L{L}.prompt_caps.npz").exists():
            caps = np.zeros((len(prompt_ids), len(cap_layers[L]), cfg.hidden_size), np.float16)
            for i, ids in enumerate(prompt_ids.values()):
                with torch.inference_mode(), capture(eng, layers=cap_layers[L], what=("mlp_in",)) as cap:
                    eng.prefill(torch.tensor([ids], device=device))
                for j, cl in enumerate(cap_layers[L]):
                    caps[i, j] = cap.engine["mlp_in"][cl][0, -1].float().cpu().numpy()
            np.savez(out_dir / f"L{L}.prompt_caps.npz", mlp_in=caps, prompts=np.array(list(prompt_ids)),
                     layers=np.array(cap_layers[L]))

        by_prompt: dict[str, list[dict]] = {}
        for j in jobs:
            by_prompt.setdefault(j["prompt"], []).append(j)
        with open(jsonl, "a") as f:
            for pk, pjobs in by_prompt.items():
                ids = torch.tensor([prompt_ids[pk]], device=device)
                plen = ids.shape[1]
                c = torch.tensor([cont[pk]], device=device)
                for b0 in range(0, len(pjobs), a.rows - 1):
                    batch = pjobs[b0:b0 + a.rows - 1]
                    tb = time.perf_counter()
                    rec_rows = run_batch(a, eng, dec, units, batch, ids, plen, c, greedy, stop_ids,
                                         cap_layers[L], first_layers[L], dec_caps)
                    write_part(f, out_dir, L, a.shard, part, rec_rows)
                    part += 1
                    dt = time.perf_counter() - tb
                    stats["rollouts"] += len(batch)
                    stats["batches"] += 1
                    stats["gen_tokens"] += sum(r["n_gen"] for r in rec_rows["records"])
                    stats["t_decode"] += dt
                    if stats["batches"] % 20 == 1:
                        logger.info("L%d %s: %d rollouts in %.2fs (%.0f rows-tok/s; ms %s); total %d rollouts, "
                                    "%.0f tok/s", L, pk, len(batch), dt,
                                    a.rows * (a.damage_tokens + a.max_tokens) / dt, json.dumps(rec_rows["timing_ms"]),
                                    stats["rollouts"], stats["gen_tokens"] / max(stats["t_decode"], 1e-9))
    logger.info("done: %s", json.dumps(stats))
    return 0


def write_part(f, out_dir: Path, layer: int, shard: int, part: int, batch_out: dict) -> None:
    """One batch's arrays as part ``part`` of the shard, then its records (each naming its part row) to ``f``: a
    record is written only once its arrays are on disk."""
    np.savez(out_dir / f"L{layer}.s{shard}.p{part}.npz", **batch_out["arrays"])
    for i, rec in enumerate(batch_out["records"]):
        f.write(json.dumps({**rec, "part": part, "part_row": i}) + "\n")
    f.flush()


@torch.inference_mode()
def run_batch(a, eng: DecodeEngine, dec: CascadeDecoder, units: Units, batch: list[dict], ids: torch.Tensor,
              plen: int, cont: torch.Tensor, params: SamplingParams, stop_ids: frozenset[int], cap_layers: list[int],
              first_layers: list[int], dec_caps: list[int]) -> dict:
    """One prompt's rollouts in one decoder batch: row 0 unsteered (the damage reference), rows 1.. the jobs."""
    L = units.layer
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    marks = [("start", time.perf_counter())]

    def mark(name: str) -> None:
        sync()
        marks.append((name, time.perf_counter()))

    for s in eng.levers.bound:
        eng.levers.release(s)
    slots, adds = [0], [0.0]
    for j in batch:
        if j["kind"] == "baseline":
            slots.append(0)
            adds.append(0.0)
            continue
        add, row = units.add_and_row(j["kind"], j["neuron"], j["dose"])
        slots.append(eng.levers.bind(LeverPlan(mlp={L: row}, span=a.mode)))
        adds.append(add)
    pad = dec.rows - len(slots)
    slots += [0] * pad
    mark("levers")
    if a.mode == "decode":
        logits = eng.prefill(ids)
        first = dec.admit(list(range(dec.rows)), logits, params, slots=slots)
    else:
        dec.admit_empty(slots)
        first = dec.force(ids, params)
    first_lp = dec.step_logprob.clone()
    first_ids, first_lps = dec.step_top_ids.clone(), dec.step_top_lps.clone()
    prompt_caps = dec.mlp_in.clone() if (a.mode == "uniform" and dec_caps) else None
    mark("prompt")
    # damage: the continuation's tokens 1.. under each row's lever, against row 0
    lp = dec.score(cont)  # (rows, D - 1)
    damage = (lp[0:1] - lp).mean(dim=1)
    mark("score")
    dec.rewind(plen, first)
    gen = [torch.tensor(first, device=lp.device)]
    lps, top_ids, top_lps = [first_lp], [first_ids], [first_lps]
    first_caps = None
    for t in range(a.max_tokens - 1):
        gen.append(dec.step(params).view(-1).clone())
        lps.append(dec.step_logprob.clone())
        top_ids.append(dec.step_top_ids.clone())
        top_lps.append(dec.step_top_lps.clone())
        if t == 0 and a.mode == "decode" and dec_caps:
            first_caps = dec.mlp_in.clone()  # the state the first generated token fed in
    mark("decode")
    gen_t = torch.stack(gen, 1).cpu().numpy()  # (rows, T)
    lps_t = torch.stack(lps, 1).cpu().numpy()
    top_ids_t = torch.stack(top_ids, 1).cpu().numpy().astype(np.int32)
    top_lps_t = torch.stack(top_lps, 1).cpu().numpy().astype(np.float16)
    damage = damage.cpu().numpy()
    tokz = TOKENIZER[0]
    records, keep = [], []
    for i, j in enumerate(batch):
        r = i + 1
        toks = gen_t[r].tolist()
        n = next((k for k, x in enumerate(toks) if x in stop_ids), len(toks))
        finish = "stop" if n < len(toks) else "length"
        toks = toks[:n]
        top_ids_t[r, n:] = -1  # steps after the stop token were decoded but are not part of the rollout
        top_lps_t[r, n:] = 0.0
        ref = None if j["kind"] == "baseline" else units.ref(j["kind"], j["neuron"])
        records.append({
            "layer": L, "module_layer": units.module_layer, "span": a.mode, "kind": j["kind"],
            "neuron": j["neuron"], "ref_neuron": ref, "prompt": j["prompt"], "dose": j["dose"], "add": adds[r],
            "rel_push": abs(j["dose"]) if ref is not None else 0.0, "damage": round(float(damage[r]), 5),
            "text": tokz.decode(toks, skip_special_tokens=True),
            "finish": finish, "n_gen": n, "token_ids": toks, "logprob": [round(float(x), 4) for x in lps_t[r][:n]],
            "capture_id": f"L{L}_{j['kind']}{j['neuron']}_{j['prompt']}_{j['dose']:+g}".replace("+", "p")
            .replace("-", "m"),
        })
        keep.append(r)
    arrays = {"top_ids": top_ids_t[keep], "top_lps": top_lps_t[keep]}
    if prompt_caps is not None:
        arrays["mlp_in"] = prompt_caps[keep][:, [dec_caps.index(x) for x in cap_layers]].float().cpu().numpy() \
            .astype(np.float16)
    if first_caps is not None and first_layers:
        arrays["mlp_in_first"] = first_caps[keep][:, [dec_caps.index(x) for x in first_layers]].float().cpu().numpy() \
            .astype(np.float16)
    mark("host")
    timing = {name: round((t - marks[i][1]) * 1000, 1) for i, (name, t) in enumerate(marks[1:])}
    return {"records": records, "arrays": arrays, "timing_ms": timing}


if __name__ == "__main__":
    raise SystemExit(main())
