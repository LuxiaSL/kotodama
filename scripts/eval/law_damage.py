#!/usr/bin/env python3
"""Damage readout of the sampling-law bench (kotodama.posttrain.lawbench): for every cell, the mean NLL of a fixed
continuation set under the cell's whole law, teacher-forced, minus the plain model's NLL of the same tokens.

One decoder batch holds every cell as a row (each with its lever slot, decode span: the prompt is prefilled
unsteered, as the server serves a decode-span request) plus an identity row. For each continuation the prompt is
prefilled once, every row admitted from it, and the continuation's tokens fed through the decode step; at each
position the row's logits pass through its law (``lawbench.law_logprobs``: repetition penalty over the tokens the
law's ``penalty_scope`` counts, temperature, top-p / min-p) and the next continuation token is scored. A token the
truncation removes is counted in ``trunc_share`` and left out of the mean. The plain reference is the identity
row's raw log-softmax (temperature 1, no penalty, no lever).

Usage (one GPU):
  tools/run_py.sh scripts/eval/law_damage.py --checkpoint <ckpt> --steer_npz <banks> \\
      --cells <run>/cells.jsonl --set <run>/damage_set.jsonl --out <run>/damage.json
Output: {"cells": {cell: {"damage", "nll", "trunc_share", "tokens", "panels": {panel: damage}}}, "plain_nll", ...}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model.decode_engine import DecodeEngine, SamplingParams  # noqa: E402
from kotodama.model.multi_decode import MultiSampleDecoder  # noqa: E402
from kotodama.posttrain import lawbench  # noqa: E402

logger = logging.getLogger("law_damage")


def cell_plan(cell: lawbench.Cell, bank: dict, meds: dict, norms: dict, eng: DecodeEngine) -> Optional[Any]:
    """A cell's steering as one lever plan, with the server's dose math (alpha x the site's median norm x the unit
    vector) and write rule; None for an unsteered cell."""
    from kotodama.serve.steering import writes_to_plan

    if not cell.steer:
        return None
    writes = []
    for key, alpha in cell.steer:
        if key not in bank:
            raise SystemExit(f"cell {cell.id}: vector {key!r} is not in the banks ({sorted(bank)})")
        site, vec = bank[key]
        writes.append((site, vec, float(alpha) * float(meds[site])))
    if cell.steer_rule == "N" and not norms:
        raise SystemExit(f"cell {cell.id}: rule N needs partial_norm_l* keys in the banks")
    return writes_to_plan(writes, cell.steer_rule, norms if cell.steer_rule == "N" else None, list(eng._boundaries),
                          eng.n_layers, cell.steer_span)


@torch.inference_mode()
def score_set(eng: DecodeEngine, dec: MultiSampleDecoder, cells: list[lawbench.Cell], slots: list[int],
              prompt_ids: list[int], cont_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(law logprobs (C, T), plain logprobs (T,)) of ``cont_ids`` after ``prompt_ids``: row 0 is the identity, rows
    1..C the cells. The first continuation token is scored on the (unsteered) prefill logits."""
    device = eng.device
    R = dec.rows
    C = len(cells)
    logits0 = eng.prefill(torch.tensor([prompt_ids], device=device))
    row_slots = [0] + slots + [0] * (R - 1 - C)
    dec.admit(list(range(R)), logits0, SamplingParams(temperature=1.0, repetition_penalty=1.0), slots=row_slots)
    temps = [c.temperature for c in cells]
    rps = [c.repetition_penalty for c in cells]
    tps = [c.top_p for c in cells]
    mps = [c.min_p for c in cells]
    law_lp, plain_lp = [], []
    logits = logits0.reshape(1, -1).expand(R, -1)
    for t, tok in enumerate(cont_ids):
        hist = cont_ids[:t]
        pen = [(prompt_ids + hist) if c.penalty_scope == "row" else hist for c in cells]
        tgt = torch.full((C,), tok, dtype=torch.long, device=device)
        law_lp.append(lawbench.law_logprobs(logits[1:1 + C], tgt, temps, rps, pen, tps, mps))
        plain_lp.append(logits[0].float().log_softmax(-1)[tok])
        if t + 1 < len(cont_ids):
            logits = dec.step_logits(torch.full((R, 1), tok, dtype=torch.long, device=device))
    return torch.stack(law_lp, 1), torch.stack(plain_lp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_size", default="1b-tsumugi")
    ap.add_argument("--no_rope_fp32", action="store_true")
    ap.add_argument("--steer_npz", required=True, help="comma list: the banks and the partial-norm table")
    ap.add_argument("--cells", required=True)
    ap.add_argument("--set", required=True, help="damage set jsonl (python -m kotodama.posttrain.lawbench damage-set)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_prompt_tokens", type=int, default=768)
    ap.add_argument("--max_cont_tokens", type=int, default=64)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from kotodama.serve import server
    from kotodama.serve.steering import load_partial_norms, load_steer_banks

    server._rope_fp32_override = not a.no_rope_fp32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = server.load_model(a.checkpoint, device, mode="base", model_size=a.model_size, engine="reference",
                                   max_seq_len=a.max_seq_len)
    cfg = model.config
    paths = a.steer_npz.split(",")
    bank, meds, _ = load_steer_banks(paths, cfg.num_layers, cfg.hidden_size, torch.device(device))
    norms = load_partial_norms(paths, cfg.num_layers)
    cells = lawbench.load_cells(Path(a.cells))
    rows = 1 + len(cells)
    eng = DecodeEngine(model, max_seq_len=a.max_seq_len, levers=max(1, sum(bool(c.steer) for c in cells)))
    dec = MultiSampleDecoder(eng, rows)
    slots = []
    for c in cells:
        plan = cell_plan(c, bank, meds, norms, eng)
        slots.append(0 if plan is None else eng.levers.bind(plan))
    items = lawbench.read_jsonl(Path(a.set))
    acc: dict[str, dict[str, Any]] = {c.id: {"d": 0.0, "nll": 0.0, "n": 0, "trunc": 0, "panels": {}} for c in cells}
    plain_tot, plain_n = 0.0, 0
    t0 = time.time()
    for i, it in enumerate(items):
        p = tok.encode(it["prompt"], add_special_tokens=False)[-a.max_prompt_tokens:]
        c = tok.encode(it["continuation"], add_special_tokens=False)[:a.max_cont_tokens]
        if not p or len(c) < 2 or len(p) + len(c) + 2 >= a.max_seq_len:
            continue
        law_lp, plain_lp = score_set(eng, dec, cells, slots, p, c)
        plain_tot += float(-plain_lp.sum())
        plain_n += len(c)
        for j, cell in enumerate(cells):
            lp = law_lp[j]
            ok = torch.isfinite(lp)
            d = (plain_lp[ok] - lp[ok]).double()
            s = acc[cell.id]
            s["d"] += float(d.sum())
            s["nll"] += float(-lp[ok].double().sum())
            s["n"] += int(ok.sum())
            s["trunc"] += int((~ok).sum())
            pp = s["panels"].setdefault(it.get("panel", ""), [0.0, 0])
            pp[0] += float(d.sum())
            pp[1] += int(ok.sum())
        if (i + 1) % 20 == 0:
            logger.info("%d/%d continuations (%.0fs)", i + 1, len(items), time.time() - t0)
    out = {"cells": {}, "plain_nll": round(plain_tot / max(plain_n, 1), 5), "tokens": plain_n,
           "checkpoint": a.checkpoint, "set": a.set, "max_cont_tokens": a.max_cont_tokens}
    for cid, s in acc.items():
        n = max(s["n"], 1)
        out["cells"][cid] = {"damage": round(s["d"] / n, 5), "nll": round(s["nll"] / n, 5), "tokens": s["n"],
                             "trunc_share": round(s["trunc"] / max(s["n"] + s["trunc"], 1), 5),
                             "panels": {k: round(v[0] / max(v[1], 1), 5) for k, v in s["panels"].items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    logger.info("wrote %s (%d cells, %d tokens, %.0fs)", a.out, len(cells), plain_n, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
