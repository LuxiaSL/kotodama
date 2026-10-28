#!/usr/bin/env python3
"""The act lever on a real checkpoint (GPU) against reference logits from an eager forward with the same rewrite.

Reads a neuron-set file (``{"sets": {name: [{"L": effective layer, "n": neuron, "mu": value}, ...]}}``) and a
directory of reference files, one per (prompt, set, alpha), each a torch file with ``ids`` (LongTensor [T]),
``prompt_len`` P, ``logits`` (float32 [C, V]: the next-token logits at positions P-1 .. T-2), ``set`` (a set name,
or None for the unsteered reference), ``alpha`` and ``span`` ("uniform": every position rewritten). For each, the
engine (compiled as the server compiles it, regional by default) prefills the prompt under the set's act lever and
steps the forced continuation; reported per file: KL(reference || engine) mean and max and top-1 agreement over
the C positions, beside KL(unsteered reference || reference), the lever's own effect, as the scale a parity
tolerance sits under.

Usage (node, GPU):
  tools/run_py.sh scripts/benchmark/check_act_lever.py --checkpoint <model.pt> --model_size 1b-tsumugi --rope_fp32 \\
      --sets <sets.json> --refs <dir> --out <dir>/check_act_lever.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch._dynamo
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model.decode_engine import DecodeEngine  # noqa: E402
from kotodama.model.levers import LeverPlan, act_rows  # noqa: E402
from kotodama.serve import server  # noqa: E402

logger = logging.getLogger("check_act_lever")


def kl_rows(ref_logits: torch.Tensor, got_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row KL(ref || got) and argmax agreement, from logits (C, V)."""
    lp_r, lp_g = F.log_softmax(ref_logits.float(), -1), F.log_softmax(got_logits.float(), -1)
    return (lp_r.exp() * (lp_r - lp_g)).sum(-1), (lp_r.argmax(-1) == lp_g.argmax(-1)).float()


@torch.inference_mode()
def engine_logits(eng: DecodeEngine, ids: torch.Tensor, prompt_len: int, slot: int) -> torch.Tensor:
    """Next-token logits at positions prompt_len-1 .. T-2: the prefill's last row, then each forced step's."""
    rows = [eng.prefill(ids[None, :prompt_len], slot=slot).view(-1).float().clone()]
    for t in range(prompt_len, ids.shape[0] - 1):
        rows.append(eng.step_logits(ids[None, t:t + 1].contiguous()).view(-1).float().clone())
    return torch.stack(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model_size", default="1b-tsumugi")
    ap.add_argument("--rope_fp32", action="store_true")
    ap.add_argument("--sets", required=True, help="the neuron-set file")
    ap.add_argument("--refs", required=True, help="directory of reference .pt files")
    ap.add_argument("--mode", default="max-autotune", help="the engine's compile mode")
    ap.add_argument("--no-regional", dest="regional", action="store_false")
    ap.add_argument("--eager", action="store_true", help="the engine uncompiled")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    torch._dynamo.config.recompile_limit = 64
    sets = json.loads(Path(a.sets).read_text())["sets"]
    refs = sorted(Path(a.refs).glob("*.pt"))
    if not refs:
        raise SystemExit(f"no reference files under {a.refs}")
    server._rope_fp32_override = a.rope_fp32
    model, _tok = server.load_model(a.checkpoint, "cuda", mode="base", model_size=a.model_size, engine="reference",
                                    max_seq_len=a.max_seq_len)
    cfg = model.config
    eng = DecodeEngine(model, max_seq_len=a.max_seq_len, levers=4)
    if not a.eager:
        eng.compile_step(mode=a.mode, regional=a.regional)
    loaded = {p.stem: torch.load(p, map_location="cpu") for p in refs}
    base = {}
    for name, ref in loaded.items():
        if ref.get("set") in (None, "base") or float(ref.get("alpha", 0.0)) == 0.0:
            base[name.split("__")[0]] = ref["logits"]
    res: dict = {"checkpoint": a.checkpoint, "compiled": not a.eager, "regional": a.regional and not a.eager,
                 "mode": a.mode, "files": {}}
    for name, ref in loaded.items():
        if ref.get("span", "uniform") != "uniform":
            raise SystemExit(f"{name}: only uniform-span references are checked here")
        ids = ref["ids"].to("cuda")
        set_name, alpha = ref.get("set"), float(ref.get("alpha", 0.0))
        slot = 0
        if set_name not in (None, "base") and alpha != 0.0:
            entries = [(e["L"], e["n"], e["mu"]) for e in sets[set_name]]
            slot = eng.levers.bind(LeverPlan(act=act_rows(entries, alpha, cfg.intermediate_size), span="uniform"))
        got = engine_logits(eng, ids, int(ref["prompt_len"]), slot)
        if slot:
            eng.levers.release(slot)
        kl, top1 = kl_rows(ref["logits"].to("cuda"), got)
        row = {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "top1": float(top1.mean()),
               "positions": int(kl.numel())}
        prompt_key = name.split("__")[0]
        if slot and prompt_key in base:
            eff, _ = kl_rows(base[prompt_key], ref["logits"])
            row["effect_kl_mean"], row["effect_kl_max"] = float(eff.mean()), float(eff.max())
        res["files"][name] = row
        logger.info("%s: %s", name, json.dumps(row))
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
