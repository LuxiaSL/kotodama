"""Signing does not change generation, and levers replay bitwise (tsumugi, GPU, compiled regional step with emulated
casts, as a server's --emulate_casts compiles it, and one kernel pin table recorded for the process, as a server's or
a drain's --kernel_pins / --pins gives it: without one, two decoders of the same kind may already autotune to
different kernels; without emulated casts, a signing layer's extra stores move fusion boundaries and with them the
rounding of its arithmetic).

    tools/run_py.sh scripts/benchmark/check_signed_generation.py --out <json>

1. An unsigned and a signing CascadeDecoder (128 rows: 16 prompts of 24..400 tokens x 8 rows, seeded, a decode-span
   attention lever on half the rows, top-5 readouts) decode the same admissions under one law: the tokens, step
   logprobs and top-5 logprobs of every row, bit for bit, at every step.
2. The signing decoder's rows (levers included) re-admitted and scored on their own tokens: every signed surface bit
   for bit as generation signed it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

LAYERS = (0, 18, 25, 34, 36, 45, 54, 56, 63, 71)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="hf:LuxiaSL/kotodama-tsumugi-1b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min_p", type=float, default=0.02)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    a = ap.parse_args()
    from kotodama.model import kernel_pins
    from kotodama.model.decode_engine import DecodeEngine, SamplingParams
    from kotodama.model.levers import LeverPlan
    from kotodama.model.multi_decode import CascadeDecoder
    from kotodama.model.signing import SignSpec
    from kotodama.serve import server

    pins = kernel_pins.install(Path(a.out).with_suffix(".pins.json"), "record")
    torch._dynamo.config.recompile_limit = 64
    server._rope_fp32_override = True
    model, _ = server.load_model(a.checkpoint, "cuda", mode="base", model_size="1b-tsumugi", engine="reference",
                                 max_seq_len=4096)
    eng = DecodeEngine(model, max_seq_len=4096, levers=4, emulate_casts=True)
    eng.compile_step(mode="reduce-overhead", regional=True)
    hidden = model.config.hidden_size
    vec = np.random.default_rng(3).standard_normal(hidden).astype(np.float32)
    lever = eng.levers.bind(LeverPlan(attn={36: 0.05 * vec / np.linalg.norm(vec) * np.sqrt(hidden)}, span="decode"))
    rng = random.Random(7)
    lens = [rng.randint(24, 400) for _ in range(16)]
    prompts = [torch.randint(3, model.config.vocab_size, (n,), generator=torch.Generator().manual_seed(n)).tolist()
               for n in lens]
    width = max(lens) + a.steps + 8
    law = SamplingParams(temperature=a.temperature, repetition_penalty=a.repetition_penalty, min_p=a.min_p)
    slots = [lever if r % 2 else 0 for r in range(128)]
    spec = SignSpec(LAYERS, tail=a.steps, depth=True, attn=True, width=width)
    out: dict = {"law": {"temperature": a.temperature, "min_p": a.min_p, "repetition_penalty": a.repetition_penalty},
                 "lever_rows": sum(1 for s in slots if s)}

    def build(sign):
        dec = CascadeDecoder(eng, 128, groups=20, group_rows=32, tail_tokens=a.steps + 8, seeded=True, sign=sign,
                             readout_k=5)
        dec.compile_step(mode="reduce-overhead", regional=True)
        return dec

    def admit(dec, seed0):
        dec.release(range(128))
        first = []
        for p, prompt in enumerate(prompts):
            rows = list(range(8 * p, 8 * p + 8))
            first += dec.admit(rows, eng.prefill(torch.tensor([prompt], device="cuda")), law,
                               slots=[slots[r] for r in rows], seeds=[seed0 + r for r in rows])
        return first

    with torch.inference_mode():
        off, on = build(None), build(spec)
        f_off, f_on = admit(off, 1000), admit(on, 1000)
        same_first = f_off == f_on
        toks, step_diff, lp_diff = [], 0, 0
        for _ in range(a.steps):
            t_off = off.step(law).view(-1).clone()
            t_on = on.step(law).view(-1).clone()
            step_diff += int((t_off != t_on).sum())
            lp_diff += int((off.step_logprob != on.step_logprob).sum())
            lp_diff += int((off.step_top_lps != on.step_top_lps).sum())
            toks.append(t_on)
        out["signing_keeps_generation"] = {"first_tokens_equal": same_first, "token_mismatches": step_diff,
                                           "logprob_mismatches": lp_diff, "steps": a.steps}
        del off
        torch.cuda.empty_cache()
        gen = [on.signed(r, device=torch.device("cuda")) for r in range(128)]
        seq = torch.cat([torch.tensor(f_on, device="cuda").view(-1, 1), torch.stack(toks, dim=1)], dim=1)
        admit(on, 5)
        on.score(seq)
        rep = [on.signed(r, device=torch.device("cuda")) for r in range(128)]
        diff = {}
        for name, x in gen[0].surfaces.items():
            if isinstance(x, torch.Tensor):
                diff[name] = {"lever_rows": sum(int((gen[r].surfaces[name] != rep[r].surfaces[name]).sum())
                                                for r in range(128) if slots[r]),
                              "plain_rows": sum(int((gen[r].surfaces[name] != rep[r].surfaces[name]).sum())
                                                for r in range(128) if not slots[r])}
        out["replay_bitwise"] = all(v["lever_rows"] == 0 and v["plain_rows"] == 0 for v in diff.values())
        out["replay_differing"] = {k: v for k, v in diff.items() if v["lever_rows"] or v["plain_rows"]}
        out["lever_slots_recorded"] = all(rep[r].lever_slot == slots[r] for r in range(128))
    out["kernel_pins"] = {"path": str(pins.save()), "kernels": len(pins.table)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
