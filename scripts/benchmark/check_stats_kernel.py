"""The fused attention-stats kernel (kotodama.model.attn_stats) on tsumugi, GPU: parity vs the torch reference, replay
bitwise, step cost.

    tools/run_py.sh scripts/benchmark/check_stats_kernel.py --out <json>

1. A signing CascadeDecoder (128 rows: 16 prompts of 24..400 tokens x 8 rows) with the fused stats decodes 32 steps,
   exports, re-admits and scores the same tokens: every surface bitwise (replay through the engine with the kernel).
2. On that decoder's KV state, random queries: kotodama.model.attn_stats.cascade_stats vs
   signing.stats_write(_cascade_probs(...)) at a sampled and a non-sampled layer: max |diff| per statistic.
3. Step time at 128 rows: unsigned, signed with reference stats, signed with fused stats.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

LAYERS = (0, 18, 25, 34, 36, 45, 54, 56, 63, 71)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="hf:LuxiaSL/kotodama-tsumugi-1b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=32)
    a = ap.parse_args()
    from kotodama.model import attn_stats, signing
    from kotodama.model.decode_engine import DecodeEngine, SamplingParams
    from kotodama.model.multi_decode import CascadeDecoder, _cascade_probs
    from kotodama.model.signing import SignSpec
    from kotodama.serve import server

    torch._dynamo.config.recompile_limit = 64
    server._rope_fp32_override = True
    model, _ = server.load_model(a.checkpoint, "cuda", mode="base", model_size="1b-tsumugi", engine="reference",
                                 max_seq_len=4096)
    eng = DecodeEngine(model, max_seq_len=4096, levers=4)
    eng.compile_step(mode="reduce-overhead", regional=True)
    rng = random.Random(7)
    lens = [rng.randint(24, 400) for _ in range(16)]
    prompts = [torch.randint(3, model.config.vocab_size, (n,), generator=torch.Generator().manual_seed(n)).tolist()
               for n in lens]
    width = max(lens) + a.steps + 8
    law = SamplingParams(temperature=0.8, repetition_penalty=1.2)
    out: dict = {}

    def build(sign: bool, fused: bool):
        spec = SignSpec(LAYERS, tail=a.steps, depth=True, attn=True, width=width) if sign else None
        dec = CascadeDecoder(eng, 128, groups=20, group_rows=32, tail_tokens=a.steps + 8, seeded=True, sign=spec)
        if not fused:
            dec._stats_fn = None
        dec.compile_step(mode="reduce-overhead", regional=True)
        return dec

    def admit(dec, seeds):
        dec.release(range(128))
        first = []
        for p, prompt in enumerate(prompts):
            rows = list(range(8 * p, 8 * p + 8))
            first += dec.admit(rows, eng.prefill(torch.tensor([prompt], device="cuda")), law,
                               seeds=[seeds + r for r in rows])
        return first

    def step_ms(dec, n=40):
        admit(dec, 0)
        for _ in range(8):
            dec.step(law)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            dec.step(law)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000 / n

    with torch.inference_mode():
        fused = build(True, True)
        out["fused_available"] = fused._stats_fn is not None
        # 1. replay through the engine, kernel on
        first = admit(fused, 500)
        toks = [fused.step(law).view(-1).clone() for _ in range(a.steps)]
        gen = [fused.signed(r, device=torch.device("cuda")) for r in range(128)]
        seq = torch.cat([torch.tensor(first, device="cuda").view(-1, 1), torch.stack(toks, dim=1)], dim=1)
        admit(fused, 9)
        fused.score(seq)
        rep = [fused.signed(r, device=torch.device("cuda")) for r in range(128)]
        diff = {}
        for name, x in gen[0].surfaces.items():
            if isinstance(x, torch.Tensor):
                diff[name] = sum(int((g.surfaces[name] != r.surfaces[name]).sum()) for g, r in zip(gen, rep))
        out["replay_bitwise"] = all(v == 0 for v in diff.values())
        out["replay_differing"] = {k: v for k, v in diff.items() if v}
        # 2. parity on the decoder's KV state (after the scoring pass), random queries
        _, kvs, state = fused._regional_attend()
        H, D = eng.n_heads, eng.head_dim
        par = {}
        for li, sampled in ((36, True), (40, False)):
            q = (torch.randn((128, H, 1, D), device="cuda") * 2).to(eng.dtype)
            bufs = {}
            for tag in ("fused", "ref"):
                st = list(fused._stats_args(li))
                st[:4] = [b.clone().zero_() for b in st[:4]]
                if tag == "fused":
                    attn_stats.cascade_stats(tuple(st), q, kvs[li], state, sampled)
                else:
                    signing.stats_write(tuple(st), _cascade_probs(q, kvs[li], state), sampled)
                bufs[tag] = st[:4]
            names = ("head_entropy", "mean_entropy", "head_scalars", "mean_rows")
            res = {}
            for nm, f, r in zip(names, bufs["fused"], bufs["ref"]):
                d = (f.double() - r.double()).abs()
                res[nm] = {"max_abs": float(d.max()), "ref_max": float(r.double().abs().max())}
            par[f"layer{li}{'_sampled' if sampled else ''}"] = res
        out["parity"] = par
        # 3. step cost at 128 rows
        out["step_ms_fused"] = step_ms(fused)
        del fused
        torch.cuda.empty_cache()
        ref = build(True, False)
        out["step_ms_reference"] = step_ms(ref)
        del ref
        torch.cuda.empty_cache()
        off = build(False, False)
        out["step_ms_unsigned"] = step_ms(off)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
