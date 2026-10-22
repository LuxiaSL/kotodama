"""Batched AttnRes Phase 1 (KOTODAMA_BATCH_P1) vs the per-sublayer path, end to end at a training shape.

The per-sublayer path routes every sublayer with its own Phase 1 call over the committed block stack; the batched
path runs Phase 1 once per block for all of the block's queries (chunks of _P1_MAX_BATCH) and Phase 2 per
sublayer. Gate: on the full model (default 1b-moonshot-loop, its fitted boundaries, bf16, doc-masked fa4 varlen
batch), loss and every parameter gradient of the batched path differ from the per-sublayer path by no more than
the per-sublayer path's own run-to-run floor (two identical runs), with a bf16 reassociation allowance (loss 1e-4,
relative grad norm 1e-3) when that floor is zero — reported per tensor family. Routing queries
are randomized (they initialize to zero, which would make every route uniform and the check vacuous). Also
times eager fwd+bwd for both.

Run on a GPU host: tools/run_py.sh scripts/utils/batch_p1_parity.py [--model_size 1b-moonshot-loop] [--mb 2]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama import presets  # noqa: E402
from kotodama.model import llama  # noqa: E402
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig  # noqa: E402

DEVICE = "cuda"
T = 4096


def build(model_size: str, seed: int) -> KotodamaModel:
    kw = presets.shape(model_size)
    kw.update(qk_norm=True, z_loss_weight=1e-5, use_liger=True, rope_fp32=True, attn_impl="fa4", attn_res=True,
              attn_res_boundaries=list(presets.DEFAULT_BOUNDARIES[model_size]))
    torch.manual_seed(seed)
    m = KotodamaModel(KotodamaModelConfig(**kw)).to(device=DEVICE, dtype=torch.bfloat16)
    g = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("res_query"):
                p.copy_(torch.randn(p.shape, device=DEVICE, generator=g) * 0.5)
    m.train()
    return m


def batch(mb: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 49152, (mb, T), generator=g)
    lens: list[int] = []
    pos = torch.zeros(mb, T, dtype=torch.long)
    for r in range(mb):
        off = 0
        while off < T:
            n = min(T - off, int(torch.randint(64, 2500, (1,), generator=g)))
            pos[r, off:off + n] = torch.arange(n)
            lens.append(n)
            off += n
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    return ids.to(DEVICE), cu.to(DEVICE), pos.to(DEVICE), max(lens)


def run(m: KotodamaModel, data, batched: bool) -> tuple[float, dict[str, torch.Tensor], float]:
    llama._BATCH_PHASE1 = batched
    ids, cu, pos, mx = data
    m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(ids, labels=ids, cu_seqlens=cu, max_seqlen=mx, position_ids=pos)["loss"]
    loss.backward()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return loss.item(), {n: p.grad.detach().float().clone() for n, p in m.named_parameters() if p.grad is not None}, dt


def family(name: str) -> str:
    parts = name.split(".")
    return ".".join(p for p in parts if not p.isdigit())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_size", default="1b-moonshot-loop")
    ap.add_argument("--mb", type=int, default=2)
    a = ap.parse_args()
    if not (llama._TRITON_ATTN_RES_AVAILABLE and torch.cuda.is_available()):
        print("needs CUDA + the Triton AttnRes kernels")
        return 2
    t0 = time.perf_counter()

    def note(msg: str) -> None:
        print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)

    m = build(a.model_size, seed=17)
    data = batch(a.mb, seed=42)
    note(f"built {a.model_size}, {data[1].numel() - 1} docs")
    for b in (False, True):  # warm-up: Triton JIT/autotune + FA4 compile
        _, _, dt = run(m, data, b)
        note(f"warm-up {'batched' if b else 'per-sublayer'} fwd+bwd {dt:.1f}s")
    l_a, g_a, t_a = run(m, data, False)
    note("per-sublayer run 1")
    l_b, g_b, t_b = run(m, data, False)
    note("per-sublayer run 2")
    l_p, g_p, t_p = run(m, data, True)
    note("batched run")
    print(f"{a.model_size} mb={a.mb} T={T}: loss per-sublayer {l_a:.6f} / rerun {l_b:.6f} / batched {l_p:.6f}")
    print(f"eager fwd+bwd s: per-sublayer {t_a:.3f} {t_b:.3f}  batched {t_p:.3f}  ({(t_p / min(t_a, t_b) - 1) * 100:+.1f}%)")
    floor: dict[str, float] = defaultdict(float)
    diff: dict[str, float] = defaultdict(float)
    for n in g_a:
        f = family(n)
        scale = g_a[n].norm().clamp_min(1e-30)
        floor[f] = max(floor[f], ((g_a[n] - g_b[n]).norm() / scale).item())
        diff[f] = max(diff[f], ((g_a[n] - g_p[n]).norm() / scale).item())
    ok = abs(l_p - l_a) <= max(abs(l_b - l_a), 1e-4)
    print(f"loss |batched - per-sublayer| {abs(l_p - l_a):.2e}  floor {abs(l_b - l_a):.2e}  {'OK' if ok else 'ABOVE'}")
    print(f"{'family':48s} {'rerun floor':>12s} {'batched':>12s}")
    for f in sorted(diff):
        within = diff[f] <= max(2.0 * floor[f], 1e-3)
        ok &= within
        print(f"{f:48s} {floor[f]:12.2e} {diff[f]:12.2e}  {'OK' if within else 'ABOVE'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
