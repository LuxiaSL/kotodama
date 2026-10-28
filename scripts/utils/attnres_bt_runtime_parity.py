"""The AttnRes kernels with the token count BT as a runtime argument, against the same kernels with BT a constexpr.

The constexpr revision's kernels/{phase_1,phase_2,configs}.py are loaded from a directory and swapped into the ops
modules, so both sides run the same autograd path. The two module copies autotune independently and can pick
different configs, which round differently, so every autotuned kernel is pinned on both sides to the config the
constexpr side picks for the shape. Gates:
  (1) the routed outputs and the block and partial gradients are bitwise equal, eager and under torch.compile,
      through the phase-1 v1 backward and both phase-2 backwards, at token counts that are 1, odd, not a multiple
      of 16, and a training shape; the query gradient, which is summed with tl.atomic_add in an order the scheduler
      sets, is within one bf16 ulp of its own scale;
  (2) eager forwards over distinct token counts stop compiling phase-1 forward kernels once the autotune configs and
      the three specializations Triton makes of an integer (1, a multiple of 16, anything else) exist: none new over
      the second half of the counts, where the constexpr kernels compile one per new count;
  (3) eager forward + backward time at the training shape, both sides on the same configs, interleaved rounds.

Run on a GPU host: tools/run_py.sh scripts/utils/attnres_bt_runtime_parity.py --old_kernels DIR [--depths 4,8]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="attnres-bt-triton-"))  # a fresh cache to count in

import torch  # noqa: E402
import triton  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from attnres_eps_compat_parity import load_old_kernels  # noqa: E402

from kotodama.model.flash_attn_res.kernels import phase_1 as new_k1, phase_2 as new_k2  # noqa: E402
from kotodama.model.flash_attn_res.ops import phase_1 as p1ops, phase_2 as p2ops  # noqa: E402

DEVICE = "cuda"
EPS = 1e-6
NAMES = ("out", "lse", "merged", "grad_blocks", "grad_query", "grad_partial")
ATOMIC = {"grad_query"}


def use(kernels) -> None:
    p1ops.phase_1, p2ops.phase_2 = kernels


def tuners(kernels) -> dict:
    return {(m.__name__.rsplit(".", 1)[-1], n): getattr(m, n) for m in kernels for n in dir(m)
            if isinstance(getattr(m, n), triton.runtime.autotuner.Autotuner)}


def route_and_grads(src, q, partial, weights):
    """Phase-1 route then the Phase-2 merge, a weighted sum of the outputs as the loss, and its gradients."""
    src, q, partial = (t.detach().requires_grad_() for t in (src, q, partial))
    out, lse = p1ops.phase_1_batched_attention_triton_op(src, q, EPS)
    merged = p2ops.phase_2_online_softmax_merge_triton_op(partial, q[0], out[0], lse[0], EPS)
    loss = (merged.float() * weights).sum() + (out.float() * weights).sum() + lse.float().sum()
    loss.backward()
    return out.detach(), lse.detach(), merged.detach(), src.grad, q.grad, partial.grad


def inputs(S: int, B: int, T: int, D: int, g: torch.Generator):
    src = torch.randn(S, B, T, D, device=DEVICE, generator=g).bfloat16()
    q = (torch.randn(1, D, device=DEVICE, generator=g) * 0.5).bfloat16()
    partial = torch.randn(B, T, D, device=DEVICE, generator=g).bfloat16()
    weights = torch.randn(B, T, D, device=DEVICE, generator=g)
    return src, q, partial, weights


class Pinned:
    """Autotunes the constexpr side on a shape, then pins every autotuned kernel on both sides to its picks."""

    def __init__(self, old, new):
        self.old, self.new = old, new
        self.ot, self.nt = tuners(old), tuners(new)
        self.full = {k: list(t.configs) for k, t in self.ot.items()}

    def pin(self, fn, args) -> None:
        for k in self.ot:
            self.ot[k].configs, self.nt[k].configs = list(self.full[k]), list(self.full[k])
            self.ot[k].cache.clear()
            self.nt[k].cache.clear()
        use(self.old)
        fn(*args)
        for k, t in self.ot.items():
            if getattr(t, "best_config", None) is not None and t.cache:
                self.ot[k].configs = self.nt[k].configs = [t.best_config]
                self.ot[k].cache.clear()
                self.nt[k].cache.clear()


def compare(a, b) -> tuple[bool, str]:
    ok, notes = True, []
    for name, x, y in zip(NAMES, a, b):
        if torch.equal(x, y):
            continue
        diff = (x.float() - y.float()).abs().max().item()
        if name in ATOMIC and diff <= 2.0 ** -7 * x.float().abs().max().item():
            notes.append(f"{name} within 1 bf16 ulp (max|Δ| {diff:.1e})")
            continue
        ok = False
        notes.append(f"{name} max|Δ| {diff:.1e}")
    return ok, "; ".join(notes) or "all bitwise"


def timed(fn, args, iters: int = 20) -> float:
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn(*args)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_kernels", type=Path, required=True)
    ap.add_argument("--depths", default="4,8")
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--mb", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--lengths", type=int, default=40, help="distinct eager token counts for gate (2)")
    a = ap.parse_args()
    old, new = load_old_kernels(a.old_kernels), (new_k1, new_k2)
    pinned = Pinned(old, new)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    ok = True

    for S in (int(x) for x in a.depths.split(",")):
        for B, T in [(1, 1), (1, 7), (2, 129), (3, 1000), (a.mb, a.seq)]:
            args = inputs(S, B, T, a.hidden, g)
            for p2_v2 in (False, True):
                p2ops._P2_BWD_V2 = p2_v2
                pinned.pin(route_and_grads, args)
                use(old)
                x = route_and_grads(*args)
                use(new)
                same, note = compare(x, route_and_grads(*args))
                ok &= same
                print(f"(1) eager S={S} B={B} T={T} phase-2 bwd {'v2' if p2_v2 else 'v1'}: {same} ({note})", flush=True)
        p2ops._P2_BWD_V2 = False
        args = inputs(S, a.mb, a.seq, a.hidden, g)
        pinned.pin(route_and_grads, args)
        results = []
        for kernels in (old, new):
            use(kernels)
            torch._dynamo.reset()
            results.append(torch.compile(route_and_grads)(*args))
        torch._dynamo.reset()
        same, note = compare(*results)
        ok &= same
        print(f"(1) compiled S={S} B={a.mb} T={a.seq}: {same} ({note})", flush=True)

    name = "phase_1_batched_attention_forward_kernel"
    root = Path(os.environ["TRITON_CACHE_DIR"])
    lengths = list(range(37, 37 + 53 * a.lengths, 53))  # distinct counts, odd and even, few multiples of 16
    growth = {}
    for label, kernels in (("constexpr BT", old), ("runtime BT", new)):
        use(kernels)
        marks = [sum(1 for _ in root.rglob(f"{name}.json"))]
        for half in (lengths[: len(lengths) // 2], lengths[len(lengths) // 2:]):
            for T in half:
                src, q, _, _ = inputs(4, 1, T, a.hidden, g)
                with torch.no_grad():
                    p1ops.phase_1_batched_attention_triton_op(src, q, EPS)
            marks.append(sum(1 for _ in root.rglob(f"{name}.json")))
        growth[label] = marks[2] - marks[1]
        print(f"(2) {label}: phase-1 forward kernels compiled over the first {len(lengths) // 2} distinct token counts "
              f"{marks[1] - marks[0]}, over the next {len(lengths) - len(lengths) // 2} {growth[label]}", flush=True)
    ok &= growth["runtime BT"] == 0 < growth["constexpr BT"]

    p2ops._P2_BWD_V2 = True
    for S in (int(x) for x in a.depths.split(",")):
        args = inputs(S, a.mb, a.seq, a.hidden, g)
        pinned.pin(route_and_grads, args)
        times = {"constexpr BT": [], "runtime BT": []}
        for r in range(7):
            order = (("constexpr BT", old), ("runtime BT", new))
            for label, kernels in order if r % 2 == 0 else order[::-1]:
                use(kernels)
                times[label].append(timed(route_and_grads, args))
        print(f"(3) S={S} B={a.mb} T={a.seq} eager fwd+bwd ms, median of 7 interleaved rounds: " +
              "  ".join(f"{k} {statistics.median(v):.3f}" for k, v in times.items()), flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
