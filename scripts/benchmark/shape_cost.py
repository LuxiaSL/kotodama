"""Forward+backward cost of a model shape on one GPU, for pricing candidate shapes (width, depth, AttnRes blocks, loop).

Builds KotodamaModel directly with the training stack's settings (FA4 varlen inputs, Liger kernels + fused linear CE,
QK-norm, fp32 RoPE, whole-model torch.compile, bf16 weights), feeds random tokens, and times fwd+bwd of one
micro-batch (no optimizer step; that cost is O(params) and modelled separately). One JSON line per run on stdout.

Pairs of runs price AttnRes routing: the same shape with --n_blocks N and with --n_blocks 0 (no AttnRes) differ by
exactly the routing work, whose per-layer share depends on N and the width, not on the depth.

  tools/run_py.sh scripts/benchmark/shape_cost.py --hidden 3072 --heads 24 --kv 8 --ffn 8192 --layers 48 --n_blocks 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kotodama.model import llama  # noqa: E402
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig  # noqa: E402

DEVICE = "cuda"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--kv", type=int, required=True)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--ffn", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True,
                    help="effective depth (num_layers; a loop reuses weights inside it, see KotodamaModelConfig)")
    ap.add_argument("--n_blocks", type=int, default=0, help="AttnRes blocks over the effective depth (0 = no AttnRes)")
    ap.add_argument("--loop_start", type=int, default=None)
    ap.add_argument("--loop_len", type=int, default=0)
    ap.add_argument("--loop_times", type=int, default=1)
    ap.add_argument("--mb", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--chain", action="store_true", help="--attnres_stack_grad_chain")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    if a.chain:
        llama._STACK_GRAD_CHAIN = True
    cfg = KotodamaModelConfig(
        hidden_size=a.hidden, num_layers=a.layers, num_attention_heads=a.heads, num_kv_heads=a.kv,
        head_dim=a.head_dim, intermediate_size=a.ffn, max_position_embeddings=a.seq, qk_norm=True,
        use_liger=True, rope_fp32=True, attn_impl="fa4",
        attn_res=a.n_blocks > 0, attn_res_n_blocks=max(a.n_blocks, 1),
        loop_start=a.loop_start, loop_len=a.loop_len, loop_times=a.loop_times,
    )
    torch.manual_seed(0)
    model = KotodamaModel(cfg).to(device=DEVICE, dtype=torch.bfloat16)
    model.train()
    compiled = torch.compile(model)
    B, T = a.mb, a.seq
    g = torch.Generator(device=DEVICE).manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=DEVICE, generator=g)
    position_ids = torch.arange(T, device=DEVICE).expand(B, T)
    cu_seqlens = (torch.arange(B + 1, device=DEVICE) * T).to(torch.int32)

    def step() -> None:
        out = compiled(ids, labels=ids, position_ids=position_ids, cu_seqlens=cu_seqlens, max_seqlen=T)
        out["loss"].backward()

    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(a.iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        step()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    ms = times[len(times) // 2]
    print(json.dumps({
        "tag": a.tag, "hidden": a.hidden, "heads": a.heads, "kv": a.kv, "ffn": a.ffn, "eff_layers": cfg.num_layers,
        "unique_layers": cfg.num_unique_layers(), "n_blocks": a.n_blocks,
        "loop": [a.loop_start, a.loop_len, a.loop_times],
        "chain": a.chain, "mb": B, "seq": T, "params": sum(p.numel() for p in model.parameters()),
        "fwd_bwd_ms": ms, "us_per_token": ms * 1e3 / (B * T), "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "compile_s": compile_s,
    }), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
