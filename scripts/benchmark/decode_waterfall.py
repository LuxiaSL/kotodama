#!/usr/bin/env python3
"""Decode cost waterfall: where does each millisecond of a decode step go?

Measures the serving decode loop the way serve.py runs it (eager, KV cat,
CPU-in-loop) and decomposes per-token cost into:

  1. Wall-clock per token vs context length, two ways:
       - "synced":  torch.cuda.synchronize per step (serve.py /generate style)
       - "free":    no per-step sync, token stays on GPU (launch-bound floor)
  2. GPU-busy fraction: total kernel time vs unprofiled wall time
  3. Kernel buckets: GEMM/attention/routing/cat-copy/norm-elementwise/other,
     plus launch counts per decode step
  4. Prefill latency at several prompt lengths
  5. Sampling-side microbench (serve.py sample_next_token cost vs seq len)

Usage (gpu-host, scratch GPU):
    source ~/workspace/.venv-shared/bin/activate
    cd ~/workspace/kotodama

    # Eager routing (selfsim-server parity)
    KOTODAMA_NO_TRITON_ATTNRES=1 CUDA_VISIBLE_DEVICES=3 \
        python scripts/benchmark/decode_waterfall.py \
        --checkpoint /models/kotodama-data/sft-selfsim/selfsim3b-lr1e-03-ep2/checkpoints/step_00000358.pt \
        --output outputs/profiles/waterfall_eager_routing.json

    # Triton routing (gateway-fleet parity)
    CUDA_VISIBLE_DEVICES=3 python scripts/benchmark/decode_waterfall.py \
        --checkpoint ... --output outputs/profiles/waterfall_triton_routing.json

    # Fast iteration without checkpoint load (timing-identical, garbage logits)
    CUDA_VISIBLE_DEVICES=3 python scripts/benchmark/decode_waterfall.py \
        --random-weights --output /tmp/waterfall_random.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model import llama as llama_mod
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

DDV1_BOUNDARIES = [0, 3, 7, 12, 21, 25]
DD3B_BOUNDARIES = [0, 1, 3, 7, 15, 19, 24]

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "proxy": dict(
        hidden_size=512, num_layers=28, num_attention_heads=4, num_kv_heads=2,
        head_dim=128, intermediate_size=1408, vocab_size=49152,
        max_position_embeddings=4096, rope_theta=500000.0, norm_eps=1e-5,
        qk_norm=True, tie_word_embeddings=True, z_loss_weight=0.0,
        use_liger=False, attn_impl="sdpa", attn_res=True,
        attn_res_boundaries=DDV1_BOUNDARIES,
    ),
    "3b": dict(
        hidden_size=3072, num_layers=28, num_attention_heads=24, num_kv_heads=8,
        head_dim=128, intermediate_size=8192, vocab_size=49152,
        max_position_embeddings=4096, rope_theta=500000.0, norm_eps=1e-5,
        qk_norm=True, tie_word_embeddings=True, z_loss_weight=0.0,
        use_liger=False, attn_impl="sdpa", attn_res=True,
        attn_res_boundaries=DD3B_BOUNDARIES,
    ),
}

# Kernel-name → bucket. First match wins; names are lowercased before matching.
# attnres routing MUST precede attention (phase kernels contain "attention" in name).
BUCKET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("attnres_routing_triton", re.compile(r"phase_1|phase_2|phase1|phase2")),
    ("gemm", re.compile(r"gemm|gemv|cutlass|nvjet|matmul|aten::mm|aten::addmm|aten::linear|splitk")),
    ("attention_sdpa", re.compile(r"fmha|flash|attention|sdpa|aten::scaled_dot_product")),
    ("cat_copy", re.compile(r"aten::cat|catarraybatched|aten::copy_|copy_kernel|aten::contiguous|aten::clone|direct_copy|cudamemcpy")),
    ("softmax", re.compile(r"softmax")),
    ("norm_elementwise", re.compile(r"elementwise|vectorized|rsqrt|aten::mul|aten::add|aten::silu|aten::neg|aten::pow|aten::mean|aten::rsqrt|reduce_kernel|unrolled")),
    ("embedding_index", re.compile(r"embedding|index|gather|scatter")),
    ("multinomial_sample", re.compile(r"multinomial|distribution|argmax|topk|sort")),
]


@dataclass
class DecodeTiming:
    context_len: int
    n_tokens: int
    synced_ms_per_token: float
    free_ms_per_token: float
    synced_tok_per_s: float
    free_tok_per_s: float


@dataclass
class ProfileSummary:
    context_len: int
    n_steps: int
    wall_ms_per_step_unprofiled: float
    gpu_kernel_ms_per_step: float
    gpu_busy_fraction: float
    kernel_launches_per_step: float
    cuda_api_calls_per_step: float
    buckets_ms_per_step: dict[str, float] = field(default_factory=dict)
    buckets_pct_of_kernel_time: dict[str, float] = field(default_factory=dict)
    buckets_launches_per_step: dict[str, float] = field(default_factory=dict)
    top_kernels: list[dict[str, Any]] = field(default_factory=list)


def build_model(args: argparse.Namespace, device: torch.device) -> LuxiaBaseModel:
    config = LuxiaModelConfig(**MODEL_CONFIGS[args.model_size])
    model = LuxiaBaseModel(config)

    if args.random_weights:
        logger.warning("Using RANDOM weights — timings valid, logits garbage")
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        logger.info("Loading checkpoint: %s", ckpt_path)
        t0 = time.time()
        if ckpt_path.suffix == ".zst":
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            with open(ckpt_path, "rb") as f_in:
                decompressed = dctx.decompress(f_in.read())
            ckpt = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
            del decompressed
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=True)
        logger.info("Checkpoint loaded in %.1fs (step %s)", time.time() - t0, ckpt.get("step", "?"))

    return model.to(device).eval().bfloat16()


def make_prompt_ids(n: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(1234)
    # Avoid special tokens 0-3 so we don't trip anything odd
    return torch.randint(4, vocab_size, (1, n), generator=g).to(device)


@torch.inference_mode()
def prefill(model: LuxiaBaseModel, input_ids: torch.Tensor) -> tuple[torch.Tensor, list]:
    out = model(input_ids, use_cache=True)
    return out["logits"][:, -1:].argmax(dim=-1), out["past_kv"]


@torch.inference_mode()
def timed_prefill(model: LuxiaBaseModel, input_ids: torch.Tensor, device: torch.device, n_repeats: int = 3) -> float:
    torch.cuda.synchronize(device)
    best = float("inf")
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model(input_ids, use_cache=True)
        torch.cuda.synchronize(device)
        best = min(best, (time.perf_counter() - t0) * 1000)
    return best


@torch.inference_mode()
def decode_synced(model: LuxiaBaseModel, tok: torch.Tensor, past_kv: list, n_tokens: int, device: torch.device) -> float:
    """serve.py /generate style: synchronize around every forward, .item() per step."""
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        torch.cuda.synchronize(device)
        out = model(tok, use_cache=True, past_kv=past_kv)
        torch.cuda.synchronize(device)
        past_kv = out["past_kv"]
        token_id = out["logits"][0, -1].argmax().item()  # host sync, like sampling does
        tok = torch.tensor([[token_id]], device=device)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000 / n_tokens


@torch.inference_mode()
def decode_free(model: LuxiaBaseModel, tok: torch.Tensor, past_kv: list, n_tokens: int, device: torch.device) -> float:
    """No CPU in the loop: token stays on GPU, single sync at the end.

    This is the floor for the CURRENT eager kernels — pure launch+kernel cost.
    """
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        out = model(tok, use_cache=True, past_kv=past_kv)
        past_kv = out["past_kv"]
        tok = out["logits"][:, -1:].argmax(dim=-1)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000 / n_tokens


@torch.inference_mode()
def grow_cache_to(model: LuxiaBaseModel, target_len: int, vocab: int, device: torch.device) -> tuple[torch.Tensor, list]:
    """Prefill target_len tokens; return (next_token, past_kv) at that context.

    A prefilled cache is structurally identical to a decode-grown one (both are
    contiguous post-cat tensors), so prefilling directly is equivalent and fast.
    """
    ids = make_prompt_ids(target_len, vocab, device)
    tok, past_kv = prefill(model, ids)
    torch.cuda.synchronize(device)
    return tok, past_kv


def _evt_self_device_us(evt: Any) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def bucket_name(name: str) -> str:
    lname = name.lower()
    for bucket, pat in BUCKET_PATTERNS:
        if pat.search(lname):
            return bucket
    return "other"


@torch.inference_mode()
def profile_decode_window(
    model: LuxiaBaseModel,
    tok: torch.Tensor,
    past_kv: list,
    n_steps: int,
    device: torch.device,
    wall_ms_unprofiled: float,
    trace_path: Optional[Path] = None,
) -> ProfileSummary:
    """Profile n_steps of free-running decode; attribute kernel time to buckets."""
    ctx_len = past_kv[0][0].shape[2]

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for step in range(n_steps):
            with record_function(f"DECODE_STEP_{step}"):
                out = model(tok, use_cache=True, past_kv=past_kv)
                past_kv = out["past_kv"]
                tok = out["logits"][:, -1:].argmax(dim=-1)
        torch.cuda.synchronize(device)

    # Aggregate: self device time over all rows = total GPU kernel time (no double count)
    buckets_us: dict[str, float] = {}
    buckets_n: dict[str, int] = {}
    total_kernel_us = 0.0
    total_launches = 0
    cuda_api_calls = 0
    top: list[tuple[str, float, int]] = []

    try:
        cuda_device_type = torch.autograd.DeviceType.CUDA
    except AttributeError:
        cuda_device_type = None

    for evt in prof.key_averages():
        name = str(evt.key)
        if name.startswith("DECODE_STEP"):
            continue
        if name.startswith("cuda") and "Launch" in name:
            cuda_api_calls += evt.count
        # Only count kernel-level events (device_type CUDA). Op-level rows
        # (aten::mm etc., device_type CPU) attribute the SAME device time as
        # the kernels they launched — summing both double-counts.
        if cuda_device_type is not None and evt.device_type != cuda_device_type:
            continue
        dev_us = _evt_self_device_us(evt)
        if dev_us <= 0:
            continue
        total_kernel_us += dev_us
        total_launches += evt.count
        b = bucket_name(name)
        buckets_us[b] = buckets_us.get(b, 0.0) + dev_us
        buckets_n[b] = buckets_n.get(b, 0) + evt.count
        top.append((name, dev_us, evt.count))

    top.sort(key=lambda t: -t[1])
    top_kernels = [
        {"name": n[:120], "us_per_step": round(us / n_steps, 1), "calls_per_step": round(c / n_steps, 1)}
        for n, us, c in top[:25]
    ]

    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
        logger.info("Chrome trace: %s", trace_path)

    kernel_ms_per_step = total_kernel_us / 1000 / n_steps
    return ProfileSummary(
        context_len=int(ctx_len),
        n_steps=n_steps,
        wall_ms_per_step_unprofiled=round(wall_ms_unprofiled, 3),
        gpu_kernel_ms_per_step=round(kernel_ms_per_step, 3),
        gpu_busy_fraction=round(kernel_ms_per_step / wall_ms_unprofiled, 4) if wall_ms_unprofiled > 0 else 0.0,
        kernel_launches_per_step=round(total_launches / n_steps, 1),
        cuda_api_calls_per_step=round(cuda_api_calls / n_steps, 1),
        buckets_ms_per_step={k: round(v / 1000 / n_steps, 3) for k, v in sorted(buckets_us.items(), key=lambda kv: -kv[1])},
        buckets_pct_of_kernel_time={k: round(100 * v / total_kernel_us, 1) for k, v in sorted(buckets_us.items(), key=lambda kv: -kv[1])} if total_kernel_us > 0 else {},
        buckets_launches_per_step={k: round(v / n_steps, 1) for k, v in sorted(buckets_n.items(), key=lambda kv: -kv[1])},
        top_kernels=top_kernels,
    )


def sample_next_token_cost(vocab_size: int, device: torch.device) -> dict[str, float]:
    """Microbench serve.py's sample_next_token at several generated-ids lengths."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from serve import sample_next_token  # noqa: PLC0415

    logits = torch.randn(vocab_size, device=device, dtype=torch.bfloat16)
    results: dict[str, float] = {}
    for n_gen in [8, 128, 512, 1024]:
        gen_ids = list(range(100, 100 + n_gen))
        # warm
        for _ in range(3):
            sample_next_token(logits, 0.9, 0, 0.0, 1.2, gen_ids)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        n_iter = 20
        for _ in range(n_iter):
            sample_next_token(logits, 0.9, 0, 0.0, 1.2, gen_ids)
        torch.cuda.synchronize(device)
        results[f"gen_len_{n_gen}_ms"] = round((time.perf_counter() - t0) * 1000 / n_iter, 3)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode cost waterfall profiler")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str, help="Path to .pt/.pt.zst checkpoint")
    src.add_argument("--random-weights", action="store_true", help="Skip checkpoint (timing-only)")
    parser.add_argument("--model-size", choices=list(MODEL_CONFIGS), default="3b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context-lens", type=int, nargs="+", default=[128, 512, 1024, 2048])
    parser.add_argument("--decode-tokens", type=int, default=32, help="Tokens per timing measurement")
    parser.add_argument("--profile-context", type=int, default=512, help="Context length for the profiler window")
    parser.add_argument("--profile-steps", type=int, default=8)
    parser.add_argument("--prefill-lens", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--trace", action="store_true", help="Export Chrome trace for the profile window")
    parser.add_argument("--output", type=str, default=None, help="Write JSON results here")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    results: dict[str, Any] = {
        "model_size": args.model_size,
        "checkpoint": args.checkpoint,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "triton_attnres_routing": bool(llama_mod._TRITON_ATTN_RES_AVAILABLE),
        "env_no_triton_attnres": os.environ.get("KOTODAMA_NO_TRITON_ATTNRES"),
        "decode_tokens_per_measurement": args.decode_tokens,
    }
    logger.info("Triton AttnRes routing active in llama.py: %s", results["triton_attnres_routing"])

    model = build_model(args, device)
    vocab = model.config.vocab_size

    # Match serve.py's backend toggle so SDPA picks the same kernels
    torch.backends.cuda.enable_cudnn_sdp(False)

    # ── Warmup: SDPA plan cache across lengths + general JIT ────────────────
    logger.info("Warmup...")
    with torch.inference_mode():
        for plen in [16, 128, 512]:
            ids = make_prompt_ids(plen, vocab, device)
            tok, past_kv = prefill(model, ids)
            for _ in range(8):
                out = model(tok, use_cache=True, past_kv=past_kv)
                past_kv = out["past_kv"]
                tok = out["logits"][:, -1:].argmax(dim=-1)
        torch.cuda.synchronize(device)
    logger.info("Warmup done")

    # ── 1. Prefill timing ────────────────────────────────────────────────────
    prefill_ms: dict[str, float] = {}
    for plen in args.prefill_lens:
        ids = make_prompt_ids(plen, vocab, device)
        ms = timed_prefill(model, ids, device)
        prefill_ms[f"prompt_{plen}_ms"] = round(ms, 2)
        logger.info("Prefill %d tokens: %.1f ms (%.0f tok/s)", plen, ms, plen / ms * 1000)
    results["prefill"] = prefill_ms

    # ── 2. Decode timing vs context length, synced vs free ──────────────────
    decode_rows: list[dict[str, Any]] = []
    for ctx in args.context_lens:
        tok, past_kv = grow_cache_to(model, ctx, vocab, device)
        # cat creates new tensors each step, so both modes can share the source cache
        synced = decode_synced(model, tok.clone(), past_kv, args.decode_tokens, device)
        free = decode_free(model, tok, past_kv, args.decode_tokens, device)
        row = DecodeTiming(
            context_len=ctx,
            n_tokens=args.decode_tokens,
            synced_ms_per_token=round(synced, 3),
            free_ms_per_token=round(free, 3),
            synced_tok_per_s=round(1000 / synced, 1),
            free_tok_per_s=round(1000 / free, 1),
        )
        decode_rows.append(asdict(row))
        logger.info(
            "ctx=%4d  synced: %6.2f ms/tok (%5.1f tok/s)   free: %6.2f ms/tok (%5.1f tok/s)",
            ctx, synced, 1000 / synced, free, 1000 / free,
        )
    results["decode_timing"] = decode_rows

    # ── 3. Profiler window at representative context ─────────────────────────
    ctx = args.profile_context
    tok, past_kv = grow_cache_to(model, ctx, vocab, device)
    # unprofiled wall reference for the same op mix
    wall_ref = decode_free(model, tok.clone(), past_kv, args.decode_tokens, device)
    trace_path = None
    if args.trace:
        out_base = Path(args.output).parent if args.output else Path("outputs/profiles")
        trace_path = out_base / f"decode_trace_ctx{ctx}.json"
    summary = profile_decode_window(model, tok, past_kv, args.profile_steps, device, wall_ref, trace_path)
    results["profile_window"] = asdict(summary)

    logger.info("=" * 70)
    logger.info("PROFILE WINDOW ctx=%d", summary.context_len)
    logger.info("  wall (unprofiled): %.2f ms/step", summary.wall_ms_per_step_unprofiled)
    logger.info("  GPU kernel time:   %.2f ms/step  → GPU busy %.1f%%", summary.gpu_kernel_ms_per_step, 100 * summary.gpu_busy_fraction)
    logger.info("  kernel launches:   %.0f /step", summary.kernel_launches_per_step)
    for k, v in summary.buckets_ms_per_step.items():
        logger.info("    %-28s %7.3f ms/step (%4.1f%%)  %5.0f launches", k, v, summary.buckets_pct_of_kernel_time.get(k, 0), summary.buckets_launches_per_step.get(k, 0))

    # ── 4. Sampling microbench (serve.py parity) ────────────────────────────
    try:
        results["sample_next_token"] = sample_next_token_cost(vocab, device)
        logger.info("sample_next_token: %s", results["sample_next_token"])
    except Exception as exc:  # serve.py import can fail outside deploy dir
        logger.warning("Sampling microbench skipped: %s", exc)
        results["sample_next_token"] = None

    results["peak_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
    logger.info("Peak GPU memory: %.2f GB", results["peak_memory_gb"])

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
