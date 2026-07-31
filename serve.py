"""
Inference server for kotodama models (3B-first; one replica = one model on one GPU).

Usage:
    python serve.py --checkpoint /path/to/model.pt [--prefix-cache]
    python serve.py --checkpoint /path/to/instruct.pt --mode chat   # mode auto-detected
                                                                    # from 'instruct' in path

Two decode paths, deliberately only two:
  * fast (default)  — DecodeEngine: static KV cache, max-autotune compiled +
    CUDA-graphed step, all sampling laws in-graph (temperature, repetition
    penalty, top-k/top-p), compiled block prefill/extend, optional token-exact
    prefix cache. ~400 tok/s decode on B200. Startup pays a one-time compile
    (~1 min warm inductor cache, minutes cold) — see gateway warmup timeouts.
  * reference       — plain eager llama.py loop. The parity oracle and debug
    path (~10x slower); NOT for production traffic.

The fleet story (gateway.py) runs many single-GPU replicas of this server;
concurrency comes from the fleet, not from batching within a replica.

Requires: fastapi, uvicorn, transformers (tokenizer only), torch
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import gc
import html
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

# CPU threading: 2 threads is optimal for single-request GPU inference.
# cuDNN SDPA needs >= 2 threads; beyond that, overhead is kernel launch
# latency (not parallelizable). Avoids 128-thread default on large Xeons.
_SERVE_THREADS = int(os.environ.get("LUXIA_SERVE_THREADS", "2"))
os.environ.setdefault("OMP_NUM_THREADS", str(_SERVE_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_SERVE_THREADS))

# expandable_segments cuts allocator fragmentation under variable-length decode
# and bursty concurrency (reserved-but-unallocated headroom). Set before the
# torch import so it applies regardless of how the server is launched.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

torch.set_num_threads(_SERVE_THREADS)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

# Fast decode engine (static cache + compiled/CUDA-graphed step). Optional.
try:
    from src.model.decode_engine import DecodeEngine, SamplingParams as EngineSamplingParams
    _ENGINE_AVAILABLE = True
except Exception as _engine_exc:
    DecodeEngine = None  # type: ignore[assignment,misc]
    EngineSamplingParams = None  # type: ignore[assignment,misc]
    _ENGINE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────────

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
DDV1_BOUNDARIES = [0, 3, 7, 12, 21, 25]
DD3B_BOUNDARIES = [0, 1, 3, 7, 15, 19, 24]

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)
IM_END_TOKEN_ID = 2

BASE_STOP_TOKEN_IDS = frozenset({0})
CHAT_STOP_TOKEN_IDS = frozenset({0, 2})

# Sampling defaults — shared by the native /generate and the OpenAI-compatible
# endpoints so every entrypoint agrees on the server's default behavior.
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 0
DEFAULT_TOP_P = 0.0
DEFAULT_REPETITION_PENALTY = 1.2

# Architecture-size label used to build the served model id
# (e.g. "kotodama-3b-instruct"). Falls back to the raw --model_size string.
SIZE_LABELS = {"proxy": "108m", "3b": "3b"}

PROXY_CONFIG = dict(
    hidden_size=512,
    num_layers=28,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=128,
    intermediate_size=1408,
    vocab_size=49152,
    max_position_embeddings=4096,
    rope_theta=500000.0,
    norm_eps=1e-5,
    qk_norm=True,
    tie_word_embeddings=True,
    z_loss_weight=0.0,
    use_liger=False,
    attn_impl="sdpa",
    attn_res=True,
    attn_res_boundaries=DDV1_BOUNDARIES,
)

CONFIG_3B = dict(
    hidden_size=3072,
    num_layers=28,
    num_attention_heads=24,
    num_kv_heads=8,
    head_dim=128,
    intermediate_size=8192,
    vocab_size=49152,
    max_position_embeddings=4096,
    rope_theta=500000.0,
    norm_eps=1e-5,
    qk_norm=True,
    tie_word_embeddings=True,
    z_loss_weight=0.0,
    use_liger=False,
    attn_impl="sdpa",
    attn_res=True,
    attn_res_boundaries=DD3B_BOUNDARIES,
)

MODEL_CONFIGS = {"proxy": PROXY_CONFIG, "3b": CONFIG_3B}


# ── Global state ────────────────────────────────────────────────────────────────

_model: LuxiaBaseModel | None = None
_tokenizer: AutoTokenizer | None = None
_device: torch.device = torch.device("cpu")
_serve_mode: str = "base"
_stop_token_ids: frozenset[int] = BASE_STOP_TOKEN_IDS
# Fast decode engine (single-stream): guarded by _engine_lock.
_engine: "DecodeEngine | None" = None
_engine_lock = asyncio.Lock()
# Prefix caching (engine extend-from-pos prefill); set by load_model.
_prefix_cache = False


def _engine_prefill(input_ids: "torch.Tensor") -> tuple["torch.Tensor", dict | None]:
    """Engine prefill, via the prefix cache when enabled.

    Returns (last-position logits, prefix-cache info dict or None).
    """
    assert _engine is not None
    if _prefix_cache:
        return _engine.prefill_cached(input_ids)
    return _engine.prefill(input_ids), None

# Cap concurrent in-flight streams so total KV/activation memory stays within
# GPU limits. Unbounded streaming concurrency + mid-stream disconnects caused a
# full-GPU OOM wedge. Override via LUXIA_MAX_CONCURRENT_STREAMS.
_MAX_CONCURRENT_STREAMS = int(os.environ.get("LUXIA_MAX_CONCURRENT_STREAMS", "4"))
_stream_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_STREAMS)

# Non-streaming generation is a fully synchronous CUDA decode loop; run directly
# on the event loop thread it starves /health + /memory for the whole generation
# (2026-07-04 incident: under saturation the gateway's health checker saw >15s
# probe silence and restarted busy replicas). Offload it to ONE dedicated worker
# thread: the loop stays responsive, while max_workers=1 preserves the exact
# serialization the loop-thread execution used to provide (the engine's static
# KV cache and the compiled decode path are stateful — never run two generations
# concurrently in this process; each replica owns a single GPU anyway).
#
# HARD INVARIANT (2026-07-04 follow-up regression): with --engine fast, ALL
# fast-engine activity — compile/cudagraph capture at startup warmup, replay,
# and every prefill/sample/step (streaming included) — must run on THIS thread.
# Inductor's cudagraph_trees keeps its tree-manager containers in THREAD-LOCAL
# storage; capture on one thread + replay on another dies with
# `assert torch._C._is_key_in_tls(attr_name)` in cudagraph_trees.get_obj.
# If you add a new engine call site, route it through _run_generation /
# run_in_executor(_generate_executor, ...). Never widen max_workers past 1.
_generate_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="koto-generate")


async def _run_generation(fn: Callable[..., "GenerateResponse"], /, *args: Any, **kwargs: Any) -> "GenerateResponse":
    """Run a blocking generation function on the dedicated generation thread.

    Exceptions (including HTTPException, e.g. prompt-too-long 400s) propagate
    unchanged to the awaiting handler.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_generate_executor, functools.partial(fn, *args, **kwargs))


@torch.inference_mode()
def _warmup_sdpa_cache(model: LuxiaBaseModel, max_seq_len: int = 4096, step: int = 64) -> None:
    """Prime SDPA plan cache by decoding across representative KV lengths.

    cuDNN SDPA selects an algorithm per unique KV shape. Without warmup,
    each novel shape costs ~300ms of CPU-side plan selection. This runs a
    single decode sweep so all shapes are cached before serving.
    """
    logger.info("Warming up SDPA plan cache (up to %d tokens, step %d)...", max_seq_len, step)
    t0 = time.time()

    dummy_ids = torch.zeros(1, 1, dtype=torch.long, device=_device)
    out = model(dummy_ids, use_cache=True)
    past_kv = out["past_kv"]
    tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

    target_positions = list(range(step, max_seq_len, step))
    pos = 1
    for target in target_positions:
        while pos < target:
            out = model(tok, use_cache=True, past_kv=past_kv)
            past_kv = out["past_kv"]
            tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
            pos += 1

    torch.cuda.synchronize()
    logger.info("SDPA warmup done in %.1fs (%d positions)", time.time() - t0, pos)


def load_model(checkpoint_path: str, device: str = "cuda", mode: str = "base", model_size: str = "3b", engine: str = "fast", prefix_cache: bool = False, warmup_sdpa: bool = False) -> tuple[LuxiaBaseModel, AutoTokenizer]:
    global _device, _serve_mode, _stop_token_ids, _engine, _prefix_cache
    _serve_mode = mode
    _stop_token_ids = CHAT_STOP_TOKEN_IDS if mode == "chat" else BASE_STOP_TOKEN_IDS
    if device.startswith("cuda") and torch.cuda.is_available():
        _device = torch.device(device)
    else:
        _device = torch.device("cpu")
    logger.info("Device: %s", _device)

    model_cfg = MODEL_CONFIGS.get(model_size)
    if model_cfg is None:
        raise ValueError(f"Unknown model_size {model_size!r}; choose from {sorted(MODEL_CONFIGS)}")
    config = LuxiaModelConfig(**model_cfg)
    logger.info("Model config (%s): %dM params", model_size, config.param_count() // 1_000_000)

    model = LuxiaBaseModel(config)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    if ckpt_path.suffix == ".zst":
        import zstandard as zstd
        logger.info("Decompressing zstd checkpoint...")
        dctx = zstd.ZstdDecompressor()
        with open(ckpt_path, "rb") as f_in:
            decompressed = dctx.decompress(f_in.read())
        ckpt = torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=False)
        del decompressed
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)
    logger.info("Checkpoint loaded (step %s, %s tokens)", ckpt.get("step", "?"), ckpt.get("tokens_consumed", "?"))

    model = model.to(_device).eval()
    if _device.type == "cuda":
        model = model.bfloat16()

    use_engine = engine == "fast" and _device.type == "cuda" and _ENGINE_AVAILABLE and config.attn_res
    if engine == "fast" and not use_engine:
        logger.warning("--engine fast requested but unavailable (cuda=%s, import=%s, attn_res=%s) — using reference path",
                       _device.type == "cuda", _ENGINE_AVAILABLE, config.attn_res)

    if _device.type == "cuda" and not use_engine:
        # Reference path only: the engine's masked SDPA relies on cuDNN flash.
        torch.backends.cuda.enable_cudnn_sdp(False)
        logger.info("cuDNN SDP disabled (using flash/math backend)")

    if use_engine:
        logger.info("Building fast decode engine (compile may take minutes on cold inductor cache)...")
        t0 = time.time()

        def _build_and_warm_engine() -> "DecodeEngine":
            eng = DecodeEngine(model)
            eng.compile_step(mode="max-autotune")
            # Compile every block-forward bucket (incl. overlap plans) now:
            # an unwarmed bucket would pay its compile on a live request.
            eng.warm_blocks()
            with torch.inference_mode():
                warm_params = EngineSamplingParams(temperature=0.9, repetition_penalty=1.2)
                trunc_params = EngineSamplingParams(temperature=0.9, repetition_penalty=1.2, top_p=0.9)
                for plen in [16, 256]:
                    dummy = torch.randint(4, config.vocab_size, (1, plen), device=_device)
                    logits = eng.prefill(dummy)
                    tok = eng.sample_first(logits, warm_params)
                    for _ in range(4):
                        tok = eng.step(tok, warm_params)
                    for _ in range(2):
                        # Warm the truncated (top-k/top-p) sampler graph too —
                        # a cold compile on the first truncated request would
                        # stall long enough to flap gateway health.
                        tok = eng.step(tok, trunc_params)
                    eng.step_logits(tok)  # warm the greedy/forward variant too
                if prefix_cache:
                    # Warm the extend path (MATH-SDPA suffix forward + match logic):
                    # the first real extend otherwise pays ~0.6s of lazy dispatch.
                    base_ids = torch.randint(4, config.vocab_size, (1, 64), device=_device)
                    eng.prefill(base_ids)
                    ext = torch.cat(
                        [base_ids, torch.randint(4, config.vocab_size, (1, 48), device=_device)], dim=1
                    )
                    eng.prefill_cached(ext)
                    # And the suffix==1 (regenerate) variant, which routes through
                    # the compiled step.
                    eng.prefill_cached(ext)
                torch.cuda.synchronize(_device)
            eng.reset()
            return eng

        # Build/compile/capture on the SAME single worker thread that serves all
        # generation (_generate_executor). max-autotune uses inductor's
        # cudagraph_trees, whose tree-manager containers live in THREAD-LOCAL
        # storage: a graph captured here on the main thread asserts
        # (torch._C._is_key_in_tls) the moment the worker thread replays it
        # (2026-07-04 production regression — 28ms 500s on every non-stream
        # request of warmed-at-startup replicas). load_model is sync (called
        # from lifespan before serving starts), so block on the future.
        _engine = _generate_executor.submit(_build_and_warm_engine).result()
        # Disable cuDNN SDPA only AFTER the engine compiled: the captured
        # decode graph keeps its baked cuDNN flash kernels, while the eager
        # FALLBACK path (non-engine requests) avoids cuDNN's ~300ms per-shape
        # plan-selection stalls by dispatching to pytorch flash instead.
        torch.backends.cuda.enable_cudnn_sdp(False)
        _prefix_cache = prefix_cache
        if _prefix_cache:
            logger.info("Prefix caching ENABLED (engine extend-from-pos prefill)")
        logger.info("Fast decode engine ready in %.1fs", time.time() - t0)
    elif _device.type == "cuda" and warmup_sdpa:
        # Reference path only: prime cuDNN plan cache so per-shape ~300ms
        # stalls don't land on live requests. ~100s; opt-in for debug use.
        _warmup_sdpa_cache(model, config.max_position_embeddings, step=64)

    # Run a few prefills at different lengths to warm any remaining caches
    if _device.type == "cuda":
        logger.info("Warming up prefill path...")
        with torch.inference_mode():
            for plen in [1, 16, 128, 512]:
                dummy = torch.zeros(1, plen, dtype=torch.long, device=_device)
                model(dummy, use_cache=True)
            torch.cuda.synchronize(_device)
        logger.info("Prefill warmup done")

        # Release warmup workspace back to the CUDA allocator
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated(_device) / 1e9
        reserved = torch.cuda.memory_reserved(_device) / 1e9
        logger.info("Post-warmup memory: %.3f GB allocated, %.3f GB reserved", alloc, reserved)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if mode == "chat":
        tokenizer.chat_template = CHATML_TEMPLATE
    logger.info("Tokenizer loaded: %s (vocab %d, mode=%s)", TOKENIZER_NAME, len(tokenizer), mode)

    return model, tokenizer


# ── Request/response schemas ────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=DEFAULT_REPETITION_PENALTY, ge=1.0, le=2.0)
    stop_strings: list[str] = Field(default_factory=list)
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    timing: dict[str, float] | None = None
    # exact token IDs for offline replay (anamnesis taste discriminator) —
    # always populated (cheap, already in hand); surfaced in the OAI response
    # only when the request sets return_token_ids
    prompt_ids: list[int] | None = None
    completion_ids: list[int] | None = None


class ModelInfo(BaseModel):
    name: str
    mode: str
    params: int
    config: dict
    device: str
    checkpoint: str
    fast_engine: bool
    prefix_cache: bool = False


def _resolve_prompt(request: GenerateRequest, tokenizer: AutoTokenizer) -> str:
    """Resolve prompt text from either raw prompt or messages array."""
    if request.messages is not None:
        if _serve_mode != "chat":
            raise HTTPException(400, "messages field requires --mode chat")
        msgs = [{"role": m.role, "content": m.content} for m in request.messages]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    if request.prompt is not None:
        return request.prompt
    raise HTTPException(400, "Either 'prompt' or 'messages' must be provided")


# ── Sampling ────────────────────────────────────────────────────────────────────

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    generated_ids: list[int],
) -> int:
    logits = logits.float()

    if repetition_penalty != 1.0 and generated_ids:
        penalty_ids = torch.tensor(generated_ids, device=logits.device, dtype=torch.long).unique()
        penalty_logits = logits[penalty_ids]
        penalty_logits = torch.where(
            penalty_logits > 0,
            penalty_logits / repetition_penalty,
            penalty_logits * repetition_penalty,
        )
        logits[penalty_ids] = penalty_logits

    if temperature == 0.0:
        return logits.argmax().item()

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = logits.topk(top_k).values[-1]
        logits = logits.masked_fill(logits < kth_val, float("-inf"))

    if top_p > 0.0 and top_p < 1.0:
        sorted_logits, sorted_indices = logits.sort(descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        mask = cumulative_probs - sorted_logits.softmax(dim=-1) >= top_p
        sorted_logits[mask] = float("-inf")
        logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ── Model forward (fast or standard) ───────────────────────────────────────────

@torch.inference_mode()
def model_forward(model: LuxiaBaseModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Plain eager forward, returning logits (reference/debug path)."""
    return model(input_ids)["logits"]


# ── Generation ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate(
    model: LuxiaBaseModel,
    tokenizer: AutoTokenizer,
    request: GenerateRequest,
) -> GenerateResponse:
    timing: dict[str, float] = {}

    # Tokenization
    t_tok = time.perf_counter()
    prompt_text = _resolve_prompt(request, tokenizer)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(_device)
    prompt_len = input_ids.shape[1]
    timing["tokenize_ms"] = (time.perf_counter() - t_tok) * 1000

    if prompt_len >= model.config.max_position_embeddings:
        raise HTTPException(400, f"Prompt too long: {prompt_len} tokens (max {model.config.max_position_embeddings})")

    generated_ids: list[int] = []
    t0 = time.perf_counter()

    decode_model = model

    # Prefill: process entire prompt, cache KV (always eager — variable prompt shapes)
    if _device.type == "cuda":
        torch.cuda.synchronize(_device)
    t_prefill = time.perf_counter()
    output = model(input_ids, use_cache=True)
    if _device.type == "cuda":
        torch.cuda.synchronize(_device)
    timing["prefill_ms"] = (time.perf_counter() - t_prefill) * 1000

    logits = output["logits"]
    past_kv = output.get("past_kv")

    t_sample = time.perf_counter()
    next_logits = logits[0, -1]
    token_id = sample_next_token(
        next_logits, request.temperature, request.top_k, request.top_p,
        request.repetition_penalty, generated_ids,
    )
    timing["first_sample_ms"] = (time.perf_counter() - t_sample) * 1000

    if token_id in _stop_token_ids:
        elapsed = time.perf_counter() - t0
        return GenerateResponse(text="", prompt_tokens=prompt_len,
                                completion_tokens=0, tokens_per_second=0.0,
                                timing=timing,
                                prompt_ids=input_ids[0].tolist(),
                                completion_ids=[])

    generated_ids.append(token_id)
    next_input = torch.tensor([[token_id]], device=_device)

    # Decode: one token at a time with KV cache (compiled if available)
    decode_forward_ms = 0.0
    decode_sample_ms = 0.0
    decode_stop_ms = 0.0
    decode_text_ms = 0.0

    for _ in range(request.max_new_tokens - 1):
        if prompt_len + len(generated_ids) >= model.config.max_position_embeddings:
            break

        if _device.type == "cuda":
            torch.cuda.synchronize(_device)
        t_fwd = time.perf_counter()
        output = decode_model(next_input, use_cache=True, past_kv=past_kv)
        if _device.type == "cuda":
            torch.cuda.synchronize(_device)
        decode_forward_ms += (time.perf_counter() - t_fwd) * 1000

        logits = output["logits"]
        past_kv = output.get("past_kv")

        t_samp = time.perf_counter()
        next_logits = logits[0, -1]
        token_id = sample_next_token(
            next_logits, request.temperature, request.top_k, request.top_p,
            request.repetition_penalty, generated_ids,
        )
        decode_sample_ms += (time.perf_counter() - t_samp) * 1000

        if token_id in _stop_token_ids:
            break

        generated_ids.append(token_id)
        next_input = torch.tensor([[token_id]], device=_device)

        if request.stop_strings:
            t_stop = time.perf_counter()
            decoded_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if any(s in decoded_so_far for s in request.stop_strings):
                decode_stop_ms += (time.perf_counter() - t_stop) * 1000
                break
            decode_stop_ms += (time.perf_counter() - t_stop) * 1000

    elapsed = time.perf_counter() - t0

    t_decode_text = time.perf_counter()
    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    timing["final_decode_ms"] = (time.perf_counter() - t_decode_text) * 1000

    n_decode_tokens = max(len(generated_ids) - 1, 0)
    timing["decode_forward_total_ms"] = round(decode_forward_ms, 2)
    timing["decode_sample_total_ms"] = round(decode_sample_ms, 2)
    timing["decode_stop_total_ms"] = round(decode_stop_ms, 2)
    if n_decode_tokens > 0:
        timing["decode_forward_per_token_ms"] = round(decode_forward_ms / n_decode_tokens, 2)
        timing["decode_sample_per_token_ms"] = round(decode_sample_ms / n_decode_tokens, 2)
    timing["total_ms"] = round(elapsed * 1000, 2)

    tps = len(generated_ids) / elapsed if elapsed > 0 else 0.0

    return GenerateResponse(
        text=completion_text,
        prompt_tokens=prompt_len,
        completion_tokens=len(generated_ids),
        tokens_per_second=round(tps, 1),
        timing=timing,
        prompt_ids=input_ids[0].tolist(),
        completion_ids=list(generated_ids),
    )


# ── Fast-engine generation ──────────────────────────────────────────────────

def _engine_compatible(request: GenerateRequest) -> bool:
    """Engine handles the full law surface: temperature + rep penalty + top-k/top-p.

    Truncated laws (top_k > 0 or 0 < top_p < 1) run the engine's dedicated
    compiled sampler variant (2026-07-05) — they used to fall back to the
    reference path at a ~10x decode penalty (the SERVING-LAWS p90/k100 cliff).
    """
    return _engine is not None


@torch.inference_mode()
def engine_generate(request: GenerateRequest) -> GenerateResponse:
    """Non-streaming generation on the fast engine. Caller must hold _engine_lock."""
    assert _engine is not None and _tokenizer is not None and _model is not None
    timing: dict[str, float] = {}
    max_pos = _model.config.max_position_embeddings

    t_tok = time.perf_counter()
    prompt_text = _resolve_prompt(request, _tokenizer)
    input_ids = _tokenizer.encode(prompt_text, return_tensors="pt").to(_device)
    prompt_len = input_ids.shape[1]
    timing["tokenize_ms"] = (time.perf_counter() - t_tok) * 1000
    if prompt_len >= max_pos:
        raise HTTPException(400, f"Prompt too long: {prompt_len} tokens (max {max_pos})")

    params = EngineSamplingParams(
        temperature=request.temperature, repetition_penalty=request.repetition_penalty,
        top_k=request.top_k, top_p=request.top_p,
    )

    t0 = time.perf_counter()
    t_prefill = time.perf_counter()
    prefill_logits, pc_info = _engine_prefill(input_ids)
    tok = _engine.sample_first(prefill_logits, params)
    token_id = int(tok.item())
    timing["prefill_ms"] = (time.perf_counter() - t_prefill) * 1000
    if pc_info is not None:
        timing["prefix_cache_hit"] = pc_info["prefix_hit"]
        timing["prefix_common_tokens"] = pc_info["common_prefix"]
        timing["prefill_suffix_tokens"] = pc_info["suffix_len"]

    generated_ids: list[int] = []
    if token_id not in _stop_token_ids:
        generated_ids.append(token_id)
        for _ in range(request.max_new_tokens - 1):
            if prompt_len + len(generated_ids) >= max_pos:
                break
            tok = _engine.step(tok, params)
            token_id = int(tok.item())
            if token_id in _stop_token_ids:
                break
            generated_ids.append(token_id)
            if request.stop_strings:
                decoded_so_far = _tokenizer.decode(generated_ids, skip_special_tokens=True)
                if any(s in decoded_so_far for s in request.stop_strings):
                    break

    elapsed = time.perf_counter() - t0
    completion_text = _tokenizer.decode(generated_ids, skip_special_tokens=True)
    timing["total_ms"] = round(elapsed * 1000, 2)
    n_decode = max(len(generated_ids) - 1, 0)
    if n_decode > 0:
        timing["decode_per_token_ms"] = round((elapsed - timing["prefill_ms"] / 1000) * 1000 / n_decode, 3)
    tps = len(generated_ids) / elapsed if elapsed > 0 else 0.0

    return GenerateResponse(
        text=completion_text,
        prompt_tokens=prompt_len,
        completion_tokens=len(generated_ids),
        tokens_per_second=round(tps, 1),
        timing=timing,
        prompt_ids=input_ids[0].tolist(),
        completion_ids=list(generated_ids),
    )


@torch.inference_mode()
async def engine_generate_stream(
    request: GenerateRequest,
    http_request: Request | None = None,
):
    """Streaming generation on the fast engine.

    Emits text via windowed incremental detokenization: pending token ids are
    decoded together and flushed once the text no longer ends in an incomplete
    UTF-8 sequence (U+FFFD) — O(window) per step instead of O(n) full redecode.

    THREADING: every engine call below is dispatched to _generate_executor —
    the single thread the engine compiled/captured on. _engine.step (T > 0)
    replays the cudagraph-captured fused step, and _engine_prefill can replay
    it too (--prefix-cache suffix==1 regenerate path); cudagraph_trees keeps
    its tree managers in THREAD-LOCAL storage, so touching either from the
    event loop thread asserts (torch._C._is_key_in_tls). sample_first/prefill
    are eager but mutate shared engine buffers, so they take the same thread.
    Cost: one ~50-100us executor hop per token. _engine_lock is held across
    the whole stream, so worker-thread engine ops never interleave requests.
    """
    assert _engine is not None and _tokenizer is not None and _model is not None
    async with _engine_lock:
        loop = asyncio.get_running_loop()
        max_pos = _model.config.max_position_embeddings
        try:
            prompt_text = _resolve_prompt(request, _tokenizer)
            input_ids = _tokenizer.encode(prompt_text, return_tensors="pt").to(_device)
            prompt_len = input_ids.shape[1]
            if prompt_len >= max_pos:
                yield f'data: {{"error": "Prompt too long: {prompt_len} tokens"}}\n\n'
                return

            params = EngineSamplingParams(
                temperature=request.temperature, repetition_penalty=request.repetition_penalty,
                top_k=request.top_k, top_p=request.top_p,
            )
            t_prefill = time.perf_counter()
            prefill_logits, pc_info = await loop.run_in_executor(
                _generate_executor, _engine_prefill, input_ids
            )
            if pc_info is not None:
                logger.debug(
                    "stream prefill: hit=%s common=%d suffix=%d in %.1fms",
                    pc_info["prefix_hit"], pc_info["common_prefix"],
                    pc_info["suffix_len"], (time.perf_counter() - t_prefill) * 1000,
                )
            tok = await loop.run_in_executor(
                _generate_executor, _engine.sample_first, prefill_logits, params
            )
            token_id = int(tok.item())

            if token_id in _stop_token_ids:
                yield f"data: {json.dumps({'done': True, 'prompt_tokens': prompt_len, 'completion_tokens': 0})}\n\n"
                return

            generated_count = 0
            emitted_text = ""
            pending: list[int] = []

            def _flush() -> str:
                nonlocal emitted_text
                text = _tokenizer.decode(pending, skip_special_tokens=True)
                if text and not text.endswith("�"):
                    emitted_text += text
                    pending.clear()
                    return text
                return ""

            generated_count = 1
            pending.append(token_id)
            delta = _flush()
            if delta:
                yield f"data: {json.dumps({'token': delta})}\n\n"

            stopped = False
            interrupted = False
            for _ in range(request.max_new_tokens - 1):
                if prompt_len + generated_count >= max_pos:
                    break
                if http_request is not None and await http_request.is_disconnected():
                    interrupted = True
                    break

                tok = await loop.run_in_executor(
                    _generate_executor, _engine.step, tok, params
                )
                token_id = int(tok.item())
                if token_id in _stop_token_ids:
                    break
                generated_count += 1
                pending.append(token_id)
                delta = _flush()
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"
                if request.stop_strings and any(s in emitted_text for s in request.stop_strings):
                    stopped = True
                    break

            if not interrupted:
                # Flush any pending tail (possibly with incomplete bytes dropped
                # by skip_special decode semantics).
                if pending and not stopped:
                    tail = _tokenizer.decode(pending, skip_special_tokens=True)
                    if tail:
                        yield f"data: {json.dumps({'token': tail})}\n\n"
                yield f"data: {json.dumps({'done': True, 'prompt_tokens': prompt_len, 'completion_tokens': generated_count})}\n\n"
        finally:
            pass


@torch.inference_mode()
async def generate_stream(
    model: LuxiaBaseModel,
    tokenizer: AutoTokenizer,
    request: GenerateRequest,
    http_request: Request | None = None,
):
    # THREADING: this reference streaming path is safe on the event loop
    # thread — it never touches the fast engine (fully eager decode, no
    # cudagraph_trees thread-local state).
    # Hold a slot for the whole stream so concurrent streams stay bounded.
    async with _stream_semaphore:
        past_kv = None
        output = None
        interrupted = False
        try:
            prompt_text = _resolve_prompt(request, tokenizer)
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(_device)
            prompt_len = input_ids.shape[1]

            if prompt_len >= model.config.max_position_embeddings:
                yield f'data: {{"error": "Prompt too long: {prompt_len} tokens"}}\n\n'
                return

            generated_ids: list[int] = []
            prev_text = ""

            decode_model = model

            # Prefill (always eager)
            output = model(input_ids, use_cache=True)
            past_kv = output.get("past_kv")
            next_logits = output["logits"][0, -1]

            token_id = sample_next_token(
                next_logits, request.temperature, request.top_k, request.top_p,
                request.repetition_penalty, generated_ids,
            )

            if token_id in _stop_token_ids:
                yield f"data: {json.dumps({'done': True, 'prompt_tokens': prompt_len, 'completion_tokens': 0})}\n\n"
                return

            generated_ids.append(token_id)
            next_input = torch.tensor([[token_id]], device=_device)

            current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            delta = current_text
            prev_text = current_text
            if delta:
                yield f"data: {json.dumps({'token': delta})}\n\n"

            # Decode with KV cache (compiled if available)
            for _ in range(request.max_new_tokens - 1):
                if prompt_len + len(generated_ids) >= model.config.max_position_embeddings:
                    break

                # Stop early for clients that have disconnected. This keeps us off
                # the mid-stream exception path, which would otherwise pin this
                # request's CUDA tensors via the traceback reference cycle.
                if http_request is not None and await http_request.is_disconnected():
                    interrupted = True
                    break

                output = decode_model(next_input, use_cache=True, past_kv=past_kv)
                past_kv = output.get("past_kv")
                next_logits = output["logits"][0, -1]

                token_id = sample_next_token(
                    next_logits, request.temperature, request.top_k, request.top_p,
                    request.repetition_penalty, generated_ids,
                )

                if token_id in _stop_token_ids:
                    break

                generated_ids.append(token_id)
                next_input = torch.tensor([[token_id]], device=_device)

                current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                delta = current_text[len(prev_text):]
                prev_text = current_text

                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"

                if request.stop_strings:
                    if any(s in current_text for s in request.stop_strings):
                        break

            if not interrupted:
                yield f"data: {json.dumps({'done': True, 'prompt_tokens': prompt_len, 'completion_tokens': len(generated_ids)})}\n\n"
        except BaseException:
            # GeneratorExit / CancelledError on client disconnect arrive here.
            interrupted = True
            raise
        finally:
            # Drop refs to this request's CUDA tensors; on the interrupted path
            # force a cyclic GC so traceback-pinned tensors are reclaimed now
            # instead of accumulating across disconnects into a full-GPU OOM.
            past_kv = None
            output = None
            if interrupted:
                gc.collect()


# ── App ─────────────────────────────────────────────────────────────────────────

_checkpoint_path = ""
_mode_arg = "base"
_model_size_arg = "3b"
_served_name_override: str | None = None
_engine_arg = "fast"
_prefix_cache_arg = False
_warmup_sdpa_arg = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer
    _model, _tokenizer = load_model(
        _checkpoint_path, _device_arg, mode=_mode_arg,
        model_size=_model_size_arg, engine=_engine_arg, prefix_cache=_prefix_cache_arg,
        warmup_sdpa=_warmup_sdpa_arg,
    )
    try:
        yield
    finally:
        # Uvicorn's graceful shutdown has already drained in-flight requests by
        # the time we get here; drop anything still queued (its awaiting request
        # was cancelled) so the non-daemon worker thread can't stall exit.
        _generate_executor.shutdown(wait=False, cancel_futures=True)


def _model_name() -> str:
    """Canonical served model id, e.g. 'kotodama-3b-instruct'.

    Priority: explicit --served_model_name > derived from architecture size
    (--model_size) + serving mode.
    """
    if _served_name_override:
        return _served_name_override
    size_label = SIZE_LABELS.get(_model_size_arg, _model_size_arg)
    mode_label = "instruct" if _serve_mode == "chat" else "base"
    return f"kotodama-{size_label}-{mode_label}"


app = FastAPI(title="luxia DD-v1", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/info")
async def info():
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    return ModelInfo(
        name=_model_name(),
        mode=_serve_mode,
        params=_model.config.param_count(),
        config=asdict(_model.config),
        device=str(_device),
        checkpoint=_checkpoint_path,
        fast_engine=_engine is not None,
        prefix_cache=_prefix_cache,
    )


@app.get("/memory")
async def memory():
    if _device.type != "cuda":
        return {"device": "cpu"}
    return {
        "device": str(_device),
        "allocated_gb": round(torch.cuda.memory_allocated(_device) / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved(_device) / 1e9, 3),
        "peak_gb": round(torch.cuda.max_memory_allocated(_device) / 1e9, 3),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest, http_request: Request):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded")

    if request.stream:
        if _engine_compatible(request):
            return StreamingResponse(
                engine_generate_stream(request, http_request),
                media_type="text/event-stream",
            )
        return StreamingResponse(
            generate_stream(_model, _tokenizer, request, http_request),
            media_type="text/event-stream",
        )

    if _engine_compatible(request):
        async with _engine_lock:
            return await _run_generation(engine_generate, request)
    return await _run_generation(generate, request=request, model=_model, tokenizer=_tokenizer)


# ── OpenAI-compatible /v1/completions (for Loom/Loomsidian) ─────────────────────

class OAICompletionRequest(BaseModel):
    prompt: str
    model: str = ""
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    n: int = Field(default=1, ge=1, le=8)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    best_of: int | None = None
    stop: list[str] | str | None = None
    return_token_ids: bool = False


@app.post("/v1/completions")
async def oai_completions(request: OAICompletionRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, detail="Model not loaded")

    stop_strings: list[str] = []
    if isinstance(request.stop, str):
        stop_strings = [request.stop]
    elif isinstance(request.stop, list):
        stop_strings = request.stop

    repetition_penalty = (
        1.0 + request.frequency_penalty
        if request.frequency_penalty is not None
        else DEFAULT_REPETITION_PENALTY
    )

    choices = []
    for i in range(request.n):
        gen_req = GenerateRequest(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=repetition_penalty,
            stop_strings=stop_strings,
        )
        if _engine_compatible(gen_req):
            async with _engine_lock:
                result = await _run_generation(engine_generate, gen_req)
        else:
            result = await _run_generation(generate, model=_model, tokenizer=_tokenizer, request=gen_req)
        choice = {
            "text": html.unescape(result.text),
            "index": i,
            "logprobs": None,
            "finish_reason": "length" if result.completion_tokens >= request.max_tokens else "stop",
        }
        if request.return_token_ids:
            choice["token_ids"] = {"prompt": result.prompt_ids,
                                   "completion": result.completion_ids}
        choices.append(choice)

    prompt_tokens = _tokenizer.encode(request.prompt, return_tensors="pt").shape[1]
    completion_tokens = sum(
        _tokenizer.encode(c["text"], return_tensors="pt").shape[1] for c in choices
    )

    return {
        "id": f"cmpl-luxia-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": _model_name(),
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/v1/models")
async def oai_models():
    # One model is loaded per process; report the one actually being served.
    active = _model_name()
    return {
        "object": "list",
        "data": [
            {
                "id": active,
                "object": "model",
                "owned_by": "aethera-gp",
                "active": True,
            }
        ],
    }


# ── OpenAI-compatible /v1/chat/completions (chat mode) ────────────────────────


class OAIChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = ""
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    stop: list[str] | str | None = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=8)
    return_token_ids: bool = False


@app.post("/v1/chat/completions")
async def oai_chat_completions(request: OAIChatRequest, http_request: Request):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, detail="Model not loaded")
    if _serve_mode != "chat":
        raise HTTPException(400, detail="/v1/chat/completions requires --mode chat")

    msgs = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt_text = _tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )

    stop_strings: list[str] = []
    if isinstance(request.stop, str):
        stop_strings = [request.stop]
    elif isinstance(request.stop, list):
        stop_strings = request.stop

    repetition_penalty = (
        1.0 + request.frequency_penalty
        if request.frequency_penalty is not None
        else DEFAULT_REPETITION_PENALTY
    )

    if request.stream:
        gen_req = GenerateRequest(
            prompt=prompt_text,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=repetition_penalty,
            stop_strings=stop_strings,
            stream=True,
        )

        if _engine_compatible(gen_req):
            source_stream = engine_generate_stream(gen_req, http_request)
        else:
            source_stream = generate_stream(_model, _tokenizer, gen_req, http_request)

        async def chat_stream():
            async for chunk in source_stream:
                data = json.loads(chunk.removeprefix("data: ").strip())
                if "token" in data:
                    oai_chunk = {
                        "id": f"chatcmpl-luxia-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": _model_name(),
                        "choices": [{
                            "index": 0,
                            "delta": {"content": data["token"]},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(oai_chunk)}\n\n"
                elif data.get("done"):
                    final_chunk = {
                        "id": f"chatcmpl-luxia-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": _model_name(),
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

        return StreamingResponse(chat_stream(), media_type="text/event-stream")

    choices = []
    total_completion_tokens = 0
    for i in range(request.n):
        gen_req = GenerateRequest(
            prompt=prompt_text,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=repetition_penalty,
            stop_strings=stop_strings,
        )
        if _engine_compatible(gen_req):
            async with _engine_lock:
                result = await _run_generation(engine_generate, gen_req)
        else:
            result = await _run_generation(generate, model=_model, tokenizer=_tokenizer, request=gen_req)
        total_completion_tokens += result.completion_tokens
        choice = {
            "index": i,
            "message": {"role": "assistant", "content": html.unescape(result.text)},
            "finish_reason": "length" if result.completion_tokens >= request.max_tokens else "stop",
        }
        if request.return_token_ids:
            choice["token_ids"] = {"prompt": result.prompt_ids,
                                   "completion": result.completion_ids}
        choices.append(choice)

    prompt_tokens = _tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").shape[1]

    return {
        "id": f"chatcmpl-luxia-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_name(),
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": prompt_tokens + total_completion_tokens,
        },
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────

_device_arg = "cuda"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kotodama inference server")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint .pt/.pt.zst file")
    parser.add_argument("--device", default="cuda", help="Device: cuda, cuda:N, or cpu")
    parser.add_argument("--mode", choices=["base", "chat"], default=None,
                        help="Serving mode (auto-detected from checkpoint path if omitted)")
    parser.add_argument("--engine", choices=["fast", "reference"], default="fast",
                        help="'fast' (default) = static-cache CUDA-graph DecodeEngine, all "
                             "sampling laws incl. top-k/top-p; 'reference' = eager debug/"
                             "parity path (~10x slower decode)")
    parser.add_argument("--prefix-cache", action="store_true",
                        help="Reuse the engine KV cache across requests sharing a token-exact "
                             "prompt prefix (multi-turn TTFT win; requires --engine fast and "
                             "per-conversation replica pinning)")
    parser.add_argument("--model_size", choices=list(MODEL_CONFIGS), default="3b",
                        help="Model architecture size (3b=2.97B, proxy=108M legacy)")
    parser.add_argument("--served_model_name", default=None,
                        help="Override the model id reported by /info and /v1/models "
                             "(default: kotodama-<size>-<base|instruct>)")
    parser.add_argument("--warmup-sdpa", action="store_true",
                        help="Reference engine only: pre-pay the ~100s cuDNN SDPA plan-cache "
                             "sweep at startup instead of ~300ms per novel shape at runtime")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2222)
    args = parser.parse_args()

    _checkpoint_path = args.checkpoint
    inferred_mode = "chat" if "instruct" in args.checkpoint else "base"

    _served_name_override = args.served_model_name
    _device_arg = args.device
    _mode_arg = args.mode if args.mode is not None else inferred_mode
    _model_size_arg = args.model_size
    _engine_arg = args.engine
    _prefix_cache_arg = args.prefix_cache
    _warmup_sdpa_arg = args.warmup_sdpa
    if _prefix_cache_arg and _engine_arg != "fast":
        logger.warning("--prefix-cache requires --engine fast; ignoring")
        _prefix_cache_arg = False

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
