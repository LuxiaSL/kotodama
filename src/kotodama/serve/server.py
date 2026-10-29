"""
Inference server for kotodama models (3B-first; one replica = one model on one GPU).

Usage:
    python -m kotodama.serve.server --checkpoint /path/to/model.pt [--prefix-cache]
    python -m kotodama.serve.server --checkpoint /path/to/instruct.pt --mode chat   # mode auto-detected
                                                                    # from 'instruct' in path
    python -m kotodama.serve.server --checkpoint ... --steer-npz algebra.npz,reppen.npz,binding.npz
                                        # per-request MULTI-SITE residual steering in the
                                        # fast path (per-key site from the _s{sublayer}
                                        # suffix; block-persistent; kotodama/serve/steering.py)

Two decode paths, deliberately only two:
  * fast (default)  — DecodeEngine: static KV cache, max-autotune compiled +
    CUDA-graphed step, all sampling laws in-graph (temperature, repetition
    penalty, top-k/top-p), compiled block prefill/extend, optional token-exact
    prefix cache. Single-stream decode is kernel-count bound (per-token cost
    grows with effective depth), so throughput comes from the batched row
    decoders (--multi_rows / --xbatch_rows; serve profiles turn them on).
    Startup pays a one-time compile per decode graph (minutes warm, much longer
    cold with many row buckets) — see gateway warmup timeouts.
  * reference       — plain eager llama.py loop. The parity oracle and debug
    path (~10x slower); NOT for production traffic.

The fleet story (kotodama.serve.gateway) runs many single-GPU replicas of this server;
concurrency comes from the fleet, not from batching within a replica.

Requires: fastapi, uvicorn, transformers (tokenizer only), torch
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import gc
import html
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

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
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import AutoTokenizer

from kotodama import ckpt, presets
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig

# Fast decode engine (static cache + compiled/CUDA-graphed step). Optional.
try:
    from kotodama.model.decode_engine import DecodeEngine, SamplingParams as EngineSamplingParams
    from kotodama.model.multi_decode import CascadeDecoder, MultiSampleDecoder, decode_rows
    from kotodama.model.prefix_cache import PrefixKVCache
    from kotodama.model.regional import check_norms as regional_check_norms
    from kotodama.serve.xbatch import XBatchJob, XBatchRunner, XBatchScheduler
    _ENGINE_AVAILABLE = True
except Exception as _engine_exc:
    DecodeEngine = None  # type: ignore[assignment,misc]
    EngineSamplingParams = None  # type: ignore[assignment,misc]
    _ENGINE_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────────

TOKENIZER_NAME = presets.TOKENIZER_NAME
DDV1_BOUNDARIES = list(presets.DDV1)
DD3B_BOUNDARIES = list(presets.DD3B)

from kotodama.serve import compile_cache  # noqa: E402
from kotodama.serve.steering import (  # noqa: E402
    STEER_DEFAULT_ALPHA,
    act_injection_to_plan,
    injection_to_plan,
    load_partial_norms,
    load_steer_banks,
    writes_to_plan,
)
from kotodama.serve.chatml import (  # noqa: E402
    BASE_STOP_TOKEN_IDS,
    CHAT_STOP_TOKEN_IDS,
    CHATML_TEMPLATE,
)

# Sampling defaults — shared by the native /generate and the OpenAI-compatible
# endpoints so every entrypoint agrees on the server's default behavior.
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 0
DEFAULT_TOP_P = 0.0
DEFAULT_REPETITION_PENALTY = 1.2

# Architecture-size label that goes into the served model id
# (e.g. "kotodama-3b-instruct"). Falls back to the raw --model_size string.
SIZE_LABELS = {"proxy": "108m", "1b": "1b", "3b": "3b"}


PROXY_CONFIG = dict(
    **presets.shape("proxy"),
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
    **presets.shape("3b"),
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

# 1B formation-ladder shape (presets "1b"); DD-3B boundaries like the 3B. The
# ladder arms trained with rope_fp32 — serve them with --rope_fp32.
CONFIG_1B = dict(CONFIG_3B, **presets.shape("1b"))

# Depth arm (presets "1b-deep": d1536 x 52 layers, the 1b's budget); DD-3B scaled to 52 layers.
CONFIG_1B_DEEP = dict(CONFIG_3B, **presets.shape("1b-deep"),
                      attn_res_boundaries=list(presets.DEFAULT_BOUNDARIES["1b-deep"]))

CONFIG_1B_DEEP_LOOP = dict(CONFIG_3B, **presets.shape("1b-deep-loop"),
                           attn_res_boundaries=list(presets.DEFAULT_BOUNDARIES["1b-deep-loop"]))

CONFIG_1B_DEEP_LOOPFIT = dict(CONFIG_3B, **presets.shape("1b-deep-loopfit"),
                              attn_res_boundaries=list(presets.DEFAULT_BOUNDARIES["1b-deep-loopfit"]))

CONFIG_1B_TSUMUGI = dict(CONFIG_3B, **presets.shape("1b-tsumugi"),
                               attn_res_boundaries=list(presets.DEFAULT_BOUNDARIES["1b-tsumugi"]))

MODEL_CONFIGS = {"proxy": PROXY_CONFIG, "1b": CONFIG_1B, "1b-deep": CONFIG_1B_DEEP,
                 "1b-deep-loop": CONFIG_1B_DEEP_LOOP, "1b-deep-loopfit": CONFIG_1B_DEEP_LOOPFIT,
                 "1b-tsumugi": CONFIG_1B_TSUMUGI, "3b": CONFIG_3B}


def model_config_for(model_size: str) -> dict | None:
    """The serving config for a size name or one of its public aliases (``presets.ALIASES``, e.g. ``1b-tsumugi``)."""
    return MODEL_CONFIGS.get(model_size) or MODEL_CONFIGS.get(presets.ALIASES.get(model_size, ""))


# ── Global state ────────────────────────────────────────────────────────────────

_model: KotodamaModel | None = None
_tokenizer: AutoTokenizer | None = None
_attn_res_override: str = "config"   # config | on | off (CLI --attn_res)
_attn_res_boundaries_override: str | None = None
_rope_fp32_override: bool = False   # CLI --rope_fp32 (1B ladder arms)
_device: torch.device = torch.device("cpu")
_serve_mode: str = "base"
_stop_token_ids: frozenset[int] = BASE_STOP_TOKEN_IDS
_max_seq_len = CONFIG_3B["max_position_embeddings"]
# Fast decode engine (single-stream): guarded by _engine_lock.
_engine: "DecodeEngine | None" = None
_engine_lock = asyncio.Lock()
# Batched multi-sample decoders over the engine (--multi_rows 4,8): an OpenAI request with 1 < n <= the largest bucket
# and a pure-temperature law decodes its n candidates in lockstep after one prefill, on the smallest bucket that fits
# (a step costs about the same at any row count, so the bucket sets the throughput). Empty = every candidate is serial.
_multi: "dict[int, MultiSampleDecoder]" = {}
# Cross-request batched decoding (--xbatch_rows 16,32,64): concurrent pure-temperature requests that share a sampling
# law and a steering config decode together, each request's rows filled from its own prefill (kotodama.serve.xbatch).
# None = off: every request takes the paths above.
_xbatch: "XBatchScheduler | None" = None
# Cascade attention in the row decoders (--cascade_attn): rows of one prompt share one copy of its KV and each row
# keeps only the KV it decodes, at most this many tokens; a request asking for more takes the serial path. 0 = the
# row decoders hold whole-context rows (no cap).
_row_token_cap: int = 0
_cascade_info: dict | None = None
# Prefix caching (engine extend-from-pos prefill); set by load_model.
_prefix_cache = False
# Multi-entry prefix KV pool (--prefix_pool_tokens): recent prompts' KV in fixed-size token blocks, so a prompt that
# shares a prefix with any pooled prompt (not only the previous request's) prefills only its suffix. None = off.
_prefix_pool: "PrefixKVCache | None" = None
# Residual steering (--steer-npz, comma-separated banks); set by load_model.
# The bank maps key -> (site_layer, UNIT vector fp32 on-device); per-site
# median norms are keyed by LAYER. The engine receives a per-request WRITES
# list [(site_layer, unit_vec, alpha * site_median_norm), ...] via set_steer.
_steer_loaded: bool = False
_adapter: dict | None = None  # the merged LoRA adapter's identity (kotodama.posttrain.lora.merge), or None
_steer_bank: dict[str, tuple[int, "torch.Tensor"]] = {}
_steer_meds: dict[int, float] = {}
_steer_aliases: dict[str, tuple[str | None, float, str]] = {}
# Per-layer median post-attention partial norms (``partial_norm_l{L}`` keys in any --steer-npz file, e.g. the
# scripts/steer/partial_norms.py sidecar): the scale table of the N write rule. Empty = rule N unavailable (400).
_steer_partial_norms: dict[int, float] = {}
# Per-request levers (--levers N, kotodama.model.levers): every steered request binds its own lever slot in the
# engine's tables for its lifetime (N slots at once), so steered and unsteered requests decode in one batch and a
# request may steer its prompt too (steer_span "uniform"). 0 = off: steering takes the engine's broadcast set_steer.
_levers: int = 0
# Seeded row decoders (--seeded_rows): a request's `seed` gives each candidate its own reproducible noise stream.
_seeded_rows: bool = False
# torch.compile mode of the engine and the row decoders (--compile_mode), and whether warmup compiles the truncated
# (top-p / min-p) step variants or leaves them to their first request (--warm_truncated), and whether both compile
# regionally (--regional).
_compile_mode: str = "max-autotune"
_warm_truncated: bool = True
_regional: bool = False
# Row buckets narrower than this compile without kernel benchmarking (--autotune_min_rows): it buys them nothing.
_autotune_min_rows: int = 0
# Startup: when this process started, how long it took to be ready (and against what budget), for /info.
_PROCESS_T0 = time.time()
_startup_info: dict | None = None
_compile_store_state = "off"
# Server-wide default repetition penalty (CLI --default_repetition_penalty):
# used by /generate when the field is absent and by the OpenAI routes when
# frequency_penalty is absent.
_default_repetition_penalty: float = DEFAULT_REPETITION_PENALTY


def resolve_oai_repetition_penalty(frequency_penalty: float | None) -> float:
    """OpenAI frequency_penalty -> kotodama repetition penalty.

    Explicit v maps to 1 + v (0 = off); absent falls back to the server's
    default (--default_repetition_penalty).
    """
    if frequency_penalty is not None:
        return 1.0 + frequency_penalty
    return _default_repetition_penalty


def _engine_lock_contended() -> bool:
    """Whether a coroutine waits on the engine lock (read from the generation thread: a length read of the lock's
    waiter deque, which the lock creates lazily). The batcher never waits on the lock while it holds it, so a waiter
    is a request outside the batch."""
    return bool(getattr(_engine_lock, "_waiters", None))


def _engine_prefill(input_ids: "torch.Tensor", slot: int = 0) -> tuple["torch.Tensor", dict | None]:
    """Engine prefill, via the prefix cache when enabled.

    ``slot``: the request's lever slot (--levers). A uniform lever's prompt KV is steered, so it bypasses the pool
    (keyed by tokens alone); the engine's own cached prefix is reused only under the same lever.

    Returns (last-position logits, prefix-cache info dict or None).
    """
    assert _engine is not None
    uniform = bool(slot) and _engine.levers is not None and _engine.levers.span(slot) == "uniform"
    if _prefix_pool is not None and not uniform:
        out = _prefix_pool.prefill(input_ids)
        _engine._arm_decode_slot(slot)
        return out
    if _prefix_cache:
        return _engine.prefill_cached(input_ids, slot=slot)
    return _engine.prefill(input_ids, slot=slot), None


# Cap concurrent in-flight streams so total KV/activation memory stays within
# GPU limits. Unbounded streaming concurrency + mid-stream disconnects caused a
# full-GPU OOM wedge. Override via LUXIA_MAX_CONCURRENT_STREAMS.
_MAX_CONCURRENT_STREAMS = int(os.environ.get("LUXIA_MAX_CONCURRENT_STREAMS", "4"))
_stream_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_STREAMS)

# Non-streaming generation is a fully synchronous CUDA decode loop; run directly
# on the event loop thread it starves /health + /memory for the whole generation
# (under saturation the gateway's health checker sees >15s probe silence and
# restarts busy replicas). Offload it to ONE dedicated worker thread: the loop
# stays responsive, while max_workers=1 keeps generations strictly serialized
# (the engine's static
# KV cache and the compiled decode path are stateful — never run two generations
# concurrently in this process; each replica owns a single GPU anyway).
#
# HARD INVARIANT: with --engine fast, ALL
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
def _warmup_sdpa_cache(model: KotodamaModel, max_seq_len: int = 4096, step: int = 64) -> None:
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


def _load_checkpoint(checkpoint_path: Path | str) -> Any:
    """Load a checkpoint dict (.pt, streamed .pt.zst, or published weights: a directory or hf:REPO) —
    see kotodama.ckpt.load."""
    return ckpt.load(checkpoint_path)


def load_model(checkpoint_path: str, device: str = "cuda", mode: str = "base", model_size: str = "3b", engine: str = "fast", prefix_cache: bool = False, warmup_sdpa: bool = False, steer_npz: str | None = None, max_seq_len: int | None = None, multi_rows: tuple[int, ...] = (), xbatch_rows: tuple[int, ...] = (), xbatch_refill: bool = False, prefix_pool_tokens: int = 0, prefix_pool_block: int = 16, cascade_attn: bool = False, cascade_groups: int = 0, cascade_tail_tokens: int = 1024, levers: int = 0, seeded_rows: bool = False, compile_store: str | None = None, compile_mode: str = "max-autotune", warm_truncated: bool = True, regional: bool = False, autotune_min_rows: int = 0, adapter: str | None = None) -> tuple[KotodamaModel, AutoTokenizer]:
    global _device, _serve_mode, _stop_token_ids, _engine, _prefix_cache, _max_seq_len, _multi, _xbatch, _prefix_pool
    global _row_token_cap, _cascade_info, _levers, _seeded_rows, _compile_mode, _warm_truncated, _regional
    global _compile_store_state, _autotune_min_rows
    _compile_mode, _warm_truncated, _regional = compile_mode, bool(warm_truncated), bool(regional)
    _autotune_min_rows = int(autotune_min_rows)
    global _steer_loaded, _steer_bank, _steer_meds, _steer_aliases, _steer_partial_norms, _adapter
    if adapter is not None and steer_npz is not None:
        # The banks' vectors and the median norms that dose them are measurements of the base weights.
        raise ValueError("--steer-npz is calibrated on the base model and does not describe it with --adapter merged")
    _serve_mode = mode
    _stop_token_ids = CHAT_STOP_TOKEN_IDS if mode == "chat" else BASE_STOP_TOKEN_IDS
    # first: a tokenizer that cannot load fails the start in seconds, not after the engine has compiled
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if mode == "chat":
        tokenizer.chat_template = CHATML_TEMPLATE
    logger.info("Tokenizer loaded: %s (vocab %d, mode=%s)", TOKENIZER_NAME, len(tokenizer), mode)
    if device.startswith("cuda") and torch.cuda.is_available():
        _device = torch.device(device)
    else:
        _device = torch.device("cpu")
    logger.info("Device: %s", _device)

    model_cfg = model_config_for(model_size)
    if model_cfg is None:
        raise ValueError(f"Unknown model_size {model_size!r}; choose from {sorted(MODEL_CONFIGS)}")
    # Serve non-AttnRes checkpoints (e.g. proxy ablation arms) without
    # editing MODEL_CONFIGS — override via --attn_res off /
    # --attn_res_boundaries "0,3,7,12,21,25".
    model_cfg = dict(model_cfg)
    if _attn_res_override == "off":
        model_cfg["attn_res"] = False
    elif _attn_res_override == "on":
        model_cfg["attn_res"] = True
    if _attn_res_boundaries_override:
        model_cfg["attn_res_boundaries"] = [
            int(x) for x in _attn_res_boundaries_override.split(",")]
    if _rope_fp32_override:
        model_cfg["rope_fp32"] = True
    config = KotodamaModelConfig(**model_cfg)
    _max_seq_len = config.max_position_embeddings if max_seq_len is None else max_seq_len
    if not 16 <= _max_seq_len <= config.max_position_embeddings:
        raise ValueError(
            f"max_seq_len must be between 16 and {config.max_position_embeddings}, got {_max_seq_len}"
        )
    logger.info("Model config (%s): %dM params", model_size, config.param_count() // 1_000_000)
    logger.info("Serving context limit: %d tokens", _max_seq_len)

    model = KotodamaModel(config)

    ckpt_path = checkpoint_path if ckpt.is_hf_ref(checkpoint_path) else Path(checkpoint_path)
    if not ckpt.is_hf_ref(ckpt_path) and not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    loaded = _load_checkpoint(ckpt_path)

    state_dict = loaded.get("model", loaded)
    model.load_state_dict(state_dict, strict=True)
    logger.info("Checkpoint loaded (step %s, %s tokens)", loaded.get("step", "?"), loaded.get("tokens_consumed", "?"))
    del state_dict
    del loaded
    gc.collect()

    model = model.to(_device).eval()
    _adapter = None
    if adapter is not None:
        # Before the bf16 cast, so the weights round once, and before the engine is built: it binds fused copies of
        # the projection weights at construction. The identity on the model names it in capture_identity.
        from kotodama.posttrain import lora

        _adapter = model.adapter_identity = lora.merge(model, adapter)
    if _device.type == "cuda":
        model = model.bfloat16()

    use_engine = engine == "fast" and _device.type == "cuda" and _ENGINE_AVAILABLE and config.attn_res
    if engine == "fast" and not use_engine:
        logger.warning("--engine fast requested but unavailable (cuda=%s, import=%s, attn_res=%s) — using reference path",
                       _device.type == "cuda", _ENGINE_AVAILABLE, config.attn_res)

    # ── Steering config: validate + load the banks BEFORE the (minutes-long)
    # engine compile so a bad npz aborts immediately, not after warmup. ──
    if steer_npz is not None:
        if not use_engine:
            # The reference path has no injection hook — refuse loudly rather
            # than silently serving unsteered generations.
            raise RuntimeError(
                "--steer-npz requires the fast engine (cuda + --engine fast + attn_res model)"
            )
        _steer_bank, _steer_meds, _steer_aliases = load_steer_banks(
            steer_npz.split(","), config.num_layers, config.hidden_size, _device
        )
        _steer_partial_norms = load_partial_norms(steer_npz.split(","), config.num_layers)
        _steer_loaded = True
        logger.info(
            "Steering banks loaded: %d vectors %s @ site layers %s (meds %s), aliases %s",
            len(_steer_bank), sorted(_steer_bank),
            sorted({s for s, _ in _steer_bank.values()}),
            {layer: round(v) for layer, v in sorted(_steer_meds.items())},
            sorted(_steer_aliases),
        )

    if _device.type == "cuda" and not use_engine:
        # Reference path only: the engine's masked SDPA relies on cuDNN flash.
        torch.backends.cuda.enable_cudnn_sdp(False)
        logger.info("cuDNN SDP disabled (using flash/math backend)")

    _levers = int(levers) if use_engine else 0
    _seeded_rows = bool(seeded_rows) and use_engine
    if levers and not use_engine:
        raise ValueError("--levers needs the fast engine (cuda + --engine fast + attn_res model)")
    store_key = store_fields = None
    store_hit = False
    store_claim = None
    if use_engine and compile_store:
        # The compiled decode graphs depend on the model code, the runtime and these flags, not on the weights:
        # a stored blob for the same key turns the compiles below into cache hits.
        store_fields = compile_cache.key_fields({
            "model_size": model_size, "rope_fp32": _rope_fp32_override, "attn_res": _attn_res_override,
            "attn_res_boundaries": _attn_res_boundaries_override, "max_seq_len": _max_seq_len, "engine": engine,
            "mode": mode, "prefix_cache": prefix_cache, "multi_rows": list(multi_rows),
            "xbatch_rows": list(xbatch_rows), "xbatch_refill": xbatch_refill,
            "prefix_pool_tokens": prefix_pool_tokens, "prefix_pool_block": prefix_pool_block,
            "cascade_attn": cascade_attn, "cascade_groups": cascade_groups, "cascade_tail_tokens": cascade_tail_tokens,
            "levers": levers, "seeded_rows": seeded_rows, "steer_npz": steer_npz is not None,
            "compile_mode": compile_mode, "warm_truncated": bool(warm_truncated), "regional": bool(regional),
            "autotune_min_rows": int(autotune_min_rows),
        })
        store_key = compile_cache.cache_key(store_fields)
        store_hit, store_claim = compile_cache.acquire(compile_store, store_key)
        _compile_store_state = "hit" if store_hit else "miss"
    if use_engine:
        logger.info("Building fast decode engine (compile may take minutes on cold inductor cache)...")
        t0 = time.time()

        def _build_and_warm_engine() -> "DecodeEngine":
            # steer_enabled=False -> the engine's code paths (and compiled
            # graphs) are bit-identical to a pre-steering build; enabled bakes
            # one per-layer row-add in unconditionally (zero rows = off) so
            # per-request writes never touch the graphs. Warmup below runs
            # with the zero buffer — the graphs it captures ARE the steered
            # graphs.
            # --levers: per-request lever tables replace the broadcast rows (banks resolve into levers).
            eng = DecodeEngine(model, max_seq_len=_max_seq_len, steer_enabled=steer_npz is not None and not levers,
                               levers=levers)
            global _regional
            if _regional:
                try:
                    regional_check_norms(model, eng.eps)
                except ValueError as exc:  # the decoders follow the engine's choice
                    logger.warning("--regional: %s; compiling the whole-step graphs instead", exc)
                    _regional = False
            # prefill blocks compile without kernel benchmarking: it buys their replays little (measured within a
            # few % on the 1B) and costs minutes per bucket
            eng.compile_step(mode=_compile_mode, regional=_regional, block_mode=_without_autotune(_compile_mode))
            # Compile every block-forward bucket (incl. overlap plans) now:
            # an unwarmed bucket would pay its compile on a live request.
            eng.warm_blocks()
            with torch.inference_mode():
                warm_params = EngineSamplingParams(temperature=0.9, repetition_penalty=1.2)
                trunc_params = EngineSamplingParams(temperature=0.9, repetition_penalty=1.2, top_p=0.9)
                warm_lengths = sorted({min(plen, _max_seq_len - 1) for plen in [16, 256]})
                for plen in warm_lengths:
                    dummy = torch.randint(4, config.vocab_size, (1, plen), device=_device)
                    logits = eng.prefill(dummy)
                    tok = eng.sample_first(logits, warm_params)
                    for _ in range(4):
                        tok = eng.step(tok, warm_params)
                    for _ in range(2 if _warm_truncated else 0):
                        # Warm the truncated (top-k/top-p) sampler graph too —
                        # a cold compile on the first truncated request would
                        # stall long enough to flap gateway health.
                        tok = eng.step(tok, trunc_params)
                    eng.step_logits(tok)  # warm the greedy/forward variant too
                if prefix_cache and _max_seq_len > 112:
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
                elif prefix_cache:
                    logger.warning("Prefix-cache warmup skipped: max_seq_len=%d is too small", _max_seq_len)
                torch.cuda.synchronize(_device)
            eng.reset()
            return eng

        def _build_and_warm_pool(eng: "DecodeEngine", n_tokens: int, block: int) -> "PrefixKVCache":
            pool = PrefixKVCache(eng, n_tokens, block_size=block)
            # A pool hit's suffix (the prompt's last partial block plus whatever differs, often one short turn)
            # extends in one block replay.
            eng.extend_single_chunk = True
            if not prefix_cache and _max_seq_len > 112:
                # The pool extends through prefill_cached; warm its eager extend like --prefix-cache does.
                base_ids = torch.randint(4, config.vocab_size, (1, 64), device=_device)
                eng.prefill(base_ids)
                eng.prefill_cached(torch.cat([base_ids, torch.randint(4, config.vocab_size, (1, 48),
                                                                      device=_device)], dim=1))
            with torch.inference_mode():
                # One store and one load (pool hit onto another prompt in the engine) warm the gather/scatter
                # kernels; the pool is emptied afterwards.
                n = min(4 * block + 8, _max_seq_len - 1)
                ids = torch.randint(4, config.vocab_size, (1, n), device=_device)
                pool.prefill(ids)
                eng.prefill(torch.randint(4, config.vocab_size, (1, 32), device=_device))
                pool.prefill(ids)
                torch.cuda.synchronize(_device)
            pool.clear()
            pool.stats.update({k: 0 for k in pool.stats})
            eng.reset()
            return pool

        # Build/compile/capture on the SAME single worker thread that serves all
        # generation (_generate_executor). max-autotune uses inductor's
        # cudagraph_trees, whose tree-manager containers live in THREAD-LOCAL
        # storage: a graph captured here on the main thread asserts
        # (torch._C._is_key_in_tls) the moment the worker thread replays it
        # (a 28ms 500 on every non-stream request of a warmed-at-startup
        # replica). load_model is sync (called
        # from lifespan before serving starts), so block on the future.
        _engine = _generate_executor.submit(_build_and_warm_engine).result()
        # One decoder per row width (--multi_rows and --xbatch_rows together); the narrower ones view the widest
        # one's KV storage, since only one decoder holds live rows at a time (the engine lock).
        decoders: dict[int, MultiSampleDecoder] = {}
        for rows in sorted(set(multi_rows) | set(xbatch_rows), reverse=True):
            # Same thread as the engine: its cudagraph trees are thread-local too.
            widest = decoders[max(decoders)] if decoders else None
            decoders[rows] = _generate_executor.submit(
                build_and_warm_row_decoder, _engine, rows, widest,
                (cascade_groups, cascade_tail_tokens) if cascade_attn else None).result()
        _multi = {r: decoders[r] for r in sorted(set(multi_rows))}
        if cascade_attn and decoders:
            widest = decoders[max(decoders)]
            assert isinstance(widest, CascadeDecoder)
            _row_token_cap = widest.tail_tokens
            _cascade_info = {"groups": {r: d.groups for r, d in sorted(decoders.items())},
                             "group_rows": {r: d.group_rows for r, d in sorted(decoders.items())},
                             "tail_tokens": widest.tail_tokens, "attention": widest.attend_backend,
                             "kv_gb": round(widest.kv_bytes() / 1e9, 2)}
            logger.info("Cascade attention ENABLED: %s", _cascade_info)
        if _multi:
            logger.info("Batched multi-sample decode ENABLED: row buckets %s", sorted(_multi))
        if xbatch_rows:
            runner = XBatchRunner({r: decoders[r] for r in sorted(set(xbatch_rows))}, prefill=_engine_prefill,
                                  set_steer=_engine.set_steer,
                                  decode=lambda ids: _tokenizer.decode(ids, skip_special_tokens=True),
                                  stop_ids=_stop_token_ids, max_seq_len=_max_seq_len, device=_device,
                                  refill=xbatch_refill, should_yield=_engine_lock_contended,
                                  levers=_engine.levers)
            _xbatch = XBatchScheduler(runner, _engine_lock, _run_generation)
            logger.info("Cross-request batched decode ENABLED: row buckets %s, refill %s", sorted(set(xbatch_rows)),
                        "on" if xbatch_refill else "off")
        # Disable cuDNN SDPA only AFTER the engine compiled: the captured
        # decode graph keeps its baked cuDNN flash kernels, while the eager
        # FALLBACK path (non-engine requests) avoids cuDNN's ~300ms per-shape
        # plan-selection stalls by dispatching to pytorch flash instead.
        torch.backends.cuda.enable_cudnn_sdp(False)
        _prefix_cache = prefix_cache or prefix_pool_tokens > 0  # the pool extends through prefill_cached
        if _prefix_cache:
            logger.info("Prefix caching ENABLED (engine extend-from-pos prefill)")
        if prefix_pool_tokens > 0:
            _prefix_pool = _generate_executor.submit(_build_and_warm_pool, _engine, prefix_pool_tokens,
                                                     prefix_pool_block).result()
            logger.info("Prefix KV pool ENABLED: %d blocks of %d tokens", _prefix_pool.n_blocks, prefix_pool_block)
        logger.info("Fast decode engine ready in %.1fs (compile store: %s; %s)", time.time() - t0,
                    "hit" if store_hit else ("miss" if store_key else "off"), compile_cache.cache_counters())
        if store_key and not store_hit:
            compile_cache.save(compile_store, store_key, store_fields)
        if store_claim is not None:
            store_claim.release()
    elif _device.type == "cuda" and warmup_sdpa:
        # Reference path only: prime cuDNN plan cache so per-shape ~300ms
        # stalls don't land on live requests. ~100s; opt-in for debug use.
        _warmup_sdpa_cache(model, _max_seq_len, step=64)

    # Run a few prefills at different lengths to warm any remaining caches
    if _device.type == "cuda":
        logger.info("Warming up prefill path...")
        with torch.inference_mode():
            for plen in sorted({min(plen, _max_seq_len - 1) for plen in [1, 16, 128, 512]}):
                dummy = torch.zeros(1, plen, dtype=torch.long, device=_device)
                model(dummy, use_cache=True)
            torch.cuda.synchronize(_device)
        logger.info("Prefill warmup done")

        # Release warmup workspace back to the CUDA allocator
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated(_device) / 1e9
        reserved = torch.cuda.memory_reserved(_device) / 1e9
        logger.info("Post-warmup memory: %.3f GB allocated, %.3f GB reserved", alloc, reserved)


    return model, tokenizer


# ── Request/response schemas ────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class SteerVectorSpec(BaseModel):
    """One member of a per-request steering stack ({"name": alias-or-raw-npz-key,
    "alpha": dose}). alpha omitted -> the
    alias default (or STEER_DEFAULT_ALPHA for raw keys)."""

    name: str
    alpha: float | None = None


class InjectionSpec(BaseModel):
    """A raw steering write (--levers): ``alpha * vectors[j]`` at effective layer ``sites[j]``, the vectors already
    carrying their dose. ``kind`` "attn" writes the attention output (block-persistent by ``rule``), "mlp" the MLP
    output at the site only (rule S). ``kind`` "act" moves MLP neurons toward fixed values instead: entry j is
    neuron ``neurons[j]`` at effective layer ``sites[j]``, whose activation (the input of down_proj) becomes
    ``act + alpha * (mu[j] - act)`` (alpha 1 replaces it with ``mu[j]``); it takes no vectors."""

    sites: list[int]
    vectors: list[list[float]] = Field(default_factory=list)
    alpha: float = 1.0
    rule: Literal["K", "N", "S"] = "K"
    kind: Literal["attn", "mlp", "act"] = "attn"
    neurons: list[int] = Field(default_factory=list)
    mu: list[float] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # kept so _refuse_unserved can see them
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    # The prompt as token ids, used as sent (the client's own tokenization; no text round trip).
    prompt_ids: list[int] | None = None
    # Seeded candidates (--seeded_rows): candidate i of a request draws from its own noise stream (this seed and i),
    # whatever else shares its batch. Pure-temperature laws on the batched decoders.
    seed: int | None = Field(default=None, ge=0)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    # Keep the tokens whose probability is at least min_p times the most likely token's (0 = off).
    min_p: float = Field(default=0.0, ge=0.0, lt=1.0)
    repetition_penalty: float = Field(
        default_factory=lambda: _default_repetition_penalty, ge=1.0, le=2.0)
    stop_strings: list[str] = Field(default_factory=list)
    stream: bool = False
    # Which tokens the repetition penalty counts: "generated" (default) the completion's own tokens, "row" the
    # prompt's too (the CTRL / Hugging Face convention).
    penalty_scope: Literal["generated", "row"] = "generated"
    # ── Residual steering (fast engine + --steer-npz only; else -> 400). ──
    # 'vectors' composes a weighted stack — each member scaled by alpha * ITS
    # OWN site's median norm; members may live at DIFFERENT sites (v2 multi-
    # site). 'steer_model' (+ optional 'steer_alpha') is single-vector use.
    vectors: list[SteerVectorSpec] | None = None
    steer_alpha: float | None = None
    steer_model: str | None = None
    # What lands at the later block entries: K = the site row unchanged (default), N = scaled to the partial norm
    # it lands in (needs partial_norm_l* in a bank), S = a single write at the site.
    steer_rule: Literal["K", "N", "S"] = "K"
    # Which positions the write reaches: "decode" = generated tokens only (default), "uniform" = the prompt too
    # (needs --levers).
    steer_span: Literal["decode", "uniform"] = "decode"
    # A raw write instead of bank vectors (needs --levers).
    injection: InjectionSpec | None = None


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    timing: dict[str, float] | None = None
    # exact token IDs for offline replay (e.g. signature capture) —
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
    max_seq_len: int
    fast_engine: bool
    prefix_cache: bool = False
    steering: dict | None = None
    default_repetition_penalty: float | None = None
    xbatch: dict | None = None
    prefix_pool: dict | None = None
    cascade: dict | None = None
    startup: dict | None = None
    adapter: dict | None = None


# ── /logprobs: batched prefill-only scoring (the four-register rider battery) ──
# One forward per BATCH, no decode loop (the computation lives in
# kotodama.eval.riders.score, shared with the training loop's in-loop riders).
# This is the fast path for logprob evals; the decode engine is irrelevant here
# because nothing is generated.

from kotodama.eval.riders.score import (  # noqa: E402
    LOGPROB_MAX_BATCH,
    LOGPROB_MAX_TOTAL_TOKENS,
    LogprobLimitError,
    score_logprob_batch,
    score_token_nll_batch,
)


class LogprobItem(BaseModel):
    id: str
    prompt: str
    candidates: list[str] = Field(..., min_length=1, max_length=64)


class LogprobRequest(BaseModel):
    items: list[LogprobItem] = Field(..., min_length=1, max_length=LOGPROB_MAX_BATCH)
    top_n: int = Field(default=5, ge=0, le=50)


@torch.inference_mode()
def _score_logprobs(req: LogprobRequest) -> dict[str, Any]:
    try:
        results = score_logprob_batch(
            lambda batch: _model(batch)["logits"], _tokenizer, req.items, req.top_n,
            _device, max_seq_len=_max_seq_len, max_total_tokens=LOGPROB_MAX_TOTAL_TOKENS)
    except LogprobLimitError as e:
        raise HTTPException(400, str(e)) from e
    return {"results": results, "model": _model_name()}


class TokenScoreItem(BaseModel):
    id: str
    prompt_ids: list[int] = Field(..., min_length=1)
    completion_ids: list[int] = Field(default_factory=list)


class TokenScoreRequest(BaseModel):
    items: list[TokenScoreItem] = Field(..., min_length=1, max_length=LOGPROB_MAX_BATCH)


@torch.inference_mode()
def _score_tokens(req: TokenScoreRequest) -> dict[str, Any]:
    try:
        results = score_token_nll_batch(
            lambda batch: _model(batch)["logits"], req.items, _device,
            vocab_size=_model.config.vocab_size, max_seq_len=_max_seq_len,
            max_total_tokens=LOGPROB_MAX_TOTAL_TOKENS)
    except LogprobLimitError as e:
        raise HTTPException(400, str(e)) from e
    return {"results": results, "model": _model_name()}


def _prompt_ids(request: GenerateRequest, tokenizer: AutoTokenizer) -> list[int]:
    """The request's prompt as token ids: ``prompt_ids`` as sent (the client's own tokenization, never re-encoded:
    a text round trip can split a boundary differently), else the prompt text or chat messages, encoded once."""
    if request.prompt_ids is not None:
        if request.prompt is not None or request.messages is not None:
            raise HTTPException(400, "Pass one of 'prompt_ids', 'prompt' or 'messages'")
        ids = list(request.prompt_ids)
        vocab = int(_model.config.vocab_size) if _model is not None else None
        if not ids or (vocab is not None and not all(0 <= t < vocab for t in ids)):
            raise HTTPException(400, f"prompt_ids must be a non-empty list of ids in [0, {vocab})")
        return ids
    return tokenizer.encode(_resolve_prompt(request, tokenizer))


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


# ── Steering composition ────────────────────────────────────────────────────────

def _engine_params(request: GenerateRequest) -> "EngineSamplingParams":
    """The request's sampling law for the engine and the row decoders."""
    return EngineSamplingParams(temperature=request.temperature, repetition_penalty=request.repetition_penalty,
                                top_k=request.top_k, top_p=request.top_p, penalty_scope=request.penalty_scope,
                                min_p=request.min_p)


def _penalty_ids(request: GenerateRequest, input_ids: "torch.Tensor", generated_ids: list[int]) -> list[int]:
    """The tokens the repetition penalty counts on the reference path (``penalty_scope``)."""
    return input_ids[0].tolist() + generated_ids if request.penalty_scope == "row" else generated_ids


def _request_wants_steering(request: GenerateRequest) -> bool:
    return (bool(request.vectors) or request.steer_model is not None or request.steer_alpha is not None
            or request.injection is not None or request.steer_span != "decode")


def _resolve_steer_member(name: str, alpha: float | None) -> tuple[str | None, float]:
    """Alias-or-raw-key -> (bank key or None for control, effective alpha)."""
    if name in _steer_aliases:
        key, default_alpha, _desc = _steer_aliases[name]
        if key is None:  # "control" — explicitly unsteered
            return None, 0.0
    else:
        key, default_alpha = name, STEER_DEFAULT_ALPHA
    if key not in _steer_bank:
        raise HTTPException(
            400,
            f"Unknown steering vector {name!r}; aliases {sorted(_steer_aliases)}, "
            f"raw keys {sorted(_steer_bank)}",
        )
    eff = alpha if alpha is not None else default_alpha
    if not math.isfinite(eff):
        raise HTTPException(400, f"Non-finite alpha for steering vector {name!r}")
    return key, float(eff)


def _steer_rule_kwargs(request: GenerateRequest) -> dict[str, Any]:
    """set_steer's rule arguments for a request; rule N without a partial-norm table is a 400."""
    if request.steer_rule == "N" and not _steer_partial_norms:
        raise HTTPException(400, "steer_rule N needs per-layer partial norms (partial_norm_l* keys) in --steer-npz")
    return {"rule": request.steer_rule, "norms": _steer_partial_norms or None}


def _request_lever(request: GenerateRequest) -> "LeverPlan | None":
    """A request's steering as one lever plan (--levers), or None when it is unsteered. Bank vectors resolve with
    the same dose math as ``_compose_steer_writes``; an ``injection`` is taken as given. 400 on anything the plan
    refuses."""
    assert _engine is not None and _engine.levers is not None
    from kotodama.model.levers import LeverError

    cfg = _engine.config
    boundaries = list(_engine._boundaries)
    try:
        if request.injection is not None:
            if request.vectors or request.steer_model is not None or request.steer_alpha is not None:
                raise HTTPException(400, "Pass either 'injection' or bank vectors, not both")
            inj = request.injection
            if inj.rule == "N" and not _steer_partial_norms:
                raise HTTPException(400, "injection rule N needs per-layer partial norms (partial_norm_l* keys)")
            if inj.kind == "act":
                return act_injection_to_plan(inj.sites, inj.neurons, inj.mu, inj.alpha, inj.vectors,
                                             request.steer_span, cfg.num_layers, cfg.intermediate_size)
            return injection_to_plan(inj.sites, inj.vectors, inj.alpha, inj.rule, inj.kind, request.steer_span,
                                     _steer_partial_norms or None, boundaries, cfg.num_layers, cfg.hidden_size)
        writes = _compose_steer_writes(request) if _request_wants_steering(request) and (
            request.vectors or request.steer_model is not None or request.steer_alpha is not None) else None
        if writes is None:
            return None
        kw = _steer_rule_kwargs(request)
        return writes_to_plan(writes, kw["rule"], kw["norms"], boundaries, cfg.num_layers, request.steer_span)
    except LeverError as exc:
        raise HTTPException(400, f"Steering request refused: {exc}") from None


def _check_span(request: GenerateRequest) -> None:
    if request.steer_span != "decode" and not _levers:
        raise HTTPException(400, "steer_span 'uniform' needs a server started with --levers")
    if request.injection is not None and not _levers:
        raise HTTPException(400, "'injection' needs a server started with --levers")
    if request.injection is not None and _adapter is not None:
        # Injection doses (and the act lever's mu) are measured on the base weights, which the merged adapter moves.
        raise HTTPException(400, f"'injection' is calibrated on the base model; not served with an adapter merged "
                                 f"({_adapter['name']})")


@contextlib.contextmanager
def _steering(request: GenerateRequest, timing: dict[str, float]):
    """Engage a request's steering for the block and yield its lever slot (0: none). Without --levers this is the
    broadcast set_steer path (slot 0); with it, the request binds a lever slot, released on exit. Everything that
    can 400 resolves before anything is engaged. Run on the generation thread."""
    assert _engine is not None
    _check_span(request)
    if not _levers:
        steer_writes = _compose_steer_writes(request)
        kwargs = _steer_rule_kwargs(request) if steer_writes is not None else {}
        try:
            if steer_writes is not None:
                _engine.set_steer(steer_writes, **kwargs)
                timing["steered"] = 1.0
                timing["steer_writes"] = float(len(steer_writes))
            yield 0
        finally:
            # ALWAYS clear (no-op on a steering-disabled engine): steering state must never leak into the next
            # request.
            _engine.set_steer(None)
        return
    plan = _request_lever(request)
    if plan is None:
        yield 0
        return
    if not _engine.levers.free:
        raise HTTPException(503, "every lever slot is in use; retry")
    slot = _engine.levers.bind(plan)
    timing["steered"] = 1.0
    timing["lever_slot"] = float(slot)
    try:
        yield slot
    finally:
        _engine.levers.release(slot)


def _compose_steer_writes(request: GenerateRequest) -> "list[tuple[int, torch.Tensor, float]] | None":
    """Resolve a request's steering fields into an engine WRITES list.

    Returns None when the request is unsteered (no fields, or an explicit
    'control'), else [(site_layer, unit_vec, alpha * that site's median
    norm), ...] — one write per resolved stack member; members at the same
    site simply sum row-wise inside engine.set_steer. Dose math per member is
    alpha * median_norm * unit, with each member scaled by ITS OWN site's
    median norm. Raises HTTPException(400)
    for steering use without --steer-npz, unknown names, or conflicting
    fields.
    """
    if not _request_wants_steering(request):
        return None
    if not _steer_loaded:
        raise HTTPException(
            400,
            "Steering fields (vectors/steer_model/steer_alpha) require a server "
            "started with --steer-npz",
        )
    if request.vectors:
        if request.steer_model is not None or request.steer_alpha is not None:
            raise HTTPException(400, "Pass either 'vectors' or steer_model/steer_alpha, not both")
        writes: list[tuple[int, torch.Tensor, float]] = []
        for item in request.vectors:
            key, alpha = _resolve_steer_member(item.name, item.alpha)
            if key is None:
                continue
            site_layer, unit = _steer_bank[key]
            writes.append((site_layer, unit, alpha * _steer_meds[site_layer]))
        return writes or None  # all-control stack -> None -> unsteered
    if request.steer_model is None:
        raise HTTPException(400, "steer_alpha requires steer_model (or use 'vectors')")
    key, alpha = _resolve_steer_member(request.steer_model, request.steer_alpha)
    if key is None:
        return None
    site_layer, unit = _steer_bank[key]
    return [(site_layer, unit, alpha * _steer_meds[site_layer])]


# ── Sampling ────────────────────────────────────────────────────────────────────

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    generated_ids: list[int],
    min_p: float = 0.0,
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

    if min_p > 0.0:
        logits = logits.masked_fill(logits < logits.max() + math.log(min_p), float("-inf"))

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ── Model forward (fast or standard) ───────────────────────────────────────────

@torch.inference_mode()
def model_forward(model: KotodamaModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Plain eager forward, returning logits (reference/debug path)."""
    return model(input_ids)["logits"]


# ── Generation ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate(
    model: KotodamaModel,
    tokenizer: AutoTokenizer,
    request: GenerateRequest,
) -> GenerateResponse:
    timing: dict[str, float] = {}

    # The reference path has no injection hook — steering is engine-only.
    # (With --steer-npz loaded the engine always exists, so this only fires
    # for steering fields on an engine-less server.)
    if _request_wants_steering(request):
        raise HTTPException(400, "Steering requires the fast engine (--engine fast with --steer-npz)")

    # Tokenization
    t_tok = time.perf_counter()
    input_ids = torch.tensor([_prompt_ids(request, tokenizer)], device=_device)
    prompt_len = input_ids.shape[1]
    timing["tokenize_ms"] = (time.perf_counter() - t_tok) * 1000

    if prompt_len >= _max_seq_len:
        raise HTTPException(400, f"Prompt too long: {prompt_len} tokens (max {_max_seq_len})")

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
        request.repetition_penalty, _penalty_ids(request, input_ids, generated_ids), request.min_p,
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
        if prompt_len + len(generated_ids) >= _max_seq_len:
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
            request.repetition_penalty, _penalty_ids(request, input_ids, generated_ids), request.min_p,
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
    compiled sampler variant rather than the reference path (which costs ~10x
    in decode).
    """
    return _engine is not None


@torch.inference_mode()
def engine_generate(request: GenerateRequest) -> GenerateResponse:
    """Non-streaming generation on the fast engine. Caller must hold _engine_lock."""
    assert _engine is not None and _tokenizer is not None and _model is not None
    timing: dict[str, float] = {}
    max_pos = _max_seq_len

    t_tok = time.perf_counter()
    input_ids = torch.tensor([_prompt_ids(request, _tokenizer)], device=_device)
    prompt_len = input_ids.shape[1]
    timing["tokenize_ms"] = (time.perf_counter() - t_tok) * 1000
    if prompt_len >= max_pos:
        raise HTTPException(400, f"Prompt too long: {prompt_len} tokens (max {max_pos})")

    params = _engine_params(request)
    t0 = time.perf_counter()
    with _steering(request, timing) as slot:
        t_prefill = time.perf_counter()
        prefill_logits, pc_info = _engine_prefill(input_ids, slot)
        tok = _engine.sample_first(prefill_logits, params)
        token_id = int(tok.item())
        timing["prefill_ms"] = (time.perf_counter() - t_prefill) * 1000
        if pc_info is not None:
            timing["prefix_cache_hit"] = pc_info["prefix_hit"]
            timing["prefix_common_tokens"] = pc_info["common_prefix"]
            timing["prefill_suffix_tokens"] = pc_info["suffix_len"]
            if "pool_tokens" in pc_info:
                timing["prefix_pool_tokens"] = pc_info["pool_tokens"]

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
    decoded together and flushed as soon as the text does not end in an incomplete
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
        max_pos = _max_seq_len
        steer_stack = contextlib.ExitStack()
        try:
            # Steering engages on the engine's thread (eager buffer mutation, no cudagraph replay, but the same
            # conservative discipline as every other engine call here); a 400 there is the stream's error.
            try:
                slot = await loop.run_in_executor(_generate_executor, steer_stack.enter_context,
                                                  _steering(request, {}))
            except HTTPException as exc:
                yield f"data: {json.dumps({'error': str(exc.detail)})}\n\n"
                return
            input_ids = torch.tensor([_prompt_ids(request, _tokenizer)], device=_device)
            prompt_len = input_ids.shape[1]
            if prompt_len >= max_pos:
                yield f'data: {{"error": "Prompt too long: {prompt_len} tokens"}}\n\n'
                return

            params = _engine_params(request)
            t_prefill = time.perf_counter()
            prefill_logits, pc_info = await loop.run_in_executor(
                _generate_executor, _engine_prefill, input_ids, slot
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
            # ALWAYS release steering (the broadcast rows, or the request's lever slot) so state never leaks
            # between requests — including the disconnect / GeneratorExit paths. Prefer the engine thread; fall
            # back to a direct call if the loop is tearing down (both are eager buffer mutations, safe off-thread
            # while the engine is idle).
            try:
                await loop.run_in_executor(_generate_executor, steer_stack.close)
            except BaseException:
                # Loop teardown / GeneratorExit / cancellation mid-await:
                # release synchronously, then let the exception propagate.
                steer_stack.close()
                _engine.set_steer(None)
                raise


@torch.inference_mode()
async def generate_stream(
    model: KotodamaModel,
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
            if _request_wants_steering(request):
                # Reference path: no injection hook (see generate()).
                yield f"data: {json.dumps({'error': 'Steering requires the fast engine (--engine fast with --steer-npz)'})}\n\n"
                return
            input_ids = torch.tensor([_prompt_ids(request, tokenizer)], device=_device)
            prompt_len = input_ids.shape[1]

            if prompt_len >= _max_seq_len:
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
                request.repetition_penalty, _penalty_ids(request, input_ids, generated_ids), request.min_p,
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
                if prompt_len + len(generated_ids) >= _max_seq_len:
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
                    request.repetition_penalty, _penalty_ids(request, input_ids, generated_ids), request.min_p,
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
_max_seq_len_arg: int | None = None
_prefix_cache_arg = False
_warmup_sdpa_arg = False
_steer_npz_arg: str | None = None
_multi_rows_arg: tuple[int, ...] = ()
_xbatch_rows_arg: tuple[int, ...] = ()
_xbatch_refill_arg = False
_prefix_pool_tokens_arg = 0
_prefix_pool_block_arg = 16
_cascade_attn_arg = False
_cascade_groups_arg = 0
_cascade_tail_tokens_arg = 1024
_levers_arg = 0
_seeded_rows_arg = False
_compile_store_arg: str | None = None
_compile_mode_arg = "max-autotune"
_warm_truncated_arg = True
_regional_arg = True
_startup_budget_s_arg = 1800.0
_autotune_min_rows_arg = 0


def _record_startup() -> None:
    """Seconds from process start to ready, against --startup_budget_s (a start over budget is logged loudly)."""
    global _startup_info
    seconds = time.time() - _PROCESS_T0
    _startup_info = {"seconds": round(seconds, 1), "budget_s": _startup_budget_s_arg, "regional": _regional,
                     "compile_mode": _compile_mode, "autotune_min_rows": _autotune_min_rows,
                     "compile_store": _compile_store_state}
    if _startup_budget_s_arg and seconds > _startup_budget_s_arg:
        logger.warning("STARTUP OVER BUDGET: ready after %.0fs, budget %.0fs (%s). A start this slow usually means "
                       "the compiled graphs multiplied: see the compile log above, or start with --regional",
                       seconds, _startup_budget_s_arg, _startup_info)
    else:
        logger.info("startup: ready after %.0fs (budget %.0fs; %s)", seconds, _startup_budget_s_arg, _startup_info)
_adapter_arg: str | None = None  # CLI --adapter: a LoRA adapter directory merged into the weights at load


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer
    _model, _tokenizer = load_model(
        _checkpoint_path, _device_arg, mode=_mode_arg,
        model_size=_model_size_arg, engine=_engine_arg, prefix_cache=_prefix_cache_arg,
        warmup_sdpa=_warmup_sdpa_arg,
        steer_npz=_steer_npz_arg,
        max_seq_len=_max_seq_len_arg,
        multi_rows=_multi_rows_arg,
        xbatch_rows=_xbatch_rows_arg,
        xbatch_refill=_xbatch_refill_arg,
        prefix_pool_tokens=_prefix_pool_tokens_arg,
        prefix_pool_block=_prefix_pool_block_arg,
        cascade_attn=_cascade_attn_arg,
        cascade_groups=_cascade_groups_arg,
        cascade_tail_tokens=_cascade_tail_tokens_arg,
        levers=_levers_arg,
        seeded_rows=_seeded_rows_arg,
        compile_store=_compile_store_arg,
        compile_mode=_compile_mode_arg,
        warm_truncated=_warm_truncated_arg,
        regional=_regional_arg,
        autotune_min_rows=_autotune_min_rows_arg,
        adapter=_adapter_arg,
    )
    _record_startup()
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
    (--model_size) + serving mode, plus ``+<adapter>`` when --adapter merged one.
    """
    if _served_name_override:
        return _served_name_override
    size_label = SIZE_LABELS.get(_model_size_arg, _model_size_arg)
    mode_label = "instruct" if _serve_mode == "chat" else "base"
    adapter_label = f"+{Path(_adapter_arg).name.removesuffix('.adapter')}" if _adapter_arg else ""
    return f"kotodama-{size_label}-{mode_label}{adapter_label}"


app = FastAPI(title="kotodama", lifespan=lifespan)
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
    steering = None
    if _steer_loaded:
        sites = sorted({s for s, _ in _steer_bank.values()})
        steering = {
            "sites": sites,  # LAYER indices; npz suffix sublayer = 2*layer
            "sublayers": [2 * s for s in sites],
            "median_norms": {str(layer): v for layer, v in sorted(_steer_meds.items())},
            "aliases": {
                a: {"key": k, "alpha": al, "desc": d,
                    "site_layer": _steer_bank[k][0] if k is not None else None}
                for a, (k, al, d) in sorted(_steer_aliases.items())
            },
            "keys": {k: {"site_layer": s} for k, (s, _) in sorted(_steer_bank.items())},
            "note": "block-persistent injection (site row + later block entries), "
                    "decode/generated positions only; multi-site stacks compose row-wise",
        }
    if _levers and _engine is not None and _engine.levers is not None:
        steering = dict(steering or {})
        steering["levers"] = {"slots": _levers, "free": _engine.levers.free, "spans": ["decode", "uniform"],
                              "sites": ["attn", "mlp"], "keyed_on": "effective layer",
                              "note": "per-request levers: each steered request binds its own slot"}
    return ModelInfo(
        name=_model_name(),
        mode=_serve_mode,
        params=_model.config.param_count(),
        config=asdict(_model.config),
        device=str(_device),
        checkpoint=_checkpoint_path,
        max_seq_len=_max_seq_len,
        fast_engine=_engine is not None,
        prefix_cache=_prefix_cache,
        steering=steering,
        default_repetition_penalty=_default_repetition_penalty,
        xbatch=None if _xbatch is None else {"rows": sorted(_xbatch.runner.decoders),
                                             "refill": _xbatch.runner.refill,
                                             "stats": dict(_xbatch.runner.stats)},
        prefix_pool=None if _prefix_pool is None else {"blocks": _prefix_pool.n_blocks,
                                                       "block_size": _prefix_pool.block_size,
                                                       "cached_blocks": len(_prefix_pool.index),
                                                       "stats": dict(_prefix_pool.stats)},
        cascade=_cascade_info,
        startup=_startup_info,
        adapter=_adapter,
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


@app.post("/logprobs")
async def logprobs_endpoint(request: LogprobRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded")
    # Same dedicated thread as generation: the GPU is single-tenant per replica
    # and eager batch prefill must not interleave with engine activity.
    return await _run_generation(_score_logprobs, request)


@app.post("/score_tokens")
async def score_tokens_endpoint(request: TokenScoreRequest):
    """Per-token NLL of exact token-id continuations under the plain model
    (temperature 1, no penalty, no truncation) — the coherence read for
    sampled text (ids as returned by /generate's completion_ids)."""
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded")
    return await _run_generation(_score_tokens, request)


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest, http_request: Request):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded")
    _refuse_unserved(request)

    if request.seed is not None:  # seeded draws come from the batched decoders only
        if request.stream:
            raise HTTPException(400, "a seeded request cannot stream")
        out = await _batched_candidates(request, 1)
        return out[0]

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

# Request fields that only route or label a request (the gateway's affinity key): accepted, and they change nothing
# that is generated.
_PASSTHROUGH_FIELDS = frozenset({"user"})


def _refuse_unserved(request: BaseModel) -> None:
    """400 on a field this server would otherwise accept and ignore: an unknown field that is on (any value but
    null / false / 0 / empty), a nonzero presence_penalty (the repetition law is multiplicative: frequency_penalty
    maps to repetition_penalty), best_of above 1. A client's law must never silently differ from what it sent."""
    bad = sorted(k for k, v in (request.model_extra or {}).items()
                 if k not in _PASSTHROUGH_FIELDS and v not in (False, 0, "", [], {}))
    if getattr(request, "presence_penalty", 0.0):
        bad.append("presence_penalty")
    if (getattr(request, "best_of", None) or 1) > 1:
        bad.append("best_of")
    if bad:
        raise HTTPException(400, f"not served on this endpoint (the server would otherwise ignore them): {bad}")


def _drop_nulls(data: Any) -> Any:
    """OpenAI clients (e.g. Loom) send unset sampling fields as JSON null; treat null as "not sent" so the field's
    default applies instead of a 422."""
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if v is not None}
    return data


class OAICompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # kept so _refuse_unserved can see them
    # Text, or the OpenAI token form: a list of token ids (sent as the prompt's ids, never re-encoded).
    prompt: str | list[int]
    model: str = ""
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    n: int = Field(default=1, ge=1, le=64)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    best_of: int | None = None
    stop: list[str] | str | None = None
    return_token_ids: bool = False
    # Residual steering, as on /generate (needs the fast engine + --steer-npz; else 400).
    vectors: list[SteerVectorSpec] | None = None
    steer_model: str | None = None
    steer_alpha: float | None = None
    steer_rule: Literal["K", "N", "S"] = "K"
    steer_span: Literal["decode", "uniform"] = "decode"
    injection: InjectionSpec | None = None
    penalty_scope: Literal["generated", "row"] = "generated"
    seed: int | None = Field(default=None, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="before")
    @classmethod
    def _tolerant(cls, data: Any) -> Any:
        data = _drop_nulls(data)
        if isinstance(data, dict) and isinstance(data.get("prompt"), list):
            p = data["prompt"]
            if p and all(isinstance(t, int) and not isinstance(t, bool) for t in p):
                return data  # token ids
            if len(p) == 1 and isinstance(p[0], list) and p[0] and all(isinstance(t, int) for t in p[0]):
                return {**data, "prompt": p[0]}  # one token-id prompt in a batch of one
            if len(p) != 1 or not isinstance(p[0], str):
                raise ValueError("prompt: send one string or one list of token ids (several prompts are not "
                                 "supported; use n)")
            data = {**data, "prompt": p[0]}
        return data


@torch.inference_mode()
def engine_generate_multi(request: GenerateRequest, n: int) -> list[GenerateResponse]:
    """n candidates of one prompt: one engine prefill (prefix cache included), then the batched decoder. Same
    tokenization, steering, stop and capacity rules as engine_generate per candidate. Caller must hold _engine_lock."""
    dec = _multi_bucket(n)
    assert _engine is not None and dec is not None and _tokenizer is not None
    timing: dict[str, float] = {}
    input_ids = torch.tensor([_prompt_ids(request, _tokenizer)], device=_device)
    prompt_len = input_ids.shape[1]
    if prompt_len >= _max_seq_len:
        raise HTTPException(400, f"Prompt too long: {prompt_len} tokens (max {_max_seq_len})")
    params = _engine_params(request)
    stops = request.stop_strings

    def stop_check(ids: list[int]) -> bool:
        return bool(stops) and any(s in _tokenizer.decode(ids, skip_special_tokens=True) for s in stops)

    t0 = time.perf_counter()
    with _steering(request, timing) as slot:
        prefill_logits, pc_info = _engine_prefill(input_ids, slot)
        timing["prefill_ms"] = (time.perf_counter() - t0) * 1000
        if pc_info is not None:
            timing["prefix_cache_hit"] = pc_info["prefix_hit"]
            timing["prefix_common_tokens"] = pc_info["common_prefix"]
        budget = min(request.max_new_tokens, _max_seq_len - prompt_len)
        rows = decode_rows(dec, prefill_logits, params, n, budget, _stop_token_ids, stop_check,
                           slots=slot if _levers else None, seed=request.seed)
    elapsed = time.perf_counter() - t0
    timing["total_ms"] = round(elapsed * 1000, 2)
    timing["batched_rows"] = float(dec.rows)
    total = sum(len(r) for r in rows)
    tps = round(total / elapsed, 1) if elapsed > 0 else 0.0
    prompt_ids = input_ids[0].tolist()
    return [GenerateResponse(text=_tokenizer.decode(r, skip_special_tokens=True), prompt_tokens=prompt_len,
                             completion_tokens=len(r), tokens_per_second=tps, timing=dict(timing),
                             prompt_ids=prompt_ids, completion_ids=list(r)) for r in rows]


def _multi_bucket(n: int) -> "MultiSampleDecoder | None":
    """The smallest row bucket that holds n candidates, or None."""
    fits = [r for r in _multi if r >= n]
    return _multi[min(fits)] if fits else None


def build_cascade_decoder(eng: "DecodeEngine", rows: int, groups: int, tail_tokens: int,
                          kv_from: "MultiSampleDecoder | None" = None) -> "CascadeDecoder":
    """A cascade row decoder of ``rows`` rows. ``groups`` (0 = rows // 4) is the widest bucket's group count; a
    narrower bucket viewing it (``kv_from``) keeps the same rows per group, within the widest one's groups."""
    if kv_from is not None:
        assert isinstance(kv_from, CascadeDecoder)
        groups = min(kv_from.groups, max(1, kv_from.groups * rows // kv_from.rows))
    return CascadeDecoder(eng, rows, groups=groups, tail_tokens=tail_tokens, kv_from=kv_from, seeded=_seeded_rows)


def _without_autotune(mode: str) -> str:
    """``mode`` without its kernel benchmarking (its CUDA-graph choice kept)."""
    if mode.startswith("max-autotune"):
        return "reduce-overhead" if mode == "max-autotune" else "default"
    return mode


def _row_compile_mode(rows: int) -> str:
    """The compile mode of a ``rows``-row decoder: the server's, without kernel benchmarking below
    --autotune_min_rows."""
    return _without_autotune(_compile_mode) if rows < _autotune_min_rows else _compile_mode


def build_and_warm_row_decoder(eng: "DecodeEngine", rows: int, kv_from: "MultiSampleDecoder | None" = None,
                               cascade: tuple[int, int] | None = None) -> "MultiSampleDecoder":
    """A compiled, warmed row decoder over ``eng`` (call on the generation thread: its CUDA graphs are thread-local).
    ``cascade`` = (groups, tail_tokens) builds a ``CascadeDecoder`` (``build_cascade_decoder``), None a per-row one."""
    if cascade is not None:
        dec = build_cascade_decoder(eng, rows, cascade[0], cascade[1], kv_from)
    else:
        dec = MultiSampleDecoder(eng, rows, kv_from=kv_from, seeded=_seeded_rows)
    dec.compile_step(mode=_row_compile_mode(rows), regional=_regional)
    with torch.inference_mode():
        params = EngineSamplingParams(temperature=0.9, repetition_penalty=1.2)
        trunc = EngineSamplingParams(temperature=0.9, repetition_penalty=1.2, top_p=0.9, min_p=0.05)
        for plen in sorted({min(plen, eng.max_seq_len - 8) for plen in [16, 256]}):
            logits = eng.prefill(torch.randint(4, eng.config.vocab_size, (1, plen), device=eng.device))
            for law in (params, trunc) if _warm_truncated else (params,):  # plain (and truncated) step variants
                dec.start(logits, law)
                for _ in range(4):
                    dec.step(law)
        torch.cuda.synchronize(eng.device)
    eng.reset()
    return dec


def _within_row_cap(request: GenerateRequest) -> bool:
    return not _row_token_cap or request.max_new_tokens <= _row_token_cap


def _multi_compatible(request: GenerateRequest, n: int) -> bool:
    return ((n > 1 or request.seed is not None) and _multi_bucket(n) is not None and _within_row_cap(request)
            and MultiSampleDecoder.supports(_engine_params(request)))


def _xbatch_compatible(request: GenerateRequest, n: int) -> bool:
    return (_xbatch is not None and 1 <= n <= _xbatch.runner.max_rows and _within_row_cap(request)
            and MultiSampleDecoder.supports(_engine_params(request)))


def _xbatch_key(request: GenerateRequest, steered: bool) -> tuple:
    """Requests with equal keys may share a batch: the same steering config (with --levers every request carries its
    own lever, so none). Each row carries its own sampling law, so the law is not part of the key."""
    steer = None
    if steered and not _levers:
        steer = (tuple((v.name, v.alpha) for v in request.vectors or ()), request.steer_model, request.steer_alpha,
                 request.steer_rule)
    return (steer,)


async def engine_generate_xbatch(request: GenerateRequest, n: int) -> list[GenerateResponse]:
    """n candidates of one prompt through the cross-request batcher: same tokenization, steering, stop and capacity
    rules as engine_generate per candidate. Bad prompts and steering stacks 400 before the request queues."""
    assert _xbatch is not None and _tokenizer is not None
    t0 = time.perf_counter()
    prompt_ids = _prompt_ids(request, _tokenizer)
    if len(prompt_ids) >= _max_seq_len:
        raise HTTPException(400, f"Prompt too long: {len(prompt_ids)} tokens (max {_max_seq_len})")
    _check_span(request)
    lever = _request_lever(request) if _levers else None
    steer_writes = _compose_steer_writes(request) if not _levers else None
    steer_kwargs = _steer_rule_kwargs(request) if steer_writes is not None else {}
    params = _engine_params(request)
    steered = steer_writes is not None or lever is not None
    job = XBatchJob(key=_xbatch_key(request, steered), prompt_ids=prompt_ids, n=n,
                    max_new_tokens=request.max_new_tokens, params=params, done=lambda _outcome: None,
                    stop_strings=tuple(request.stop_strings), steer_writes=steer_writes, steer_kwargs=steer_kwargs,
                    lever=lever, seed=request.seed)
    result = await _xbatch.submit(job)
    elapsed = time.perf_counter() - t0
    timing = dict(result.timing)
    if steered:
        timing["steered"] = 1.0
    total = sum(len(r) for r in result.rows)
    tps = round(total / elapsed, 1) if elapsed > 0 else 0.0
    return [GenerateResponse(text=_tokenizer.decode(r, skip_special_tokens=True), prompt_tokens=len(prompt_ids),
                             completion_tokens=len(r), tokens_per_second=tps, timing=dict(timing),
                             prompt_ids=list(prompt_ids), completion_ids=list(r)) for r in result.rows]


async def _batched_candidates(request: GenerateRequest, n: int) -> list[GenerateResponse]:
    """All n candidates from one batched path (cross-request batcher, else the n-row decoder), or [] when neither
    takes the request and the caller generates candidates one at a time. A seeded request that no batched path
    takes is a 400: only the seeded row decoders draw from per-candidate seeds."""
    if request.seed is not None:
        if not _seeded_rows:
            raise HTTPException(400, "'seed' needs a server started with --seeded_rows")
        if not (_xbatch_compatible(request, n) or _multi_compatible(request, n)):
            raise HTTPException(400, "'seed' needs a pure-temperature law (temperature > 0, no top-k/top-p) and "
                                     "the batched decoders (--xbatch_rows or --multi_rows)")
    if _xbatch_compatible(request, n):
        return await engine_generate_xbatch(request, n)
    if _multi_compatible(request, n):
        async with _engine_lock:
            return await _run_generation(engine_generate_multi, request, n)
    if _xbatch is not None and MultiSampleDecoder.supports(_engine_params(request)):
        # a law the batcher samples, refused anyway (a candidate longer than the row cap, more candidates than the
        # widest bucket): served one candidate at a time on the single-stream engine
        logger.info("xbatch: n=%d, max_new_tokens=%d served single-stream (row cap %s, widest bucket %d)",
                    n, request.max_new_tokens, _row_token_cap or "none", _xbatch.runner.max_rows)
    return []


def oai_completion_to_generate(request: OAICompletionRequest) -> GenerateRequest:
    """The /generate request one /v1/completions choice runs: sampling, stops and steering carried over."""
    stop_strings: list[str] = []
    if isinstance(request.stop, str):
        stop_strings = [request.stop]
    elif isinstance(request.stop, list):
        stop_strings = request.stop
    ids = request.prompt if isinstance(request.prompt, list) else None
    return GenerateRequest(
        prompt=None if ids is not None else request.prompt,
        prompt_ids=ids,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=resolve_oai_repetition_penalty(request.frequency_penalty),
        stop_strings=stop_strings,
        vectors=request.vectors,
        steer_model=request.steer_model,
        steer_alpha=request.steer_alpha,
        steer_rule=request.steer_rule,
        steer_span=request.steer_span,
        injection=request.injection,
        penalty_scope=request.penalty_scope,
        seed=request.seed,
        min_p=request.min_p,
    )


@app.post("/v1/completions")
async def oai_completions(request: OAICompletionRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, detail="Model not loaded")
    _refuse_unserved(request)

    choices = []
    gen_req = oai_completion_to_generate(request)
    batched = await _batched_candidates(gen_req, request.n)
    for i in range(request.n):
        if batched:
            result = batched[i]
        elif _engine_compatible(gen_req):
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

    prompt_tokens = (len(request.prompt) if isinstance(request.prompt, list)
                     else _tokenizer.encode(request.prompt, return_tensors="pt").shape[1])
    completion_tokens = sum(
        _tokenizer.encode(c["text"], return_tensors="pt").shape[1] for c in choices
    )

    return {
        "id": f"cmpl-kotodama-{int(time.time())}",
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
    # Steering aliases ride in a separate non-standard key rather than as
    # extra data[] entries — the gateway routes on the model field, and fake
    # model ids would be routable.
    active = _model_name()
    resp: dict[str, Any] = {
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
    if _steer_loaded:
        resp["steering"] = {
            "sites": sorted({s for s, _ in _steer_bank.values()}),
            "aliases": sorted(_steer_aliases),
            "keys": sorted(_steer_bank),
        }
    return resp


# ── OpenAI-compatible /v1/chat/completions (chat mode) ────────────────────────


class OAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # kept so _refuse_unserved can see them
    messages: list[ChatMessage]
    model: str = ""
    max_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    stop: list[str] | str | None = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=64)
    return_token_ids: bool = False

    @model_validator(mode="before")
    @classmethod
    def _tolerant(cls, data: Any) -> Any:
        return _drop_nulls(data)


@app.post("/v1/chat/completions")
async def oai_chat_completions(request: OAIChatRequest, http_request: Request):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, detail="Model not loaded")
    if _serve_mode != "chat":
        raise HTTPException(400, detail="/v1/chat/completions requires --mode chat")
    _refuse_unserved(request)

    msgs = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt_text = _tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )

    stop_strings: list[str] = []
    if isinstance(request.stop, str):
        stop_strings = [request.stop]
    elif isinstance(request.stop, list):
        stop_strings = request.stop

    repetition_penalty = resolve_oai_repetition_penalty(request.frequency_penalty)

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
                        "id": f"chatcmpl-kotodama-{int(time.time())}",
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
                        "id": f"chatcmpl-kotodama-{int(time.time())}",
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
    gen_req = GenerateRequest(
        prompt=prompt_text,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=repetition_penalty,
        stop_strings=stop_strings,
    )
    batched = await _batched_candidates(gen_req, request.n)
    for i in range(request.n):
        if batched:
            result = batched[i]
        elif _engine_compatible(gen_req):
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
        "id": f"chatcmpl-kotodama-{int(time.time())}",
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

def build_arg_parser() -> argparse.ArgumentParser:
    """The server's command line (every flag a serve profile may set)."""
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
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Serving context limit (16-4096 for the 3B model). Lower values "
                             "reduce the fast engine's static KV-cache memory; default is 4096")
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=False,
                        help="Reuse the engine KV cache across requests sharing a token-exact "
                             "prompt prefix (multi-turn TTFT win; requires --engine fast and "
                             "per-conversation replica pinning)")
    parser.add_argument("--model_size", choices=list(MODEL_CONFIGS) + [a for a, k in presets.ALIASES.items() if k in MODEL_CONFIGS], default="3b",
                        help="Model architecture size (3b=2.97B, 1b=1.42B ladder, proxy=108M legacy)")
    parser.add_argument("--served_model_name", default=None,
                        help="Override the model id reported by /info and /v1/models "
                             "(default: kotodama-<size>-<base|instruct>)")
    parser.add_argument("--warmup-sdpa", action=argparse.BooleanOptionalAction, default=False,
                        help="Reference engine only: pre-pay the ~100s cuDNN SDPA plan-cache "
                             "sweep at startup instead of ~300ms per novel shape at runtime")
    parser.add_argument("--steer-npz", default=None,
                        help="COMMA-SEPARATED npz banks of unit steering vectors + per-site "
                             "median norms (vector keys like "
                             "'Alg_echo_base_s46' + 'median_norm_s46'; per-key site from the "
                             "_s{sublayer} suffix, sublayer = 2*layer after-attn). Enables "
                             "per-request multi-site residual steering on the fast engine. "
                             "See kotodama/serve/steering.py")
    parser.add_argument("--attn_res", choices=["config", "on", "off"], default="config",
                        help="Override the size-config's attn_res (proxy 2x2 arms: "
                             "baseline/NCA-only checkpoints have no AttnRes params)")
    parser.add_argument("--attn_res_boundaries", default=None,
                        help="Override boundaries, comma ints (proxy DD-v1: 0,3,7,12,21,25)")
    parser.add_argument("--rope_fp32", action=argparse.BooleanOptionalAction, default=False,
                        help="Model config rope_fp32=True (the 1B ladder arms trained with it)")
    parser.add_argument("--default_repetition_penalty", type=float,
                        default=DEFAULT_REPETITION_PENALTY,
                        help="Repetition penalty used when a request does not set one "
                             "(/generate without repetition_penalty, OpenAI routes without "
                             "frequency_penalty). 1.0 = off; range [1.0, 2.0]")
    parser.add_argument("--multi_rows", default="",
                        help="Batched multi-sample decode for OpenAI-route n > 1 (pure-temperature laws): comma row "
                             "buckets, e.g. 4,8 — n candidates decode in lockstep after one prefill on the smallest "
                             "bucket that fits (~0.9 GB of KV per row at 4096 context on the 1B). Empty = off (serial)")
    parser.add_argument("--xbatch_rows", default="",
                        help="Cross-request batched decode for the OpenAI routes (pure-temperature laws): comma row "
                             "buckets, e.g. 16,32,64 — concurrent requests with the same sampling law and steering "
                             "config decode together, each filled from its own prefill, on the narrowest bucket that "
                             "holds their rows. All buckets (and --multi_rows) share the widest one's KV memory. "
                             "Empty = off")
    parser.add_argument("--xbatch_refill", action=argparse.BooleanOptionalAction, default=False,
                        help="With --xbatch_rows: refill rows freed by finished requests from the queue between "
                             "decode steps (continuous batching) instead of waiting for the batch's longest row")
    parser.add_argument("--prefix_pool_tokens", type=int, default=0,
                        help="Multi-entry prefix KV cache: pool this many prompt tokens of KV (fixed-size blocks, "
                             "least recently used evicted) so a prompt sharing a prefix with ANY recently prefilled "
                             "prompt prefills only its suffix (~0.22 MB per token on the 1B at bf16). Implies "
                             "--prefix-cache. 0 = off")
    parser.add_argument("--levers", type=int, default=0,
                        help="Per-request steering: lever tables with this many slots (kotodama.model.levers). Every "
                             "steered request binds its own slot, so steered and unsteered requests batch together, "
                             "a request may steer its prompt (steer_span uniform) or send a raw 'injection'; "
                             "--steer-npz banks resolve into levers. 0 = off (the broadcast steering rows)")
    parser.add_argument("--compile_store", default=os.environ.get("KOTODAMA_COMPILE_STORE") or None,
                        help="Directory of compiled engine artifacts keyed by model code, runtime, GPU and engine flags "
                             "(kotodama.serve.compile_cache): a stored key is loaded before compiling (the compiles "
                             "become cache hits), a missing one is saved after warmup. Default $KOTODAMA_COMPILE_STORE; "
                             "unset = off")
    parser.add_argument("--compile_mode", default="max-autotune",
                        choices=["max-autotune", "reduce-overhead", "max-autotune-no-cudagraphs", "default"],
                        help="torch.compile mode of the engine and the row decoders: max-autotune benchmarks kernel "
                             "choices for every graph (slowest to start, fastest steps); reduce-overhead keeps the "
                             "CUDA graphs and skips the benchmarking")
    parser.add_argument("--warm_truncated", action=argparse.BooleanOptionalAction, default=True,
                        help="Warmup compiles the truncated (top-p / min-p) step variants; --no-warm_truncated leaves "
                             "each to its first request, which then waits for that compile")
    parser.add_argument("--regional", action=argparse.BooleanOptionalAction, default=True,
                        help="Regional compile (kotodama.model.regional): the engine and the row decoders compile one "
                             "graph per layer signature instead of one graph of the whole depth per variant and "
                             "width, and capture each step whole as one CUDA graph; a fraction of the compile time at "
                             "the same or better step time. --no-regional compiles the whole-step graphs (a model "
                             "whose norms the regions cannot run falls back to them, logged)")
    parser.add_argument("--autotune_min_rows", type=int, default=0,
                        help="Row buckets narrower than this compile without kernel benchmarking (max-autotune's "
                             "choices buy nothing at narrow widths on the regional step, and cost minutes each); "
                             "0 = every bucket compiles under --compile_mode")
    parser.add_argument("--startup_budget_s", type=float, default=1800.0,
                        help="Seconds from process start to ready this server should take; a start over it is logged "
                             "as a warning and reported in /info (startup). 0 = no budget")
    parser.add_argument("--seeded_rows", action=argparse.BooleanOptionalAction, default=False,
                        help="Row decoders draw each candidate's noise from its own seed and step: a request's 'seed' "
                             "then reproduces its candidates whatever shares their batch (up to kernel rounding of "
                             "the logits, which can flip a near-tie). Without it a 'seed' is a 400")
    parser.add_argument("--prefix_pool_block", type=int, default=16,
                        help="Token block size of the prefix KV pool (>= 16): prefixes are shared in whole blocks")
    parser.add_argument("--cascade_attn", action=argparse.BooleanOptionalAction, default=False,
                        help="Row decoders (--multi_rows, --xbatch_rows) share each prompt's KV among its rows and "
                             "attend to it once per group (cascade attention, merged by log-sum-exp): less attention "
                             "time per step and KV memory per row. A row then generates at most "
                             "--cascade_tail_tokens tokens; requests asking for more take the serial path. Start the "
                             "buckets at 16: on the 1B the 8-row cascade step compiles to a slower step than the "
                             "16-row one")
    parser.add_argument("--cascade_groups", type=int, default=0,
                        help="With --cascade_attn: prompts the widest bucket holds at once (each up to 32 rows; "
                             "~0.9 GB of KV per prompt slot on the 1B at 4096 context). 0 = rows // 4")
    parser.add_argument("--cascade_tail_tokens", type=int, default=1024,
                        help="With --cascade_attn: tokens a row can generate (its own KV slot)")
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter directory (kotodama.posttrain.sft --lora_rank) merged into the weights at "
                             "load; the served model id gains +<directory name, less any .adapter suffix> and /info "
                             "reports its rank, alpha, modules and digest. Steering calibrated on the base model "
                             "(--steer-npz, 'injection' requests) is refused with it")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--profile", default=None,
                        help="Serve profile: a name in configs/serve/ (e.g. tsumugi-1b) or a YAML path whose keys are "
                             "these flags; it supplies their defaults and the server's environment "
                             "(kotodama.serve.profiles), and any flag given here overrides it")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    _profile_parser = argparse.ArgumentParser(add_help=False)
    _profile_parser.add_argument("--profile", default=None)
    _profile_arg = _profile_parser.parse_known_args()[0].profile
    if _profile_arg:
        from kotodama.serve import profiles as _profiles
        _profiles.apply(parser, _profile_arg)
    args = parser.parse_args()

    _checkpoint_path = args.checkpoint
    inferred_mode = "chat" if "instruct" in args.checkpoint else "base"

    _served_name_override = args.served_model_name
    _adapter_arg = args.adapter
    _device_arg = args.device
    _mode_arg = args.mode if args.mode is not None else inferred_mode
    _model_size_arg = args.model_size
    _attn_res_override = args.attn_res
    _attn_res_boundaries_override = args.attn_res_boundaries
    _rope_fp32_override = args.rope_fp32
    _engine_arg = args.engine
    _max_seq_len_arg = args.max_seq_len
    _prefix_cache_arg = args.prefix_cache
    _warmup_sdpa_arg = args.warmup_sdpa
    if not 1.0 <= args.default_repetition_penalty <= 2.0:
        parser.error("--default_repetition_penalty must be in [1.0, 2.0]")
    _default_repetition_penalty = args.default_repetition_penalty
    if _prefix_cache_arg and _engine_arg != "fast":
        logger.warning("--prefix-cache requires --engine fast; ignoring")
        _prefix_cache_arg = False

    if args.steer_npz is not None and _engine_arg != "fast":
        # Unlike --prefix-cache this is FATAL, not ignored: a steering server
        # that silently serves unsteered generations poisons downstream data.
        parser.error("--steer-npz requires --engine fast")
    _steer_npz_arg = args.steer_npz
    try:
        _multi_rows_arg = tuple(int(x) for x in args.multi_rows.split(",") if x.strip())
    except ValueError:
        parser.error(f"--multi_rows: comma-separated ints, got {args.multi_rows!r}")
    if any(r < 2 for r in _multi_rows_arg):
        parser.error("--multi_rows buckets must be >= 2")
    if _multi_rows_arg and _engine_arg != "fast":
        parser.error("--multi_rows requires --engine fast")
    try:
        _xbatch_rows_arg = tuple(int(x) for x in args.xbatch_rows.split(",") if x.strip())
    except ValueError:
        parser.error(f"--xbatch_rows: comma-separated ints, got {args.xbatch_rows!r}")
    if any(r < 2 for r in _xbatch_rows_arg):
        parser.error("--xbatch_rows buckets must be >= 2")
    if _xbatch_rows_arg and _engine_arg != "fast":
        parser.error("--xbatch_rows requires --engine fast")
    if args.xbatch_refill and not _xbatch_rows_arg:
        parser.error("--xbatch_refill needs --xbatch_rows")
    _xbatch_refill_arg = args.xbatch_refill
    if args.prefix_pool_tokens < 0 or args.prefix_pool_block < 16:
        parser.error("--prefix_pool_tokens must be >= 0 and --prefix_pool_block >= 16")
    if args.prefix_pool_tokens and _engine_arg != "fast":
        parser.error("--prefix_pool_tokens requires --engine fast")
    if 0 < args.prefix_pool_tokens < args.prefix_pool_block:
        parser.error("--prefix_pool_tokens must hold at least one --prefix_pool_block")
    _prefix_pool_tokens_arg = args.prefix_pool_tokens
    _prefix_pool_block_arg = args.prefix_pool_block
    if args.cascade_attn and not (_multi_rows_arg or _xbatch_rows_arg):
        parser.error("--cascade_attn needs --multi_rows or --xbatch_rows")
    if args.cascade_groups < 0 or args.cascade_tail_tokens < 1:
        parser.error("--cascade_groups must be >= 0 and --cascade_tail_tokens >= 1")
    _cascade_attn_arg = args.cascade_attn
    _cascade_groups_arg = args.cascade_groups
    _cascade_tail_tokens_arg = args.cascade_tail_tokens
    if args.levers < 0 or (args.levers and _engine_arg != "fast"):
        parser.error("--levers needs the fast engine and a slot count >= 0")
    _levers_arg = args.levers
    _seeded_rows_arg = args.seeded_rows
    _compile_store_arg = args.compile_store
    _compile_mode_arg = args.compile_mode
    _warm_truncated_arg = args.warm_truncated
    _regional_arg = args.regional
    _startup_budget_s_arg = args.startup_budget_s
    _autotune_min_rows_arg = args.autotune_min_rows

    import uvicorn

    from kotodama.serve.logs import uvicorn_log_config
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", log_config=uvicorn_log_config())
