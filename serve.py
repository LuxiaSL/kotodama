"""
Inference server for kotodama 108M models.

Usage:
    python serve.py --model kotodama-108m-base-fc [--compile]
    python serve.py --model kotodama-108m-instruct-fc [--compile]
    python serve.py --checkpoint /path/to/custom.pt --mode chat

Models (resolved from checkpoints/serving/):
    kotodama-108m-base-fc       — fullcorpus pretrained (text completion)
    kotodama-108m-base-bcpt     — books CPT pretrained (text completion)
    kotodama-108m-instruct-fc   — fullcorpus SFT (chat, auto-detected)
    kotodama-108m-instruct-bcpt — books CPT SFT (chat, auto-detected)

Requires: fastapi, uvicorn, transformers (tokenizer only), torch
Optional: triton (enables fused AttnRes kernels, ~2x routing speedup)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
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

import torch
import torch.nn.functional as F

torch.set_num_threads(_SERVE_THREADS)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from src.model.llama import LuxiaBaseModel, LuxiaModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Fast AttnRes kernels (optional, requires triton) ───────────────────────────

_FAST_ATTNRES_AVAILABLE = False
try:
    from src.model.flash_attn_res.ops.phase_1 import phase_1_batched_attention_triton_op as phase_1_forward
    from src.model.flash_attn_res.ops.phase_2 import phase_2_online_softmax_merge_triton_op as phase_2_merge

    _FAST_ATTNRES_AVAILABLE = True
except Exception:
    phase_1_forward = None  # type: ignore[assignment]
    phase_2_merge = None  # type: ignore[assignment]


# ── Defaults ────────────────────────────────────────────────────────────────────

SERVING_DIR = Path("checkpoints/serving")
KNOWN_MODELS: dict[str, str] = {
    "kotodama-108m-base-fc": "kotodama-108m-base-fc.pt.zst",
    "kotodama-108m-base-bcpt": "kotodama-108m-base-bcpt.pt.zst",
    "kotodama-108m-instruct-fc": "kotodama-108m-instruct-fc.pt",
    "kotodama-108m-instruct-bcpt": "kotodama-108m-instruct-bcpt.pt",
}
DEFAULT_CHECKPOINT = str(SERVING_DIR / KNOWN_MODELS["kotodama-108m-base-fc"])
TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
DDV1_BOUNDARIES = [0, 3, 7, 12, 21, 25]

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


# ── Fast AttnRes forward ───────────────────────────────────────────────────────


class FastAttnResContext:
    """Pre-computed state for Triton-accelerated AttnRes forward."""

    def __init__(self, model: LuxiaBaseModel) -> None:
        config = model.config
        self.eps = config.norm_eps

        # Fold norm weights into queries: effective_q = query * norm.weight
        # Layout: [attn_q_0, mlp_q_0, attn_q_1, mlp_q_1, ..., final_q]
        effective_queries: list[torch.Tensor] = []
        for layer in model.layers:
            effective_queries.append(layer.attn_res_query * layer.attn_res_norm.weight)
            effective_queries.append(layer.mlp_res_query * layer.mlp_res_norm.weight)
        effective_queries.append(model.final_res_query * model.final_res_norm.weight)
        self.pseudo_queries = torch.stack(effective_queries, dim=0)  # [57, D]

        # DD-v1 layer boundaries → sublayer boundaries (2 sublayers per layer)
        layer_boundaries = sorted(model._attn_res_boundary_set)
        self.sublayer_boundaries = [2 * b for b in layer_boundaries]

        # Build sublayer callables: each returns the RESIDUAL UPDATE
        rope_cos = model.rope_cos
        rope_sin = model.rope_sin
        self.sublayers: list[Callable] = []
        for layer in model.layers:
            self.sublayers.append(
                lambda x, ly=layer: ly.attn(ly.attn_norm(x), rope_cos, rope_sin)
            )
            self.sublayers.append(
                lambda x, ly=layer: ly.ffn(ly.ffn_norm(x))
            )

        self.final_norm = model.norm


def fast_forward_attn_res(ctx: FastAttnResContext, embed: torch.Tensor) -> torch.Tensor:
    """AttnRes forward using Triton-fused phase 1/2 kernels with variable block sizes."""
    blocks = [embed]
    sublayers = ctx.sublayers
    pq = ctx.pseudo_queries
    eps = ctx.eps
    num_sublayers = len(sublayers)

    # Compute block ranges from sublayer boundaries
    boundaries = ctx.sublayer_boundaries
    block_ranges: list[tuple[int, int]] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else num_sublayers
        block_ranges.append((start, end))

    for block_start, block_end in block_ranges:
        num_queries = block_end - block_start
        values = torch.stack(blocks, dim=0)

        phase1_out, phase1_lse = phase_1_forward(
            values,
            pq[block_start: block_start + num_queries],
            eps,
        )

        curr_block = None
        for query_offset in range(num_queries):
            sublayer_idx = block_start + query_offset

            if query_offset == 0:
                layer_input = phase1_out[0]
                curr_block = sublayers[sublayer_idx](layer_input)
            else:
                layer_input = phase_2_merge(
                    curr_block,
                    pq[sublayer_idx],
                    phase1_out[query_offset],
                    phase1_lse[query_offset],
                    eps,
                )
                curr_block = curr_block + sublayers[sublayer_idx](layer_input)

        blocks.append(curr_block)

    # Final aggregation over all committed blocks
    final_out, _ = phase_1_forward(
        torch.stack(blocks, dim=0),
        pq[-1:],
        eps,
    )

    return ctx.final_norm(final_out[0].to(embed.dtype))


# ── Global state ────────────────────────────────────────────────────────────────

_model: LuxiaBaseModel | None = None
_compiled_model: torch.nn.Module | None = None
_tokenizer: AutoTokenizer | None = None
_device: torch.device = torch.device("cpu")
_fast_ctx: FastAttnResContext | None = None
_serve_mode: str = "base"
_stop_token_ids: frozenset[int] = BASE_STOP_TOKEN_IDS


@torch.inference_mode()
def _warmup_triton_kernels(model: LuxiaBaseModel, ctx: FastAttnResContext) -> None:
    """Run a dummy forward pass to trigger Triton JIT compilation for all kernel variants."""
    logger.info("Warming up Triton kernels (compiling %d block variants)...", len(ctx.sublayer_boundaries))
    t0 = time.time()
    dummy = torch.randn(1, 8, model.config.hidden_size, device=_device, dtype=torch.bfloat16)
    fast_forward_attn_res(ctx, dummy)
    torch.cuda.synchronize()
    logger.info("Triton warmup done in %.1fs", time.time() - t0)


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


@torch.inference_mode()
def _warmup_compile(model: LuxiaBaseModel, compiled_model: torch.nn.Module) -> None:
    """Trigger torch.compile specialization for the decode path.

    Warms compiled decode after prefills at multiple prompt lengths to cover
    the shape specializations that real requests will hit. The first 2-3
    compiled decode steps trigger inductor compilation (~30-45s total on cold
    cache, faster with inductor cache). After that, decode is stable.
    """
    logger.info("Warming up torch.compile decode path...")
    t0 = time.time()

    for plen in [16, 128, 1024]:
        dummy_ids = torch.zeros(1, plen, dtype=torch.long, device=_device)
        out = model(dummy_ids, use_cache=True)
        torch.cuda.synchronize(_device)
        past_kv = out["past_kv"]
        tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)

        for i in range(4):
            t_step = time.time()
            out = compiled_model(tok, use_cache=True, past_kv=past_kv)
            torch.cuda.synchronize(_device)
            elapsed = time.time() - t_step
            past_kv = out["past_kv"]
            tok = out["logits"][0, -1].argmax().unsqueeze(0).unsqueeze(0)
            if elapsed > 1.0:
                logger.info("  Compile warmup plen=%d step %d: %.1fs (compilation)", plen, i, elapsed)

    logger.info("Compile warmup done in %.1fs", time.time() - t0)


def load_model(checkpoint_path: str, device: str = "cuda", compile: bool = False, mode: str = "base") -> tuple[LuxiaBaseModel, torch.nn.Module | None, AutoTokenizer]:
    global _device, _fast_ctx, _serve_mode, _stop_token_ids
    _serve_mode = mode
    _stop_token_ids = CHAT_STOP_TOKEN_IDS if mode == "chat" else BASE_STOP_TOKEN_IDS
    if device.startswith("cuda") and torch.cuda.is_available():
        _device = torch.device(device)
    else:
        _device = torch.device("cpu")
    logger.info("Device: %s", _device)

    use_fast = _FAST_ATTNRES_AVAILABLE and _device.type == "cuda"

    config = LuxiaModelConfig(**PROXY_CONFIG)
    logger.info("Model config: %dM params", config.param_count() // 1_000_000)

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

    if use_fast:
        _fast_ctx = FastAttnResContext(model)
        logger.info("Fast AttnRes kernels enabled (Triton-fused phase 1/2)")
        _warmup_triton_kernels(model, _fast_ctx)
    else:
        _fast_ctx = None
        if _device.type == "cuda":
            logger.info("Triton not available, using standard AttnRes forward")

    if _device.type == "cuda":
        torch.backends.cuda.enable_cudnn_sdp(False)
        logger.info("cuDNN SDP disabled (using flash/math backend)")

    compiled_model: torch.nn.Module | None = None
    if compile and _device.type == "cuda":
        logger.info("Creating torch.compile(dynamic=True) decode model...")
        compiled_model = torch.compile(model, dynamic=True)
        _warmup_compile(model, compiled_model)
    else:
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

    return model, compiled_model, tokenizer


# ── Request/response schemas ────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0)
    stop_strings: list[str] = Field(default_factory=list)
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    timing: dict[str, float] | None = None


class ModelInfo(BaseModel):
    name: str
    mode: str
    params: int
    config: dict
    device: str
    checkpoint: str
    triton_attn_res: bool
    compiled_decode: bool


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
    """Run model forward, returning logits. Uses fast AttnRes kernels when available."""
    if _fast_ctx is not None:
        embed = model.embed_tokens(input_ids)
        hidden = fast_forward_attn_res(_fast_ctx, embed)
        return F.linear(hidden, model.get_lm_head_weight())

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

    # Use compiled model for decode if available, eager for prefill
    decode_model = _compiled_model if _compiled_model is not None else model

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
                                timing=timing)

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
    )


@torch.inference_mode()
async def generate_stream(
    model: LuxiaBaseModel,
    tokenizer: AutoTokenizer,
    request: GenerateRequest,
):
    prompt_text = _resolve_prompt(request, tokenizer)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(_device)
    prompt_len = input_ids.shape[1]

    if prompt_len >= model.config.max_position_embeddings:
        yield f'data: {{"error": "Prompt too long: {prompt_len} tokens"}}\n\n'
        return

    generated_ids: list[int] = []
    prev_text = ""

    decode_model = _compiled_model if _compiled_model is not None else model

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

    yield f"data: {json.dumps({'done': True, 'prompt_tokens': prompt_len, 'completion_tokens': len(generated_ids)})}\n\n"


# ── App ─────────────────────────────────────────────────────────────────────────

_checkpoint_path = DEFAULT_CHECKPOINT
_compile_arg = False
_mode_arg = "base"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _compiled_model, _tokenizer
    _model, _compiled_model, _tokenizer = load_model(
        _checkpoint_path, _device_arg, compile=_compile_arg, mode=_mode_arg,
    )
    yield


def _model_name() -> str:
    ckpt_stem = Path(_checkpoint_path).stem.removesuffix(".pt")
    for name in KNOWN_MODELS:
        if ckpt_stem.startswith(name):
            return name
    suffix = "instruct" if _serve_mode == "chat" else "base"
    return f"kotodama-108m-{suffix}-unknown"


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
        triton_attn_res=_FAST_ATTNRES_AVAILABLE and _device.type == "cuda",
        compiled_decode=_compiled_model is not None,
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
async def generate_endpoint(request: GenerateRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded")

    if request.stream:
        return StreamingResponse(
            generate_stream(_model, _tokenizer, request),
            media_type="text/event-stream",
        )

    return generate(request=request, model=_model, tokenizer=_tokenizer)


# ── OpenAI-compatible /v1/completions (for Loom/Loomsidian) ─────────────────────

class OAICompletionRequest(BaseModel):
    prompt: str
    model: str = ""
    max_tokens: int = Field(default=256, ge=1, le=2048)
    n: int = Field(default=1, ge=1, le=8)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    best_of: int | None = None
    stop: list[str] | str | None = None


@app.post("/v1/completions")
async def oai_completions(request: OAICompletionRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, detail="Model not loaded")

    stop_strings: list[str] = []
    if isinstance(request.stop, str):
        stop_strings = [request.stop]
    elif isinstance(request.stop, list):
        stop_strings = request.stop

    choices = []
    for i in range(request.n):
        gen_req = GenerateRequest(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=1.0 + request.frequency_penalty,
            stop_strings=stop_strings,
        )
        result = generate(model=_model, tokenizer=_tokenizer, request=gen_req)
        choices.append({
            "text": result.text,
            "index": i,
            "logprobs": None,
            "finish_reason": "length" if result.completion_tokens >= request.max_tokens else "stop",
        })

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
    active = _model_name()
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "owned_by": "aethera-gp",
                "active": name == active,
            }
            for name in KNOWN_MODELS
        ],
    }


# ── OpenAI-compatible /v1/chat/completions (chat mode) ────────────────────────


class OAIChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = ""
    max_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    stop: list[str] | str | None = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=8)


@app.post("/v1/chat/completions")
async def oai_chat_completions(request: OAIChatRequest):
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

    if request.stream:
        gen_req = GenerateRequest(
            prompt=prompt_text,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=1.0 + request.frequency_penalty,
            stop_strings=stop_strings,
            stream=True,
        )

        async def chat_stream():
            async for chunk in generate_stream(_model, _tokenizer, gen_req):
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
            repetition_penalty=1.0 + request.frequency_penalty,
            stop_strings=stop_strings,
        )
        result = generate(model=_model, tokenizer=_tokenizer, request=gen_req)
        total_completion_tokens += result.completion_tokens
        choices.append({
            "index": i,
            "message": {"role": "assistant", "content": result.text},
            "finish_reason": "length" if result.completion_tokens >= request.max_tokens else "stop",
        })

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
    parser = argparse.ArgumentParser(
        description="kotodama inference server",
        epilog="Available models: " + ", ".join(KNOWN_MODELS),
    )
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument("--model", choices=list(KNOWN_MODELS), metavar="NAME",
                            help="Model name (resolves to checkpoints/serving/)")
    ckpt_group.add_argument("--checkpoint", help="Direct path to checkpoint .pt/.pt.zst file")
    parser.add_argument("--device", default="cuda", help="Device: cuda, cuda:N, or cpu")
    parser.add_argument("--mode", choices=["base", "chat"], default=None,
                        help="Serving mode (auto-detected from model name if omitted)")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile(dynamic=True) for faster inference")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2222)
    args = parser.parse_args()

    if args.model:
        _checkpoint_path = str(SERVING_DIR / KNOWN_MODELS[args.model])
        inferred_mode = "chat" if "instruct" in args.model else "base"
    elif args.checkpoint:
        _checkpoint_path = args.checkpoint
        inferred_mode = "chat" if "instruct" in args.checkpoint else "base"
    else:
        _checkpoint_path = DEFAULT_CHECKPOINT
        inferred_mode = "base"

    _device_arg = args.device
    _mode_arg = args.mode if args.mode is not None else inferred_mode
    _compile_arg = args.compile

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
