"""
Inference server for luxia-base proxy DD-v1 model.

Usage:
    python serve.py [--checkpoint PATH] [--device cuda|cpu] [--port 8000] [--host 0.0.0.0]

Requires: fastapi, uvicorn, transformers (tokenizer only), torch
Optional: triton (enables fused AttnRes kernels, ~2x routing speedup)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
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

DEFAULT_CHECKPOINT = "checkpoints/lang-full-ddv1/step_00045775.pt"
TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
DDV1_BOUNDARIES = [0, 3, 7, 12, 21, 25]

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
_tokenizer: AutoTokenizer | None = None
_device: torch.device = torch.device("cpu")
_fast_ctx: FastAttnResContext | None = None


@torch.inference_mode()
def _warmup_triton_kernels(model: LuxiaBaseModel, ctx: FastAttnResContext) -> None:
    """Run a dummy forward pass to trigger Triton JIT compilation for all kernel variants."""
    logger.info("Warming up Triton kernels (compiling %d block variants)...", len(ctx.sublayer_boundaries))
    t0 = time.time()
    dummy = torch.randn(1, 8, model.config.hidden_size, device=_device, dtype=torch.bfloat16)
    fast_forward_attn_res(ctx, dummy)
    torch.cuda.synchronize()
    logger.info("Triton warmup done in %.1fs", time.time() - t0)


def load_model(checkpoint_path: str, device: str = "cuda", compile: bool = False) -> tuple[LuxiaBaseModel, AutoTokenizer]:
    global _device, _fast_ctx
    _device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", _device)

    use_fast = _FAST_ATTNRES_AVAILABLE and _device.type == "cuda"

    config = LuxiaModelConfig(**PROXY_CONFIG)
    logger.info("Model config: %dM params", config.param_count() // 1_000_000)

    model = LuxiaBaseModel(config)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=True)
    logger.info("Checkpoint loaded (step %s, %s tokens)", ckpt.get("step", "?"), ckpt.get("tokens_consumed", "?"))

    model = model.to(_device).eval()
    if _device.type == "cuda":
        model = model.bfloat16()

    if compile and _device.type == "cuda":
        logger.info("Compiling model with torch.compile(dynamic=True)...")
        model = torch.compile(model, dynamic=True)
        logger.info("Compilation registered (will compile on first forward)")

    if use_fast:
        _fast_ctx = FastAttnResContext(model)
        logger.info("Fast AttnRes kernels enabled (Triton-fused phase 1/2)")
        _warmup_triton_kernels(model, _fast_ctx)
    else:
        _fast_ctx = None
        if _device.type == "cuda":
            logger.info("Triton not available, using standard AttnRes forward")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded: %s (vocab %d)", TOKENIZER_NAME, len(tokenizer))

    return model, tokenizer


# ── Request/response schemas ────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
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


class ModelInfo(BaseModel):
    name: str
    params: int
    config: dict
    device: str
    checkpoint: str
    fast_attnres: bool


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
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(_device)
    prompt_len = input_ids.shape[1]

    if prompt_len >= model.config.max_position_embeddings:
        raise HTTPException(400, f"Prompt too long: {prompt_len} tokens (max {model.config.max_position_embeddings})")

    generated_ids: list[int] = []
    t0 = time.perf_counter()

    # Prefill: process entire prompt, cache KV
    output = model(input_ids, use_cache=True)
    logits = output["logits"]
    past_kv = output.get("past_kv")

    next_logits = logits[0, -1]
    token_id = sample_next_token(
        next_logits, request.temperature, request.top_k, request.top_p,
        request.repetition_penalty, generated_ids,
    )

    if token_id == tokenizer.eos_token_id:
        elapsed = time.perf_counter() - t0
        return GenerateResponse(text="", prompt_tokens=prompt_len,
                                completion_tokens=0, tokens_per_second=0.0)

    generated_ids.append(token_id)
    next_input = torch.tensor([[token_id]], device=_device)

    # Decode: one token at a time with KV cache
    for _ in range(request.max_new_tokens - 1):
        if prompt_len + len(generated_ids) >= model.config.max_position_embeddings:
            break

        output = model(next_input, use_cache=True, past_kv=past_kv)
        logits = output["logits"]
        past_kv = output.get("past_kv")

        next_logits = logits[0, -1]
        token_id = sample_next_token(
            next_logits, request.temperature, request.top_k, request.top_p,
            request.repetition_penalty, generated_ids,
        )

        if token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(token_id)
        next_input = torch.tensor([[token_id]], device=_device)

        if request.stop_strings:
            decoded_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if any(s in decoded_so_far for s in request.stop_strings):
                break

    elapsed = time.perf_counter() - t0
    completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tps = len(generated_ids) / elapsed if elapsed > 0 else 0.0

    return GenerateResponse(
        text=completion_text,
        prompt_tokens=prompt_len,
        completion_tokens=len(generated_ids),
        tokens_per_second=round(tps, 1),
    )


@torch.inference_mode()
async def generate_stream(
    model: LuxiaBaseModel,
    tokenizer: AutoTokenizer,
    request: GenerateRequest,
):
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(_device)
    prompt_len = input_ids.shape[1]

    if prompt_len >= model.config.max_position_embeddings:
        yield f'data: {{"error": "Prompt too long: {prompt_len} tokens"}}\n\n'
        return

    generated_ids: list[int] = []
    prev_text = ""

    # Prefill
    output = model(input_ids, use_cache=True)
    past_kv = output.get("past_kv")
    next_logits = output["logits"][0, -1]

    token_id = sample_next_token(
        next_logits, request.temperature, request.top_k, request.top_p,
        request.repetition_penalty, generated_ids,
    )

    if token_id == tokenizer.eos_token_id:
        yield f"data: {json.dumps({'done': True, 'prompt_tokens': prompt_len, 'completion_tokens': 0})}\n\n"
        return

    generated_ids.append(token_id)
    next_input = torch.tensor([[token_id]], device=_device)

    current_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    delta = current_text
    prev_text = current_text
    if delta:
        yield f"data: {json.dumps({'token': delta})}\n\n"

    # Decode with KV cache
    for _ in range(request.max_new_tokens - 1):
        if prompt_len + len(generated_ids) >= model.config.max_position_embeddings:
            break

        output = model(next_input, use_cache=True, past_kv=past_kv)
        past_kv = output.get("past_kv")
        next_logits = output["logits"][0, -1]

        token_id = sample_next_token(
            next_logits, request.temperature, request.top_k, request.top_p,
            request.repetition_penalty, generated_ids,
        )

        if token_id == tokenizer.eos_token_id:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer
    _model, _tokenizer = load_model(_checkpoint_path, _device_arg, compile=_compile_arg)
    yield


app = FastAPI(title="luxia-base DD-v1", lifespan=lifespan)
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
        name="luxia-base-proxy-ddv1",
        params=_model.config.param_count(),
        config=asdict(_model.config),
        device=str(_device),
        checkpoint=_checkpoint_path,
        fast_attnres=_fast_ctx is not None,
    )


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
    model: str = "luxia-base-proxy-ddv1"
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
        "model": "luxia-base-proxy-ddv1",
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/v1/models")
async def oai_models():
    return {
        "object": "list",
        "data": [{
            "id": "luxia-base-proxy-ddv1",
            "object": "model",
            "owned_by": "aethera-gp",
        }],
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────

_device_arg = "cuda"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="luxia-base DD-v1 inference server")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to checkpoint .pt file")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile(dynamic=True) for faster inference")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2222)
    args = parser.parse_args()

    _checkpoint_path = args.checkpoint
    _device_arg = args.device
    _compile_arg = args.compile

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
