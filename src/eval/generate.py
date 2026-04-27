"""Unified text generation for evaluation.

Consolidates generation loops from eval_lang_full.py,
eval_lang_full_multisample.py, and other eval scripts.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from src.model.llama import LuxiaBaseModel

logger = logging.getLogger(__name__)

# Default EOS tokens for SmolLM2 tokenizer
DEFAULT_EOS_TOKENS = [0, 1, 2]


@torch.no_grad()
def generate(
    model: LuxiaBaseModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float | None = None,
    eos_tokens: list[int] | None = None,
    max_seq_len: int = 2048,
) -> torch.Tensor:
    """Autoregressive generation with temperature and optional top-p sampling.

    Uses a sliding window to respect the model's context length. Generation
    stops when an EOS token is produced or max_new_tokens is reached.

    Args:
        model: The model in eval mode.
        input_ids: Prompt token IDs, shape (1, prompt_len).
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Must be > 0.
        top_p: Nucleus sampling threshold. None disables top-p.
        eos_tokens: Token IDs that signal end of generation.
            Defaults to [0, 1, 2] (SmolLM2 EOS tokens).
        max_seq_len: Maximum sequence length for sliding window.

    Returns:
        Full sequence tensor (prompt + generated), shape (1, total_len).

    Raises:
        ValueError: If temperature <= 0 or input is not batch size 1.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")
    if input_ids.shape[0] != 1:
        raise ValueError(f"generate() expects batch size 1, got {input_ids.shape[0]}")

    if eos_tokens is None:
        eos_tokens = DEFAULT_EOS_TOKENS
    eos_set = set(eos_tokens)

    generated = input_ids.clone()
    device = input_ids.device

    for _ in range(max_new_tokens):
        # Sliding window to respect context length
        context = generated[:, -max_seq_len:]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(context)
            logits = output["logits"] if isinstance(output, dict) else output

        # Sample next token
        next_logits = logits[:, -1, :] / temperature

        if top_p is not None:
            next_logits = _apply_top_p(next_logits, top_p)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() in eos_set:
            break

    return generated


def generate_text(
    model: LuxiaBaseModel,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float | None = None,
    eos_tokens: list[int] | None = None,
    max_seq_len: int = 2048,
) -> dict[str, Any]:
    """Generate text from a prompt string.

    Convenience wrapper around generate() that handles tokenization and
    decoding.

    Args:
        model: The model in eval mode.
        tokenizer: HuggingFace tokenizer instance.
        prompt: The text prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Optional nucleus sampling threshold.
        eos_tokens: EOS token IDs.
        max_seq_len: Context window size.

    Returns:
        Dict with keys:
            - prompt: The input prompt text.
            - continuation: Generated text (excluding prompt).
            - n_tokens: Number of generated tokens.
            - stopped_by: "eos" or "max_tokens".
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    generated = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_tokens=eos_tokens,
        max_seq_len=max_seq_len,
    )

    total_len = generated.shape[1]
    n_new = total_len - prompt_len

    # Decode the full output and strip the prompt prefix
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    continuation = full_text[len(prompt):]

    # Determine stop reason
    if eos_tokens is None:
        eos_tokens = DEFAULT_EOS_TOKENS
    eos_set = set(eos_tokens)
    last_token = generated[0, -1].item()
    stopped_by = "eos" if last_token in eos_set else "max_tokens"

    return {
        "prompt": prompt,
        "continuation": continuation,
        "n_tokens": n_new,
        "stopped_by": stopped_by,
    }


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering to logits.

    Sets logits below the cumulative probability threshold to -inf.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    # Shift right so the token that crosses the threshold is kept
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    # Scatter back to original indices
    return sorted_logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)
