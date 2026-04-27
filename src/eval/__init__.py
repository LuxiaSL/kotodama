"""Shared evaluation utilities for luxia-base analysis pipeline.

Consolidates model loading, forward passes, generation, perplexity,
metrics I/O, and visualization defaults into reusable modules.
"""

from src.eval.forward import ForwardResult, compute_attention_weights, forward_with_states
from src.eval.generate import generate, generate_text
from src.eval.metrics_io import extract_series, get_nearest_step, load_metrics
from src.eval.model_loader import CheckpointInfo, load_checkpoint_registry, load_model, load_model_config
from src.eval.perplexity import compute_perplexity

__all__ = [
    # model_loader
    "load_model",
    "load_model_config",
    "load_checkpoint_registry",
    "CheckpointInfo",
    # forward
    "forward_with_states",
    "compute_attention_weights",
    "ForwardResult",
    # generate
    "generate",
    "generate_text",
    # perplexity
    "compute_perplexity",
    # metrics_io
    "load_metrics",
    "get_nearest_step",
    "extract_series",
]
