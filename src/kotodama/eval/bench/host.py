"""The benchmark's one door into the rest of kotodama.

The benchmark packages (``kotodama.eval.bench``, ``.bindprobes``, ``.binding``, ``.riders``) import nothing from
kotodama outside themselves except through this module; ``tests/test_bench_boundary.py`` checks that. Everything
here is either a constant the item generators need (the tokenizer that sizes and checks items, the ChatML
renderer for the riders chat register) or the loader for a kotodama checkpoint (``koto:`` model specs). A
benchmark that lives outside this repository replaces this module with a plugin that imports kotodama.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kotodama.presets import TOKENIZER_NAME
from kotodama.serve.chatml import render_chatml

if TYPE_CHECKING:
    from kotodama.eval.binding.models import LoadedModel, ModelSpec

__all__ = ["TOKENIZER_NAME", "render_chatml", "load_koto"]


def load_koto(spec: "ModelSpec", device: str) -> "LoadedModel":
    """A kotodama checkpoint as a scorer: logits from the parity-tested capture forward (AttnRes models) or the
    model's own forward (plain residual), with the kotodama tokenizer."""
    import torch
    from transformers import AutoTokenizer

    from kotodama.eval.binding.models import LoadedModel
    from kotodama.model import capture
    from kotodama.serve import server

    server._rope_fp32_override = True
    server._attn_res_override = spec.attn_res
    server._attn_res_boundaries_override = spec.bounds
    model, _ = server.load_model(spec.path, device=device, model_size=spec.size)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if getattr(model.config, "attn_res", False):
        def fwd(ids: torch.Tensor) -> torch.Tensor:
            return capture.instrumented_forward(model, ids, light=True)[0]
    else:   # the capture walker is AttnRes-only; a plain-residual model's own forward is the reference
        def fwd(ids: torch.Tensor) -> torch.Tensor:
            return model(ids)["logits"]
    msl = getattr(model.config, "max_seq_len", None) or getattr(model.config, "max_position_embeddings", None)
    return LoadedModel(spec, fwd, tok, next(model.parameters()).device, msl)
