"""Model specs for the binding suite: a kotodama checkpoint or a Hugging Face model, loaded once per run.

Spec strings (CLI ``--model``):
    koto:CHECKPOINT[?size=1b-tsumugi][&bounds=0,1,7,…][&attn_res=config|on|off]
    hf:MODEL                (a Hugging Face repo id or a local model directory)

A koto checkpoint loads through ``kotodama.serve.server.load_model`` with ``rope_fp32`` on (the serve law for the
ladder and tsumugi lineages). Logits come from ``kotodama.model.capture.instrumented_forward`` (the parity-tested
eager walker) when the model has AttnRes, else from the model's own forward. These are exactly the paths the stage
gauges on record were scored with, so suite numbers are comparable with them. AttnRes geometry is explicit: a checkpoint whose boundaries differ from its preset loads silently wrong
(routing params have the same shapes), so pass ``bounds=`` for any non-default geometry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import torch

LogitsFn = Callable[[torch.Tensor], torch.Tensor]


class SpecError(ValueError):
    """A model spec string that cannot be parsed."""


@dataclass(frozen=True)
class ModelSpec:
    kind: Literal["koto", "hf"]
    path: str
    size: str = "1b-tsumugi"
    bounds: Optional[str] = None
    attn_res: Literal["config", "on", "off"] = "config"

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        kind, sep, rest = spec.partition(":")
        if not sep or not rest:
            raise SpecError(f"model spec must be koto:PATH[?k=v&…] or hf:PATH, got {spec!r}")
        path, _, query = rest.partition("?")
        if kind == "hf":
            if query:
                raise SpecError(f"hf specs take no options, got {spec!r}")
            return cls(kind="hf", path=path)
        if kind != "koto":
            raise SpecError(f"model spec kind must be koto or hf, got {kind!r}")
        kw: dict[str, str] = {}
        for part in filter(None, query.split("&")):
            k, eq, v = part.partition("=")
            if not eq or k not in ("size", "bounds", "attn_res"):
                raise SpecError(f"bad koto option {part!r} in {spec!r} (size= bounds= attn_res=)")
            kw[k] = v
        if kw.get("attn_res", "config") not in ("config", "on", "off"):
            raise SpecError(f"attn_res must be config|on|off in {spec!r}")
        return cls(kind="koto", path=path, size=kw.get("size", cls.size), bounds=kw.get("bounds"),
                   attn_res=kw.get("attn_res", "config"))  # type: ignore[arg-type]


@dataclass
class LoadedModel:
    spec: ModelSpec
    logits_fn: LogitsFn
    tokenizer: Any
    device: torch.device
    max_seq_len: Optional[int]


def load(spec: ModelSpec, device: str = "cuda") -> LoadedModel:
    if device == "cpu":   # torch's thread pools ignore OMP_NUM_THREADS; cap them on a shared node
        torch.set_num_threads(int(os.environ.get("SCORE_THREADS", "8")))
        torch.set_num_interop_threads(1)
    if spec.kind == "koto":
        return _load_koto(spec, device)
    return _load_hf(spec, device)


def _load_koto(spec: ModelSpec, device: str) -> LoadedModel:
    from transformers import AutoTokenizer

    from kotodama import presets
    from kotodama.model import capture
    from kotodama.serve import server

    server._rope_fp32_override = True
    server._attn_res_override = spec.attn_res
    server._attn_res_boundaries_override = spec.bounds
    model, _ = server.load_model(spec.path, device=device, model_size=spec.size)
    model.eval()
    tok = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
    if getattr(model.config, "attn_res", False):
        def fwd(ids: torch.Tensor) -> torch.Tensor:
            return capture.instrumented_forward(model, ids, light=True)[0]
    else:   # the capture walker is AttnRes-only; a plain-residual model's own forward is the reference
        def fwd(ids: torch.Tensor) -> torch.Tensor:
            return model(ids)["logits"]
    msl = getattr(model.config, "max_seq_len", None) or getattr(model.config, "max_position_embeddings", None)
    return LoadedModel(spec, fwd, tok, next(model.parameters()).device, msl)


def _load_hf(spec: ModelSpec, device: str) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.path)
    dt = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(spec.path, torch_dtype=dt,
                                                 device_map=None if device == "cpu" else "auto")
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    def fwd(ids: torch.Tensor) -> torch.Tensor:
        return model(input_ids=ids).logits

    msl = getattr(model.config, "max_position_embeddings", None)
    return LoadedModel(spec, fwd, tok, next(model.parameters()).device, msl)
