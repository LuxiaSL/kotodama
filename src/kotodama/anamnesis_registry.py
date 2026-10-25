"""kotodama's rows in the anamnesis model registry, built from kotodama's own tables.

anamnesis reads its model facts from registry files: the one it ships, then every file named in
the ``ANAMNESIS_MODELS`` environment variable (separated like ``PATH``). A model it does not ship
joins by a file of the same shape; anamnesis never learns the model's name or paths. This module
builds that file for kotodama's models from :mod:`kotodama.presets` (shape, AttnRes boundaries,
tokenizer), :mod:`kotodama.serve.laws` (sampling) and :mod:`kotodama.serve.chatml` (stop tokens),
so no fact is restated here. :data:`REGISTRY_FILE` is the generated file, pinned by
``tests/test_anamnesis_registry.py`` and regenerated on purpose with
``KOTODAMA_WRITE_ANAMNESIS_REGISTRY=1``.

Use it with::

    export ANAMNESIS_MODELS="$(python -c 'from kotodama import anamnesis_registry as r; print(r.REGISTRY_FILE)')"

**Layer plan.** Sampled layers follow the rule every anamnesis model of record uses: the 8B row's
layers ``0, 8, 16, 20, 24, 28, 31`` of 32 rescaled to the target's effective depth, rounded half to
even, with the top layer anchored to the target's last. At depth 28 that is kotodama-3b's set,
``0, 7, 14, 18, 21, 24, 27``. A looped model adds each sampled layer's *loop twins*: the other
effective positions that run the same attention and FFN weights on another pass of the core. A
twin pair reads the same computation at a different point of the stream, which is what the loop
adds. Twins join ``sampled_layers`` only; the PCA, trajectory and contrastive layers and the
cutoffs stay the rule's, so the depth reading stays comparable with every other model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from kotodama import presets
from kotodama.model.llama import KotodamaModelConfig
from kotodama.serve.chatml import BASE_STOP_TOKEN_IDS
from kotodama.serve.laws import law

REGISTRY_FILE = Path(__file__).with_name("anamnesis_models.json")
"""The generated registry file, to name in ``ANAMNESIS_MODELS``."""

WRITE_ENV = "KOTODAMA_WRITE_ANAMNESIS_REGISTRY"
"""Set to 1 to regenerate :data:`REGISTRY_FILE` from the tables (the pinning test then writes it)."""

MODELS: dict[str, dict[str, str]] = {
    "kotodama-tsumugi-1b": dict(
        shape="1b-moonshot-loop",
        law="base",
        torch_dtype="bfloat16",
        calibration_dir="calibration/kotodama_tsumugi_1b",
        notes=(
            "kotodama-tsumugi 1B base: 52 unique / 72 effective layers (a 20-layer core from layer 19, "
            "run twice), AttnRes over 8 committed blocks, SmolLM2 tokenizer. A raw kotodama checkpoint: "
            "load and capture through kotodama.model.capture, never an HF cast (it would drop the "
            "routing). The sampling fields are the kotodama base law until the model has its own."
        ),
    ),
}
"""anamnesis registry key -> the kotodama shape, sampling law, dtype and calibration directory."""

_8B_NUM_LAYERS = 32
_8B_SAMPLED = (0, 8, 16, 20, 24, 28, 31)
_8B_PCA = (8, 16, 20, 24, 28)
_8B_EARLY, _8B_LATE = 8, 24
MAX_NEW_TOKENS = 512
"""The anamnesis extraction protocol's generation length."""


def rescale_layer(layer_8b: int, num_layers: int) -> int:
    """An 8B layer index at the same fractional depth of ``num_layers`` (the top layer anchored)."""
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if layer_8b == _8B_NUM_LAYERS - 1:
        return num_layers - 1
    return max(0, min(num_layers - 1, round(layer_8b * num_layers / _8B_NUM_LAYERS)))


def loop_twins(config: KotodamaModelConfig, layer: int) -> list[int]:
    """The other effective positions that run ``layer``'s attention and FFN weights."""
    source = config.layer_source(layer)
    return [i for i in range(config.num_layers) if i != layer and config.layer_source(i) == source]


def layer_plan(shape: str) -> dict[str, Any]:
    """The anamnesis layer fields for a kotodama shape: the rescaling rule, plus loop twins."""
    config = KotodamaModelConfig(**presets.shape(shape))
    n = config.num_layers
    rule = [rescale_layer(layer, n) for layer in _8B_SAMPLED]
    if len(set(rule)) != len(rule):
        raise ValueError(f"rescaling the 8B layers to depth {n} collided: {rule}")
    twins = {t for layer in rule for t in loop_twins(config, layer)}
    pca = [rescale_layer(layer, n) for layer in _8B_PCA]
    return dict(
        sampled_layers=sorted(set(rule) | twins),
        pca_layers=pca,
        trajectory_layers=list(pca),
        contrastive_layers=list(pca),
        early_layer_cutoff=rescale_layer(_8B_EARLY, n),
        late_layer_cutoff=rescale_layer(_8B_LATE, n),
        rule_layers=rule,
        loop_twin_layers=sorted(twins),
    )


def preset_row(key: str) -> dict[str, Any]:
    """One anamnesis registry row for the model ``key`` names in :data:`MODELS`."""
    spec = MODELS[key]
    shape = presets.shape(spec["shape"])
    plan = layer_plan(spec["shape"])
    sampling = law(spec["law"])
    return dict(
        name=key,
        model_id=key,
        torch_dtype=spec["torch_dtype"],
        num_layers=shape["num_layers"],
        hidden_dim=shape["hidden_size"],
        num_attention_heads=shape["num_attention_heads"],
        num_kv_heads=shape["num_kv_heads"],
        head_dim=shape["head_dim"],
        sampled_layers=plan["sampled_layers"],
        pca_layers=plan["pca_layers"],
        trajectory_layers=plan["trajectory_layers"],
        contrastive_layers=plan["contrastive_layers"],
        early_layer_cutoff=plan["early_layer_cutoff"],
        late_layer_cutoff=plan["late_layer_cutoff"],
        temperature=sampling["temperature"],
        # The laws spell "top-p off" as 0; anamnesis spells it as the whole mass.
        top_p=sampling["top_p"] if sampling["top_p"] > 0 else 1.0,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_ids=sorted(BASE_STOP_TOKEN_IDS),
        calibration_root="outputs",
        calibration_dir=spec["calibration_dir"],
        notes=(f"{spec['notes']} Sampled layers: the rescaling rule {plan['rule_layers']} plus loop twins "
               f"{plan['loop_twin_layers']}. Tokenizer {presets.TOKENIZER_NAME}; AttnRes boundaries "
               f"{list(presets.DEFAULT_BOUNDARIES[spec['shape']])}."),
    )


def registry() -> dict[str, Any]:
    """The whole registry file, as anamnesis reads it."""
    return dict(
        description=("kotodama's models for the anamnesis registry, generated by "
                     "kotodama.anamnesis_registry from kotodama's own tables; name this file in "
                     "ANAMNESIS_MODELS."),
        presets={key: preset_row(key) for key in MODELS},
    )


def render() -> str:
    """The registry file's exact bytes."""
    return json.dumps(registry(), indent=2) + "\n"


def write(path: Path = REGISTRY_FILE) -> Path:
    """Write the registry file and return its path."""
    path.write_text(render())
    return path


if __name__ == "__main__":
    print(write() if os.environ.get(WRITE_ENV) == "1" else render(), end="")
