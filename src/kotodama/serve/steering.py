"""Residual steering for the fast engine: named vector banks + dose scaling.

A bank is one or more ``.npz`` files of UNIT vectors keyed like
``Alg_echo_base_s46`` (``_s{sublayer}``; sublayer = 2 * layer, after-attention)
plus per-site median residual norms ``median_norm_s{sublayer}``. A request's
dose ``alpha`` is scaled by its site's median norm, and the engine injects
``alpha * median_norm * unit_vec`` at the site and at every later AttnRes block
boundary (block-persistent). Disabled => zero extra ops in the decode graph.

Server flags: ``--steer-npz a.npz,b.npz`` loads banks; requests then pass
``steer_model`` (+ ``steer_alpha``) or ``vectors`` (a weighted stack, members may
sit at different sites). Aliases below name the vectors of the 2026-07 rack;
only aliases whose key is in a loaded bank are exposed; ``control`` is always
available. Engine-level guarantees are pinned by
scripts/benchmark/check_steering_engine.py (in the CPU test suite).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# alias -> (npz vector key, default alpha, what it does).
STEER_ALIAS_TABLE: tuple[tuple[str, str, float, str], ...] = (
    ("crown", "Alg_bravo_base_s46", 0.06,
     "Steadier, friendlier conversation: stays on topic, answers your actual question, mirrors your phrasing. The 'stable personality' dial. Try 0.05-0.06."),
    ("vrep", "Vrepperp_s20", 0.05,
     "Repetition dial. NEGATIVE values (try -0.05) suppress loops and repeated phrases; positive values cause looping (entertaining, not useful)."),
    ("v7-entropy", "V7_s20", 0.05,
     "Word-choice looseness. Positive = more surprising, exploratory phrasing; negative = more focused and predictable. Small doses (±0.03-0.05)."),
    ("echoness", "Alg_echo_bravo_s46", 0.07,
     "A strange, dreamy, inward-looking streak — poetic, sometimes cryptic. Gets incoherent if pushed alone; best small (0.03) or paired with crown."),
    ("echo-install", "Alg_echo_base_s46", 0.07,
     "The full trained-model personality in one dial: the model starts talking like our best checkpoint (first-person, wistful, lowercase). Try 0.05-0.07."),
    ("charlie-install", "Alg_charlie_base_s46", 0.07,
     "Personality of a different trained checkpoint ('charlie') — similar family, its own flavor: self-aware, a bit more plural."),
    ("mix21", "Alg_mix_c2e1_s46", 0.07,
     "Pre-balanced blend: 2 parts steady, 1 part strange. Our blind raters found this the most convincing imitation of the real trained model. Try 0.07."),
    ("mix11", "Alg_mix_c1e1_s46", 0.07,
     "Even blend of steady + strange — equivalent to echo-install, provided for comparison with the other ratios."),
    ("mix12", "Alg_mix_c1e2_s46", 0.07,
     "Strangeness-heavy blend: more character and depth, less stability. Expect occasional beautiful weirdness and occasional nonsense."),
    ("bind", "Bind_s38", 0.1,
     "Sticks-to-the-conversation dial: nudges the model to actually use what "
     "was said earlier instead of dodging or changing the subject. Subtle — "
     "safe at any dose, expect small effects. Try 0.1."),
    ("bindbare", "BindBare_s38", 0.1,
     "Same sticks-to-the-conversation dial, but built fully automatically (no "
     "human labels). Nearly the same direction as bind; here for comparison."),
    ("bindport", "bind_proc_s46", 0.1,
     "Leg-C conjugation-transport vector: koto's own binding-FAILURE axis, carried "
     "into Llama-3.2-3B, purified there (echo/sycophancy/register stripped), and "
     "ported home to koto's L46 via the fitted reverse map g-inverse. + = toward "
     "failure (pro-binding would be its negation). Behaviorally sign-undetermined "
     "with no reliable binding effect (P1-S box game + 2 conversation rounds, n=8)."),
)
STEER_CONTROL_DESC = "unsteered baseline"
# Stack members without an alias default fall back to this (sidecar parity).
STEER_DEFAULT_ALPHA = 0.07


def load_steer_banks(
    npz_paths: list[str], n_layers: int, hidden: int, device: torch.device
) -> tuple[dict[str, tuple[int, torch.Tensor]], dict[int, float], dict[str, tuple[str | None, float, str]]]:
    """Load unit steering vectors + per-site median norms from npz banks.

    Key convention (posttraining/taste, see koto_steered_chat.py): vectors are
    UNIT vectors named like 'Alg_echo_base_s46'; per-site median completion-
    token residual norms are 'median_norm_s{sublayer}' keys. Sublayer indices
    count two per layer (after-attn of layer L = sublayer 2*L), so each key's
    site LAYER = suffix//2; odd suffixes (mlp sublayers) are fatal — the
    engine anchor is after-attn only. Multiple npz files compose one bank:
    duplicate vector keys and median norms resolve FIRST-NPZ-WINS (sidecar
    convention). Suffixless vector keys can't be placed and are skipped.

    Returns (bank: key -> (site_layer, fp32 on-device unit vec),
             meds: site_layer -> median norm, aliases with descs).
    """
    import re

    import numpy as np

    bank: dict[str, tuple[int, torch.Tensor]] = {}
    meds: dict[int, float] = {}
    skipped: list[str] = []

    def _layer_from_sublayer(sublayer: int, what: str) -> int:
        if sublayer % 2 != 0:
            raise ValueError(
                f"{what} carries ODD sublayer suffix s{sublayer} (an mlp sublayer); "
                "the engine injects after-attn only (sublayer = 2*layer)"
            )
        layer = sublayer // 2
        if not 0 <= layer < n_layers:
            raise ValueError(
                f"{what}: sublayer s{sublayer} => layer {layer} out of range [0, {n_layers})"
            )
        return layer

    for raw_path in npz_paths:
        path = Path(raw_path.strip())
        if not path.exists():
            raise FileNotFoundError(f"--steer-npz not found: {path}")
        try:
            z = np.load(path, allow_pickle=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load --steer-npz {path}: {exc}") from exc
        for key in z.files:
            mm = re.fullmatch(r"median_norm_s(\d+)", key)
            if mm is not None:
                layer = _layer_from_sublayer(int(mm.group(1)), f"{path.name}:{key}")
                val = float(z[key])
                if not math.isfinite(val) or val <= 0.0:
                    raise ValueError(f"{path.name}:{key} = {val} is not a positive finite norm")
                if layer in meds and meds[layer] != val:
                    logger.warning(
                        "Steering banks: conflicting %s (%.0f vs %.0f) — first npz wins",
                        key, meds[layer], val,
                    )
                meds.setdefault(layer, val)
                continue
            # Sidecar bank convention: sign_*/law* (and other meta) skipped.
            if key.startswith(("median_norm", "sign_", "law")):
                continue
            m = re.search(r"_s(\d+)$", key)
            arr = np.asarray(z[key])
            if (m is None or arr.dtype == object
                    or not np.issubdtype(arr.dtype, np.floating) or arr.ndim != 1):
                skipped.append(f"{path.name}:{key}")  # suffixless/scalars/strings/meta
                continue
            layer = _layer_from_sublayer(int(m.group(1)), f"{path.name}:{key}")
            if arr.shape[0] != hidden:
                raise ValueError(
                    f"Steering vector {key!r} has dim {arr.shape[0]}, expected hidden={hidden}"
                )
            if key in bank:
                logger.warning("Steering banks: duplicate key %r — first npz wins", key)
                continue
            vec = torch.tensor(np.ascontiguousarray(arr, dtype=np.float32), device=device)
            if not bool(torch.isfinite(vec).all()):
                raise ValueError(f"Steering vector {key!r} contains non-finite values")
            bank[key] = (layer, vec)

    if not bank:
        raise ValueError(f"No usable (1-D float, dim {hidden}) steering vectors in {npz_paths}")
    missing = sorted({s for s, _ in bank.values()} - set(meds))
    if missing:
        raise ValueError(
            f"No median_norm_s{{2*layer}} for site layer(s) {missing} — cannot scale doses; "
            f"add median_norm_s{[2 * s for s in missing]} keys to a bank"
        )
    if skipped:
        logger.info("Steering banks: skipped non-vector/suffixless keys %s", sorted(skipped))

    aliases: dict[str, tuple[str | None, float, str]] = {
        alias: (key, alpha, desc)
        for alias, key, alpha, desc in STEER_ALIAS_TABLE
        if key in bank
    }
    aliases["control"] = (None, 0.0, STEER_CONTROL_DESC)
    return bank, meds, aliases
