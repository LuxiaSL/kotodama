"""Residual steering for the fast engine: named vector banks + dose scaling.

A bank is one or more ``.npz`` files of UNIT vectors keyed like
``Alg_echo_base_s46`` (``_s{sublayer}``; sublayer = 2 * layer, after-attention)
plus per-site median residual norms ``median_norm_s{sublayer}``. A request's
dose ``alpha`` is scaled by its site's median norm, and the engine injects
``alpha * median_norm * unit_vec`` at the site and at every later AttnRes block
boundary (block-persistent). Disabled => zero extra ops in the decode graph.

Server flags: ``--steer-npz a.npz,b.npz`` loads banks; requests then pass
``steer_model`` (+ ``steer_alpha``) or ``vectors`` (a weighted stack, members may
sit at different sites). Aliases below name the vectors of the steering rack;
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
    # kotodama-tsumugi (1b-tsumugi): the same two dials, built at effective layer 29 (scripts/steer/build_egv.py).
    ("vrep", "Vrepperp_s58", -0.05,
     "Repetition dial. The default (-0.05) suppresses loops and repeated phrases with entropy held flat; positive values cause looping."),
    ("v7-entropy", "V7_s58", 0.05,
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
     "Conjugation-transport vector: koto's own binding-FAILURE axis, carried "
     "into Llama-3.2-3B, purified there (echo/sycophancy/register stripped), and "
     "ported home to koto's L46 via the fitted reverse map g-inverse. + = toward "
     "failure (pro-binding would be its negation). Behaviorally sign-undetermined "
     "with no reliable binding effect (a box-game probe + 2 conversation rounds, n=8)."),
)
STEER_CONTROL_DESC = "unsteered baseline"
# Stack members without an alias default fall back to this.
STEER_DEFAULT_ALPHA = 0.07


def load_steer_banks(
    npz_paths: list[str], n_layers: int, hidden: int, device: torch.device
) -> tuple[dict[str, tuple[int, torch.Tensor]], dict[int, float], dict[str, tuple[str | None, float, str]]]:
    """Load unit steering vectors + per-site median norms from npz banks.

    Key convention: vectors are
    UNIT vectors named like 'Alg_echo_base_s46'; per-site median completion-
    token residual norms are 'median_norm_s{sublayer}' keys. Sublayer indices
    count two per layer (after-attn of layer L = sublayer 2*L), so each key's
    site LAYER = suffix//2; odd suffixes (mlp sublayers) are fatal — the
    engine anchor is after-attn only. Multiple npz files compose one bank:
    duplicate vector keys and median norms resolve first-npz-wins. Suffixless
    vector keys can't be placed and are skipped.

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
            if key.startswith(("median_norm", "sign_", "law", "partial_norm")):
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

    # An alias may name one vector per model (e.g. vrep at the 3B's s20 and tsumugi's s58); the first row whose key
    # is in the loaded banks wins.
    aliases: dict[str, tuple[str | None, float, str]] = {}
    for alias, key, alpha, desc in STEER_ALIAS_TABLE:
        if key in bank and alias not in aliases:
            aliases[alias] = (key, alpha, desc)
    aliases["control"] = (None, 0.0, STEER_CONTROL_DESC)
    return bank, meds, aliases


def load_partial_norms(npz_paths: list[str], n_layers: int) -> dict[int, float]:
    """Per-layer median post-attention partial norms from ``partial_norm_l{L}`` keys in any of the npz files (first
    file wins per layer) — the scale table of the N write rule. Empty when no file carries them."""
    import re

    import numpy as np

    norms: dict[int, float] = {}
    for raw_path in npz_paths:
        with np.load(Path(raw_path.strip()), allow_pickle=False) as z:
            for key in z.files:
                m = re.fullmatch(r"partial_norm_l(\d+)", key)
                if m is None:
                    continue
                layer, val = int(m.group(1)), float(z[key])
                if not 0 <= layer < n_layers:
                    raise ValueError(f"{raw_path}:{key}: layer {layer} out of range [0, {n_layers})")
                if not math.isfinite(val) or val <= 0.0:
                    raise ValueError(f"{raw_path}:{key} = {val} is not a positive finite norm")
                norms.setdefault(layer, val)
    if norms and len(norms) != n_layers:
        logger.warning("partial norms cover %d of %d layers; rule N refuses writes that land elsewhere",
                       len(norms), n_layers)
    return norms



def writes_to_plan(writes: list[tuple[int, torch.Tensor, float]], rule: str, norms: dict[int, float] | None,
                   boundaries: list[int], n_layers: int, span: str) -> "LeverPlan":
    """The engine's WRITES list ``[(site_layer, unit_vec, scale), ...]`` (``_compose`` output, the arguments of
    ``DecodeEngine.set_steer``) as one per-request lever: members at one site sum to that site's float32 row
    ``sum(scale * unit)``, then the write rule places the rows (kotodama.model.levers.write_plan)."""
    import numpy as np

    from kotodama.model.levers import LeverPlan, write_plan

    rows: dict[int, np.ndarray] = {}
    for site, vec, scale in writes:
        v = float(scale) * vec.detach().float().reshape(-1).cpu().numpy()
        rows[int(site)] = rows[int(site)] + v if int(site) in rows else v.astype(np.float32)
    sites = sorted(rows)
    plan = write_plan(sites, np.stack([rows[s] for s in sites]), boundaries, n_layers, rule,
                      norms if rule == "N" else None)
    return LeverPlan(attn=plan, span=span)


def act_injection_to_plan(sites: list[int], neurons: list[int], mu: list[float], alpha: float,
                          vectors: list[list[float]], span: str, n_layers: int, intermediate: int) -> "LeverPlan":
    """An ``act`` injection as one lever: neuron ``neurons[j]`` at effective layer ``sites[j]`` moves toward
    ``mu[j]`` by ``alpha`` (kotodama.model.levers.act_rows)."""
    from kotodama.model.levers import LeverError, LeverPlan, act_rows

    if vectors:
        raise LeverError("an act injection takes neurons and mu, not vectors")
    if not len(sites) == len(neurons) == len(mu) or not sites:
        raise LeverError(f"an act injection needs one site, neuron and mu per entry, got {len(sites)} sites, "
                         f"{len(neurons)} neurons, {len(mu)} mu")
    if any(not 0 <= int(s) < n_layers for s in sites):
        raise LeverError(f"sites outside [0, {n_layers})")
    return LeverPlan(act=act_rows(zip(sites, neurons, mu), alpha, intermediate), span=span)


def injection_to_plan(sites: list[int], vectors: list[list[float]], alpha: float, rule: str, kind: str, span: str,
                      norms: dict[int, float] | None, boundaries: list[int], n_layers: int,
                      hidden: int) -> "LeverPlan":
    """A raw injection (``alpha * vectors[j]`` at ``sites[j]``, the vectors already carrying their dose, never
    renormalised) as one lever. ``kind`` "attn" writes the attention output with the block-persistent rule;
    "mlp" writes the MLP output at the site only (the rule must be S)."""
    import numpy as np

    from kotodama.model.levers import LeverError, LeverPlan, write_plan

    rows = np.asarray(vectors, dtype=np.float32)
    if rows.ndim != 2 or rows.shape != (len(sites), hidden):
        raise LeverError(f"injection vectors must be [n_sites={len(sites)}, hidden={hidden}], got {rows.shape}")
    if not math.isfinite(alpha):
        raise LeverError("injection alpha must be finite")
    rows = (np.float32(alpha) * rows).astype(np.float32)
    if kind == "mlp":
        if rule != "S":
            raise LeverError("an mlp injection writes the site only: use rule S")
        if len(set(sites)) != len(sites):
            raise LeverError(f"duplicate sites {sites}")
        if any(not 0 <= int(s) < n_layers for s in sites):
            raise LeverError(f"sites {sites} outside [0, {n_layers})")
        return LeverPlan(mlp={int(s): r for s, r in zip(sites, rows)}, span=span)
    if kind != "attn":
        raise LeverError(f"injection kind must be attn or mlp, got {kind!r}")
    if any(not 0 <= int(s) < n_layers for s in sites):
        raise LeverError(f"sites {sites} outside [0, {n_layers})")
    return LeverPlan(attn=write_plan([int(s) for s in sites], rows, boundaries, n_layers, rule,
                                     norms if rule == "N" else None), span=span)
