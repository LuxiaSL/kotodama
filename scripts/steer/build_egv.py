#!/usr/bin/env python3
"""Entropy-gradient steering vectors (EGV) for koto, in the serving engine's write space.

For each functional F and site layer L (an EFFECTIVE position; loop repeats count separately):

    g_F  = (1/N) Σ_n mean_{t ∈ G_n} ∂(mean_t F_n)/∂h_{n,t}        N = --n-grad sampled continuations
    U    = eigvecs of Cov(h) at descending ranks [16, 256)         Σ from --n-sigma DISJOINT continuations
    V_F  = unit(U Uᵀ g_F)

h is the AttnRes running ``partial`` right after layer L's attention sublayer — exactly where
``DecodeEngine._forward_step`` adds ``steer_buf[L]`` (the server's ``--steer-npz`` injection point), so
the gradient is taken in the space of the write. The site leaf is a detached clone that requires grad
(gradient through the downstream path only), installed by wrapping that layer's ``attn_sublayer`` — no
forward mirror. G_n = generated positions p..T-1 (the decode steps the engine steers).

Functionals (per-position terms; ``logits[t]`` predicts ids[t+1]):
  entropy  H(softmax z_t),                          t = p..T-1        -> V7
  selfrep  P(next ∈ generated-so-far ∖ prompt),     t = p..T-2        -> Vrep    (the 3B looping dial)
  repmass  P(next ∈ ids[0..t]) (egv; incl. prompt), t = p-1..T-2      -> Vrepmass
  margin   top-1 − top-2 log-prob,                  t = p-1..T-2      -> Vmargin
  eos      log P(eos),                              t = p-1..T-2      -> Veos
  loglik   log P(ids[t+1]) — the realized token,     t = p-1..T-2      -> Vll     (no contrast: "more like what I wrote")
  typdev   |−log P(ids[t+1]) − H(z_t)|,              t = p-1..T-2      -> Vtyp    (distance from local typicality)
Every non-entropy V also gets ``<name>perp = unit(V − (V·V7) V7)`` (Gram-Schmidt against V7; the ⊥ is
load-bearing — it removes the entropy leak). With selfrep also requested, every other non-entropy V gets
``<name>perp2`` = its Gram-Schmidt residual against both V7 and Vrepperp (what is left once entropy AND repetition
are removed). +Vrep raises self-repeat mass, so the anti-repetition
lever is NEGATIVE alpha on Vrepperp. ``Rband{1,2,3}`` are matched-support random controls (unit(U c)).

Outputs:
  <out>.npz      server bank: unit vectors ``<name>_s{2L}`` + ``median_norm_s{2L}`` (median ‖h_t‖ over
                 Σ generated tokens = the α scale) — loadable by kotodama.serve.steering.load_steer_banks
  <out>_raw.npz  raw (unprojected) mean gradients + per-sequence band coords, for audit
  <out>.json     band_mass, agree, pairwise_cos, cos(Vrep, V7), ⊥ residual fractions, sample counts
  <out>_samples.jsonl  the sampled continuations (ids + text), for audit / reuse

Usage (node, GPU):
  tools/run_py.sh scripts/steer/build_egv.py --ckpt <step.pt> --config-section 1b-moonshot-loop \
      --boundaries 0,1,7,9,17,19,39,59 --rope-fp32 --sites 29,49 --out /path/egv.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kotodama import ckpt, presets  # noqa: E402
from kotodama.model.capture import cached_forward  # noqa: E402
from kotodama.model.decode_engine import compute_steer_points  # noqa: E402
from kotodama.model.llama import KotodamaModel, KotodamaModelConfig  # noqa: E402
from kotodama.serve.chatml import BASE_STOP_TOKEN_IDS  # noqa: E402

logger = logging.getLogger("build_egv")

BAND = (16, 256)
RANDOM_SEED = 20261024
N_RANDOM = 3
# npz vector name per functional (V7 = the entropy vector, the 3B rack's name).
VEC_NAME = {"entropy": "V7", "selfrep": "Vrep", "repmass": "Vrepmass", "margin": "Vmargin", "eos": "Veos",
            "loglik": "Vll", "typdev": "Vtyp"}


def sublayer_key(name: str, layer: int) -> str:
    """Server bank key: after-attn sublayer index = 2 * effective layer (kotodama.serve.steering)."""
    return f"{name}_s{2 * layer}"


# ── site leaf + block-persistent injection (wraps the per-position attn_sublayer; no forward mirror) ──


@contextmanager
def _wrap_attn_sublayer(model: KotodamaModel, layer: int, post: Callable[[Tensor], Tensor]) -> Iterator[None]:
    """Replace ``model.layers[layer].attn_sublayer`` so its output (the running partial after attention —
    the engine's steer row) passes through ``post``. Each effective position owns its TransformerBlock
    (loop repeats share only attn/ffn), so this fires at exactly one position per forward."""
    if not model.config.attn_res:
        raise ValueError("the steering site is defined on the AttnRes forward only")
    if not 0 <= layer < model.config.num_layers:
        raise ValueError(f"site layer {layer} out of range [0, {model.config.num_layers})")
    blk = model.layers[layer]
    prev = blk.__dict__.get("attn_sublayer")  # an instance-level override (compile_regions) to restore
    inner = blk.attn_sublayer

    def wrapped(*args, **kwargs):
        return post(inner(*args, **kwargs))

    blk.attn_sublayer = wrapped
    try:
        yield
    finally:
        if prev is None:
            del blk.attn_sublayer
        else:
            blk.attn_sublayer = prev


@contextmanager
def site_leaf(model: KotodamaModel, layer: int) -> Iterator[dict[str, Tensor]]:
    """``box['h']`` = the site partial of the latest forward, as a detached grad-requiring leaf that the
    rest of the forward consumes (gradient reaches it through the downstream path only)."""
    box: dict[str, Tensor] = {}

    def post(out: Tensor) -> Tensor:
        leaf = out.detach().clone().requires_grad_(True)
        box["h"] = leaf
        return leaf

    with _wrap_attn_sublayer(model, layer, post):
        yield box


@contextmanager
def inject(model: KotodamaModel, writes: Sequence[tuple[int, Tensor, float]], start: int) -> Iterator[None]:
    """Teacher-forced replica of the engine's block-persistent write: ``scale * vec`` added to the partial
    after attention at each ``compute_steer_points`` row of each write, at sequence positions >= ``start``
    (the decode steps — prefill is never steered). Rows from several writes sum, like ``set_steer``."""
    rows: dict[int, Tensor] = {}
    hidden = model.config.hidden_size
    for site, vec, scale in writes:
        v = vec.reshape(hidden).float() * float(scale)
        for row in compute_steer_points(int(site), model._attn_res_boundary_set, model.config.num_layers):
            rows[row] = rows[row] + v if row in rows else v.clone()
    with ExitStack() as stack:
        for row, delta in rows.items():
            def post(out: Tensor, d: Tensor = delta) -> Tensor:
                add = torch.zeros_like(out)
                add[:, start:] = d.to(device=out.device, dtype=out.dtype)
                return out + add
            stack.enter_context(_wrap_attn_sublayer(model, row, post))
        yield


# ── functionals: f(logits [T,V] fp32, ids [T], p) -> scalar ──────────────────────────────────────────

Functional = Callable[[Tensor, Tensor, int], Tensor]


def f_entropy(logits: Tensor, ids: Tensor, p: int) -> Tensor:
    lp = torch.log_softmax(logits[p:], dim=-1)
    return -(lp.exp() * lp).sum(-1).mean()


def f_selfrep(logits: Tensor, ids: Tensor, p: int) -> Tensor:
    """Mean over t = p..T-2 of P(next ∈ {ids[p..t]} ∖ prompt); steps with an empty set are skipped
    (steer_formula_pulse.s_selfrep, vectorized)."""
    T, V = logits.shape
    if T - 1 <= p:
        raise ValueError("selfrep needs >= 2 generated tokens")
    gen = ids[p:T - 1]                                            # tokens available as 'seen' at t = p..T-2
    in_prompt = torch.zeros(V, dtype=torch.bool, device=ids.device)
    in_prompt[ids[:p]] = True
    onehot = torch.zeros(len(gen), V, dtype=torch.bool, device=ids.device)
    keep = ~in_prompt[gen]
    onehot[torch.arange(len(gen), device=ids.device)[keep], gen[keep]] = True
    seen = torch.cumsum(onehot.to(torch.int32), dim=0) > 0       # row i: set after emitting ids[p..p+i]
    probs = torch.softmax(logits[p:T - 1], dim=-1)               # logits at t = p..T-2
    mass = (probs * seen.to(probs.dtype)).sum(-1)
    ok = seen.any(-1)
    if not bool(ok.any()):
        raise ValueError("selfrep: no generated token outside the prompt")
    return mass[ok].mean()


def f_repmass(logits: Tensor, ids: Tensor, p: int) -> Tensor:
    T, V = logits.shape
    probs = torch.softmax(logits[p - 1:T - 1], dim=-1)
    base = torch.zeros(V, dtype=torch.bool, device=ids.device)
    base[ids[:p]] = True
    gen = ids[p - 1:T - 1]
    onehot = torch.zeros(len(gen), V, dtype=torch.bool, device=ids.device)
    onehot[torch.arange(len(gen), device=ids.device), gen] = True
    seen = (torch.cumsum(onehot.to(torch.int32), dim=0) > 0) | base
    return (probs * seen.to(probs.dtype)).sum(-1).mean()


def f_margin(logits: Tensor, ids: Tensor, p: int) -> Tensor:
    lp = torch.log_softmax(logits[p - 1:-1], dim=-1)
    top2 = torch.topk(lp.detach(), 2, dim=-1).indices
    return (lp.gather(-1, top2[:, :1]) - lp.gather(-1, top2[:, 1:])).mean()


def f_eos(logits: Tensor, ids: Tensor, p: int) -> Tensor:
    lp = torch.log_softmax(logits[p - 1:-1], dim=-1)
    return torch.logsumexp(lp[:, sorted(BASE_STOP_TOKEN_IDS)], dim=-1).mean()


def f_loglik(logits: Tensor, ids: Tensor, p: int) -> Tensor:
    lp = torch.log_softmax(logits[p - 1:-1], dim=-1)
    return lp.gather(-1, ids[p:, None]).squeeze(-1).mean()


def f_typdev(logits: Tensor, ids: Tensor, p: int) -> Tensor:
    """Mean |surprisal of the realized token − entropy| — how far each step sits from local typicality."""
    lp = torch.log_softmax(logits[p - 1:-1], dim=-1)
    H = -(lp.exp() * lp).sum(-1)
    return (-lp.gather(-1, ids[p:, None]).squeeze(-1) - H).abs().mean()


FUNCTIONALS: dict[str, Functional] = {
    "entropy": f_entropy, "selfrep": f_selfrep, "repmass": f_repmass, "margin": f_margin, "eos": f_eos,
    "loglik": f_loglik, "typdev": f_typdev,
}


# ── model + sampling ─────────────────────────────────────────────────────────────────────────────────


def build_config(section: str, boundaries: list[int] | None, rope_fp32: bool) -> KotodamaModelConfig:
    """Shape from kotodama.presets (+ its default boundaries); the serving numerics (sdpa, no liger)."""
    bnd = list(boundaries) if boundaries else list(presets.DEFAULT_BOUNDARIES[presets.resolve(section)])
    return KotodamaModelConfig(**presets.shape(section), z_loss_weight=0.0, use_liger=False,
                               attn_impl="sdpa", attn_res=True, attn_res_boundaries=bnd, rope_fp32=rope_fp32)


def load_model(path: str, cfg: KotodamaModelConfig, device: torch.device) -> KotodamaModel:
    model = KotodamaModel(cfg)
    state = ckpt.load(path)
    model.load_state_dict(ckpt.model_state(state), strict=True)
    del state
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.bfloat16()
    model.requires_grad_(False)
    return model


@dataclass
class Seq:
    ids: Tensor        # [T] long, prompt + generated (first stop token trimmed off)
    p: int             # prompt length
    prompt_idx: int


@torch.no_grad()
def sample_batch(model: KotodamaModel, prompt_ids: list[int], k: int, max_new: int, temperature: float,
                 rep_penalty: float, stops: frozenset[int], gen: torch.Generator) -> list[list[int]]:
    """k continuations of one prompt, batched; server sampling law (presence penalty over GENERATED
    tokens: logit>0 divided, else multiplied; pure temperature; top-k/top-p off). Each row stops at
    its first stop token (excluded from the returned ids)."""
    dev = next(model.parameters()).device
    x = torch.tensor([prompt_ids] * k, dtype=torch.long, device=dev)
    logits, _cap, cache = cached_forward(model, x, past=None, position_offset=0, light=True, capture_hidden=False)
    step = logits[:, -1, :].float()
    V = step.shape[-1]
    presence = torch.zeros(k, V, dtype=torch.bool, device=dev)
    alive = torch.ones(k, dtype=torch.bool, device=dev)
    out = torch.zeros(k, 0, dtype=torch.long, device=dev)
    stop_t = torch.tensor(sorted(stops), device=dev)
    pos = len(prompt_ids)
    for _ in range(max_new):
        if pos >= model.rope_cos.shape[0]:
            break
        lg = step
        if rep_penalty != 1.0:
            pen = torch.where(lg > 0, lg / rep_penalty, lg * rep_penalty)
            lg = torch.where(presence, pen, lg)
        probs = torch.softmax(lg / max(temperature, 1e-6), dim=-1)
        nxt = torch.multinomial(probs, 1, generator=gen)          # (k, 1)
        nxt[~alive] = int(stop_t[0])
        presence.scatter_(1, nxt, True)
        out = torch.cat([out, nxt], dim=1)
        alive &= ~torch.isin(nxt.squeeze(1), stop_t)
        if not bool(alive.any()):
            break
        logits, _cap, cache = cached_forward(model, nxt, past=cache, position_offset=pos, light=True,
                                             capture_hidden=False)
        pos += 1
        step = logits[:, -1, :].float()
    rows: list[list[int]] = []
    for r in out.tolist():
        cut = next((i for i, t in enumerate(r) if t in stops), len(r))
        rows.append(r[:cut])
    return rows


def sample_seqs(model, tok, prompts: list[str], n: int, per_prompt: int, max_new: int, temperature: float,
                rep_penalty: float, min_new: int, seed: int, prompt_offset: int = 0) -> list[Seq]:
    """n continuations, ``per_prompt`` at a time, cycling prompts from ``prompt_offset``."""
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    out: list[Seq] = []
    j = prompt_offset
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > 4 * max(1, n // per_prompt) + 8:
            raise RuntimeError(f"only {len(out)}/{n} continuations with >= {min_new} new tokens")
        pi = j % len(prompts)
        j += 1
        pids = tok.encode(prompts[pi])
        for g in sample_batch(model, pids, per_prompt, max_new, temperature, rep_penalty,
                              BASE_STOP_TOKEN_IDS, gen):
            if len(g) >= min_new and len(out) < n:
                out.append(Seq(torch.tensor(pids + g, dtype=torch.long, device=dev), len(pids), pi))
    return out


# ── construction ─────────────────────────────────────────────────────────────────────────────────────


@dataclass
class SiteSpectrum:
    evals: np.ndarray            # descending
    evecs: np.ndarray            # columns
    median_norm: float           # median ‖h_t‖ over all pooled generated tokens (the α scale)
    median_of_medians: float     # egv's convention, recorded for comparison
    n_positions: int


@torch.no_grad()
def spectrum(model: KotodamaModel, seqs: list[Seq], layer: int) -> SiteSpectrum:
    sx: np.ndarray | None = None
    sxx: np.ndarray | None = None
    n = 0
    norms: list[np.ndarray] = []
    with site_leaf(model, layer) as box:
        for s in seqs:
            model(s.ids[None])
            r = box["h"][0, s.p:].double().cpu().numpy()
            if sx is None:
                sx, sxx = np.zeros(r.shape[1]), np.zeros((r.shape[1],) * 2)
            sx += r.sum(0)
            sxx += r.T @ r
            n += len(r)
            norms.append(np.linalg.norm(r, axis=1))
    assert sx is not None and sxx is not None
    mu = sx / n
    w, V = np.linalg.eigh((sxx - n * np.outer(mu, mu)) / (n - 1))
    order = np.argsort(w)[::-1]
    allnorm = np.concatenate(norms)
    return SiteSpectrum(w[order], V[:, order], float(np.median(allnorm)),
                        float(np.median([np.median(x) for x in norms])), n)


def gradients(model: KotodamaModel, seqs: list[Seq], layer: int, names: list[str]
              ) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Per sequence, the mean over generated positions of ∂F/∂h_t: {name: [N, d]}; plus F values."""
    out: dict[str, list[np.ndarray]] = {k: [] for k in names}
    vals: dict[str, list[float]] = {k: [] for k in names}
    with site_leaf(model, layer) as box, torch.enable_grad():
        for s in seqs:
            logits = model(s.ids[None])["logits"][0].float()
            leaf = box["h"]
            for i, k in enumerate(names):
                F = FUNCTIONALS[k](logits, s.ids, s.p)
                (g,) = torch.autograd.grad(F, leaf, retain_graph=i < len(names) - 1)
                gg = g[0, s.p:].double().mean(0).cpu().numpy()
                if not np.all(np.isfinite(gg)):
                    raise FloatingPointError(f"non-finite gradient: {k} @ layer {layer}")
                out[k].append(gg)
                vals[k].append(float(F.detach()))
            del logits, leaf
            box.clear()
    return {k: np.stack(v) for k, v in out.items()}, vals


def unit(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if not math.isfinite(n) or n <= 1e-12:
        raise ValueError("zero / non-finite vector")
    return (x / n).astype(np.float32)


def diagnostics(G: np.ndarray, U: np.ndarray) -> dict[str, float | str]:
    """egv.diagnostics: band_mass = ‖Uᵀg‖/‖g‖; agree = #seqs whose band coords align with the mean's;
    pairwise_cos = mean cosine between per-sequence band coordinates."""
    g = G.mean(0)
    B = G @ U
    Bu = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-30, None)
    iu = np.triu_indices(len(G), 1)
    ref = U.T @ g
    return {
        "band_mass": float(np.linalg.norm(ref) / np.linalg.norm(g)),
        "pairwise_cos": float((Bu @ Bu.T)[iu].mean()) if len(G) > 1 else float("nan"),
        "agree": f"{int((B @ ref > 0).sum())}/{len(G)}",
        "raw_norm": float(np.linalg.norm(g)),
    }


@dataclass
class SiteResult:
    bank: dict[str, np.ndarray] = field(default_factory=dict)
    raw: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def build_site(model: KotodamaModel, sig_seqs: list[Seq], grad_seqs: list[Seq], layer: int,
               names: list[str], band: tuple[int, int]) -> SiteResult:
    t0 = time.time()
    spec = spectrum(model, sig_seqs, layer)
    lo, hi = band
    if not 0 <= lo < hi <= len(spec.evals):
        raise ValueError(f"band {band} outside [0, {len(spec.evals)}]")
    U = spec.evecs[:, lo:hi]
    G, vals = gradients(model, grad_seqs, layer, names)
    res = SiteResult()
    vec: dict[str, np.ndarray] = {}
    diag: dict[str, dict] = {}
    for k, Gk in G.items():
        vec[k] = unit(U @ (U.T @ Gk.mean(0)))
        diag[k] = diagnostics(Gk, U)
        diag[k]["F_mean"] = float(np.mean(vals[k]))
        res.raw[sublayer_key(f"raw_{VEC_NAME[k]}", layer)] = Gk.mean(0).astype(np.float32)
        res.raw[sublayer_key(f"seqgrads_{VEC_NAME[k]}", layer)] = Gk.astype(np.float32)
        res.bank[sublayer_key(VEC_NAME[k], layer)] = vec[k]
    cos = {f"{VEC_NAME[a]}·{VEC_NAME[b]}": float(vec[a] @ vec[b]) for a, b in combinations(names, 2)}
    perp: dict[str, dict] = {}
    if "entropy" in vec:
        v7 = vec["entropy"]
        for k in names:
            if k == "entropy":
                continue
            c = float(vec[k] @ v7)
            r = vec[k] - c * v7
            pn = float(np.linalg.norm(r))
            if pn < 1e-6:
                raise ValueError(f"{VEC_NAME[k]} @ layer {layer} collapses onto V7 (⊥ residual {pn:.2e})")
            u = unit(r)
            if abs(float(u @ v7)) > 1e-5:
                raise AssertionError(f"⊥ failed: {VEC_NAME[k]}perp·V7 = {float(u @ v7):.2e}")
            res.bank[sublayer_key(f"{VEC_NAME[k]}perp", layer)] = u
            perp[VEC_NAME[k]] = {"cos_to_V7": c, "perp_residual_fraction": pn}
        if "selfrep" in vec:
            rep = res.bank[sublayer_key("Vrepperp", layer)]
            for k in names:
                if k in ("entropy", "selfrep"):
                    continue
                r = vec[k] - float(vec[k] @ v7) * v7
                c2 = float(r @ rep)
                r = r - c2 * rep
                pn2 = float(np.linalg.norm(r))
                if pn2 < 1e-6:
                    raise ValueError(f"{VEC_NAME[k]} @ layer {layer} collapses onto span(V7, Vrepperp)")
                res.bank[sublayer_key(f"{VEC_NAME[k]}perp2", layer)] = unit(r)
                pn1 = perp[VEC_NAME[k]]["perp_residual_fraction"]
                perp[VEC_NAME[k]].update({"cos_perp_to_Vrepperp": c2 / pn1, "perp2_residual_fraction": pn2})
    rng = np.random.default_rng(RANDOM_SEED + layer)
    rnd = [unit(U @ rng.standard_normal(U.shape[1])) for _ in range(N_RANDOM)]
    for i, r in enumerate(rnd, 1):
        res.bank[sublayer_key(f"Rband{i}", layer)] = r
    res.bank[f"median_norm_s{2 * layer}"] = np.array(spec.median_norm, dtype=np.float64)
    res.raw[sublayer_key("evals", layer)] = spec.evals.astype(np.float32)
    rand_cos = {f"Rband{i}·{VEC_NAME[k]}": float(r @ vec[k]) for i, r in enumerate(rnd, 1) for k in names}
    res.meta = {
        "layer": layer, "sublayer": 2 * layer,
        "loop_source_layer": model.config.layer_source(layer),
        "steer_rows": list(compute_steer_points(layer, model._attn_res_boundary_set, model.config.num_layers)),
        "median_norm": spec.median_norm, "median_of_medians": spec.median_of_medians,
        "n_sigma_positions": spec.n_positions,
        "band_eval_range": [float(spec.evals[lo]), float(spec.evals[hi - 1])],
        "band_var_fraction": float(spec.evals[lo:hi].sum() / spec.evals.sum()),
        "diagnostics": {VEC_NAME[k]: d for k, d in diag.items()},
        "cos": cos, "perp": perp, "random_cos": rand_cos,
        "max_abs_random_cos": float(max(abs(v) for v in rand_cos.values())) if rand_cos else None,
        "seconds": round(time.time() - t0, 1),
    }
    return res


# ── cli ──────────────────────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-section", default="1b-moonshot-loop", help="kotodama.presets shape name")
    ap.add_argument("--boundaries", default=None, help="comma list; default = the preset's")
    ap.add_argument("--rope-fp32", action="store_true")
    ap.add_argument("--sites", required=True, help="comma list of EFFECTIVE layer indices")
    ap.add_argument("--functionals", nargs="+", default=["entropy", "selfrep", "margin"],
                    choices=sorted(FUNCTIONALS))
    ap.add_argument("--n-grad", type=int, default=20)
    ap.add_argument("--n-sigma", type=int, default=60)
    ap.add_argument("--per-prompt", type=int, default=4, help="continuations sampled per prompt (one batch)")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--min-new", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--rep-penalty", type=float, default=1.2)
    ap.add_argument("--prompts", type=Path, required=True,
                    help="plain-text openings, one per line; the vectors are only as general as these prompts")
    ap.add_argument("--band", type=int, nargs=2, default=list(BAND))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    names = list(dict.fromkeys(a.functionals))
    if "repmass" in names and "selfrep" in names:
        logger.warning("both repmass and selfrep requested — distinct functionals, both banked")
    sites = [int(x) for x in a.sites.split(",")]
    bnd = [int(x) for x in a.boundaries.split(",")] if a.boundaries else None
    cfg = build_config(a.config_section, bnd, a.rope_fp32)
    for s in sites:
        if not 0 <= s < cfg.num_layers:
            raise SystemExit(f"site {s} outside [0, {cfg.num_layers})")
    prompts = [ln.strip() for ln in a.prompts.read_text().splitlines() if ln.strip()]
    if not prompts:
        raise SystemExit(f"no prompts in {a.prompts}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
    dev = torch.device(a.device)
    t0 = time.time()
    model = load_model(a.ckpt, cfg, dev)
    logger.info("model loaded (%s, %d effective layers, boundaries %s) in %.0fs", a.config_section,
                cfg.num_layers, sorted(model._attn_res_boundary_set), time.time() - t0)

    # Σ and g from DISJOINT continuations: Σ cycles prompts from 0, g starts half-way round the list.
    t0 = time.time()
    sig = sample_seqs(model, tok, prompts, a.n_sigma, a.per_prompt, a.max_new, a.temperature,
                      a.rep_penalty, a.min_new, a.seed)
    grd = sample_seqs(model, tok, prompts, a.n_grad, a.per_prompt, a.max_new, a.temperature,
                      a.rep_penalty, a.min_new, a.seed + 1, prompt_offset=len(prompts) // 2)
    logger.info("sampled %d + %d continuations in %.0fs (mean new tokens %.0f / %.0f)", len(sig), len(grd),
                time.time() - t0, np.mean([len(s.ids) - s.p for s in sig]),
                np.mean([len(s.ids) - s.p for s in grd]))

    bank: dict[str, np.ndarray] = {}
    raw: dict[str, np.ndarray] = {}
    meta: dict = {
        "ckpt": a.ckpt, "config_section": a.config_section, "boundaries": sorted(model._attn_res_boundary_set),
        "rope_fp32": a.rope_fp32, "functionals": names, "band": a.band, "n_sigma": len(sig), "n_grad": len(grd),
        "sampling": {"temperature": a.temperature, "rep_penalty": a.rep_penalty, "top_p": None,
                     "max_new": a.max_new, "min_new": a.min_new, "per_prompt": a.per_prompt, "seed": a.seed,
                     "chat_template": False},
        "site": "AttnRes running partial after layer L's attention sublayer (= DecodeEngine steer_buf[L])",
        "keys": "unit vectors <name>_s{2L}; median_norm_s{2L}; ⊥ = Gram-Schmidt vs V7_s{2L}",
        "sign": "+Vrep raises self-repeat mass: anti-repetition = NEGATIVE alpha on Vrepperp",
        "sites": {},
    }
    for layer in sites:
        r = build_site(model, sig, grd, layer, names, tuple(a.band))
        bank.update(r.bank)
        raw.update(r.raw)
        meta["sites"][str(layer)] = r.meta
        d = r.meta["diagnostics"]
        logger.info("site %d (s%d): %s | cos %s | perp %s | med_norm %.1f", layer, 2 * layer,
                    {k: (round(v["band_mass"], 3), v["agree"], round(v["pairwise_cos"], 3)) for k, v in d.items()},
                    {k: round(v, 3) for k, v in r.meta["cos"].items()},
                    {k: round(v["cos_to_V7"], 3) for k, v in r.meta["perp"].items()}, r.meta["median_norm"])

    out = a.out if a.out.suffix == ".npz" else a.out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **bank)
    np.savez(out.with_name(out.stem + "_raw.npz"), **raw)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    with out.with_name(out.stem + "_samples.jsonl").open("w") as fh:
        for split, seqs in (("sigma", sig), ("grad", grd)):
            for s in seqs:
                ids = s.ids.tolist()
                fh.write(json.dumps({"split": split, "prompt_idx": s.prompt_idx, "p": s.p, "ids": ids,
                                     "text": tok.decode(ids[s.p:])}) + "\n")
    # Round-trip through the server's loader: the bank must be servable as written.
    from kotodama.serve.steering import load_steer_banks
    loaded, meds, _aliases = load_steer_banks([str(out)], cfg.num_layers, cfg.hidden_size, torch.device("cpu"))
    logger.info("bank %s: %d vectors, sites %s, meds %s", out, len(loaded),
                sorted({s for s, _ in loaded.values()}), {k: round(v, 1) for k, v in meds.items()})
    print(json.dumps(meta["sites"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
