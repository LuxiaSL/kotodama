#!/usr/bin/env python3
"""Steering v2 (per-layer block-persistent residual write) unit battery.

CPU-only, tiny fp32 random-weight model — no checkpoint, no GPU, exact
comparisons (fp32 + MATH SDPA, same trick as test_prefix_cache part 1).
What it proves, per docs/STEERING-SERVE.md:

  0. steer-points — compute_steer_points matches the reference persist
     semantics of posttraining/taste/steer_inject.forward_inject (sublayer
     s = 2*layer; {site} + boundaries after it) for every site, incl. the
     production case (site 23, DD-3B boundaries -> [23, 24])
  1. off == absent — an engine built with steer_enabled=True but a zero
     buffer is bitwise-identical to an engine built without the ctor arg
     (prefill + every greedy decode step)
  2. prefill unshifted — engaged writes never touch prefill logits
  3. decode injected exactly — a steered decode step equals an independent
     in-test reference forward that adds the row contents at the documented
     points; plus the running partial at the site layer shifts by EXACTLY
     the write (route-input capture)
  3b. multi-site rows — set_steer populates exactly the expected buffer rows
     (site + later block entries per write); overlapping rows SUM; two
     writes at different sites compose bitwise vs the reference forward
  4. set_steer(None) (and []) restores the unsteered baseline bitwise
  5. error surface — bad shapes/NaNs/scales/malformed writes/disabled
     engines raise
  6. prefix-cache regenerate guard — with writes ENGAGED (incl. multi-site),
     the suffix==1 shortcut (which runs the injecting compiled step on a
     PROMPT token) is bypassed: regenerate logits match an unsteered full
     prefill

GPU/compiled variants share the exact traced code these eager paths run, but
the compiled/cudagraph gates (bitwise-off vs HEAD, steered-serve vs mirror,
throughput) are a GPU battery — see docs/STEERING-SERVE.md.

Usage:
    python scripts/benchmark/test_steering_engine.py            # CPU, anywhere
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

torch.set_num_threads(2)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.decode_engine import DecodeEngine, compute_steer_points  # noqa: E402
from src.model.llama import LuxiaBaseModel, LuxiaModelConfig  # noqa: E402

_FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if cond:
        logger.info("  PASS  %s", msg)
    else:
        logger.error("  FAIL  %s", msg)
        _FAILURES.append(msg)


# Three blocks ([0,2) [2,4) [4,6)) so a site in block 0 exercises TWO
# re-assertion points — richer than SMALL_CONFIG's two blocks.
TINY_CONFIG = dict(
    hidden_size=256, num_layers=6, num_attention_heads=4, num_kv_heads=2,
    head_dim=64, intermediate_size=512, vocab_size=1024,
    max_position_embeddings=512, rope_theta=10000.0, norm_eps=1e-5,
    qk_norm=True, tie_word_embeddings=True, z_loss_weight=0.0,
    use_liger=False, attn_impl="sdpa", attn_res=True,
    attn_res_boundaries=[0, 2, 4],
)
HIDDEN = TINY_CONFIG["hidden_size"]
N_LAYERS = TINY_CONFIG["num_layers"]
BOUNDARIES = TINY_CONFIG["attn_res_boundaries"]
DD3B_BOUNDARIES = [0, 1, 3, 7, 15, 19, 24]


def make_engine(device: torch.device, enabled: bool, max_seq_len: int = 128) -> DecodeEngine:
    torch.manual_seed(7)  # identical weights per construction
    model = LuxiaBaseModel(LuxiaModelConfig(**TINY_CONFIG)).to(device).eval()
    # fp32 + GQA has no flash/efficient kernel — MATH is exact, ideal for units.
    return DecodeEngine(
        model, max_seq_len=max_seq_len, dtype=torch.float32,
        prefill_backends=[SDPBackend.MATH], steer_enabled=enabled,
    )


def rand_ids(n: int, seed: int, vocab: int = 1024) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, vocab, (1, n), generator=g)


def make_vec(scale: float, seed: int = 11) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(HIDDEN, generator=g) * scale


def expected_rows(
    writes: list[tuple[int, torch.Tensor, float]], device: torch.device
) -> torch.Tensor:
    """Reference row accumulation — same order/ops as set_steer's staging."""
    staged = torch.zeros((N_LAYERS, HIDDEN), device=device, dtype=torch.float32)
    for site, vec, scale in writes:
        v = vec.reshape(HIDDEN).to(device=device, dtype=torch.float32)
        for row in compute_steer_points(site, BOUNDARIES, N_LAYERS):
            staged[row] += float(scale) * v
    return staged


# ────────────────────────── reference implementations ───────────────────────

def reference_steer_points(site_layer: int, boundaries: list[int], n_layers: int) -> tuple[int, ...]:
    """Direct port of steer_inject.forward_inject persist_points, translated
    from sublayer indices (after-attn of layer L = sublayer 2*L)."""
    site_sub = 2 * site_layer
    points_sub = {site_sub}
    for bs in boundaries:
        if bs < n_layers and 2 * bs > site_sub:
            points_sub.add(2 * bs)
    return tuple(sorted(s // 2 for s in points_sub))


@torch.inference_mode()
def manual_injected_step(
    eng: DecodeEngine, token: torch.Tensor, row_add: dict[int, torch.Tensor]
) -> torch.Tensor:
    """Port of DecodeEngine._forward_step with the DOCUMENTED steering write
    applied in-test: row_add[layer] added to `partial` right after
    `partial + attn_out` at each populated layer. Runs on a steering-DISABLED
    engine — the independent oracle a steered engine's step must match
    bitwise."""
    model = eng.model
    eng.attn_bias.index_fill_(3, eng.pos, 0.0)
    eng.token_history.index_copy_(0, eng.pos, token.view(1))

    embed = model.embed_tokens(token)
    committed: list[torch.Tensor] = []
    partial = embed
    for ls in eng.schedule:
        layer = model.layers[ls.layer_idx]
        if committed:
            h_attn = eng._route(committed + [partial], layer.attn_res_query, layer.attn_res_norm.weight)
        else:
            h_attn = partial
        if ls.is_boundary:
            committed.append(partial)
            partial = torch.zeros_like(embed)
        attn_out = eng._attn(ls.layer_idx, layer.attn_norm(h_attn))
        partial = partial + attn_out
        if ls.layer_idx in row_add:
            partial = partial + row_add[ls.layer_idx]
        h_mlp = eng._route(committed + [partial], layer.mlp_res_query, layer.mlp_res_norm.weight)
        gu = eng._linear(layer.ffn_norm(h_mlp), eng.w_gate_ups[ls.layer_idx])
        gate, up = gu.split(eng.config.intermediate_size, dim=-1)
        partial = partial + eng._linear(F.silu(gate) * up, layer.ffn.down_proj.weight)

    x = eng._route(committed + [partial], model.final_res_query, model.final_res_norm.weight)
    x = model.norm(x)
    logits = eng._linear(x, model.get_lm_head_weight())
    eng.pos.add_(1)
    return logits


def rows_from_buf(buf: torch.Tensor) -> dict[int, torch.Tensor]:
    """Nonzero buffer rows as a row_add map (fp32 clones)."""
    out: dict[int, torch.Tensor] = {}
    for i in range(buf.shape[0]):
        if bool((buf[i] != 0).any()):
            out[i] = buf[i].to(torch.float32).clone()
    return out


# ──────────────────────────────── the battery ───────────────────────────────

def run_steer_points() -> None:
    logger.info("== 0: steer-points vs the forward_inject reference ==")
    got = compute_steer_points(23, DD3B_BOUNDARIES, 28)
    check(got == (23, 24), f"production case: site 23, DD-3B -> (23, 24) (got {got})")
    check(compute_steer_points(10, DD3B_BOUNDARIES, 28) == (10, 15, 19, 24),
          "site 10 -> (10, 15, 19, 24)")
    check(compute_steer_points(0, DD3B_BOUNDARIES, 28) == (0, 1, 3, 7, 15, 19, 24),
          "site 0 -> itself + every later boundary")
    check(compute_steer_points(27, DD3B_BOUNDARIES, 28) == (27,), "last layer -> itself only")
    check(compute_steer_points(24, DD3B_BOUNDARIES, 28) == (24,),
          "boundary site: no later boundaries -> itself only")
    all_match = all(
        compute_steer_points(s, DD3B_BOUNDARIES, 28) == reference_steer_points(s, DD3B_BOUNDARIES, 28)
        for s in range(28)
    )
    check(all_match, "matches forward_inject persist semantics for ALL 28 sites")
    for bad in (-1, 28):
        try:
            compute_steer_points(bad, DD3B_BOUNDARIES, 28)
            check(False, f"out-of-range site {bad} raises")
        except ValueError:
            check(True, f"out-of-range site {bad} raises")


@torch.inference_mode()
def greedy_logits(eng: DecodeEngine, first_logits: torch.Tensor, n: int) -> list[torch.Tensor]:
    tok = first_logits.argmax(-1).view(1, 1)
    out = []
    for _ in range(n):
        logits = eng.step_logits(tok)
        out.append(logits.clone())
        tok = logits[0, -1].argmax().view(1, 1)
    return out


@torch.inference_mode()
def run_off_identity(device: torch.device) -> None:
    logger.info("== 1: enabled-but-zero engine bitwise-identical to steer-absent engine ==")
    prompt = rand_ids(32, seed=1)
    eng_absent = make_engine(device, enabled=False)
    eng_zero = make_engine(device, enabled=True)
    check(eng_absent.steer_buf is None and not eng_absent.steer_enabled,
          "disabled engine has no buffer")
    check(tuple(eng_zero.steer_buf.shape) == (N_LAYERS, HIDDEN),
          f"per-layer buffer shape ({N_LAYERS}, {HIDDEN})")
    pf_a = eng_absent.prefill(prompt)
    pf_z = eng_zero.prefill(prompt)
    check(torch.equal(pf_a, pf_z), "prefill logits identical")
    steps_a = greedy_logits(eng_absent, pf_a, 8)
    steps_z = greedy_logits(eng_zero, pf_z, 8)
    check(all(torch.equal(a, z) for a, z in zip(steps_a, steps_z)),
          "8 greedy decode steps identical (zero buffer = exact no-op)")


@torch.inference_mode()
def run_prefill_unshifted(device: torch.device) -> None:
    logger.info("== 2: engaged writes never touch prefill ==")
    prompt = rand_ids(32, seed=1)
    eng_absent = make_engine(device, enabled=False)
    eng_s = make_engine(device, enabled=True)
    eng_s.set_steer([(2, make_vec(3.0), 1.0), (0, make_vec(2.0, seed=13), 0.5)])
    check(bool(eng_s._steer_active), "writes engaged (_steer_active)")
    pf_a = eng_absent.prefill(prompt)
    pf_s = eng_s.prefill(prompt)
    check(torch.equal(pf_a, pf_s), "prefill logits identical under engaged multi-site writes")


@torch.inference_mode()
def run_injected_exact(device: torch.device) -> None:
    logger.info("== 3: steered decode step == independent injected reference ==")
    prompt = rand_ids(32, seed=1)
    vec = make_vec(3.0)
    vec32 = vec.reshape(HIDDEN).to(device=device, dtype=torch.float32)

    for site in (0, 2, 5):
        eng_s = make_engine(device, enabled=True)
        eng_m = make_engine(device, enabled=False)  # oracle host: NO engine injection
        eng_b = make_engine(device, enabled=False)  # unsteered baseline
        pf = eng_s.prefill(prompt)
        eng_m.prefill(prompt)
        eng_b.prefill(prompt)
        tok = pf.argmax(-1).view(1, 1)

        eng_s.set_steer([(site, vec, 1.0)])
        points = compute_steer_points(site, BOUNDARIES, N_LAYERS)
        logits_s = eng_s.step_logits(tok)
        logits_m = manual_injected_step(eng_m, tok, {p: vec32 for p in points})
        logits_b = eng_b.step_logits(tok)
        check(torch.equal(logits_s, logits_m),
              f"site {site} (points {points}): engine == injected reference, bitwise")
        check(not torch.equal(logits_s, logits_b), f"site {site}: injection actually moves logits")

    # Route-input capture: the running partial at the site layer shifts by
    # EXACTLY the write (pre-MLP route of layer L is call index 2L).
    site = 2
    caps: dict[str, list[torch.Tensor]] = {}
    for name, writes in (("zero", None), ("vec", [(site, vec, 1.0)])):
        eng = make_engine(device, enabled=True)
        pf = eng.prefill(prompt)
        tok = pf.argmax(-1).view(1, 1)
        if writes is not None:
            eng.set_steer(writes)
        rec: list[torch.Tensor] = []
        orig_route = eng._route

        def spy(sources, query, norm_weight, _rec=rec, _orig=orig_route):
            _rec.append(sources[-1].clone())
            return _orig(sources, query, norm_weight)

        eng._route = spy  # type: ignore[method-assign]
        eng.step_logits(tok)
        caps[name] = rec
    idx = 2 * site
    check(all(torch.equal(a, b) for a, b in zip(caps["zero"][:idx], caps["vec"][:idx])),
          "all route inputs BEFORE the site are untouched")
    check(torch.equal(caps["vec"][idx], caps["zero"][idx] + vec32),
          "partial at the site layer shifts by EXACTLY the write")


@torch.inference_mode()
def run_multisite_rows(device: torch.device) -> None:
    logger.info("== 3b: multi-site writes populate exactly the expected rows ==")
    prompt = rand_ids(32, seed=1)
    va, vb = make_vec(1.0, seed=11), make_vec(1.0, seed=12)
    writes = [(0, va, 2.0), (2, vb, 1.0)]
    # boundaries [0,2,4]: write A (site 0) -> rows {0,2,4} += 2*va;
    #                     write B (site 2) -> rows {2,4} += vb (rows SUM).
    eng = make_engine(device, enabled=True)
    eng.set_steer(writes)
    exp = expected_rows(writes, device)
    check(torch.equal(eng.steer_buf, exp),
          "buffer rows == expected (site + later block entries per write; overlaps sum)")
    check(all(not bool((eng.steer_buf[r] != 0).any()) for r in (1, 3, 5)),
          "non-steer rows (1, 3, 5) stay exactly zero")
    check(bool((eng.steer_buf[0] != 0).any()) and torch.equal(eng.steer_buf[0], 2.0 * va.to(device)),
          "row 0 = 2*va only (write B does not reach earlier blocks)")

    # Behavioral: two writes at different sites compose — engine step ==
    # reference forward fed the engine's own nonzero rows.
    eng_m = make_engine(device, enabled=False)
    eng_b = make_engine(device, enabled=False)
    pf = eng.prefill(prompt)
    eng_m.prefill(prompt)
    eng_b.prefill(prompt)
    tok = pf.argmax(-1).view(1, 1)
    logits_s = eng.step_logits(tok)
    logits_m = manual_injected_step(eng_m, tok, rows_from_buf(exp))
    logits_b = eng_b.step_logits(tok)
    check(torch.equal(logits_s, logits_m),
          "two-site stack: engine == injected reference, bitwise")
    check(not torch.equal(logits_s, logits_b), "two-site stack moves logits")

    # Same-site duplicate writes sum in place.
    eng2 = make_engine(device, enabled=True)
    eng2.set_steer([(2, va, 1.0), (2, va, 1.0)])
    eng3 = make_engine(device, enabled=True)
    eng3.set_steer([(2, va, 2.0)])
    check(torch.equal(eng2.steer_buf, eng3.steer_buf),
          "duplicate same-site writes sum (2 x 1.0 == 1 x 2.0)")


@torch.inference_mode()
def run_none_restores(device: torch.device) -> None:
    logger.info("== 4: set_steer(None)/[] restores the unsteered baseline bitwise ==")
    prompt = rand_ids(32, seed=1)
    eng_b = make_engine(device, enabled=False)
    eng_s = make_engine(device, enabled=True)
    pf_b = eng_b.prefill(prompt)
    base_steps = greedy_logits(eng_b, pf_b, 6)

    eng_s.set_steer([(2, make_vec(3.0), 1.0), (0, make_vec(2.0, seed=13), 1.0)])
    pf1 = eng_s.prefill(prompt)
    steered = greedy_logits(eng_s, pf1, 6)
    check(not all(torch.equal(a, b) for a, b in zip(steered, base_steps)),
          "steered generation diverges from baseline (precondition)")

    eng_s.set_steer(None)
    check(not eng_s._steer_active, "set_steer(None) clears the active flag")
    check(not bool((eng_s.steer_buf != 0).any()), "set_steer(None) zeroes every row")
    pf2 = eng_s.prefill(prompt)
    restored = greedy_logits(eng_s, pf2, 6)
    check(torch.equal(pf2, pf_b) and all(torch.equal(a, b) for a, b in zip(restored, base_steps)),
          "set_steer(None): prefill + 6 decode steps bitwise-restored")

    eng_s.set_steer([(2, make_vec(3.0), 1.0)])
    eng_s.set_steer([])
    check(not eng_s._steer_active and not bool((eng_s.steer_buf != 0).any()),
          "empty writes list also clears")
    pf3 = eng_s.prefill(prompt)
    empty_steps = greedy_logits(eng_s, pf3, 6)
    check(all(torch.equal(a, b) for a, b in zip(empty_steps, base_steps)),
          "set_steer([]) == baseline bitwise")


@torch.inference_mode()
def run_error_surface(device: torch.device) -> None:
    logger.info("== 5: error surface ==")
    eng_plain = make_engine(device, enabled=False)
    try:
        eng_plain.set_steer([(2, make_vec(1.0), 1.0)])
        check(False, "set_steer on steering-disabled engine raises")
    except RuntimeError:
        check(True, "set_steer on steering-disabled engine raises")
    try:
        eng_plain.set_steer(None)
        check(True, "set_steer(None) is a no-op on a disabled engine (finally-safe)")
    except Exception:
        check(False, "set_steer(None) is a no-op on a disabled engine (finally-safe)")

    eng = make_engine(device, enabled=True)
    try:
        eng.set_steer([(2, torch.zeros(HIDDEN + 1), 1.0)])
        check(False, "wrong-size vector raises")
    except ValueError:
        check(True, "wrong-size vector raises")
    bad = make_vec(1.0)
    bad[0] = float("nan")
    try:
        eng.set_steer([(2, bad, 1.0)])
        check(False, "non-finite vector raises")
    except ValueError:
        check(True, "non-finite vector raises")
    try:
        eng.set_steer([(2, make_vec(1.0), float("inf"))])
        check(False, "non-finite scale raises")
    except ValueError:
        check(True, "non-finite scale raises")
    try:
        eng.set_steer([(99, make_vec(1.0), 1.0)])
        check(False, "out-of-range site raises")
    except ValueError:
        check(True, "out-of-range site raises")
    try:
        eng.set_steer([(2, make_vec(1.0))])  # not a triple
        check(False, "malformed write item raises")
    except TypeError:
        check(True, "malformed write item raises")
    check(not eng._steer_active, "failed set_steer calls leave steering off")


@torch.inference_mode()
def run_prefix_regen_guard(device: torch.device) -> None:
    logger.info("== 6: prefix-cache regenerate (suffix==1) guard under multi-site writes ==")
    prompt = rand_ids(32, seed=1)
    huge = make_vec(50.0)  # a write this hot on the tip token would be unmissable

    eng_ctl = make_engine(device, enabled=False)
    ref = eng_ctl.prefill(prompt)  # unsteered full-prefill logits at the tip

    eng = make_engine(device, enabled=True)
    eng.min_cached_prefix = 8
    eng.prefill_cached(prompt)  # miss -> full prefill, cache primed
    eng.set_steer([(2, huge, 1.0), (0, huge, 0.5)])
    logits, info = eng.prefill_cached(prompt)  # regenerate: common=31, suffix=1
    check(info["prefix_hit"] is True and info["suffix_len"] == 1,
          f"regenerate hit with suffix==1 (got {info})")
    d = float((logits - ref).abs().max())
    # Extend path: KV from the same reference batch path, only mask/backend
    # form differs -> near-exact class (test_prefix_cache GATE_MAX_DELTA_SMALL).
    check(d < 1e-3, f"tip token NOT injected: regenerate ~= unsteered prefill (maxD={d:.2e})")

    # Counterfactual: the same tip token through the decode STEP path IS
    # injected — the guard is protecting against a real, visible write.
    eng_cf = make_engine(device, enabled=True)
    eng_cf.prefill(prompt[:, :31])
    eng_cf.set_steer([(2, huge, 1.0)])
    logits_cf = eng_cf.step_logits(prompt[:, 31:32].contiguous())
    d_cf = float((logits_cf[0, -1:].float() - ref).abs().max())
    check(d_cf > 1e-2, f"counterfactual steered step differs materially (maxD={d_cf:.2e})")

    eng.set_steer(None)
    check(not eng._steer_active, "cleared after request (suffix==1 fast path restored)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda[:N]")
    args = parser.parse_args()
    device = torch.device(args.device)

    run_steer_points()
    run_off_identity(device)
    run_prefill_unshifted(device)
    run_injected_exact(device)
    run_multisite_rows(device)
    run_none_restores(device)
    run_error_surface(device)
    run_prefix_regen_guard(device)

    if _FAILURES:
        logger.error("OVERALL: FAIL (%d)", len(_FAILURES))
        for f in _FAILURES:
            logger.error("  - %s", f)
        return 1
    logger.info("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
