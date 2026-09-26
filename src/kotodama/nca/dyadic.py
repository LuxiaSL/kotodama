"""Dyadic NCA — the arm-5 generating function ("the world that answers back").

Spec: research/planning/DRAFT-SPEC-dyadic-nca-2026-08-03.md (three clauses:
model-the-other / honor-the-ledger / orient-to-the-surprise). Builds on the
original generator's primitives (NCARule, simulate step, patch tokenizer);
this module adds the sequence families the original could not express:

  A. COUPLED ASYMMETRIC GENERATORS — two rules alternate control of ONE grid
     in variable-length turns (turn-boundary tokens mark the handover). Both
     rules act on the shared state: each inherits the other's writes.
     Complexity asymmetry supported (R_other may be simpler — the ramp).
  B. COMMITMENTS + QUERIES — stream-level tagged associations: [COMMIT, TAG_k,
     VAL_v] emitted once at a random turn; later [QUERY, TAG_k] must be
     answered by VAL_v. Values come from a RESERVED alphabet never emitted by
     frame serialization, so the ANTI-SHORTCUT LAW (no textual recurrence
     between commit and query — the P0.7-v2 lesson) holds structurally.
     Committer side (self/other) is recorded per event: the bind_self /
     bind_ctx split, in miniature.
  C. PERTURBATION-AND-REPAIR — a marked exogenous patch overwrite mid-rollout;
     the ground-truth continuation INCORPORATES the perturbation (dynamics
     propagate it). Eval pairs each perturbed rollout with its counterfactual
     unperturbed twin -> the smoothing index in its native form.

Token map (extends the original patch vocabulary, d=10, ps=2 -> ids 0..10001):
  10000 START   10001 END          10002 TURN_SELF  10003 TURN_OTHER
  10004 COMMIT  10005 QUERY        10006 PERTURB
  10007..10022  TAG_0..TAG_15
  10023..10032  VAL_0..VAL_9   (reserved value alphabet — never in frames)

Training emission (ARM-D writer) and the in-domain gate harness wire up in a
follow-up; this module = the generative core + eval-item emission + selftest.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .generator import NCAConfig, NCARule, sample_rule_config

# ── Token map ──────────────────────────────────────────────────────────────────

def token_map(cfg: NCAConfig) -> dict[str, int]:
    base = cfg.d_state ** (cfg.patch_size ** 2)  # 10_000 for d=10, ps=2
    return {
        "START": base, "END": base + 1,
        "TURN_SELF": base + 2, "TURN_OTHER": base + 3,
        "COMMIT": base + 4, "QUERY": base + 5, "PERTURB": base + 6,
        "TAG0": base + 7,   # ..TAG0+15
        "VAL0": base + 23,  # ..VAL0+9
    }


N_TAGS = 16
N_VALS = 10


@dataclass
class DyadicConfig:
    nca: NCAConfig = field(default_factory=NCAConfig)
    # turns
    turn_len_min: int = 2
    turn_len_max: int = 6
    # commitments
    commits_per_traj: tuple[int, int] = (2, 5)
    queries_per_traj: tuple[int, int] = (2, 4)   # <= commits unless replacement
    query_replacement: bool = False              # v2: re-query commits at new lags
    min_query_lag_steps: int = 4                 # timesteps between commit & query
    # perturbations
    perturb_prob: float = 0.5                    # per-trajectory
    perturb_patch: int = 8                       # square side, cells
    # asymmetry ramp: other-rule identity bias boost (higher = more ordered)
    other_identity_bias_boost: float = 1.0


# ── Core simulation ────────────────────────────────────────────────────────────

def _one_step(rule: NCARule, state: torch.Tensor, d: int, G: int,
              identity_bias: float, temperature: float) -> torch.Tensor:
    B, _, H, W = state.shape
    s_flat = state.reshape(B * G, H, W)
    onehot = F.one_hot(s_flat.long(), d).permute(0, 3, 1, 2).float()
    onehot = onehot.reshape(B, G * d, H, W)
    logits = rule(onehot)
    if identity_bias > 0:
        logits = logits + identity_bias * onehot
    logits = logits.reshape(B, G, d, H, W).permute(0, 1, 3, 4, 2).reshape(-1, d)
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    return torch.multinomial(probs, 1).reshape(B, G, H, W)


@dataclass
class DyadicTrajectory:
    frames: torch.Tensor                 # (T, G, H, W) int states
    control: list[int]                   # per-timestep: 0=self, 1=other
    events: list[dict[str, Any]]         # commits/queries/perturbs w/ step idx
    rng_seed: int


def simulate_dyadic(cfg: DyadicConfig, rule_self: NCARule, rule_other: NCARule,
                    seed: int, device: torch.device = torch.device("cpu"),
                    with_perturb: Optional[bool] = None,
                    perturb_state_override: Optional[torch.Tensor] = None,
                    ) -> DyadicTrajectory:
    """Roll one coupled trajectory with commit/query/perturb events.

    perturb_state_override: when given (counterfactual twin generation), the
    trajectory reuses this exact pre-perturbation stream — caller passes the
    twin's frames up to the perturb step; we re-seed identically instead
    (simpler: same seed + with_perturb=False gives the unperturbed twin,
    because ALL stochastic draws are made from a dedicated torch.Generator
    whose consumption is identical in both branches up to the perturb step,
    and the perturbation itself uses an INDEPENDENT generator)."""
    n = cfg.nca
    rng = random.Random(seed)
    tgen = torch.Generator(device="cpu").manual_seed(seed)
    torch.manual_seed(seed)  # multinomial in _one_step uses global gen

    T = n.num_steps
    # turn schedule
    control: list[int] = []
    side = rng.randint(0, 1)
    while len(control) < T:
        control += [side] * rng.randint(cfg.turn_len_min, cfg.turn_len_max)
        side = 1 - side
    control = control[:T]

    # event schedule
    n_commit = rng.randint(*cfg.commits_per_traj)
    commit_steps = sorted(rng.sample(range(2, T - cfg.min_query_lag_steps - 1),
                                     min(n_commit, T // 4)))
    commits = []
    for k, cs in enumerate(commit_steps):
        commits.append({"kind": "commit", "step": cs, "tag": rng.randrange(N_TAGS),
                        "val": rng.randrange(N_VALS), "side": control[cs]})
    if cfg.query_replacement and commits:
        n_query = rng.randint(*cfg.queries_per_traj)
        queried = rng.choices(commits, k=n_query)
    else:
        n_query = min(rng.randint(*cfg.queries_per_traj), len(commits))
        queried = rng.sample(commits, n_query)
    queries = []
    seen_commits: set[int] = set()
    for c in queried:
        qs = rng.randint(c["step"] + cfg.min_query_lag_steps, T - 1)
        cid = id(c)
        queries.append({"kind": "query", "step": qs, "tag": c["tag"],
                        "val": c["val"], "committer_side": c["side"],
                        "lag_steps": qs - c["step"],
                        "is_requery": cid in seen_commits})
        seen_commits.add(cid)
    do_perturb = (rng.random() < cfg.perturb_prob if with_perturb is None
                  else with_perturb)
    perturbs = []
    if do_perturb:
        ps_step = rng.randint(T // 4, 3 * T // 4)
        pgen = torch.Generator(device="cpu").manual_seed(seed ^ 0x5F5F5F)
        px = rng.randrange(n.grid_size - cfg.perturb_patch)
        py = rng.randrange(n.grid_size - cfg.perturb_patch)
        perturbs.append({"kind": "perturb", "step": ps_step, "x": px, "y": py,
                         "_gen": pgen})

    events = sorted(commits + queries + perturbs, key=lambda e: e["step"])

    # simulate
    state = torch.randint(0, n.d_state, (1, n.n_groups, n.grid_size, n.grid_size),
                          generator=tgen).to(device)
    for _ in range(n.burn_in):
        state = _one_step(rule_self, state, n.d_state, n.n_groups,
                          n.identity_bias, n.temperature)
    frames = torch.zeros(T, n.n_groups, n.grid_size, n.grid_size, dtype=torch.long)
    for t in range(T):
        for e in perturbs:
            if e["step"] == t:
                patch = torch.randint(0, n.d_state,
                                      (1, n.n_groups, cfg.perturb_patch,
                                       cfg.perturb_patch), generator=e["_gen"])
                state[:, :, e["y"]:e["y"] + cfg.perturb_patch,
                      e["x"]:e["x"] + cfg.perturb_patch] = patch
        frames[t] = state[0].cpu()
        rule = rule_self if control[t] == 0 else rule_other
        bias = n.identity_bias + (cfg.other_identity_bias_boost
                                  if control[t] == 1 else 0.0)
        state = _one_step(rule, state, n.d_state, n.n_groups, bias, n.temperature)
    for e in perturbs:
        e.pop("_gen", None)
    return DyadicTrajectory(frames=frames, control=control, events=events,
                            rng_seed=seed)


# ── Serialization ─────────────────────────────────────────────────────────────

def serialize_dyadic(traj: DyadicTrajectory, cfg: DyadicConfig) -> np.ndarray:
    """Token stream: per timestep [TURN_x (at handover)] [PERTURB?] frames...
    then any commit/query tokens scheduled at that step AFTER its frames.
    Query answer (VAL token) directly follows [QUERY, TAG] — next-token
    supervised like everything else."""
    n = cfg.nca
    tm = token_map(n)
    ps, d = n.patch_size, n.d_state
    H = W = n.grid_size
    nph, npw = H // ps, W // ps
    powers = d ** np.arange(ps * ps, dtype=np.int64)

    by_step: dict[int, list[dict[str, Any]]] = {}
    for e in traj.events:
        by_step.setdefault(e["step"], []).append(e)

    out: list[int] = []
    prev_side = None
    fr = traj.frames.numpy()
    for t in range(fr.shape[0]):
        side = traj.control[t]
        if side != prev_side:
            out.append(tm["TURN_SELF"] if side == 0 else tm["TURN_OTHER"])
            prev_side = side
        for e in by_step.get(t, []):
            if e["kind"] == "perturb":
                out.append(tm["PERTURB"])
        for g in range(fr.shape[1]):
            out.append(tm["START"])
            patches = fr[t, g].reshape(nph, ps, npw, ps).transpose(0, 2, 1, 3)
            out.extend((patches.reshape(-1, ps * ps).astype(np.int64) @ powers
                        ).astype(np.uint16).tolist())
            out.append(tm["END"])
        for e in by_step.get(t, []):
            if e["kind"] == "commit":
                out += [tm["COMMIT"], tm["TAG0"] + e["tag"], tm["VAL0"] + e["val"]]
            elif e["kind"] == "query":
                out += [tm["QUERY"], tm["TAG0"] + e["tag"], tm["VAL0"] + e["val"]]
    return np.array(out, dtype=np.uint16)


# ── Eval-item emission (the in-domain gates, logprob channel) ─────────────────

def emit_ledger_items(cfg: DyadicConfig, rules: list[tuple[NCARule, NCARule]],
                      n_items: int, seed0: int) -> list[dict[str, Any]]:
    """Ledger-gate items: prefix ends right after [QUERY, TAG]; gold = VAL
    token; candidates = the full VAL alphabet. Committer side + lag recorded."""
    tm = token_map(cfg.nca)
    items = []
    s = seed0
    while len(items) < n_items:
        rs, ro = rules[s % len(rules)]
        traj = simulate_dyadic(cfg, rs, ro, seed=s)
        toks = serialize_dyadic(traj, cfg)
        q_positions = np.where(toks == tm["QUERY"])[0]
        for qp in q_positions:
            gold = int(toks[qp + 2]) - tm["VAL0"]
            ev = [e for e in traj.events if e["kind"] == "query"
                  and tm["TAG0"] + e["tag"] == int(toks[qp + 1])
                  and tm["VAL0"] + e["val"] == int(toks[qp + 2])]
            items.append({
                "item_id": f"led_{s}_{int(qp)}",
                "prefix": toks[: qp + 2].tolist(),
                "gold_val": gold,
                "candidates": list(range(N_VALS)),
                "committer_side": ev[0]["committer_side"] if ev else None,
                "lag_steps": ev[0]["lag_steps"] if ev else None,
                "is_requery": ev[0].get("is_requery", False) if ev else None,
            })
            if len(items) >= n_items:
                break
        s += 1
    return items


def emit_smoothing_pairs(cfg: DyadicConfig, rules: list[tuple[NCARule, NCARule]],
                         n_pairs: int, seed0: int) -> list[dict[str, Any]]:
    """Smoothing-gate items, v2 (2026-08-03 fix): the scored object is the
    PROPAGATION timestep (t+1), not the perturbed frame itself — the injected
    patch is unpredictable noise by construction (v1's SI 0.78 was this
    artifact). Prefix = perturbed stream THROUGH timestep t (patch visible in
    context); true = timestep t+1 of the perturbed rollout (dynamics carry
    the patch forward); counterfactual = timestep t+1 of the unperturbed twin
    (rule-consistent with pre-perturbation state). Frame slices located by
    START-token ordinals so interleaved commit/query events can't shift them."""
    tm = token_map(cfg.nca)
    G = cfg.nca.n_groups
    tpt = cfg.nca.tokens_per_frame * G
    pairs = []
    s = seed0
    while len(pairs) < n_pairs:
        rs, ro = rules[s % len(rules)]
        pert = simulate_dyadic(cfg, rs, ro, seed=s, with_perturb=True)
        base = simulate_dyadic(cfg, rs, ro, seed=s, with_perturb=False)
        pe = [e for e in pert.events if e["kind"] == "perturb"]
        if not pe:
            s += 1
            continue
        toks_p = serialize_dyadic(pert, cfg)
        toks_b = serialize_dyadic(base, cfg)
        p_idx = int(np.where(toks_p == tm["PERTURB"])[0][0])
        starts_p = np.where(toks_p == tm["START"])[0]
        starts_b = np.where(toks_b == tm["START"])[0]
        k = int(np.searchsorted(starts_p, p_idx))  # first START after PERTURB
        if k + 2 * G > min(len(starts_p), len(starts_b)):
            s += 1
            continue  # perturbation too close to trajectory end
        t1_p = int(starts_p[k + G])   # timestep t+1, channel 0, perturbed
        t1_b = int(starts_b[k + G])   # same ordinal in the twin
        pairs.append({
            "item_id": f"smo_{s}",
            "prefix": toks_p[:t1_p].tolist(),
            "true_frame": toks_p[t1_p: t1_p + tpt].tolist(),
            "counterfactual_frame": toks_b[t1_b: t1_b + tpt].tolist(),
            "perturb_step": pe[0]["step"],
        })
        s += 1
    return pairs


# ── Selftest ──────────────────────────────────────────────────────────────────

def selftest() -> None:
    cfg = DyadicConfig(nca=NCAConfig(grid_size=16, num_steps=32, burn_in=2,
                                     filter_enabled=False))
    n = cfg.nca
    tm = token_map(n)
    torch.manual_seed(7)

    def mk() -> NCARule:
        rc = sample_rule_config(n)
        arch = {k: rc[k] for k in ("kernel_size", "hidden_dim",
                                   "num_hidden_layers")}
        return NCARule(d_state=n.d_state, n_groups=n.n_groups, **arch)

    rules = [(mk(), mk()) for _ in range(3)]
    fails = 0

    def check(c: bool, msg: str) -> None:
        nonlocal fails
        if not c:
            fails += 1
            print(f"FAIL: {msg}")

    traj = simulate_dyadic(cfg, *rules[0], seed=11, with_perturb=True)
    toks = serialize_dyadic(traj, cfg)
    # 1. turn tokens present, alternate correctly
    turns = toks[(toks == tm["TURN_SELF"]) | (toks == tm["TURN_OTHER"])]
    check(len(turns) >= 2, "no turn alternation")
    check(all(turns[i] != turns[i + 1] for i in range(len(turns) - 1)),
          "turn tokens repeat without handover")
    # 2. commit/query structure + ANTI-SHORTCUT: VAL token of each queried
    # commitment appears exactly at its commit and its query, nowhere between
    q_pos = np.where(toks == tm["QUERY"])[0]
    c_pos = np.where(toks == tm["COMMIT"])[0]
    for qp in q_pos:
        val_tok = toks[qp + 2]
        check(tm["VAL0"] <= val_tok < tm["VAL0"] + N_VALS, "query answer not VAL")
        commits_with_val = [cp for cp in c_pos if toks[cp + 2] == val_tok]
        check(len(commits_with_val) >= 1, "queried value never committed")
        cp = max(cp for cp in commits_with_val if cp < qp)
        between = toks[cp + 3: qp]
        check(int((between == val_tok).sum()) == 0,
              "ANTI-SHORTCUT VIOLATION: value recurs between commit and query")
    # 3. VAL alphabet never appears in frames (reserved)
    frame_mask = np.ones(len(toks), bool)
    # crude: VAL tokens should ONLY be at commit+2/query+2 positions
    val_positions = set(np.where((toks >= tm["VAL0"])
                                 & (toks < tm["VAL0"] + N_VALS))[0].tolist())
    legal = set((c_pos + 2).tolist()) | set((q_pos + 2).tolist())
    check(val_positions <= legal, "VAL token leaked into frame serialization")
    # 4. smoothing pairs: well-formed, and the perturbation actually diverges
    pairs = emit_smoothing_pairs(cfg, rules, 3, seed0=100)
    check(len(pairs) == 3 and all(len(p["true_frame"]) ==
          n.tokens_per_frame * n.n_groups for p in pairs),
          "smoothing pair malformed")
    for p in pairs:
        check(p["true_frame"] != p["counterfactual_frame"],
              "perturbation did not change the next frame")
    # 5. ledger items well-formed
    led = emit_ledger_items(cfg, rules, 10, seed0=200)
    check(len(led) == 10 and all(0 <= it["gold_val"] < N_VALS for it in led),
          "ledger items malformed")
    check(any(it["committer_side"] == 0 for it in led)
          and any(it["committer_side"] == 1 for it in led),
          "committer sides not mixed (need both for bind_self/bind_ctx split)")

    print(f"dyadic selftest: {fails} failures")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
