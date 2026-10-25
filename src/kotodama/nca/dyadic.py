"""Dyadic NCA — the "world that answers back" generating family, v3.

Three design clauses: model-the-other / honor-the-ledger /
orient-to-the-surprise. Builds on the original generator's primitives (NCARule, simulate step, patch tokenizer);
this module adds the sequence families the original could not express:

  A. COUPLED ASYMMETRIC GENERATORS — two rules alternate control of ONE grid
     in variable-length turns. Both rules act on the shared state: each
     inherits the other's writes. Per-side dynamics (identity_bias /
     temperature) supported. ``mark_turns=False`` drops the TURN tokens: the
     handover must be detected from the dynamics alone (the writer mixes
     marked and unmarked episodes ~half/half).
  B. COMMITMENTS + QUERIES, last-write-wins (v3) — [COMMIT, TAG_k, VAL_v]
     events write a tagged association; [QUERY, TAG_k] must be answered by
     the value of the MOST RECENT commit to that tag — well-posed by
     construction (an overwritten commit is never the gold).
     ``p_tag_reuse`` dials how often commits overwrite a live tag: 0.0 =
     unique-tag pure retrieval, 1.0 = heavy state-update demand. The writer
     ramps it across the stream (unique → overwrite). VALs are
     distinct within an episode, so the ANTI-SHORTCUT LAW holds structurally:
     a gold value's token appears exactly at its own commit and at queries it
     answers, nowhere else.
  C. perturbation and repair — a marked exogenous patch overwrite mid-rollout;
     the ground-truth continuation INCORPORATES the perturbation. Eval pairs
     each perturbed rollout with its counterfactual unperturbed twin.

Token map (d=10, ps=2):
  v1 layout (``reserve_eos=False``, kept readable for replaying v1 bins):  0..9999 patches · 10000 START · 10001 END ·
  10002/10003 TURN_SELF/OTHER · 10004 COMMIT · 10005 QUERY · 10006 PERTURB ·
  10007..10022 TAG0-15 · 10023..10032 VAL0-9.
  v3 layout (``reserve_eos=True``): every id above shifted +1; **id 0 = EOS
  episode separator**, matching the EOS=0 doc-masking contract in
  kotodama.data.dataset (episodes stop bleeding into each other's windows).

The dataset writer (token-level mixing, gzip banding, ramp, sidecar) is
``kotodama.nca.gen_dyadic_data``; this module = the generative core +
eval-item emission + selftest (``python -m kotodama.nca.dyadic --selftest``).
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
    off = 1 if cfg.reserve_eos else 0
    base = cfg.d_state ** (cfg.patch_size ** 2) + off  # 10_000 (+1) for d=10, ps=2
    tm = {
        "START": base, "END": base + 1,
        "TURN_SELF": base + 2, "TURN_OTHER": base + 3,
        "COMMIT": base + 4, "QUERY": base + 5, "PERTURB": base + 6,
        "TAG0": base + 7,   # ..TAG0+15
        "VAL0": base + 23,  # ..VAL0+9
    }
    if off:
        tm["EOS"] = 0
    return tm


N_TAGS = 16
N_VALS = 10


@dataclass
class DyadicConfig:
    nca: NCAConfig = field(default_factory=NCAConfig)
    # turns
    turn_len_min: int = 2
    turn_len_max: int = 6
    mark_turns: bool = True                      # False: no TURN tokens — detect
    # commitments (last-write-wins semantics, v3)
    commits_per_traj: tuple[int, int] = (2, 5)   # capped at N_VALS (distinct vals)
    queries_per_traj: tuple[int, int] = (2, 4)
    p_tag_reuse: float = 0.0                     # 0 = unique tags; 1 = heavy overwrite
    min_query_lag_steps: int = 4                 # timesteps between (last) write & query
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
                    dyn_self: Optional[dict[str, float]] = None,
                    dyn_other: Optional[dict[str, float]] = None,
                    ) -> DyadicTrajectory:
    """Roll one coupled trajectory with commit/query/perturb events.

    Counterfactual twins: same seed + with_perturb=False reproduces the
    unperturbed rollout, because all stochastic draws come from generators
    whose consumption is identical up to the perturb step, and the
    perturbation itself uses an INDEPENDENT generator.

    dyn_self / dyn_other: optional per-side dynamics dicts with
    ``identity_bias`` / ``temperature`` keys (the sampled rule params; without
    them every rule runs at the config defaults). Default None = cfg.nca values.
    """
    n = cfg.nca
    rng = random.Random(seed)
    torch.manual_seed(seed)  # multinomial in _one_step uses global gen
    tgen = torch.Generator(device="cpu").manual_seed(seed)
    d_self = {"identity_bias": n.identity_bias, "temperature": n.temperature}
    d_other = dict(d_self)
    if dyn_self:
        d_self.update(dyn_self)
    if dyn_other:
        d_other.update(dyn_other)

    T = n.num_steps
    # turn schedule
    control: list[int] = []
    side = rng.randint(0, 1)
    while len(control) < T:
        control += [side] * rng.randint(cfg.turn_len_min, cfg.turn_len_max)
        side = 1 - side
    control = control[:T]

    # commit schedule — last-write-wins semantics (v3):
    # tags may repeat (p_tag_reuse); vals are DISTINCT within the episode so
    # the anti-shortcut law holds structurally and gold is never ambiguous.
    n_commit = min(rng.randint(*cfg.commits_per_traj), T // 4, N_VALS)
    commit_steps = sorted(rng.sample(range(2, T - cfg.min_query_lag_steps - 1),
                                     max(n_commit, 0)))
    vals = rng.sample(range(N_VALS), len(commit_steps))
    unused_tags = list(range(N_TAGS))
    rng.shuffle(unused_tags)
    used_tags: list[int] = []
    commits: list[dict[str, Any]] = []
    for k, cs in enumerate(commit_steps):
        if used_tags and (not unused_tags or rng.random() < cfg.p_tag_reuse):
            tag = rng.choice(used_tags)
        else:
            tag = unused_tags.pop()
            used_tags.append(tag)
        commits.append({"kind": "commit", "step": cs, "tag": tag,
                        "val": vals[k], "side": control[cs]})

    # query schedule — target a specific write; the query lands strictly
    # inside [write + min_lag, next write to the same tag), so gold IS the
    # last write before the query, by construction.
    by_tag: dict[int, list[dict[str, Any]]] = {}
    for c in commits:
        by_tag.setdefault(c["tag"], []).append(c)
    n_query = rng.randint(*cfg.queries_per_traj) if commits else 0
    queries: list[dict[str, Any]] = []
    seen_writes: set[int] = set()
    attempts = 0
    while len(queries) < n_query and attempts < n_query * 8:
        attempts += 1
        w = rng.choice(commits)
        writes = by_tag[w["tag"]]
        i = writes.index(w)
        lo = w["step"] + cfg.min_query_lag_steps
        hi = (writes[i + 1]["step"] - 1) if i + 1 < len(writes) else T - 1
        if lo > hi:
            continue
        qs = rng.randint(lo, hi)
        wid = id(w)
        queries.append({"kind": "query", "step": qs, "tag": w["tag"],
                        "val": w["val"], "committer_side": w["side"],
                        "lag_steps": qs - w["step"],
                        "writes_before": i + 1,
                        "post_overwrite": i > 0,
                        "is_requery": wid in seen_writes})
        seen_writes.add(wid)

    do_perturb = (rng.random() < cfg.perturb_prob if with_perturb is None
                  else with_perturb)
    perturbs = []
    if do_perturb:
        ps_step = rng.randint(T // 4, 3 * T // 4)
        pgen = torch.Generator(device="cpu").manual_seed(seed ^ 0x5F5F5F)
        # a perturbation is a PATCH: never larger than half the grid (small
        # grids with the default patch size would hit an empty randrange)
        pp = min(cfg.perturb_patch, n.grid_size // 2)
        px = rng.randrange(n.grid_size - pp)
        py = rng.randrange(n.grid_size - pp)
        perturbs.append({"kind": "perturb", "step": ps_step, "x": px, "y": py,
                         "size": pp, "_gen": pgen})

    events = sorted(commits + queries + perturbs, key=lambda e: e["step"])

    # simulate
    state = torch.randint(0, n.d_state, (1, n.n_groups, n.grid_size, n.grid_size),
                          generator=tgen).to(device)
    for _ in range(n.burn_in):
        state = _one_step(rule_self, state, n.d_state, n.n_groups,
                          d_self["identity_bias"], d_self["temperature"])
    frames = torch.zeros(T, n.n_groups, n.grid_size, n.grid_size, dtype=torch.long)
    for t in range(T):
        for e in perturbs:
            if e["step"] == t:
                pp = e["size"]
                patch = torch.randint(0, n.d_state, (1, n.n_groups, pp, pp),
                                      generator=e["_gen"])
                state[:, :, e["y"]:e["y"] + pp, e["x"]:e["x"] + pp] = patch
        frames[t] = state[0].cpu()
        dyn = d_self if control[t] == 0 else d_other
        rule = rule_self if control[t] == 0 else rule_other
        bias = dyn["identity_bias"] + (cfg.other_identity_bias_boost
                                       if control[t] == 1 else 0.0)
        state = _one_step(rule, state, n.d_state, n.n_groups, bias,
                          dyn["temperature"])
    for e in perturbs:
        e.pop("_gen", None)
    return DyadicTrajectory(frames=frames, control=control, events=events,
                            rng_seed=seed)


# ── Serialization ─────────────────────────────────────────────────────────────

def serialize_dyadic(traj: DyadicTrajectory, cfg: DyadicConfig) -> np.ndarray:
    """Token stream: per timestep [TURN_x (at handover, if mark_turns)]
    [PERTURB?] frames... then any commit/query tokens scheduled at that step
    AFTER its frames. Query answer (VAL token) directly follows [QUERY, TAG]
    — next-token supervised like everything else. With reserve_eos, patch ids
    are shifted +1 (the writer appends the EOS 0 separator; it never appears
    inside an episode)."""
    n = cfg.nca
    tm = token_map(n)
    off = 1 if n.reserve_eos else 0
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
        if cfg.mark_turns and side != prev_side:
            out.append(tm["TURN_SELF"] if side == 0 else tm["TURN_OTHER"])
        prev_side = side
        for e in by_step.get(t, []):
            if e["kind"] == "perturb":
                out.append(tm["PERTURB"])
        for g in range(fr.shape[1]):
            out.append(tm["START"])
            patches = fr[t, g].reshape(nph, ps, npw, ps).transpose(0, 2, 1, 3)
            out.extend((patches.reshape(-1, ps * ps).astype(np.int64) @ powers
                        + off).astype(np.uint16).tolist())
            out.append(tm["END"])
        for e in by_step.get(t, []):
            if e["kind"] == "commit":
                out += [tm["COMMIT"], tm["TAG0"] + e["tag"], tm["VAL0"] + e["val"]]
            elif e["kind"] == "query":
                out += [tm["QUERY"], tm["TAG0"] + e["tag"], tm["VAL0"] + e["val"]]
    return np.array(out, dtype=np.uint16)


def queries_in_stream_order(traj: DyadicTrajectory) -> list[dict[str, Any]]:
    """Query events in exactly the order serialize_dyadic emits them, so the
    k-th QUERY token in the stream maps to the k-th element here."""
    by_step: dict[int, list[dict[str, Any]]] = {}
    for e in traj.events:
        by_step.setdefault(e["step"], []).append(e)
    out = []
    for t in range(traj.frames.shape[0]):
        out.extend(e for e in by_step.get(t, []) if e["kind"] == "query")
    return out


# ── Eval-item emission (the in-domain gates, logprob channel) ─────────────────

RulePair = tuple  # (rule_self, rule_other[, dyn_self, dyn_other])


def _unpack_pair(pair: RulePair):
    if len(pair) == 4:
        return pair
    rs, ro = pair
    return rs, ro, None, None


def emit_ledger_items(cfg: DyadicConfig, rules: list[RulePair],
                      n_items: int, seed0: int) -> list[dict[str, Any]]:
    """Ledger-gate items: prefix ends right after [QUERY, TAG]; gold = VAL of
    the LAST write to that tag (well-posed); candidates = the VAL alphabet.
    Labels carried per item: committer side, lag from the gold write,
    writes_before / post_overwrite (the unique-vs-overwrite eval split),
    is_requery, marked."""
    tm = token_map(cfg.nca)
    items = []
    s = seed0
    while len(items) < n_items:
        rs, ro, ds, do = _unpack_pair(rules[s % len(rules)])
        traj = simulate_dyadic(cfg, rs, ro, seed=s, dyn_self=ds, dyn_other=do)
        toks = serialize_dyadic(traj, cfg)
        q_positions = np.where(toks == tm["QUERY"])[0]
        q_events = queries_in_stream_order(traj)
        assert len(q_positions) == len(q_events), "query stream-order mismatch"
        for qp, ev in zip(q_positions, q_events):
            gold = int(toks[qp + 2]) - tm["VAL0"]
            assert gold == ev["val"], "serialized gold != event gold"
            items.append({
                "item_id": f"led_{s}_{int(qp)}",
                "prefix": toks[: qp + 2].tolist(),
                "gold_val": gold,
                "candidates": list(range(N_VALS)),
                "committer_side": ev["committer_side"],
                "lag_steps": ev["lag_steps"],
                "writes_before": ev["writes_before"],
                "post_overwrite": ev["post_overwrite"],
                "is_requery": ev["is_requery"],
                "marked": cfg.mark_turns,
                "p_tag_reuse": cfg.p_tag_reuse,
            })
            if len(items) >= n_items:
                break
        s += 1
    return items


def emit_smoothing_pairs(cfg: DyadicConfig, rules: list[RulePair],
                         n_pairs: int, seed0: int) -> list[dict[str, Any]]:
    """Smoothing-gate items: the scored object is the PROPAGATION timestep
    (t+1), not the perturbed frame itself — the injected patch is unpredictable
    noise by construction, and scoring it reads a spurious high SI (~0.78).
    Prefix = perturbed stream THROUGH timestep t (patch visible in context);
    true = timestep t+1 of the perturbed rollout (dynamics carry the patch
    forward); counterfactual = timestep t+1 of the unperturbed twin
    (rule-consistent with pre-perturbation state). Frame slices located by
    START-token ordinals so interleaved commit/query events can't shift them."""
    tm = token_map(cfg.nca)
    G = cfg.nca.n_groups
    tpt = cfg.nca.tokens_per_frame * G
    pairs = []
    s = seed0
    while len(pairs) < n_pairs:
        rs, ro, ds, do = _unpack_pair(rules[s % len(rules)])
        pert = simulate_dyadic(cfg, rs, ro, seed=s, with_perturb=True,
                               dyn_self=ds, dyn_other=do)
        base = simulate_dyadic(cfg, rs, ro, seed=s, with_perturb=False,
                               dyn_self=ds, dyn_other=do)
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
    fails = 0

    def check(c: bool, msg: str) -> None:
        nonlocal fails
        if not c:
            fails += 1
            print(f"FAIL: {msg}")

    def mk(n: NCAConfig) -> NCARule:
        rc = sample_rule_config(n)
        arch = {k: rc[k] for k in ("kernel_size", "hidden_dim",
                                   "num_hidden_layers")}
        return NCARule(d_state=n.d_state, n_groups=n.n_groups, **arch)

    for reserve_eos, p_reuse, marked in [(False, 0.0, True), (True, 1.0, True),
                                         (True, 0.5, False)]:
        cfg = DyadicConfig(
            nca=NCAConfig(grid_size=16, num_steps=32, burn_in=2,
                          filter_enabled=False, reserve_eos=reserve_eos),
            commits_per_traj=(4, 8), queries_per_traj=(3, 6),
            p_tag_reuse=p_reuse, mark_turns=marked, min_query_lag_steps=3)
        n = cfg.nca
        tm = token_map(n)
        torch.manual_seed(7)
        rules = [(mk(n), mk(n)) for _ in range(3)]
        tag = f"[eos={reserve_eos} reuse={p_reuse} marked={marked}]"

        traj = simulate_dyadic(cfg, *rules[0], seed=11, with_perturb=True)
        toks = serialize_dyadic(traj, cfg)
        # 1. turn tokens: present+alternating when marked, absent when not
        turns = toks[(toks == tm["TURN_SELF"]) | (toks == tm["TURN_OTHER"])]
        if marked:
            check(len(turns) >= 2, f"{tag} no turn alternation")
            check(all(turns[i] != turns[i + 1] for i in range(len(turns) - 1)),
                  f"{tag} turn tokens repeat without handover")
        else:
            check(len(turns) == 0, f"{tag} TURN token leaked into unmarked ep")
        # 2. alphabet: EOS id never emitted; ids within bounds
        if reserve_eos:
            check(int((toks == 0).sum()) == 0, f"{tag} EOS inside episode")
        check(int(toks.max()) <= tm["VAL0"] + N_VALS - 1, f"{tag} id overflow")
        # 3. Last-write-wins gold: for every query, the answer equals the val
        # of the most recent commit to that tag earlier in the token stream
        q_pos = np.where(toks == tm["QUERY"])[0]
        c_pos = np.where(toks == tm["COMMIT"])[0]
        check(len(q_pos) >= 1, f"{tag} no queries emitted")
        for qp in q_pos:
            q_tag = int(toks[qp + 1])
            ans = int(toks[qp + 2])
            check(tm["VAL0"] <= ans < tm["VAL0"] + N_VALS,
                  f"{tag} query answer not VAL")
            prior = [cp for cp in c_pos if cp < qp and toks[cp + 1] == q_tag]
            check(len(prior) >= 1, f"{tag} query with no prior write")
            if prior:
                check(int(toks[max(prior) + 2]) == ans,
                      f"{tag} gold != last write (ill-posed ledger)")
        # 4. anti-shortcut: each VAL token occurs only at its own commit or at
        # queries it answers (vals are distinct per episode)
        commit_vals = [int(toks[cp + 2]) for cp in c_pos]
        check(len(set(commit_vals)) == len(commit_vals),
              f"{tag} duplicate VAL across commits")
        val_positions = set(np.where((toks >= tm["VAL0"])
                                     & (toks < tm["VAL0"] + N_VALS))[0].tolist())
        legal = set((c_pos + 2).tolist()) | set((q_pos + 2).tolist())
        check(val_positions <= legal, f"{tag} VAL leaked into frames")
        # 5. overwrite regime actually produces post-overwrite queries
        if p_reuse >= 1.0:
            evs = queries_in_stream_order(traj)
            check(any(e["post_overwrite"] for e in evs)
                  or len({c["tag"] for c in traj.events
                          if c["kind"] == "commit"}) == 1,
                  f"{tag} reuse=1.0 produced no overwrites")
        # 6. smoothing pairs well-formed + divergent
        pairs = emit_smoothing_pairs(cfg, rules, 2, seed0=100)
        check(len(pairs) == 2 and all(len(p["true_frame"]) ==
              n.tokens_per_frame * n.n_groups for p in pairs),
              f"{tag} smoothing pair malformed")
        for p in pairs:
            check(p["true_frame"] != p["counterfactual_frame"],
                  f"{tag} perturbation did not change the next frame")
        # 7. ledger items well-formed, both committer sides present
        led = emit_ledger_items(cfg, rules, 10, seed0=200)
        check(len(led) == 10 and all(0 <= it["gold_val"] < N_VALS for it in led),
              f"{tag} ledger items malformed")
        check(any(it["committer_side"] == 0 for it in led)
              and any(it["committer_side"] == 1 for it in led),
              f"{tag} committer sides not mixed")

    print(f"dyadic selftest: {fails} failures")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
