# STEERING-SERVE — residual steering in the production fast path (v2, multi-site)

> STATUS: LIVE (v2 multi-site implementation 2026-07-21; v1 single-site passed its
> first GPU gates — identity-off 5/5 token-exact vs HEAD, throughput −0.5%. The v2
> generalization re-runs the SAME gates before production — see §5).
> Companion docs: the pilot record
> `research/planning/{PLAN,RESULTS}-steering-pilot-2026-07-21.md` (repo root), the
> reference semantics `posttraining/taste/steer_inject.py::forward_inject`, and the
> sidecar server this API mirrors `posttraining/taste/koto_steered_chat.py`.

Adds the pilot's block-persistent residual write to `DecodeEngine` + `serve.py` so
steered generation runs at fast-path speed (~400 tok/s decode) instead of the
HF-eager mirror (~10x slower). Same math, same dose currency, same request dialect
as the sidecar steering rack.

## 1. The write primitive (why not a single-site add)

RESULTS §3, the architectural finding: a single-site residual add is destroyed at
the next DD-3B block boundary (commit-and-reset + routing recombination attenuate
it ~100x; final logits KL 0.002 nats — the model never sees it). The koto write
primitive is therefore **block-persistent**: inject at the site, then re-assert at
the first after-attn position of every block AFTER the site's block (KL 0.78 at
α=0.1 — the real lever).

**Layer/sublayer convention.** The pilot indexes SUBLAYERS (two per layer); the
after-attn sublayer of layer L is `s = 2*L`. The engine takes **LAYER** indices;
npz banks keep the sublayer suffix, and in v2 **each key carries its own site**:
`Alg_echo_base_s46` ⇔ layer 23, `Vrepperp_s20` ⇔ layer 10, `Bind_s38` ⇔ layer 19.
Odd suffixes (mlp sublayers) are rejected — the engine anchor is after-attn only.

**Write points** (`src/model/decode_engine.py::compute_steer_points`):
`{site} + {boundary b : b > site}`. Example — site layer 23, DD-3B boundaries
`[0, 1, 3, 7, 15, 19, 24]` → points `[23, 24]` (site is in block [19,24); one
block after it, entered at layer 24). Site 10 → `[10, 15, 19, 24]`.

**Generated positions only.** Injection happens exclusively in the decode step
(`_forward_step`); prefill and block/extend forwards (`_forward_block`, the eager
extend) never inject. This reproduces the pilot's `w0` window for free — with the
documented consequence that re-prefilled prior generated turns in multi-turn are
**unsteered** (accepted pilot choice; "re-fed context is not re-steered").
Prefix-cache edge: the suffix==1 regenerate shortcut normally routes through the
compiled decode step, which WOULD inject into a prompt position — with a nonzero
vector engaged the engine bypasses that shortcut (block/eager extend instead,
slightly slower, semantics exact). Vector off → path unchanged.

## 2. Engine API (`src/model/decode_engine.py`)

```python
engine = DecodeEngine(model, steer_enabled=True)  # False (default) = feature absent
engine.steer_buf          # (n_layers, hidden) buffer; zero rows = off
engine.set_steer([        # engage: list of (site_layer, vec, scale) writes
    (23, unit_a, 0.05 * med23),   # scale = alpha * that site's median norm
    (10, unit_b, -0.1 * med10),
])
engine.set_steer(None)    # clear (also: []). No-op on a disabled engine.
compute_steer_points(23, boundaries, n_layers)  # (23, 24) — row placement rule
```

* v2: sites are **per-write, at request time** — the compiled graphs contain one
  per-layer row-add (`partial + steer_buf[layer]`) regardless of which rows are
  populated, so no rebuild is needed to change sites. `set_steer` stages rows in
  fp32 then copies into the buffer in-place: each write lands `scale*vec` in the
  site row AND every later block-entry row (`compute_steer_points`, now applied
  at content-time); overlapping rows SUM.
* Buffer dtype = engine dtype (bf16 in prod — koto residual scales ~1e5 are safe
  in bf16; it was fp16 that clipped). `set_steer` rejects wrong sizes,
  non-finite vectors/scales, malformed items, out-of-range sites.
* `reset()` deliberately does NOT clear the buffer — it is per-request state
  owned by the caller (serve.py clears in a `finally` after every request).

## 3. serve.py API

```bash
python serve.py --checkpoint /models/.../base.pt \
    --steer-npz algebra_vectors.npz,reppen_formula.npz,binding.npz
    # comma-separated banks; per-key site from the _s{sublayer} suffix
```

* `--steer-npz` — COMMA-SEPARATED banks of UNIT vectors + per-site median norms,
  the posttraining/taste convention: vector keys like `Alg_echo_base_s46`
  (site = suffix//2), median keys `median_norm_s{2*layer}` per site. Duplicate
  keys / conflicting medians across files resolve FIRST-NPZ-WINS (sidecar
  convention). Every vector's site MUST have a median norm (else fatal — doses
  can't be scaled); odd sublayer suffixes are fatal (after-attn only);
  suffixless vector keys are skipped (can't be placed); metadata
  (`median_norm*`/`sign_*`/`law*`, scalars, strings) is skipped. Requires
  `--engine fast` — a steering flag on the reference engine is a startup error,
  never a silent unsteered server.
* Aliases: `STEER_ALIAS_TABLE` in serve.py copies the sidecar's `ALIASES`
  (crown/vrep/v7-entropy/echoness/…/bind/bindbare + `control`) and adds a
  human-readable `desc` per alias (surfaced in `/info`); only aliases whose key
  is present in a loaded bank are exposed. Keep the two tables in sync by hand.

Per-request (`POST /generate`, streaming included — same fields):

```jsonc
{ "messages": [...],
  "vectors": [ {"name": "crown", "alpha": 0.05},   // alias or raw npz key
               {"name": "vrep", "alpha": -0.1},    // DIFFERENT site — fine in v2
               {"name": "bind"} ] }                // per-member: alpha * ITS site's med * unit
// or single-vector:
{ "prompt": "...", "steer_model": "echo-install", "steer_alpha": 0.07 }
```

* Omitted `alpha` → alias default (raw keys: 0.07, the sidecar fallback).
  `"control"` (or an all-control stack) → unsteered. `vectors` and
  `steer_model/steer_alpha` are mutually exclusive (400).
* Composition: each member becomes one engine write
  `(site_layer, unit, alpha · that site's median_norm)` — same-site members sum
  row-wise in the engine, cross-site members land in their own rows (dose math
  per member identical to the sidecar rack). **`engine.set_steer(None)` runs in
  a `finally` after every request** (streaming: including disconnect paths), so
  state can never leak across requests.
* No `--steer-npz` → the request fields exist but any use of them is a clear 400;
  behavior is otherwise byte-identical to a pre-steering server. The OpenAI
  endpoints (`/v1/completions`, `/v1/chat/completions`) do not expose steering.
* Discovery: `/info` gains a `steering` block (sites/sublayers, per-site
  median_norms, aliases with defaults + per-alias `desc` + site_layer, raw keys
  with sites); `/v1/models` gains a top-level `"steering"` key
  (sites/aliases/keys). Aliases are deliberately NOT injected into `data[]` as
  fake model ids — the gateway routes on the model field.

## 4. Compile-safety rules (load-bearing — do not "simplify" these away)

1. `steer_buf` (n_layers, hidden) is a registered buffer created ONCE in
   `__init__` and only ever mutated **in-place** (`copy_`/`zero_`). Never
   reassign it — `torch.compile(mode="max-autotune")`/cudagraph trees specialize
   on tensor identity; a reassignment silently detaches the live graphs from the
   buffer.
2. The per-layer row-add (`partial + steer_buf[layer]`, static index per
   unrolled layer) is **unconditional in the compiled graph** of a
   steering-enabled engine (zero rows = off). set_steer changes CONTENT only —
   no recompiles, no guard churn, no capture invalidation, regardless of which
   sites a request uses.
3. A steering-**disabled** engine (`steer_enabled=False`, a trace-time constant)
   traces zero new ops, code paths bit-identical to HEAD — the off-gate is
   bitwise by construction, never by an "adding zeros is identity" argument.
4. All three compiled decode variants (`_compiled_forward`,
   `_compiled_step_sampled`, `_compiled_step_truncated`) trace through the same
   `_forward_step`, so one injection site covers every decode graph. The compiled
   block forward (`_compiled_block`, prefill/extend) intentionally has NO
   injection. There are no other captured graphs.
5. `set_steer` costs one host sync per write (vector finiteness validation) at
   request granularity — never call it inside the token loop. The activity flag
   is derived host-side from the writes (no device sync).
6. serve.py threading: in the non-stream path `set_steer` runs on the dedicated
   generation thread (inside `engine_generate`); in the stream path it is routed
   through `_generate_executor` like every other engine call (it is eager buffer
   mutation, no cudagraph replay, but the single-thread discipline stays intact).

## 5. GPU gates — MUST pass before production use (run in the MAIN session, not here)

The CPU battery (§6) proves semantics on the eager paths; the compiled/cudagraph
behavior and throughput are only provable on a B200. Gate battery, in order:

1. **Bitwise-off gate vs HEAD, fixed seeds.** (a) A server built WITHOUT
   `--steer-npz` at this commit vs HEAD: token-identical outputs on a fixed
   prompt pack (the disabled engine adds zero ops — anything non-identical is a
   bug). (b) A server WITH `--steer-npz` but no steering fields in the requests
   vs HEAD: expected token-identical too (x+0 is exact elementwise); if compile
   fusion reorders surrounding kernels and dust appears, STOP and characterize
   before proceeding — the off-state must be either bitwise or formally accepted
   as trunk-dust class by Luxia.
2. **Steered-serve vs steered-mirror trunk-dust tolerance.** Teacher-forced
   logit comparison: engine decode with a vector engaged vs
   `forward_inject(..., persist=True)` on the same checkpoint/tokens/site/scale.
   Acceptance = the established benign trunk-delta class (G0.3: top-1 agreement
   100%, mean |Δlogit| ≈ 0.07–0.11), NOT exactness — serve and mirror are
   different trunk paths (LAW: pin one trunk path per activation analysis; the
   mirror stays THE trunk for state reads).
3. **Throughput within ~2%** of HEAD on the standard decode benchmark
   (`scripts/benchmark/bench_engine.py`), steering built-in + vector off AND
   vector on (one fused elementwise add per steer point — expect noise-level).
4. Standard batteries still green: `validate_engine.py`,
   `test_prefix_cache.py` (unit + real), `test_sampler_dist.py`.

## 6. Tests (CPU, run anywhere)

```bash
python scripts/benchmark/test_steering_engine.py    # 46 checks, ~3 s, CPU
```

Proves: steer-point computation == `forward_inject` persist semantics for all 28
production sites; zero-buffer/absent bitwise identity; prefill untouched under
multi-site writes; steered step bitwise-equal to an independent injected
reference (sites in first/middle/last block); exact-write shift of the running
partial at the site; multi-site writes populate exactly the expected rows
(site + later block entries; overlaps sum; duplicate same-site writes sum) and
compose bitwise; `set_steer(None)`/`[]` bitwise restore; error surface (incl.
malformed writes and bad scales); the prefix-cache regenerate guard under
multi-site writes (tip token not injected, counterfactual visibly would be).
