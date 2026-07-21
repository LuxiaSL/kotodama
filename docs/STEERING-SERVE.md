# STEERING-SERVE — residual steering in the production fast path

> STATUS: LIVE (implementation 2026-07-21; GPU gates PENDING — see §5, nothing is
> production-legal until they pass). Companion docs: the pilot record
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
after-attn sublayer of layer L is `s = 2*L`. The engine and `--steer-site` take
the **LAYER** index; npz banks keep the sublayer suffix. So `--steer-site 23`
⇔ pilot site `s46` ⇔ the write lands after layer 23's
`partial = partial + attn_out`.

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
engine = DecodeEngine(model, steer_site=23)   # LAYER index; None (default) = feature absent
engine.steer_site      # 23
engine.steer_points    # (23, 24) — introspection
engine.set_steer(vec)  # engage: FULL-SCALE composite (see §3), any dtype/shape (hidden,)
engine.set_steer(None) # clear (also: all-zero vector). No-op on a disabled engine.
```

* The site is **fixed at construction** (it's baked into the compiled graphs).
  Different site ⇒ rebuild the engine.
* `set_steer` casts to engine dtype (bf16 in prod — koto residual scales ~1e5 are
  safe in bf16; it was fp16 that clipped) and rejects wrong sizes / non-finite
  values. Callers own scaling: pass `sum(alpha * median_norm * unit_vec)`.
* `reset()` deliberately does NOT clear the vector — it is per-request state owned
  by the caller (serve.py clears in a `finally` after every request).

## 3. serve.py API

```bash
python serve.py --checkpoint /models/.../base.pt \
    --steer-npz /models/kotodama-data/steering_pilot/smoke/algebra_vectors.npz \
    --steer-site 23        # layer; npz must carry median_norm_s46 + *_s46 keys
```

* `--steer-npz` — bank of UNIT vectors + per-site median norms, the
  posttraining/taste convention: vector keys like `Alg_echo_base_s46`, median key
  `median_norm_s{2*layer}`. Keys suffixed with a DIFFERENT sublayer are **fatal**
  (mixed-site bank = silent dose error); metadata keys
  (`median_norm*`/`sign_*`/`law*`, scalars, strings) are skipped. Requires
  `--engine fast` — a steering flag on the reference engine is a startup error,
  never a silent unsteered server.
* Aliases: `STEER_ALIAS_TABLE` in serve.py is a verbatim copy of the sidecar's
  `ALIASES` (crown/echoness/echo-install/…/bind/bindbare + `control`); only
  aliases whose key is present in the bank are exposed. Keep the two tables in
  sync by hand.

Per-request (`POST /generate`, streaming included — same fields):

```jsonc
{ "messages": [...],
  "vectors": [ {"name": "crown", "alpha": 0.05},      // alias or raw npz key
               {"name": "echoness", "alpha": 0.03} ]  // composed: sum of alpha*med*unit
}
// or single-vector:
{ "prompt": "...", "steer_model": "echo-install", "steer_alpha": 0.07 }
```

* Omitted `alpha` → alias default (raw keys: 0.07, the sidecar fallback).
  `"control"` (or an all-control stack) → unsteered. `vectors` and
  `steer_model/steer_alpha` are mutually exclusive (400).
* Composition math is exactly the sidecar's: `Σ alphaᵢ · median_norm · unitᵢ`,
  fp32, then `engine.set_steer(composite)`; **`engine.set_steer(None)` runs in a
  `finally` after every request** (streaming: including disconnect paths), so
  state can never leak across requests.
* No `--steer-npz` → the request fields exist but any use of them is a clear 400;
  behavior is otherwise byte-identical to a pre-steering server. The OpenAI
  endpoints (`/v1/completions`, `/v1/chat/completions`) do not expose steering.
* Discovery: `/info` gains a `steering` block (site_layer, sublayer, median_norm,
  steer_points, aliases with defaults, raw keys); `/v1/models` gains a top-level
  `"steering"` key (aliases/keys). Aliases are deliberately NOT injected into
  `data[]` as fake model ids — the gateway routes on the model field.

## 4. Compile-safety rules (load-bearing — do not "simplify" these away)

1. `steer_vec` is a registered buffer created ONCE in `__init__` and only ever
   mutated **in-place** (`copy_`/`zero_`). Never reassign it —
   `torch.compile(mode="max-autotune")`/cudagraph trees specialize on tensor
   identity; a reassignment silently detaches the live graphs from the vector.
2. The add is **unconditional in the compiled graph** of a steering-enabled
   engine (zeros = off). Toggling via `set_steer` therefore never changes graph
   structure — no recompiles, no guard churn, no capture invalidation.
3. A steering-**disabled** engine (`steer_site=None`) traces an EMPTY point set:
   zero new ops, code paths bit-identical to HEAD. The membership test
   `ls.layer_idx in self._steer_point_set` resolves at trace time.
4. All three compiled decode variants (`_compiled_forward`,
   `_compiled_step_sampled`, `_compiled_step_truncated`) trace through the same
   `_forward_step`, so one injection site covers every decode graph. The compiled
   block forward (`_compiled_block`, prefill/extend) intentionally has NO
   injection. There are no other captured graphs.
5. `set_steer` costs two host syncs (finiteness + nonzero check) at request
   granularity — never call it inside the token loop.
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
python scripts/benchmark/test_steering_engine.py    # 33 checks, ~2 s, CPU
```

Proves: steer-point computation == `forward_inject` persist semantics for all 28
production sites; zeros/absent bitwise identity; prefill untouched; steered step
bitwise-equal to an independent injected reference (sites in first/middle/last
block); exact-vector shift of the running partial at the site; `set_steer(None)`
bitwise restore; error surface; the prefix-cache regenerate guard (tip token not
injected, counterfactual visibly would be).
