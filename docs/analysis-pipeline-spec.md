# Analysis Pipeline Spec

A unified analysis suite for luxia-base pretrained models. Transforms scattered ad-hoc scripts into a coherent, repeatable pipeline with clear stages, shared utilities, and standard output formats.

Last updated: 2026-04-10

---

## 0. Design Principles

- **Six tracks, one suite.** Each track answers a different question. They share utilities but produce independent outputs. No monolithic script.
- **Shared core, thin tracks.** Model loading, forward passes, generation, JSONL parsing — all live in shared modules. Track scripts are thin orchestrators.
- **Standard output.** Every track writes JSON (metrics/analysis) or NPZ (tensors). Plots are secondary outputs, not primary.
- **GPU vs CPU separation.** Extraction requires GPU. Analysis and visualization are CPU-only. Clear boundary so you know what runs where.
- **Repeatable.** Given a checkpoint path and a config, the full suite produces deterministic output. No hardcoded paths in track scripts.

---

## 1. The Six Tracks

### Track 1: Training Metrics
**Question:** How did the training run go? Loss, throughput, learning rate, gradient norms.
**When:** Logged during training (automatic). Analyzed post-hoc.
**Input:** `metrics.jsonl` (written by WandbLogger during training)
**Output:** JSON summary + trajectory analysis

### Track 2: Geometric Monitoring
**Question:** Is the model geometrically healthy? Spectral properties, attention behavior, representational structure.
**When:** Logged during training at Tier 1 (every 500 steps) and Tier 2 (every 5000 steps). Analyzed post-hoc.
**Input:** Same `metrics.jsonl` (geometric metrics interleaved with training metrics)
**Output:** JSON summary + geometric health analysis

### Track 3: Eval Generation
**Question:** What does the model produce? Text samples for qualitative and quantitative assessment.
**When:** Post-training (or at intermediate checkpoints).
**Input:** Model checkpoint + prompt set + generation config
**Output:** JSON with prompt/continuation pairs
**Requires:** GPU

### Track 4: Text Quality
**Question:** What are the semantic properties of the generated text? 59-metric profiling across lexical, coherence, structural, repetition, creativity dimensions.
**When:** After Track 3 produces generations.
**Input:** Track 3 output JSON
**Output:** JSON with per-sample metrics + model profiles + distinctive features
**Requires:** CPU only (NLTK + sklearn)

### Track 5: Activation Geometry
**Question:** What do the model's internal representations look like geometrically? Point clouds, eigenspectra, trajectories, attention patterns.
**When:** Post-training extraction from checkpoints.
**Input:** Model checkpoint + eval tokens
**Output:** NPZ arrays (point clouds, eigenspectra, trajectories, attention weights, head entropy, effective rank)
**Requires:** GPU for extraction, CPU for derived metrics and visualization

### Track 6: Concept Geometry
**Question:** Does the model organize semantic concepts into structured manifolds? Ordinal, cyclic, geographic structure in the residual stream.
**When:** Post-training extraction + analysis.
**Input:** Model checkpoint + concept set definitions + templates
**Output:** NPZ activations (extraction, GPU) → JSON analysis results with statistical tests (analysis, CPU)
**Requires:** GPU for extraction, CPU for analysis

---

## 2. Dependency Graph

```
Training run
  │
  ├─→ metrics.jsonl ──→ [Track 1: Training Metrics Analysis] ──→ training_summary.json
  │                  ──→ [Track 2: Geometric Health Analysis] ──→ geometric_summary.json
  │
  └─→ checkpoint(s)
        │
        ├─→ [Track 3: Eval Generation] ──→ generations.json
        │         │
        │         └─→ [Track 4: Text Quality] ──→ text_quality.json
        │
        ├─→ [Track 5: Activation Geometry Extraction] ──→ activation_geometry/*.npz
        │         │
        │         └─→ [Track 5: Activation Geometry Analysis] ──→ activation_analysis.json
        │                                                     ──→ plots/
        │
        └─→ [Track 6: Concept Geometry Extraction] ──→ concept_geometry/*.npz
                  │
                  └─→ [Track 6: Concept Geometry Analysis] ──→ concept_analysis.json
                                                           ──→ plots/
```

Tracks 1-2 share input (metrics.jsonl) but are independent analyses.
Tracks 3→4 are sequential (generation feeds quality analysis).
Tracks 5-6 are independent of each other and of 3-4.
All tracks are independent of each other except 3→4.

---

## 3. Shared Modules

These are utilities extracted from patterns duplicated across 5-15 existing scripts. They live in `src/` and are imported by track scripts.

### 3.1 Model Loading (`src/eval/model_loader.py`)

Currently duplicated in ~10 scripts with minor variations.

**Consolidates:**
- Checkpoint loading with `_orig_mod.` prefix stripping (torch.compile artifact)
- AttnRes auto-detection via state dict key inspection
- Explicit AttnRes config override (preferred over auto-detection for reliability)
- Config loading from YAML
- Device placement

```python
def load_model(
    checkpoint_path: Path,
    config_path: Path = "configs/model.yaml",
    config_section: str = "proxy",
    attn_res_config: dict | None = None,
    device: str = "cuda:0",
) -> LuxiaBaseModel:
    ...
```

### 3.2 Forward Pass with State Capture (`src/eval/forward.py`)

Currently duplicated in extract_shapes.py, extract_shapes_v2.py, extract_manifolds.py, geometric.py — each with slightly different AttnRes handling.

**Consolidates:**
- Standard forward with hidden state capture at every layer
- AttnRes-aware forward with boundary-clone routing
- Manual attention weight computation (for when SDPA/Flash won't return weights)
- Consistent `states[0] = embedding, states[i+1] = after layer i` convention

```python
def forward_with_states(
    model: LuxiaBaseModel,
    input_ids: Tensor,
    capture_attention: bool = False,
    attention_layers: list[int] | None = None,
) -> ForwardResult:
    """Returns hidden states at all layers, optionally attention weights."""
    ...
```

### 3.3 Text Generation (`src/eval/generate.py`)

Currently duplicated in ~6 eval scripts with variations in temperature, top-p, max tokens, EOS handling.

**Consolidates:**
- Autoregressive generation loop
- Temperature + top-p sampling
- Configurable EOS token set
- Context window management (truncation to max_position_embeddings)

```python
def generate(
    model: LuxiaBaseModel,
    input_ids: Tensor,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float | None = None,
    eos_tokens: list[int] = [0, 1, 2],
) -> Tensor:
    ...
```

### 3.4 Perplexity (`src/eval/perplexity.py`)

Currently duplicated in ~4 eval scripts.

**Consolidates:**
- Memmap-based eval data loading
- Batched cross-entropy computation with autocast
- Configurable sequence length, batch size, max batches

```python
def compute_perplexity(
    model: LuxiaBaseModel,
    eval_data_path: Path,
    seq_len: int = 2048,
    batch_size: int = 4,
    max_seqs: int = 400,
    device: str = "cuda:0",
) -> dict:  # {"loss": float, "perplexity": float, "n_tokens": int}
    ...
```

### 3.5 JSONL Metrics Loading (`src/eval/metrics_io.py`)

Currently duplicated in ~5 analysis scripts with different merge strategies.

**Consolidates:**
- Line-by-line JSONL loading with error handling
- Step-based merging (multiple records per step → single dict)
- Nearest-step lookup with tolerance
- Metric extraction by key pattern
- Time-series extraction for any metric

```python
def load_metrics(path: Path, merge: bool = True) -> dict[int, dict[str, float]]:
    ...

def get_nearest_step(data: dict, target: int, tolerance: int = 500) -> int | None:
    ...

def extract_series(data: dict, key: str) -> list[tuple[int, float]]:
    ...
```

### 3.6 Checkpoint Registry (`configs/checkpoints.yaml` or similar)

Currently hardcoded in every script. Centralizes:
- Checkpoint name → path mapping
- AttnRes config per checkpoint
- Grouping (proxy sweep, NCA matrix, lang-full, etc.)

### 3.7 Prompt Sets (`configs/prompts.yaml`)

Currently hardcoded in every eval script. Two sets:
- Standard 5-prompt set (used across all eval scripts)
- Extended 19-prompt set (used for temperature matrix)
- Trajectory 12-prompt set (used for shapes extraction)

### 3.8 Concept Sets (`configs/concepts.yaml` or module)

Currently defined in extract_manifolds.py. Centralizes:
- Concept set definitions (items, topology, tier, category)
- Templates per concept set
- Geographic coordinates (for US states)
- Year ranges

### 3.9 Visualization Defaults (`src/eval/plot_utils.py`)

Currently duplicated across plotting scripts:
- Checkpoint color schemes
- Marker styles
- Abbreviation maps
- Style setup (dark_background vs publication white)

---

## 4. Track Specifications

### Track 1+2: Unified Metrics Analyzer

**Replaces:** `analyze_metrics.py`, `compare_runs.py`, `interaction_analysis_v2.py`, `analyze_lang_diagnostics.py`, `viz_alpha_rank.py`

**Script:** `scripts/analysis/analyze_run.py` (or similar)

**Input:** One or more `metrics.jsonl` files via CLI

**Modes:**
- `--summary`: Endpoint summary (what analyze_metrics.py does now)
- `--dynamics`: Multi-resolution windowed analysis (NEW)
- `--compare`: Cross-run comparison tables (what compare_runs.py does)
- `--factorial`: 2×2 interaction analysis (what interaction_analysis_v2.py does)
- `--reference <path>`: Compare against reference trajectory (generalized from analyze_lang_diagnostics.py)
- `--full`: All of the above

**Analyses:**

#### A. Endpoint Summary (existing, cleaned up)
- Final loss, perplexity, throughput
- RankMe trajectory (initial → min → final, rebound ratio)
- WW alpha (final, healthy fraction)
- TwoNN intrinsic dimension
- Tokens consumed, steps completed

#### B. Training Dynamics (NEW — the big missing piece)

**Multi-resolution windowed slopes:**
For each of `train/loss`, `geo/rankme_last`, and optionally any specified metric:
- Compute slope over windows of [100, 500, 1000, 5000] steps
- Report slope at each window size at landmark steps (10%, 25%, 50%, 75%, 90% of training)
- Flag sign changes (convergence → divergence transitions)

**Second derivatives (slopes of slopes):**
- Compute slope-of-slope at 1000-step resolution
- Detect: convergence tightening (magnitude decreasing), oscillation (sign alternation), divergence acceleration
- Report inflection points (where second derivative crosses zero)

**Rolling statistics:**
- Windowed mean/std/p25/p75 at 500-step windows for key metrics
- Stability score: ratio of late-training std to early-training std (lower = more stable)

**Phase transition detection:**
- Plateau detection: regions where |slope| < threshold for N consecutive windows
- Jump detection: step-over-step delta > 3σ from running mean
- Report suspected phase transitions with step ranges

#### C. Geometric Health Profile (NEW)

For each sampled layer, at landmark steps:
- Stable rank (per projection type: q/k/o/gate/down)
- Attention entropy (mean, std across heads)
- Anisotropy
- Dead unit fraction
- BOS attention mass (if attention weights available)

Cross-layer profile:
- Depth gradient: metric at L0 vs L14 vs L27 (or equivalent sampled layers)
- Gradient flow: attn/mlp gradient ratio per layer
- Spectral budget: total stable rank across layers (conservation check)

Composite health score (experimental):
- Weighted combination of RankMe rebound, WW healthy fraction, dead unit fraction, entropy consistency
- Reported as single number for quick comparison, with decomposition available

#### D. Cross-Run Comparison (existing, generalized)
- Side-by-side table of endpoint metrics
- Delta computation with direction awareness (lower-is-better for loss)
- Highlight best/worst per metric

#### E. Factorial Analysis (existing, generalized)
- Define factors via CLI: `--factorial "NCA:run1,run2 AttnRes:run3,run4"`
- Compute main effects + interaction terms for each metric
- Report super-additive / sub-additive / additive classification

#### F. Reference Trajectory Comparison (existing pattern, generalized)
- Load reference values from a prior run's summary JSON
- At matched landmark steps, compute deltas and percent deviations
- Flag metrics that deviate > threshold from reference
- Useful for: "is the 3B run tracking the proxy trajectory?"

**Output:** `analysis/run_analysis.json` containing all requested analyses, machine-readable.

---

### Track 3: Eval Generation

**Replaces:** `eval_lang_full.py`, `eval_lang_full_multisample.py`, `eval_full_matrix.py`, `eval_temperature_matrix.py`, `eval_nca_vs_p3.py`, `eval_proxy_sweep.py`

**Preferred method: serve.py + eval_via_server.py**

The fastest generation path uses `serve.py` (inference server with KV cache, bf16, optional torch.compile, Triton-fused AttnRes) combined with `eval_via_server.py` (client that hits the server and produces Track 3 JSON).

```bash
# 1. Launch serve.py on a GPU node (via Heimdall or directly)
heimdall submit 'tools/run_eval.sh serve.py --checkpoint <path> --device cuda --port 2222 --compile' \
    --name eval-serve --type custom --gpus 1 --node node2 \
    --workdir /home/athuser/luxi-files/kotodama

# 2. Hit it from local with the eval client
python scripts/analysis/eval_via_server.py \
    --url http://node2.datasci.ath:2222 \
    --name <run-name> \
    --prompt-set extended --n-samples 3 --temperature 0.7 --max-tokens 512

# 3. Kill the server when done
heimdall cancel <job-id>
```

**Alternative: eval_generate.py** (standalone, slower)

`scripts/analysis/eval_generate.py` loads the model directly and generates without a server. Uses KV cache but does not use the Triton-fused AttnRes path. Suitable for quick tests or when serve.py isn't available.

```bash
# Via Heimdall
heimdall submit 'tools/run_eval.sh scripts/analysis/eval_generate.py \
    --checkpoint <path> --name <run-name> \
    --attn-res "boundaries=0,3,7,12,21,25" --no-ppl \
    --prompt-set extended --n-samples 3' \
    --name eval-gen --type custom --gpus 1 --node node2 \
    --workdir /home/athuser/luxi-files/kotodama
```

**Output:** `analysis/{run_name}/generations.json`
```json
{
  "config": {"temperature": 0.7, "top_p": null, "max_tokens": 512, "n_samples": 3},
  "runs": [
    {
      "name": "owt-ddv1",
      "checkpoint": "path/to/checkpoint.pt",
      "samples": [
        {
          "prompt_idx": 0,
          "prompt": "The most interesting thing about language is",
          "sample_idx": 0,
          "continuation": "...",
          "n_tokens": 512,
          "tokens_per_second": 45.6,
          "stopped_by": "max_tokens"
        }
      ]
    }
  ]
}
```

**Performance:** serve.py with KV cache: ~45 tok/s on B200 (108M model, no compile). Without KV cache: ~2.4 tok/s (O(n²) recomputation — do not use).

**Requires:** GPU

---

### Track 4: Text Quality

**Replaces:** `analyze_text_quality.py`

**Script:** `scripts/analysis/analyze_text_quality.py` (largely unchanged — it's already well-structured)

**Input:** Track 3 output JSON
**Output:** `analysis/text_quality.json`

The 59-metric 5-layer framework (lexical, coherence, structural, repetition, creativity) is bespoke and well-designed. Main changes:
- Accept Track 3 output format directly (currently expects a slightly different schema)
- Write to standard output location
- Add CLI args for input/output paths

**Requires:** CPU only

---

### Track 5: Activation Geometry

**Replaces:** `extract_shapes.py`, `extract_shapes_v2.py`, `extract_shapes_lang_full.py`, `visualize_shapes.py`, `visualize_shapes_v2.py`

**Two scripts, clear separation:**

#### 5a. Extraction: `scripts/analysis/extract_activation_geometry.py`

**Input:** Checkpoint path(s) + eval tokens
**Output:** `analysis/activation_geometry/*.npz`

Unifies v1 and v2 extraction into one script with `--components` flag:

| Component | What | GPU? | Source |
|-----------|------|------|--------|
| `point_clouds` | Mean-pooled hidden states (500 seqs) | Yes | v1 |
| `topo_clouds` | Token-level activations for persistent homology (50 seqs) | Yes | v1 |
| `trajectories` | Last-token hidden state through all layers (12 prompts) | Yes | v1 |
| `eigenspectra` | SVD of all weight matrices | CPU | v1 |
| `attention_weights` | Per-head attention patterns (all layers) | Yes | v2 |
| `attention_outputs` | Pre-residual attention contributions (500 seqs) | Yes | v2 |
| `head_entropy` | Per-head attention entropy distribution (200 seqs) | Yes | v2 |
| `effective_rank` | Participation ratio per weight matrix | CPU | v2 (derived from eigenspectra) |
| `procrustes` | PCA-aligned point clouds for cross-run comparison | CPU | v2 (derived from point_clouds) |

Default: `--components all`. Can run `--components eigenspectra,effective_rank` for CPU-only subset.

Uses shared modules: model_loader, forward_with_states, checkpoint_registry.

#### 5b. Visualization: `scripts/analysis/visualize_activation_geometry.py`

**Input:** Track 5a NPZ outputs
**Output:** `analysis/activation_geometry/plots/*.png`

Unifies visualize_shapes.py + visualize_shapes_v2.py. CPU-only.

---

### Track 6: Concept Geometry

**Replaces:** `extract_manifolds.py`, `extract_manifolds_v3.py`, `analyze_manifolds_v2.py`, `plot_manifolds_paper_v3.py`, `plot_manifolds_paper_v3_extra.py`

**Three scripts:**

#### 6a. Extraction: `scripts/analysis/extract_concept_geometry.py`

**Input:** Checkpoint path(s) + concept set config + templates
**Output:** `analysis/concept_geometry/activations.npz` + `metadata.json`

Unifies v1 (17 concept sets) and v3 (states + years) into one extractor.
Uses shared modules: model_loader, forward_with_states, concept_sets config.

**Requires:** GPU

#### 6b. Analysis: `scripts/analysis/analyze_concept_geometry.py`

**Input:** Track 6a NPZ outputs + metadata
**Output:** `analysis/concept_geometry/results.json`

This is `analyze_manifolds_v2.py` — already well-structured with corrected statistical tests. Main changes:
- Read from standard Track 6a output location
- Include v3 concept sets (states, years) in the same analysis pass
- Write to standard output location

**Requires:** CPU only

#### 6c. Visualization: `scripts/analysis/visualize_concept_geometry.py`

**Input:** Track 6b results JSON + Track 6a activations
**Output:** `analysis/concept_geometry/plots/*.png`

Consolidates plot_manifolds_paper_v3.py + _extra.py.

**Requires:** CPU only

---

## 5. Scripts to Retire

Once the unified suite is built, these become dead code:

| Script | Replaced by | Notes |
|--------|-------------|-------|
| `analyze_metrics.py` | Track 1+2 unified analyzer | Endpoint-only summary |
| `compare_runs.py` | Track 1+2 `--compare` mode | Misplaced in data/, bespoke |
| `interaction_analysis.py` | Track 1+2 `--factorial` mode | v1, superseded by v2 |
| `interaction_analysis_v2.py` | Track 1+2 `--factorial` mode | Hardcoded to NCA×AttnRes |
| `analyze_lang_diagnostics.py` | Track 1+2 `--reference` mode | Hardcoded to 3 diagnostic runs |
| `viz_alpha_rank.py` | Track 1+2 visualization | Hardcoded to proxy sweep |
| `eval_lang_full.py` | Track 3 | Hardcoded to 2 checkpoints |
| `eval_lang_full_multisample.py` | Track 3 | Hardcoded to 2 checkpoints |
| `eval_full_matrix.py` | Track 3 | Hardcoded to 4 checkpoints |
| `eval_temperature_matrix.py` | Track 3 | Keeps top-p; merge into Track 3 |
| `eval_nca_vs_p3.py` | Track 3 | Hardcoded to 2 checkpoints |
| `eval_proxy_sweep.py` | Track 3 | Keeps Heimdall logging pattern |
| `extract_shapes.py` | Track 5a | v1 core |
| `extract_shapes_v2.py` | Track 5a | v2 supplementary |
| `extract_shapes_lang_full.py` | Track 5a | Monkeypatch wrapper |
| `visualize_shapes.py` | Track 5b | v1 viz |
| `visualize_shapes_v2.py` | Track 5b | v2 viz |
| `extract_manifolds.py` | Track 6a | v1 (17 concepts) |
| `extract_manifolds_v3.py` | Track 6a | v3 (states + years) |
| `analyze_manifolds.py` | **DELETE** | BROKEN, replaced by v2 |
| `analyze_manifolds_v2.py` | Track 6b | Corrected analysis |
| `plot_manifolds_paper.py` | Track 6c | v1 plots |
| `plot_manifolds_paper_v2.py` | Track 6c | v2 plots |
| `plot_manifolds_paper_v3.py` | Track 6c | v3 plots |
| `plot_manifolds_paper_v3_extra.py` | Track 6c | Supplementary |
| `analyze_dynamics.py` | Track 6b | Subsumed into concept analysis |
| `fig7_block_boundary.py` | Track 6c | Paper-specific figure |
| `plot_attention_no_bos.py` | Track 5b | Attention visualization |

**Keep as-is (utility, not analysis):**
- `tokenize_data.py` — Data prep utility, not part of analysis suite
- `smoke_test.py` — Validation utility
- `profile_step.py` — Performance profiling
- `bench_attnres.py` / `bench_attnres_v2.py` — Architecture benchmarking
- `visualize_nca.py` — NCA data visualization (not model analysis)

---

## 6. Standard Output Layout

All analysis outputs for a given run go under one directory:

```
analysis/
  {run_name}/
    training_summary.json        # Track 1+2: endpoint metrics
    training_dynamics.json       # Track 1+2: windowed slopes, inflections, health
    generations.json             # Track 3: text samples
    text_quality.json            # Track 4: 59-metric profiles
    activation_geometry/         # Track 5
      point_clouds.npz
      eigenspectra.npz
      trajectories.npz
      attention_weights.npz
      attention_outputs.npz
      head_entropy.npz
      effective_rank.npz
      procrustes.npz
      plots/
    concept_geometry/            # Track 6
      activations.npz
      metadata.json
      results.json
      plots/
```

For cross-run analyses (comparisons, factorial):
```
analysis/
  comparisons/
    {comparison_name}.json       # e.g., "lang_full_baseline_vs_ddv1.json"
```

---

## 7. CLI Interface

All track scripts follow the same pattern:

```bash
# Track 1+2: Training metrics
python -m scripts.analyze_run data/lang_full_ddv1_metrics.jsonl --full
python -m scripts.analyze_run data/*_metrics.jsonl --compare
python -m scripts.analyze_run data/*_metrics.jsonl --factorial "NCA:nca,baseline AttnRes:ar,noar"

# Track 3: Generation
python -m scripts.eval_generate --checkpoint path/to/ckpt.pt --name lang-ddv1 --n-samples 5

# Track 4: Text quality
python -m scripts.analyze_text_quality --input analysis/lang-ddv1/generations.json

# Track 5: Activation geometry
python -m scripts.extract_activation_geometry --checkpoint path/to/ckpt.pt --name lang-ddv1
python -m scripts.visualize_activation_geometry --input analysis/lang-ddv1/activation_geometry/

# Track 6: Concept geometry
python -m scripts.extract_concept_geometry --checkpoint path/to/ckpt.pt --name lang-ddv1
python -m scripts.analyze_concept_geometry --input analysis/lang-ddv1/concept_geometry/
python -m scripts.visualize_concept_geometry --input analysis/lang-ddv1/concept_geometry/
```

---

## 8. Implementation Order

### Phase A: Shared Modules (prerequisite for everything) — DONE (2026-04-11)
- [x] `src/eval/model_loader.py` — checkpoint loading, AttnRes config, YAML config
- [x] `src/eval/forward.py` — unified forward_with_states + attention weight computation
- [x] `src/eval/generate.py` — autoregressive generation with temperature + top-p
- [x] `src/eval/perplexity.py` — memmap-based held-out PPL
- [x] `src/eval/metrics_io.py` — JSONL loading, step merging, series extraction
- [x] `src/eval/plot_utils.py` — checkpoint colors, markers, style setup
- [x] `src/eval/__init__.py` — package exports
- [x] `configs/checkpoints.yaml` — 9 checkpoints with AttnRes config + metrics file mapping
- [x] `configs/prompts.yaml` — standard (5), trajectory (12), extended (19) prompt sets
- [x] `configs/concepts.yaml` — 19 concept sets (v1+v3) with templates + state coordinates

### Phase B: Track 1+2 Unified Analyzer (highest immediate value) — DONE (2026-04-11)
- [x] `src/eval/run_analysis.py` — analysis engine: endpoint_summary, training_dynamics, geometric_health, compare_runs, factorial_analysis, reference_comparison
- [x] `scripts/analysis/analyze_run.py` — CLI orchestrator with --summary, --dynamics, --health, --compare, --factorial, --reference, --full modes
- [x] Validated: endpoint summary matches analyze_metrics.py exactly (11/11 metrics)
- [x] Validated: factorial analysis matches interaction_analysis_v2.py exactly (loss interaction = +0.0118)

### Phase C: Track 3+4 Eval Pipeline (depends on Phase A) — DONE (2026-04-11)
- [x] `scripts/analysis/eval_generate.py` — unified generation with --checkpoint/--checkpoints, --prompt-set, --n-samples, --temperature/--top-p
- [x] Updated `scripts/analysis/analyze_text_quality.py` — accepts both legacy list format and new Track 3 {"config", "runs"} format

### Phase D: Track 5 Activation Geometry (depends on Phase A) — DONE (2026-04-11)
- [x] `scripts/analysis/extract_activation_geometry.py` — unifies v1+v2: 9 components (6 GPU, 3 CPU), --components flag, registry or single checkpoint
- [x] `scripts/analysis/visualize_activation_geometry.py` — PCA, eigenspectra, trajectories, head entropy plots; extensible dispatcher

### Phase E: Track 6 Concept Geometry (depends on Phase A) — DONE (2026-04-11)
- [x] `scripts/analysis/extract_concept_geometry.py` — unified extraction with configs/concepts.yaml, --tier/--concept-sets filtering
- [x] `scripts/analysis/analyze_concept_geometry.py` — 6 analyses (cyclic, ordinal, smoothness, template sensitivity, subspace, spectral entropy) with permutation nulls
- [x] `scripts/analysis/visualize_concept_geometry.py` — Mantel profiles, PCA scatter, template sensitivity plots

### Phase F: Integration — DONE (2026-04-11)
- [x] All phases A–E implemented and tested
- [x] 28 retired scripts archived to ~/Documents/.pt-analyses-archive/
- [x] data/README.md updated with pipeline quick reference
- [ ] Full GPU validation against proxy run outputs (deferred — needs node1)

---

## 9. Decisions (Resolved)

- **Naming:** "Activation geometry" and "concept geometry" confirmed. Replaces "shapes" and "manifolds" terminology.
- **Paper figures:** No dedicated paper figure scripts. Visualization scripts produce all available plots, well-organized and labeled. Paper-worthy figures identified after analysis, not pre-designed.
- **Heimdall integration:** Optional `--heimdall-job-id` flag in Track 3 if desired, but not a priority. Separate concern from the core pipeline.
- **Backward compatibility:** None required. This is a clean break — all forward-looking. Old scripts and old output formats do not need to be supported.
- **Track 1+2 plots:** Defer decision. JSON output is the priority; plots can be added later as a visualization pass (same pattern as Tracks 5b/6c).
- **Cross-track correlation:** Defer until Tracks 1-6 are stable. Natural candidate for a future Track 7.
