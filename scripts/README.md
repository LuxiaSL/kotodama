# Scripts

Full spec: `docs/analysis-pipeline-spec.md`

## analysis/ — 6-Track Analysis Pipeline

Run in track order. Tracks 1-4 can run independently; Track 5/6 extraction requires GPU, visualization is CPU-only.

| Track | Script | What it does | Requires |
|-------|--------|-------------|----------|
| 1+2 | `analyze_run.py` | Training metrics, loss curves, geometric health from JSONL logs | CPU |
| 3 | `eval_generate.py` | Generate text samples from checkpoints with held-out perplexity | GPU |
| 4 | `analyze_text_quality.py` | 59-metric semantic quality profiling on generated text | CPU |
| 5a | `extract_activation_geometry.py` | Extract point clouds, eigenspectra, attention patterns from checkpoints | GPU |
| 5b | `visualize_activation_geometry.py` | PCA, eigenspectra, trajectory plots from Track 5a outputs | CPU |
| 6a | `extract_concept_geometry.py` | Extract residual stream activations at concept token positions | CPU |
| 6b | `analyze_concept_geometry.py` | Cyclic, ordinal, smoothness, spectral entropy analysis on 6a outputs | CPU |
| 6c | `visualize_concept_geometry.py` | Mantel profiles, PCA scatter, template sensitivity plots | CPU |

### Quick start

```bash
cd pretraining

# Track 1+2: training run summary
python scripts/analysis/analyze_run.py --summary --metrics data/sweep_p3_muon002_metrics.jsonl

# Track 3: generate from a checkpoint
python scripts/analysis/eval_generate.py --checkpoint checkpoints/proxy_sweep/p3-muon-002/step_00045775.pt --prompt-set standard

# Track 4: analyze generated text
python scripts/analysis/analyze_text_quality.py outputs/generations/p3-muon-002/

# Track 5: activation geometry (GPU then CPU)
python scripts/analysis/extract_activation_geometry.py --checkpoint checkpoints/proxy_sweep/p3-muon-002/step_00045775.pt
python scripts/analysis/visualize_activation_geometry.py outputs/geometry/p3-muon-002/

# Track 6: concept geometry
python scripts/analysis/extract_concept_geometry.py --checkpoint checkpoints/proxy_sweep/p3-muon-002/step_00045775.pt
python scripts/analysis/analyze_concept_geometry.py outputs/concepts/p3-muon-002/
python scripts/analysis/visualize_concept_geometry.py outputs/concepts/p3-muon-002/
```

All scripts support `--help`. Checkpoint registry: `configs/checkpoints.yaml`. Prompt sets: `configs/prompts.yaml`. Concept sets: `configs/concepts.yaml`.

## eval/ — lm-eval battery

| Script | Purpose |
|--------|---------|
| `run_lm_eval.py` | lm-evaluation-harness on Luxia checkpoints (batched loglikelihood since 2026-07-04; `--batch-size` rows/forward, `--max-batch-tokens` memory cap) |
| `merge_lmeval_shards.py` | Union per-shard results.json into the standard `analysis/lm_eval/<ckpt>/results.json` |
| `check_lmeval_equivalence.py` | Equivalence gate: diff new results.json vs a banked one; nonzero exit on point-metric drift beyond `--tol` |
| `assemble_lmeval_table.py` | Multi-checkpoint comparison table from results.json files |

Fast path for the standard 10-task battery (gpu-host): `tools/launch_lmeval_sharded.sh <ckpt> <gpu,gpu,gpu> [bs]`
— 3 task shards balanced by cost (hellaswag alone ≈ half the battery), auto-merge, never
clobbers an existing results.json.

```bash
# One checkpoint, 3 free GPUs (~3 min):
tools/launch_lmeval_sharded.sh /models/kotodama-data/<ckpt>.pt.zst 4,5,6 32

# Multiple checkpoints, one GPU each (~5.5 min each, run in parallel):
CUDA_VISIBLE_DEVICES=<gpu> KOTODAMA_NO_TRITON_ATTNRES=1 HF_HOME=/models/huggingface \
TMPDIR=/models/kotodama-data/tmp python -m scripts.eval.run_lm_eval \
  --checkpoint <ckpt> --config-section model \
  --attn-res-boundaries 0,1,3,7,15,19,24 --batch-size 32 --output-dir analysis/lm_eval_v2
```

v2 harness config (frozen 2026-07-04): batched bs=32, max-batch-tokens 32768, **eager — no
`--compile`** (compile = different kernel path; it was the source of the ≤0.26pp v1 drift).
v2 anchor rows for the 4 pretraining checkpoints: `analysis/lm_eval_v2/<ckpt>/results.json`.
Never mix v1 (2026-06-09, compiled) and v2 numbers in one comparison; when diagnosing a
posttrained checkpoint, re-run its parent/base arm same-harness in the same session.

**Equivalence protocol:** after ANY harness change (batching, kernels, sharding), re-run at
least one banked checkpoint+task and gate with `check_lmeval_equivalence.py` before the new
numbers enter a comparison. Banked anchor: `analysis/lm_eval/3b-language-FINAL-step195311.pt/results.json`
(2026-06-09, lm-eval 0.4.11, bf16). A fast benchmark that drifted is worse than a slow one.

## utils/ — Tools

| Script | Purpose |
|--------|---------|
| `tokenize_data.py` | Tokenize HF datasets into flat binary for `TokenizedDataset` |
| `smoke_test.py` | Quick single-GPU end-to-end validation (model + Muon + z-loss) |
| `profile_step.py` | `torch.profiler` single-step kernel breakdown |
| `visualize_nca.py` | NCA trajectory grid visualizations |

## legacy/ — Completed Proxy Phase

Sweep submission and benchmark scripts from the 108M proxy validation phase. Kept for reference; not needed for current work.
