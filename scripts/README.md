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

## utils/ — Tools

| Script | Purpose |
|--------|---------|
| `tokenize_data.py` | Tokenize HF datasets into flat binary for `TokenizedDataset` |
| `smoke_test.py` | Quick single-GPU end-to-end validation (model + Muon + z-loss) |
| `profile_step.py` | `torch.profiler` single-step kernel breakdown |
| `visualize_nca.py` | NCA trajectory grid visualizations |

## legacy/ — Completed Proxy Phase

Sweep submission and benchmark scripts from the 108M proxy validation phase. Kept for reference; not needed for current work.
