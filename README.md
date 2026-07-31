# Kotodama

Kotodama is an experimental PyTorch codebase for training and serving decoder-only language models. It contains the current 3B and 7B training paths, Attention Residuals, optional Neural Cellular Automata (NCA) pre-training, distributed Muon, evaluation tooling, and an OpenAI-compatible inference server.

This repository contains source code, configurations, and small benchmark artifacts. It does not include training data, checkpoints, credentials, or deployment-specific infrastructure.

## Current Capabilities

- 3B and 7B Llama-style decoder models with grouped-query attention, RoPE, RMSNorm, SwiGLU, and tied embeddings.
- Block Attention Residuals with Triton and PyTorch implementations.
- DDP training with Muon for matrix parameters and AdamW for the remaining parameters.
- Optional FP8 and MXFP8 paths, activation checkpointing, FlashAttention/SDPA/flex attention backends, and `torch.compile`.
- NCA trajectory generation and NCA-to-language initialization experiments.
- Checkpoint loading, generation, lm-eval integration, parity checks, and profiling utilities.
- FastAPI inference server and a multi-replica gateway with health checks and prefix-cache affinity.

The 108M proxy experiments and their reports remain in the tree as historical research artifacts. The actively maintained configs are the 3B and 7B training, serving, and benchmark paths.

## Requirements

- Python 3.12+
- PyTorch with a CUDA build for GPU training or fast inference
- CUDA-compatible NVIDIA hardware for the optional compiled, FlashAttention, and FP8 paths

Install the base package and the extras appropriate to your workflow:

```bash
uv sync --extra training --extra eval
```

`flash-attn`, `liger-kernel`, `torchao`, and `lm-eval` are optional and feature-dependent. Consult `pyproject.toml` and the selected config before running a GPU workflow.

## Quick Start

Run a short 7B canary using synthetic data:

```bash
torchrun --nproc_per_node=8 -m src.training.train \
  --config configs/canary-7b.yaml
```

The canary is designed for a multi-GPU CUDA environment and performs no useful checkpoint save. For other experiments, start from a config in `configs/` and provide a tokenized data path or use its `random_data` mode.

Serve a compatible checkpoint:

```bash
python serve.py \
  --checkpoint /path/to/checkpoint.pt \
  --model_size 3b \
  --mode chat \
  --prefix-cache
```

The server listens on port 2222 by default. It exposes `POST /v1/chat/completions`, `POST /v1/completions`, `POST /generate`, `GET /v1/models`, `GET /info`, and `GET /health`.

Start a gateway fleet after adapting a gateway YAML file to local checkpoint paths and runtime settings:

```bash
python gateway.py --config configs/gateway-minitest.yaml --gpus 0
```

Gateway configs use `python: python` and `workdir: .` as portable defaults. Checkpoint locations, caches, GPU placement, and any scheduler integration are operator configuration.

## Evaluation And Validation

The repository includes focused parity, smoke, and benchmark tools:

```bash
python scripts/utils/phase1_bwd_parity.py
python scripts/utils/phase2_bwd_parity.py
python scripts/utils/fa4_varlen_parity.py
python scripts/benchmark/validate_engine.py --checkpoint /path/to/checkpoint.pt
python -m scripts.eval.run_lm_eval --help
```

Most scripts are hardware- and checkpoint-dependent. Run `--help` before invoking a tool and treat configurations in `scripts/legacy/` as archived experiment launchers, not production defaults.

## Repository Layout

```text
configs/       Training, serving, and benchmark configurations
docs/          Research notes and handoff documents
scripts/       Analysis, evaluation, benchmark, utility, and legacy scripts
src/data/      Dataset loading
src/eval/      Evaluation, generation, and model-loading utilities
src/model/     Model, decode engine, and Attention Residual implementations
src/nca/       NCA data generation
src/training/  Distributed training, Muon, and checkpoint handling
tests/         CPU-focused equivalence tests
tools/         Portable runner and convenience scripts
```

## Configuration And Security

Do not commit checkpoints, datasets, scheduler URLs, private hostnames, local paths, or API keys. The launcher wrappers accept `KOTODAMA_WORKDIR`, `KOTODAMA_VENV`, `KOTODAMA_CACHE_ROOT`, and `KOTODAMA_SCHEDULER_URL` for environment-specific setup. Telemetry credentials such as `WANDB_API_KEY` must be supplied through the environment or a secret manager.

For prior experimental context, see `PROXY-REPORT.md` and the documents in `docs/`; they should be read as dated research notes rather than a deployment guide.
