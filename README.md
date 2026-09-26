# Kotodama

A from-scratch language-model pretraining stack: Llama-style decoders with
**Block Attention Residuals** (AttnRes), **NCA pre-pretraining** (training on
neural-cellular-automata trajectories before language), and **Muon** for matrix
parameters. The shipped model is
[kotodama-3b-base](https://huggingface.co/aethera-gp/kotodama-3b-base-final)
(384B tokens, DD-3B AttnRes boundaries).

Architecture: GQA, RoPE (θ=500K), RMSNorm + QK-norm, SwiGLU, tied embeddings,
SmolLM2 vocabulary (49,152), z-loss. Shapes live in one table:
`src/kotodama/presets.py`.

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[training,eval]"
cp site.env.example site.env   # optional: data root, venv, caches for this machine
```

Every launcher in `tools/` sources `tools/_env.sh`, which loads `site.env`,
activates the venv, and sets the package path and compile caches.

## Golden paths

| task | command |
|---|---|
| NCA data | `tools/nca_gen.sh` (defaults = the 3B recipe: 3500 rules × 20 sims, seed 17) |
| pretrain (NCA → language, CPT) | `tools/run_train.sh --config configs/<run>.yaml [--flag value ...]` |
| lm-eval anchors | `tools/launch_lmeval_sharded.sh <ckpt> 0,1,2` · single GPU: `tools/run_py.sh -m scripts.eval.run_lm_eval --checkpoint <ckpt> --config-section 3b` |
| binding rider grid | `python -m kotodama.eval.riders {gen,run,aggregate}` (logprob battery against a server) |
| checkpoints | `python -m kotodama.ckpt {info,strip,list}` — names from `configs/checkpoints.yaml` |
| capture / readout | `kotodama.model.capture` (instrumented AttnRes forward, parity-tested) |
| serve one model | `tools/run_serve.sh --checkpoint <ckpt> [--mode chat]` · public base: `tools/serve_3b_base.sh` |
| serve a fleet | `tools/run_gateway.sh --config configs/gateway.example.yaml` |
| chat | `python -m kotodama.serve.chat [complete]` (sampling laws: `kotodama.serve.laws`) |
| any script | `tools/run_py.sh <script.py or -m module> ...` |

Configs are flat YAML (keys = `train.py` flags) with `extends:` inheritance;
shared blocks live in `configs/base/`. The 3B lineage configs
(`nca-3b-phase1` → `nca-3b-phase3-cotrain` → `3b-language`, plus
`3b-chinchilla-decay`) are the record of the shipped run.

The server exposes `/v1/chat/completions`, `/v1/completions`, `/generate`
(full sampling control, streaming), `/logprobs` (batched prefill scoring),
`/v1/models`, `/info`, `/health`.

## Layout

```text
src/kotodama/
  model/      llama.py (model + AttnRes), flash_attn_res/ (Triton), decode_engine, capture
  nca/        NCA trajectory generator (+ dyadic variant)
  training/   train.py, Muon, checkpointing
  data/       uint16 token-stream dataset
  eval/       model loading, lm-eval adapter, riders/ (binding battery)
  serve/      server, gateway, chatml, laws, chat client
  presets.py  shapes + AttnRes boundaries + tokenizer
  ckpt.py     checkpoint load/strip/info/registry
configs/      training runs (+ base/), gateway example, checkpoint registry
scripts/      eval/ (lm-eval), benchmark/ (serving perf + parity batteries), utils/ (kernel parity, data, smoke)
tools/        launchers (all via _env.sh)
tests/        CPU suite: pytest
```

## Tests

```bash
.venv/bin/python -m pytest
```

CPU-only; GPU batteries skip without CUDA. Kernel parity checks
(`scripts/utils/*_parity.py`) and serving batteries (`scripts/benchmark/check_*.py`)
run standalone on a GPU.
