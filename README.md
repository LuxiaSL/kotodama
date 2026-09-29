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
| NCA v3 data + gates | `python -m kotodama.nca.gen_dyadic_data --out <dir> ...` (token-level mix, banded rules, sidecar; `--finalize` emits eval sets) · score: `python -m kotodama.eval.nca_gates --checkpoint <ckpt> --eval-dir <dir> --tag <t>` · every checkpoint of a run (formation curve): `tools/run_py.sh -m scripts.eval.nca_gates_sweep --ckpt-dir <run> --eval-dir <dir> --init-seed <seed>` |
| pretrain (NCA → language, CPT) | `tools/run_train.sh --config configs/<run>.yaml [--flag value ...]` |
| lm-eval anchors | `tools/launch_lmeval_sharded.sh <ckpt> 0,1,2` · single GPU: `tools/run_py.sh -m scripts.eval.run_lm_eval --checkpoint <ckpt> --config-section 3b` |
| induction score | `tools/run_py.sh -m kotodama.eval.induction --checkpoint <ckpt> --config-section proxy --attn-res-boundaries ddv1 --tag <t>` (random-token repeat loss drop + per-head prefix matching on AttnRes models) |
| telemetry (heimdall) | post per-checkpoint results: `python scripts/eval/hm_backfill.py --job <id> --gates <dir>` · read a family back as PNG + text gist: `python scripts/eval/hm_portrait.py --job <id> --x 'norm_gain/layer_{i}/q_norm' [--y 'geo/layer_{i}/attn_entropy_mean'] --out <png>` (comet with `--y`, heatmap without; repeat `--job` to compare runs) |
| spike resilience | `python scripts/eval/resilience.py --job <id> [--signal train/grad_norm]` (per spike: does geo/rankme_last recover to its pre-spike level, and when) |
| node parity | `python scripts/utils/node_parity.py --hosts <h1> <h2> --mirror <path> --python <venv python>` (same stack on every node before hopping runs) |
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
