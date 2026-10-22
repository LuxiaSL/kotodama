# Kotodama

A from-scratch language-model pretraining stack: Llama-style decoders with
**Block Attention Residuals** (AttnRes), **NCA pre-pretraining** (training on
neural-cellular-automata trajectories before language), and **Muon** for matrix
parameters. The shipped model is
[kotodama-3b-base](https://huggingface.co/aethera-gp/kotodama-3b-base-final)
(384B tokens, DD-3B AttnRes boundaries).

Architecture: GQA, RoPE (θ=500K), RMSNorm + QK-norm, SwiGLU, tied embeddings,
SmolLM2 vocabulary (49,152), z-loss. Shapes and their default AttnRes boundaries
live in one table: `src/kotodama/presets.py`.

**Looped depth** (opt-in, model config `loop_start` / `loop_len` / `loop_times`):
a core run of `loop_len` layers starting at `loop_start` executes `loop_times`
times with shared attention + FFN weights. `num_layers` counts effective
positions (AttnRes boundaries, KV caches and capture see one layer per position),
and each position keeps its own pre-norms and AttnRes routing parameters. Presets:
`1b-deep-loop` (52 effective layers from 28 unique), `1b-deep-loopfit` (52 from 42,
boundaries `CF52_FIT`), `1b-moonshot-loop` (72 from 52, boundaries `MOONSHOT_LOOP`);
the server loads each by `--model_size`.

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
| 1B trunk + branch cooldowns | `tools/run_train.sh --config configs/moonshot-1b-shapeA.yaml --global_batch_tokens 2097152 --data_path <train.bin> --mix_data_path <mix.bin> --checkpoint_dir <dir>` (constant LR, no decay; full checkpoints at `--anchor_tokens` land in `<dir>/anchors/`) · cooldown branch at anchor step S: copy `<dir>/anchors/step_<S>.pt` (8-digit step) into a fresh dir and launch there with `--total_steps S+1+K --decay_start_pct (S+1)/(S+1+K) --save_schedule 1B:end --anchor_tokens ""` (training resumes from the newest `step_*.pt` in `--checkpoint_dir`) |
| in-loop evals | add `--inloop_evals riders,induction` to a train launch (see [In-loop evals](#in-loop-evals)) |
| lm-eval anchors | `tools/launch_lmeval_sharded.sh <ckpt> 0,1,2` · single GPU: `tools/run_py.sh -m scripts.eval.run_lm_eval --checkpoint <ckpt> --config-section 3b` |
| Uncheatable BPB | `tools/run_py.sh -m kotodama.eval.uncheatable --checkpoint <ckpt> --config-section 3b --attn-res-boundaries dd3b --out <json>` (per-domain bits/byte on the Uncheatable Eval release, same math as the 3B's published 0.852; data = `--data-dir` / `$KOTODAMA_UNCHEATABLE_DATA`, else the HF dataset; formation sweeps: `--max-docs-per-domain N`; post with `hm_backfill.py --uncheatable <json>`) |
| AttnRes boundaries (DD method) | profile one no-AttnRes checkpoint: `tools/run_py.sh scripts/eval/nca_profile.py --checkpoint <ckpt> --config-section 1b --out <json>` (the algorithm DD-3B was derived with) · fit a run's snapshots until the cuts settle: `tools/run_py.sh scripts/eval/fit_boundaries.py --run <label> --checkpoint <run>/weights/step_*.pt --config-section 1b --out-dir <dir>`, then `--summarize --run <label> --out-dir <dir>` |
| AttnRes routing matrix | `tools/run_py.sh scripts/eval/routing_matrix.py --run <label> --checkpoint <ckpt> [--checkpoint ...] --config-section 1b --attn-res-boundaries dd3b --out-dir <dir>` (every routing site × source on held-out text, per domain; `--plot` renders finished JSONs) |
| repeated-bigram stats | `python scripts/utils/bigram_stats.py name=<stream.bin> [...] --out <json>` (in-context repeated-bigram opportunity / reliability over uint16 streams) |
| induction score | `tools/run_py.sh -m kotodama.eval.induction --checkpoint <ckpt> --config-section proxy --attn-res-boundaries ddv1 --tag <t>` (random-token repeat loss drop + per-head prefix matching on AttnRes models) |
| telemetry (scheduler metrics API at `$KOTODAMA_SCHEDULER_URL`) | post per-checkpoint results: `python scripts/eval/hm_backfill.py --job <id> --gates <dir>` · read a family back as PNG + text gist: `python scripts/eval/hm_portrait.py --job <id> --x 'norm_gain/layer_{i}/q_norm' [--y 'geo/layer_{i}/attn_entropy_mean'] --out <png>` (comet with `--y`, heatmap without; repeat `--job` to compare runs) · geometry diff of two runs (every per-layer family as A / B / B−A heatmaps + gist): `python scripts/eval/hm_geodiff.py --job <A> --job <B> --out-dir <dir> [--max-step N]` |
| spike resilience | `python scripts/eval/resilience.py --job <id> [--signal train/grad_norm]` (per spike: does geo/rankme_last recover to its pre-spike level, and when) |
| node parity | `python scripts/utils/node_parity.py --hosts <h1> <h2> --mirror <path> --python <venv python>` (same stack on every node before hopping runs) |
| binding rider grid | `python -m kotodama.eval.riders {gen,run,aggregate}` (logprob battery against a server) |
| binding probes (other shapes) | `python -m kotodama.eval.bindprobes {gen,run,aggregate}` (core battery: attr / codevar / dialogue / deixis × {0shot, 3shot, 3shot_collide}; same `/logprobs` scorer as the riders) · deixis validity gates V1–V5: `... bindprobes gates --grids DIR --items items.jsonl --chance-model M --learn-model M` (`--pilot` for the small pilot from `gen --families deixis --per-cell deixis=10`) |
| binding probes v3 | `python -m kotodama.eval.bindprobes gen --out DIR --families setbind,answer,adjacency,codevarmut --verify`, then `run` / `aggregate` as above (generated only when named; the default `gen` emits the core battery). setbind: many-to-one sets (hit@1 / exact@k / margin); answer: production-side deixis (`me` / `you` / a name, plus a body-hidden null variant for calibration); adjacency: vocative-free "you" resolved by request → grant adjacency; codevarmut: Python list aliasing (mutate / rebind / copy traps) |
| checkpoints | `python -m kotodama.ckpt {info,strip,list}` — names from `configs/checkpoints.yaml` |
| mirror to a machine without git | `tools/sync.sh [-n] [host:/path]` (exactly the tracked files; default target `$KOTODAMA_SYNC_TARGET` from `site.env`; the remote's `site.env`, venvs, caches, `logs/` and `analysis/` are kept) |
| capture / readout | `kotodama.model.capture` (instrumented AttnRes forward, parity-tested) |
| serve one model | `tools/run_serve.sh --checkpoint <ckpt> [--mode chat]` · public base: `tools/serve_3b_base.sh` |
| serve a fleet | `tools/run_gateway.sh --config configs/gateway.example.yaml` |
| chat | `python -m kotodama.serve.chat [complete]` (sampling laws: `kotodama.serve.laws`) |
| any script | `tools/run_py.sh <script.py or -m module> ...` |

Configs are flat YAML (keys = `train.py` flags) with `extends:` inheritance;
shared blocks live in `configs/base/`. The 3B lineage configs
(`nca-3b-phase1` → `nca-3b-phase3-cotrain` → `3b-language`, plus
`3b-chinchilla-decay`) are the record of the shipped run.
The `ladder-*` configs are 1B-budget arms: language arms extend `ladder-a0.yaml`
(`1b` shape, DD-3B boundaries, 30B tokens; the `-deep*` arms swap in the 52-layer
shapes), NCA-phase arms extend `ladder-a1-nca.yaml`. `moonshot-1b-shapeA.yaml`
runs the `1b-moonshot-loop` shape as a constant-LR trunk.

### Run-shape flags

| flag | effect |
|---|---|
| `--save_schedule 250M:4B,1B:end` | weights-only saves on a piecewise token cadence → `<checkpoint_dir>/weights/` (never pruned) |
| `--anchor_tokens 10B,20B,...` | protected full checkpoints → `<checkpoint_dir>/anchors/` (fork points for cooldown branches) |
| `--stop_at_step N` | exit with a final checkpoint after step N−1; the LR / batch schedule is unchanged (`--total_steps` redefines it) |
| `--init_scheme` | `standard` (default, no-op) / `mimetic` / `orthogonal` / `depth_graded` / `small_attn_out`, applied over the standard init (`kotodama.model.init_schemes`) |
| `--mix_data_path`, `--mix_fraction`, `--mix_window START:END` | interleave a second token bin at an exact token fraction (token-debt, resumable), optionally only inside a global-token window |
| `--data_seed` | data-order seed, independent of the init `--seed` (default = `--seed`) |

### In-loop evals

`train.py --inloop_evals <suites>` scores small suites on the live weights during
training (in-process, every DDP rank takes a shard; off by default; not with
`--tp_size > 1`):

| suite | what | needs |
|---|---|---|
| `riders` (`on` = this alone) | riders-lite: 200 balanced std items × 4 registers × {0shot, 3shot, 3shot_relet} (shots re-lettered F/G/H→A/B/C), scored by the same function as `/logprobs`; acc, chance-corrected acc, margin, error types (`wrong_box` / `init_query` / `other`) | nothing (generates the standard battery; or `--inloop_riders_items` + `--inloop_riders_shots`) |
| `induction` | random-token repeat loss drop | nothing |
| `uncheatable` | bits/byte on the first `--inloop_uncheatable_docs` docs per domain | `--inloop_uncheatable_path` |
| `nca_gates` | NCA ledger accuracy (NCA-phase runs) | `--inloop_nca_gates_dir` |

Cadence `--inloop_schedule` (default `0:0.025,decay:0.005`: every 2.5% of tokens,
every 0.5% once the LR decay starts, plus the first and final steps). Each point
posts one `hm_log` batch (`inloop/riders/narrative/3shot/acc`, …, `inloop/time_s`)
and writes every row to `<checkpoint_dir>/inloop/step_<N>.jsonl` (same weights as
checkpoint `step_<N>`); resumes skip points already written. Size knobs:
`--inloop_riders_n`, `--inloop_max_batch_tokens`, `--no_inloop_riders_relet`.

The server exposes `/v1/chat/completions`, `/v1/completions`, `/generate`
(full sampling control, streaming), `/logprobs` (batched prefill scoring),
`/v1/models`, `/info`, `/health`.

## Layout

```text
src/kotodama/
  model/      llama.py (model + AttnRes + looped depth), flash_attn_res/ (Triton), decode_engine, capture, init_schemes
  nca/        NCA trajectory generator (+ dyadic variant)
  training/   train.py, Muon, checkpointing, inloop (in-loop evals)
  data/       uint16 token-stream dataset
  eval/       model loading, lm-eval adapter, uncheatable, induction, nca_gates, riders/ (binding battery), bindprobes/ (binding probes)
  monitoring/ per-layer geometric telemetry, wandb callback
  serve/      server, gateway, chatml, laws, chat client
  presets.py  shapes + AttnRes boundaries + tokenizer
  ckpt.py     checkpoint load/strip/info/registry
configs/      training runs (+ base/), gateway example, checkpoint registry
scripts/      eval/ (lm-eval, boundary fitting, routing, telemetry readers), benchmark/ (serving perf + parity batteries), utils/ (kernel parity, data, smoke)
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
