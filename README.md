# Kotodama

Kotodama is a from-scratch language-model pretraining stack in one Python
package, `kotodama`. It trains Llama-style decoders with **Block Attention
Residuals** (AttnRes) whose block boundaries are placed from the network's own
layer geometry, optimizes matrix parameters with **Muon**, supports **looped
depth** (a run of layers executed several times with shared weights), and can
pre-pretrain on **neural-cellular-automata (NCA) trajectories** before language.
Around the trainer sit an evaluation suite (lm-eval, Uncheatable BPB, induction,
judge-free binding batteries), per-layer telemetry, and an OpenAI-compatible
serving stack.

The shipped model is
[kotodama-3b-base](https://huggingface.co/aethera-gp/kotodama-3b-base-final): a
3B base model trained on 384B tokens (about 48B unique; see
`--legacy_rank0_data` in `train.py` for the data-loader regime it ran under),
with DD-3B AttnRes boundaries `0,1,3,7,15,19,24`. Its architecture is frozen.

## Architecture

- **Decoder:** grouped-query attention, RoPE (θ = 500K), RMSNorm with QK-norm,
  SwiGLU feed-forward, tied input/output embeddings, z-loss on the logits.
- **Tokenizer:** SmolLM2 (`HuggingFaceTB/SmolLM2-135M`, vocabulary 49,152).
- **Block Attention Residuals** ([Moonshot AI, 2026](https://arxiv.org/abs/2603.15031)): instead of one running
  residual sum, the layers are grouped into blocks; each sublayer reads a
  softmax-weighted mixture of the committed block outputs and the current
  block's partial sum, routed by a learned per-layer pseudo-query. Fused Triton
  kernels for the two routing phases are vendored in
  `kotodama/model/flash_attn_res/`; a PyTorch fallback runs on CPU.
- **Muon** (momentum orthogonalized by Newton–Schulz) for hidden matrices,
  AdamW for embeddings, norms and scalars (`kotodama/training/muon.py`,
  recipe in `configs/base/optim-muon.yaml`).
- **Looped depth** (optional): see [Looped depth](#looped-depth).

Model shapes, their default AttnRes boundaries and the tokenizer live in one
table, `src/kotodama/presets.py`. Every other model constant (RoPE θ, norm eps,
QK-norm, tied embeddings, z-loss weight) is a `KotodamaModelConfig` default in
`kotodama/model/llama.py`.

| shape | layers (effective) | width | notes |
|---|---|---|---|
| `3b` | 28 | 3072 | the shipped model; DD-3B boundaries |
| `7b` | 32 | 4096 | throughput configs only |
| `1b` | 28 | 2048 | ~1.42B; DD-3B boundaries verbatim |
| `1b-deep` | 52 | 1536 | the `1b` parameter budget spent on depth |
| `1b-deep-loop` | 52 from 28 unique | 1536 | 12-layer core run 3× |
| `1b-deep-loopfit` | 52 from 42 unique | 1536 | 10-layer core run 2×; boundaries `CF52_FIT` |
| `1b-moonshot-loop` | 72 from 52 unique | 1536 | 20-layer core run 2×; boundaries `MOONSHOT_LOOP` |
| `intermediate`, `proxy`, `smoke` | 28 / 28 / 4 | 1024 / 512 / 256 | small-scale and test shapes |

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[training,eval]"
cp site.env.example site.env   # optional: data root, venv, caches for this machine
```

The `nca` extra adds the JAX stack; `all` installs every extra. Python ≥ 3.12.

`site.env` (gitignored) holds everything machine-specific: `KOTODAMA_VENV`,
`KOTODAMA_DATA_ROOT` (token streams, NCA data, checkpoints),
`KOTODAMA_CACHE_ROOT` (persistent torch.compile / Triton caches),
`KOTODAMA_SCHEDULER_URL` and `KOTODAMA_SCHEDULER_JOB_ID_VAR` (metrics API),
`KOTODAMA_UNCHEATABLE_DATA`, `HF_HOME`. Every launcher in `tools/` sources
`tools/_env.sh`, which loads `site.env`, activates the venv, and exports the
package path and compile caches, so one machine's paths live in one file.

## Golden paths

One command per task. `tools/run_py.sh <script.py | -m module> ...` runs any
single-process job inside the launcher environment.

### Data

| task | command |
|---|---|
| NCA token stream (3B recipe) | `tools/nca_gen.sh` (defaults: 3500 rules × 20 sims, seed 17; split into train/eval and verified) |
| NCA v3 stream + eval sets | `python -m kotodama.nca.gen_dyadic_data --out <dir> ...` (token-level mix, banded rules, per-episode sidecar), then `--finalize --out <dir>` to concatenate and emit the eval sets |
| repeated-bigram statistics | `python scripts/utils/bigram_stats.py name=<stream.bin> [...] --out <json>` (in-context repeated-bigram opportunity and reliability over uint16 streams) |

### Train

| task | command |
|---|---|
| pretrain (NCA, language, CPT) | `tools/run_train.sh --config configs/<run>.yaml [--flag value ...]` (torchrun over `KOTODAMA_NPROC` GPUs, default 8) |
| 1B trunk | `tools/run_train.sh --config configs/moonshot-1b-shapeA.yaml --global_batch_tokens 2097152 --data_path <train.bin> --mix_data_path <mix.bin> --checkpoint_dir <dir>` |
| cooldown branch at anchor step S | copy `<dir>/anchors/step_<S>.pt` (8-digit step) into a fresh `--checkpoint_dir` and launch there with `--total_steps S+1+K --decay_start_pct (S+1)/(S+1+K) --save_schedule 1B:end --anchor_tokens ""` |
| in-loop evals | add `--inloop_evals riders,induction` to a train launch |

### Post-train

| task | command |
|---|---|
| chat SFT | `tools/run_py.sh -m kotodama.posttrain.sft --config configs/posttrain/tsumugi-sft.yaml --checkpoint <base.pt> --train_jsonl <chat.jsonl> --out_dir <dir>` (rows `{"messages": [...]}`; assistant turns train) |
| DPO / DPOP / IPO | `tools/run_py.sh -m kotodama.posttrain.pref --config configs/posttrain/tsumugi-dpo.yaml --checkpoint <init.pt> --data <pairs.jsonl> --out_dir <dir>` |
| KTO against self-copying | `python -m kotodama.posttrain.selfcopy label --in <samples.jsonl> --out <kto.jsonl>`, then `pref` with `configs/posttrain/tsumugi-kto-repetition.yaml` |
| instruction data → transcripts | `python -m kotodama.posttrain.transcript convert-messages --in <*.parquet> --out <transcripts.jsonl>` (`User:` / `Assistant:`; `convert-markdown` for `**Name:**` dialogues) |
| self-play seeds | `python -m kotodama.posttrain.seeds --endpoint <url> --n 2000 --out <seeds.jsonl>` (the model writes each cast and setting) |
| self-play transcripts | `python -m kotodama.posttrain.selfplay --seeds <seeds.jsonl> --endpoint <url> --out <run>` (K candidates per turn, mechanical vetoes, uniform choice among survivors, every candidate banked) |
| self-play → SFT data / health / coherence study | `python -m kotodama.posttrain.corpus {assemble,meters,precision-sample,rate,score} <run> ...` |

Conversations are plain-text transcripts with no special tokens: an optional prose header, then `Name: text`
turns separated by blank lines (`kotodama.posttrain.transcript`); `--train_speakers all` puts loss on every turn.
`kotodama.posttrain` trains single-GPU on pretraining's precision recipe (bf16
weights, fp32 masters in Muon + AdamW) with fp32 AttnRes routing, the numerics the
serving engine computes. Preference runs precompute reference log-probs once (disk-cached, verified on
reuse) and refuse more than one epoch. The self-copy labeller flags verbatim reuse
of the model's own text and template loops, never reuse of names or objects.

### Evaluate

| task | command |
|---|---|
| lm-eval battery | `tools/launch_lmeval_sharded.sh <ckpt> 0,1,2` (task-sharded over GPUs) · single GPU: `tools/run_py.sh -m scripts.eval.run_lm_eval --checkpoint <ckpt> --config-section 3b` |
| Uncheatable BPB | `tools/run_py.sh -m kotodama.eval.uncheatable --checkpoint <ckpt> --config-section 3b --attn-res-boundaries dd3b --out <json>` (data: `--data-dir`, else `$KOTODAMA_UNCHEATABLE_DATA`, else the HF dataset; checkpoint sweeps: `--max-docs-per-domain N`) |
| induction score | `tools/run_py.sh -m kotodama.eval.induction --checkpoint <ckpt> --config-section proxy --attn-res-boundaries ddv1 --tag <t>` |
| NCA gates | `python -m kotodama.eval.nca_gates --checkpoint <ckpt> --eval-dir <dir> --tag <t>` · every checkpoint of a run: `tools/run_py.sh -m scripts.eval.nca_gates_sweep --ckpt-dir <run> --eval-dir <dir> --init-seed <seed>` |
| binding suite (every battery, one model load) | `tools/run_py.sh -m kotodama.eval.binding run --model koto:<ckpt> --name <m> --batteries scorecard --out <dir>` (`hf:<path>` for public models; `list` shows batteries and groups) · scorecard: `python -m kotodama.eval.binding report --scores <dir>` · capacity size law: `python -m kotodama.eval.binding capacity --scores <dir> --params <tsv>` |
| rider grid | `python -m kotodama.eval.riders {gen,run,aggregate}` (against a server's `/logprobs`) |
| binding probes, core | `python -m kotodama.eval.bindprobes {gen,run,aggregate}` · deixis validity gates: `python -m kotodama.eval.bindprobes gates --grids <dir> --items items.jsonl --chance-model <m> --learn-model <m> [--pilot]` |
| binding probes, v3 families | `python -m kotodama.eval.bindprobes gen --out <dir> --families setbind,answer,adjacency,codevarmut --verify`, then `run` / `aggregate` |
| AttnRes boundaries from one checkpoint | `tools/run_py.sh scripts/eval/nca_profile.py --checkpoint <ckpt> --config-section 1b --out <json>` |
| AttnRes boundaries over a run | `tools/run_py.sh scripts/eval/fit_boundaries.py --run <label> --checkpoint <run>/weights/step_*.pt --config-section 1b --out-dir <dir>`, then `--summarize --run <label> --out-dir <dir>` |
| AttnRes routing matrix | `tools/run_py.sh scripts/eval/routing_matrix.py --run <label> --checkpoint <ckpt> [--checkpoint ...] --config-section 1b --attn-res-boundaries dd3b --out-dir <dir>` (every routing site × source on held-out text, per domain; `--plot` renders finished JSONs) |
| capture / readout | `kotodama.model.capture` (instrumented AttnRes forward, parity-tested against `llama.py`) |

### Serve

| task | command |
|---|---|
| one model | `tools/run_serve.sh --checkpoint <ckpt> [--mode chat] [--model_size <shape>]` |
| the public 3B base | `tools/serve_3b_base.sh` (downloads the checkpoint once, then serves it) |
| the released kotodama-tsumugi 1B base | `tools/run_serve.sh --checkpoint hf:LuxiaSL/kotodama-tsumugi-1b --model_size 1b-tsumugi --rope_fp32 --mode base --law tsumugi` (published weights load from the Hub or a downloaded directory) |
| a fleet behind one port | `tools/run_gateway.sh --config configs/gateway.example.yaml` |
| chat / completion client | `python -m kotodama.serve.chat [complete]` (sampling laws: `kotodama.serve.laws`) |

The server exposes `/v1/chat/completions`, `/v1/completions`, `/generate` (full
sampling control, streaming), `/logprobs` (batched prefill scoring, the
binding batteries' scorer), `/v1/models`, `/info` and `/health`. The default
decode path is a static-KV-cache, compiled, CUDA-graphed engine
(`kotodama/model/decode_engine.py`); the eager `llama.py` loop is the parity
reference. ChatML and stop tokens live in `kotodama/serve/chatml.py`.

### Telemetry and tooling

| task | command |
|---|---|
| post per-checkpoint results to the metrics API | `python scripts/eval/hm_backfill.py --job <id> --gates <dir>` (Uncheatable JSONs: `--uncheatable <json>`) |
| read a metric family back | `python scripts/eval/hm_portrait.py --job <id> --x 'norm_gain/layer_{i}/q_norm' [--y 'geo/layer_{i}/attn_entropy_mean'] --out <png>` (heatmap without `--y`, trajectory with it; repeat `--job` to compare runs) |
| geometry diff of two runs | `python scripts/eval/hm_geodiff.py --job <A> --job <B> --out-dir <dir> [--max-step N]` |
| spike resilience | `python scripts/eval/resilience.py --job <id> [--signal train/grad_norm]` (per loss spike: whether and when `geo/rankme_last` recovers) |
| checkpoints | `python -m kotodama.ckpt {info,strip,list}` (names from `configs/checkpoints.yaml`) |
| node parity | `python scripts/utils/node_parity.py --hosts <h1> <h2> --mirror <path> --python <venv python>` (same stack on every machine before moving a run) |
| mirror to a machine without git | `tools/sync.sh [-n] [host:/path]` (exactly the tracked files; default target `$KOTODAMA_SYNC_TARGET`; the remote's `site.env`, venvs, caches, `logs/` and `analysis/` are kept) |

The telemetry readers talk to the scheduler metrics API at
`$KOTODAMA_SCHEDULER_URL`; `train.py` posts to it when a job id is present.

## Configs

Configs are flat YAML whose keys are `train.py` flags, with `extends:`
inheritance; CLI flags override config values. Shared blocks live in
`configs/base/` (`optim-muon`, `runtime-production`, `3b-lineage`,
`7b-throughput`). `tests/data/config_snapshot.json` pins every config's
effective values; it is regenerated on purpose with
`KOTODAMA_WRITE_CONFIG_SNAPSHOT=1`.

- **The 3B lineage** (`nca-3b-phase1` → `nca-3b-phase3-cotrain` → `3b-language`,
  plus `3b-chinchilla-decay`) is the record of the shipped run: NCA
  pre-pretraining without AttnRes, an NCA co-train with AttnRes on the
  boundaries derived from that run's geometry, then language pretraining from
  the NCA checkpoint with re-initialized embeddings (`--resume_nca`).
- **`ladder-*`** are 1B-budget arms: language arms extend `ladder-a0.yaml`
  (`1b` shape, DD-3B boundaries, 30B tokens; the `-deep*` arms use the 52-layer
  shapes), NCA-phase arms extend `ladder-a1-nca.yaml`, and `ladder-cf-*` are
  no-AttnRes counterfactuals for boundary fitting.
- **`moonshot-1b-shapeA.yaml`** runs the `1b-moonshot-loop` shape as a
  constant-LR trunk with anchors (see [Trunk and cooldown branches](#trunk-and-cooldown-branches)).
- **`canary-7b*`, `bench-7b-profile`, `soak-7b`** are 7B throughput configs
  (canaries: `tools/run_canary.sh 7b`); their uniform AttnRes blocks are a
  placeholder, not a geometry fit.

## How the distinctive pieces work

### AttnRes boundaries from layer geometry

Kotodama places block starts from the model's own geometry rather than
uniformly. The DD method profiles a checkpoint trained **without** AttnRes,
scores layer-to-layer transitions (attention entropy; in `fit_boundaries.py`
also linear CKA between adjacent layers' outputs), and suggests
block starts under explicit rules (an isolated embedding layer, a cushioned
last block, a contiguous middle block). `scripts/eval/nca_profile.py` is the
archived algorithm that produced DD-3B, kept verbatim and pinned by a test.
`scripts/eval/fit_boundaries.py` fits every snapshot of a no-AttnRes run on
held-out language text and declares a fit settled at the first snapshot after
which its cuts never change. The resulting tuples (`DD3B`, `DDV1`,
`DD3B_DEEP52`, `CF52_FIT`, `MOONSHOT_LOOP`) are named in `presets.py`.

### Looped depth

Model config `loop_start` / `loop_len` / `loop_times`: a core of `loop_len`
layers starting at `loop_start` executes `loop_times` times with shared
attention and FFN weights. `num_layers` counts effective positions: AttnRes
boundaries, KV caches and capture see one layer per position, and each position
keeps its own pre-norms and AttnRes routing parameters, so every pass routes
separately. The looped presets load by name in training, evaluation and the
server (`--model_size`).

### Trunk and cooldown branches

A trunk trains at constant LR to a token ceiling and never decays
(`decay_start_pct: 1.0`). Two checkpoint streams come off it:

| flag | effect |
|---|---|
| `--save_schedule 250M:4B,1B:end` | weights-only saves on a piecewise token cadence → `<checkpoint_dir>/weights/` (never pruned) |
| `--anchor_tokens 10B,20B,...` | protected full (resumable) checkpoints → `<checkpoint_dir>/anchors/` |

An annealed read at any anchor is a **branch**: the anchor is copied into a
fresh directory and training resumes there (from the newest `step_*.pt` in
`--checkpoint_dir`) with a short LR decay that starts exactly at the anchor step
(the cooldown command under [Train](#train)). The trunk is never disturbed, so
one run yields annealed models at every anchor.

Other run-shape flags:

| flag | effect |
|---|---|
| `--stop_at_step N` | exit with a final checkpoint after step N−1; the LR and batch schedule are unchanged (`--total_steps` redefines them) |
| `--init_scheme` | `standard` (default) / `mimetic` / `orthogonal` / `depth_graded` / `small_attn_out`, applied over the standard init (`kotodama.model.init_schemes`) |
| `--mix_data_path`, `--mix_fraction`, `--mix_window START:END` | interleave a second token stream at an exact token fraction (token-debt scheduler, resumable), optionally only inside a global-token window |
| `--data_seed` | data-order seed, independent of the init `--seed` (default: `--seed`) |
| `--legacy_rank0_data` | every DDP rank reads rank 0's data (effective batch = global / world size), for reproducing runs trained under that regime |

### In-loop evals

`train.py --inloop_evals <suites>` scores small suites on the live weights
in-process during training. Every DDP rank scores a shard, rank 0 gathers; off
by default, and not available with `--tp_size > 1`.

| suite | what | needs |
|---|---|---|
| `riders` | riders-lite: `--inloop_riders_n` balanced items (default 200) × 4 registers × {0shot, 3shot, 3shot_relet}, scored by the same function as `/logprobs`; accuracy, chance-corrected accuracy, margin, error types | nothing (or `--inloop_riders_items` + `--inloop_riders_shots`) |
| `induction` | random-token repeat loss drop | nothing |
| `uncheatable` | bits/byte on the first `--inloop_uncheatable_docs` documents per domain | `--inloop_uncheatable_path` |
| `nca_gates` | NCA ledger accuracy, for NCA-phase runs | `--inloop_nca_gates_dir` |

Cadence is `--inloop_schedule` (default `0:0.025,decay:0.005`: every 2.5% of
tokens, every 0.5% once the LR decay starts, plus the first and final steps). A
point fires on the same step rule as `--save_schedule`, so eval `step_N` and
checkpoint `step_N` hold the same weights. Each point posts one metrics batch
(`inloop/...` keys) and writes every row to
`<checkpoint_dir>/inloop/step_<N>.jsonl`; resumes skip points already written.
Size knobs: `--inloop_max_batch_tokens`, `--no_inloop_riders_relet`.

### Binding batteries

The binding batteries are judge-free: each item has a closed candidate set of
single-token answers, and the read is the logprob of each candidate at the
answer position, scored through the server's `/logprobs`
(`kotodama.eval.riders.score`). Chance is 1/C for C candidates;
aggregates report chance-corrected accuracy, (acc − chance) / (1 − chance),
alongside margin (gold logprob minus the best distractor) and typed errors.
Every item also runs under few-shot conditions with exemplars.

- **Rider grid** (`kotodama.eval.riders`): entity-state tracking. N boxes go
  through M operations (swap, move, put, overwrite) and one box is queried;
  the same abstract item is rendered in four registers (Python code, narrative
  prose, screenplay, chat). Candidates are the objects mentioned, so chance is
  1/C per item; the gold is never the most recently mentioned object outside a
  recency-control subset. A difficulty-0 copy tier (1 box, 0 ops, 8 candidates)
  gates harness validity per register. The `3shot_relet` condition re-letters
  the exemplars' boxes to separate following a procedure from copying labels.
- **Bindprobes, core** (`kotodama.eval.bindprobes`): the same question in
  other shapes, each under `0shot`, `3shot` (collision-free exemplars) and
  `3shot_collide` (exemplars re-labeled into the item's own names, different
  bindings).
  - `attr`: N ∈ {2, 4, 6, 8} people each hold one object, plus unbound
    distractors; one is queried.
  - `codevar`: variable binding in Python with copies, reassignments and swaps;
    candidates are every literal in the program.
  - `dialogue`: multi-party prose in which objects change hands, partly through
    pronouns; query the holder of an object or the object of a holder.
  - `deixis`: a chat transcript where ownership is stated only through
    I / me / you / your; at least 3 candidates (chance ≤ 1/3); each item comes
    with a third-person twin and, at three speakers, a speaker-transposed twin,
    so the deixis penalty and twin consistency are read off the pairs.
    `gates` runs the validity gates V1–V5.
- **Bindprobes, v3 families** (generated only when named):
  - `setbind`: one person holds a set of k ∈ {1, 2, 3} objects among 8 in
    play; hit@1 (chance k/8), exact@k (chance 1/C(8, k)) and set margin.
  - `answer`: production-side deixis: a speaker names the holder as they would
    say it (`me` / `you` / a name; chance 1/4), with a body-hidden null variant
    for contextual calibration.
  - `adjacency`: a vocative-free "you" at three speakers, resolved by request →
    grant adjacency (3 names, chance 1/3).
  - `codevarmut`: Python list aliasing with mutate / rebind / copy traps; 5
    literals per item (chance 1/5); only alias-graph tracking clears all three
    traps.

- **The binding suite** (`kotodama.eval.binding`): one runner for every battery
  above and below. It loads a model once (a kotodama checkpoint or any Hugging Face
  model), scores each requested battery in-process with the same scorer as
  `/logprobs`, and writes resumable rows. Prompt files are pinned by sha256, so a
  recorded number always names its exact strings. Beyond the bindprobes families
  it carries:
  - stage gauges `joinlite-v1` (static I / you and a speaker-swap counterfactual)
    and `swapdyn-v1` (the swap on transfer dialogues). A shift above +.02 at the
    low end of its 95% CI binds; one below −.02 at the high end inverts.
  - `stage5-decomp-v1`: dynamic "you" split into its prerequisites (third-person
    possession transfer, deictic transfer speech acts, query-time deixis), each
    a prompt variant over the same episodes.
  - `capacity-v1`: binding capacity after arXiv 2609.30634. K "Name: task" lines
    (K = 1…64), with or without ~256 tokens of filler, then one query. K50 is
    where recall falls halfway to chance; `capacity` fits K50 ∝ N^α over
    baselines scored on this same harness and places any model on the law as a
    parameter-equivalent size.
  `report` prints one scorecard row per model.

Rider generation is byte-identical per seed and name pool, so banked grids stay
comparable. Shortcut controls (recency, last speaker, last literal) are
scheduled at the chance rate and audited.

## Layout

```text
src/kotodama/
  model/       llama.py (model, AttnRes, looped depth), flash_attn_res/ (Triton kernels),
               decode_engine, capture, init_schemes, routing_read
  nca/         NCA trajectory generator, dyadic variant, v3 dataset writer
  training/    train.py, Muon, checkpointing, inloop (in-loop evals)
  data/        uint16 token-stream dataset
  eval/        model loading, lm-eval adapter, uncheatable, induction, nca_gates,
               riders/ (rider grid), bindprobes/ (binding probes), binding/ (the suite: one runner + scorecard)
  monitoring/  per-layer geometric telemetry, wandb callback
  serve/       server, gateway, chatml, laws, steering, chat client
  presets.py   shapes, AttnRes boundaries, tokenizer
  ckpt.py      checkpoint load / strip / info / registry
configs/       training runs (+ base/), gateway example, checkpoint registry
scripts/       eval/ (lm-eval, boundary fitting, routing, telemetry readers),
               benchmark/ (serving performance and parity batteries),
               utils/ (kernel parity, data, smoke test)
tools/         launchers (all via _env.sh)
tests/         CPU test suite
```

## Tests

```bash
.venv/bin/python -m pytest
```

The suite runs on CPU; GPU-only tests skip without CUDA. Kernel parity checks
(`scripts/utils/*_parity.py`) and serving batteries
(`scripts/benchmark/check_*.py`) run standalone on a GPU.
