# Tensor Parallelism Investigation Handoff

## Goal

Implement TP (tensor parallelism) for the kotodama training loop to improve MFU from ~23% to ~40-50%. Current throughput on 8xB200 DDP: 3B @ 240K tok/s, 8B @ 121K tok/s. Memory-bandwidth bound. TP enables larger effective matmul tiles and higher arithmetic intensity.

## What makes this non-trivial

Two custom components interact with TP in non-obvious ways:

1. **Muon optimizer**: Newton-Schulz orthogonalization requires the full (unsharded) weight matrix. Under TP, weight matrices are sharded across GPUs. Solution exists: all-gather gradient → run NS on full matrix → slice result back. Multiple reference implementations available.

2. **Block Attention Residuals (AttnRes)**: Learned depth-wise routing with per-layer pseudo-query vectors and committed activation buffers. Pseudo-queries are small (d_model each) and should be replicated. Committed buffers flow through hidden_dim which is the TP-sharded dimension — need to verify whether routing softmax works on the local shard or needs a gather.

## Reference implementations to clone and study

```bash
# Clone to /tmp/ for investigation — don't install, just read
cd /tmp

# Megatron-LM: TensorParallelMuon + general TP infrastructure
git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git

# Key files:
#   megatron/core/optimizer/emerging_optimizers.py — TensorParallelMuon, newton_schulz_tp()
#   megatron/core/tensor_parallel/layers.py — ColumnParallelLinear, RowParallelLinear
#   megatron/core/tensor_parallel/mappings.py — all-reduce / all-gather primitives

# torchtitan: PyTorch-native Muon+TP (closest to our codebase style)
git clone --depth 1 https://github.com/pytorch/torchtitan.git

# Key files:
#   Check PR #1630 for DiSCO/distributed Scion — gather_tp_shard(), update_bucket_params()
#   torchtitan/parallelisms/ — TP application patterns
#   torchtitan/models/llama/ — how TP is applied to a Llama model

# Community Megatron fork with AttnRes + TP
git clone --depth 1 https://github.com/Smu-Tan/Megatron-LM.git megatron-attnres

# Key: search for AttnRes, attention_residual, or similar in the codebase
# This has Full AttnRes (not Block) working with TP and sequence parallelism
```

## Our codebase (source of truth)

All code is at `~/projects/kotodama/pretraining/` (local) and `~/luxi-files/kotodama/` (node1).

- **Model**: `src/model/llama.py` — LuxiaBaseModel with AttnRes. AttnRes mechanism starts at line ~480. Pseudo-queries are `attn_res_query` and `mlp_res_query` per TransformerBlock, plus `final_res_query` on the model. Routing is in `_route_static()` (line ~545). Committed/partial buffer management is in the forward pass of LuxiaBaseModel.
- **Muon**: `src/training/muon.py` — `newton_schulz_orthogonalize()` and `_batched_newton_schulz()`. The step function batches same-shape matrices for efficiency. NS runs in fp16.
- **Training loop**: `src/training/train.py` — DDP wrapping at line ~297, optimizer construction at ~302, training loop at ~519.
- **Model config**: `src/model/llama.py:LuxiaModelConfig` dataclass and `MODEL_CONFIGS` dict in train.py.

## What needs to change (estimated)

### 1. TP for model layers (~100-150 lines)

Use `torch.distributed.tensor.parallel` APIs:
- `ColwiseParallel`: Q, K, V, gate, up projections (shard output dim)
- `RowwiseParallel`: O, down projections (shard input dim, all-reduce output)
- AttnRes pseudo-queries: leave as regular replicated parameters (they're 1D vectors, not worth sharding)
- AttnRes committed buffers: follow the residual stream, should flow through TP all-reduce naturally. **VERIFY**: does `_route_static()` work correctly when hidden_dim is sharded? The einsum with the pseudo-query might need the full hidden_dim.

### 2. Muon gather-before-NS (~30 lines)

In `Muon.step()`, before the NS call:
```python
# Pseudocode — adapt from torchtitan PR #1630
if tp_group is not None:
    full_grad = all_gather(local_grad, tp_group, shard_dim)
    full_orth = newton_schulz_orthogonalize(full_grad, ...)
    local_orth = full_orth.narrow(shard_dim, rank * shard_size, shard_size)
else:
    local_orth = newton_schulz_orthogonalize(local_grad, ...)
```

The batched NS path also needs this — gather per-shape groups, NS, slice back.

### 3. Process group setup (~50 lines)

Initialize TP process groups alongside DDP. On a single 8-GPU node, options:
- TP=8, no DDP (full TP within node)
- TP=4, DDP=2 (hybrid — probably not worth the complexity for single node)
- TP=8 is simplest and most impactful

Apply TP to model first, then wrap with DDP if using hybrid.

## How to test

### Smoke test (local or node1)

```bash
# Minimal: 2 GPUs, smoke model, verify it runs
torchrun --nproc_per_node=2 -m src.training.train \
    --model_size smoke --random_data --total_steps 20 \
    --compile --attn_res --attn_res_n_blocks 2 \
    --micro_batch_size 2 --sequence_length 512

# Then compare: same config, 2 GPUs with TP=2 (once implemented)
# Loss curves should be numerically similar (not identical due to TP all-reduce order)
```

### Throughput validation (node1, 8xB200)

Run the existing canary configs with TP enabled and compare tok/s:
- `configs/canary-3b-fp8.yaml` — baseline: 240K tok/s
- `configs/canary-8b-fp8.yaml` — baseline: 121K tok/s

Target: 30-40% improvement. If MFU reaches 35%+ it's working.

### Correctness validation

- Compare loss curves (first 100 steps) between DDP and TP runs on same seed
- Check that AttnRes routing weights are learning (not stuck at uniform)
- Verify gradient norms are in the same range as DDP baseline
- Run geometric monitoring (`--geo_monitor`) to check stable rank / RankMe aren't degraded

## Node access

- **node1**: `ssh node1`, user `athuser`, working dir `~/luxi-files/kotodama/`
- **Shared venv**: `~/luxi-files/.venv-shared/` — use `uv` for package management, NEVER `pip`
- **8x B200 GPUs**, 183 GB HBM3e each, NVLink interconnect
- **IMPORTANT**: This is a shared machine. See global CLAUDE.md for safety constraints. Never install packages outside the activated venv.
- Sync code from local: `sync-kotodama` (defined in ~/.bashrc) or manual tar+ssh

## Memory files with context

- `memory/canary_throughput.md` — current throughput numbers and profile breakdown
- `memory/tensor_parallelism_research.md` — full research on Muon+TP+AttnRes compatibility, reference repos, open questions
- `memory/optimization_pass.md` — prior optimization work (Liger, FA2/FA4, compile benchmarks)
- `memory/pretraining.md` — full project state

## Key constraints

- Muon NS must see full weight matrix (all-gather required under TP)
- AttnRes routing is incompatible with per-layer activation checkpointing (routing shares state across layers)
- FA4 (CuTeDSL) is incompatible with torch.compile — stick with FA2/SDPA
- FP8 via torchao Float8Linear should compose with TP (Float8Linear wraps nn.Linear, TP shards nn.Linear — order matters, apply TP first then FP8 conversion)
