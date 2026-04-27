# Kotodama

From-scratch transformer language model. NCA pre-pretraining + Block Attention Residuals + Muon optimizer.

Research model targeting rich geometric structure formation and conversational quality at the 3B–8B scale. Full writeup at [aetherawi.red/kotodama](https://aetherawi.red/kotodama).

## Architecture

Two model scales sharing the same training infrastructure, tokenizer, and methodology.

### 3B

| Parameter | Value |
|-----------|-------|
| Parameters | 2.97B |
| Hidden size | 3072 |
| Layers | 28 |
| Query heads | 24 |
| KV heads | 8 (GQA 3:1) |
| Head dim | 128 |
| FFN intermediate | 8192 (SwiGLU, ~2.67x) |
| Vocab | 49,152 |
| Max context | 4,096 |
| Tied embeddings | Yes |
| Positional encoding | RoPE (θ=500,000) |
| Normalization | RMSNorm + QK-norm |
| Stability | z-loss (1e-5) |
| Bias | None |

### 8B

| Parameter | Value |
|-----------|-------|
| Parameters | 7.18B |
| Hidden size | 4096 |
| Layers | 32 |
| Query heads | 32 |
| KV heads | 8 (GQA 4:1) |
| Head dim | 128 |
| FFN intermediate | 14,336 (SwiGLU, 3.5x) |
| Vocab | 49,152 |
| Max context | 4,096 |

All other flags (RoPE θ, RMSNorm, QK-norm, z-loss, tied embeddings, no bias) are shared with the 3B.

### Tokenizer

[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) BPE tokenizer, 49,152 tokens. Chosen for favorable gradient flow properties at our hidden dimensions — the embedding layer gradient bottleneck ([Godey et al., 2024](https://arxiv.org/abs/2402.13571)) is ~94% at d_model=3072/vocab=49K vs ~98% at 3072/128K with larger vocabularies. The smaller vocabulary reduces the fraction of gradient signal lost in the embedding projection.

### Architecture details

**QK-norm**: RMSNorm applied to query and key projections after linear transformation, before RoPE. Stabilizes attention logit magnitudes and enables higher learning rates without attention entropy collapse.

**z-loss**: Auxiliary loss term (weight 1e-5) on the log-sum-exp of output logits. Prevents logit explosion during training by penalizing large logit magnitudes.

## Block Attention Residuals

[Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals/) (Moonshot AI, 2026) generalize residual connections from depth-wise linear accumulation to depth-wise softmax attention. In a standard transformer, the residual stream at layer l is the unweighted sum of all prior layer outputs: `h_l = Σ v_i`. AttnRes replaces this with a learned weighted combination: `h_l = Σ α_{i→l} · v_i`, where α is computed via softmax over learned per-layer pseudo-queries.

### Mechanism

Layers are divided into blocks (e.g., 28 layers into 7 blocks of 4). The forward pass maintains two buffers:

- **committed**: frozen summary activations from completed blocks, shape `(num_completed_blocks, B, T, D)`
- **partial**: accumulation buffer for the current in-progress block, shape `(B, T, D)`

At each layer, routing proceeds as:

1. **Build sources**: concatenate all committed block summaries + current partial buffer
2. **Normalize**: apply a per-layer learned RMSNorm to each source to prevent magnitude differences from biasing attention
3. **Compute logits**: `logits = einsum("d, s b t d -> s b t", w_l, K)` where `w_l` is a learned pseudo-query vector `(D,)` for this layer, initialized to zero (so initial routing is uniform)
4. **Route**: `weights = softmax(logits, dim=0)` — each token position independently decides how much to draw from each depth source
5. **Aggregate**: `output = einsum("s b t, s b t d -> b t d", weights, sources)` — this replaces the standard residual as input to attention or MLP

At block boundaries, the partial buffer is frozen into committed and a new partial begins. Two routing points per layer (pre-attention and pre-MLP) plus a final aggregation at model output.

### Why it matters

AttnRes gives each layer an explicit handle to selectively retrieve earlier representations rather than receiving everything diluted through additive residual accumulation. This prevents **BOS-sink collapse** — a failure mode where deep-layer attention concentrates 89-90% of mass on the first (BOS) token, effectively bypassing the residual stream. AttnRes with data-driven block boundaries (DD-v1) halves BOS concentration at layer 14 and maintains this through billions of subsequent tokens.

The overhead is <4% wall-clock training time and ~58K additional parameters (0.05% of 3B model). Block boundaries are derived from NCA pre-pretraining profiles — the NCA phase reveals which layer ranges develop coherent computational circuits, and those circuit boundaries become AttnRes block boundaries. For throughput benchmarking, uniform blocks of 4 layers are used.

## NCA Pre-pretraining

Neural Cellular Automata pre-pretraining exposes the model to discrete cellular automata dynamics before language training. The model processes 64×64 grids with 8 state channels, tokenized as 2×2 patches (~10,000-token vocabulary), over 64–256 step trajectories. The training objective is standard next-token prediction on these spatial sequences — no language is involved.

Rules are selected for Class IV (edge-of-chaos) dynamics via gzip compression scoring. Class IV rules produce long-range information propagation through local interactions, multi-scale hierarchical structure, and context-sensitive state transitions — properties that share structural similarity with natural language without any semantic content.

### What it bootstraps

NCA forces **computational scaffolding before content**. After 300–850M tokens of NCA training (<0.5% of total training budget), attention heads develop dramatic specialization:

- Head entropy standard deviation increases **14×** (from 0.08 to 1.14) — some heads sharpen to entropy below 2.0, attending to a handful of positions rather than spreading mass uniformly
- Output projection participation ratio increases **18×** (172 vs 9–10 effective dimensions) — standard training collapses o_proj to near-rank-1, while NCA preserves the full mixing matrix
- Layer 14 BOS attention concentration drops from 86% to 75%

These attention circuits — structured routing patterns with form but no semantic content — persist through the transition to language training and provide the geometric substrate that AttnRes needs to function.

### NCA → Language transition

After NCA training, the model transitions to language by reinitializing the embedding and LM head layers to the language vocabulary (49,152 tokens) while keeping all transformer weights (attention, MLP, norms). AttnRes routing weights are co-trained during NCA rather than initialized fresh at language transition — this co-training is critical, as fresh AttnRes without NCA-trained structure actually degrades representation quality.

Reference implementation: [nca-pre-pretraining](https://github.com/danihyunlee/nca-pre-pretraining)

## Training

### Optimizer

Hybrid [Muon](https://github.com/KellerJordan/Muon) + AdamW:
- **Muon** (94.9% of params): All 2D weight matrices (attention projections, MLP layers). Muon orthogonalizes the momentum matrix via 5 iterations of Newton-Schulz before each parameter update, producing spectral-norm descent rather than Frobenius-norm descent. Uses [Gram-NS coefficients](https://github.com/Dao-AILab/gram-newton-schulz) (per-iteration optimized coefficients for faster convergence). LR 0.02, momentum 0.95, Nesterov, weight decay 0.01.
- **AdamW** (5.1% of params): 1D parameters — embeddings, norms, LM head, AttnRes pseudo-queries. LR 6e-4, betas (0.9, 0.95), weight decay 0.1.

Muon requires full (unsharded) weight matrices for Newton-Schulz orthogonalization, which makes it incompatible with FSDP. Training uses DDP instead.

### Schedule

Warmup-Stable-Decay (WSD): 2,000 step warmup, 88% of training at stable LR, sqrt decay to 0 over the final 10% of steps. Batch size ramp from 512K to 2M tokens over the first 5% of training steps.

### Precision

- **Forward/backward matmuls**: FP8 (E4M3) via [torchao](https://github.com/pytorch/ao) Float8Linear with dynamic per-tensor scaling. Non-matmul operations (norms, activations, AttnRes routing) remain in bf16.
- **Muon Newton-Schulz**: fp16. NS iterates near unit norm where fp16's 10 mantissa bits outperform bf16's 7 — the extra range of bf16 is wasted when magnitudes are controlled ([Dao-AILab analysis](https://github.com/Dao-AILab/gram-newton-schulz)).
- **Optimizer states**: fp32 (AdamW), bf16 (Muon momentum buffer)
- **Gradient accumulation**: fp32

### Infrastructure

- **Hardware**: 8× NVIDIA B200 (183 GB HBM3e each)
- **Parallelism**: DDP with gradient-as-bucket-view
- **Compile**: torch.compile (whole model). [Liger](https://github.com/linkedin/Liger-Kernel) fused Triton kernels for RMSNorm, SwiGLU inner multiply, and FusedLinearCrossEntropy (fuses lm_head + CE + z-loss, skips logit materialization)
- **Attention**: Flash Attention 2 via PyTorch scaled dot-product attention auto-dispatch
- **Checkpointing**: Async — synchronous CPU state dict clone (~0.5s), then background thread writes to /dev/shm + zstd compression + atomic copy to persistent storage

### Throughput (8×B200, 4096 context, FP8)

| Model | tok/s | MFU | 80B tokens | 150B tokens | 200B tokens |
|-------|-------|-----|-----------|------------|------------|
| 3B | 240K | ~23% | 3.9 days | 7.2 days | 9.6 days |
| 8B | 121K | ~22% | 7.6 days | 14.3 days | 19.1 days |

MFU (model FLOP utilization) is computed against theoretical peak bf16 FLOPS (FP8 tensor core peak is higher, so effective hardware utilization is better than the MFU number suggests). The primary bottleneck is memory bandwidth, not compute — the models are small enough per GPU that data movement dominates arithmetic. Profile breakdown: Flash Attention fwd+bwd accounts for 22% of CUDA time, AttnRes routing softmax ~8%, DDP all-reduce ~2%.

## Data

~246B tokens pre-dedup (~210B estimated post-dedup) across 14 sources. Single-epoch training — no data is seen twice. Source weights for phase assembly are not yet finalized; the current corpus is heavily skewed toward code (~51%) and academic text (~20%), with conversational data at <1%. Phase assembly will subsample overweight sources and the planned additions below will improve register diversity.

| Source | Est. Tokens | Domain |
|--------|------------|--------|
| The Stack v1 | ~126B | Code (130 languages, license-filtered) |
| peS2o | ~50B | Academic papers (Semantic Scholar, 38.8M papers) |
| OpenCoder Reasoning | ~21B | Code reasoning (R1 + QwQ critiques + verified solutions) |
| Pile of Law | ~14B | Legal (US court opinions + congressional hearings) |
| StackExchange | ~10B | Q&A (22 high-value sites, Mar 2025 dump) |
| OpenWebMath | ~9B | Math (web pages from Common Crawl) |
| FineMath | ~7B | Math (quality-scored, 4+ on 0-5 scale) |
| Project Gutenberg | ~5B | Books (71K public domain, via common-pile) |
| Wikipedia | ~4B | Encyclopedia (6.4M English articles, Nov 2023) |
| SmolTalk | ~1B | Synthetic conversation (Llama 405B multi-turn) |
| WildChat | ~0.3B | Real user-LLM conversations (filtered for quality, language, safety) |
| SODA | ~0.2B | Synthetic social dialogue |
| Enron | ~0.2B | Corporate email correspondence |
| OASST2 | ~0.02B | Human-written multi-turn conversations |

### Planned additions

| Source | Est. Tokens | Domain |
|--------|------------|--------|
| Common Pile books (pre-1929, LoC, BHL, DOAB) | ~35B | Public domain books and open access texts |
| OpenSubtitles | ~2B | Movie/TV dialogue (conversational register) |
| USPTO Patents | ~30B | Technical/legal prose |
| Common Pile (caselaw, hansard, youtube, usgpo) | ~40B | Legal, parliamentary, spoken, government |

### Data pipeline

Cleaning (PII scrub, unicode normalization, length filtering) → cross-source MinHash LSH dedup (14 bands × 8 rows) → tokenization to uint16 binary → phase assembly with source weight balancing. Code in the `curation/` directory (separate from pretraining).

## Project structure

```
configs/           YAML configs (model, training, analysis, benchmarks)
src/
  model/llama.py   Model + AttnRes + Liger kernels + FA2
  training/        DDP training loop, Muon+AdamW, async checkpoints
  eval/            Shared analysis modules (model loading, generation, metrics)
  monitoring/      Geometric health monitoring (RankMe, WeightWatcher, Anamnesis)
  data/            Dataset classes (tokenized binary + random)
  nca/             NCA trajectory generation
scripts/
  analysis/        6-track analysis pipeline (see scripts/README.md)
  utils/           Tokenization, profiling, smoke tests
  legacy/          Completed proxy-phase scripts
tools/             Training runners, benchmarks, profiling
docs/              Specs and pipeline documentation
```

## Proxy validation

Completed 108M-parameter proxy phase: 5-run optimizer sweep, NCA/AttnRes validation matrix, 6B token training runs. Key findings:

- **Muon LR 0.02**: Pareto optimum across the 0.01–0.04 sweep (matches AdamW loss, 2–4× higher stable rank)
- **AttnRes DD-v1**: 4× gradient uniformity, BOS-sink prevention, full depth utilization across all 28 layers
- **NCA + AttnRes synergy**: Combined loss is sub-additive — the pair adds only +0.008 nats over the sum of individual effects. NCA provides the geometric substrate (head specialization, attention structure); AttnRes provides the dimensional headroom for that structure to express itself. Neither works as well alone.
- **BOS-sink**: Baseline develops 89–90% BOS attention at deep layers by 6B tokens. DD-v1 with co-trained routing reduces this to ~52% and maintains it through subsequent training.

Full report: `PROXY-REPORT.md`
