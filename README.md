# Kotodama

From-scratch transformer language model. NCA pre-pretraining + Block Attention Residuals + Muon optimizer.

Research model targeting rich geometric structure formation and conversational quality at the 3B–8B scale.

## Architecture

Two model scales sharing the same training infrastructure, tokenizer, and methodology.

### 3B (primary)

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

[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) BPE tokenizer, 49,152 tokens. Chosen for favorable gradient flow properties at our hidden dimensions — the Godey gradient bottleneck is ~94% at 3072/49K vs ~98% at 3072/128K with larger vocabularies.

### Block Attention Residuals

[Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals/) (Moonshot AI, 2026) add learned depth-wise routing across transformer blocks. Layers are grouped into blocks; at each block boundary, accumulated activations are committed to a buffer. Each layer routes its residual stream across all committed blocks via masked softmax with learned pseudo-queries.

Configuration is data-driven: block boundaries are derived from NCA pre-pretraining profiles. For throughput benchmarking, uniform blocks of 4 layers are used (7 blocks on 28 layers, 8 blocks on 32 layers). Overhead is <4% wall-clock, <0.05% additional parameters.

## Training

### Optimizer

Hybrid Muon + AdamW:
- **Muon** (94.9% of params): 2D weight matrices. Newton-Schulz orthogonalization with Gram-NS coefficients (Dao-AILab). LR 0.02, momentum 0.95, Nesterov, weight decay 0.01. NS iterations run in fp16 for precision.
- **AdamW** (5.1% of params): Embeddings, norms, LM head. LR 6e-4, betas (0.9, 0.95), weight decay 0.1.

### Schedule

Warmup-Stable-Decay (WSD): 2,000 step warmup, 88% stable, sqrt decay to 0 over final 10%. Batch size ramp from 512K to 2M tokens over first 5%.

### Precision

- Forward/backward: FP8 via torchao Float8Linear (bf16 fallback)
- Muon Newton-Schulz: fp16
- Optimizer states: fp32 (AdamW), bf16 (Muon momentum)
- Accumulation: fp32

### NCA Pre-pretraining

Neural Cellular Automata pre-pretraining bootstraps attention circuit structure before language training. The model trains on NCA trajectory prediction (~750M tokens), then transitions to language with embedding reinitialization. NCA-trained attention heads develop structured routing patterns that persist through language training and prevent BOS-sink attention collapse at deep layers.

Reference implementation: [nca-pre-pretraining](https://github.com/MoonshotAI/Attention-Residuals/)

### Infrastructure

- **Hardware**: 8× NVIDIA B200 (183 GB HBM3e each)
- **Parallelism**: DDP (Muon requires full weight matrices, incompatible with FSDP)
- **Compile**: torch.compile with Liger fused kernels (RMSNorm, SwiGLU, FusedLinearCrossEntropy)
- **Attention**: Flash Attention 2 via PyTorch SDPA auto-dispatch
- **Checkpointing**: Async CPU clone + background SHM write + zstd compression

### Throughput (8×B200, 4096 context, FP8)

| Model | tok/s | 80B tokens | 150B tokens | 200B tokens |
|-------|-------|-----------|------------|------------|
| 3B | 240K | 3.9 days | 7.2 days | 9.6 days |
| 8B | 121K | 7.6 days | 14.3 days | 19.1 days |

## Data

~246B tokens pre-dedup (~210B estimated post-dedup) across 14 sources. Single-epoch training — no data is seen twice.

| Source | Est. Tokens | Domain |
|--------|------------|--------|
| The Stack v1 | ~126B | Code (130 languages, license-filtered) |
| peS2o | ~50B | Academic papers (Semantic Scholar, 38.8M papers) |
| OpenCoder Reasoning | ~21B | Code reasoning (R1 + QwQ critiques + verified solutions) |
| Pile of Law | ~14B | Legal (US court opinions + congressional hearings) |
| StackExchange | ~10B | Q&A (22 high-value sites, Mar 2025 dump) |
| OpenWebMath | ~9B | Math (web pages from Common Crawl) |
| FineMath | ~7B | Math (quality-scored, 4+ threshold) |
| Project Gutenberg | ~5B | Books (71K public domain, via common-pile) |
| Wikipedia | ~4B | Encyclopedia (6.4M English articles, Nov 2023) |
| SmolTalk | ~1B | Synthetic conversation (Llama 405B multi-turn) |
| WildChat | ~0.3B | Real user-LLM conversations (aggressively filtered) |
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

Cleaning → cross-source MinHash LSH dedup → tokenization → phase assembly. Code in `~/projects/kotodama/curation/`.

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

- **Muon LR 0.02**: Pareto optimum (matches AdamW loss, 2–4× stable rank)
- **AttnRes DD-v1**: 4× gradient uniformity, BOS-sink prevention, full depth utilization
- **NCA + AttnRes**: Sub-additive on loss (+0.008 nats) but preserves geometry everywhere
- **BOS-sink**: Baseline develops 89–90% BOS attention at deep layers by 6B tokens; DD-v1 prevents this entirely

Full report: `PROXY-REPORT.md`
