# luxia-base 108M Proxy Matrix — Full Spec Sheet

## 1. Architecture

| | Value |
|---|---|
| **Type** | Decoder-only transformer (Llama-family) |
| **Parameters** | ~108M |
| **d_model** | 512 |
| **Layers** | 28 (same depth as target 3B) |
| **Query heads** | 4 |
| **KV heads** | 2 (GQA ratio 2:1) |
| **Head dim** | 128 |
| **FFN** | SwiGLU, intermediate=1408 (~2.75x) |
| **Norm** | Pre-RMSNorm (eps=1e-5) + QK-norm |
| **Position** | RoPE, theta=500,000, max 4096 |
| **Vocab** | 49,152 (SmolLM2 tokenizer) |
| **Embeddings** | Tied input/output |
| **Bias** | None |
| **z-loss** | 1e-5 |
| **Init** | Normal(0, 0.02); residual projections scaled by 1/sqrt(2*28) |

**Design rationale:** Depth held at 28 so layer specialization and gradient flow phenomena transfer to the 3B. Width scaled down (512 vs 3072). Same GQA, same RoPE theta, same QK-norm — minimizes proxy-to-full transfer uncertainty.

**Differences from stock Llama 3.2-3B:** QK-norm added (training stability, enables higher LR). z-loss added (prevents logit explosion). Vocab 49K instead of 128K (reduces Godey gradient bottleneck: ~94% destruction at 3072/49K vs ~98% at 3072/128K).

---

## 2. Matrix Design (2x2: NCA x LR)

| Run | Muon LR | NCA | Comparison |
|---|---|---|---|
| **P3** | 0.02 | No | Baseline for NCA-002 |
| **NCA-Muon-002** | 0.02 | 300M NCA tokens | Tests NCA effect at optimal LR |
| **P4** | 0.03 | No | Baseline for NCA-003 |
| **NCA-Muon-003** | 0.03 | 300M NCA tokens | Tests NCA effect at secondary LR |

All 4 runs trained 6B language tokens on FineWeb-Edu, 45,776 steps, identical hyperparameters except LR and initialization.

The prior **5-run LR sweep** (P1-P5) established P3 as Pareto-optimal and P4 as secondary:

| Run | Optimizer | LR | Purpose |
|---|---|---|---|
| P1 | AdamW | 8e-4 | Baseline (no Muon) |
| P2 | Muon | 0.01 | Conservative |
| P3 | Muon | 0.02 | **Primary (Pareto winner)** |
| P4 | Muon | 0.03 | Secondary |
| P5 | Muon | 0.04 | Aggressive |

---

## 3. Optimizer

Hybrid Muon + AdamW. Muon handles 2D weight matrices (~77% of params); AdamW handles embeddings, norms, QK-norm params.

### Muon (Q/K/V/O projections, FFN gate/up/down)

| | Value |
|---|---|
| LR | 0.02 (P3/NCA-002) or 0.03 (P4/NCA-003) |
| Momentum | 0.95, Nesterov |
| Weight decay | 0.01 (decoupled) |
| Newton-Schulz iters | 5 |
| NS precision | BF16 |
| NS coefficients | a=3.4445, b=-4.7750, c=2.0315 |
| Tall-matrix optimization | Transpose if rows > cols |
| Scaling | `max(1, sqrt(rows/cols))` per matrix |
| Batched NS | Matrices of same shape batched together |

### AdamW (embeddings, norms, QK-norm)

| | Value |
|---|---|
| LR | 6e-4 |
| Betas | (0.9, 0.95) |
| Weight decay | 0.1 |

**P1 baseline:** Pure AdamW, lr=8e-4, betas=(0.9, 0.95), wd=0.1, all params.

---

## 4. Schedule (WSD)

| Phase | Steps | % | Description |
|---|---|---|---|
| Warmup | 0 → 2,000 | 4.4% | Linear ramp to base LR |
| Stable | 2,000 → 41,198 | 85.6% | Constant LR |
| Decay | 41,198 → 45,776 | 10% | sqrt decay to 0 |

Decay schedule validated by separate sweep (10%, 12.5%, 15%, 20%, 25%): longer decay uniformly hurts (+0.002 loss / +2.5% extra). Geometry is decay-invariant across all variants.

---

## 5. Training Configuration

| | Value |
|---|---|
| Total tokens | 6B (~55.7 tok/param) |
| Total steps | 45,776 |
| Sequence length | 2048 |
| Global batch | 131,072 tokens (128K) |
| Micro-batch | 4 seqs/GPU |
| Precision | BF16 (torch.autocast) |
| Gradient clip | 1.0 |
| torch.compile | Yes |
| Activation checkpointing | Yes |
| Parallelism | DDP, 8x B200 |
| Checkpoint interval | Every 2,000 steps |
| Seed | 42 |

**Why DDP over FSDP2:** Muon's Newton-Schulz orthogonalization requires global matrix operations (X @ X.T, Frobenius norm) that cannot operate correctly on FSDP2-sharded tensors without explicit all-gather at every iteration (5 NS iters x 3 matmuls = 15 collectives per parameter per step). DDP all-reduces gradients so Muon operates on the full replicated weight matrices locally.

---

## 6. Data

| | Value |
|---|---|
| **Source** | FineWeb-Edu |
| **Training** | `fineweb_edu_6b.bin` — 6B tokens, flat uint16 memmap |
| **Eval** | `fineweb_edu_eval_5m.bin` — tail 5M tokens, held out |
| **Tokenizer** | SmolLM2 (`HuggingFaceTB/SmolLM2-135M`), 49,152 vocab, byte-fallback |
| **Packing** | No document boundaries at proxy scale |

---

## 7. NCA Pre-Pre-Training (NCA-002 and NCA-003 only)

Neural Cellular Automata trajectories used to bootstrap attention circuits before language training. Adapted from Han et al. (arxiv.org/html/2603.10055v1).

### NCA Data Generation

| | Value |
|---|---|
| NCA tokens | 300M |
| Grid | 32x32 |
| Cell states | 10 (discrete) |
| Channels | 4 (serialized as sequential frames) |
| Patch tokenization | 2x2 → vocab of 10,002 (10^4 + START + END) |
| Tokens/frame | 258 (256 patches + START + END) |
| Tokens/timestep | 1,032 (258 x 4 channels) |
| Trajectory length | 128 steps (after 10-step burn-in) |
| Rules | 5,000 randomly sampled, complexity-filtered |
| Sims/rule | 8 |

### Rule Architecture (Mixed Complexity)

| | Value |
|---|---|
| Kernel sizes | 3x3, 5x5 (random choice per rule) |
| Hidden layers | 1, 2, or 3 (random) |
| Hidden dim | 16, 32, or 48 (random) |
| Identity bias | Uniform(0.5, 2.0) |
| Temperature | Uniform(0.3, 1.0) |
| Complexity filter | gzip ratio in [0.4, 0.7] (Class IV edge-of-chaos) |
| Cross-channel | Shared perception convolution over all channels |

### NCA Design Brief

The NCA trajectories are designed to bootstrap:
- **Multi-scale hierarchical structure** (Class IV / edge-of-chaos dynamics)
- **Context-sensitive transitions** (same local pattern, different meaning in different context)
- **Multi-channel parallel tracking** (multiple simultaneous information threads)
- **Long-range propagation through local interactions** (information transforms coherently over distance)
- **Natural boundary discovery** (segmentation, cutting at joints)

### NCA → Language Transition

Load NCA checkpoint → reinitialize embeddings to 49K language vocab → keep MLPs and attention weights (embed-only reinit) → reset optimizer states → train 6B language tokens identically to baselines.

---

## 8. Geometric Monitoring

| Metric | Interval |
|---|---|
| Tier 1 (RankMe, stable rank, anisotropy, attn entropy) | Every 200 steps |
| Tier 2 (WeightWatcher alpha per layer) | Every 2,000 steps |
| Per-layer gradient norms (attn/MLP split) | Every 50 steps |
| Logging targets | Wandb (`luxia-base` project) + Heimdall metrics API |

---

## 9. Results

### LR Sweep (P1-P5, all at 45,776 steps)

| Run | Optimizer | LR | Final Loss | Final PPL | RankMe |
|---|---|---|---|---|---|
| P2 | Muon | 0.01 | **2.914** | 18.6 | 431 |
| P3 | Muon | 0.02 | 2.934 | 18.8 | 434 |
| P1 | AdamW | 8e-4 | 2.939 | 18.9 | 430 |
| P4 | Muon | 0.03 | 2.946 | 19.1 | 435 |
| P5 | Muon | 0.04 | 2.949 | 19.1 | 434 |

### 2x2 NCA Matrix Eval

| Run | Eval Loss | Eval PPL | NCA Delta |
|---|---|---|---|
| P3 (Muon 0.02, no NCA) | 2.9245 | 18.62 | — |
| NCA-Muon-002 (Muon 0.02, NCA) | 2.9309 | 18.74 | +0.0064 / +0.12 PPL |
| P4 (Muon 0.03, no NCA) | 2.9381 | 18.88 | — |
| NCA-Muon-003 (Muon 0.03, NCA) | 2.9417 | 18.95 | +0.0036 / +0.07 PPL |

NCA shows a small loss penalty at both LRs. The primary analysis target is the geometric trajectory, not loss.

### Key Geometric Findings

- **Muon vs AdamW is the dominant effect.** 2-4x stable rank advantage, early anisotropy spike, preserved L0 intrinsic dimensionality.
- **WW alpha reframe.** AdamW has "healthier" alpha (5.15, 39.6% healthy) vs Muon (7.2-7.7, 17-18%). But this measures spectral tail shape, not representational capacity. Muon's profile = committed structure (high rank + steep tails). AdamW's = hedging (low rank + fat tails).
- **NCA builds attention structure, not MLP structure.** Attention projections reach near-healthy WW alpha (5-7), MLPs stay unhealthy (9-15). L0 attention becomes sharply specialized with heterogeneous heads.
- **Decay schedule is settled at 10%.** Geometry is decay-invariant across all sweep variants.
- **P3 qualitative text is most "interesting."** Bold associations, narrative coherence, willingness to commit — maps directly to its geometric profile.
