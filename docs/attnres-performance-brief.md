# Block Attention Residuals: Performance Analysis Brief

## Context

We are training a 3B parameter transformer (28 layers, D=3072) with Block Attention Residuals (AttnRes), a depth-wise routing mechanism from Moonshot/Kimi (2026). At proxy scale (108M params, D=512), the mechanism adds 54% overhead to training throughput. We need to understand whether this is fixable, expected, or an artifact of scale — and what optimizations are available.

## Architecture

Standard Llama-family transformer: 28 layers, pre-norm (RMSNorm), SwiGLU FFN, GQA (query heads > KV heads), RoPE, tied embeddings. Trained with Muon+AdamW hybrid optimizer, DDP across 8×B200 GPUs, torch.compile, bf16 autocast, activation checkpointing.

## What Block Attention Residuals Does

The standard transformer residual stream is `x = x + sublayer(norm(x))` — a single additive skip connection per sublayer. AttnRes replaces this with a **block-based depth routing** system.

### Mechanism

The 28 layers are divided into N blocks (we use N=4, so blocks of 7 layers each, with boundaries at layers 0, 7, 14, 21).

**State maintained during the forward pass:**
- `committed`: tensor of shape `(num_completed_blocks, B, T, D)` — frozen summaries of completed blocks
- `partial`: tensor of shape `(B, T, D)` — accumulation buffer for the current in-progress block

**At each layer** (called once before the attention sublayer):

1. **Build sources**: `sources = concat(committed, partial.unsqueeze(0))` → shape `(S, B, T, D)` where S = number of completed blocks + 1 (for current partial). S grows from 1 to N+1 as blocks complete.

2. **Normalize keys**: `K = RMSNorm(sources)` → `(S, B, T, D)`. The RMSNorm uses a learned weight vector `(D,)` unique to this layer. This normalization is critical — it prevents magnitude differences between block summaries (which accumulate more layers) and the partial (fewer layers) from biasing the attention weights.

3. **Compute attention logits**: `logits = einsum("d, s b t d -> s b t", query, K)` where `query` is a learned pseudo-query vector `(D,)` unique to this layer. Zero-initialized → uniform attention at start of training.

4. **Softmax over depth**: `weights = softmax(logits, dim=0)` → `(S, B, T)`. This is attention over the depth dimension — each position independently selects how much to draw from each block summary and the current partial.

5. **Weighted aggregation**: `output = einsum("s b t, s b t d -> b t d", weights, sources)` → `(B, T, D)`. The output is what gets fed to the attention (or MLP) sublayer instead of the standard residual.

**At block boundaries** (layer indices 0, 7, 14, 21 for N=4):
- Freeze current `partial` into `committed` (append along dim 0)
- Reset `partial` to zeros

**Final aggregation**: After all 28 layers, one more routing call produces the final output fed to the LM head.

### Call count

With once-per-layer routing: 28 layers + 1 final = **29 routing calls** per forward pass. The paper also supports twice-per-layer (pre-attention + pre-MLP = 57 calls), but we use once-per-layer.

### Parameter overhead

Per layer: 1 pseudo-query `(D,)` + 1 RMSNorm weight `(D,)` = 2D parameters.
Total: 28 layers × 2D + 2D (final) = 58D parameters.
At D=3072: 58 × 3072 = 178K params (0.006% of 3B). Negligible.

## The Performance Problem

### Measured numbers (proxy scale, D=512, 28 layers, B=4 per GPU, T=2048)

| Config | tok/s (1 GPU) | tok/s (8 GPU DDP) |
|---|---|---|
| Standard + compile | 256K | 1,183K |
| AttnRes + compile | 117K | 650K |
| **Overhead** | **54%** | **45%** |

### Why 54% when the FLOP overhead is ~4%?

The routing operation is **memory-bandwidth-bound**, not compute-bound. Each routing call:
- Reads the entire source buffer `(S, B, T, D)` for RMSNorm
- Reads it again for the dot-product einsum
- Reads it again for the weighted-sum einsum
- Writes the output `(B, T, D)`

At D=512, B=4, T=2048, S=5 (average): each call reads ~40MB and writes ~8MB. With 29 calls: ~1.4GB of memory traffic added to a forward pass that otherwise does ~0.8GB of memory traffic for layer computation. The routing adds more memory traffic than the entire rest of the model.

Additionally:
- **29 separate kernel launches** (each routing call = 4 CUDA kernels: RMSNorm, einsum, softmax, einsum = ~116 kernel launches total)
- **Variable S dimension** (grows from 1 to 5 across calls) prevents PyTorch's inductor from caching/reusing kernel configurations
- **Inductor disables "online softmax"** optimization because the reduction dimension varies, falling back to a slower softmax path

### The scaling argument

| Property | D=512 (proxy) | D=3072 (3B) | Scaling |
|---|---|---|---|
| Per-layer compute (matmuls) | ~1.1ms | ~25ms | **D² = 36x** |
| Per routing call (memory-bound) | ~0.19ms | ~1.1ms | **D = 6x** |
| Kernel launch overhead per call | ~10μs | ~10μs | **1x (constant)** |
| Total routing (29 calls) | ~5.5ms | ~32ms | 6x |
| Total model forward | ~32ms | ~720ms | 22x |
| **Routing as % of forward** | **~17%** | **~4.4%** |
| **Measured overhead (with all overheads)** | **54%** | **est. 8-15%** |

The measured 54% exceeds the theoretical 17% because of: Python loop overhead in the layer iteration, buffer allocation/zeroing at each boundary, torch.compile graph fragmentation from the variable-S dimension, and the backward pass amplifying all of these.

At 3B, the per-layer compute dominates so heavily (25ms vs 1.1ms per routing call) that the routing overhead becomes a small fraction. The kernel launch overhead (constant ~10μs) also becomes negligible relative to kernel execution time.

## Current Implementation

```python
# Simplified pseudocode of the forward pass with AttnRes
def _forward_attn_res(self, embed, mask):
    committed = None  # (N, B, T, D) or None
    partial = embed    # (B, T, D)

    for i, layer in enumerate(self.layers):
        # Build sources tensor
        p = partial.unsqueeze(0)  # (1, B, T, D)
        if committed is not None:
            sources = torch.cat([committed, p], dim=0)  # (S, B, T, D)
        else:
            sources = p

        # Routing: softmax attention over depth
        K = layer.attn_res_norm(sources)           # RMSNorm keys
        logits = einsum("d, s b t d -> s b t", layer.attn_res_query, K)
        weights = softmax(logits, dim=0)
        h = einsum("s b t, s b t d -> b t d", weights, sources)

        # Block boundary: freeze partial, reset
        if i in boundary_set:  # {0, 7, 14, 21} for N=4
            committed = torch.cat([committed, partial.unsqueeze(0)], dim=0)
            partial = torch.zeros_like(embed)

        # Standard sublayers using routed input h
        partial = partial + layer.attn(layer.attn_norm(h), rope_cos, rope_sin, mask)
        # (pre-MLP routing would go here if using twice-per-layer)
        partial = partial + layer.ffn(layer.ffn_norm(h))

    # Final routing for output
    sources = torch.cat([committed, partial.unsqueeze(0)], dim=0)
    K = self.final_res_norm(sources)
    logits = einsum("d, s b t d -> s b t", self.final_res_query, K)
    weights = softmax(logits, dim=0)
    output = einsum("s b t, s b t d -> b t d", weights, sources)
    return self.norm(output)
```

## What We've Tried

1. **torch.compile (default mode)**: Currently in use. Provides ~2x speedup over eager for AttnRes (342K → 650K on 8 GPU). But inductor can't fully optimize the variable-S softmax.
2. **torch.cat instead of list+stack**: Replaced Python list operations with tensor concatenation. No measurable difference — the bottleneck is the routing computation itself, not the data structure.
3. **enable_gqa=True in SDPA**: Eliminates KV head expansion via repeat_interleave. Helps the attention sublayer but doesn't affect routing.

## Questions for Analysis

1. **Is there a way to batch multiple routing calls into fewer operations?** Within a block (between boundaries), S is constant. Could we process all layers in a block with a single batched operation?

2. **Can the routing operation (RMSNorm + dot product + softmax + weighted sum) be fused into a single kernel?** This would eliminate the multiple DRAM round-trips per call. The operation per call is small: sources `(S, B, T, D)` + query `(D,)` + norm_weight `(D,)` → output `(B, T, D)` with S ≤ 5 for N=4.

3. **Is amortized routing viable?** Route only at block boundaries (8 calls total instead of 29), use standard residuals within blocks. The trade-off: layers within a block lose per-layer depth selection. How much does this matter for training quality?

4. **Could a cheaper routing mechanism (linear combination, gating, fixed weights) approximate softmax attention over depth?** The softmax is the most expensive part because it requires reading the full buffer twice (once for logits, once for weighted sum).

5. **Pre-allocation and memory layout**: Currently the `committed` tensor grows via `torch.cat` at each boundary. Could pre-allocating a fixed-size buffer `(max_S, B, T, D)` and using masking/indexing improve memory access patterns?

6. **Reducing routing frequency**: Route every K layers instead of every layer. What's the theoretical cost? The committed blocks only change at boundaries (every 7 layers for N=4), so within a block, only the `partial` slot changes between routing calls. The information gain from per-layer routing within a block may be small.

7. **Are there tricks from mixture-of-depths, early-exit, or efficient routing literature that apply here?**

8. **Given the scaling argument (8-15% at 3B), should we even optimize at proxy scale?** Or accept the proxy overhead and focus engineering effort on the 3B implementation?

## Constraints

- Mathematical semantics must be preserved (or changes must be justified and validated)
- Must work with torch.compile and activation checkpointing
- Must support both uniform blocks (N=4) and variable-size blocks (arbitrary boundary list)
- Training stability is paramount — no autograd-unsafe optimizations
- We're on PyTorch 2.10, CUDA 12.9, B200 GPUs
