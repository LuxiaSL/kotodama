# luxia-base Pretraining Optimization Plan

**Source**: Audit of 4p-sc (SFT/LoRA codebase) vs luxia-base, 2026-03-31.
**Full risk assessment**: `/home/luxia/projects/CROSS_POLLINATION_RISK_ASSESSMENT.md`
**Scope**: Kernel and infrastructure optimizations ported from 4p-sc. No architectural changes.

---

## Overview

Three optimization tracks, ordered by implementation priority. Each is independent —
they can be done in any order, but the suggested order maximizes reward/effort.

**Track 1: Liger Fused Kernels (Phase 1)** — lowest risk, broadest speedup
**Track 2: Flash Attention 2** — medium risk, attention-specific speedup
**Track 3: Async Checkpointing to SHM** — infrastructure, non-blocking saves

All changes are in two files unless noted:
- `src/model/llama.py` — Tracks 1 and 2
- `src/training/checkpoint.py` — Track 3

---

## Track 1: Liger Fused Kernels (Phase 1)

### What to do

Three drop-in replacements inside `src/model/llama.py`. No config changes, no new dependencies
(`liger-kernel` is already in `pyproject.toml:27`).

### 1a. RMSNorm → LigerRMSNorm (5 standard norms)

Replace the `RMSNorm` class usage for these norms only:
- `TransformerBlock.attn_norm` (line 183)
- `TransformerBlock.ffn_norm` (line 185)
- `GQAttention.q_norm` (line 126)
- `GQAttention.k_norm` (line 127)
- `LuxiaBaseModel.norm` (final norm, line 226 area)

**DO NOT replace** AttnRes norms (`TransformerBlock.attn_res_norm`, `.mlp_res_norm`,
`LuxiaBaseModel.final_res_norm`). Reason: `_block_attn_res_from_list()` at line 314-315
accesses `norm.eps` directly. `LigerRMSNorm` stores epsilon as `self.variance_epsilon`,
not `self.eps` — this would cause an `AttributeError`. AttnRes norms should keep the
custom `RMSNorm` class.

**Implementation approach**: The cleanest way is a factory function or config flag that
selects which norm class to use. The custom `RMSNorm` class should stay in the file
(AttnRes needs it). Example:

```python
from liger_kernel.transformers.rms_norm import LigerRMSNorm

# In LuxiaModelConfig, add:
use_liger: bool = False

# In TransformerBlock.__init__:
NormClass = LigerRMSNorm if config.use_liger else RMSNorm
self.attn_norm = NormClass(config.hidden_size, eps=config.norm_eps)
self.ffn_norm = NormClass(config.hidden_size, eps=config.norm_eps)

# AttnRes norms always use custom RMSNorm regardless of use_liger:
if config.attn_res:
    self.attn_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
    ...
```

### 1b. SwiGLU inner multiply → LigerSiLUMulFunction

In `SwiGLUFFN.forward()` (line 174), replace:
```python
# Before:
return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# After:
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))
```

Do NOT use `LigerSwiGLUMLP` as a class replacement — it expects `config.hidden_act`
which `LuxiaModelConfig` doesn't have, and creates its own projection layers.

### 1c. CrossEntropy → LigerCrossEntropyLoss (with z-loss)

In `LuxiaBaseModel.forward()` (lines 409-425), replace the manual CE + z-loss with:

```python
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

# In __init__, create the loss function:
self.ce_loss = LigerCrossEntropyLoss(
    ignore_index=-100,
    lse_square_scale=config.z_loss_weight,  # maps to z_loss_weight
    return_z_loss=True,
)

# In forward(), replace lines 414-425:
loss, z_loss = self.ce_loss(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
)
output["loss"] = loss    # CE + z_loss already combined
output["z_loss"] = z_loss
```

This still materializes logits (the `F.linear` at line 405 stays). The big memory win
(skipping logit materialization via `LigerFusedLinearCrossEntropyLoss`) is a separate
follow-up that requires restructuring the forward method — prototype it separately.

### What NOT to touch

- **RoPE**: Liger's `liger_rotary_pos_emb()` expects HF-style full-dimension cos/sin
  and processes (q, k) together. The custom RoPE uses half-dimension buffers and applies
  to q and k separately. Incompatible buffer format — not worth the rewrite.
- **AttnRes norms**: `.eps` vs `.variance_epsilon` attribute mismatch breaks routing.
- **LigerSwiGLUMLP class**: Wrong constructor interface for our config.

### Verification

1. Run 10 steps with `use_liger=False`, record per-step loss values.
2. Run 10 steps with `use_liger=True`, same seed/data. Loss should match within ~1e-3 (bf16 tolerance).
3. Verify `z_loss` appears in output and matches the manual computation to similar tolerance.
4. Benchmark tokens/sec with and without Liger on the `full` 3B config.

---

## Track 2: Flash Attention 2

### What to do

Replace `F.scaled_dot_product_attention` in `GQAttention.forward()` (line 154) with an
explicit `flash_attn_func` call. This gives deterministic kernel selection instead of
relying on SDPA's backend dispatch.

### Key context

- **FA2, not FA4.** FA4's Blackwell backward is unoptimized (runs Ampere-era kernels)
  and has no `torch.compile` integration. FA2 has mature forward+backward kernels and
  registers `torch.library.custom_op` for both, making it fully `torch.compile`-compatible.
- **AttnRes is NOT affected.** AttnRes routing happens entirely outside the attention
  computation — it changes what goes INTO `GQAttention.forward()` but does not modify
  the Q@K^T@V operation itself. FA2 is a drop-in for the inner SDPA call regardless of
  whether AttnRes is on or off.
- **QK-norm is NOT affected.** QK-norm applies before RoPE, before attention. Purely
  pre-attention — no interaction with FA kernels.
- `flash-attn` is already in `pyproject.toml:26`.

### Implementation

In `GQAttention.forward()`, the changes are:

1. **Remove the `.transpose(1, 2)` calls.** FA2 expects `(B, S, nheads, head_dim)`.
   The current code transposes to `(B, nheads, S, head_dim)` for SDPA.

2. **Adjust `apply_rope`** to work with `(B, S, nheads, head_dim)` layout instead of
   `(B, nheads, S, head_dim)`. The cos/sin unsqueeze dimensions change.

3. **Replace the SDPA call** with `flash_attn_func`.

4. **Remove the post-attention transpose** — output is already `(B, S, nheads, head_dim)`.

```python
from flash_attn import flash_attn_func

def forward(self, x, rope_cos, rope_sin, mask=None):
    bsz, seq_len, _ = x.shape

    # Shape: (B, S, nheads, head_dim) — NO transpose for FA2
    q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
    k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
    v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

    if self.qk_norm:
        q = self.q_norm(q)
        k = self.k_norm(k)

    # RoPE — needs adjustment for (B, S, nheads, head_dim) layout
    q = apply_rope(q, rope_cos[:seq_len], rope_sin[:seq_len])
    k = apply_rope(k, rope_cos[:seq_len], rope_sin[:seq_len])

    # FA2 handles GQA natively via shape: Q has more heads than K/V
    # causal=True for standard pretraining (mask is always None)
    attn_output = flash_attn_func(q, k, v, causal=True)

    # Output is (B, S, nheads, head_dim) — reshape directly, no transpose
    attn_output = attn_output.contiguous().view(bsz, seq_len, -1)
    return self.o_proj(attn_output)
```

### apply_rope adjustment

The current `apply_rope` (lines 94-105) assumes `(B, nheads, S, head_dim)`:
```python
cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, S, head_dim//2) for (B, nheads, S, D)
```

For FA2's `(B, S, nheads, head_dim)` layout:
```python
cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, head_dim//2) for (B, S, nheads, D)
sin = sin.unsqueeze(0).unsqueeze(2)
```

### Fallback path

Add a config flag and import guard so SDPA remains available:

```python
# In LuxiaModelConfig:
attn_impl: str = "auto"  # "auto", "fa2", "sdpa"

# At module level:
try:
    from flash_attn import flash_attn_func
    _FA2_AVAILABLE = True
except ImportError:
    _FA2_AVAILABLE = False
```

When `attn_impl="auto"`: use FA2 if available, else SDPA.
When a custom mask is passed (non-None): always fall back to SDPA
(FA2 doesn't accept arbitrary attention masks; `causal=True` handles the standard case).

### What about torch.compile?

FA2 registers `torch.library.custom_op` for both forward and backward, so it works
inside `torch.compile` graphs without causing breaks. The recommended benchmark is:

1. `SDPA` (baseline)
2. `torch.compile + SDPA` (auto-dispatch)
3. `FA2` (explicit)
4. `torch.compile + FA2` (explicit + compiled surrounding ops)

If `torch.compile + SDPA` already auto-dispatches to FlashAttention AND fuses surrounding
ops better than explicit FA2, the explicit FA2 path adds complexity for no gain. The
benchmark determines whether explicit FA2 is worth keeping.

### Verification

1. **Numerical parity**: 10 steps SDPA vs FA2, same seed. Loss within ~1e-3.
2. **Gradient parity**: Single forward+backward, compare gradient tensors. Max abs diff < 1e-2.
3. **GQA shape assertion**: At the FA2 call site, assert Q is `(B, S, 24, 128)` and K/V
   are `(B, S, 8, 128)` (for the `full` config).
4. **AttnRes smoke test**: Run with `--attn_res` and FA2 enabled. Loss curve should match
   SDPA+AttnRes within tolerance.
5. **Benchmark**: tokens/sec on `full` 3B with `--activation_checkpointing`, 8xB200.

---

## Track 3: Async Checkpointing to SHM

### What to do

Modify `CheckpointManager` in `src/training/checkpoint.py` to:
1. Clone state dicts to CPU synchronously (the blocking cost)
2. `torch.save` to `/dev/shm` in a background thread
3. Background zstd compress + move to NVMe
4. Rotation (delete old checkpoints)

Keep the single-file `.pt` schema. Keep atomic rename. Keep SIGTERM on the blocking path.

### Why this matters

- Node1 has **1 TB** `/dev/shm` — 24 GB checkpoints fit trivially.
- NVMe is consistently 70-80% full — checkpoints living in SHM during training and only
  hitting disk after compression saves significant disk space.
- Current blocking save time is unknown — measure first. If `torch.save` of ~24 GB to
  NVMe takes 10-30s, that's 10-30s of idle GPUs per checkpoint.

### Implementation sketch

```python
import threading
from queue import Queue

class AsyncCheckpointManager(CheckpointManager):
    """Extends CheckpointManager with background SHM writes + zstd compression."""

    def __init__(self, checkpoint_dir, rank=0, keep_last_n=5,
                 shm_dir="/dev/shm/luxia-base-ckpts", compress=True):
        super().__init__(checkpoint_dir, rank, keep_last_n)
        self.shm_dir = Path(shm_dir)
        self.compress = compress
        self._queue: Queue = Queue(maxsize=3)  # backpressure
        self._worker = threading.Thread(target=self._bg_worker, daemon=True)
        self._worker.start()

    def save(self, step, model, muon_opt, adamw_opt, scheduler,
             tokens_consumed=0, data_state=None, extra=None):
        """Synchronous CPU clone + async SHM write."""
        if self.rank != 0:
            # Non-rank-0: just wait at barrier
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.barrier()
            return None

        # --- Synchronous: clone state to CPU ---
        raw_model = model.module if hasattr(model, "module") else model
        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        state = {
            "step": step,
            "tokens_consumed": tokens_consumed,
            "model": {k: v.cpu().clone() for k, v in raw_model.state_dict().items()},
            "muon_optimizer": _deep_cpu(muon_opt.state_dict()),
            "adamw_optimizer": _deep_cpu(adamw_opt.state_dict()),
            "scheduler": _scheduler_state(scheduler),
        }
        if data_state is not None:
            state["data"] = data_state
        if extra is not None:
            state.update(extra)

        # --- Barrier: all ranks sync after CPU clone, before background write ---
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

        # --- Async: enqueue for background thread ---
        self._queue.put((step, state))
        return self.checkpoint_dir / f"step_{step:08d}.pt"

    def _bg_worker(self):
        """Background: write to SHM, compress, move to NVMe."""
        while True:
            item = self._queue.get()
            if item is None:  # shutdown sentinel
                break
            step, state = item

            self.shm_dir.mkdir(parents=True, exist_ok=True)
            shm_path = self.shm_dir / f"step_{step:08d}.pt"
            shm_tmp = shm_path.with_suffix(".tmp")
            final_path = self.checkpoint_dir / f"step_{step:08d}.pt"

            # Write to SHM (fast — RAM bandwidth)
            torch.save(state, shm_tmp)
            shm_tmp.rename(shm_path)  # atomic in SHM

            if self.compress:
                # Compress in SHM, then move to NVMe
                zst_path = shm_path.with_suffix(".pt.zst")
                subprocess.run(["zstd", "-3", str(shm_path), "-o", str(zst_path)],
                               check=True, capture_output=True)
                shm_path.unlink()
                shutil.move(str(zst_path), str(final_path.with_suffix(".pt.zst")))
            else:
                # Move uncompressed to NVMe
                shutil.move(str(shm_path), str(final_path))

            self._cleanup_old(keep=self.keep_last_n)

    def save_blocking(self, step, model, muon_opt, adamw_opt, scheduler,
                      tokens_consumed=0, data_state=None, extra=None):
        """SIGTERM path: synchronous save, no background thread."""
        return super().save(step, model, muon_opt, adamw_opt, scheduler,
                           tokens_consumed, data_state, extra)

    def shutdown(self):
        """Drain queue and stop background worker."""
        self._queue.put(None)
        self._worker.join(timeout=60)
```

### SIGTERM integration

In `train.py`, the SIGTERM path (lines 638-653) should call `save_blocking()`, not `save()`:
```python
if sigterm.received:
    ckpt_mgr.save_blocking(step, model, muon_opt, adamw_opt, ...)
    break
```

Normal periodic saves use `save()` (async). SIGTERM always blocks.

### Resume from compressed checkpoints

`load_latest()` needs to handle `.pt.zst` files:
```python
checkpoints = sorted(
    list(self.checkpoint_dir.glob("step_*.pt"))
    + list(self.checkpoint_dir.glob("step_*.pt.zst"))
)
# For .zst: decompress to /tmp, load, clean up
```

### Measure first

Before implementing, measure the baseline blocking cost:
```python
# Add timing to current CheckpointManager.save():
t0 = time.time()
torch.save(state, tmp_path)
elapsed = time.time() - t0
logger.info("torch.save took %.1fs for %.1f GB", elapsed, tmp_path.stat().st_size / 1e9)
```

If blocking saves are already < 5s, async may not be worth the complexity. If they're
10-30s (likely for 24 GB to a busy NVMe), async is clearly worth it.

### Verification

1. **Round-trip test**: Save async, load, verify model output matches pre-save output.
2. **SIGTERM test**: Send SIGTERM during a background save, verify the blocking emergency
   checkpoint is valid and resumable.
3. **Compression ratio**: Measure `.pt` vs `.pt.zst` size for a full checkpoint.
4. **Timing**: Compare blocking save time (current) vs CPU clone time (new blocking portion).

---

## Dependencies to add

Only if needed (most are already declared):

```toml
# In pyproject.toml [project.optional-dependencies] training:
# Already present:
#   "flash-attn"     — Track 2
#   "liger-kernel"    — Track 1
# Add:
    "zstandard",      # Track 3: Python fallback for zstd compression
```

Ensure `zstd` CLI is available on node1 (preferred over Python fallback for speed).

---

## Quick reference: what NOT to do

| Temptation | Why not |
|---|---|
| Use FA4 instead of FA2 | Unoptimized backward on Blackwell, no torch.compile support |
| Replace AttnRes norms with LigerRMSNorm | `.eps` vs `.variance_epsilon` breaks `_block_attn_res_from_list()` |
| Use `LigerSwiGLUMLP` class | Expects `config.hidden_act`, creates own projections |
| Use `liger_rotary_pos_emb()` | Expects full-dim cos/sin + joint (q,k) — incompatible with half-dim separate application |
| Port 4p-sc's async_checkpoint.py directly | It's an HF TrainerCallback with adapter-oriented serialization. Borrow the pattern, not the code. |
| Use LigerFusedLinearCrossEntropyLoss in phase 1 | Needs forward-path restructuring to skip logit materialization. Separate prototype. |
