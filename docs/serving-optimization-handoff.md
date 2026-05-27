# Handoff: Serving Optimization

## Current State (2026-05-09)

Phases 0-1 complete. The server runs at **100-117 tok/s** (7.8ms/tok decode) with `--compile` flag, up from 50-52 tok/s. Gate 1 logit parity passed (224/224 top-1 matches). Next target is last-logits-only prefill (Phase 2), then static KV cache (Phase 3).

## What Was Done

### Phase 0 (2026-05-08)
1. **Root cause found and fixed:** cuDNN SDPA plan selection was costing ~300ms per novel KV shape. Fixed by disabling cuDNN SDP backend and adding startup warmup.
2. **CPU threading optimized:** Default 128 threads → 2 threads. cuDNN needs ≥2 but more provides zero benefit for single-request GPU inference.
3. **Benchmark infrastructure built:** Harness, prompt pack, logit parity test, torch.profiler script, thread sweep, length sweep, SDPA backend matrix.
4. **Timing instrumentation added to serve.py:** Prefill, per-token decode forward/sample/stop, tokenize, final decode — all in response JSON.
5. **zstd checkpoint support added** to serve.py and all benchmark scripts.

### Phase 1 (2026-05-09)
6. **torch.compile decode path:** Eager prefill + compiled decode strategy. `_forward_attn_res_cached` has 0 graph breaks (1634-op single graph). The "stall" was inductor compilation time, not graph breaks.
7. **Compile warmup added:** 4 decode steps at startup trigger the two shape specializations. ~33s on warm inductor cache, ~80s cold.
8. **Prefill warmup added:** Short eager prefills at varied lengths to avoid first-request latency spike.
9. **compile_parity.py created:** Eager vs compiled logit comparison at 6 prompt lengths.
10. **compile_diagnose.py created:** `torch._dynamo.explain()` wrapper + compile+decode timing test.

## Key Files

### Modified
- `pretraining/serve.py` — flash SDP backend, threading defaults, timing instrumentation, `/memory` endpoint, `cuda:N` device support, zstd checkpoint loading, `--compile` flag with eager-prefill/compiled-decode strategy, compile warmup, prefill warmup

### Created (all under `pretraining/scripts/benchmark/`)
- `generate_prompt_pack.py` — builds prompts at exact token counts (16/128/1024/3072 × 4 domains)
- `bench.py` — server benchmark harness, outputs JSON + Markdown
- `logit_parity.py` — Gate 1: full-forward vs cached-forward at 22 positions + extended decode
- `profile_decode.py` — torch.profiler kernel-level breakdown with Chrome trace export
- `thread_sweep.py` — OMP/torch thread count sweep + torch.compile test
- `length_sweep.py` — prompt/decode length scaling analysis
- `sdpa_backend_matrix.py` — SDPA backend comparison + shape-cache priming + randomized order
- `compile_diagnose.py` — torch._dynamo.explain() diagnostics + compile timing test
- `compile_parity.py` — eager vs compiled decode logit comparison

### Results on gpu-host (at `~/workspace/kotodama/scripts/benchmark/results/`)
- `baseline.json` / `baseline.md` — pre-fix benchmark (cuDNN, 128 threads)
- `flash-sdp-baseline.json` / `flash-sdp-baseline.md` — Phase 0 baseline (flash SDP, 2 threads, 50-52 tok/s)
- `logit-parity-baseline.json` / `logit-parity-baseline.md` — Phase 0 Gate 1 results
- `compile-decode.json` / `compile-decode.md` — Phase 1 benchmark (compiled decode, 100-117 tok/s)
- `compile-parity.json` — Phase 1 eager vs compiled logit parity (224/224 top-1)

### Plan
- `pretraining/docs/serving-optimization-and-quality-plan.md` — updated with Phase 0 findings and revised phase ordering

## How to Work With GPU host

### SSH and paths
```bash
ssh gpu-host
# Project: ~/workspace/kotodama/
# Shared venv: /home/cluster-user/workspace/.venv-shared/
# Python: /home/cluster-user/workspace/.venv-shared/bin/python
```

### Syncing code
Code is edited locally at `~/projects/kotodama/pretraining/` and synced to gpu-host:
```bash
# Sync specific files
rsync -avz <repository-root>/serve.py gpu-host:~/workspace/kotodama/serve.py
rsync -avz <repository-root>/scripts/benchmark/ gpu-host:~/workspace/kotodama/scripts/benchmark/

# DO NOT use rsync --relative — it creates nested path structures
```

### Running the server
```bash
# With compile (recommended — 2x faster decode)
ssh gpu-host 'CUDA_VISIBLE_DEVICES=7 nohup /home/cluster-user/workspace/.venv-shared/bin/python \
    /home/cluster-user/workspace/kotodama/serve.py \
    --checkpoint /home/cluster-user/workspace/kotodama/checkpoints/fullcorpus-ddv1/step_00081252.pt.zst \
    --device cuda --port 2223 --compile > /tmp/serve_benchmark.log 2>&1 &'

# Without compile (eager mode, lower memory)
ssh gpu-host 'CUDA_VISIBLE_DEVICES=7 nohup /home/cluster-user/workspace/.venv-shared/bin/python \
    /home/cluster-user/workspace/kotodama/serve.py \
    --checkpoint /home/cluster-user/workspace/kotodama/checkpoints/fullcorpus-ddv1/step_00081252.pt.zst \
    --device cuda --port 2223 > /tmp/serve_benchmark.log 2>&1 &'

# Wait for ready
ssh gpu-host "until curl -s http://localhost:2223/health 2>/dev/null | grep -q model_loaded; do sleep 2; done; echo ready"

# Check startup log
ssh gpu-host "tail -10 /tmp/serve_benchmark.log"
```

Startup with `--compile`: ~47s (zstd ~4s, Triton ~7s, compile warmup ~33s warm/~80s cold, prefill ~1s).
Startup without `--compile`: ~80s (zstd ~4s, Triton ~7s, SDPA warmup ~70s).

### Running benchmarks
```bash
# Against running server
ssh gpu-host "cd ~/workspace/kotodama && /home/cluster-user/workspace/.venv-shared/bin/python \
    scripts/benchmark/bench.py --url http://localhost:2223 \
    --output scripts/benchmark/results/my-test"

# Direct model tests (no server needed, loads model directly)
ssh gpu-host "CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    /home/cluster-user/workspace/.venv-shared/bin/python \
    ~/workspace/kotodama/scripts/benchmark/logit_parity.py \
    --checkpoint /home/cluster-user/workspace/kotodama/checkpoints/fullcorpus-ddv1/step_00081252.pt.zst \
    --device cuda"
```

### GPU allocation
GPU 7 was used for all benchmark work. Check what's running before claiming a GPU:
```bash
ssh gpu-host "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,name --format=csv"
```

## Gotchas

1. **Quote escaping in SSH commands.** Complex bash commands with nested quotes fail over SSH. Write scripts as files and sync them rather than inlining Python in ssh commands.

2. **rsync --relative creates nested paths.** `rsync --relative <local-user-home>/.../serve.py gpu-host:~/workspace/kotodama/` creates `~/workspace/kotodama<local-user-home>/.../serve.py`. Don't use `--relative`.

3. **grep -v filters eat print output.** Piping SSH output through `grep -v "WARNING|..."` to suppress torch warnings also drops legitimate output if the filter is too broad. When running long jobs, skip the filter or tee to a file.

4. **Background SSH commands need careful quoting.** Use single quotes for the outer SSH command and avoid `&&` chains with nohup. These patterns work:
   ```bash
   # Works
   ssh gpu-host 'CUDA_VISIBLE_DEVICES=7 nohup ... > /tmp/log 2>&1 &'
   
   # Fails (exit 255)
   ssh gpu-host "pkill -f 'something' 2>/dev/null; sleep 1; nohup ... &"
   ```

5. **torch.compile compilation time.** `_forward_attn_res_cached` has 0 graph breaks but is a 1634-op single graph. First compilation takes ~80s cold / ~33s with inductor cache. The Phase 1 fix uses eager prefill + compiled decode with startup warmup. Different prompt lengths trigger ~30s prefill recompilation if the compiled model is used for prefill — this is why the server uses eager prefill.

6. **Shared node safety.** Never run pip/conda outside the venv. Use `uv` for package management. The venv at `/home/cluster-user/workspace/.venv-shared/` is shared across projects. See global CLAUDE.md.

7. **Checkpoint is zstd compressed.** All benchmark/test scripts handle `.pt.zst` files via `zstandard` library. Always use absolute paths for checkpoints on gpu-host.

8. **OMP_NUM_THREADS must be set before torch import.** The serve.py sets it at module level before `import torch`. For standalone scripts, set it via environment variable in the shell command or at the top of the script before torch import.

## Optimization Loop

For each optimization phase:

1. **Edit code locally** at `~/projects/kotodama/pretraining/`
2. **Sync to gpu-host:** `rsync -avz <file> gpu-host:~/workspace/kotodama/<path>`
3. **Run logit parity test** to verify no quality regression
4. **Start server** on GPU 7, port 2223
5. **Run benchmark harness** to measure latency/throughput
6. **Compare** against `compile-decode.json` results (current best baseline)
7. **Record** results in `scripts/benchmark/results/<phase-name>.json`

If any gate fails, inspect logits before text. The plan document has detailed failure diagnosis per phase.

## Next Up: Phase 2 (Last-Logits-Only Prefill)

Prefill currently computes logits for all prompt positions, but generation only needs the last. This is a small win (prefill is already 19ms) but reduces memory.

Target: add an `inference_only_last_logit=True` path in `LuxiaBaseModel.forward()` that slices `x[:, -1:]` before the LM head projection during `use_cache=True` inference.

Required gates: baseline vs optimized logits parity for final prompt position, greedy exact match.

## Phase 3: Static KV Cache

Replace the `torch.cat([past_kv, new_kv], dim=2)` pattern with pre-allocated cache tensors. At proxy scale KV concat is 11% of GPU time; at 3B it dominates.

**torch.compile memory note:** Phase 1 showed torch.compile reserves ~167GB on B200 at proxy scale. Static KV cache + compile at 3B needs careful memory budgeting.
