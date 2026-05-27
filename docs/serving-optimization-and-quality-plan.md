# Serving Optimization And Quality Plan

## Purpose

This document defines the path for improving `serve.py` performance without repeating the previous failure mode where generation got faster but output quality regressed.

The core principle is: optimize only behind measurement and distribution parity. Text samples are still useful, but they should be the final review layer, not the first signal that something broke.

## Current Serving Shape

The current serving entry point is `serve.py`.

The hosted model is a proxy DD-v1 Luxia model with:

- Llama-family decoder architecture
- 28 layers
- hidden size 512
- 4 query heads
- 2 KV heads
- head dim 128
- QK-norm
- tied embeddings
- 4096 context
- AttnRes enabled
- DD-v1 AttnRes boundaries: `[0, 3, 7, 12, 21, 25]`

Important current observations:

- `serve.py` builds and warms a `FastAttnResContext`, but the `/generate` path calls `model(..., use_cache=True)` directly, so the server-level `fast_forward_attn_res()` path is not used for served generation.
- Cached decode currently grows K/V with `torch.cat([past_kv, new_kv], dim=2)` in every layer on every token. At proxy scale this is NOT the dominant cost (only 11% of GPU time), but will matter at 3B.
- Prefill computes logits for every prompt token even though generation only needs the final prompt position.
- `use_cache=True` forces the SDPA attention path, so FA2/FA4 attention optimizations do not currently apply to served generation. FA2 does not implement KV cache; SDPA flash backend handles it natively.
- Sampling and stop handling perform repeated CPU synchronization and repeated decoding work.
- The server handles generation synchronously per request. Throughput under concurrency will need explicit scheduling or batching after single-stream decode is healthy.

## Phase 0 Findings (Completed 2026-05-08)

### Root cause: cuDNN SDPA plan selection overhead

The dominant bottleneck at proxy scale was NOT KV cache copying or GPU compute. It was cuDNN's per-call CPU-side plan selection for `F.scaled_dot_product_attention`, costing ~300ms per novel KV shape. This manifested as:

- Decode at 3-4 tok/s (300-350ms/tok) whenever cuDNN encountered a new KV sequence length
- 17x slowdown ratios as generation progressed through novel shapes
- Misleading pattern where some prompt lengths appeared fast (shapes already cached from prior runs)

Evidence:
- `torch.profiler` showed 34ms total CUDA kernel time vs 2.7s total CPU time for 9 forward passes
- `_cudnn_attention_forward` consumed 85% of CPU time (2.35s) for 1.3ms of CUDA work
- Disabling cuDNN SDP (`math_only` or `flash_only` backend) eliminated all spikes
- Shape-cache priming (decoding all lengths before serving) eliminated all spikes on default backend

### Fix applied

1. `torch.backends.cuda.enable_cudnn_sdp(False)` — forces flash/math SDPA backend
2. Startup decode warmup — primes remaining plan caches for all KV lengths up to max_seq_len
3. CPU threading set to 2 — cuDNN needs ≥2 threads but >2 provides no benefit; avoids 128-thread default on large Xeons

### Result

- Decode: 127-348ms/tok → **19.1ms/tok stable** (6.7-17x improvement)
- Throughput: 3-52 tok/s (variable) → **50-52 tok/s stable** across all prompt/gen lengths
- Gate 1 logit parity: **54/54 positions top-1 match**, max KL ~1e-5 (bf16 noise)

### Additional finding: torch.compile

`torch.compile(dynamic=True)` yields an additional 2.3x decode speedup (19ms → 7.8ms/tok, ~128 tok/s) at optimal threading. However, the initial compilation takes 30-80s for the 1634-op graph (zero graph breaks, but large inductor workload). Different prompt lengths trigger prefill recompilation (~30s each). See Phase 1 Findings for the solution.

### GPU compute breakdown (proxy scale, batch=1, seq_len=1)

At proxy scale, actual GPU compute per decode step is ~3.8ms. The breakdown:

| Component | CUDA Time | % |
|---|---:|---:|
| aten::mul (RMSNorm, routing) | 9.2ms | 27% |
| aten::mm (linear projections) | 6.8ms | 20% |
| aten::cat (KV cache concat) | 3.6ms | 11% |
| aten::add (residuals) | 3.5ms | 10% |
| aten::mean (RMSNorm) | 2.8ms | 8% |
| phase_1 Triton (AttnRes routing) | 1.6ms | 5% |
| SDPA (attention) | 1.3ms | 4% |
| aten::stack (pad_and_stack) | 0.8ms | 2% |

These proportions will shift at 3B scale where matmuls dominate.

## Non-Negotiable Guardrail

Every serving optimization must answer this before quality review:

> Given the same prompt and same prefix tokens, does the optimized path produce the same next-token distribution as the baseline path?

Most serious quality regressions from cache or serving rewrites show up as logit drift before they show up as visibly bad text.

Common causes:

- RoPE position offset mismatch
- cache cursor off by one
- K/V write layout mismatch
- K/V read layout mismatch
- layer index mismatch in cache storage
- head or KV-head grouping mismatch
- causal mask mismatch
- stale cache after request reset
- incorrect prompt/decode split
- changed dtype in a sensitive operation
- AttnRes routing state mismatch
- sampling semantics changed while model logits stayed correct

## Baseline Benchmark Harness

Create a fixed benchmark harness that can run against the current server and future variants.

Prompt lengths:

- 16 tokens
- 128 tokens
- 1024 tokens
- 3072 tokens

Generation lengths:

- 32 new tokens
- 256 new tokens
- 1024 new tokens

Decode modes:

- greedy: `temperature=0`
- top-k: `temperature=0.7`, `top_k=50`
- top-p: `temperature=0.7`, `top_p=0.9`
- streaming
- non-streaming

Concurrency levels:

- 1 request first
- 2, 4, and 8 requests only after single-request performance is understood

Metrics to capture:

- prompt tokens
- completion tokens
- time to first token
- total latency
- prefill latency
- prefill tokens/sec
- decode latency per token
- decode tokens/sec
- total tokens/sec
- GPU allocated memory
- GPU reserved memory
- peak GPU memory
- CPU sampling time
- CPU tokenization time
- CPU text decoding time
- stop-string check time
- generated token IDs for deterministic modes

The benchmark output should be machine-readable JSON plus a small Markdown summary table.

## Instrumentation Plan

Add timing around the major phases. Use `time.perf_counter()` for CPU sections and CUDA events for GPU sections.

Sections to measure:

- request validation
- tokenization
- prefill model forward
- prefill LM-head projection
- first-token sampling
- per-token decode model forward
- per-token LM-head projection
- per-token sampling
- next-input construction
- text decode
- stop-string checks
- response serialization

For GPU profiling, use a representative small matrix first:

- 128 prompt tokens, 256 generated tokens
- 1024 prompt tokens, 256 generated tokens
- 3072 prompt tokens, 256 generated tokens

Profiler goals:

- identify whether wall time is dominated by KV copying, AttnRes routing, attention, LM head, sampling, or CPU overhead
- identify repeated allocations in decode
- identify graph breaks if using `torch.compile`
- identify whether routing kernels dominate after the cache path is fixed

## Correctness And Quality Gates

### Gate 1: Full Forward Vs Cached Forward

For each prompt, compare:

1. Full forward over `prompt[:t]`
2. Incremental cached forward up to position `t`

For each `t`, compare the next-token logits.

Track:

- max absolute logit difference
- mean absolute logit difference
- KL divergence between softmax distributions
- top-1 token match
- top-10 overlap
- top-50 overlap

Run this over multiple prompt lengths, especially around suspicious boundaries.

Boundary lengths:

- 1
- 2
- 3
- 7
- 12
- 21
- 25
- 31
- 32
- 127
- 128
- 255
- 256
- 511
- 512
- 1023
- 1024
- 2047
- 2048
- 3071
- 3072
- 4095

The early small numbers cover AttnRes boundary-adjacent behavior. The powers of two and near-powers of two catch common cache and kernel shape bugs.

### Gate 2: Baseline Server Vs Optimized Server

Use the current implementation as the behavioral baseline.

For fixed prompt and fixed generated prefix, compare baseline next-token logits against optimized next-token logits before sampling.

This catches cases where the optimized cache path is internally self-consistent but no longer matches the old model behavior.

### Gate 3: Greedy Exact Match

With `temperature=0`, generated token IDs should match exactly for the same prompt.

Run across:

- short prompts
- medium prompts
- long prompts
- near-context-limit prompts
- prompts with punctuation-heavy text
- prompts with code-like text
- prompts with repeated phrases

If exact match fails, inspect logits before inspecting prose.

### Gate 4: Sampling Semantics

Sampling behavior should be tested separately from model logits.

Test:

- `temperature=0`
- `temperature>0`
- `top_k=0`
- `top_k=1`
- `top_k=50`
- `top_p=0`
- `top_p=0.9`
- repetition penalty disabled
- repetition penalty enabled
- EOS handling
- stop strings
- streaming deltas

For stochastic modes, do not expect identical text unless RNG behavior is explicitly controlled. Instead, compare logits before sampling and verify that filtering masks are identical.

### Gate 5: Perplexity Smoke

Run a small fixed held-out perplexity check before and after each optimization.

This should be small enough to run frequently but broad enough to catch obvious distribution damage.

Record:

- baseline loss
- optimized loss
- absolute delta
- relative delta

Any meaningful movement is a blocker unless the optimization intentionally changes numerics and the change has been justified.

### Gate 6: Automated Text Quality Signals

After logit parity and perplexity pass, generate a fixed text-quality sample pack.

Use the existing analysis tooling where possible:

- `scripts/analysis/eval_via_server.py`
- `scripts/analysis/eval_generate.py`
- `scripts/analysis/analyze_text_quality.py`

Track:

- repetition rates
- unique token ratio
- average completion length
- EOS frequency
- stop reason distribution
- degeneracy indicators
- entity density if available from the existing quality analysis
- prompt adherence notes from human review

These metrics are secondary. They help catch sampling bugs and quality shifts that do not show up in a small perplexity set.

## Optimization Sequence

### Phase 0: Benchmark And Profiling Foundation

Deliverables:

- server benchmark harness
- fixed prompt set
- timing breakdown in `/generate`
- optional profiler script or command
- baseline JSON results
- baseline Markdown summary

Exit criteria:

- current server has a reproducible latency and throughput profile
- current server has reproducible greedy outputs for the fixed prompt set
- current cache path has known full-forward parity numbers

Do not optimize before this phase is complete.

**Status: COMPLETE (2026-05-08).** See Phase 0 Findings section above. Baseline established at 50-52 tok/s stable with flash SDP backend. Gate 1 logit parity passed.

### Phase 1: Make torch.compile Viable For Cached Decode

Problem:

`torch.compile(dynamic=True)` yields 2.3x decode speedup (19ms → 7.8ms/tok) but appeared to stall on the AttnRes cached path during Phase 0 investigation.

Target change:

- make `_forward_attn_res_cached` compile-friendly by replacing dynamic list ops with fixed-shape tensor operations
- alternatively, isolate the AttnRes routing from the compiled graph and only compile the per-layer forward
- test with fixed decode shape first, then growing cache length, to isolate which operations cause recompilation

Expected benefit:

- 2.3x decode speedup (50 tok/s → ~128 tok/s)
- prefill also benefits (17.9ms → 7.2ms)

Quality risk:

- low if the compile produces identical numerics (it should)

Required gates:

- greedy exact match before and after compile
- logit parity at boundary positions
- verify no graph breaks during steady-state decode (use `torch._dynamo.config.log_level` or `TORCH_LOGS`)

Failure diagnosis:

- if compile stalls: use `torch._dynamo.explain()` to find graph break locations
- if numerics drift: compare compiled vs eager logits at each layer
- if only some lengths work: the issue is shape specialization — test with `dynamic=True` vs explicit `torch._dynamo.mark_dynamic()`

**Status: COMPLETE (2026-05-09).** See Phase 1 Findings below.

#### Phase 1 Findings

**Root cause re-diagnosed:** `torch._dynamo.explain()` showed **zero graph breaks** in `_forward_attn_res_cached` — the entire 1634-op graph traces as a single unit. The "stall" from Phase 0 was inductor compilation time for the large graph, not graph breaks or recompilation loops.

**Solution: eager prefill + compiled decode.**
- Prefill uses eager model (variable prompt shapes would trigger 28-30s recompilation each)
- Decode uses `torch.compile(model, dynamic=True)` — compiles once, then stable at 7.8ms/tok
- Startup warmup runs 4 compiled decode steps to trigger the two shape specializations
- Inductor cache persists across restarts (33s warmup on warm cache vs 80s cold)

**Benchmark results (vs Phase 0 baseline):**

| Metric | Phase 0 (eager) | Phase 1 (compiled) | Improvement |
|---|---:|---:|---:|
| Greedy 16+256 tok/s | 51.7 | 117.0 | 2.26x |
| Greedy 128+1024 tok/s | 51.9 | 106.6 | 2.05x |
| Greedy 3072+1024 tok/s | 50.5 | 110.5 | 2.19x |
| Decode forward ms/tok | 19.1 | 7.8 | 2.45x |
| Prefill ms | 19.2 | 19.5 | same |
| GPU reserved GB | 1.2 | 0.4 (after empty_cache) | same |

**Gate 1 logit parity:** 224/224 top-1 matches (100%). Max KL=2.03e-3 at one position (bf16 compile rounding noise). All top-10 overlaps ≥90%.

**Memory note:** torch.compile initially reserves ~167GB for inductor workspace during compilation, but `torch.cuda.empty_cache()` after warmup releases it to 0.4GB with no performance impact. nvidia-smi shows 1.14GB total process footprint.

**Server changes:**
- `serve.py` now accepts `--compile` flag
- Eager prefill + compiled decode — no model code changes needed
- Startup: zstd (~4s) + Triton (~7s) + compile warmup (~33s warm / ~80s cold) + prefill warmup (<1s)

### Phase 2: Last-Logits-Only Prefill

Note: prefill is already fast (~19ms at all prompt lengths). This is a small win. Prioritize only after Phase 1 (compile) is resolved.

Problem:

Prefill currently projects all hidden states through the LM head, but generation only needs the final position logits.

Target change:

- add an inference option that returns only final-position logits during generation
- preserve full logits for training, perplexity, and analysis paths

Expected benefit:

- lower prefill latency for long prompts
- lower peak memory during prefill

Quality risk:

- low, if hidden states are unchanged

Required gates:

- baseline vs optimized logits parity for final prompt position
- greedy exact match
- perplexity smoke

Failure diagnosis:

- if hidden-state parity holds but logits differ, inspect LM-head slice/indexing
- if prefill gets slower, inspect whether slicing introduced an extra copy or graph break

### Phase 3: Static Or Preallocated KV Cache

Problem:

Decode currently concatenates K/V tensors on every token for every layer. This repeatedly copies the existing cache.

Target change:

- allocate fixed K/V cache storage per request up to maximum needed length
- write new K/V at the current cursor position
- pass cache storage and valid length into attention
- avoid reallocating or copying the whole cache each token

Expected benefit:

- major decode speedup for medium and long generations
- lower allocation churn
- more stable latency per token

Quality risk:

- high

Required gates:

- full forward vs cached forward parity at many positions
- baseline cache vs static cache logits parity
- greedy exact match
- long-generation greedy exact match
- boundary-length tests
- perplexity smoke

Failure diagnosis:

- top-1 diverges at token 1: inspect prefill cache write and first decode position
- divergence starts at a specific length: inspect cursor update, page/block boundary, or RoPE offset
- only long prompts diverge: inspect cache capacity, truncation, or position indexing
- only sampled text regresses: inspect sampling, not K/V
- perplexity changes but greedy sample looks okay: inspect distribution drift and top-k overlap

### Phase 4: Sampling And Text Handling Cleanup

Problem:

Sampling currently returns Python ints and stop handling decodes growing token lists repeatedly. This adds CPU sync and O(T^2) text work.

Target change:

- reduce avoidable `.item()` synchronizations
- avoid repeated full decode for stop checks where possible
- keep streaming deltas correct
- avoid constructing fresh tensors in hot paths where practical

Expected benefit:

- lower per-token CPU overhead
- better streaming behavior
- more stable latency

Quality risk:

- medium

Required gates:

- logits parity before sampling
- sampling mask parity
- EOS behavior tests
- stop-string behavior tests
- streaming delta tests
- fixed-seed sampling smoke if RNG control is available

Failure diagnosis:

- logits match but output changes under greedy: inspect argmax and token conversion
- logits match but sampled outputs degrade: inspect top-k, top-p, repetition penalty, dtype, and RNG
- streaming text duplicates or drops characters: inspect delta construction

### Phase 5: Cached Fast AttnRes Path

Problem:

The server warms a custom fast AttnRes path, but served cached generation does not use it. After KV cache is fixed, AttnRes routing may become the dominant decode cost.

Target change:

- profile cached decode after Phase 2
- determine whether `_route_static` phase-1 Triton is enough
- if needed, implement a decode-specialized AttnRes route using the existing phase-2 merge idea

Expected benefit:

- lower per-token model forward time if routing dominates

Quality risk:

- high

Required gates:

- hidden-state parity at each layer or selected layer checkpoints
- final logits parity
- greedy exact match
- perplexity smoke
- long-prompt tests

Failure diagnosis:

- hidden states diverge before attention: inspect AttnRes route input assembly
- hidden states diverge after boundary layer: inspect committed/partial block state
- final logits diverge with close hidden states: inspect final aggregation or norm dtype

### Phase 6: Batching And Concurrency

Problem:

The current server is synchronous per request. After single-stream generation is efficient, throughput under concurrent workloads will require scheduling.

Target change:

- add a request queue or microbatching layer
- batch prefill where practical
- batch decode for active requests with compatible shapes
- preserve per-request stop conditions and streaming output

Expected benefit:

- improved GPU utilization under concurrent serving

Quality risk:

- medium to high

Required gates:

- single-request parity still passes
- batched vs unbatched logits parity
- per-request cache isolation test
- mixed-length request test
- early-stop request mixed with long request test
- streaming ordering test

Failure diagnosis:

- only concurrent requests fail: inspect request/cache indexing
- short request affects long request: inspect cache reuse and reset
- streaming outputs interleave incorrectly: inspect response routing, not model logits

### Phase 7: vLLM Spike Decision

Do this only after Phases 1 and 2 establish a strong custom baseline.

Spike questions:

- Can Luxia load and generate correctly through vLLM at all?
- Can vLLM manage the standard token-attention KV cache while preserving AttnRes behavior?
- Does the Transformers backend work, and if so, how slow is it relative to custom serving?
- Is a native vLLM model port required?
- Does vLLM batching beat the optimized custom server for the real workload?

Decision criteria:

- If single-stream eval generation is the main workload and custom serving is fast enough, defer vLLM.
- If concurrency, OpenAI compatibility, prefix caching, production metrics, or multi-user throughput matter, invest in a native vLLM port.
- Do not use a slow Transformers-backend vLLM port as the final target unless it already beats the custom server in the intended workload.

## Regression Report Template

Every optimization should produce a small report.

Required fields:

- change name
- git commit or patch identifier
- benchmark command
- prompt set
- hardware
- dtype
- compile setting
- baseline result path
- optimized result path
- prefill latency delta
- decode latency delta
- total throughput delta
- peak memory delta
- greedy exact-match status
- logits parity summary
- perplexity delta
- profiler top changes
- known risks
- decision: keep, revise, or revert

Example table:

| Metric | Baseline | Optimized | Delta |
|---|---:|---:|---:|
| TTFT, 128 prompt | | | |
| Decode ms/token, 256 gen | | | |
| Total tok/s | | | |
| Peak memory GB | | | |
| Max logit diff | | | |
| Mean KL | | | |
| Top-1 match | | | |
| PPL smoke | | | |

## Stop Conditions

Pause optimization and debug before continuing if any of these happen:

- greedy exact-match fails unexpectedly
- top-1 logit match drops below 100 percent for deterministic parity tests
- top-10 overlap drops materially
- KL divergence increases beyond established baseline numerical noise
- perplexity moves meaningfully
- EOS rate changes materially
- completion length distribution changes materially
- repeated text rate increases materially
- profiler shows a new large copy or allocation in the decode loop
- compile introduces graph breaks or cold compiles during request handling

## Suggested Implementation Order (Revised After Phase 0)

1. ~~Add benchmark harness and fixed prompt pack.~~ DONE.
2. ~~Add logit parity tests for full-forward vs cached-forward behavior.~~ DONE.
3. ~~Add phase timing to `serve.py`.~~ DONE.
4. ~~Capture baseline results.~~ DONE.
5. ~~Identify and fix cuDNN SDPA overhead.~~ DONE. Flash backend + warmup.
6. ~~Make `torch.compile` work for cached AttnRes decode (Phase 1). Expected 2.3x.~~ DONE. Eager prefill + compiled decode. 2.0-2.5x decode speedup.
7. ~~Re-run all gates and benchmark.~~ DONE. 224/224 top-1 parity, benchmark at 100-117 tok/s.
8. Implement last-logits-only prefill (Phase 2). Small win.
9. Implement static/preallocated KV cache (Phase 3). Matters more at 3B.
10. Re-run all gates and benchmark.
11. Clean up sampling and text handling (Phase 4).
12. Profile AttnRes routing and decide on cached fast path (Phase 5).
13. Consider batching only after single-stream behavior is stable (Phase 6).

## Expected Outcome

The near-term goal is not just higher tokens/sec. The goal is an optimized serving path where:

- performance changes are measured
- output distribution changes are caught before text review
- quality regressions are debuggable to a specific layer, cache position, routing step, or sampling step
- every accepted optimization has a before/after report
- future vLLM work has a strong custom-server baseline to compare against
