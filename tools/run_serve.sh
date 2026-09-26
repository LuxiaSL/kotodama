#!/bin/bash
# One model server replica (OpenAI-compatible + /generate + /logprobs).
# Usage: tools/run_serve.sh --checkpoint <ckpt.pt[.zst]> [--mode chat] [--port 2224] [--prefix-cache]
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"
# 2 threads is optimal for single-request GPU decode.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec python -m kotodama.serve.server "$@"
