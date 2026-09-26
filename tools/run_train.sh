#!/bin/bash
# Distributed training (NCA pre-pretraining, language pretraining, CPT): torchrun
# over the one trainer. GPUs per node: KOTODAMA_NPROC (default 8).
# Usage: tools/run_train.sh --config configs/<run>.yaml [overrides...]
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1  # persist FA4 JIT kernels to disk
exec torchrun --nproc_per_node="${KOTODAMA_NPROC:-8}" -m kotodama.training.train "$@"
