#!/bin/bash
# Wrapper for luxia-base training jobs on cluster nodes.
# Activates shared venv, sets env, runs torchrun.
# Usage: tools/run_train.sh --config configs/proxy-benchmark.yaml [extra args...]
set -e
ROOT="${KOTODAMA_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
# KOTODAMA_VENV overrides for experiment venvs (e.g. the MXFP8 throwaway
# venv at /models/kotodama-data/venv-mxfp8 — never modify .venv-shared).
VENV="${KOTODAMA_VENV:-$ROOT/.venv}"
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export OMP_NUM_THREADS=16
export CPATH="${KOTODAMA_CPATH:-${CPATH:-}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1  # persist FA4 JIT kernels to disk

# Persist torch.compile / Triton caches across restarts — otherwise every
# crash/resume pays full Inductor+Triton recompilation on all 8 ranks
# (minutes of wall-clock each time; the 3B run lost ~20% to downtime).
CACHE_ROOT="${KOTODAMA_CACHE_ROOT:-$ROOT/.cache/compile}"
if mkdir -p "$CACHE_ROOT" 2>/dev/null; then
  export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
  export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
else
  echo "WARN: $CACHE_ROOT not writable — compile caches will not persist" >&2
fi

exec torchrun --nproc_per_node=8 -m src.training.train "$@"
