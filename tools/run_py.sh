#!/bin/bash
# Wrapper for non-torchrun utility jobs (parity tests, micro-benches, analysis)
# on cluster nodes. Same env as run_train.sh: shared venv, unbuffered output,
# persistent Triton/Inductor caches — bare `python` submits JIT cold and
# buffer stdout until exit (invisible in scheduler logs).
# Usage: tools/run_py.sh scripts/utils/phase2_bwd_parity.py [args...]
set -e
ROOT="${KOTODAMA_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
# KOTODAMA_VENV overrides for experiment venvs (never modify .venv-shared).
VENV="${KOTODAMA_VENV:-$ROOT/.venv}"
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export CPATH="${KOTODAMA_CPATH:-${CPATH:-}}"

CACHE_ROOT="${KOTODAMA_CACHE_ROOT:-$ROOT/.cache/compile}"
if mkdir -p "$CACHE_ROOT" 2>/dev/null; then
  export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
  export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
fi

exec python -u "$@"
