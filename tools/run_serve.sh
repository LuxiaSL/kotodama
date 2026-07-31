#!/bin/bash
# Wrapper for kotodama inference serving on cluster nodes.
# Activates shared venv, sets env, runs serve.py.
#
# Usage:
#   tools/run_serve.sh --checkpoint /path/to/model.pt [--prefix-cache] [--port 2224]
#   tools/run_serve.sh --checkpoint /path/to/instruct.pt --mode chat
#
set -e
ROOT="${KOTODAMA_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

VENV="${KOTODAMA_VENV:-$ROOT/.venv}"
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONUNBUFFERED=1

# 2 threads optimal for single-request GPU inference (see serve.py)
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Reduce CUDA allocator fragmentation under variable-length decode.
# (serve.py also setdefault()s this, so direct launches get it too.)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

exec "${KOTODAMA_PYTHON:-python}" serve.py "$@"
