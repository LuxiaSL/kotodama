#!/bin/bash
# Wrapper for the kotodama inference gateway on cluster nodes.
# The gateway supervises a fleet of serve.py replicas and exposes one
# OpenAI-compatible port that routes/load-balances across them.
#
# Usage:
#   tools/run_gateway.sh --config configs/gateway.yaml
#   tools/run_gateway.sh --config configs/gateway.yaml --gpus 0,1   # subset (test)
set -e
ROOT="${KOTODAMA_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

VENV="${KOTODAMA_VENV:-$ROOT/.venv}"
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONUNBUFFERED=1

# The gateway itself is a lightweight async proxy; the replicas it spawns set
# their own threading/alloc env (see serve.py). HF_HOME is passed to replicas
# via the config.
exec "${KOTODAMA_PYTHON:-python}" gateway.py "$@"
