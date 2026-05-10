#!/bin/bash
# Wrapper for kotodama inference serving on cluster nodes.
# Activates shared venv, sets env, runs serve.py.
#
# Usage:
#   tools/run_serve.sh --model kotodama-108m-instruct-fc [--compile] [--port 2224]
#   tools/run_serve.sh --checkpoint /path/to/custom.pt --mode chat
#
# Scheduler:
#   scheduler submit "tools/run_serve.sh --model kotodama-108m-instruct-fc --compile" \
#     --name kotodama-serve --gpus 1 --node gpu-host --always-on \
#     --workdir ~/workspace/kotodama --port 2224
set -e
cd ~/workspace/kotodama

VENV=~/workspace/.venv-shared
export PATH="$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
export PYTHONUNBUFFERED=1

# 2 threads optimal for single-request GPU inference (see serve.py)
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

exec python3 serve.py "$@"
