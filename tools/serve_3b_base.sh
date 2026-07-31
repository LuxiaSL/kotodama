#!/usr/bin/env bash
# Download (once) and serve the public Kotodama 3B base checkpoint.
set -euo pipefail

ROOT="${KOTODAMA_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"

MODEL_ID="${KOTODAMA_MODEL_ID:-aethera-gp/kotodama-3b-base-final}"
CHECKPOINT_NAME="${KOTODAMA_CHECKPOINT_NAME:-step_00195311.pt.zst}"
MODEL_DIR="${KOTODAMA_MODEL_DIR:-$ROOT/models/kotodama-3b-base-final}"
CHECKPOINT_PATH="$MODEL_DIR/$CHECKPOINT_NAME"
ENGINE="${KOTODAMA_ENGINE:-fast}"
MAX_SEQ_LEN="${KOTODAMA_MAX_SEQ_LEN:-4096}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.cache/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.cache/triton}"
mkdir -p "$MODEL_DIR" "$HF_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Downloading $MODEL_ID/$CHECKPOINT_NAME to $MODEL_DIR" >&2
  hf download "$MODEL_ID" "$CHECKPOINT_NAME" --local-dir "$MODEL_DIR"
fi

exec "${KOTODAMA_PYTHON:-python}" serve.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --model_size 3b \
  --mode base \
  --engine "$ENGINE" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --served_model_name kotodama-3b-base-final \
  --host "${KOTODAMA_HOST:-0.0.0.0}" \
  --port "${KOTODAMA_PORT:-2222}" \
  "$@"
