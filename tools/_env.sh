# Shared launcher environment. Source it; do not execute it.
#
#   source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"
#
# Resolves the repo root, loads the machine's site profile ($ROOT/site.env,
# uncommitted — copy site.env.example), activates the venv, and exports the
# package path and persistent compile caches. Every launcher in tools/ goes
# through here, so one machine's paths live in exactly one file.

ROOT="${KOTODAMA_WORKDIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ -f "$ROOT/site.env" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/site.env"
fi
cd "$ROOT"

VENV="${KOTODAMA_VENV:-$ROOT/.venv}"
if [ -x "$VENV/bin/python" ]; then
  export PATH="$VENV/bin:$PATH"
  export VIRTUAL_ENV="$VENV"
elif [ -n "${KOTODAMA_VENV:-}" ]; then
  echo "ERROR: KOTODAMA_VENV=$KOTODAMA_VENV has no bin/python" >&2
  exit 1
fi  # else: no venv configured -> whatever python is already on PATH
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CPATH="${KOTODAMA_CPATH:-${CPATH:-}}"

# Scheduler integration: train.py reads KOTODAMA_SCHEDULER_JOB_ID. A site whose
# scheduler exports its job id under another name maps it here.
if [ -z "${KOTODAMA_SCHEDULER_JOB_ID:-}" ] && [ -n "${KOTODAMA_SCHEDULER_JOB_ID_VAR:-}" ]; then
  export KOTODAMA_SCHEDULER_JOB_ID="${!KOTODAMA_SCHEDULER_JOB_ID_VAR:-}"
fi

# Persist torch.compile / Triton caches across restarts (a cold recompile on
# all ranks costs minutes per resume). Never leave them in ~/.triton: on a
# small root filesystem the JIT cache alone can fill it.
CACHE_ROOT="${KOTODAMA_CACHE_ROOT:-$ROOT/.cache/compile}"
if mkdir -p "$CACHE_ROOT" 2>/dev/null; then
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/inductor}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
else
  echo "WARN: $CACHE_ROOT not writable — compile caches will not persist" >&2
fi

KOTODAMA_DATA_ROOT="${KOTODAMA_DATA_ROOT:-$ROOT/data}"
export KOTODAMA_DATA_ROOT
