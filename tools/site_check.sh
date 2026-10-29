#!/bin/bash
# Check this machine's site profile (site.env) the way every launcher sees it: source tools/_env.sh, then verify each
# configured path exists and each service answers. One line per check, "ok" or "FAIL"; exit 1 if any check fails.
# Secrets are checked for readability only, never printed.
#
#   tools/site_check.sh
set -uo pipefail
# shellcheck disable=SC1091
source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"

FAILS=0
ok()   { printf 'ok    %-26s %s\n' "$1" "$2"; }
fail() { printf 'FAIL  %-26s %s\n' "$1" "$2"; FAILS=$((FAILS + 1)); }
skip() { printf '-     %-26s %s\n' "$1" "unset"; }

PY="$(command -v python || true)"
if [ -n "$PY" ] && v=$("$PY" -c 'import torch, sys; print(sys.version.split()[0], "torch", torch.__version__)' 2>/dev/null); then
  ok python "$PY ($v)"
else
  fail python "no python with torch on PATH (KOTODAMA_VENV=${KOTODAMA_VENV:-unset})"
fi

# Triton builds a small C helper at first launch and needs Python.h on CPATH or in the interpreter's include dir.
inc=""
[ -n "$PY" ] && inc=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])' 2>/dev/null)
found=""
IFS=: read -r -a dirs <<< "${CPATH:-}"
for d in "${dirs[@]}" "$inc"; do
  [ -n "$d" ] && [ -f "$d/Python.h" ] && { found="$d"; break; }
done
for d in "${dirs[@]}"; do
  [ -n "$d" ] && [ ! -d "$d" ] && fail KOTODAMA_CPATH "$d does not exist"
done
[ -n "$found" ] && ok Python.h "$found" || fail Python.h "not in CPATH (${CPATH:-empty}) or $inc"

for var in KOTODAMA_DATA_ROOT HF_HOME KOTODAMA_COMPILE_STORE; do
  p="${!var:-}"
  if [ -z "$p" ]; then skip "$var"; elif [ -d "$p" ]; then ok "$var" "$p"; else fail "$var" "$p is not a directory"; fi
done
c="${KOTODAMA_CACHE_ROOT:-}"
if [ -z "$c" ]; then skip KOTODAMA_CACHE_ROOT
elif [ -w "$c" ]; then ok KOTODAMA_CACHE_ROOT "$c"
else fail KOTODAMA_CACHE_ROOT "$c is not writable"; fi

r="${KOTODAMA_CKPT_REGISTRY:-}"
if [ -z "$r" ]; then skip KOTODAMA_CKPT_REGISTRY
elif [ -r "$r" ]; then ok KOTODAMA_CKPT_REGISTRY "$r"
else fail KOTODAMA_CKPT_REGISTRY "$r is not readable"; fi

t="${HF_TOKEN_PATH:-}"
if [ -z "$t" ]; then skip HF_TOKEN_PATH
elif [ -r "$t" ] && [ -s "$t" ]; then ok HF_TOKEN_PATH "$t (readable, not shown)"
else fail HF_TOKEN_PATH "$t is missing or empty"; fi

[ -n "${HF_HUB_DISABLE_XET:-}" ] && ok HF_HUB_DISABLE_XET "${HF_HUB_DISABLE_XET}" || skip HF_HUB_DISABLE_XET

u="${KOTODAMA_SCHEDULER_URL:-}"
if [ -z "$u" ]; then skip KOTODAMA_SCHEDULER_URL
elif code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$u/" 2>/dev/null) && [ "$code" != 000 ]; then
  ok KOTODAMA_SCHEDULER_URL "$u (HTTP $code)"
else
  fail KOTODAMA_SCHEDULER_URL "$u did not answer"
fi

if [ "$FAILS" -gt 0 ]; then echo "site check: $FAILS FAILED"; exit 1; fi
echo "site check: all ok"
