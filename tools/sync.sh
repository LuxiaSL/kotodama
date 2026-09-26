#!/bin/bash
# Mirror this repo's TRACKED files to a remote checkout (a machine without git):
# exactly `git ls-files`, extras deleted — except the remote's own site.env,
# .venv*, caches and __pycache__, which are protected.
#
# Usage: tools/sync.sh [host:/path]      (default: $KOTODAMA_SYNC_TARGET from site.env)
#        tools/sync.sh -n [host:/path]   (dry run)
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/site.env" ] && source "$ROOT/site.env"
DRY=(); [ "${1:-}" = "-n" ] && { DRY=(--dry-run --itemize-changes); shift; }
TARGET="${1:-${KOTODAMA_SYNC_TARGET:-}}"
[ -n "$TARGET" ] || { echo "usage: tools/sync.sh [-n] host:/path (or set KOTODAMA_SYNC_TARGET)" >&2; exit 2; }
cd "$ROOT"
[ -z "$(git status --porcelain --untracked-files=no)" ] || echo "WARN: uncommitted changes are synced as-is" >&2
LIST="$(mktemp)"; trap 'rm -f "$LIST"' EXIT
git ls-files > "$LIST"
rsync -a --delete --delete-excluded "${DRY[@]}" \
  --filter='P site.env' --filter='P .venv*/' --filter='P .cache/' --filter='P __pycache__/' --filter='P logs/' \
  --include-from=<(sed 's|^|/|' "$LIST"; awk -F/ '{p=""; for(i=1;i<NF;i++){p=p"/"$i; print p"/"}}' "$LIST" | sort -u) \
  --exclude='*' ./ "$TARGET/"
echo "synced $(wc -l < "$LIST") tracked files -> $TARGET"
