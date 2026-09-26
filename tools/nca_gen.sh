#!/bin/bash
# Generate an NCA pre-pretraining token stream, split train/eval, verify.
# Defaults = the 3B recipe of record (3500 rules x 20 sims, seed 17, ~9.2B
# tokens -> 9B train + ~200M eval). Output lands in $KOTODAMA_DATA_ROOT.
#
# Usage: tools/nca_gen.sh [--name nca_3b_seed17] [--tokens 9200000000]
#          [--train-tokens 9000000000] [--rules 3500] [--sims 20] [--seed 17]
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_env.sh"

NAME=nca_3b_seed17; TOKENS=9200000000; TRAIN_TOKENS=9000000000
RULES=3500; SIMS=20; SEED=17; DEVICE=cuda:0
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --tokens) TOKENS="$2"; shift 2 ;;
    --train-tokens) TRAIN_TOKENS="$2"; shift 2 ;;
    --rules) RULES="$2"; shift 2 ;;
    --sims) SIMS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUT="$KOTODAMA_DATA_ROOT"; mkdir -p "$OUT"
FULL="$OUT/${NAME}_full.bin"; TRAIN="$OUT/${NAME}_train.bin"; EVAL="$OUT/${NAME}_eval.bin"
for f in "$TRAIN" "$EVAL"; do
  [ -e "$f" ] && { echo "refusing to overwrite $f" >&2; exit 1; }
done

echo "=== generate $TOKENS tokens ($RULES rules x $SIMS sims, seed $SEED) -> $FULL"
python -m kotodama.nca.generator --output "$FULL" --tokens "$TOKENS" \
  --num_rules "$RULES" --sims_per_rule "$SIMS" --seed "$SEED" --device "$DEVICE"
echo "=== split -> $TRAIN ($TRAIN_TOKENS) + $EVAL"
python tools/split_nca_data.py --input "$FULL" --train "$TRAIN" --eval "$EVAL" \
  --train_tokens "$TRAIN_TOKENS"
echo "=== verify"
python -m kotodama.nca.generator --verify "$TRAIN"
python -m kotodama.nca.generator --verify "$EVAL"
rm "$FULL"
echo "=== done: $TRAIN $EVAL"
