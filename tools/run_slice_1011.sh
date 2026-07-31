#!/usr/bin/env bash
# PROBES §10.11 benchmark-transfer slice (round-3 session, 2026-07-04).
# 9 ckpts (ck0 + round posterboys + r3 gradient) x 4 tasks
# (lambada_openai,winogrande,sciq,wikitext) on the FROZEN v2 harness config.
# GPUs 4,5,6 (6 shared with the :2399 chat replica — B200 has headroom).
# Per-model output dirs (step-stem collisions across rounds).
set -uo pipefail
cd ~/workspace/kotodama
PY=~/workspace/.venv-shared/bin/python
export KOTODAMA_NO_TRITON_ATTNRES=1 HF_HOME=/models/huggingface
export TMPDIR=/models/kotodama-data/tmp
TASKS=lambada_openai,winogrande,sciq,wikitext
DPO=/models/kotodama-data/dpo
mkdir -p logs

run() { # gpu tag ckpt
  CUDA_VISIBLE_DEVICES=$1 $PY -m scripts.eval.run_lm_eval \
    --checkpoint "$3" --config-section model \
    --attn-res-boundaries 0,1,3,7,15,19,24 --batch-size 32 \
    --tasks $TASKS --output-dir analysis/lm_eval_v2/slice_$2 \
    > logs/lmeval_slice_$2.log 2>&1 \
    && echo "OK  $2" || echo "FAIL $2 (logs/lmeval_slice_$2.log)"
}

( run 4 ck0    /models/kotodama-data/tmp/kotodama_checkpoints/3b-language-FINAL-step195311.pt
  run 4 r3s12  $DPO/round3-961/checkpoints/step_00000012.pt
  run 4 r3s36  $DPO/round3-961/checkpoints/step_00000036.pt ) &
( run 5 r1s5   $DPO/round1-base-primary/checkpoints/step_00000005.pt
  run 5 r2s35  $DPO/round2-base-mixed/checkpoints/step_00000035.pt
  run 5 r3s24  $DPO/round3-961/checkpoints/step_00000024.pt ) &
( run 6 r2bs35 $DPO/round2b-lr5e5/checkpoints/step_00000035.pt
  run 6 r3s48  $DPO/round3-961/checkpoints/step_00000048.pt
  run 6 r3s60  $DPO/round3-961/checkpoints/step_00000060.pt ) &
wait
echo "SLICE-1011 COMPLETE"
