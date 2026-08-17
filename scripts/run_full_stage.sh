#!/usr/bin/env bash
set -euo pipefail

ROOT="${AIBLINK_ROOT:-/root/ai_blink}"
DATA="${AIBLINK_DATA:-/data/ai_blink}"
STEPS="${AIBLINK_STEPS:-4500}"
MANIFEST="${AIBLINK_MANIFEST:-$DATA/full/manifest.csv}"
RUN_ROOT="${AIBLINK_RUN_ROOT:-$DATA/full/runs}"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$DATA/full/logs"

candidates=(commfor_384 sieve_ft1 commfor_224)
for candidate in "${candidates[@]}"; do
  initial="$DATA/tournament/runs/$candidate/pilot.pt"
  output="$RUN_ROOT/$candidate"
  progress="$output/progress.json"
  mkdir -p "$output"

  if [[ -f "$progress" ]] && "$ROOT/.venv/bin/python" -c \
    'import json,sys; raise SystemExit(0 if int(json.load(open(sys.argv[1]))["step"]) >= int(sys.argv[2]) else 1)' \
    "$progress" "$STEPS"; then
    echo "$candidate already completed $STEPS steps"
    continue
  fi

  resume=()
  if [[ -f "$output/last.pt" ]]; then
    resume=(--resume "$output/last.pt")
  fi
  echo "starting $candidate at $(date --iso-8601=seconds)"
  "$ROOT/.venv/bin/aiblink-val" train-full \
    --manifest "$MANIFEST" \
    --initial-checkpoint "$initial" \
    --steps "$STEPS" \
    --batch-size 96 \
    --lr 2e-5 \
    --head-multiplier 1.0 \
    --weight-decay 0.05 \
    --warmup-steps 300 \
    --workers 4 \
    --compile-mode reduce-overhead \
    --validation-every 750 \
    --checkpoint-every 250 \
    --validation-cap 2000 \
    --ema-decay 0.999 \
    --source-sampling-exponent 0.5 \
    --threshold 0.65 \
    --out "$output" \
    "${resume[@]}" \
    2>&1 | tee -a "$DATA/full/logs/$candidate.log"
done
