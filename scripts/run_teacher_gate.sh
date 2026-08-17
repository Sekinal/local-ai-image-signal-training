#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/ai_blink
DATA=/data/ai_blink
TRAIN="$DATA/frontier_training"
WARM="$DATA/frontier_warmstarts"
CHALLENGERS="$DATA/frontier_challengers"
OUT="$DATA/frontier_teacher_gate"
TEACHER=convnext_large_teacher_scratch
SUBSET="$DATA/tournament/subset/manifest.csv"
mkdir -p "$OUT/smoke" "$OUT/warmstart" "$OUT/run" "$OUT/evals"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "teacher-superiority pipeline started $(date --iso-8601=seconds)"
while [[ -f "$DATA/frontier_warmstarts.pid" ]] && kill -0 "$(cat "$DATA/frontier_warmstarts.pid")" 2>/dev/null; do
  echo "waiting for current warmstarts $(date --iso-8601=seconds)"
  sleep 30
done

exec 9>"$DATA/frontier_gpu.lock"
echo "waiting for GPU lease for teacher capacity smoke and warmstart $(date --iso-8601=seconds)"
flock -x 9
echo "acquired GPU lease for teacher capacity smoke and warmstart $(date --iso-8601=seconds)"

if [[ ! -f "$OUT/chosen_batch.txt" ]]; then
  chosen=""
  for batch in 24 16 12 8; do
    if "$ROOT/.venv/bin/aiblink-val" train-pilot \
      --manifest "$SUBSET" --candidate "$TEACHER" --steps 1 --batch-size "$batch" \
      --lr 1e-4 --head-multiplier 1 --workers 2 --compile-mode reduce-overhead \
      --out "$OUT/smoke/batch-$batch"; then
      chosen="$batch"
      break
    fi
    echo "teacher batch $batch did not fit; trying a smaller batch"
  done
  if [[ -z "$chosen" ]]; then
    echo "no safe ConvNeXt-Large training batch fit the L40S" >&2
    exit 1
  fi
  printf '%s\n' "$chosen" > "$OUT/chosen_batch.txt"
fi
batch=$(cat "$OUT/chosen_batch.txt")
case "$batch" in
  24) lr=1.5e-4 ;;
  16) lr=1.0e-4 ;;
  12) lr=7.5e-5 ;;
  8) lr=5.0e-5 ;;
  *) echo "unexpected chosen teacher batch $batch" >&2; exit 1 ;;
esac

# Match the scratch Small warmstart's 192,000 examples exactly.
pilot_steps=$((192000 / batch))
if [[ ! -f "$OUT/warmstart/pilot.pt" ]]; then
  "$ROOT/.venv/bin/aiblink-val" train-pilot \
    --manifest "$SUBSET" --candidate "$TEACHER" --steps "$pilot_steps" --batch-size "$batch" \
    --lr "$lr" --head-multiplier 1 --weight-decay 0.05 --workers 4 \
    --seed 323 --compile-mode reduce-overhead --out "$OUT/warmstart"
fi
flock -u 9
echo "teacher warmstart complete; released GPU lease $(date --iso-8601=seconds)"

while [[ ! -f "$CHALLENGERS/ranking.json" || ! -f "$TRAIN/runs/commfor_384/training.json" ]]; do
  echo "waiting for completed small-model comparison $(date --iso-8601=seconds)"
  sleep 60
done

echo "waiting for GPU lease for controlled full teacher run $(date --iso-8601=seconds)"
flock -x 9
echo "acquired GPU lease for controlled full teacher run $(date --iso-8601=seconds)"

# Match scratch Small's 1,152,000 full-training examples exactly.
full_steps=$((1152000 / batch))
validation_every=$((full_steps / 24))
resume=()
if [[ -f "$OUT/run/last.pt" ]]; then
  resume=(--resume "$OUT/run/last.pt")
fi
if [[ ! -f "$OUT/run/training.json" ]]; then
  "$ROOT/.venv/bin/aiblink-val" train-full \
    --manifest "$TRAIN/manifest.csv" --initial-checkpoint "$OUT/warmstart/pilot.pt" \
    --steps "$full_steps" --batch-size "$batch" --validation-batch-size "$batch" \
    --lr "$lr" --head-multiplier 1 --weight-decay 0.05 \
    --warmup-steps "$validation_every" --workers 4 --seed 323 --compile-mode reduce-overhead \
    --validation-every "$validation_every" --checkpoint-every "$validation_every" \
    --validation-cap 2000 --ema-decay 0.9995 --source-sampling-exponent 0.35 \
    --threshold 0.65 --out "$OUT/run" "${resume[@]}"
fi

"$ROOT/.venv/bin/aiblink-val" score-trained \
  --manifest "$TRAIN/manifest.csv" --checkpoint "$OUT/run/best.pt" --role calibration \
  --views clean,web,hard --batch-size "$batch" --workers 4 \
  --out "$OUT/evals/$TEACHER.jsonl"

"$ROOT/.venv/bin/aiblink-val" rank-tournament \
  --manifest "$TRAIN/manifest.csv" \
  --predictions \
    "$CHALLENGERS/evals/calibration/commfor_384.jsonl" \
    "$CHALLENGERS/evals/calibration/convnext_small_pretrained.jsonl" \
    "$CHALLENGERS/evals/calibration/convnext_small_scratch.jsonl" \
    "$CHALLENGERS/evals/calibration/forensic_convnext_scratch.jsonl" \
    "$OUT/evals/$TEACHER.jsonl" \
  --threshold 0.65 --out "$OUT/ranking_with_teacher.json"

"$ROOT/.venv/bin/python" "$ROOT/scripts/validate_teacher_gate.py" \
  --manifest "$TRAIN/manifest.csv" --ranking "$OUT/ranking_with_teacher.json" \
  --teacher-predictions "$OUT/evals/$TEACHER.jsonl" --teacher-training "$OUT/run/training.json" \
  --student-prediction "commfor_384=$CHALLENGERS/evals/calibration/commfor_384.jsonl" \
  --student-prediction "convnext_small_scratch=$CHALLENGERS/evals/calibration/convnext_small_scratch.jsonl" \
  --student-prediction "forensic_convnext_scratch=$CHALLENGERS/evals/calibration/forensic_convnext_scratch.jsonl" \
  --student-training "commfor_384=$TRAIN/runs/commfor_384/training.json" \
  --student-training "convnext_small_scratch=$CHALLENGERS/runs/convnext_small_scratch/training.json" \
  --student-training "forensic_convnext_scratch=$CHALLENGERS/runs/forensic_convnext_scratch/training.json" \
  --minimum-oof-ba-delta 0.01 --maximum-view-auc-regression 0.005 \
  --minimum-stable-validations 2 --bootstrap-replicates 2000 \
  --out "$OUT/gate.json"

flock -u 9
echo "teacher-superiority pipeline completed $(date --iso-8601=seconds)"
