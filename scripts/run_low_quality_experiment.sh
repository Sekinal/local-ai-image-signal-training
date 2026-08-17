#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/ai_blink
DATA=/data/ai_blink
MANIFEST="$DATA/frontier_training/manifest.csv"
INITIAL="$DATA/final_commfor_20260816/run/best.pt"
OUT="$DATA/low_quality_experiment_20260816"
CONFIG="$ROOT/configs/low_quality_canary.yaml"
mkdir -p "$OUT/canary" "$OUT/scaled" "$OUT/evals" "$OUT/reports"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "low-quality experiment started $(date --iso-8601=seconds)"
exec 9>"$DATA/frontier_gpu.lock"
flock -x 9

if [[ ! -s "$OUT/evals/baseline_calibration.jsonl" ]]; then
  "$ROOT/.venv/bin/aiblink-val" score-trained \
    --manifest "$MANIFEST" --checkpoint "$INITIAL" --role calibration \
    --attack-config "$CONFIG" --profile gate --batch-size 96 --workers 4 \
    --out "$OUT/evals/baseline_calibration.jsonl"
fi

if [[ ! -s "$OUT/canary/training.json" ]]; then
  "$ROOT/.venv/bin/aiblink-val" train-full \
    --manifest "$MANIFEST" --initial-checkpoint "$INITIAL" \
    --steps 400 --batch-size 96 --validation-batch-size 96 \
    --lr 3e-6 --head-multiplier 1 --weight-decay 0.05 --warmup-steps 40 \
    --workers 4 --seed 1323 --compile-mode reduce-overhead \
    --validation-every 200 --checkpoint-every 200 --validation-cap 1000 \
    --ema-decay 0.99 --source-sampling-exponent 0.35 --threshold 0.65 \
    --augmentation-profile low-quality --out "$OUT/canary"
fi

if [[ ! -s "$OUT/evals/canary_calibration.jsonl" ]]; then
  "$ROOT/.venv/bin/aiblink-val" score-trained \
    --manifest "$MANIFEST" --checkpoint "$OUT/canary/last.pt" --role calibration \
    --attack-config "$CONFIG" --profile gate --batch-size 96 --workers 4 \
    --out "$OUT/evals/canary_calibration.jsonl"
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/analyze_low_quality_ablation.py" \
  --manifest "$MANIFEST" --baseline "$OUT/evals/baseline_calibration.jsonl" \
  --candidate "$OUT/evals/canary_calibration.jsonl" --stage canary \
  --out "$OUT/reports/canary.json"

if ! jq -e '.promote == true' "$OUT/reports/canary.json" >/dev/null; then
  echo "canary did not pass the frozen promotion gate; scaled training will not start"
  flock -u 9
  exit 0
fi

echo "canary passed; starting scaled low-quality training $(date --iso-8601=seconds)"
if [[ ! -s "$OUT/scaled/training.json" ]]; then
  "$ROOT/.venv/bin/aiblink-val" train-full \
    --manifest "$MANIFEST" --initial-checkpoint "$INITIAL" \
    --steps 2400 --batch-size 96 --validation-batch-size 96 \
    --lr 3e-6 --head-multiplier 1 --weight-decay 0.05 --warmup-steps 120 \
    --workers 4 --seed 2323 --compile-mode reduce-overhead \
    --validation-every 400 --checkpoint-every 400 --validation-cap 3912 \
    --ema-decay 0.999 --source-sampling-exponent 0.35 --threshold 0.65 \
    --augmentation-profile low-quality --out "$OUT/scaled"
fi

if [[ ! -s "$OUT/evals/scaled_calibration.jsonl" ]]; then
  "$ROOT/.venv/bin/aiblink-val" score-trained \
    --manifest "$MANIFEST" --checkpoint "$OUT/scaled/last.pt" --role calibration \
    --attack-config "$CONFIG" --profile gate --batch-size 96 --workers 4 \
    --out "$OUT/evals/scaled_calibration.jsonl"
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/analyze_low_quality_ablation.py" \
  --manifest "$MANIFEST" --baseline "$OUT/evals/baseline_calibration.jsonl" \
  --candidate "$OUT/evals/scaled_calibration.jsonl" --stage scaled \
  --out "$OUT/reports/scaled.json"

flock -u 9
echo "low-quality experiment completed $(date --iso-8601=seconds)"
