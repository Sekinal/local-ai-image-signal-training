#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/ai_blink
DATA=/data/ai_blink
TRAIN="$DATA/frontier_training"
WARM="$DATA/frontier_warmstarts"
OUT="$DATA/frontier_challengers"
mkdir -p "$OUT/runs" "$OUT/evals/calibration" "$OUT/evals/openrouter" "$OUT/evals/recent_holdout"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "challenger pipeline started $(date --iso-8601=seconds)"
while [[ ! -f "$TRAIN/manifest.csv" || ! -f "$TRAIN/runs/commfor_384/training.json" ]]; do
  echo "waiting for audited frontier manifest and incumbent $(date --iso-8601=seconds)"
  sleep 30
done
for candidate in convnext_small_pretrained convnext_small_scratch forensic_convnext_scratch; do
  while [[ ! -f "$WARM/$candidate/pilot.pt" ]]; do
    echo "waiting for $candidate warmstart $(date --iso-8601=seconds)"
    sleep 30
  done
done

exec 9>"$DATA/frontier_gpu.lock"
echo "waiting for exclusive GPU lease $(date --iso-8601=seconds)"
flock -x 9
echo "acquired exclusive GPU lease $(date --iso-8601=seconds)"

train_candidate() {
  local candidate="$1"
  local steps="$2"
  local batch="$3"
  local lr="$4"
  local warmup="$5"
  local ema="$6"
  local run="$OUT/runs/$candidate"
  mkdir -p "$run"
  resume=()
  if [[ -f "$run/last.pt" ]]; then
    resume=(--resume "$run/last.pt")
  fi
  "$ROOT/.venv/bin/aiblink-val" train-full \
    --manifest "$TRAIN/manifest.csv" \
    --initial-checkpoint "$WARM/$candidate/pilot.pt" \
    --steps "$steps" \
    --batch-size "$batch" \
    --lr "$lr" \
    --head-multiplier 1 \
    --weight-decay 0.05 \
    --warmup-steps "$warmup" \
    --workers 4 \
    --seed 323 \
    --compile-mode reduce-overhead \
    --validation-every 1000 \
    --checkpoint-every 500 \
    --validation-cap 2000 \
    --ema-decay "$ema" \
    --source-sampling-exponent 0.35 \
    --threshold 0.65 \
    --out "$run" \
    "${resume[@]}"
}

# The pretrained run is a diagnostic ceiling because its checkpoint is Apache-2.0.
train_candidate convnext_small_pretrained 8000 48 1e-5 400 0.999
# Scratch candidates receive a longer, architecture-appropriate optimization budget.
train_candidate convnext_small_scratch 24000 48 2e-4 1200 0.9995
train_candidate forensic_convnext_scratch 20000 64 2e-4 1000 0.9995

score_candidate() {
  local candidate="$1"
  local checkpoint="$2"
  "$ROOT/.venv/bin/aiblink-val" score-trained \
    --manifest "$TRAIN/manifest.csv" --checkpoint "$checkpoint" --role calibration \
    --views clean,web,hard --batch-size 64 --workers 4 \
    --out "$OUT/evals/calibration/$candidate.jsonl"
  "$ROOT/.venv/bin/aiblink-val" score-trained \
    --manifest "$DATA/recent_openrouter/manifest.csv" --checkpoint "$checkpoint" --role train \
    --views clean,web,hard --batch-size 64 --workers 4 \
    --out "$OUT/evals/openrouter/$candidate.jsonl"
  "$ROOT/.venv/bin/aiblink-val" score-trained \
    --manifest "$TRAIN/recent_holdout_manifest.csv" --checkpoint "$checkpoint" --role test \
    --views clean,web,hard --batch-size 64 --workers 4 \
    --out "$OUT/evals/recent_holdout/$candidate.jsonl"
}

score_candidate commfor_384 "$TRAIN/runs/commfor_384/best.pt"
score_candidate convnext_small_pretrained "$OUT/runs/convnext_small_pretrained/best.pt"
score_candidate convnext_small_scratch "$OUT/runs/convnext_small_scratch/best.pt"
score_candidate forensic_convnext_scratch "$OUT/runs/forensic_convnext_scratch/best.pt"

"$ROOT/.venv/bin/aiblink-val" rank-tournament \
  --manifest "$TRAIN/manifest.csv" \
  --predictions \
    "$OUT/evals/calibration/commfor_384.jsonl" \
    "$OUT/evals/calibration/convnext_small_pretrained.jsonl" \
    "$OUT/evals/calibration/convnext_small_scratch.jsonl" \
    "$OUT/evals/calibration/forensic_convnext_scratch.jsonl" \
  --threshold 0.65 \
  --out "$OUT/ranking.json"

echo "challenger pipeline completed $(date --iso-8601=seconds)"
