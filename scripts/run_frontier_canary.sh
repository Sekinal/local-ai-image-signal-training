#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/ai_blink
DATA=/data/ai_blink
MANIFEST="$DATA/frontier_training/manifest.csv"
OUT="$DATA/frontier_canary_20260816"
mkdir -p "$OUT/runs"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "equal-exposure robust canary started $(date --iso-8601=seconds)"
if [[ ! -f "$MANIFEST" ]]; then
  echo "audited frontier manifest is missing" >&2
  exit 1
fi
exec 9>"$DATA/frontier_gpu.lock"
flock -x 9

run_canary() {
  local candidate="$1"
  local initial="$2"
  local lr="$3"
  local run="$OUT/runs/$candidate"
  mkdir -p "$run"
  if [[ -f "$run/training.json" ]]; then
    echo "canary already complete for $candidate"
    return
  fi
  resume=()
  if [[ -f "$run/last.pt" ]]; then
    resume=(--resume "$run/last.pt")
  fi
  "$ROOT/.venv/bin/aiblink-val" train-full \
    --manifest "$MANIFEST" --initial-checkpoint "$initial" \
    --steps 600 --batch-size 48 --validation-batch-size 48 \
    --lr "$lr" --head-multiplier 1 --weight-decay 0.05 --warmup-steps 60 \
    --workers 4 --seed 323 --compile-mode reduce-overhead \
    --validation-every 200 --checkpoint-every 600 --validation-cap 1000 \
    --ema-decay 0.999 --source-sampling-exponent 0.35 --threshold 0.65 \
    --out "$run" "${resume[@]}"
}

# Same 28,800 robust examples and sampler seed for every candidate.
run_canary commfor_384 "$DATA/full/runs/commfor_384/best.pt" 1e-5
run_canary convnext_small_pretrained "$DATA/frontier_warmstarts/convnext_small_pretrained/pilot.pt" 1e-5
run_canary convnext_small_scratch "$DATA/frontier_warmstarts/convnext_small_scratch/pilot.pt" 2e-4
run_canary forensic_convnext_scratch "$DATA/frontier_warmstarts/forensic_convnext_scratch/pilot.pt" 2e-4

"$ROOT/.venv/bin/python" "$ROOT/scripts/analyze_frontier_canary.py" \
  --training "$OUT/runs/commfor_384/training.json" \
  --training "$OUT/runs/convnext_small_pretrained/training.json" \
  --training "$OUT/runs/convnext_small_scratch/training.json" \
  --training "$OUT/runs/forensic_convnext_scratch/training.json" \
  --out "$OUT/summary.json"

flock -u 9
echo "equal-exposure robust canary completed $(date --iso-8601=seconds)"
