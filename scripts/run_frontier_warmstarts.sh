#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/ai_blink
DATA=/data/ai_blink
MANIFEST="$DATA/tournament/subset/manifest.csv"
OUT="$DATA/frontier_warmstarts"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "warmstarts started $(date --iso-8601=seconds)"

"$ROOT/.venv/bin/aiblink-val" train-pilot \
  --manifest "$MANIFEST" \
  --candidate convnext_small_pretrained \
  --steps 1200 \
  --batch-size 48 \
  --lr 2e-5 \
  --head-multiplier 10 \
  --weight-decay 0.05 \
  --workers 4 \
  --seed 323 \
  --compile-mode reduce-overhead \
  --out "$OUT/convnext_small_pretrained"

"$ROOT/.venv/bin/aiblink-val" train-pilot \
  --manifest "$MANIFEST" \
  --candidate convnext_small_scratch \
  --steps 4000 \
  --batch-size 48 \
  --lr 3e-4 \
  --head-multiplier 1 \
  --weight-decay 0.05 \
  --workers 4 \
  --seed 323 \
  --compile-mode reduce-overhead \
  --out "$OUT/convnext_small_scratch"

"$ROOT/.venv/bin/aiblink-val" train-pilot \
  --manifest "$MANIFEST" \
  --candidate forensic_convnext_scratch \
  --steps 4000 \
  --batch-size 64 \
  --lr 3e-4 \
  --head-multiplier 1 \
  --weight-decay 0.05 \
  --workers 4 \
  --seed 323 \
  --compile-mode reduce-overhead \
  --out "$OUT/forensic_convnext_scratch"

echo "warmstarts completed $(date --iso-8601=seconds)"
