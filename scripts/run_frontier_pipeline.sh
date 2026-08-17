#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/ai_blink
DATA=/data/ai_blink
ACQ="$DATA/acquisitions"
PUBLIC="$DATA/frontier_synthetic_public/stage"
TRAIN="$DATA/frontier_training"
LOG="$DATA/frontier_pipeline.log"

exec > >(tee -a "$LOG") 2>&1
echo "pipeline started $(date --iso-8601=seconds)"

wait_pidfile() {
  local pidfile="$1"
  local label="$2"
  while [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; do
    echo "waiting for $label $(date --iso-8601=seconds)"
    sleep 20
  done
}

wait_pidfile "$ACQ/qwen_image_bench/download-direct.pid" qwen-image-bench
wait_pidfile "$ACQ/rapidata/flux2/download-direct.pid" rapidata-flux2
wait_pidfile "$ACQ/rapidata/recraft3/download-direct.pid" rapidata-recraft3
wait_pidfile "$ACQ/rapidata/seedream3/download-direct.pid" rapidata-seedream3

qwen_count=$(find "$ACQ/qwen_image_bench/source/images" -type f ! -name '*.part' | wc -l)
flux_count=$(find "$ACQ/rapidata/flux2/source/data" -type f -name '*.parquet' | wc -l)
recraft_count=$(find "$ACQ/rapidata/recraft3/source/data" -type f -name '*.parquet' | wc -l)
seedream_count=$(find "$ACQ/rapidata/seedream3/source/data" -type f -name '*.parquet' | wc -l)
if [[ "$qwen_count" -ne 18000 || "$flux_count" -ne 69 || "$recraft_count" -ne 122 || "$seedream_count" -ne 101 ]]; then
  echo "incomplete acquisition: qwen=$qwen_count flux=$flux_count recraft=$recraft_count seedream=$seedream_count" >&2
  exit 1
fi
echo "all pinned acquisitions complete"

"$ROOT/.venv/bin/python" "$ROOT/scripts/build_frontier_public_dataset.py" \
  --acquisitions "$ACQ" \
  --recent "$DATA/recent_huggingface" \
  --output "$PUBLIC" \
  --replace

"$ROOT/.venv/bin/python" "$ROOT/scripts/prepare_frontier_training.py" \
  --stage "$PUBLIC" \
  --base-manifest "$DATA/full/manifest.csv" \
  --output "$TRAIN" \
  --cap-per-generator 12000 \
  --recent-holdout-mod 5 \
  --replace

wait_pidfile "$DATA/frontier_warmstarts.pid" frontier-model-warmstarts

exec 9>"$DATA/frontier_gpu.lock"
echo "waiting for exclusive GPU lease $(date --iso-8601=seconds)"
flock -x 9
echo "acquired exclusive GPU lease $(date --iso-8601=seconds)"

mkdir -p "$TRAIN/runs/commfor_384" "$TRAIN/evals"
"$ROOT/.venv/bin/aiblink-val" train-full \
  --manifest "$TRAIN/manifest.csv" \
  --initial-checkpoint "$DATA/full/runs/commfor_384/best.pt" \
  --steps 6000 \
  --batch-size 96 \
  --lr 1e-5 \
  --head-multiplier 1.0 \
  --weight-decay 0.05 \
  --warmup-steps 300 \
  --workers 4 \
  --compile-mode reduce-overhead \
  --validation-every 750 \
  --checkpoint-every 250 \
  --validation-cap 2000 \
  --ema-decay 0.999 \
  --source-sampling-exponent 0.35 \
  --threshold 0.65 \
  --out "$TRAIN/runs/commfor_384"

"$ROOT/.venv/bin/aiblink-val" score-trained \
  --manifest "$DATA/recent_openrouter/manifest.csv" \
  --checkpoint "$TRAIN/runs/commfor_384/best.pt" \
  --role train \
  --views clean,web,hard \
  --batch-size 96 \
  --workers 4 \
  --out "$TRAIN/evals/openrouter.jsonl"

"$ROOT/.venv/bin/aiblink-val" score-trained \
  --manifest "$TRAIN/recent_holdout_manifest.csv" \
  --checkpoint "$TRAIN/runs/commfor_384/best.pt" \
  --role test \
  --views clean,web,hard \
  --batch-size 96 \
  --workers 4 \
  --out "$TRAIN/evals/recent_holdout.jsonl"

echo "pipeline completed $(date --iso-8601=seconds)"
