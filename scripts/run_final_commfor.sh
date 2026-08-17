#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/ai_blink
DATA=/data/ai_blink
MANIFEST="$DATA/frontier_training/manifest.csv"
OUT="$DATA/final_commfor_20260816"
RUN="$OUT/run"
EVAL="$OUT/evals"
mkdir -p "$RUN" "$EVAL" "$OUT/report" "$OUT/export"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "final Community Forensics pipeline started $(date --iso-8601=seconds)"
sha=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
if [[ "$sha" != "af629187fca31ab2f4f7f4c81ea3a761c13ea7cb8d4ee8a9bb9e7bd8e1eb6f36" ]]; then
  echo "unexpected final manifest hash $sha" >&2
  exit 1
fi

if [[ ! -f "$OUT/data_audit.json" ]]; then
  "$ROOT/.venv/bin/aiblink-val" audit --manifest "$MANIFEST" --phash-distance 4 --out "$OUT/data_audit.json"
fi

exec 9>"$DATA/frontier_gpu.lock"
flock -x 9

if [[ ! -f "$RUN/training.json" ]]; then
  resume=()
  if [[ -f "$RUN/last.pt" ]]; then resume=(--resume "$RUN/last.pt"); fi
  "$ROOT/.venv/bin/aiblink-val" train-full \
    --manifest "$MANIFEST" \
    --initial-checkpoint "$DATA/frontier_canary_20260816/runs/commfor_384/best.pt" \
    --steps 6000 --batch-size 96 --validation-batch-size 96 \
    --lr 1e-5 --head-multiplier 1 --weight-decay 0.05 --warmup-steps 300 \
    --workers 4 --seed 323 --compile-mode reduce-overhead \
    --validation-every 750 --checkpoint-every 500 --validation-cap 3912 \
    --ema-decay 0.999 --source-sampling-exponent 0.35 --threshold 0.65 \
    --out "$RUN" "${resume[@]}"
fi

"$ROOT/.venv/bin/aiblink-val" score-trained \
  --manifest "$MANIFEST" --checkpoint "$RUN/best.pt" --role calibration \
  --views clean,web,hard --batch-size 96 --workers 4 --out "$EVAL/calibration_pytorch.jsonl"

"$ROOT/.venv/bin/aiblink-val" rank-tournament \
  --manifest "$MANIFEST" --predictions "$EVAL/calibration_pytorch.jsonl" \
  --threshold 0.65 --out "$OUT/ranking.json"

"$ROOT/.venv/bin/python" "$ROOT/scripts/prepare_final_validation.py" \
  --manifest "$MANIFEST" --ranking "$OUT/ranking.json" \
  --redteam-manifest "$OUT/development_redteam_manifest.csv" \
  --calibrator "$OUT/calibrator.json"

"$ROOT/.venv/bin/aiblink-val" score-trained \
  --manifest "$DATA/recent_openrouter/manifest.csv" --checkpoint "$RUN/best.pt" --role train \
  --views clean,web,hard --batch-size 96 --workers 4 --out "$EVAL/openrouter.jsonl"

"$ROOT/.venv/bin/aiblink-val" score-trained \
  --manifest "$DATA/frontier_training/recent_holdout_manifest.csv" --checkpoint "$RUN/best.pt" --role test \
  --views clean,web,hard --batch-size 96 --workers 4 --out "$EVAL/recent_holdout.jsonl"

"$ROOT/.venv/bin/aiblink-val" score-trained \
  --manifest "$MANIFEST" --checkpoint "$RUN/best.pt" --role calibration \
  --attack-config "$ROOT/configs/redteam.yaml" --profile core \
  --batch-size 96 --workers 4 --out "$EVAL/redteam_core.jsonl"

set +e
"$ROOT/.venv/bin/aiblink-val" redteam-report \
  --manifest "$OUT/development_redteam_manifest.csv" --predictions "$EVAL/redteam_core.jsonl" \
  --calibration "$OUT/calibrator.json" --attack-config "$ROOT/configs/redteam.yaml" \
  --profile core --threshold 0.65 --replicates 2000 --seed 323 --out "$OUT/redteam_report"
redteam_status=$?
set -e
if (( redteam_status != 0 )); then
  if [[ -s "$OUT/redteam_report/redteam_report.json" ]]; then
    echo "red-team gates failed; preserving the diagnostic result and continuing finalization"
  else
    echo "red-team reporting failed without producing an artifact" >&2
    exit "$redteam_status"
  fi
fi

"$ROOT/.venv/bin/aiblink-val" export-trained-onnx \
  --checkpoint "$RUN/best.pt" --device cuda --precision fp16 --opset 17 \
  --parity-batch 8 --parity-seed 323 --out "$OUT/export/community_forensics_final_fp16.onnx"

revision=$(sha256sum "$OUT/export/community_forensics_final_fp16.onnx" | cut -d' ' -f1)
"$ROOT/.venv/bin/aiblink-val" onnx-infer \
  --manifest "$MANIFEST" --role calibration \
  --model-path "$OUT/export/community_forensics_final_fp16.onnx" \
  --model-id commfor_384_final_fp16 --revision "$revision" --input-size 384 \
  --views clean,web,hard --batch-size 96 --workers 4 --device cuda \
  --out "$EVAL/calibration_onnx.jsonl"

"$ROOT/.venv/bin/python" "$ROOT/scripts/report_final_validation.py" \
  --manifest "$MANIFEST" --preparation "$DATA/frontier_training/preparation.json" \
  --audit "$OUT/data_audit.json" --training "$RUN/training.json" --ranking "$OUT/ranking.json" \
  --calibration-predictions "$EVAL/calibration_pytorch.jsonl" \
  --onnx-predictions "$EVAL/calibration_onnx.jsonl" \
  --openrouter-predictions "$EVAL/openrouter.jsonl" \
  --recent-predictions "$EVAL/recent_holdout.jsonl" \
  --redteam "$OUT/redteam_report/redteam_report.json" \
  --export "$OUT/export/community_forensics_final_fp16.onnx.json" \
  --out "$OUT/report"

flock -u 9
echo "final Community Forensics pipeline completed $(date --iso-8601=seconds)"
