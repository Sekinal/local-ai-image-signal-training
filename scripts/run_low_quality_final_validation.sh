#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/ai_blink
DATA=/data/ai_blink
MANIFEST="$DATA/frontier_training/manifest.csv"
ORIGINAL="$DATA/final_commfor_20260816"
EXPERIMENT="$DATA/low_quality_experiment_20260816"
CHECKPOINT="$EXPERIMENT/scaled/last.pt"
OUT="$DATA/low_quality_final_20260816"
EVAL="$OUT/evals"
CONFIG="$ROOT/configs/low_quality_canary.yaml"
mkdir -p "$EVAL" "$OUT/export" "$OUT/report" "$OUT/redteam_report" "$OUT/promotion"
exec > >(tee -a "$OUT/run.log") 2>&1

echo "low-quality final validation started $(date --iso-8601=seconds)"
exec 9>"$DATA/frontier_gpu.lock"
flock -x 9

if [[ ! -s "$EVAL/calibration_pytorch.jsonl" ]]; then
  "$ROOT/.venv/bin/aiblink-val" score-trained \
    --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --role calibration \
    --views clean,web,hard --batch-size 96 --workers 4 \
    --out "$EVAL/calibration_pytorch.jsonl"
fi

"$ROOT/.venv/bin/aiblink-val" rank-tournament \
  --manifest "$MANIFEST" --predictions "$EVAL/calibration_pytorch.jsonl" \
  --threshold 0.65 --out "$OUT/ranking.json"

"$ROOT/.venv/bin/python" "$ROOT/scripts/prepare_final_validation.py" \
  --manifest "$MANIFEST" --ranking "$OUT/ranking.json" \
  --redteam-manifest "$OUT/development_redteam_manifest.csv" \
  --calibrator "$OUT/calibrator.json"

if [[ ! -s "$EVAL/openrouter.jsonl" ]]; then
  "$ROOT/.venv/bin/aiblink-val" score-trained \
    --manifest "$DATA/recent_openrouter/manifest.csv" --checkpoint "$CHECKPOINT" --role train \
    --views clean,web,hard --batch-size 96 --workers 4 --out "$EVAL/openrouter.jsonl"
fi
if [[ ! -s "$EVAL/recent_holdout.jsonl" ]]; then
  "$ROOT/.venv/bin/aiblink-val" score-trained \
    --manifest "$DATA/frontier_training/recent_holdout_manifest.csv" \
    --checkpoint "$CHECKPOINT" --role test --views clean,web,hard \
    --batch-size 96 --workers 4 --out "$EVAL/recent_holdout.jsonl"
fi

for cohort in recent_hf openrouter; do
  if [[ "$cohort" == "recent_hf" ]]; then
    cohort_manifest="$DATA/frontier_training/recent_holdout_manifest.csv"
    cohort_role=test
  else
    cohort_manifest="$DATA/recent_openrouter/manifest.csv"
    cohort_role=train
  fi
  if [[ ! -s "$EVAL/${cohort}_baseline_low_quality.jsonl" ]]; then
    "$ROOT/.venv/bin/aiblink-val" score-trained \
      --manifest "$cohort_manifest" --checkpoint "$ORIGINAL/run/best.pt" \
      --role "$cohort_role" --attack-config "$CONFIG" --profile gate \
      --batch-size 96 --workers 4 --out "$EVAL/${cohort}_baseline_low_quality.jsonl"
  fi
  if [[ ! -s "$EVAL/${cohort}_candidate_low_quality.jsonl" ]]; then
    "$ROOT/.venv/bin/aiblink-val" score-trained \
      --manifest "$cohort_manifest" --checkpoint "$CHECKPOINT" \
      --role "$cohort_role" --attack-config "$CONFIG" --profile gate \
      --batch-size 96 --workers 4 --out "$EVAL/${cohort}_candidate_low_quality.jsonl"
  fi
done

if [[ ! -s "$EVAL/redteam_core.jsonl" ]]; then
  "$ROOT/.venv/bin/aiblink-val" score-trained \
    --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --role calibration \
    --attack-config "$ROOT/configs/redteam.yaml" --profile core \
    --batch-size 96 --workers 4 --out "$EVAL/redteam_core.jsonl"
fi

set +e
"$ROOT/.venv/bin/aiblink-val" redteam-report \
  --manifest "$OUT/development_redteam_manifest.csv" --predictions "$EVAL/redteam_core.jsonl" \
  --calibration "$OUT/calibrator.json" --attack-config "$ROOT/configs/redteam.yaml" \
  --profile core --threshold 0.65 --replicates 2000 --seed 2323 \
  --out "$OUT/redteam_report"
redteam_status=$?
set -e
if (( redteam_status != 0 )) && [[ ! -s "$OUT/redteam_report/redteam_report.json" ]]; then
  echo "red-team reporting failed without producing an artifact" >&2
  exit "$redteam_status"
fi

if [[ ! -s "$OUT/export/community_forensics_low_quality_fp16.onnx" ]]; then
  "$ROOT/.venv/bin/aiblink-val" export-trained-onnx \
    --checkpoint "$CHECKPOINT" --device cuda --precision fp16 --opset 17 \
    --parity-batch 8 --parity-seed 2323 \
    --out "$OUT/export/community_forensics_low_quality_fp16.onnx"
fi

revision=$(sha256sum "$OUT/export/community_forensics_low_quality_fp16.onnx" | cut -d' ' -f1)
if [[ ! -s "$EVAL/calibration_onnx.jsonl" ]]; then
  "$ROOT/.venv/bin/aiblink-val" onnx-infer \
    --manifest "$MANIFEST" --role calibration \
    --model-path "$OUT/export/community_forensics_low_quality_fp16.onnx" \
    --model-id commfor_384_low_quality --revision "$revision" --input-size 384 \
    --views clean,web,hard --batch-size 96 --workers 4 --device cuda \
    --out "$EVAL/calibration_onnx.jsonl"
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/report_final_validation.py" \
  --manifest "$MANIFEST" --preparation "$DATA/frontier_training/preparation.json" \
  --audit "$ORIGINAL/data_audit.json" --training "$EXPERIMENT/scaled/training.json" \
  --checkpoint "$CHECKPOINT" --ranking "$OUT/ranking.json" \
  --calibration-predictions "$EVAL/calibration_pytorch.jsonl" \
  --onnx-predictions "$EVAL/calibration_onnx.jsonl" \
  --openrouter-predictions "$EVAL/openrouter.jsonl" \
  --recent-predictions "$EVAL/recent_holdout.jsonl" \
  --redteam "$OUT/redteam_report/redteam_report.json" \
  --export "$OUT/export/community_forensics_low_quality_fp16.onnx.json" \
  --out "$OUT/report"

"$ROOT/.venv/bin/python" "$ROOT/scripts/report_low_quality_promotion.py" \
  --manifest "$MANIFEST" --original-report "$ORIGINAL/report/report.json" \
  --candidate-report "$OUT/report/report.json" \
  --original-calibrator "$ORIGINAL/calibrator.json" \
  --candidate-calibrator "$OUT/calibrator.json" \
  --baseline-low-quality "$EXPERIMENT/evals/baseline_calibration.jsonl" \
  --candidate-low-quality "$EXPERIMENT/evals/scaled_calibration.jsonl" \
  --recent-hf-baseline-low-quality "$EVAL/recent_hf_baseline_low_quality.jsonl" \
  --recent-hf-candidate-low-quality "$EVAL/recent_hf_candidate_low_quality.jsonl" \
  --openrouter-baseline-low-quality "$EVAL/openrouter_baseline_low_quality.jsonl" \
  --openrouter-candidate-low-quality "$EVAL/openrouter_candidate_low_quality.jsonl" \
  --original-redteam "$ORIGINAL/redteam_report/redteam_report.json" \
  --candidate-redteam "$OUT/redteam_report/redteam_report.json" \
  --out "$OUT/promotion/report.json"

flock -u 9
echo "low-quality final validation completed $(date --iso-8601=seconds)"
