# AI Blink validation pipeline

[![tests](https://github.com/Sekinal/local-ai-image-signal-training/actions/workflows/test.yml/badge.svg)](https://github.com/Sekinal/local-ai-image-signal-training/actions/workflows/test.yml)

> **August 2026 release:** the exact source tree, pinned environment, recovery inventory, and commands for the promoted low-quality Community Forensics detector are documented in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Large immutable artifacts live in the linked Hugging Face revision; this repository intentionally does not commit model checkpoints or source-licensed image bytes to Git.

This repository starts with the measurement system, before model fine-tuning. Its job is to
make an inflated result difficult to produce and a real improvement easy to verify.

The pipeline separates four concerns:

1. **Manifest**: one immutable row per original image, with provenance and content hashes.
2. **Integrity gate**: exact and perceptual duplicate checks across train, calibration, and
   test roles. A failed gate blocks the report by default.
3. **Prediction ledger**: one raw logit per `(sample_id, view)` from any backend. Model
   inference and statistics never share hidden state.
4. **Calibration/report**: group-cross-validated monotone calibration, evaluation at the
   bounty's fixed `0.65` threshold, clustered bootstrap intervals, subgroup tails, and paired
   clean-to-web degradation analysis.

This design keeps the final test blind: calibrator selection sees only `role=calibration`;
the locked `role=test` rows are opened once for the final report. Generator families and real
sources are the split groups, never individual images.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test,inference]'

# Validate a manifest before spending GPU time.
aiblink-val audit --manifest data/manifests/all.csv --out artifacts/audit.json

# Score all deterministic views with Community Forensics.
aiblink-val infer \
  --manifest data/manifests/all.csv \
  --role calibration,test \
  --model OwensLab/commfor-model-384 \
  --views clean,web,hard \
  --out artifacts/predictions.jsonl

# Fit on calibration groups and report only on the locked test role.
aiblink-val report \
  --manifest data/manifests/all.csv \
  --predictions artifacts/predictions.jsonl \
  --config configs/validation.yaml \
  --out artifacts/report

# Compare a candidate to the champion with paired, image-clustered uncertainty.
aiblink-val compare \
  --champion artifacts/champion/test_predictions.csv \
  --challenger artifacts/candidate/test_predictions.csv \
  --out artifacts/comparison.json

# Produce frozen-calibrator red-team logits and worst-case metrics.
aiblink-val infer \
  --manifest data/manifests/all.csv --role test \
  --attack-config configs/redteam.yaml --profile core \
  --out artifacts/redteam/predictions.jsonl
aiblink-val redteam-report \
  --manifest data/manifests/all.csv \
  --predictions artifacts/redteam/predictions.jsonl \
  --calibration artifacts/champion/calibration.json \
  --attack-config configs/redteam.yaml --profile core \
  --out artifacts/redteam/report

# Use the identical attack registry with a fine-tuned tournament checkpoint.
aiblink-val score-trained \
  --manifest data/manifests/all.csv --role test \
  --checkpoint artifacts/champion/best.pt \
  --attack-config configs/redteam.yaml --profile core \
  --out artifacts/redteam/trained_predictions.jsonl

# Export the frozen EMA checkpoint to a browser-oriented ONNX graph and verify ORT parity.
aiblink-val export-trained-onnx \
  --checkpoint artifacts/champion/best.pt \
  --precision fp16 \
  --out artifacts/champion/model.fp16.onnx

# Detect resolution/codec/file-size shortcuts before training or release.
aiblink-val metadata-audit \
  --manifest data/manifests/all.csv --role train,calibration,test \
  --out artifacts/metadata_leakage.json

# Run exact-model attacks using the registry's per-epsilon step/restart budget.
aiblink-val whitebox-eval \
  --manifest data/manifests/all.csv --role test \
  --attack-config configs/redteam.yaml --profile adaptive \
  --out artifacts/redteam/whitebox_predictions.jsonl

# Frozen DINOv3-S comparison with out-of-fold probe logits.
aiblink-val dinov3-probe \
  --manifest data/manifests/all.csv --input-size 256 \
  --out artifacts/dinov3/probe_256
aiblink-val report \
  --manifest data/manifests/all.csv \
  --predictions artifacts/dinov3/probe_256/predictions.jsonl \
  --config configs/validation.yaml \
  --out artifacts/dinov3/probe_256/report
```

For data already arranged in folders, create a manifest without losing provenance:

```bash
aiblink-val manifest-from-folders \
  --root /data/synthbuster \
  --dataset synthbuster \
  --role test \
  --layout 'generator/label' \
  --out data/manifests/synthbuster.csv
```

The directory labels recognized by default are `real`, `nature`, `0`, `fake`, `ai`, and `1`.
For a serious run, review and version the generated CSV; do not infer dataset roles at report
time.

## Required manifest columns

`sample_id,path,label,dataset,source,group_id,role,content_sha256,phash`

- `label`: `0` real, `1` AI-generated.
- `source`: generator/model for fakes; camera/collection/provider for reals.
- `group_id`: the indivisible provenance unit used for isolation. Use generator family for
  fakes and collection/capture family for reals.
- `role`: `train`, `calibration`, or `test`.
- `content_sha256`: hash of original bytes.
- `phash`: 64-bit perceptual hash as 16 lowercase hex digits.

Optional columns such as `license`, `release_date`, `width`, `height`, `mime`, `url`, and
`parent_id` are preserved and become report slices where applicable.

## Non-negotiable protocol

- Threshold semantics are `score >= 0.65 => AI` everywhere.
- Calibration candidates are monotone and include identity/no remapping. A transform must beat
  identity in generator-held-out validation; it may move the operating point but cannot improve ranking.
- Method selection leaves one fake-generator group out at a time on calibration data only;
  real samples are cross-fitted into matching folds.
- Final metrics use test data only and include 95% clustered bootstrap intervals.
- Views of the same image are resampled together. Treating recompressions as independent
  images would produce falsely narrow intervals.
- Resolution is measured twice: original short edge and effective short edge after web
  degradation. Reports include the resulting resize factor into the 440→384 model path and
  retain generator groups beside every bucket so resolution is not mistaken for generator effects.
- The report is invalid if hashes, role isolation, prediction coverage, or class support fail.
- Red-team ledgers must carry the exact family, severity, and operation parameters declared in
  the attack registry. Missing or mismatched execution metadata invalidates the report.
- Macro group tails are reported beside pooled scores so one large/easy generator cannot hide
  failures on smaller generators.

See [`configs/validation.yaml`](configs/validation.yaml) for the default gate thresholds and
[`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the release protocol. The adversarial scope and
browser-hostile test matrix are specified in [`docs/RED_TEAM.md`](docs/RED_TEAM.md).
Executable hostile-page fixtures and their required state contract live in
[`browser_fixtures/`](browser_fixtures/README.md).
The controlled DINOv3 ViT-S comparison and licensing caveat are recorded in
[`artifacts/dinov3/`](artifacts/dinov3/README.md).
The pinned multi-model comparison, resolution ablation, and full-core challenger results are
recorded in [`artifacts/model_eval/`](artifacts/model_eval/README.md).
The equal-budget compiled fine-tuning ranking and promoted top-three set are recorded in
[`artifacts/finetune_tournament/`](artifacts/finetune_tournament/README.md).

## Full robust fine-tuning

`scripts/prepare_full_training.py` builds a deterministic training/development manifest from
OpenFake train/validation, the generator-balanced GenImage pilot subset, SynthBuster, COCO
train2017, and WikiArt. It excludes every exact hash in the locked baseline manifest and never
reads OpenFake test. Keep the downloaded masters; only bounded materialized copies enter the
manifest.

```bash
python scripts/prepare_full_training.py \
  --downloads /data/ai_blink/full/downloads \
  --materialized /data/ai_blink/full/materialized \
  --pilot-manifest /data/ai_blink/tournament/subset/manifest.csv \
  --exclude-manifest /data/ai_blink/baseline/manifest.csv \
  --out /data/ai_blink/full/manifest.csv

aiblink-val train-full \
  --manifest /data/ai_blink/full/manifest.csv \
  --initial-checkpoint /data/ai_blink/tournament/runs/commfor_384/pilot.pt \
  --out /data/ai_blink/full/runs/commfor_384
```

The full trainer makes each batch exactly class-balanced, samples provenance sources with a
square-root temperature so rare generators matter without being repeated as aggressively as
uniform-source sampling, and applies the same web-laundering mixture to both classes. It uses
BF16/TF32, fused AdamW, `torch.compile`, EMA model selection, atomic checkpoints every 250
steps, and leave-one-generator-out calibration metrics on development only. Pass the run's
`last.pt` to `--resume` after an interruption.
