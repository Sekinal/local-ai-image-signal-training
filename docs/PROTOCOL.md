# Validation and model-release protocol

## Dataset lifecycle

1. Materialize original bytes into versioned, read-only dataset directories.
2. Build manifests with SHA-256 and perceptual hashes. Record license and provenance.
3. Assign entire provenance groups—not images—to `train`, `calibration`, or `test`.
4. Run `aiblink-val audit`. Resolve every cross-role exact/near duplicate before training.
5. Lock the manifest SHA-256 in the experiment metadata.

The final test manifest should not be used for checkpoint selection, early stopping,
calibrator selection, augmentation choice, or ensemble weighting. If it influences a choice,
it becomes calibration data and a new final test must be established.

## Model lifecycle

1. Produce raw logits for calibration and test data under every declared view.
2. Select `bias` versus positive-slope `platt` calibration using leave-one-group-out folds on
   calibration data. The fold score is macro-balanced across held-out groups.
3. Refit the chosen monotone mapping on all calibration rows. Its decision boundary is placed
   exactly at displayed confidence `0.65`.
4. Freeze `calibration.json`; never retune it on test predictions.
5. Generate the final report at threshold `0.65`, including group tails and paired stress
   degradation.
6. Export ONNX and then score a parity corpus through both Python and the shipped browser
   runtime. Maximum absolute logit/probability drift and decision mismatches are release gates.

## Required comparisons

Every candidate checkpoint is compared with the current champion on identical rows. Preserve
per-sample ledgers so use paired bootstrap deltas, not overlapping standalone confidence
intervals. Promote only when the lower confidence bound of the primary improvement is positive
and no critical generator/source slice regresses beyond the declared tolerance.

## Interpretation

Balanced accuracy is `(AI recall + real specificity) / 2`. Pooled balanced accuracy answers
the bounty question but is not sufficient for model development. Also inspect:

- macro balanced accuracy across provenance groups;
- 10th-percentile group recall/specificity and worst groups;
- clean-to-web and clean-to-hard score drift and decision flips;
- class-conditional performance by original short edge and by the effective
  downscale/upscale factor entering the 440→384 preprocessing path;
- AUROC/average precision (ranking), Brier/log loss/ECE (confidence quality);
- clustered bootstrap intervals at the original-image level.

Do not call a run better based on a point estimate alone.
