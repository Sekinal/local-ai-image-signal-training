#!/usr/bin/env python3
"""Report frozen Sieve diagnostic performance on the recent HF synthetic cohort."""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np


VIEW_ORDER = ("clean", "web", "hard")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(rows: list[dict]) -> dict:
    probabilities = np.asarray([row["probability"] for row in rows], dtype=np.float64)
    detections = int(sum(row["detected_fake"] for row in rows))
    return {
        "n": len(rows),
        "detected": detections,
        "missed": len(rows) - detections,
        "recall": detections / len(rows),
        "recall_wilson_95": wilson(detections, len(rows)),
        "score": {
            "mean": float(np.mean(probabilities)),
            "std": float(np.std(probabilities)),
            "min": float(np.min(probabilities)),
            "p05": float(np.quantile(probabilities, 0.05)),
            "p25": float(np.quantile(probabilities, 0.25)),
            "median": float(np.median(probabilities)),
            "p75": float(np.quantile(probabilities, 0.75)),
            "p95": float(np.quantile(probabilities, 0.95)),
            "max": float(np.max(probabilities)),
        },
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def score_text(summary: dict) -> str:
    score = summary["score"]
    return f"{score['median']:.4f} ({score['p05']:.4f}–{score['p95']:.4f})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--leakage-audit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest = {row["sample_id"]: row for row in manifest_rows}
    if len(manifest_rows) != 1000 or len(manifest) != 1000:
        raise RuntimeError("diagnostic manifest must contain exactly 1,000 unique samples")
    if any(int(row["label"]) != 1 or row["role"] != "test" for row in manifest_rows):
        raise RuntimeError("diagnostic cohort must be test-role synthetic-only data")

    provenance = {}
    for line in args.ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("error"):
            provenance[record["content_sha256"]] = record

    prediction_rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_keys = {(sample_id, view) for sample_id in manifest for view in VIEW_ORDER}
    actual_keys = {(row["sample_id"], row["view"]) for row in prediction_rows}
    if len(prediction_rows) != 3000 or actual_keys != expected_keys:
        raise RuntimeError("prediction ledger does not have exact 1,000 x 3 coverage")
    if any(row.get("error") or row.get("raw_logit") is None for row in prediction_rows):
        raise RuntimeError("prediction ledger contains failures")

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))["final"]
    threshold = float(calibration["target_threshold"])
    raw_boundary = float(calibration["raw_boundary"])
    slope = float(calibration["slope"])
    intercept = float(calibration["intercept"])
    logits = np.asarray([row["raw_logit"] for row in prediction_rows], dtype=np.float64)
    probabilities = sigmoid(slope * logits + intercept)

    enriched = []
    for prediction, probability in zip(prediction_rows, probabilities, strict=True):
        item = manifest[prediction["sample_id"]]
        source = item["source"].removeprefix("HuggingFace:")
        detail = provenance.get(item["content_sha256"], {})
        detected = bool(probability >= threshold)
        if detected != bool(float(prediction["raw_logit"]) >= raw_boundary):
            raise RuntimeError("calibrated and frozen raw-boundary decisions disagree")
        enriched.append(
            {
                "sample_id": prediction["sample_id"],
                "view": prediction["view"],
                "model_id": source,
                "model_slug": detail.get("model_slug", source.rsplit("/", 1)[-1]),
                "generator_revision": item.get("generator_revision", ""),
                "path": item["path"],
                "width": int(item["width"]),
                "height": int(item["height"]),
                "prompt": detail.get("prompt", ""),
                "seed": detail.get("seed", item.get("generation_seed", "")),
                "raw_logit": float(prediction["raw_logit"]),
                "probability": float(probability),
                "detected_fake": detected,
            }
        )

    by_view = {
        view: summarize([row for row in enriched if row["view"] == view])
        for view in VIEW_ORDER
    }
    model_ids = sorted({row["model_id"] for row in enriched})
    by_model = {}
    for model_id in model_ids:
        by_model[model_id] = {
            view: summarize(
                [row for row in enriched if row["model_id"] == model_id and row["view"] == view]
            )
            for view in VIEW_ORDER
        }

    by_sample = collections.defaultdict(dict)
    for row in enriched:
        by_sample[row["sample_id"]][row["view"]] = row
    all_views_detected = sum(
        all(rows[view]["detected_fake"] for view in VIEW_ORDER) for rows in by_sample.values()
    )
    at_least_one_detected = sum(
        any(rows[view]["detected_fake"] for view in VIEW_ORDER) for rows in by_sample.values()
    )
    clean_misses = [row for row in enriched if row["view"] == "clean" and not row["detected_fake"]]
    clean_misses.sort(key=lambda row: row["probability"])

    transform_deltas = {}
    for view in ("web", "hard"):
        deltas = np.asarray(
            [rows[view]["probability"] - rows["clean"]["probability"] for rows in by_sample.values()]
        )
        transform_deltas[view] = {
            "mean_score_delta_vs_clean": float(np.mean(deltas)),
            "median_score_delta_vs_clean": float(np.median(deltas)),
            "clean_detected_to_missed": sum(
                rows["clean"]["detected_fake"] and not rows[view]["detected_fake"]
                for rows in by_sample.values()
            ),
            "clean_missed_to_detected": sum(
                not rows["clean"]["detected_fake"] and rows[view]["detected_fake"]
                for rows in by_sample.values()
            ),
        }

    leakage = json.loads(args.leakage_audit.read_text(encoding="utf-8"))
    if not leakage.get("valid") or leakage.get("issues"):
        raise RuntimeError("leakage audit is not clean")
    report = {
        "protocol": "aiblink-recent-hf-frozen-diagnostic/0.1.0",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "diagnostic_only": True,
        "submission_eligible": False,
        "disqualification_reason": "Sieve initialization came from another bounty participant",
        "cohort": {
            "samples": 1000,
            "positive_label": "synthetic/AI-generated",
            "real_samples": 0,
            "generators": 4,
            "samples_per_generator": 250,
            "balanced_accuracy": None,
            "roc_auc": None,
            "limitation": "positive-only cohort; measures synthetic recall, not balanced detector performance",
        },
        "frozen_decision": {
            "calibrator": calibration,
            "probability_threshold": threshold,
            "raw_logit_boundary": raw_boundary,
            "tuning_on_cohort": False,
        },
        "integrity": {
            "cross_role_leakage_audit_valid": True,
            "prediction_failures": 0,
            "prediction_coverage": "3000/3000",
            "manifest_sha256": sha256_file(args.manifest),
            "predictions_sha256": sha256_file(args.predictions),
            "calibration_sha256": sha256_file(args.calibration),
            "leakage_audit_sha256": sha256_file(args.leakage_audit),
            "onnx_sha256": sha256_file(args.model),
            "training_checkpoint_sha256": sha256_file(args.checkpoint),
        },
        "overall_by_view": by_view,
        "by_generator": by_model,
        "cross_view": {
            "all_three_views_detected": all_views_detected,
            "all_three_views_recall": all_views_detected / 1000,
            "at_least_one_view_detected": at_least_one_detected,
            "at_least_one_view_recall": at_least_one_detected / 1000,
        },
        "transform_effects": transform_deltas,
        "clean_misses": len(clean_misses),
    }

    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = [
        "sample_id", "view", "model_id", "model_slug", "generator_revision", "path",
        "width", "height", "prompt", "seed", "raw_logit", "probability", "detected_fake",
    ]
    with (args.out / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(enriched, key=lambda row: (row["model_id"], row["sample_id"], row["view"])))
    with (args.out / "clean_misses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean_misses)

    lines = [
        "# Frozen Sieve FT1 on recent Hugging Face generators",
        "",
        "> Diagnostic only. Sieve is disqualified from submission because its initialization came from another bounty participant.",
        "",
        "## Result",
        "",
        f"At the untouched `0.65` calibrated threshold, clean synthetic recall is **{pct(by_view['clean']['recall'])}** "
        f"({by_view['clean']['detected']}/1,000; Wilson 95% CI "
        f"{pct(by_view['clean']['recall_wilson_95'][0])}–{pct(by_view['clean']['recall_wilson_95'][1])}).",
        "",
        "| View | Detected | Recall | 95% CI | Score median (p05–p95) |",
        "|---|---:|---:|---:|---:|",
    ]
    for view in VIEW_ORDER:
        item = by_view[view]
        lines.append(
            f"| {view} | {item['detected']}/1,000 | {pct(item['recall'])} | "
            f"{pct(item['recall_wilson_95'][0])}–{pct(item['recall_wilson_95'][1])} | {score_text(item)} |"
        )
    lines += [
        "",
        "## Per generator",
        "",
        "| Generator | Clean recall | Web recall | Hard recall | Clean score median (p05–p95) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id in model_ids:
        items = by_model[model_id]
        lines.append(
            f"| `{model_id}` | {items['clean']['detected']}/250 ({pct(items['clean']['recall'])}) | "
            f"{items['web']['detected']}/250 ({pct(items['web']['recall'])}) | "
            f"{items['hard']['detected']}/250 ({pct(items['hard']['recall'])}) | "
            f"{score_text(items['clean'])} |"
        )
    lines += [
        "",
        "## Cross-view robustness",
        "",
        f"- Detected in all three views: {all_views_detected}/1,000 ({pct(all_views_detected / 1000)}).",
        f"- Detected in at least one view: {at_least_one_detected}/1,000 ({pct(at_least_one_detected / 1000)}).",
        f"- Clean misses: {len(clean_misses)}; see `clean_misses.csv`.",
        "",
        "## Protocol and interpretation",
        "",
        f"The frozen FP16 ONNX graph (`{report['integrity']['onnx_sha256']}`) and untouched Platt calibrator were used once, with no tuning on this cohort. The combined train/calibration/diagnostic manifest passed exact and pHash-distance-4 cross-role leakage checks. All 3,000 predictions completed without failure.",
        "",
        "This is a positive-only diagnostic cohort: it measures recall on four recent generators. Balanced accuracy, specificity, false-positive rate, and AUROC are undefined because there are no real images. It therefore cannot establish overall detector quality or competition performance.",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
