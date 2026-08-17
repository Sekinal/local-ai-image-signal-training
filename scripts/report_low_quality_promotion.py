#!/usr/bin/env python3
"""Issue a conservative promotion verdict for the low-quality checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from aiblink_validation.calibration import Calibrator
from aiblink_validation.io import atomic_json, read_jsonl, read_manifest
from aiblink_validation.metrics import binary_metrics


def load_calibrator(path: Path) -> Calibrator:
    payload = json.loads(path.read_text())
    return Calibrator(**payload.get("final", payload))


def balanced_views(manifest_path: Path, predictions_path: Path, calibrator: Calibrator) -> dict:
    manifest = {
        row["sample_id"]: row
        for row in read_manifest(manifest_path)
        if row["role"] == "calibration"
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for prediction in read_jsonl(predictions_path):
        if not prediction.get("error"):
            grouped[str(prediction["view"])].append(prediction)
    result = {}
    for view, rows in sorted(grouped.items()):
        labels = np.asarray([int(manifest[row["sample_id"]]["label"]) for row in rows])
        probabilities = calibrator.transform(
            np.asarray([float(row["raw_logit"]) for row in rows], dtype=np.float64)
        )
        metrics = binary_metrics(labels, probabilities, 0.65)
        result[view] = {
            key: metrics[key]
            for key in ("n", "balanced_accuracy", "fake_recall", "real_specificity", "roc_auc")
        }
    return result


def positive_views(predictions_path: Path, calibrator: Calibrator) -> dict:
    grouped: dict[str, list[bool]] = defaultdict(list)
    by_sample: dict[str, list[bool]] = defaultdict(list)
    for row in read_jsonl(predictions_path):
        if row.get("error"):
            continue
        detected = bool(calibrator.transform(np.asarray([float(row["raw_logit"])]))[0] >= 0.65)
        grouped[str(row["view"])].append(detected)
        by_sample[str(row["sample_id"])].append(detected)
    return {
        "unique_images": len(by_sample),
        "recall_by_view": {key: float(np.mean(value)) for key, value in sorted(grouped.items())},
        "macro_view_recall": statistics.fmean(float(np.mean(value)) for value in grouped.values()),
        "all_views_recall": float(np.mean([all(value) for value in by_sample.values()])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--original-calibrator", type=Path, required=True)
    parser.add_argument("--candidate-calibrator", type=Path, required=True)
    parser.add_argument("--baseline-low-quality", type=Path, required=True)
    parser.add_argument("--candidate-low-quality", type=Path, required=True)
    parser.add_argument("--recent-hf-baseline-low-quality", type=Path, required=True)
    parser.add_argument("--recent-hf-candidate-low-quality", type=Path, required=True)
    parser.add_argument("--openrouter-baseline-low-quality", type=Path, required=True)
    parser.add_argument("--openrouter-candidate-low-quality", type=Path, required=True)
    parser.add_argument("--original-redteam", type=Path, required=True)
    parser.add_argument("--candidate-redteam", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    original_report = json.loads(args.original_report.read_text())
    candidate_report = json.loads(args.candidate_report.read_text())
    original_calibrator = load_calibrator(args.original_calibrator)
    candidate_calibrator = load_calibrator(args.candidate_calibrator)
    original_lowq = balanced_views(
        args.manifest, args.baseline_low_quality, original_calibrator
    )
    candidate_lowq = balanced_views(
        args.manifest, args.candidate_low_quality, candidate_calibrator
    )
    lowq_views = sorted(set(original_lowq) & set(candidate_lowq) - {"clean"})

    def mean_metric(values: dict[str, dict], metric: str) -> float:
        return statistics.fmean(values[view][metric] for view in lowq_views)

    original_redteam = json.loads(args.original_redteam.read_text())
    candidate_redteam = json.loads(args.candidate_redteam.read_text())
    original_standard = original_report["calibration"]["view_metrics"]
    candidate_standard = candidate_report["calibration"]["view_metrics"]
    original_recent_hf = positive_views(
        args.recent_hf_baseline_low_quality, original_calibrator
    )
    candidate_recent_hf = positive_views(
        args.recent_hf_candidate_low_quality, candidate_calibrator
    )
    original_openrouter = positive_views(
        args.openrouter_baseline_low_quality, original_calibrator
    )
    candidate_openrouter = positive_views(
        args.openrouter_candidate_low_quality, candidate_calibrator
    )

    deltas = {
        "clean_balanced_accuracy": candidate_standard["clean"]["balanced_accuracy"]
        - original_standard["clean"]["balanced_accuracy"],
        "web_balanced_accuracy": candidate_standard["web"]["balanced_accuracy"]
        - original_standard["web"]["balanced_accuracy"],
        "hard_balanced_accuracy": candidate_standard["hard"]["balanced_accuracy"]
        - original_standard["hard"]["balanced_accuracy"],
        "low_quality_macro_balanced_accuracy": mean_metric(
            candidate_lowq, "balanced_accuracy"
        )
        - mean_metric(original_lowq, "balanced_accuracy"),
        "low_quality_macro_fake_recall": mean_metric(candidate_lowq, "fake_recall")
        - mean_metric(original_lowq, "fake_recall"),
        "recent_hf_low_quality_macro_recall": candidate_recent_hf["macro_view_recall"]
        - original_recent_hf["macro_view_recall"],
        "openrouter_low_quality_macro_recall": candidate_openrouter["macro_view_recall"]
        - original_openrouter["macro_view_recall"],
        "resolution_worst_balanced_accuracy": candidate_redteam["families"]["resolution"][
            "worst_balanced_accuracy"
        ]
        - original_redteam["families"]["resolution"]["worst_balanced_accuracy"],
    }
    gates = {
        "clean_regression_at_most_1_point": deltas["clean_balanced_accuracy"] >= -0.01,
        "web_regression_at_most_1_point": deltas["web_balanced_accuracy"] >= -0.01,
        "hard_regression_at_most_1_point": deltas["hard_balanced_accuracy"] >= -0.01,
        "low_quality_ba_gain_at_least_8_points": deltas[
            "low_quality_macro_balanced_accuracy"
        ]
        >= 0.08,
        "low_quality_recall_gain_at_least_15_points": deltas[
            "low_quality_macro_fake_recall"
        ]
        >= 0.15,
        "recent_hf_low_quality_recall_not_regressed": deltas[
            "recent_hf_low_quality_macro_recall"
        ]
        >= 0.0,
        "openrouter_low_quality_recall_not_regressed": deltas[
            "openrouter_low_quality_macro_recall"
        ]
        >= 0.0,
        "resolution_worst_ba_gain_at_least_5_points": deltas[
            "resolution_worst_balanced_accuracy"
        ]
        >= 0.05,
        "onnx_decision_disagreement_at_most_half_percent": candidate_report["deployment"][
            "full_calibration_ledger_parity"
        ]["threshold_decision_disagreement_rate"]
        <= 0.005,
    }
    payload = {
        "protocol": "aiblink-low-quality-promotion/0.1.0",
        "threshold": 0.65,
        "selection_role": "calibration",
        "locked_competition_test_opened": False,
        "promote": all(gates.values()),
        "gates": gates,
        "deltas": deltas,
        "standard": {"original": original_standard, "candidate": candidate_standard},
        "low_quality": {
            "views": lowq_views,
            "original": original_lowq,
            "candidate": candidate_lowq,
        },
        "protected_recent_hf_low_quality": {
            "original": original_recent_hf,
            "candidate": candidate_recent_hf,
        },
        "protected_openrouter_low_quality": {
            "original": original_openrouter,
            "candidate": candidate_openrouter,
        },
        "candidate_redteam_valid": candidate_redteam["valid"],
        "candidate_redteam_gates": candidate_redteam["gates"],
        "candidate_onnx_parity": candidate_report["deployment"][
            "full_calibration_ledger_parity"
        ],
    }
    atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
