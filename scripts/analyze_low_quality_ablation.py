#!/usr/bin/env python3
"""Compare two checkpoints on a frozen low-quality calibration ledger."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from aiblink_validation.calibration import select_and_fit
from aiblink_validation.io import atomic_json, read_jsonl, read_manifest
from aiblink_validation.metrics import binary_metrics


def summarize(manifest: dict[str, dict], path: Path) -> dict:
    predictions = [row for row in read_jsonl(path) if not row.get("error")]
    clean = [
        {**manifest[row["sample_id"]], **row}
        for row in predictions
        if row["view"] == "clean"
    ]
    calibrator, diagnostics = select_and_fit(clean, ["bias"], 0.65)
    by_view: dict[str, dict] = {}
    for view in sorted({str(row["view"]) for row in predictions}):
        rows = [{**manifest[row["sample_id"]], **row} for row in predictions if row["view"] == view]
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
        probabilities = calibrator.transform(
            np.asarray([float(row["raw_logit"]) for row in rows], dtype=np.float64)
        )
        metrics = binary_metrics(labels, probabilities, 0.65)
        by_view[view] = {
            key: metrics[key]
            for key in (
                "n",
                "balanced_accuracy",
                "fake_recall",
                "real_specificity",
                "roc_auc",
            )
        }
    low_quality = [metrics for view, metrics in by_view.items() if view != "clean"]
    critical = [
        metrics
        for view, metrics in by_view.items()
        if view.startswith(("resize_96", "resize_64", "resize_48", "resize_32", "lowq_"))
    ]
    return {
        "prediction_path": str(path),
        "model_revision": predictions[0]["model_revision"],
        "calibrator": calibrator.to_dict(),
        "clean_oof_balanced_accuracy": diagnostics["candidates"]["bias"][
            "logo_pooled_balanced_accuracy"
        ],
        "views": by_view,
        "low_quality_macro_balanced_accuracy": statistics.fmean(
            row["balanced_accuracy"] for row in low_quality
        ),
        "low_quality_macro_fake_recall": statistics.fmean(
            row["fake_recall"] for row in low_quality
        ),
        "low_quality_macro_real_specificity": statistics.fmean(
            row["real_specificity"] for row in low_quality
        ),
        "critical_macro_balanced_accuracy": statistics.fmean(
            row["balanced_accuracy"] for row in critical
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--stage", choices=("canary", "scaled"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = {
        row["sample_id"]: row
        for row in read_manifest(args.manifest)
        if row["role"] == "calibration"
    }
    baseline = summarize(manifest, args.baseline)
    candidate = summarize(manifest, args.candidate)
    deltas = {
        "clean_balanced_accuracy": candidate["views"]["clean"]["balanced_accuracy"]
        - baseline["views"]["clean"]["balanced_accuracy"],
        "low_quality_macro_balanced_accuracy": candidate[
            "low_quality_macro_balanced_accuracy"
        ]
        - baseline["low_quality_macro_balanced_accuracy"],
        "low_quality_macro_fake_recall": candidate["low_quality_macro_fake_recall"]
        - baseline["low_quality_macro_fake_recall"],
        "low_quality_macro_real_specificity": candidate[
            "low_quality_macro_real_specificity"
        ]
        - baseline["low_quality_macro_real_specificity"],
        "critical_macro_balanced_accuracy": candidate["critical_macro_balanced_accuracy"]
        - baseline["critical_macro_balanced_accuracy"],
    }
    gates = {
        "low_quality_ba_gain_at_least_5_points": deltas[
            "low_quality_macro_balanced_accuracy"
        ]
        >= 0.05,
        "critical_ba_gain_at_least_5_points": deltas["critical_macro_balanced_accuracy"]
        >= 0.05,
        "low_quality_fake_recall_gain_at_least_8_points": deltas[
            "low_quality_macro_fake_recall"
        ]
        >= 0.08,
        "clean_ba_regression_at_most_1_point": deltas["clean_balanced_accuracy"] >= -0.01,
        "low_quality_specificity_regression_at_most_3_points": deltas[
            "low_quality_macro_real_specificity"
        ]
        >= -0.03,
    }
    payload = {
        "protocol": "aiblink-low-quality-ablation/0.1.0",
        "stage": args.stage,
        "selection_role": "calibration",
        "test_role_opened": False,
        "baseline": baseline,
        "candidate": candidate,
        "deltas": deltas,
        "gates": gates,
        "promote": all(gates.values()),
    }
    atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
