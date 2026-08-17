#!/usr/bin/env python3
"""Create the final combined training, robustness, diagnostic, and deployment report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1 + exp_values)
    return output


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.65) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(probabilities) >= threshold
    fake = labels == 1
    real = ~fake
    recall = float(predictions[fake].mean())
    specificity = float((~predictions[real]).mean())
    return {
        "n": len(labels),
        "balanced_accuracy": (recall + specificity) / 2,
        "fake_recall": recall,
        "real_specificity": specificity,
        "roc_auc": auc(labels, probabilities),
    }


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probabilities(path: Path, calibration: dict) -> list[dict]:
    rows = read_jsonl(path)
    logits = np.asarray([float(row["raw_logit"]) for row in rows if not row.get("error")])
    transformed = sigmoid(calibration["slope"] * logits + calibration["intercept"])
    result = []
    cursor = 0
    for row in rows:
        if row.get("error"):
            continue
        result.append({**row, "probability": float(transformed[cursor])})
        cursor += 1
    return result


def positive_diagnostic(path: Path, calibration: dict, threshold: float = 0.65) -> dict:
    rows = probabilities(path, calibration)
    by_view: dict[str, list[bool]] = defaultdict(list)
    by_sample: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        detected = row["probability"] >= threshold
        by_view[str(row["view"])].append(detected)
        by_sample[str(row["sample_id"])].append(detected)
    return {
        "unique_images": len(by_sample),
        "recall_by_view": {view: float(np.mean(values)) for view, values in sorted(by_view.items())},
        "all_views_recall": float(np.mean([all(values) for values in by_sample.values()])),
    }


def ledger_parity(pytorch_path: Path, onnx_path: Path, calibration: dict, threshold: float = 0.65) -> dict:
    def keyed(path: Path) -> dict[tuple[str, str], float]:
        return {
            (str(row["sample_id"]), str(row["view"])): float(row["raw_logit"])
            for row in read_jsonl(path)
            if not row.get("error") and row.get("raw_logit") is not None
        }
    pytorch, onnx = keyed(pytorch_path), keyed(onnx_path)
    if pytorch.keys() != onnx.keys():
        raise ValueError(f"deployment ledger mismatch: pytorch={len(pytorch)} onnx={len(onnx)}")
    keys = sorted(pytorch)
    first = np.asarray([pytorch[key] for key in keys])
    second = np.asarray([onnx[key] for key in keys])
    delta = np.abs(first - second)
    first_probability = sigmoid(calibration["slope"] * first + calibration["intercept"])
    second_probability = sigmoid(calibration["slope"] * second + calibration["intercept"])
    disagreement = (first_probability >= threshold) != (second_probability >= threshold)
    return {
        "predictions": len(keys),
        "max_abs_logit_delta": float(delta.max()),
        "mean_abs_logit_delta": float(delta.mean()),
        "threshold_decision_disagreements": int(disagreement.sum()),
        "threshold_decision_disagreement_rate": float(disagreement.mean()),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--onnx-predictions", type=Path, required=True)
    parser.add_argument("--openrouter-predictions", type=Path, required=True)
    parser.add_argument("--recent-predictions", type=Path, required=True)
    parser.add_argument("--redteam", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    preparation = json.loads(args.preparation.read_text())
    audit = json.loads(args.audit.read_text())
    training = json.loads(args.training.read_text())
    ranking = json.loads(args.ranking.read_text())
    redteam = json.loads(args.redteam.read_text())
    export = json.loads(args.export.read_text())
    selected = ranking["ranking"][0]
    calibration = selected["final_calibrator"]
    with args.manifest.open(newline="") as handle:
        manifest = {row["sample_id"]: row for row in csv.DictReader(handle) if row["role"] == "calibration"}
    calibrated_rows = probabilities(args.calibration_predictions, calibration)
    by_view: dict[str, list[dict]] = defaultdict(list)
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in calibrated_rows:
        annotated = {**row, "label": int(manifest[row["sample_id"]]["label"])}
        by_view[str(row["view"])].append(annotated)
        by_sample[str(row["sample_id"])].append(annotated)
    view_metrics = {
        view: metrics(
            np.asarray([row["label"] for row in rows]),
            np.asarray([row["probability"] for row in rows]),
        )
        for view, rows in sorted(by_view.items())
    }
    robust_labels, robust_probabilities = [], []
    for rows in by_sample.values():
        label = rows[0]["label"]
        robust_labels.append(label)
        robust_probabilities.append(min(row["probability"] for row in rows) if label else max(row["probability"] for row in rows))
    robust = metrics(np.asarray(robust_labels), np.asarray(robust_probabilities))
    parity = ledger_parity(args.calibration_predictions, args.onnx_predictions, calibration)
    valid = bool(
        audit.get("valid")
        and not training["decode_failures"]
        and selected["oof_macro_balanced_accuracy"] >= 0.75
        and parity["threshold_decision_disagreement_rate"] <= 0.001
    )
    payload = {
        "protocol": "aiblink-final-community-forensics/0.1.0",
        "valid": valid,
        "candidate_id": "commfor_384",
        "competition_status": selected["competition_status"],
        "license": selected["license"],
        "threshold": 0.65,
        "test_role_opened": False,
        "dataset": {
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "combined_rows": preparation["combined_rows"],
            "new_train_rows": preparation["new_train_rows"],
            "diagnostic_recent_holdout_rows": preparation["diagnostic_recent_holdout_rows"],
            "audit_valid": audit.get("valid"),
            "audit_issues": audit.get("issues", []),
        },
        "training": {
            "steps": training["steps"],
            "samples_seen": training["samples_seen"],
            "best_step": training["best_step"],
            "best_metric": training["best_metric"],
            "seconds": training["seconds"],
            "decode_failures": training["decode_failures"],
            "selected_checkpoint": "explicit" if args.checkpoint else "best",
            "selected_checkpoint_step": training["steps"] if args.checkpoint else training["best_step"],
            "checkpoint_sha256": (
                sha256_file(args.checkpoint)
                if args.checkpoint
                else training["best_checkpoint_sha256"]
            ),
        },
        "calibration": {
            "method": selected["selected_calibrator"],
            "parameters": calibration,
            "oof_macro_balanced_accuracy": selected["oof_macro_balanced_accuracy"],
            "view_metrics": view_metrics,
            "all_views_worst_case": robust,
        },
        "diagnostics": {
            "openrouter_90": positive_diagnostic(args.openrouter_predictions, calibration),
            "recent_hf_holdout": positive_diagnostic(args.recent_predictions, calibration),
        },
        "redteam": {
            "development_calibration_only": True,
            "valid": redteam["valid"],
            "worst_declared_attack": redteam["worst_declared_attack"],
            "worst_variant_per_image": redteam["worst_variant_per_image"],
            "attack_success": redteam["attack_success"],
            "gates": redteam["gates"],
        },
        "deployment": {"export": export, "full_calibration_ledger_parity": parity},
    }
    atomic_json(args.out / "report.json", payload)
    lines = [
        "# Final Community Forensics validation",
        "",
        f"**Status:** {'PASS' if valid else 'REVIEW REQUIRED'}",
        "",
        f"Combined manifest: {payload['dataset']['combined_rows']:,} rows; new frontier training images: {payload['dataset']['new_train_rows']:,}.",
        f"Best training step: {payload['training']['best_step']}; calibration-only OOF macro BA: **{selected['oof_macro_balanced_accuracy']:.4f}**.",
        f"All-clean/web/hard worst-view BA: **{robust['balanced_accuracy']:.4f}**.",
        f"OpenRouter all-view recall: **{payload['diagnostics']['openrouter_90']['all_views_recall']:.4f}**.",
        f"Recent-HF holdout all-view recall: **{payload['diagnostics']['recent_hf_holdout']['all_views_recall']:.4f}**.",
        f"ONNX decision disagreement rate: **{parity['threshold_decision_disagreement_rate']:.6f}**.",
        "",
        "The locked proxy test was not reopened. The 33-condition red-team report uses the calibration cohort and is development evidence, not an unbiased test estimate.",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
