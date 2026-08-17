#!/usr/bin/env python3
"""Decide whether a large scratch teacher is strong enough to justify distillation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return float("nan")
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


def load_predictions(path: Path, manifest: dict[str, dict]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for row in read_jsonl(path):
        if row.get("error") or row.get("raw_logit") is None:
            continue
        sample_id = str(row["sample_id"])
        if sample_id not in manifest:
            raise ValueError(f"prediction {sample_id!r} is absent from calibration manifest")
        result[sample_id][str(row["view"])] = float(row["raw_logit"])
    return dict(result)


def view_aucs(predictions: dict[str, dict[str, float]], manifest: dict[str, dict]) -> dict[str, float]:
    views = sorted(set.intersection(*(set(values) for values in predictions.values())))
    output = {}
    for view in views:
        sample_ids = sorted(predictions)
        labels = np.asarray([int(manifest[sample_id]["label"]) for sample_id in sample_ids])
        scores = np.asarray([predictions[sample_id][view] for sample_id in sample_ids])
        output[view] = roc_auc(labels, scores)
    return output


def paired_group_bootstrap(
    teacher: dict[str, dict[str, float]],
    student: dict[str, dict[str, float]],
    manifest: dict[str, dict],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    common = sorted(set(teacher) & set(student))
    views = sorted(set.intersection(*(set(teacher[sample_id]) & set(student[sample_id]) for sample_id in common)))
    clusters: dict[int, dict[str, list[str]]] = {0: defaultdict(list), 1: defaultdict(list)}
    for sample_id in common:
        row = manifest[sample_id]
        clusters[int(row["label"])][str(row["group_id"])].append(sample_id)
    if any(not clusters[label] for label in (0, 1)):
        raise ValueError("teacher gate requires both real and fake calibration groups")
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    group_ids = {label: np.asarray(sorted(clusters[label]), dtype=object) for label in (0, 1)}
    for replicate in range(replicates):
        sampled: list[str] = []
        for label in (0, 1):
            chosen = rng.choice(group_ids[label], size=len(group_ids[label]), replace=True)
            for group_id in chosen:
                sampled.extend(clusters[label][str(group_id)])
        labels = np.asarray([int(manifest[sample_id]["label"]) for sample_id in sampled])
        teacher_auc = np.mean([
            roc_auc(labels, np.asarray([teacher[sample_id][view] for sample_id in sampled]))
            for view in views
        ])
        student_auc = np.mean([
            roc_auc(labels, np.asarray([student[sample_id][view] for sample_id in sampled]))
            for view in views
        ])
        deltas[replicate] = teacher_auc - student_auc
    return {
        "replicates": replicates,
        "confidence": 0.90,
        "mean_delta": float(np.nanmean(deltas)),
        "low": float(np.nanquantile(deltas, 0.05)),
        "high": float(np.nanquantile(deltas, 0.95)),
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
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--teacher-id", default="convnext_large_teacher_scratch")
    parser.add_argument("--teacher-predictions", type=Path, required=True)
    parser.add_argument("--teacher-training", type=Path, required=True)
    parser.add_argument("--student-prediction", action="append", default=[], help="candidate_id=path")
    parser.add_argument("--student-training", action="append", default=[], help="candidate_id=path")
    parser.add_argument("--minimum-oof-ba-delta", type=float, default=0.01)
    parser.add_argument("--maximum-view-auc-regression", type=float, default=0.005)
    parser.add_argument("--minimum-stable-validations", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=323)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="") as handle:
        manifest = {
            str(row["sample_id"]): row
            for row in csv.DictReader(handle)
            if row["role"] == "calibration"
        }
    ranking = json.loads(args.ranking.read_text())
    all_ranked = ranking["ranking"] + ranking.get("excluded", [])
    by_id = {row["candidate_id"]: row for row in all_ranked}
    if args.teacher_id not in by_id:
        raise ValueError("teacher is missing from ranking")
    students = [row for row in ranking["ranking"] if row["candidate_id"] != args.teacher_id]
    if not students:
        raise ValueError("no eligible smaller student candidate is present")
    best_student = max(students, key=lambda row: row["oof_macro_balanced_accuracy"])
    student_id = best_student["candidate_id"]

    prediction_paths = dict(item.split("=", 1) for item in args.student_prediction)
    training_paths = dict(item.split("=", 1) for item in args.student_training)
    if student_id not in prediction_paths or student_id not in training_paths:
        raise ValueError(f"missing prediction or training input for selected student {student_id}")

    teacher_rank = by_id[args.teacher_id]
    oof_delta = float(teacher_rank["oof_macro_balanced_accuracy"] - best_student["oof_macro_balanced_accuracy"])
    teacher_predictions = load_predictions(args.teacher_predictions, manifest)
    student_predictions = load_predictions(Path(prediction_paths[student_id]), manifest)
    teacher_aucs = view_aucs(teacher_predictions, manifest)
    student_aucs = view_aucs(student_predictions, manifest)
    shared_views = sorted(set(teacher_aucs) & set(student_aucs))
    view_deltas = {view: teacher_aucs[view] - student_aucs[view] for view in shared_views}

    teacher_training = json.loads(args.teacher_training.read_text())
    student_training = json.loads(Path(training_paths[student_id]).read_text())
    stable_threshold = float(student_training["best_metric"]) + 0.005
    stable_validations = sum(
        int(validation["step"] > 0 and validation["logo_macro_balanced_accuracy"] >= stable_threshold)
        for validation in teacher_training["validations"]
    )
    bootstrap = paired_group_bootstrap(
        teacher_predictions, student_predictions, manifest, args.bootstrap_replicates, args.seed
    )

    checks = {
        "oof_macro_ba_margin": oof_delta >= args.minimum_oof_ba_delta,
        "paired_bootstrap_auc_lower_bound_positive": bootstrap["low"] > 0.0,
        "no_material_per_view_auc_regression": all(
            delta >= -args.maximum_view_auc_regression for delta in view_deltas.values()
        ),
        "repeatable_across_validation_checkpoints": stable_validations >= args.minimum_stable_validations,
    }
    passed = all(checks.values())
    payload = {
        "protocol": "aiblink-teacher-superiority-gate/0.1.0",
        "selection_role": "calibration",
        "test_role_opened": False,
        "teacher_id": args.teacher_id,
        "best_student_id": student_id,
        "teacher_oof_macro_balanced_accuracy": teacher_rank["oof_macro_balanced_accuracy"],
        "student_oof_macro_balanced_accuracy": best_student["oof_macro_balanced_accuracy"],
        "oof_macro_balanced_accuracy_delta": oof_delta,
        "minimum_required_oof_delta": args.minimum_oof_ba_delta,
        "teacher_view_roc_auc": teacher_aucs,
        "student_view_roc_auc": student_aucs,
        "view_roc_auc_deltas": view_deltas,
        "maximum_allowed_view_auc_regression": args.maximum_view_auc_regression,
        "paired_group_bootstrap_mean_view_auc_delta": bootstrap,
        "stable_validation_threshold": stable_threshold,
        "stable_validation_count": stable_validations,
        "minimum_stable_validations": args.minimum_stable_validations,
        "checks": checks,
        "gate": "pass" if passed else "fail",
        "distillation_authorized": passed,
        "reason": (
            "teacher demonstrated a meaningful and repeatable robust-calibration advantage"
            if passed
            else "teacher superiority was not established; do not distill"
        ),
    }
    if any(not math.isfinite(float(value)) for value in view_deltas.values()):
        raise ValueError("non-finite view comparison")
    atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
