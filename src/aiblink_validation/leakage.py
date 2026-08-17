from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_json, read_manifest
from .metrics import binary_metrics


def _feature_row(row: dict[str, Any]) -> list[float]:
    width, height = max(1, int(row.get("width") or 1)), max(1, int(row.get("height") or 1))
    try:
        file_size = Path(row["path"]).stat().st_size
    except OSError:
        file_size = 0
    mime = str(row.get("mime", "")).lower()
    suffix = Path(row["path"]).suffix.lower()
    pixels = width * height
    return [
        math.log(width),
        math.log(height),
        math.log(width / height),
        math.log(pixels),
        math.log1p(file_size),
        file_size / pixels,
        float(width > height),
        float(width == height),
        float("jpeg" in mime or suffix in {".jpg", ".jpeg"}),
        float("png" in mime or suffix == ".png"),
        float("webp" in mime or suffix == ".webp"),
    ]


FEATURE_NAMES = [
    "log_width", "log_height", "log_aspect", "log_pixels", "log_file_bytes",
    "bytes_per_pixel", "landscape", "square", "jpeg", "png", "webp",
]


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    positives, negatives = max(1, int(labels.sum())), max(1, int((labels == 0).sum()))
    weights = np.where(labels == 1, len(labels) / (2 * positives), len(labels) / (2 * negatives))
    design = np.column_stack((features, np.ones(len(features))))
    beta = np.zeros(design.shape[1])
    ridge = np.eye(design.shape[1]) * 1e-3
    ridge[-1, -1] = 1e-8
    for _ in range(100):
        logits = np.clip(design @ beta, -40, 40)
        probability = 1 / (1 + np.exp(-logits))
        gradient = design.T @ (weights * (probability - labels)) + ridge @ beta
        curvature = weights * probability * (1 - probability)
        hessian = (design.T * curvature) @ design + ridge
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.linalg.norm(step) < 1e-8:
            break
    return beta


def metadata_leakage_audit(
    manifest_path: str | Path, output: str | Path, roles: set[str], folds: int = 5, seed: int = 323
) -> dict[str, Any]:
    rows = [row for row in read_manifest(manifest_path) if row["role"] in roles]
    if len(rows) < 20 or len({row["label"] for row in rows}) < 2:
        raise ValueError("metadata audit requires at least 20 rows and both classes")
    raw = np.array([_feature_row(row) for row in rows], dtype=np.float64)
    labels = np.array([row["label"] for row in rows], dtype=np.int8)
    class_group_ids = {
        label: sorted({str(row["group_id"]) for row in rows if int(row["label"]) == label})
        for label in (0, 1)
    }
    effective_folds = min(folds, *(len(values) for values in class_group_ids.values()))
    group_cv_available = effective_folds >= 2
    if group_cv_available:
        group_folds = {}
        for label, group_ids in class_group_ids.items():
            ordered = sorted(
                group_ids,
                key=lambda group_id: hashlib.sha256(f"{seed}\0{label}\0{group_id}".encode()).digest(),
            )
            for index, group_id in enumerate(ordered):
                group_folds[(label, group_id)] = index % effective_folds
        fold_ids = np.array(
            [group_folds[(int(row["label"]), str(row["group_id"]))] for row in rows]
        )
        cross_validation = "class-stratified provenance-group folds"
    else:
        effective_folds = folds
        fold_ids = np.array(
            [
                int.from_bytes(
                    hashlib.sha256(f"{seed}\0{row['label']}\0{row['sample_id']}".encode()).digest()[:8],
                    "big",
                )
                % effective_folds
                for row in rows
            ]
        )
        cross_validation = "fallback class-stratified image folds (insufficient provenance groups)"
    probabilities = np.full(len(rows), np.nan)
    coefficients = []
    for fold in range(effective_folds):
        train, test = fold_ids != fold, fold_ids == fold
        if not test.any() or len(np.unique(labels[train])) < 2:
            continue
        mean, std = raw[train].mean(axis=0), raw[train].std(axis=0)
        std[std < 1e-8] = 1
        normalized_train = (raw[train] - mean) / std
        beta = _fit_logistic(normalized_train, labels[train])
        design_test = np.column_stack(((raw[test] - mean) / std, np.ones(test.sum())))
        probabilities[test] = 1 / (1 + np.exp(-np.clip(design_test @ beta, -40, 40)))
        coefficients.append(beta[:-1])
    keep = np.isfinite(probabilities)
    metrics = binary_metrics(labels[keep], probabilities[keep], 0.5)
    standardized_differences = {}
    for index, name in enumerate(FEATURE_NAMES):
        pooled_std = raw[:, index].std()
        standardized_differences[name] = (
            float((raw[labels == 1, index].mean() - raw[labels == 0, index].mean()) / pooled_std)
            if pooled_std > 1e-12
            else 0.0
        )
    class_groups = {str(label): len(class_group_ids[label]) for label in (0, 1)}
    metadata_gate = metrics["roc_auc"] <= 0.60
    provenance_gate = group_cv_available
    result = {
        "valid": metadata_gate and provenance_gate,
        "roles": sorted(roles),
        "n": len(rows),
        "features": FEATURE_NAMES,
        "cross_validation": cross_validation,
        "requested_folds": folds,
        "effective_folds": effective_folds,
        "provenance_group_cv_available": group_cv_available,
        "provenance_groups_by_class": class_groups,
        "metrics": metrics,
        "mean_absolute_coefficients": {
            name: float(np.mean(np.abs(np.array(coefficients)[:, index])))
            for index, name in enumerate(FEATURE_NAMES)
        },
        "standardized_class_differences": standardized_differences,
        "gate": {"name": "metadata_only_roc_auc", "passed": metadata_gate, "maximum": 0.60},
        "gates": [
            {"name": "metadata_only_roc_auc", "passed": metadata_gate, "maximum": 0.60},
            {
                "name": "provenance_group_cross_validation",
                "passed": provenance_gate,
                "detail": f"groups_by_class={class_groups}",
            },
        ],
    }
    atomic_json(output, result)
    return result
