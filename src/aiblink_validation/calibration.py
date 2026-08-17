from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import numpy as np

from .metrics import binary_metrics, optimal_balanced_threshold, sigmoid


@dataclass(frozen=True)
class Calibrator:
    method: str
    slope: float
    intercept: float
    raw_boundary: float
    target_threshold: float

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(self.slope * np.asarray(logits, dtype=np.float64) + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_logistic_slope(logits: np.ndarray, y: np.ndarray) -> float:
    """Fit a class-balanced logistic slope with Newton updates, constrained positive."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    positive = max(1, int(y.sum()))
    negative = max(1, len(y) - positive)
    weights = np.where(y == 1, len(y) / (2 * positive), len(y) / (2 * negative))
    x = np.column_stack((logits, np.ones(len(logits))))
    beta = np.array([1.0, 0.0], dtype=np.float64)
    ridge = np.diag([1e-4, 1e-8])
    for _ in range(100):
        probability = sigmoid(x @ beta)
        gradient = x.T @ (weights * (probability - y)) + ridge @ beta
        curvature = weights * probability * (1 - probability)
        hessian = (x.T * curvature) @ x + ridge
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return 1.0
        beta -= step
        if np.linalg.norm(step) < 1e-8:
            break
    return float(np.clip(beta[0], 1e-3, 100.0)) if np.isfinite(beta[0]) else 1.0


def fit_calibrator(logits: np.ndarray, y: np.ndarray, method: str, target_threshold: float = 0.65) -> Calibrator:
    if method == "identity":
        target_logit = float(np.log(target_threshold / (1 - target_threshold)))
        return Calibrator(method, 1.0, 0.0, target_logit, target_threshold)
    raw_boundary, _ = optimal_balanced_threshold(y, logits)
    slope = 1.0 if method == "bias" else _positive_logistic_slope(logits, y)
    if method not in {"bias", "platt"}:
        raise ValueError(f"unknown calibration method {method!r}")
    target_logit = float(np.log(target_threshold / (1 - target_threshold)))
    intercept = target_logit - slope * raw_boundary
    return Calibrator(method, slope, intercept, raw_boundary, target_threshold)


def _macro_heldout_score(rows: list[dict[str, Any]], threshold: float) -> float:
    group_rates: dict[tuple[int, str], list[bool]] = {}
    for row in rows:
        key = (int(row["label"]), str(row["group_id"]))
        correct = row["probability"] >= threshold if key[0] else row["probability"] < threshold
        group_rates.setdefault(key, []).append(bool(correct))
    per_class = []
    for label in (0, 1):
        rates = [np.mean(values) for (group_label, _), values in group_rates.items() if group_label == label]
        if rates:
            per_class.append(float(np.mean(rates)))
    return float(np.mean(per_class)) if per_class else float("nan")


def select_and_fit(
    rows: list[dict[str, Any]], candidates: list[str], target_threshold: float = 0.65
) -> tuple[Calibrator, dict[str, Any]]:
    """Select calibration complexity with leave-one-provenance-group-out predictions."""
    # "Leave one generator out" means fake-generator groups. Reals remain the
    # negative reference population, but are cross-fitted by sample so a real
    # image never trains and scores in the same fold.
    groups = sorted({str(row["group_id"]) for row in rows if int(row["label"]) == 1})
    if len(groups) < 3:
        raise ValueError("at least three calibration groups are required")
    diagnostics: dict[str, Any] = {"folds": len(groups), "candidates": {}}
    for method in candidates:
        heldout_rows: list[dict[str, Any]] = []
        fold_details = []
        for fold_index, held_group in enumerate(groups):
            def real_fold(row: dict[str, Any]) -> int:
                digest = hashlib.sha256(str(row["sample_id"]).encode()).digest()
                return int.from_bytes(digest[:8], "big") % len(groups)

            held = [
                dict(row)
                for row in rows
                if str(row["group_id"]) == held_group
                or (int(row["label"]) == 0 and real_fold(row) == fold_index)
            ]
            train = [
                row
                for row in rows
                if str(row["group_id"]) != held_group
                and not (int(row["label"]) == 0 and real_fold(row) == fold_index)
            ]
            train_y = np.array([row["label"] for row in train], dtype=np.int8)
            if len(np.unique(train_y)) < 2:
                raise ValueError(f"fold without both classes after holding out {held_group}")
            calibrator = fit_calibrator(
                np.array([row["raw_logit"] for row in train]), train_y, method, target_threshold
            )
            probabilities = calibrator.transform(np.array([row["raw_logit"] for row in held]))
            for row, probability in zip(held, probabilities, strict=True):
                row["probability"] = float(probability)
            heldout_rows.extend(held)
            correct = [
                row["probability"] >= target_threshold if row["label"] else row["probability"] < target_threshold
                for row in held
            ]
            fold_details.append({"group_id": held_group, "n": len(held), "accuracy": float(np.mean(correct))})
        y = np.array([row["label"] for row in heldout_rows])
        probability = np.array([row["probability"] for row in heldout_rows])
        pooled = binary_metrics(y, probability, target_threshold)
        diagnostics["candidates"][method] = {
            "logo_macro_balanced_accuracy": _macro_heldout_score(heldout_rows, target_threshold),
            "logo_pooled_balanced_accuracy": pooled["balanced_accuracy"],
            "logo_log_loss": pooled["log_loss"],
            "logo_brier": pooled["brier"],
            "folds": fold_details,
        }
    # BA is primary. Prefer lower log loss only outside a numerical tie; prefer bias on exact ties.
    selected = sorted(
        candidates,
        key=lambda method: (
            -diagnostics["candidates"][method]["logo_macro_balanced_accuracy"],
            diagnostics["candidates"][method]["logo_log_loss"],
            {"identity": 0, "bias": 1, "platt": 2}.get(method, 3),
        ),
    )[0]
    y = np.array([row["label"] for row in rows], dtype=np.int8)
    final = fit_calibrator(np.array([row["raw_logit"] for row in rows]), y, selected, target_threshold)
    diagnostics["selected"] = selected
    diagnostics["final"] = final.to_dict()
    return final, diagnostics
