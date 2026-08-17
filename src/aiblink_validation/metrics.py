from __future__ import annotations

import math
from typing import Any

import numpy as np

EPS = np.finfo(np.float64).eps


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1 + exp_values)
    return output


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    positives = int(y.sum())
    if not positives:
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    sorted_y = y[order]
    cumulative = np.cumsum(sorted_y)
    precision = cumulative / np.arange(1, len(y) + 1)
    return float((precision * sorted_y).sum() / positives)


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(y, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if mask.any():
            result += mask.mean() * abs(float(y[mask].mean() - probability[mask].mean()))
    return float(result) if total else float("nan")


def binary_metrics(y: np.ndarray, probability: np.ndarray, threshold: float = 0.65) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    if len(y) != len(probability) or not len(y):
        raise ValueError("labels and probabilities must be non-empty and equal length")
    prediction = probability >= threshold
    tp = int(((y == 1) & prediction).sum())
    tn = int(((y == 0) & ~prediction).sum())
    fp = int(((y == 0) & prediction).sum())
    fn = int(((y == 1) & ~prediction).sum())
    tpr = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(tp * tn - fp * fn, denominator)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    log_loss = -float(np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    return {
        "n": len(y),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "threshold": threshold,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "balanced_accuracy": float((tpr + tnr) / 2),
        "accuracy": float((tp + tn) / len(y)),
        "fake_recall": tpr,
        "real_specificity": tnr,
        "precision": precision,
        "negative_predictive_value": npv,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc(y, probability),
        "average_precision": average_precision(y, probability),
        "brier": float(np.mean((probability - y) ** 2)),
        "log_loss": log_loss,
        "ece_15": expected_calibration_error(y, probability),
    }


def optimal_balanced_threshold(y: np.ndarray, raw_score: np.ndarray) -> tuple[float, float]:
    """Exact optimum for the `score >= threshold` decision rule.

    Ties choose the midpoint of the optimal threshold interval nearest zero,
    which is stable and avoids a quantile-grid approximation.
    """
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(raw_score, dtype=np.float64)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if not positives or not negatives:
        raise ValueError("both classes are required to optimize balanced accuracy")
    unique = np.unique(score)
    candidates = np.concatenate(([np.nextafter(unique[0], -np.inf)], unique, [np.nextafter(unique[-1], np.inf)]))
    best_value = -1.0
    best: list[float] = []
    for threshold in candidates:
        prediction = score >= threshold
        value = 0.5 * (float(prediction[y == 1].mean()) + float((~prediction[y == 0]).mean()))
        if value > best_value + 1e-15:
            best_value, best = value, [float(threshold)]
        elif abs(value - best_value) <= 1e-15:
            best.append(float(threshold))
    threshold = min(best, key=lambda value: (abs(value), value))
    return threshold, best_value


def macro_group_tail(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[bool]] = {}
    for row in rows:
        key = (int(row["label"]), str(row["group_id"]))
        correct = row["probability"] >= threshold if key[0] else row["probability"] < threshold
        grouped.setdefault(key, []).append(bool(correct))
    by_class: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for (label, group), values in grouped.items():
        by_class[label].append({"group_id": group, "n": len(values), "rate": float(np.mean(values))})
    for entries in by_class.values():
        entries.sort(key=lambda entry: (entry["rate"], entry["group_id"]))
    fake_rates = np.array([entry["rate"] for entry in by_class[1]], dtype=float)
    real_rates = np.array([entry["rate"] for entry in by_class[0]], dtype=float)
    macro_fake = float(fake_rates.mean()) if len(fake_rates) else float("nan")
    macro_real = float(real_rates.mean()) if len(real_rates) else float("nan")
    return {
        "macro_balanced_accuracy": (macro_fake + macro_real) / 2,
        "macro_fake_recall": macro_fake,
        "macro_real_specificity": macro_real,
        "fake_group_p10": float(np.quantile(fake_rates, 0.1)) if len(fake_rates) else float("nan"),
        "real_group_p10": float(np.quantile(real_rates, 0.1)) if len(real_rates) else float("nan"),
        "worst_fake_groups": by_class[1][:10],
        "worst_real_groups": by_class[0][:10],
    }


def clustered_bootstrap(
    rows: list[dict[str, Any]], threshold: float, replicates: int, seed: int, confidence: float
) -> dict[str, dict[str, float]]:
    """Stratified cluster bootstrap; all views of an original image move together."""
    clusters: dict[int, dict[str, list[int]]] = {0: {}, 1: {}}
    y = np.array([int(row["label"]) for row in rows], dtype=np.int8)
    probability = np.array([float(row["probability"]) for row in rows], dtype=np.float64)
    for index, row in enumerate(rows):
        clusters[int(row["label"])].setdefault(str(row["sample_id"]), []).append(index)
    if not clusters[0] or not clusters[1]:
        raise ValueError("both classes are required for bootstrap")
    rng = np.random.default_rng(seed)
    tracked = ("balanced_accuracy", "fake_recall", "real_specificity", "roc_auc", "brier")
    values = {name: np.empty(replicates, dtype=np.float64) for name in tracked}
    ids = {label: np.array(list(class_clusters), dtype=object) for label, class_clusters in clusters.items()}
    for replicate in range(replicates):
        indexes: list[int] = []
        for label in (0, 1):
            selected = rng.choice(ids[label], size=len(ids[label]), replace=True)
            for sample_id in selected:
                indexes.extend(clusters[label][str(sample_id)])
        metrics = binary_metrics(y[indexes], probability[indexes], threshold)
        for name in tracked:
            values[name][replicate] = metrics[name]
    alpha = (1 - confidence) / 2
    return {
        name: {
            "low": float(np.nanquantile(series, alpha)),
            "high": float(np.nanquantile(series, 1 - alpha)),
        }
        for name, series in values.items()
    }

