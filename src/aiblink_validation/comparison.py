from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import atomic_json
from .metrics import binary_metrics


def _read(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["sample_id"], row["view"])
            if key in rows:
                raise ValueError(f"duplicate comparison key {key}")
            row["label"] = int(row["label"])
            row["probability"] = float(row["probability"])
            rows[key] = row
    return rows


def _paired_intervals(
    rows: list[dict[str, Any]], threshold: float, replicates: int, seed: int, confidence: float
) -> dict[str, Any]:
    clusters: dict[int, dict[str, list[int]]] = {0: {}, 1: {}}
    for index, row in enumerate(rows):
        clusters[row["label"]].setdefault(row["sample_id"], []).append(index)
    rng = np.random.default_rng(seed)
    values = {name: np.empty(replicates) for name in ("balanced_accuracy", "fake_recall", "real_specificity")}
    for replicate in range(replicates):
        indexes = []
        for label in (0, 1):
            ids = list(clusters[label])
            for sample_id in rng.choice(ids, len(ids), replace=True):
                indexes.extend(clusters[label][sample_id])
        labels = np.array([rows[index]["label"] for index in indexes])
        champion = np.array([rows[index]["champion_probability"] for index in indexes])
        challenger = np.array([rows[index]["challenger_probability"] for index in indexes])
        left = binary_metrics(labels, champion, threshold)
        right = binary_metrics(labels, challenger, threshold)
        for name in values:
            values[name][replicate] = right[name] - left[name]
    alpha = (1 - confidence) / 2
    return {
        name: {"low": float(np.quantile(value, alpha)), "high": float(np.quantile(value, 1 - alpha))}
        for name, value in values.items()
    }


def compare(
    champion_path: str | Path,
    challenger_path: str | Path,
    output: str | Path,
    threshold: float = 0.65,
    replicates: int = 2000,
    seed: int = 323,
    confidence: float = 0.95,
) -> dict[str, Any]:
    champion = _read(champion_path)
    challenger = _read(challenger_path)
    if champion.keys() != challenger.keys():
        missing_left = sorted(challenger.keys() - champion.keys())[:5]
        missing_right = sorted(champion.keys() - challenger.keys())[:5]
        raise ValueError(f"prediction sets differ: champion_missing={missing_left}, challenger_missing={missing_right}")
    rows = []
    for key in champion:
        left, right = champion[key], challenger[key]
        if left["label"] != right["label"] or left["group_id"] != right["group_id"]:
            raise ValueError(f"label/provenance mismatch at {key}")
        rows.append(
            {
                "sample_id": key[0],
                "view": key[1],
                "label": left["label"],
                "group_id": left["group_id"],
                "champion_probability": left["probability"],
                "challenger_probability": right["probability"],
            }
        )
    labels = np.array([row["label"] for row in rows])
    left_probability = np.array([row["champion_probability"] for row in rows])
    right_probability = np.array([row["challenger_probability"] for row in rows])
    left_metrics = binary_metrics(labels, left_probability, threshold)
    right_metrics = binary_metrics(labels, right_probability, threshold)
    tracked = ("balanced_accuracy", "fake_recall", "real_specificity", "roc_auc", "brier")
    delta = {name: right_metrics[name] - left_metrics[name] for name in tracked}
    intervals = _paired_intervals(rows, threshold, replicates, seed, confidence)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    group_delta = []
    for group_id, subset in grouped.items():
        label = subset[0]["label"]
        left_correct = [row["champion_probability"] >= threshold for row in subset]
        right_correct = [row["challenger_probability"] >= threshold for row in subset]
        if not label:
            left_correct = [not value for value in left_correct]
            right_correct = [not value for value in right_correct]
        group_delta.append(
            {
                "group_id": group_id,
                "label": label,
                "n": len(subset),
                "champion_rate": float(np.mean(left_correct)),
                "challenger_rate": float(np.mean(right_correct)),
                "delta": float(np.mean(right_correct) - np.mean(left_correct)),
            }
        )
    result = {
        "paired": True,
        "threshold": threshold,
        "n_rows": len(rows),
        "n_unique_images": len({row["sample_id"] for row in rows}),
        "champion": left_metrics,
        "challenger": right_metrics,
        "delta": delta,
        "delta_bootstrap_95": intervals,
        "group_delta": sorted(group_delta, key=lambda row: (row["delta"], row["group_id"])),
        "promotion_signal": intervals["balanced_accuracy"]["low"] > 0,
    }
    atomic_json(output, result)
    return result

