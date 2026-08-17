#!/usr/bin/env python3
"""Summarize equal-exposure canary loss, gradient, and validation curves."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from pathlib import Path


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


def linear_slope(points: list[tuple[float, float]]) -> float:
    x_mean = statistics.fmean(x for x, _ in points)
    y_mean = statistics.fmean(y for _, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator if denominator else 0.0


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text())
    history = data["history"]
    validations = data["validations"]
    losses = [float(row["loss"]) for row in history]
    gradients = [float(row["gradient_norm"]) for row in history]
    if len(losses) < 6 or len(validations) < 3:
        raise ValueError(f"incomplete canary curve in {path}")
    window = min(3, len(losses) // 2)
    initial_loss = statistics.median(losses[:window])
    final_loss = statistics.median(losses[-window:])
    loss_points = [(float(row["step"]), float(row["loss"])) for row in history]
    loss_slope_per_100_steps = linear_slope(loss_points) * 100.0
    tail = losses[len(losses) // 2 :]
    tail_median = statistics.median(tail)
    tail_mad = statistics.median(abs(value - tail_median) for value in tail)
    sorted_gradients = sorted(gradients)
    gradient_p95 = sorted_gradients[min(len(sorted_gradients) - 1, round(0.95 * (len(sorted_gradients) - 1)))]
    validation_curve = [
        {
            "step": int(row["step"]),
            "macro_balanced_accuracy": float(row["logo_macro_balanced_accuracy"]),
            "pooled_balanced_accuracy": float(row["logo_pooled_balanced_accuracy"]),
            "log_loss": float(row["logo_log_loss"]),
        }
        for row in validations
    ]
    finite = all(math.isfinite(value) for value in losses + gradients)
    checks = {
        "finite_loss_and_gradients": finite,
        "no_decode_failures": int(data["decode_failures"]) == 0,
        "final_loss_not_materially_worse": final_loss <= 1.10 * initial_loss,
        # Ignore a single step-1 transient: sustained instability is captured by p95.
        "gradient_norm_not_exploding": gradient_p95 < 100.0,
        "robust_validation_not_regressing": validation_curve[-1]["macro_balanced_accuracy"] >= validation_curve[0]["macro_balanced_accuracy"] - 0.01,
    }
    return {
        "candidate_id": data["candidate"]["candidate_id"],
        "competition_status": data["candidate"]["competition_status"],
        "parameter_count": data.get("parameter_count"),
        "steps": int(data["steps"]),
        "samples_seen": int(data["samples_seen"]),
        "seconds": float(data["seconds"]),
        "samples_per_second": float(data["samples_per_second"]),
        "initial_loss_median": initial_loss,
        "final_loss_median": final_loss,
        "relative_loss_reduction": (initial_loss - final_loss) / max(initial_loss, 1e-12),
        "loss_slope_per_100_steps": loss_slope_per_100_steps,
        "tail_loss_mad": tail_mad,
        "gradient_norm_p95": gradient_p95,
        "gradient_norm_max": max(gradients),
        "initial_validation_macro_ba": validation_curve[0]["macro_balanced_accuracy"],
        "final_validation_macro_ba": validation_curve[-1]["macro_balanced_accuracy"],
        "best_validation_macro_ba": max(row["macro_balanced_accuracy"] for row in validation_curve),
        "validation_curve": validation_curve,
        "checks": checks,
        "healthy": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidates = [summarize(path) for path in args.training]
    healthy = [row for row in candidates if row["healthy"]]
    healthiest = min(
        healthy,
        key=lambda row: (row["final_loss_median"], row["loss_slope_per_100_steps"], row["gradient_norm_p95"]),
    ) if healthy else None
    best_validation = max(
        healthy,
        key=lambda row: (row["final_validation_macro_ba"], row["best_validation_macro_ba"]),
    ) if healthy else None
    eligible = [row for row in healthy if row["competition_status"] == "eligible"]
    recommended = max(
        eligible,
        key=lambda row: (row["final_validation_macro_ba"], row["best_validation_macro_ba"]),
    ) if eligible else None
    payload = {
        "protocol": "aiblink-equal-exposure-robust-canary/0.1.0",
        "selection_role": "calibration",
        "test_role_opened": False,
        "equal_steps": len({row["steps"] for row in candidates}) == 1,
        "equal_samples": len({row["samples_seen"] for row in candidates}) == 1,
        "candidates": candidates,
        "healthiest_loss_curve_candidate": healthiest["candidate_id"] if healthiest else None,
        "best_canary_validation_candidate": best_validation["candidate_id"] if best_validation else None,
        "recommended_eligible_candidate": recommended["candidate_id"] if recommended else None,
        "automatic_full_training_started": False,
    }
    atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
