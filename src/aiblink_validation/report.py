from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .audit import audit_manifest
from .calibration import select_and_fit
from .io import atomic_json, file_fingerprint, read_jsonl
from .metrics import binary_metrics, clustered_bootstrap, macro_group_tail, sigmoid


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def _load_predictions(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates = []
    for row in read_jsonl(path):
        key = (str(row["sample_id"]), str(row["view"]))
        if key in predictions:
            duplicates.append(key)
        predictions[key] = row
    if duplicates:
        raise ValueError(f"duplicate prediction keys, examples: {duplicates[:5]}")
    return predictions


def _join(
    manifest: list[dict[str, Any]], predictions: dict[tuple[str, str], dict[str, Any]], role: str, views: list[str]
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    rows: list[dict[str, Any]] = []
    missing = []
    for item in manifest:
        if item["role"] != role:
            continue
        for view in views:
            key = (str(item["sample_id"]), view)
            prediction = predictions.get(key)
            if prediction is None or prediction.get("raw_logit") is None:
                missing.append(key)
                continue
            annotated = {**item, "view": view, "raw_logit": float(prediction["raw_logit"])}
            if prediction.get("model_resize_short_edge") is not None:
                annotated["model_resize_short_edge"] = int(prediction["model_resize_short_edge"])
            rows.append(annotated)
    return rows, missing


def _slices(rows: list[dict[str, Any]], threshold: float, minimum: int) -> dict[str, Any]:
    for row in rows:
        try:
            minimum_side = min(int(row["width"]), int(row["height"]))
            row["resolution_bucket"] = (
                "min_side_lt_128"
                if minimum_side < 128
                else "min_side_128_to_255"
                if minimum_side < 256
                else "min_side_256_to_511"
                if minimum_side < 512
                else "min_side_512_to_1023"
                if minimum_side < 1024
                else "min_side_ge_1024"
            )
        except (KeyError, TypeError, ValueError):
            row["resolution_bucket"] = "unknown"
    result: dict[str, Any] = {}
    for field in ("view", "dataset", "resolution_bucket"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[field])].append(row)
        result[field] = {}
        for name, subset in sorted(grouped.items()):
            if len(subset) < minimum or len({row["label"] for row in subset}) < 2:
                continue
            result[field][name] = binary_metrics(
                np.array([row["label"] for row in subset]),
                np.array([row["probability"] for row in subset]),
                threshold,
            )
    return result


def _resolution_diagnostics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Class-conditional resolution results, including one-class buckets.

    Original resolution and effective pre-resize resolution are separate: web
    views can first downscale the image, after which the model resizes the short
    edge to 440. This exposes destructive downscale-then-upscale paths.
    """
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    for row in rows:
        try:
            width, height = int(row["width"]), int(row["height"])
            short, long = min(width, height), max(width, height)
        except (KeyError, TypeError, ValueError):
            missing += 1
            continue
        resolution_bucket = (
            "min_side_lt_128"
            if short < 128
            else "min_side_128_to_255"
            if short < 256
            else "min_side_256_to_511"
            if short < 512
            else "min_side_512_to_1023"
            if short < 1024
            else "min_side_ge_1024"
        )
        degradation_target = {"web": 768, "hard": 512}.get(str(row["view"]))
        effective_short = short
        if degradation_target and long > degradation_target:
            effective_short = max(1, round(short * degradation_target / long))
        model_resize_short_edge = int(row.get("model_resize_short_edge", 440))
        scale = model_resize_short_edge / effective_short
        scale_bucket = (
            "upsample_gt_2x"
            if scale > 2
            else "upsample_1_to_2x"
            if scale > 1
            else "downsample_1_to_2x"
            if scale >= 0.5
            else "downsample_gt_2x"
        )
        annotated = {
            **row,
            "original_min_side": short,
            "effective_min_side": effective_short,
            "model_resize_factor": scale,
            "model_resize_short_edge": model_resize_short_edge,
        }
        grouped[(str(row["view"]), resolution_bucket, scale_bucket, int(row["label"]))].append(annotated)
        row.update(
            original_min_side=short,
            effective_min_side=effective_short,
            model_resize_factor=scale,
            model_resize_bucket=scale_bucket,
            model_resize_short_edge=model_resize_short_edge,
        )
    buckets = []
    for (view, resolution_bucket, scale_bucket, label), subset in grouped.items():
        probabilities = np.array([row["probability"] for row in subset])
        correct = probabilities >= threshold if label else probabilities < threshold
        buckets.append(
            {
                "view": view,
                "original_resolution_bucket": resolution_bucket,
                "model_resize_bucket": scale_bucket,
                "label": label,
                "metric": "fake_recall" if label else "real_specificity",
                "n": len(subset),
                "unique_images": len({row["sample_id"] for row in subset}),
                "rate": float(np.mean(correct)),
                "mean_probability": float(np.mean(probabilities)),
                "mean_original_min_side": float(np.mean([row["original_min_side"] for row in subset])),
                "mean_effective_min_side": float(np.mean([row["effective_min_side"] for row in subset])),
                "model_resize_short_edge": int(subset[0]["model_resize_short_edge"]),
                "mean_model_resize_factor": float(np.mean([row["model_resize_factor"] for row in subset])),
                "groups": sorted({str(row["group_id"]) for row in subset}),
            }
        )
    return {
        "definition": "original short edge; effective short edge after deterministic view degradation; model resize target is taken from each prediction ledger (legacy default 440)",
        "missing_metadata_rows": missing,
        "class_conditional_buckets": sorted(
            buckets,
            key=lambda item: (
                item["view"], item["original_resolution_bucket"], item["model_resize_bucket"], item["label"]
            ),
        ),
    }


def _group_view_rates(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[bool]] = defaultdict(list)
    for row in rows:
        label = int(row["label"])
        correct = row["probability"] >= threshold if label else row["probability"] < threshold
        grouped[(str(row["view"]), str(row["group_id"]), label)].append(bool(correct))
    result = []
    for (view, group_id, label), correct in grouped.items():
        result.append(
            {
                "view": view,
                "group_id": group_id,
                "label": label,
                "metric": "fake_recall" if label else "real_specificity",
                "n": len(correct),
                "rate": float(np.mean(correct)),
            }
        )
    return sorted(result, key=lambda row: (row["view"], row["label"], row["rate"], row["group_id"]))


def _paired_stress(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_sample[str(row["sample_id"])][str(row["view"])] = row
    result = {}
    for stress in ("web", "hard"):
        pairs = [(views["clean"], views[stress]) for views in by_sample.values() if "clean" in views and stress in views]
        if not pairs:
            continue
        clean = np.array([pair[0]["probability"] for pair in pairs])
        changed = np.array([pair[1]["probability"] for pair in pairs])
        labels = np.array([pair[0]["label"] for pair in pairs])
        clean_metrics = binary_metrics(labels, clean, threshold)
        stress_metrics = binary_metrics(labels, changed, threshold)
        result[stress] = {
            "n_pairs": len(pairs),
            "mean_score_delta": float(np.mean(changed - clean)),
            "mean_absolute_score_delta": float(np.mean(np.abs(changed - clean))),
            "decision_flip_rate": float(np.mean((clean >= threshold) != (changed >= threshold))),
            "balanced_accuracy_delta": stress_metrics["balanced_accuracy"] - clean_metrics["balanced_accuracy"],
            "fake_recall_delta": stress_metrics["fake_recall"] - clean_metrics["fake_recall"],
            "real_specificity_delta": stress_metrics["real_specificity"] - clean_metrics["real_specificity"],
        }
    return result


def _markdown(report: dict[str, Any]) -> str:
    primary = report["evaluation"]["primary"]
    ci = report["evaluation"]["bootstrap_95"]["balanced_accuracy"]
    lines = [
        "# Validation report",
        "",
        f"**Status:** {'PASS' if report['valid'] else 'FAIL'}",
        "",
        f"Fixed-threshold balanced accuracy: **{primary['balanced_accuracy']:.4f}** "
        f"(95% clustered bootstrap CI {ci['low']:.4f}–{ci['high']:.4f}) at threshold "
        f"`{primary['threshold']:.2f}`.",
        "",
        f"AI recall: {primary['fake_recall']:.4f}; real specificity: {primary['real_specificity']:.4f}; "
        f"AUROC: {primary['roc_auc']:.4f}.",
        "",
        "## Views",
        "",
        "| View | N | Balanced accuracy | AI recall | Real specificity | AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for view, metrics in report["evaluation"]["slices"]["view"].items():
        lines.append(
            f"| {view} | {metrics['n']} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['fake_recall']:.4f} | {metrics['real_specificity']:.4f} | {metrics['roc_auc']:.4f} |"
        )
    lines += [
        "",
        "## Resolution diagnostics",
        "",
        "Rows with fewer than 20 unique images remain in JSON but are omitted here. Generator groups are shown to expose confounding.",
        "",
        "| View | Original short edge | Model resize | Metric | Images | Rate | Groups |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for bucket in report["evaluation"]["resolution_diagnostics"]["class_conditional_buckets"]:
        if bucket["unique_images"] < 20:
            continue
        lines.append(
            f"| {bucket['view']} | {bucket['original_resolution_bucket']} | {bucket['model_resize_bucket']} | "
            f"{bucket['metric']} | {bucket['unique_images']} | {bucket['rate']:.4f} | "
            f"{', '.join(bucket['groups'])} |"
        )
    lines += ["", "## Gates", ""]
    for gate in report["gates"]:
        lines.append(f"- {'PASS' if gate['passed'] else 'FAIL'} — {gate['name']}: {gate['detail']}")
    lines += ["", "## Calibration", ""]
    final = report["calibration"]["final"]
    lines.append(
        f"Selected `{final['method']}` with slope `{final['slope']:.8g}`, intercept "
        f"`{final['intercept']:.8g}`, raw decision boundary `{final['raw_boundary']:.8g}`."
    )
    return "\n".join(lines) + "\n"


def generate_report(
    manifest_path: str | Path, prediction_path: str | Path, config_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    from .io import read_manifest

    config = yaml.safe_load(Path(config_path).read_text())
    manifest = read_manifest(manifest_path)
    predictions = _load_predictions(prediction_path)
    threshold = float(config["threshold"])
    calibration_views = list(config["calibration"]["views"])
    evaluation_views = list(config["evaluation"]["views"])
    audit = audit_manifest(manifest, int(config["gates"]["phash_distance"]), check_files=True)
    calibration_rows, missing_calibration = _join(manifest, predictions, "calibration", calibration_views)
    test_rows, missing_test = _join(manifest, predictions, "test", evaluation_views)
    if missing_calibration or missing_test:
        missing = missing_calibration + missing_test
        raise ValueError(f"prediction coverage incomplete ({len(missing)} missing), examples: {missing[:5]}")
    if not calibration_rows or not test_rows:
        raise ValueError("both calibration and test predictions are required")

    calibrator, calibration_diagnostics = select_and_fit(
        calibration_rows, list(config["calibration"]["candidates"]), threshold
    )
    probabilities = calibrator.transform(np.array([row["raw_logit"] for row in test_rows]))
    for row, probability in zip(test_rows, probabilities, strict=True):
        row["probability"] = float(probability)
    y = np.array([row["label"] for row in test_rows])
    primary = binary_metrics(y, probabilities, threshold)
    primary["n_unique_images"] = len({row["sample_id"] for row in test_rows})
    uncalibrated = binary_metrics(
        y, sigmoid(np.array([row["raw_logit"] for row in test_rows])), threshold
    )
    bootstrap = clustered_bootstrap(
        test_rows,
        threshold,
        int(config["evaluation"]["bootstrap_replicates"]),
        int(config["evaluation"]["bootstrap_seed"]),
        float(config["evaluation"]["confidence"]),
    )
    slices = _slices(test_rows, threshold, int(config["evaluation"]["min_slice_size"]))
    resolution = _resolution_diagnostics(test_rows, threshold)
    tails = macro_group_tail(test_rows, threshold)
    stress = _paired_stress(test_rows, threshold)

    test_counts = Counter(int(row["label"]) for row in manifest if row["role"] == "test")
    minimum = int(config["gates"]["min_test_per_class"])
    ci_width = bootstrap["balanced_accuracy"]["high"] - bootstrap["balanced_accuracy"]["low"]
    gates = [
        {"name": "manifest_integrity", "passed": audit["valid"], "detail": f"{len(audit['issues'])} issue(s)"},
        {
            "name": "minimum_test_support",
            "passed": min(test_counts[0], test_counts[1]) >= minimum,
            "detail": f"real={test_counts[0]}, fake={test_counts[1]}, required_each={minimum}",
        },
        {
            "name": "balanced_accuracy",
            "passed": primary["balanced_accuracy"] >= float(config["gates"]["min_balanced_accuracy"]),
            "detail": f"{primary['balanced_accuracy']:.4f} >= {config['gates']['min_balanced_accuracy']}",
        },
        {
            "name": "uncertainty_width",
            "passed": ci_width <= float(config["gates"]["max_balanced_accuracy_ci_width"]),
            "detail": f"width={ci_width:.4f} <= {config['gates']['max_balanced_accuracy_ci_width']}",
        },
    ]
    report = _finite(
        {
            "valid": all(gate["passed"] for gate in gates),
            "protocol": "aiblink-validation/0.1.0",
            "inputs": {
                "manifest": str(Path(manifest_path).resolve()),
                "manifest_sha256": file_fingerprint(manifest_path),
                "predictions": str(Path(prediction_path).resolve()),
                "predictions_sha256": file_fingerprint(prediction_path),
                "config": str(Path(config_path).resolve()),
                "config_sha256": file_fingerprint(config_path),
            },
            "audit": audit,
            "calibration": calibration_diagnostics,
            "evaluation": {
                "primary_definition": "equal-weight mixture of declared deterministic views; bootstrap clusters by original image",
                "primary": primary,
                "uncalibrated_fixed_threshold": uncalibrated,
                "bootstrap_95": bootstrap,
                "group_tails": tails,
                "group_view_rates": _group_view_rates(test_rows, threshold),
                "resolution_diagnostics": resolution,
                "slices": slices,
                "paired_stress": stress,
            },
            "gates": gates,
        }
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "report.json", report)
    atomic_json(output_dir / "calibration.json", calibration_diagnostics)
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "sample_id", "label", "dataset", "source", "group_id", "view", "width", "height",
            "original_min_side", "effective_min_side", "model_resize_short_edge", "model_resize_factor", "model_resize_bucket",
            "raw_logit", "probability",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(test_rows)
    return report
