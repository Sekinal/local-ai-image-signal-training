from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .attacks import load_attack_profile
from .calibration import Calibrator
from .io import atomic_json, file_fingerprint, read_jsonl, read_manifest
from .metrics import binary_metrics


def _hierarchical_intervals(
    rows: list[dict[str, Any]], threshold: float, replicates: int, seed: int, confidence: float
) -> dict[str, dict[str, float]]:
    """Bootstrap provenance groups, then original images within each group.

    Class rates are macro-averaged over sampled provenance groups, so one large
    generator cannot dominate uncertainty about an unseen generator population.
    """
    groups: dict[int, dict[str, list[dict[str, Any]]]] = {0: defaultdict(list), 1: defaultdict(list)}
    for row in rows:
        groups[int(row["label"])][str(row["group_id"])].append(row)
    if not groups[0] or not groups[1]:
        raise ValueError("hierarchical bootstrap requires both classes")
    rng = np.random.default_rng(seed)
    values = {"balanced_accuracy": [], "fake_recall": [], "real_specificity": []}
    for _ in range(replicates):
        class_rates = {}
        for label in (0, 1):
            group_ids = list(groups[label])
            selected_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
            rates = []
            for group_id in selected_groups:
                population = groups[label][str(group_id)]
                indexes = rng.integers(0, len(population), size=len(population))
                probabilities = np.array([population[index]["probability"] for index in indexes])
                rates.append(float(np.mean(probabilities >= threshold if label else probabilities < threshold)))
            class_rates[label] = float(np.mean(rates))
        values["real_specificity"].append(class_rates[0])
        values["fake_recall"].append(class_rates[1])
        values["balanced_accuracy"].append((class_rates[0] + class_rates[1]) / 2)
    alpha = (1 - confidence) / 2
    return {
        name: {"low": float(np.quantile(series, alpha)), "high": float(np.quantile(series, 1 - alpha))}
        for name, series in values.items()
    }


def _markdown(report: dict[str, Any]) -> str:
    robust = report["worst_variant_per_image"]
    lines = [
        "# Red-team validation report",
        "",
        f"**Status:** {'PASS' if report['valid'] else 'INCOMPLETE/FAIL'}",
        "",
        f"Worst-variant-per-image balanced accuracy at 0.65: **{robust['metrics']['balanced_accuracy']:.4f}**.",
        f"Any-attack fake evasion rate among clean-correct fakes: **{report['attack_success']['fake_any_attack']:.4f}**.",
        "",
        "| Attack | Family | Severity | BA | AI recall | Real specificity | Fake ASR |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for case in sorted(report["cases"], key=lambda item: (item["metrics"]["balanced_accuracy"], item["attack_id"])):
        metrics = case["metrics"]
        lines.append(
            f"| {case['attack_id']} | {case['family']} | {case['severity']} | "
            f"{metrics['balanced_accuracy']:.4f} | {metrics['fake_recall']:.4f} | "
            f"{metrics['real_specificity']:.4f} | {case['fake_attack_success_rate']:.4f} |"
        )
    lines += ["", "## Gates", ""]
    for gate in report["gates"]:
        lines.append(f"- {'PASS' if gate['passed'] else 'FAIL'} — {gate['name']}: {gate['detail']}")
    return "\n".join(lines) + "\n"


def generate_redteam_report(
    manifest_path: str | Path,
    prediction_path: str | Path,
    calibration_path: str | Path,
    attack_config_path: str | Path,
    profile: str,
    output_dir: str | Path,
    threshold: float = 0.65,
    replicates: int = 2000,
    seed: int = 323,
) -> dict[str, Any]:
    manifest = {str(row["sample_id"]): row for row in read_manifest(manifest_path) if row["role"] == "test"}
    attacks, attack_config = load_attack_profile(attack_config_path, profile)
    attack_by_id = {attack.attack_id: attack for attack in attacks}
    prediction_rows = list(read_jsonl(prediction_path))
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    for prediction in prediction_rows:
        key = (str(prediction["sample_id"]), str(prediction["view"]))
        if key in predictions:
            raise ValueError(f"duplicate red-team prediction {key}")
        predictions[key] = prediction
    expected = {(sample_id, attack.attack_id) for sample_id in manifest for attack in attacks}
    missing = sorted(key for key in expected if key not in predictions or predictions[key].get("raw_logit") is None)
    if missing:
        raise ValueError(f"red-team prediction coverage incomplete ({len(missing)}), examples: {missing[:5]}")

    protocol_mismatches = []
    for sample_id in manifest:
        for attack in attacks:
            prediction = predictions[(sample_id, attack.attack_id)]
            expected_metadata = attack.metadata()
            for field in ("attack_id", "attack_family", "attack_severity", "attack_operations"):
                if prediction.get(field) != expected_metadata[field]:
                    protocol_mismatches.append(
                        {
                            "sample_id": sample_id,
                            "attack_id": attack.attack_id,
                            "field": field,
                            "expected": expected_metadata[field],
                            "observed": prediction.get(field),
                        }
                    )
                    break

    calibration_data = json.loads(Path(calibration_path).read_text())
    final = calibration_data.get("final", calibration_data)
    calibrator = Calibrator(**final)
    rows = []
    for sample_id, item in manifest.items():
        logits = np.array([float(predictions[(sample_id, attack.attack_id)]["raw_logit"]) for attack in attacks])
        probabilities = calibrator.transform(logits)
        for attack, logit, probability in zip(attacks, logits, probabilities, strict=True):
            rows.append(
                {
                    **item,
                    **attack.metadata(),
                    "raw_logit": float(logit),
                    "probability": float(probability),
                }
            )
    by_attack: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_attack[row["attack_id"]].append(row)
        by_sample[row["sample_id"]].append(row)
    clean_by_sample = {row["sample_id"]: row for row in by_attack["clean"]}
    cases = []
    family_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attack in attacks:
        subset = by_attack[attack.attack_id]
        labels = np.array([row["label"] for row in subset])
        probability = np.array([row["probability"] for row in subset])
        metrics = binary_metrics(labels, probability, threshold)
        attacked_fake = [
            row
            for row in subset
            if int(row["label"]) == 1 and clean_by_sample[row["sample_id"]]["probability"] >= threshold
        ]
        attack_success = (
            float(np.mean([row["probability"] < threshold for row in attacked_fake])) if attacked_fake else float("nan")
        )
        deltas = [row["probability"] - clean_by_sample[row["sample_id"]]["probability"] for row in subset]
        entry = {
            "attack_id": attack.attack_id,
            "family": attack.family,
            "severity": attack.severity,
            "operations": list(attack.operations),
            "metrics": metrics,
            "fake_attack_success_rate": attack_success,
            "mean_score_delta_from_clean": float(np.mean(deltas)),
            "mean_absolute_score_delta_from_clean": float(np.mean(np.abs(deltas))),
        }
        cases.append(entry)
        family_metrics[attack.family].append(entry)

    worst_rows = []
    fake_clean_correct = 0
    fake_any_evaded = 0
    real_clean_correct = 0
    real_any_false_positive = 0
    for sample_id, variants in by_sample.items():
        label = int(variants[0]["label"])
        clean = clean_by_sample[sample_id]
        if label:
            selected = min(variants, key=lambda row: row["probability"])
            if clean["probability"] >= threshold:
                fake_clean_correct += 1
                fake_any_evaded += int(selected["probability"] < threshold)
        else:
            selected = max(variants, key=lambda row: row["probability"])
            if clean["probability"] < threshold:
                real_clean_correct += 1
                real_any_false_positive += int(selected["probability"] >= threshold)
        worst_rows.append(dict(selected))
    worst_labels = np.array([row["label"] for row in worst_rows])
    worst_probability = np.array([row["probability"] for row in worst_rows])
    worst_metrics = binary_metrics(worst_labels, worst_probability, threshold)
    hierarchical_ci = _hierarchical_intervals(worst_rows, threshold, replicates, seed, 0.95)

    non_clean = [entry for entry in cases if entry["attack_id"] != "clean"]
    worst_case = min(non_clean, key=lambda entry: entry["metrics"]["balanced_accuracy"])
    critical = [entry for entry in non_clean if entry["severity"] == "critical"]
    observed_families = {attack.family for attack in attacks}
    required_external = set(attack_config.get("gates", {}).get("required_external_families", []))
    missing_external = sorted(required_external - observed_families)
    gates_config = attack_config.get("gates", {})
    gates = [
        {
            "name": "attack_protocol_match",
            "passed": not protocol_mismatches,
            "detail": "mismatches=" + str(len(protocol_mismatches)),
        },
        {
            "name": "worst_attack_balanced_accuracy",
            "passed": worst_case["metrics"]["balanced_accuracy"]
            >= float(gates_config.get("min_worst_attack_balanced_accuracy", 0.75)),
            "detail": f"{worst_case['attack_id']}={worst_case['metrics']['balanced_accuracy']:.4f}",
        },
        {
            "name": "critical_fake_recall",
            "passed": not critical
            or min(entry["metrics"]["fake_recall"] for entry in critical)
            >= float(gates_config.get("min_critical_fake_recall", 0.60)),
            "detail": f"minimum={min((entry['metrics']['fake_recall'] for entry in critical), default=float('nan')):.4f}",
        },
        {
            "name": "external_cohort_coverage",
            "passed": not missing_external,
            "detail": "missing=" + (",".join(missing_external) if missing_external else "none"),
        },
    ]
    report = {
        "valid": all(gate["passed"] for gate in gates),
        "protocol": "aiblink-redteam/0.1.0",
        "profile": profile,
        "threshold": threshold,
        "inputs": {
            "manifest_sha256": file_fingerprint(manifest_path),
            "predictions_sha256": file_fingerprint(prediction_path),
            "calibration_sha256": file_fingerprint(calibration_path),
            "attack_config_sha256": file_fingerprint(attack_config_path),
        },
        "coverage": {
            "unique_images": len(manifest),
            "attacks_per_image": len(attacks),
            "prediction_rows": len(rows),
            "observed_families": sorted(observed_families),
            "missing_external_families": missing_external,
            "protocol_mismatch_count": len(protocol_mismatches),
            "protocol_mismatch_examples": protocol_mismatches[:10],
        },
        "calibrator": final,
        "cases": cases,
        "families": {
            family: {
                "case_count": len(entries),
                "worst_balanced_accuracy": min(entry["metrics"]["balanced_accuracy"] for entry in entries),
                "worst_fake_recall": min(entry["metrics"]["fake_recall"] for entry in entries),
                "worst_real_specificity": min(entry["metrics"]["real_specificity"] for entry in entries),
            }
            for family, entries in family_metrics.items()
        },
        "worst_declared_attack": worst_case,
        "worst_variant_per_image": {"metrics": worst_metrics, "hierarchical_bootstrap_95": hierarchical_ci},
        "attack_success": {
            "fake_clean_correct": fake_clean_correct,
            "fake_any_attack_evaded": fake_any_evaded,
            "fake_any_attack": fake_any_evaded / fake_clean_correct if fake_clean_correct else None,
            "real_clean_correct": real_clean_correct,
            "real_any_attack_false_positive": real_any_false_positive,
            "real_any_attack": real_any_false_positive / real_clean_correct if real_clean_correct else None,
        },
        "gates": gates,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "redteam_report.json", report)
    (output_dir / "redteam_report.md").write_text(_markdown(report), encoding="utf-8")
    return report
