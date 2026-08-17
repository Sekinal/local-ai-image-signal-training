import json

from aiblink_validation.io import atomic_jsonl, write_manifest
from aiblink_validation.report import generate_report


def test_end_to_end_report(tmp_path):
    rows = []
    predictions = []
    index = 0
    specs = [
        ("calibration", 0, "real-cal", -3.0),
        ("calibration", 1, "fake-a", 2.0),
        ("calibration", 1, "fake-b", 3.0),
        ("calibration", 1, "fake-c", 4.0),
        ("test", 0, "real-test", -3.0),
        ("test", 1, "fake-test", 3.0),
    ]
    for role, label, group, center in specs:
        for sample in range(3):
            sample_id = f"s-{index}"
            index += 1
            rows.append(
                {
                    "sample_id": sample_id,
                    "path": __file__,
                    "label": label,
                    "dataset": "fixture",
                    "source": group,
                    "group_id": group,
                    "role": role,
                    "content_sha256": f"{index:064x}",
                    "phash": f"{(index << (32 if role == 'test' else 0)):016x}",
                    "width": 512,
                    "height": 512,
                }
            )
            for view in ("clean", "web", "hard"):
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "view": view,
                        "raw_logit": center,
                        "model_resize_short_edge": 256,
                    }
                )
    manifest = tmp_path / "manifest.csv"
    ledger = tmp_path / "predictions.jsonl"
    config = tmp_path / "config.yaml"
    write_manifest(manifest, rows)
    atomic_jsonl(ledger, predictions)
    config.write_text(
        """threshold: 0.65
calibration: {candidates: [identity, bias, platt], views: [clean, web, hard]}
evaluation: {views: [clean, web, hard], bootstrap_replicates: 20, bootstrap_seed: 1, confidence: 0.95, min_slice_size: 1}
gates: {phash_distance: 0, min_test_per_class: 1, min_balanced_accuracy: 0.75, max_balanced_accuracy_ci_width: 0.5}
"""
    )
    report = generate_report(manifest, ledger, config, tmp_path / "report")
    assert report["valid"]
    assert report["evaluation"]["primary"]["balanced_accuracy"] == 1.0
    assert report["evaluation"]["primary"]["n_unique_images"] == 6
    assert "min_side_512_to_1023" in report["evaluation"]["slices"]["resolution_bucket"]
    resolution = report["evaluation"]["resolution_diagnostics"]
    assert resolution["missing_metadata_rows"] == 0
    assert {bucket["label"] for bucket in resolution["class_conditional_buckets"]} == {0, 1}
    assert {bucket["model_resize_short_edge"] for bucket in resolution["class_conditional_buckets"]} == {256}
    assert json.loads((tmp_path / "report" / "report.json").read_text())["valid"]
