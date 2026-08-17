from pathlib import Path

from aiblink_validation.io import write_manifest
from aiblink_validation.leakage import metadata_leakage_audit


def test_metadata_probe_finds_resolution_leakage(tmp_path):
    rows = []
    for label in (0, 1):
        for index in range(30):
            rows.append(
                {
                    "sample_id": f"{label}-{index}",
                    "path": __file__,
                    "label": label,
                    "dataset": "fixture",
                    "source": f"source-{label}",
                    "group_id": f"group-{label}",
                    "role": "test",
                    "content_sha256": f"{label * 100 + index + 1:064x}",
                    "phash": f"{label * 100 + index + 1:016x}",
                    "width": 128 if label == 0 else 1024,
                    "height": 128 if label == 0 else 1024,
                    "mime": "image/jpeg" if label == 0 else "image/png",
                }
            )
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    result = metadata_leakage_audit(manifest, tmp_path / "leakage.json", {"test"}, folds=3)
    assert result["metrics"]["roc_auc"] > 0.95
    assert not result["valid"]
    assert not result["provenance_group_cv_available"]


def test_metadata_probe_holds_out_whole_provenance_groups(tmp_path):
    rows = []
    for label in (0, 1):
        for group in range(3):
            for index in range(10):
                rows.append(
                    {
                        "sample_id": f"{label}-{group}-{index}",
                        "path": __file__,
                        "label": label,
                        "dataset": "fixture",
                        "source": f"source-{label}-{group}",
                        "group_id": f"group-{label}-{group}",
                        "role": "test",
                        "content_sha256": f"{label * 1000 + group * 100 + index + 1:064x}",
                        "phash": f"{label * 1000 + group * 100 + index + 1:016x}",
                        "width": 256 if label == 0 else 1024,
                        "height": 256 if label == 0 else 1024,
                        "mime": "image/jpeg" if label == 0 else "image/png",
                    }
                )
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    result = metadata_leakage_audit(manifest, tmp_path / "leakage.json", {"test"}, folds=5)
    assert result["provenance_group_cv_available"]
    assert result["effective_folds"] == 3
    assert result["cross_validation"] == "class-stratified provenance-group folds"
    assert result["metrics"]["roc_auc"] > 0.95
