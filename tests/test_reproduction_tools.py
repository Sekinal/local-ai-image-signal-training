import csv
import gzip
import hashlib
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "rehydrate_manifest.py"
SPEC = importlib.util.spec_from_file_location("rehydrate_manifest", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rehydrate_manifest_matches_exact_bytes(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    first = images / "a.jpg"
    second = images / "b.png"
    first.write_bytes(b"first encoded image")
    second.write_bytes(b"second encoded image")
    inventory = tmp_path / "inventory.csv.gz"
    fields = ["sample_id", "label", "content_sha256", "role"]
    with gzip.open(inventory, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "second",
                "label": 1,
                "content_sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
                "role": "train",
            }
        )
        writer.writerow(
            {
                "sample_id": "first",
                "label": 0,
                "content_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
                "role": "calibration",
            }
        )
    output = tmp_path / "manifest.csv"
    result = MODULE.rehydrate(inventory, images, output)
    assert result == {"inventory_rows": 2, "written_rows": 2, "missing": 0, "duplicate_bytes": 0}
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in rows] == ["second", "first"]
    assert Path(rows[0]["path"]) == second.resolve()


def test_rehydrate_manifest_fails_closed_on_missing_bytes(tmp_path: Path):
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("sample_id,label,content_sha256,role\nmissing,1," + "0" * 64 + ",train\n")
    try:
        MODULE.rehydrate(inventory, tmp_path, tmp_path / "out.csv")
    except RuntimeError as error:
        assert "missing 1 of 1" in str(error)
    else:
        raise AssertionError("missing bytes must fail closed")
