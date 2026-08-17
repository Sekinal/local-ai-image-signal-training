import csv

from aiblink_validation.comparison import compare


def test_paired_comparison(tmp_path):
    columns = ["sample_id", "view", "label", "group_id", "probability"]
    left, right = tmp_path / "left.csv", tmp_path / "right.csv"
    rows_left, rows_right = [], []
    for label in (0, 1):
        for index in range(10):
            for view in ("clean", "web"):
                base = {"sample_id": f"{label}-{index}", "view": view, "label": label, "group_id": f"g-{label}"}
                rows_left.append({**base, "probability": 0.7})
                rows_right.append({**base, "probability": 0.1 if label == 0 else 0.9})
    for path, rows in ((left, rows_left), (right, rows_right)):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    result = compare(left, right, tmp_path / "comparison.json", replicates=50)
    assert result["delta"]["balanced_accuracy"] == 0.5
    assert result["promotion_signal"]

