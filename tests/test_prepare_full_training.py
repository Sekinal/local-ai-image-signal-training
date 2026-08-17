import io
import importlib.util
from pathlib import Path

from PIL import Image


_SPEC = importlib.util.spec_from_file_location(
    "prepare_full_training", Path(__file__).parents[1] / "scripts" / "prepare_full_training.py"
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_select = _MODULE._select
_web_tier = _MODULE._web_tier
_drop_cross_role_phash = _MODULE._drop_cross_role_phash


def test_source_caps_are_deterministic():
    rows = [
        {"source": source, "locator": f"{source}-{index}"}
        for source in ("a", "b")
        for index in range(20)
    ]
    first = _select(rows, {"a": 3, "b": 5}, 323)
    second = _select(reversed(rows), {"a": 3, "b": 5}, 323)
    assert [(row["source"], row["locator"]) for row in first] == [
        (row["source"], row["locator"]) for row in second
    ]
    assert sum(row["source"] == "a" for row in first) == 3
    assert sum(row["source"] == "b" for row in first) == 5


def test_openfake_web_tier_bounds_short_edge_and_is_jpeg():
    source = io.BytesIO()
    Image.new("RGB", (1800, 1200), "orange").save(source, "PNG")
    encoded = _web_tier(source.getvalue(), max_short_edge=512)
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.format == "JPEG"
        assert min(image.size) == 512


def test_near_duplicate_cleanup_drops_only_calibration_rows():
    rows = [
        {
            "sample_id": "train",
            "role": "train",
            "phash": "0000000000000000",
            "content_sha256": "a" * 64,
        },
        {
            "sample_id": "calibration-near",
            "role": "calibration",
            "phash": "0000000000000001",
            "content_sha256": "b" * 64,
        },
        {
            "sample_id": "calibration-far",
            "role": "calibration",
            "phash": "ffffffffffffffff",
            "content_sha256": "c" * 64,
        },
    ]
    cleaned, dropped = _drop_cross_role_phash(rows, distance=1)
    assert dropped == 1
    assert {row["sample_id"] for row in cleaned} == {"train", "calibration-far"}
