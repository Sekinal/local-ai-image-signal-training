import hashlib
import random

from PIL import Image

from aiblink_validation.io import atomic_jsonl, write_manifest
from aiblink_validation.architectures import (
    build_convnext_binary,
    build_convnext_large_binary,
    build_forensic_convnext_binary,
)
from aiblink_validation.training import (
    _BalancedSourceBatchSampler,
    _FullTrainDataset,
    _full_train_augment,
    _low_quality_train_augment,
    _resize_crop,
    _validation_subset,
    rank_calibration,
)


def _row(sample_id, label, group):
    return {
        "sample_id": sample_id,
        "path": "/unused.jpg",
        "label": label,
        "dataset": "toy",
        "source": group,
        "group_id": group,
        "role": "calibration",
        "content_sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
        "phash": hashlib.sha256((sample_id + "p").encode()).hexdigest()[:16],
    }


def test_tournament_resize_crop_has_fixed_size():
    image = Image.new("RGB", (91, 53), "navy")
    assert _resize_crop(image, 224, random_crop=False).size == (224, 224)


def test_scratch_convnext_architectures_are_binary_and_browser_scale():
    import torch

    cases = (
        (build_convnext_binary, 49_455_457),
        (build_forensic_convnext_binary, 26_900_321),
    )
    for builder, expected_parameters in cases:
        model = builder().eval()
        assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters
        with torch.inference_mode():
            output = model(torch.randn(1, 3, 64, 64))
        assert output.shape == (1, 1)
        assert torch.isfinite(output).all()


def test_forensic_convnext_high_pass_bank_is_fixed():
    model = build_forensic_convnext_binary()
    assert model.high_pass_weight.shape == (12, 1, 5, 5)
    assert not model.high_pass_weight.requires_grad


def test_scratch_large_teacher_is_materially_larger_than_student():
    import torch

    student_parameters = sum(parameter.numel() for parameter in build_convnext_binary().parameters())
    teacher = build_convnext_large_binary().eval()
    teacher_parameters = sum(parameter.numel() for parameter in teacher.parameters())
    assert teacher_parameters > 3 * student_parameters
    assert teacher(torch.randn(1, 3, 64, 64)).shape == (1, 1)


def test_full_augmentation_is_seeded_and_fixed_size():
    image = Image.new("RGB", (640, 431), "navy")
    first = _full_train_augment(image, 384, random.Random(323))
    second = _full_train_augment(image, 384, random.Random(323))
    assert first.size == (384, 384)
    assert first.tobytes() == second.tobytes()


def test_low_quality_augmentation_is_seeded_fixed_size_and_distinct():
    image = Image.effect_noise((640, 431), 80).convert("RGB")
    first = _low_quality_train_augment(image, 384, random.Random(323))
    second = _low_quality_train_augment(image, 384, random.Random(323))
    web = _full_train_augment(image, 384, random.Random(323))
    assert first.size == (384, 384)
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != web.tobytes()


def test_full_dataset_rejects_unknown_augmentation_profile():
    try:
        _FullTrainDataset([], 384, "unknown")
    except ValueError as error:
        assert "unknown augmentation profile" in str(error)
    else:
        raise AssertionError("invalid augmentation profile was accepted")


def test_full_sampler_exactly_balances_classes_and_is_reproducible():
    rows = [
        {"label": 0, "source": "real-a"},
        {"label": 0, "source": "real-b"},
        {"label": 1, "source": "fake-a"},
        {"label": 1, "source": "fake-b"},
        {"label": 1, "source": "fake-b"},
    ]
    first = list(_BalancedSourceBatchSampler(rows, 8, 3, 7, 323))
    second = list(_BalancedSourceBatchSampler(rows, 8, 3, 7, 323))
    assert first == second
    assert len(first) == 4
    for batch in first:
        labels = [rows[index]["label"] for index, _ in batch]
        assert labels.count(0) == labels.count(1) == 4


def test_validation_subset_balances_classes_and_round_robins_sources():
    rows = []
    for label in (0, 1):
        for source, count in ((f"{label}-large", 20), (f"{label}-small", 2)):
            rows.extend(
                {"sample_id": f"{source}-{index}", "label": label, "source": source}
                for index in range(count)
            )
    selected = _validation_subset(rows, 12, 323)
    assert sum(row["label"] == 0 for row in selected) == 6
    assert sum(row["label"] == 1 for row in selected) == 6
    assert {row["source"] for row in selected} == {"0-large", "0-small", "1-large", "1-small"}


def test_tournament_ranking_consumes_prediction_ledgers(tmp_path):
    rows = [_row(f"real-{i}", 0, "real") for i in range(30)]
    for group in range(3):
        rows.extend(_row(f"fake-{group}-{i}", 1, f"fake-{group}") for i in range(10))
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, rows)
    paths = []
    for candidate, direction in (("commfor_384", 1), ("convnext_nano", -1)):
        path = tmp_path / f"{candidate}.jsonl"
        atomic_jsonl(
            path,
            [
                {
                    "sample_id": row["sample_id"],
                    "view": "clean",
                    "raw_logit": direction * (3 if row["label"] else -3),
                    "error": None,
                    "model_id": candidate,
                }
                for row in rows
            ],
        )
        paths.append(path)
    result = rank_calibration(manifest, paths, tmp_path / "ranking.json")
    assert result["test_role_opened"] is False
    assert result["selected_candidate_id"] == "commfor_384"
    assert [row["candidate_id"] for row in result["ranking"]] == ["commfor_384"]
    assert [row["candidate_id"] for row in result["excluded"]] == ["convnext_nano"]
