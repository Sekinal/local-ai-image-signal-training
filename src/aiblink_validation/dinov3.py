from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .attacks import AttackCase, apply_attack
from .io import atomic_json, atomic_jsonl, sha256_file


DEFAULT_MODEL = "vit_small_patch16_dinov3.lvd1689m"
DEFAULT_REPOSITORY = "timm/vit_small_patch16_dinov3.lvd1689m"
DEFAULT_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class Probe:
    weight: np.ndarray
    bias: float

    def logits(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64) @ self.weight + self.bias


def preprocess_dinov3(image: Image.Image, input_size: int) -> np.ndarray:
    image = image.convert("RGB")
    width, height = image.size
    factor = input_size / min(width, height)
    resized = image.resize(
        (max(input_size, round(width * factor)), max(input_size, round(height * factor))),
        Image.Resampling.BICUBIC,
    )
    left = round((resized.width - input_size) / 2)
    top = round((resized.height - input_size) / 2)
    array = np.asarray(resized.crop((left, top, left + input_size, top + input_size)), dtype=np.float32) / 255
    return ((array - MEAN) / STD).transpose(2, 0, 1).copy()


def pool_tokens(tokens, prefix_tokens: int):
    """Official-style linear-probe representation: CLS plus mean patch token."""
    import torch

    if tokens.ndim != 3 or tokens.shape[1] <= prefix_tokens:
        raise ValueError(f"unexpected DINOv3 token shape {tuple(tokens.shape)}")
    representation = torch.cat((tokens[:, 0], tokens[:, prefix_tokens:].mean(dim=1)), dim=1)
    return torch.nn.functional.normalize(representation, dim=1)


def pool_backbone_output(model, output, pooling: str):
    import torch

    if pooling == "cls_mean_patch":
        return pool_tokens(output, int(model.num_prefix_tokens))
    if pooling == "cls":
        if output.ndim != 3:
            raise ValueError(f"CLS pooling requires token output, got {tuple(output.shape)}")
        representation = output[:, 0]
    elif pooling == "model_head":
        representation = model.forward_head(output, pre_logits=True)
    elif pooling == "spatial_mean":
        if output.ndim != 4:
            raise ValueError(f"spatial mean pooling requires BCHW output, got {tuple(output.shape)}")
        representation = output.mean(dim=(-2, -1))
    else:
        raise ValueError(f"unknown backbone pooling mode {pooling!r}")
    if representation.ndim != 2:
        raise ValueError(f"pooling produced unexpected shape {tuple(representation.shape)}")
    return torch.nn.functional.normalize(representation, dim=1)


def calibration_fold(row: dict[str, Any], fake_groups: list[str]) -> int:
    """Hold out a fake generator or a deterministic real-image partition."""
    if int(row["label"]) == 1:
        return fake_groups.index(str(row["group_id"]))
    digest = hashlib.sha256(str(row["sample_id"]).encode()).digest()
    return int.from_bytes(digest[:8], "big") % len(fake_groups)


def fit_probe(features: np.ndarray, labels: np.ndarray, ridge: float = 1e-3) -> tuple[Probe, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    features_tensor = torch.as_tensor(features, dtype=torch.float64)
    labels_tensor = torch.as_tensor(labels, dtype=torch.float64)
    if features_tensor.ndim != 2 or len(features_tensor) != len(labels_tensor):
        raise ValueError("probe features and labels have incompatible shapes")
    if len(torch.unique(labels_tensor)) != 2:
        raise ValueError("probe fitting requires both classes")
    positive = max(1, int(labels_tensor.sum().item()))
    negative = max(1, len(labels_tensor) - positive)
    sample_weight = torch.where(
        labels_tensor == 1,
        torch.tensor(len(labels_tensor) / (2 * positive), dtype=torch.float64),
        torch.tensor(len(labels_tensor) / (2 * negative), dtype=torch.float64),
    )
    weight = torch.zeros(features_tensor.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=200, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        logits = features_tensor @ weight + bias
        losses = functional.binary_cross_entropy_with_logits(logits, labels_tensor, reduction="none")
        loss = torch.mean(sample_weight * losses) + 0.5 * ridge * torch.sum(weight.square())
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final_logits = features_tensor @ weight + bias
        final_losses = functional.binary_cross_entropy_with_logits(final_logits, labels_tensor, reduction="none")
        final_loss = float((torch.mean(sample_weight * final_losses) + 0.5 * ridge * torch.sum(weight.square())).item())
    probe = Probe(weight.detach().numpy().copy(), float(bias.detach().item()))
    return probe, {"ridge": ridge, "optimizer": "LBFGS", "max_iter": 200, "final_loss": final_loss}


class _FeatureDataset:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        attack: AttackCase,
        input_size: int,
        background_rows: list[dict[str, Any]],
    ):
        self.rows = rows
        self.attack = attack
        self.input_size = input_size
        self.background_rows = sorted(
            [row for row in background_rows if int(row["label"]) == 0], key=lambda row: str(row["sample_id"])
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        try:
            with Image.open(row["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                background = None
                if any(operation["op"] == "composite" for operation in self.attack.operations):
                    if not self.background_rows:
                        raise ValueError("no real backgrounds available for composite attack")
                    digest = hashlib.sha256(
                        f"{row['sample_id']}\0{self.attack.attack_id}\0background".encode()
                    ).digest()
                    background_index = int.from_bytes(digest[:8], "big") % len(self.background_rows)
                    background_row = self.background_rows[background_index]
                    if background_row["sample_id"] == row["sample_id"] and len(self.background_rows) > 1:
                        background_row = self.background_rows[(background_index + 1) % len(self.background_rows)]
                    with Image.open(background_row["path"]) as background_source:
                        background = ImageOps.exif_transpose(background_source).convert("RGB").copy()
                attacked = apply_attack(image, self.attack, str(row["sample_id"]), background=background)
                tensor = torch.from_numpy(preprocess_dinov3(attacked, self.input_size))
            return tensor, index, ""
        except Exception as exc:
            return torch.zeros(3, self.input_size, self.input_size), index, f"{type(exc).__name__}: {exc}"


def load_backbone(model_name: str, repository: str, revision: str, device: str):
    import timm
    from huggingface_hub import hf_hub_download

    checkpoint = hf_hub_download(repository, "model.safetensors", revision=revision)
    # Load the checkpoint with its native classifier shape first.  Some tagged
    # timm checkpoints (for example ConvNeXt D1H) contain a supervised head,
    # while self-supervised ViT checkpoints do not.  Resetting after the load
    # supports both without weakening checkpoint-key validation.
    model_options = {"pretrained": False, "checkpoint_path": checkpoint}
    if model_name.startswith(("vit_", "eva")):
        model_options["dynamic_img_size"] = True
    model = timm.create_model(model_name, **model_options)
    model.reset_classifier(0)
    model.eval().to(device)
    return model, checkpoint


def extract_features(
    rows: list[dict[str, Any]],
    attacks: list[AttackCase],
    model,
    input_size: int,
    batch_size: int,
    workers: int,
    device: str,
    pooling: str = "cls_mean_patch",
) -> tuple[dict[tuple[str, str], np.ndarray], int]:
    import torch
    from torch.utils.data import DataLoader

    features: dict[tuple[str, str], np.ndarray] = {}
    failures = 0
    for attack in attacks:
        dataset = _FeatureDataset(rows, attack, input_size, rows)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(max(0, workers), os.cpu_count() or 1),
            pin_memory=device == "cuda",
            persistent_workers=workers > 0 and bool(rows),
        )
        with torch.inference_mode():
            for images, indexes, errors in loader:
                tokens = model.forward_features(images.to(device, non_blocking=True))
                pooled = pool_backbone_output(model, tokens, pooling).float().cpu().numpy()
                for feature, index, error in zip(pooled, indexes.tolist(), errors, strict=True):
                    failures += int(bool(error))
                    if not error:
                        features[(str(rows[index]["sample_id"]), attack.attack_id)] = feature
    return features, failures


def _flatten(
    rows: list[dict[str, Any]], attacks: list[AttackCase], features: dict[tuple[str, str], np.ndarray]
) -> tuple[np.ndarray, np.ndarray, list[tuple[dict[str, Any], AttackCase]]]:
    metadata = [(row, attack) for row in rows for attack in attacks]
    missing = [
        (str(row["sample_id"]), attack.attack_id)
        for row, attack in metadata
        if (str(row["sample_id"]), attack.attack_id) not in features
    ]
    if missing:
        raise ValueError(f"DINOv3 feature coverage incomplete ({len(missing)}), examples: {missing[:5]}")
    matrix = np.stack([features[(str(row["sample_id"]), attack.attack_id)] for row, attack in metadata])
    labels = np.array([int(row["label"]) for row, _ in metadata], dtype=np.int8)
    return matrix, labels, metadata


def _write_head(path: Path, probe: Probe) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, weight=probe.weight, bias=np.array(probe.bias))
    os.replace(temporary, path)


def load_head(path: str | Path) -> tuple[Probe, dict[str, Any]]:
    path = Path(path)
    with np.load(path) as archive:
        probe = Probe(np.asarray(archive["weight"], dtype=np.float64), float(archive["bias"]))
    metadata = json.loads(path.with_name("metadata.json").read_text())
    if sha256_file(path) != metadata["head_sha256"]:
        raise ValueError("DINOv3 head fingerprint does not match metadata")
    return probe, metadata


def run_probe(
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    attacks: list[AttackCase],
    output_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    repository: str = DEFAULT_REPOSITORY,
    revision: str = DEFAULT_REVISION,
    input_size: int = 384,
    ridge: float = 1e-3,
    batch_size: int = 64,
    workers: int = 4,
    device: str = "cuda",
    pooling: str = "cls_mean_patch",
    license_label: str = "dinov3-license (evaluation only; not MIT-compatible for bounty redistribution)",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model, checkpoint = load_backbone(model_name, repository, revision, device)
    patch_size = getattr(getattr(model, "patch_embed", None), "patch_size", None)
    if patch_size:
        patch_edge = int(patch_size[0] if isinstance(patch_size, tuple) else patch_size)
        if input_size % patch_edge:
            raise ValueError(f"input size {input_size} must be a multiple of patch size {patch_edge}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    calibration_features, calibration_failures = extract_features(
        calibration_rows, attacks, model, input_size, batch_size, workers, device, pooling
    )
    test_features, test_failures = extract_features(
        test_rows, attacks, model, input_size, batch_size, workers, device, pooling
    )
    calibration_matrix, calibration_labels, calibration_metadata = _flatten(
        calibration_rows, attacks, calibration_features
    )
    test_matrix, _, test_metadata = _flatten(test_rows, attacks, test_features)
    fake_groups = sorted({str(row["group_id"]) for row in calibration_rows if int(row["label"]) == 1})
    if len(fake_groups) < 3:
        raise ValueError("DINOv3 probe requires at least three fake calibration groups")
    folds = np.array([calibration_fold(row, fake_groups) for row, _ in calibration_metadata])
    oof_logits = np.full(len(calibration_matrix), np.nan)
    fold_diagnostics = []
    for fold, fake_group in enumerate(fake_groups):
        train, held = folds != fold, folds == fold
        probe, diagnostics = fit_probe(calibration_matrix[train], calibration_labels[train], ridge)
        oof_logits[held] = probe.logits(calibration_matrix[held])
        fold_diagnostics.append(
            {"fold": fold, "fake_group": fake_group, "train_rows": int(train.sum()), "heldout_rows": int(held.sum()), **diagnostics}
        )
    if not np.all(np.isfinite(oof_logits)):
        raise RuntimeError("cross-fitted DINOv3 probe did not score every calibration row")
    final_probe, final_diagnostics = fit_probe(calibration_matrix, calibration_labels, ridge)
    test_logits = final_probe.logits(test_matrix)
    records = []
    for (row, attack), logit in zip(calibration_metadata, oof_logits, strict=True):
        records.append(
            {
                "sample_id": row["sample_id"], "view": attack.attack_id, "raw_logit": float(logit),
                "model_id": repository, "model_revision": revision, "model_architecture": model_name,
                "model_input_size": input_size, "model_resize_short_edge": input_size,
                "probe_fit": "out_of_fold", "error": None, **attack.metadata(),
            }
        )
    for (row, attack), logit in zip(test_metadata, test_logits, strict=True):
        records.append(
            {
                "sample_id": row["sample_id"], "view": attack.attack_id, "raw_logit": float(logit),
                "model_id": repository, "model_revision": revision, "model_architecture": model_name,
                "model_input_size": input_size, "model_resize_short_edge": input_size,
                "probe_fit": "all_calibration", "error": None, **attack.metadata(),
            }
        )
    head_path = output_dir / "head.npz"
    _write_head(head_path, final_probe)
    metadata = {
        "protocol": "aiblink-frozen-backbone-linear-probe/0.2.0",
        "model": {"architecture": model_name, "repository": repository, "revision": revision},
        "model_checkpoint_sha256": sha256_file(checkpoint),
        "license": license_label,
        "pooling": pooling,
        "representation": f"L2-normalized {pooling}",
        "backbone_parameter_count": parameter_count,
        "estimated_fp16_backbone_bytes": parameter_count * 2,
        "input_size": input_size,
        "training_views": [attack.attack_id for attack in attacks],
        "calibration_images": len(calibration_rows),
        "test_images": len(test_rows),
        "fake_calibration_groups": fake_groups,
        "folds": fold_diagnostics,
        "final_fit": final_diagnostics,
        "feature_dimension": int(final_probe.weight.shape[0]),
        "head_parameter_count": int(final_probe.weight.size + 1),
        "head_sha256": sha256_file(head_path),
        "failures": calibration_failures + test_failures,
        "seconds": time.monotonic() - started,
    }
    atomic_jsonl(output_dir / "predictions.jsonl", records)
    atomic_json(output_dir / "metadata.json", metadata)
    return metadata


def run_inference(
    rows: list[dict[str, Any]],
    attacks: list[AttackCase],
    head_path: str | Path,
    output: str | Path,
    batch_size: int = 64,
    workers: int = 4,
    device: str = "cuda",
) -> dict[str, Any]:
    started = time.monotonic()
    probe, metadata = load_head(head_path)
    model_settings = metadata["model"]
    model, checkpoint = load_backbone(
        model_settings["architecture"], model_settings["repository"], model_settings["revision"], device
    )
    if sha256_file(checkpoint) != metadata["model_checkpoint_sha256"]:
        raise ValueError("DINOv3 backbone fingerprint does not match trained-head metadata")
    input_size = int(metadata["input_size"])
    pooling = str(metadata.get("pooling", "cls_mean_patch"))
    features, failures = extract_features(
        rows, attacks, model, input_size, batch_size, workers, device, pooling
    )
    matrix, _, flattened = _flatten(rows, attacks, features)
    logits = probe.logits(matrix)
    records = [
        {
            "sample_id": row["sample_id"], "view": attack.attack_id, "raw_logit": float(logit),
            "model_id": model_settings["repository"], "model_revision": model_settings["revision"],
            "model_architecture": model_settings["architecture"], "model_input_size": input_size,
            "model_resize_short_edge": input_size, "probe_fit": "all_calibration", "error": None,
            **attack.metadata(),
        }
        for (row, attack), logit in zip(flattened, logits, strict=True)
    ]
    atomic_jsonl(output, records)
    return {
        "rows": len(rows), "predictions": len(records), "failures": failures,
        "views": [attack.attack_id for attack in attacks], "seconds": time.monotonic() - started,
    }
