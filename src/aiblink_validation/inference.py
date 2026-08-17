from __future__ import annotations

import io
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .attacks import AttackCase, apply_attack, legacy_case


def _resize_shorter_edge(image: Image.Image, target: int) -> Image.Image:
    width, height = image.size
    if width <= height:
        output = (target, max(target, round(height * target / width)))
    else:
        output = (max(target, round(width * target / height)), target)
    return image.resize(output, Image.Resampling.BICUBIC)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = round((width - size) / 2)
    top = round((height - size) / 2)
    return image.crop((left, top, left + size, top + size))


def degrade(image: Image.Image, view: str) -> Image.Image:
    if view == "clean":
        return image
    settings = {"web": (768, 60), "hard": (512, 40)}
    if view not in settings:
        raise ValueError(f"unknown view {view!r}")
    target, quality = settings[view]
    width, height = image.size
    longest = max(width, height)
    if longest > target:
        scale = target / longest
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR
        )
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        return encoded.convert("RGB").copy()


def preprocess(image: Image.Image, input_size: int = 384) -> np.ndarray:
    resize_size = 440 if input_size == 384 else 256
    image = _center_crop(_resize_shorter_edge(image, resize_size), input_size)
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return ((array - mean) / std).transpose(2, 0, 1).copy()


class _ImageDataset:
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
                tensor = torch.from_numpy(
                    preprocess(
                        apply_attack(image, self.attack, str(row["sample_id"]), background=background),
                        self.input_size,
                    )
                )
            return tensor, index, ""
        except Exception as exc:  # retained in the ledger; coverage gate will fail
            return torch.zeros(3, self.input_size, self.input_size), index, f"{type(exc).__name__}: {exc}"


def _load_model(model_id: str, revision: str | None, device: str):
    import timm
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    snapshot = snapshot_download(
        repo_id=model_id,
        revision=revision,
        allow_patterns=["config.json", "model.safetensors"],
    )
    config = json.loads(Path(snapshot, "config.json").read_text())
    input_size = int(config.get("input_size", 384))
    model_size = config.get("model_size", "small")
    patch_size = int(config.get("patch_size", 16))
    name = f"vit_{model_size}_patch{patch_size}_{input_size}.augreg_in21k_ft_in1k"
    model = timm.create_model(name, pretrained=False)
    model.head = torch.nn.Linear(model.head.in_features, 1)
    state = load_file(str(Path(snapshot, "model.safetensors")), device="cpu")
    # PyTorchModelHubMixin saves the wrapper's `vit.` prefix.
    if any(key.startswith("vit.") for key in state):
        state = {key.removeprefix("vit."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval().to(device)
    return model, input_size, snapshot


def run_inference(
    rows: list[dict[str, Any]],
    output: str | Path,
    model_id: str,
    revision: str | None,
    views: list[str],
    batch_size: int,
    workers: int,
    device: str,
    attacks: list[AttackCase] | None = None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, input_size, snapshot = _load_model(model_id, revision, device)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if output.exists():
        from .io import read_jsonl

        existing = {(row["sample_id"], row["view"]): row for row in read_jsonl(output)}
    temporary = output.with_suffix(output.suffix + ".tmp")
    started = time.monotonic()
    total_new = 0
    failures = 0
    attack_cases = attacks or [legacy_case(view) for view in views]
    with temporary.open("w", encoding="utf-8") as handle:
        for row in existing.values():
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        for attack in attack_cases:
            pending = [row for row in rows if (row["sample_id"], attack.attack_id) not in existing]
            dataset = _ImageDataset(pending, attack, input_size, rows)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(max(0, workers), os.cpu_count() or 1),
                pin_memory=device == "cuda",
                persistent_workers=workers > 0 and bool(pending),
            )
            with torch.inference_mode():
                for images, indexes, errors in loader:
                    logits = model(images.to(device, non_blocking=True)).squeeze(-1).float().cpu().numpy()
                    for logit, index, error in zip(logits, indexes.tolist(), errors, strict=True):
                        source = pending[index]
                        failed = bool(error)
                        failures += int(failed)
                        record = {
                            "sample_id": source["sample_id"],
                            "view": attack.attack_id,
                            "raw_logit": None if failed else float(logit),
                            "error": error or None,
                            "model_id": model_id,
                            "model_revision": revision or Path(snapshot).name,
                            **attack.metadata(),
                        }
                        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                        total_new += 1
                    handle.flush()
    os.replace(temporary, output)
    elapsed = time.monotonic() - started
    return {
        "rows": len(rows),
        "views": [attack.attack_id for attack in attack_cases],
        "predictions_written": total_new,
        "resumed_predictions": len(existing),
        "failures": failures,
        "seconds": elapsed,
        "images_per_second": total_new / elapsed if elapsed else None,
        "model_snapshot": snapshot,
    }
