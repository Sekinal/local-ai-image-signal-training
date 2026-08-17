from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .attacks import AttackCase, apply_attack, legacy_case
from .calibration import select_and_fit
from .io import atomic_json, atomic_jsonl, read_manifest, sha256_file
from .metrics import binary_metrics, sigmoid


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    kind: str
    model_name: str
    repository: str
    revision: str
    input_size: int
    license: str
    competition_status: str
    onnx_path: str | None = None


CANDIDATES = {
    spec.candidate_id: spec
    for spec in (
        CandidateSpec("commfor_384", "community_forensics", "vit_small_patch16_384", "OwensLab/commfor-model-384", "6076002bf0d9dd37537f965ee2f06f826c333b61", 384, "MIT", "eligible"),
        CandidateSpec("sieve_ft1", "sieve_onnx", "vit_small_patch16_384", "Phineas1500/sieve-ai-image-detector", "v0.1.0", 384, "MIT", "disqualified-participant-model", "/data/ai_blink/models/sieve/ft1_best_fp16.onnx"),
        CandidateSpec("commfor_224", "community_forensics", "vit_small_patch16_224", "OwensLab/commfor-model-224", "26afc31e6b40c312c3fd42c05a758be62446215b", 224, "MIT", "eligible"),
        CandidateSpec("dinov2_small", "timm", "vit_small_patch14_dinov2.lvd142m", "timm/vit_small_patch14_dinov2.lvd142m", "4610ca143709d58a633b6397a74412c2c3842454", 252, "Apache-2.0", "needs-organizer-approval"),
        CandidateSpec("eva02_small", "timm", "eva02_small_patch14_224.mim_in22k", "timm/eva02_small_patch14_224.mim_in22k", "79c7d4274f6dbf202549d8f976ae24eeaf97e5ad", 224, "MIT", "eligible"),
        CandidateSpec("convnext_nano", "timm", "convnext_nano.d1h_in1k", "timm/convnext_nano.d1h_in1k", "fe9cba6df2f17352639e23ca89e1aa6b37f51462", 288, "Apache-2.0", "needs-organizer-approval"),
        CandidateSpec("convnext_small_pretrained", "timm", "convnext_small.fb_in22k_ft_in1k_384", "timm/convnext_small.fb_in22k_ft_in1k_384", "14a837353b6ffeced1c635dc5c8c799356e316f4", 384, "Apache-2.0", "needs-organizer-approval"),
        CandidateSpec("convnext_small_scratch", "convnext_scratch", "convnext_small_384", "local/convnext-small-scratch", "architecture-v1", 384, "MIT", "eligible"),
        CandidateSpec("convnext_large_teacher_scratch", "convnext_large_scratch", "convnext_large_384", "local/convnext-large-teacher-scratch", "architecture-v1", 384, "MIT", "eligible-teacher-only"),
        CandidateSpec("forensic_convnext_scratch", "forensic_convnext_scratch", "forensic_convnext_384", "local/forensic-convnext-scratch", "architecture-v1", 384, "MIT", "eligible"),
        CandidateSpec("dinov3_small", "timm", "vit_small_patch16_dinov3.lvd1689m", "timm/vit_small_patch16_dinov3.lvd1689m", "3bf4720a82ec2066db88137180ff1f83a675cef0", 256, "DINOv3 License", "diagnostic-only"),
    )
}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sieve_from_onnx(path: str):
    import onnx
    import timm
    import torch
    from onnx import numpy_helper

    model = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
    model.reset_classifier(1)
    graph = onnx.load(path, load_external_data=True).graph
    initializers = list(graph.initializer)
    state: dict[str, Any] = {}
    unnamed = []
    for initializer in initializers:
        value = torch.from_numpy(np.array(numpy_helper.to_array(initializer), copy=True)).float()
        if initializer.name.startswith("m.vit."):
            state[initializer.name.removeprefix("m.vit.")] = value
        elif initializer.name.startswith("onnx::MatMul_"):
            unnamed.append(value)
        else:
            raise RuntimeError(f"unrecognized Sieve ONNX initializer {initializer.name!r}")
    expected = []
    for block in range(12):
        expected.extend(
            (
                f"blocks.{block}.attn.qkv.weight",
                f"blocks.{block}.attn.proj.weight",
                f"blocks.{block}.mlp.fc1.weight",
                f"blocks.{block}.mlp.fc2.weight",
            )
        )
    if len(unnamed) != len(expected):
        raise RuntimeError(f"Sieve ONNX has {len(unnamed)} folded matrices; expected {len(expected)}")
    for key, value in zip(expected, unnamed, strict=True):
        state[key] = value.T.contiguous()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Sieve ONNX reconstruction mismatch: missing={missing}, unexpected={unexpected}")
    return model, sha256_file(path)


def load_candidate(spec: CandidateSpec, device: str = "cpu"):
    import torch

    if spec.kind == "community_forensics":
        from .inference import _load_model

        model, _, snapshot = _load_model(spec.repository, spec.revision, device)
        fingerprint = hashlib.sha256(str(snapshot).encode()).hexdigest()
    elif spec.kind == "sieve_onnx":
        if not spec.onnx_path:
            raise ValueError("Sieve candidate has no ONNX path")
        model, fingerprint = _sieve_from_onnx(spec.onnx_path)
        model.to(device)
    elif spec.kind in {"convnext_scratch", "convnext_large_scratch", "forensic_convnext_scratch"}:
        from .architectures import (
            build_convnext_binary,
            build_convnext_large_binary,
            build_forensic_convnext_binary,
        )

        builders = {
            "convnext_scratch": build_convnext_binary,
            "convnext_large_scratch": build_convnext_large_binary,
            "forensic_convnext_scratch": build_forensic_convnext_binary,
        }
        model = builders[spec.kind]()
        model.to(device)
        architecture_path = Path(__file__).with_name("architectures.py")
        fingerprint = hashlib.sha256(
            f"{sha256_file(architecture_path)}\0{spec.candidate_id}\0{spec.revision}".encode()
        ).hexdigest()
    else:
        from .dinov3 import load_backbone

        model, checkpoint = load_backbone(spec.model_name, spec.repository, spec.revision, device)
        model.reset_classifier(1)
        model.to(device)
        fingerprint = sha256_file(checkpoint)
    if not isinstance(model.get_classifier(), torch.nn.Module):
        raise RuntimeError(f"{spec.candidate_id} exposes no trainable classifier")
    return model, fingerprint


def _uses_channels_last(spec: CandidateSpec) -> bool:
    return "convnext" in spec.kind or spec.model_name.startswith("convnext")


def _resize_crop(image: Image.Image, input_size: int, random_crop: bool) -> Image.Image:
    resize_size = 440 if input_size == 384 else max(input_size, round(input_size / 0.875))
    width, height = image.size
    scale = resize_size / min(width, height)
    image = image.resize(
        (max(input_size, round(width * scale)), max(input_size, round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    max_left, max_top = image.width - input_size, image.height - input_size
    if random_crop:
        left = random.randint(0, max_left) if max_left else 0
        top = random.randint(0, max_top) if max_top else 0
    else:
        left, top = round(max_left / 2), round(max_top / 2)
    return image.crop((left, top, left + input_size, top + input_size))


def _tensor(image: Image.Image):
    import torch

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class _TrainDataset:
    def __init__(self, rows: list[dict[str, Any]], input_size: int):
        self.rows = rows
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        try:
            with Image.open(row["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image = _resize_crop(image, self.input_size, random_crop=True)
                if random.random() < 0.5:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                return _tensor(image), torch.tensor(float(row["label"])), index
        except Exception:
            return torch.zeros(3, self.input_size, self.input_size), torch.tensor(-1.0), index


def _codec_roundtrip(image: Image.Image, codec: str, quality: int) -> Image.Image:
    output = io.BytesIO()
    if codec == "jpeg":
        image.convert("RGB").save(output, format="JPEG", quality=quality, subsampling="4:2:0")
    elif codec == "webp":
        image.convert("RGB").save(output, format="WEBP", quality=quality, method=0)
    else:
        raise ValueError(f"unknown codec {codec!r}")
    output.seek(0)
    with Image.open(output) as decoded:
        return decoded.convert("RGB").copy()


def _full_train_augment(image: Image.Image, input_size: int, rng: random.Random) -> Image.Image:
    """Class-symmetric mixture of clean, CDN, and multi-hop web transformations."""
    interpolations = (
        Image.Resampling.NEAREST,
        Image.Resampling.BILINEAR,
        Image.Resampling.BICUBIC,
        Image.Resampling.LANCZOS,
    )
    if rng.random() < 0.18 and min(image.size) >= 96:
        left = round(image.width * rng.uniform(0.0, 0.08))
        right = round(image.width * rng.uniform(0.0, 0.08))
        top = round(image.height * rng.uniform(0.0, 0.08))
        bottom = round(image.height * rng.uniform(0.0, 0.08))
        image = image.crop((left, top, max(left + 1, image.width - right), max(top + 1, image.height - bottom)))
    if rng.random() < 0.72:
        scale = rng.uniform(0.40, 1.30)
        image = image.resize(
            (max(64, round(image.width * scale)), max(64, round(image.height * scale))),
            rng.choice(interpolations),
        )
    if rng.random() < 0.08:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.75)))
    for probability in (0.82, 0.30):
        if rng.random() < probability:
            codec = "webp" if rng.random() < 0.22 else "jpeg"
            image = _codec_roundtrip(image, codec, rng.randint(30, 95))
    resize_size = 440 if input_size == 384 else max(input_size, round(input_size / 0.875))
    scale = resize_size / min(image.size)
    image = image.resize(
        (max(input_size, round(image.width * scale)), max(input_size, round(image.height * scale))),
        rng.choice((Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS)),
    )
    max_left, max_top = image.width - input_size, image.height - input_size
    left = rng.randint(0, max_left) if max_left else 0
    top = rng.randint(0, max_top) if max_top else 0
    image = image.crop((left, top, left + input_size, top + input_size))
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image


def _low_quality_train_augment(
    image: Image.Image, input_size: int, rng: random.Random
) -> Image.Image:
    """Preserve clean exposure while explicitly simulating tiny web-source pixels."""
    if rng.random() < 0.30:
        return _full_train_augment(image, input_size, rng)

    interpolations = (
        Image.Resampling.NEAREST,
        Image.Resampling.BILINEAR,
        Image.Resampling.BICUBIC,
        Image.Resampling.LANCZOS,
    )
    if rng.random() < 0.25 and min(image.size) >= 96:
        left = round(image.width * rng.uniform(0.0, 0.10))
        right = round(image.width * rng.uniform(0.0, 0.10))
        top = round(image.height * rng.uniform(0.0, 0.10))
        bottom = round(image.height * rng.uniform(0.0, 0.10))
        image = image.crop(
            (left, top, max(left + 1, image.width - right), max(top + 1, image.height - bottom))
        )

    target_short = rng.choices(
        (32, 48, 64, 80, 96, 128, 160, 192, 256),
        weights=(2, 3, 5, 5, 6, 6, 4, 3, 2),
        k=1,
    )[0]
    if min(image.size) > target_short:
        factor = target_short / min(image.size)
        image = image.resize(
            (max(1, round(image.width * factor)), max(1, round(image.height * factor))),
            rng.choice(interpolations),
        )

    if rng.random() < 0.16:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.20, 1.10)))
    if rng.random() < 0.12:
        array = np.asarray(image, dtype=np.float32).copy()
        noise = np.random.default_rng(rng.getrandbits(64)).normal(
            0.0, rng.uniform(1.0, 5.0), array.shape
        )
        image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "RGB")
    if rng.random() < 0.12:
        image = image.filter(
            ImageFilter.UnsharpMask(
                radius=rng.uniform(0.5, 1.5),
                percent=rng.randint(110, 190),
                threshold=rng.randint(1, 4),
            )
        )

    generations = rng.choices((0, 1, 2, 3), weights=(1, 6, 3, 1), k=1)[0]
    for _ in range(generations):
        codec = "webp" if rng.random() < 0.30 else "jpeg"
        image = _codec_roundtrip(image, codec, rng.randint(12, 78))

    resize_size = 440 if input_size == 384 else max(input_size, round(input_size / 0.875))
    factor = resize_size / min(image.size)
    image = image.resize(
        (max(input_size, round(image.width * factor)), max(input_size, round(image.height * factor))),
        rng.choice(
            (Image.Resampling.NEAREST, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC)
        ),
    )
    max_left, max_top = image.width - input_size, image.height - input_size
    left = rng.randint(0, max_left) if max_left else 0
    top = rng.randint(0, max_top) if max_top else 0
    image = image.crop((left, top, left + input_size, top + input_size))
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image


class _FullTrainDataset:
    def __init__(
        self, rows: list[dict[str, Any]], input_size: int, augmentation_profile: str = "web"
    ):
        if augmentation_profile not in {"web", "low-quality"}:
            raise ValueError(f"unknown augmentation profile {augmentation_profile!r}")
        self.rows = rows
        self.input_size = input_size
        self.augmentation_profile = augmentation_profile

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, request):
        import torch

        index, draw_seed = request
        row = self.rows[index]
        try:
            with Image.open(row["path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                augment = (
                    _low_quality_train_augment
                    if self.augmentation_profile == "low-quality"
                    else _full_train_augment
                )
                image = augment(image, self.input_size, random.Random(draw_seed))
                return _tensor(image), torch.tensor(float(row["label"])), index
        except Exception:
            return torch.zeros(3, self.input_size, self.input_size), torch.tensor(-1.0), index


class _BalancedSourceBatchSampler:
    """Stateless exact-class batches with temperature-balanced source sampling."""

    def __init__(
        self,
        rows: list[dict[str, Any]],
        batch_size: int,
        start_step: int,
        end_step: int,
        seed: int,
        source_exponent: float = 0.5,
    ):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("full-training batch size must be positive and even")
        self.batch_size = batch_size
        self.start_step = start_step
        self.end_step = end_step
        self.seed = seed
        if not 0.0 <= source_exponent <= 1.0:
            raise ValueError("source sampling exponent must be between zero and one")
        self.source_exponent = source_exponent
        grouped: dict[int, dict[str, list[int]]] = {0: defaultdict(list), 1: defaultdict(list)}
        for index, row in enumerate(rows):
            grouped[int(row["label"])][str(row["source"])].append(index)
        if not grouped[0] or not grouped[1]:
            raise ValueError("full-training manifest requires both classes and at least one source per class")
        self.grouped = grouped
        self.sources = {label: sorted(groups) for label, groups in grouped.items()}
        self.source_weights = {
            label: [len(groups[source]) ** source_exponent for source in self.sources[label]]
            for label, groups in grouped.items()
        }

    def __len__(self) -> int:
        return max(0, self.end_step - self.start_step)

    def __iter__(self):
        half = self.batch_size // 2
        for step in range(self.start_step, self.end_step):
            digest = hashlib.sha256(f"{self.seed}\0{step}".encode()).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
            batch = []
            for label in (0, 1):
                for _ in range(half):
                    source = rng.choices(self.sources[label], weights=self.source_weights[label], k=1)[0]
                    index = rng.choice(self.grouped[label][source])
                    batch.append((index, rng.getrandbits(64)))
            rng.shuffle(batch)
            yield batch


class _EvalDataset:
    def __init__(self, rows: list[dict[str, Any]], attack: AttackCase, input_size: int):
        self.rows = rows
        self.attack = attack
        self.input_size = input_size
        self.real_rows = sorted((row for row in rows if int(row["label"]) == 0), key=lambda row: row["sample_id"])

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
                    digest = hashlib.sha256(f"{row['sample_id']}\0{self.attack.attack_id}".encode()).digest()
                    background_row = self.real_rows[int.from_bytes(digest[:8], "big") % len(self.real_rows)]
                    with Image.open(background_row["path"]) as background_source:
                        background = ImageOps.exif_transpose(background_source).convert("RGB").copy()
                image = apply_attack(image, self.attack, str(row["sample_id"]), background=background)
                image = _resize_crop(image, self.input_size, random_crop=False)
                return _tensor(image), index, ""
        except Exception as exc:
            return torch.zeros(3, self.input_size, self.input_size), index, f"{type(exc).__name__}: {exc}"


def _worker_seed(worker_id: int) -> None:
    import torch

    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _state_dict(model):
    return getattr(model, "_orig_mod", model).state_dict()


def train_pilot(
    manifest_path: str | Path,
    candidate_id: str,
    output_dir: str | Path,
    steps: int = 600,
    batch_size: int = 96,
    learning_rate: float = 2e-5,
    head_multiplier: float = 10.0,
    weight_decay: float = 0.05,
    warmup_fraction: float = 0.10,
    workers: int = 4,
    seed: int = 323,
    compile_mode: str = "reduce-overhead",
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    if candidate_id not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate_id!r}; choose from {sorted(CANDIDATES)}")
    spec = CANDIDATES[candidate_id]
    rows = [row for row in read_manifest(manifest_path) if row["role"] == "train"]
    counts = {label: sum(int(row["label"]) == label for row in rows) for label in (0, 1)}
    if not rows or counts[0] != counts[1]:
        raise ValueError(f"pilot manifest must be nonempty and class-balanced, got {counts}")
    _seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    model, initial_fingerprint = load_candidate(spec, "cuda")
    channels_last = _uses_channels_last(spec)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    classifier = model.get_classifier()
    head_ids = {id(parameter) for parameter in classifier.parameters()}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    head_parameters = list(classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": learning_rate},
            {"params": head_parameters, "lr": learning_rate * head_multiplier},
        ],
        weight_decay=weight_decay,
        fused=True,
    )
    eager_model = model
    compile_started = time.monotonic()
    model = torch.compile(model, mode=compile_mode, fullgraph=False, dynamic=False)
    compile_setup_seconds = time.monotonic() - compile_started
    warmup_steps = max(1, round(steps * warmup_fraction))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        _TrainDataset(rows, spec.input_size),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=min(workers, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=True,
        worker_init_fn=_worker_seed,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    failures = 0
    samples = 0
    started = time.monotonic()
    step = 0
    model.train()
    while step < steps:
        for images, labels, _ in loader:
            keep = labels >= 0
            failures += int((~keep).sum())
            if int(keep.sum()) < 2:
                continue
            images = images[keep].cuda(non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            labels = labels[keep].cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images).reshape(-1)
                loss = functional.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(eager_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            samples += int(keep.sum())
            if step == 1 or step % 50 == 0 or step == steps:
                elapsed = time.monotonic() - started
                entry = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "backbone_lr": optimizer.param_groups[0]["lr"],
                    "head_lr": optimizer.param_groups[1]["lr"],
                    "samples_per_second": samples / elapsed,
                    "max_cuda_bytes": torch.cuda.max_memory_allocated(),
                }
                history.append(entry)
                print(json.dumps(entry), flush=True)
            if step >= steps:
                break
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    checkpoint = output_dir / "pilot.pt"
    torch.save(
        {
            "candidate": asdict(spec),
            "model_state_dict": _state_dict(model),
            "training": {
                "steps": steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "head_multiplier": head_multiplier,
                "weight_decay": weight_decay,
                "warmup_fraction": warmup_fraction,
                "seed": seed,
                "compile_mode": compile_mode,
                "amp_dtype": "bfloat16",
                "fused_adamw": True,
                "tf32": True,
                "channels_last": channels_last,
                "augmentation": "resize-short/random-crop + horizontal-flip; no degradation",
            },
        },
        checkpoint,
    )
    metadata = {
        "protocol": "aiblink-equal-budget-finetune/0.1.0",
        "candidate": asdict(spec),
        "manifest_sha256": sha256_file(manifest_path),
        "initial_checkpoint_fingerprint": initial_fingerprint,
        "output_checkpoint_sha256": sha256_file(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in eager_model.parameters()),
        "train_rows": len(rows),
        "class_counts": counts,
        "steps": steps,
        "samples_seen": samples,
        "decode_failures": failures,
        "seconds": seconds,
        "samples_per_second": samples / seconds,
        "compile_setup_seconds": compile_setup_seconds,
        "max_cuda_bytes": torch.cuda.max_memory_allocated(),
        "history": history,
    }
    atomic_json(output_dir / "training.json", metadata)
    return metadata


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _validation_subset(rows: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    if cap <= 0 or len(rows) <= cap:
        return sorted(rows, key=lambda row: str(row["sample_id"]))
    per_class = cap // 2
    selected = []
    for label in (0, 1):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if int(row["label"]) == label:
                groups[str(row["source"])].append(row)
        for source in groups:
            groups[source].sort(
                key=lambda row: hashlib.sha256(f"{seed}\0{row['sample_id']}".encode()).digest()
            )
        sources = sorted(groups)
        offsets = {source: 0 for source in sources}
        while len([row for row in selected if int(row["label"]) == label]) < per_class:
            progressed = False
            for source in sources:
                offset = offsets[source]
                if offset < len(groups[source]):
                    selected.append(groups[source][offset])
                    offsets[source] += 1
                    progressed = True
                    if len([row for row in selected if int(row["label"]) == label]) >= per_class:
                        break
            if not progressed:
                break
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def _evaluate_training_model(
    model,
    spec: CandidateSpec,
    rows: list[dict[str, Any]],
    views: tuple[str, ...],
    batch_size: int,
    workers: int,
    threshold: float,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    records = []
    failures = 0
    channels_last = _uses_channels_last(spec)
    model.eval()
    with torch.inference_mode():
        for view in views:
            attack = legacy_case(view)
            loader = DataLoader(
                _EvalDataset(rows, attack, spec.input_size),
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(workers, os.cpu_count() or 1),
                pin_memory=True,
                persistent_workers=workers > 0,
                prefetch_factor=2 if workers > 0 else None,
            )
            for images, indexes, errors in loader:
                images = images.cuda(non_blocking=True)
                if channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(images).reshape(-1).float().cpu().numpy()
                for logit, index, error in zip(logits, indexes.tolist(), errors, strict=True):
                    source = rows[index]
                    failures += int(bool(error))
                    if not error:
                        records.append(
                            {
                                **source,
                                "view": view,
                                "raw_logit": float(logit),
                            }
                        )
    model.train()
    if failures:
        raise RuntimeError(f"development evaluation had {failures} decode failures")
    # Bias-only calibration is monotone and cheap enough for recurring checkpoint
    # selection. Run the full identity/bias/Platt comparison once after training.
    calibrator, diagnostics = select_and_fit(records, ["bias"], threshold)
    selected = diagnostics["selected"]
    metrics = diagnostics["candidates"][selected]
    y = np.asarray([int(row["label"]) for row in records])
    logits = np.asarray([float(row["raw_logit"]) for row in records])
    raw = binary_metrics(y, sigmoid(logits), threshold)
    return {
        "rows": len(rows),
        "predictions": len(records),
        "views": list(views),
        "selected_calibrator": selected,
        "calibrator": calibrator.to_dict(),
        "logo_macro_balanced_accuracy": metrics["logo_macro_balanced_accuracy"],
        "logo_pooled_balanced_accuracy": metrics["logo_pooled_balanced_accuracy"],
        "logo_log_loss": metrics["logo_log_loss"],
        "raw_roc_auc": raw["roc_auc"],
    }


def train_full(
    manifest_path: str | Path,
    initial_checkpoint_path: str | Path,
    output_dir: str | Path,
    steps: int = 4500,
    batch_size: int = 96,
    learning_rate: float = 2e-5,
    head_multiplier: float = 1.0,
    weight_decay: float = 0.05,
    warmup_steps: int = 300,
    workers: int = 4,
    seed: int = 323,
    compile_mode: str = "reduce-overhead",
    validation_every: int = 750,
    checkpoint_every: int = 250,
    validation_cap: int = 6000,
    ema_decay: float = 0.999,
    source_sampling_exponent: float = 0.5,
    resume_path: str | Path | None = None,
    threshold: float = 0.65,
    validation_batch_size: int | None = None,
    augmentation_profile: str = "web",
) -> dict[str, Any]:
    """Resumable, source-balanced full fine-tune with class-symmetric web laundering."""
    import torch
    import torch.nn.functional as functional
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    from torch.utils.data import DataLoader

    manifest_path = Path(manifest_path)
    initial_checkpoint_path = Path(initial_checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest_path)
    train_rows = [row for row in rows if row["role"] == "train"]
    validation_rows = _validation_subset(
        [row for row in rows if row["role"] == "calibration"], validation_cap, seed
    )
    if len({str(row["group_id"]) for row in validation_rows if int(row["label"]) == 1}) < 3:
        raise ValueError("full training requires at least three fake development groups")
    class_counts = {label: sum(int(row["label"]) == label for row in train_rows) for label in (0, 1)}
    source_counts = {
        label: len({str(row["source"]) for row in train_rows if int(row["label"]) == label})
        for label in (0, 1)
    }
    if not all(class_counts.values()):
        raise ValueError(f"full-training manifest must contain both classes, got {class_counts}")

    _seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    saved_initial = torch.load(initial_checkpoint_path, map_location="cpu", weights_only=False)
    spec = CandidateSpec(**saved_initial["candidate"])
    model, base_fingerprint = load_candidate(spec, "cpu")
    model.load_state_dict(saved_initial["model_state_dict"], strict=True)
    model.cuda()
    channels_last = _uses_channels_last(spec)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    classifier = model.get_classifier()
    head_ids = {id(parameter) for parameter in classifier.parameters()}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    head_parameters = list(classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": learning_rate},
            {"params": head_parameters, "lr": learning_rate * head_multiplier},
        ],
        weight_decay=weight_decay,
        fused=True,
    )

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    ema_model = AveragedModel(
        model, device="cuda", multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True
    )
    start_step = 0
    best_metric = -math.inf
    best_step = 0
    history: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    failures = 0
    samples = 0
    resume_path = Path(resume_path) if resume_path else None
    if resume_path and resume_path.exists():
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resumed["candidate"]["candidate_id"] != spec.candidate_id:
            raise ValueError("resume candidate does not match initial checkpoint")
        if resumed["manifest_sha256"] != sha256_file(manifest_path):
            raise ValueError("resume manifest fingerprint does not match")
        model.load_state_dict(resumed.get("training_model_state_dict", resumed["model_state_dict"]), strict=True)
        if "ema_state_dict" in resumed:
            ema_model.load_state_dict(resumed["ema_state_dict"], strict=True)
        else:
            ema_model.module.load_state_dict(resumed["model_state_dict"], strict=True)
            ema_model.n_averaged.fill_(max(1, int(resumed["step"])))
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scheduler.load_state_dict(resumed["scheduler_state_dict"])
        start_step = int(resumed["step"])
        best_metric = float(resumed.get("best_metric", best_metric))
        best_step = int(resumed.get("best_step", best_step))
        history = list(resumed.get("history", []))
        validations = list(resumed.get("validations", []))
        failures = int(resumed.get("decode_failures", 0))
        samples = int(resumed.get("samples_seen", 0))
        if "torch_rng_state" in resumed:
            torch.set_rng_state(resumed["torch_rng_state"])
        if "cuda_rng_state" in resumed:
            torch.cuda.set_rng_state(resumed["cuda_rng_state"])
    if start_step >= steps:
        raise ValueError(f"resume checkpoint is already at step {start_step}, target is {steps}")

    loader = DataLoader(
        _FullTrainDataset(train_rows, spec.input_size, augmentation_profile),
        batch_sampler=_BalancedSourceBatchSampler(
            train_rows, batch_size, start_step, steps, seed, source_sampling_exponent
        ),
        num_workers=min(workers, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        worker_init_fn=_worker_seed,
    )
    eager_model = model
    compile_started = time.monotonic()
    model = torch.compile(model, mode=compile_mode, fullgraph=False, dynamic=False)
    compile_setup_seconds = time.monotonic() - compile_started
    started = time.monotonic()

    def checkpoint_payload(step: int) -> dict[str, Any]:
        return {
            "protocol": "aiblink-full-finetune/0.1.0",
            "candidate": asdict(spec),
            "manifest_sha256": sha256_file(manifest_path),
            "initial_checkpoint_sha256": sha256_file(initial_checkpoint_path),
            "model_state_dict": ema_model.module.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "training_model_state_dict": eager_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "steps": steps,
            "best_metric": best_metric,
            "best_step": best_step,
            "samples_seen": samples,
            "decode_failures": failures,
            "history": history,
            "validations": validations,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "training": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "head_multiplier": head_multiplier,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "seed": seed,
                "compile_mode": compile_mode,
                "ema_decay": ema_decay,
                "sampler": "exact-class-balanced/source-temperature/stateless-per-step",
                "source_sampling_exponent": source_sampling_exponent,
                "validation_batch_size": evaluation_batch_size,
                "augmentation_profile": augmentation_profile,
                "augmentation": (
                    "class-symmetric 30% web mixture plus 70% tiny-source 32-256px/"
                    "JPEG+WebP/blur/noise/unsharp mixture"
                    if augmentation_profile == "low-quality"
                    else "class-symmetric clean/CDN/multi-hop JPEG+WebP/resize/crop/blur mixture"
                ),
            },
        }

    evaluation_batch_size = validation_batch_size or max(batch_size, 96)
    if evaluation_batch_size < 1:
        raise ValueError("validation batch size must be positive")
    step = start_step
    if start_step == 0 and not validations and validation_every > 0:
        validation_started = time.monotonic()
        metrics = _evaluate_training_model(
            ema_model.module, spec, validation_rows, ("clean", "web", "hard"),
            evaluation_batch_size, workers, threshold,
        )
        metrics["step"] = 0
        metrics["seconds"] = time.monotonic() - validation_started
        validations.append(metrics)
        best_metric = float(metrics["logo_macro_balanced_accuracy"])
        best_step = 0
        print(json.dumps({"initial_validation": metrics}), flush=True)
        _atomic_torch_save(checkpoint_payload(0), output_dir / "best.pt")

    started = time.monotonic()
    run_start_samples = samples
    model.train()
    for images, labels, _ in loader:
        keep = labels >= 0
        failures += int((~keep).sum())
        if int(keep.sum()) < 2:
            continue
        images = images[keep].cuda(non_blocking=True)
        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        labels = labels[keep].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images).reshape(-1)
            loss = functional.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(eager_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema_model.update_parameters(eager_model)
        step += 1
        samples += int(keep.sum())

        if step == start_step + 1 or step % 50 == 0 or step == steps:
            elapsed = time.monotonic() - started
            entry = {
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                "backbone_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
                "samples_per_second": max(0, samples - run_start_samples) / max(1e-9, elapsed),
                "max_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)

        if validation_every > 0 and (step % validation_every == 0 or step == steps):
            torch.cuda.synchronize()
            validation_started = time.monotonic()
            metrics = _evaluate_training_model(
                ema_model.module, spec, validation_rows, ("clean", "web", "hard"),
                evaluation_batch_size, workers, threshold,
            )
            metrics["step"] = step
            metrics["seconds"] = time.monotonic() - validation_started
            validations.append(metrics)
            print(json.dumps({"validation": metrics}), flush=True)
            metric = float(metrics["logo_macro_balanced_accuracy"])
            if metric > best_metric:
                best_metric = metric
                best_step = step
                _atomic_torch_save(checkpoint_payload(step), output_dir / "best.pt")
                print(json.dumps({"new_best": best_metric, "step": best_step}), flush=True)

        if step % checkpoint_every == 0 or step == steps:
            _atomic_torch_save(checkpoint_payload(step), output_dir / "last.pt")
            atomic_json(
                output_dir / "progress.json",
                {
                    "candidate": spec.candidate_id,
                    "step": step,
                    "steps": steps,
                    "best_metric": best_metric,
                    "best_step": best_step,
                    "samples_seen": samples,
                    "decode_failures": failures,
                    "updated_unix": time.time(),
                },
            )
        if step >= steps:
            break

    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    best_path = output_dir / "best.pt"
    if not best_path.exists() and resume_path:
        resumed_best = resume_path.parent / "best.pt"
        if resumed_best.exists():
            shutil.copy2(resumed_best, best_path)
    if not best_path.exists():
        _atomic_torch_save(checkpoint_payload(step), best_path)
    metadata = {
        "protocol": "aiblink-full-finetune/0.1.0",
        "candidate": asdict(spec),
        "manifest_sha256": sha256_file(manifest_path),
        "initial_checkpoint_sha256": sha256_file(initial_checkpoint_path),
        "base_checkpoint_fingerprint": base_fingerprint,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "class_counts": class_counts,
        "source_counts": source_counts,
        "start_step": start_step,
        "steps": step,
        "samples_seen": samples,
        "decode_failures": failures,
        "seconds": seconds,
        "samples_per_second": max(0, samples - run_start_samples) / max(1e-9, seconds),
        "compile_setup_seconds": compile_setup_seconds,
        "max_cuda_bytes": torch.cuda.max_memory_allocated(),
        "best_metric": best_metric,
        "best_step": best_step,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(output_dir / "last.pt"),
        "history": history,
        "validations": validations,
        "test_role_opened": False,
    }
    atomic_json(output_dir / "training.json", metadata)
    return metadata


def score_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    role: str = "calibration",
    views: tuple[str, ...] = ("clean", "web", "hard"),
    batch_size: int = 128,
    workers: int = 4,
    attacks: list[AttackCase] | None = None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from .inference import _ImageDataset

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = CandidateSpec(**saved["candidate"])
    model, _ = load_candidate(spec, "cpu")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval().cuda()
    checkpoint_fingerprint = sha256_file(checkpoint_path)
    channels_last = _uses_channels_last(spec)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    rows = [row for row in read_manifest(manifest_path) if row["role"] == role]
    attack_cases = attacks or [legacy_case(view) for view in views]
    records = []
    failures = 0
    started = time.monotonic()
    with torch.inference_mode():
        for attack in attack_cases:
            loader = DataLoader(
                _ImageDataset(rows, attack, spec.input_size, rows), batch_size=batch_size, shuffle=False,
                num_workers=min(workers, os.cpu_count() or 1), pin_memory=True,
                persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
            )
            for images, indexes, errors in loader:
                images = images.cuda(non_blocking=True)
                if channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(images).reshape(-1).float().cpu().numpy()
                for logit, index, error in zip(logits, indexes.tolist(), errors, strict=True):
                    source = rows[index]
                    failures += int(bool(error))
                    records.append(
                        {
                            "sample_id": source["sample_id"], "view": attack.attack_id,
                            "raw_logit": None if error else float(logit), "error": error or None,
                            "model_id": spec.candidate_id, "model_revision": checkpoint_fingerprint,
                            "model_input_size": spec.input_size,
                            "model_resize_short_edge": 440 if spec.input_size == 384 else round(spec.input_size / 0.875),
                            **attack.metadata(),
                        }
                    )
    atomic_jsonl(output_path, records)
    elapsed = time.monotonic() - started
    return {
        "candidate": spec.candidate_id,
        "rows": len(rows),
        "views": [attack.attack_id for attack in attack_cases],
        "predictions": len(records),
        "failures": failures,
        "seconds": elapsed,
    }


def rank_calibration(
    manifest_path: str | Path,
    prediction_paths: list[str | Path],
    output_path: str | Path,
    threshold: float = 0.65,
) -> dict[str, Any]:
    manifest = {row["sample_id"]: row for row in read_manifest(manifest_path) if row["role"] == "calibration"}
    ranking = []
    for path in prediction_paths:
        from .io import read_jsonl

        predictions = list(read_jsonl(path))
        rows = []
        for prediction in predictions:
            source = manifest[prediction["sample_id"]]
            if prediction.get("error") or prediction.get("raw_logit") is None:
                continue
            rows.append({**source, **prediction})
        calibrator, diagnostics = select_and_fit(rows, ["identity", "bias", "platt"], threshold)
        selected = diagnostics["selected"]
        oof = diagnostics["candidates"][selected]
        y = np.asarray([int(row["label"]) for row in rows])
        logits = np.asarray([float(row["raw_logit"]) for row in rows])
        uncalibrated = binary_metrics(y, sigmoid(logits), threshold)
        candidate_id = str(predictions[0]["model_id"])
        spec = CANDIDATES[candidate_id]
        ranking.append(
            {
                "candidate_id": candidate_id,
                "license": spec.license,
                "competition_status": spec.competition_status,
                "oof_macro_balanced_accuracy": oof["logo_macro_balanced_accuracy"],
                "oof_pooled_balanced_accuracy": oof["logo_pooled_balanced_accuracy"],
                "oof_log_loss": oof["logo_log_loss"],
                "raw_roc_auc": uncalibrated["roc_auc"],
                "selected_calibrator": selected,
                "final_calibrator": calibrator.to_dict(),
                "prediction_sha256": sha256_file(path),
            }
        )
    ranking.sort(key=lambda row: (-row["oof_macro_balanced_accuracy"], -row["raw_roc_auc"]))
    eligible = [row for row in ranking if row["competition_status"] == "eligible"]
    excluded = [row for row in ranking if row["competition_status"] != "eligible"]
    if not eligible:
        raise ValueError("tournament has no competition-eligible candidates")
    result = {
        "protocol": "aiblink-tournament-ranking/0.1.0",
        "selection_role": "calibration",
        "test_role_opened": False,
        "selected_candidate_id": eligible[0]["candidate_id"],
        "ranking": eligible,
        "excluded": excluded,
    }
    atomic_json(output_path, result)
    return result
