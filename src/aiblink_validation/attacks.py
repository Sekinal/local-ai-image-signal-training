from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class AttackCase:
    attack_id: str
    family: str
    severity: str
    operations: tuple[dict[str, Any], ...]
    seed: int = 323

    def metadata(self) -> dict[str, Any]:
        return {
            "attack_id": self.attack_id,
            "attack_family": self.family,
            "attack_severity": self.severity,
            "attack_operations": list(self.operations),
        }


def load_attack_profile(path: str | Path, profile: str) -> tuple[list[AttackCase], dict[str, Any]]:
    config = yaml.safe_load(Path(path).read_text())
    if profile not in config["profiles"]:
        raise ValueError(f"unknown attack profile {profile!r}")
    seed = int(config.get("seed", 323))
    cases = []
    for attack_id in config["profiles"][profile]:
        try:
            case = config["cases"][attack_id]
        except KeyError as exc:
            raise ValueError(f"profile references missing case {attack_id!r}") from exc
        cases.append(
            AttackCase(
                attack_id=attack_id,
                family=str(case["family"]),
                severity=str(case["severity"]),
                operations=tuple(dict(operation) for operation in case.get("operations", [])),
                seed=seed,
            )
        )
    if not any(case.attack_id == "clean" for case in cases):
        raise ValueError("every attack profile must include a clean control")
    return cases, config


def legacy_case(view: str) -> AttackCase:
    if view == "clean":
        return AttackCase("clean", "clean", "control", ())
    settings = {"web": (768, 60), "hard": (512, 40)}
    if view not in settings:
        raise ValueError(f"unknown legacy view {view!r}")
    target, quality = settings[view]
    return AttackCase(
        view,
        "web_degradation",
        "medium" if view == "web" else "high",
        (
            {"op": "limit_long", "pixels": target, "filter": "bilinear"},
            {"op": "jpeg", "quality": quality, "subsampling": 2},
        ),
    )


def _rng(sample_id: str, attack: AttackCase) -> np.random.Generator:
    digest = hashlib.sha256(f"{attack.seed}\0{sample_id}\0{attack.attack_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _resampling(name: str) -> Image.Resampling:
    values = {
        "nearest": Image.Resampling.NEAREST,
        "box": Image.Resampling.BOX,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown resize filter {name!r}") from exc


def _roundtrip(image: Image.Image, image_format: str, **save_options: Any) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=image_format, **save_options)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _resize_short(image: Image.Image, pixels: int, method: Image.Resampling) -> Image.Image:
    width, height = image.size
    factor = pixels / min(width, height)
    return image.resize((max(1, round(width * factor)), max(1, round(height * factor))), method)


def _apply(
    image: Image.Image,
    operation: dict[str, Any],
    rng: np.random.Generator,
    background: Image.Image | None = None,
) -> Image.Image:
    name = operation["op"]
    if name == "limit_long":
        width, height = image.size
        target = int(operation["pixels"])
        if max(width, height) <= target:
            return image
        factor = target / max(width, height)
        return image.resize(
            (max(1, round(width * factor)), max(1, round(height * factor))),
            _resampling(operation.get("filter", "bilinear")),
        )
    if name == "resize_short":
        return _resize_short(image, int(operation["pixels"]), _resampling(operation.get("filter", "lanczos")))
    if name == "scale":
        factor = float(operation["factor"])
        return image.resize(
            (max(1, round(image.width * factor)), max(1, round(image.height * factor))),
            _resampling(operation.get("filter", "bicubic")),
        )
    if name == "scale_cycle":
        original = image.size
        factor = float(operation["scale"])
        method = _resampling(operation.get("filter", "lanczos"))
        image = image.resize((max(1, round(original[0] * factor)), max(1, round(original[1] * factor))), method)
        return image.resize(original, method)
    if name == "jpeg":
        for _ in range(int(operation.get("generations", 1))):
            image = _roundtrip(
                image,
                "JPEG",
                quality=int(operation["quality"]),
                subsampling=int(operation.get("subsampling", 2)),
            )
        return image
    if name == "webp":
        for _ in range(int(operation.get("generations", 1))):
            image = _roundtrip(image, "WEBP", quality=int(operation["quality"]), method=4)
        return image
    if name == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(float(operation["radius"])))
    if name == "median":
        return image.filter(ImageFilter.MedianFilter(int(operation.get("size", 3))))
    if name == "unsharp":
        return image.filter(
            ImageFilter.UnsharpMask(
                radius=float(operation.get("radius", 2)),
                percent=int(operation.get("percent", 150)),
                threshold=int(operation.get("threshold", 3)),
            )
        )
    if name == "gaussian_noise":
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        array += rng.normal(0, float(operation["sigma"]), array.shape)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    if name == "crop":
        retained = float(operation["retained"])
        side_factor = math.sqrt(retained)
        crop_width, crop_height = max(1, round(image.width * side_factor)), max(1, round(image.height * side_factor))
        left = int(rng.integers(0, max(1, image.width - crop_width + 1)))
        top = int(rng.integers(0, max(1, image.height - crop_height + 1)))
        return image.crop((left, top, left + crop_width, top + crop_height))
    if name == "pad":
        scale = float(operation["canvas_scale"])
        width, height = round(image.width * scale), round(image.height * scale)
        mean = tuple(int(value) for value in np.asarray(image.resize((1, 1))).reshape(3))
        canvas = Image.new("RGB", (width, height), mean)
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        return canvas
    if name == "overlay":
        coverage = float(operation["coverage"])
        overlay_height = max(1, round(image.height * coverage))
        output = image.copy()
        draw = ImageDraw.Draw(output, "RGBA")
        top = image.height - overlay_height if rng.random() < 0.5 else 0
        draw.rectangle((0, top, image.width, top + overlay_height), fill=(245, 245, 245, 235))
        stripe = max(2, overlay_height // 9)
        for index in range(1, 7, 2):
            y = top + min(overlay_height - 1, index * stripe)
            draw.rectangle((image.width // 12, y, image.width * 10 // 12, min(top + overlay_height, y + stripe)), fill=(25, 25, 25, 210))
        return output
    if name == "composite":
        if background is None:
            raise ValueError("composite attack requires a class-matched real background")
        canvas = background.convert("RGB").copy()
        fraction = float(operation["foreground_fraction"])
        target_area = max(1, round(canvas.width * canvas.height * fraction))
        ratio = image.width / image.height
        target_width = min(canvas.width, max(1, round(math.sqrt(target_area * ratio))))
        target_height = min(canvas.height, max(1, round(target_width / ratio)))
        foreground = image.copy()
        foreground.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
        max_x, max_y = max(0, canvas.width - foreground.width), max(0, canvas.height - foreground.height)
        x = int(rng.integers(0, max_x + 1)) if max_x else 0
        y = int(rng.integers(0, max_y + 1)) if max_y else 0
        canvas.paste(foreground, (x, y))
        return canvas
    if name == "screenshot_frame":
        scale = float(operation.get("canvas_scale", 1.5))
        width, height = round(image.width * scale), round(image.height * scale)
        top_bar = max(18, round(height * 0.08))
        canvas = Image.new("RGB", (width, height + top_bar), (238, 239, 242))
        fitted = image.copy()
        fitted.thumbnail((round(width * 0.9), round(height * 0.85)), Image.Resampling.LANCZOS)
        x, y = (width - fitted.width) // 2, top_bar + (height - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        draw = ImageDraw.Draw(canvas)
        for index, color in enumerate(((240, 80, 75), (245, 185, 65), (75, 190, 90))):
            draw.ellipse((8 + index * 18, 6, 20 + index * 18, 18), fill=color)
        return canvas
    if name == "moire":
        amplitude, period = float(operation.get("amplitude", 4)), float(operation.get("period", 6))
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        yy, xx = np.indices(array.shape[:2])
        pattern = amplitude * np.sin(2 * np.pi * (xx + 0.71 * yy) / period)
        array += pattern[..., None]
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    if name == "perspective":
        magnitude = float(operation.get("magnitude", 0.02))
        width, height = image.size
        dx, dy = magnitude * width, magnitude * height
        coefficients = (1, magnitude, -dx, -magnitude, 1, dy, 0.00005, -0.00005)
        return image.transform(image.size, Image.Transform.PERSPECTIVE, coefficients, Image.Resampling.BICUBIC)
    if name == "gamma":
        gamma = float(operation["value"])
        table = [round(255 * ((value / 255) ** gamma)) for value in range(256)]
        return image.point(table * 3)
    if name == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(operation["factor"]))
    raise ValueError(f"unsupported attack operation {name!r}")


def apply_attack(
    image: Image.Image,
    attack: AttackCase,
    sample_id: str,
    background: Image.Image | None = None,
) -> Image.Image:
    output = image.convert("RGB")
    rng = _rng(sample_id, attack)
    for operation in attack.operations:
        output = _apply(output, operation, rng, background)
    return output
