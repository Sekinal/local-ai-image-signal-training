from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .hashing import image_metadata_and_phash
from .io import sha256_file, write_manifest

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".avif"}
REAL_NAMES = {"real", "nature", "0", "human", "photo"}
FAKE_NAMES = {"fake", "ai", "1", "generated", "synthetic"}


def _label(name: str) -> int:
    normalized = name.lower().replace("-", "_")
    if normalized in REAL_NAMES:
        return 0
    if normalized in FAKE_NAMES:
        return 1
    raise ValueError(f"unrecognized label directory {name!r}")


def _inspect(task: tuple[Path, Path, str, str, str, str]) -> dict[str, Any]:
    path, root, dataset, role, source, label_name = task
    width, height, image_format, phash = image_metadata_and_phash(path)
    content_hash = sha256_file(path)
    relative = path.relative_to(root).as_posix()
    sample_id = hashlib.sha256(f"{dataset}\0{relative}\0{content_hash}".encode()).hexdigest()[:24]
    label = _label(label_name)
    group_id = f"{dataset}:{'fake' if label else 'real'}:{source}"
    return {
        "sample_id": sample_id,
        "path": str(path.resolve()),
        "label": label,
        "dataset": dataset,
        "source": source,
        "group_id": group_id,
        "role": role,
        "content_sha256": content_hash,
        "phash": phash,
        "width": width,
        "height": height,
        "mime": f"image/{image_format}",
    }


def from_folders(
    root: str | Path,
    dataset: str,
    role: str,
    layout: str,
    output: str | Path,
    workers: int = 4,
) -> list[dict[str, Any]]:
    if layout not in {"generator/label", "label/generator"}:
        raise ValueError("layout must be 'generator/label' or 'label/generator'")
    root = Path(root).resolve()
    tasks = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = path.relative_to(root).parts
        if len(parts) < 3:
            raise ValueError(f"expected at least two directories beneath root: {path}")
        if layout == "generator/label":
            source, label_name = parts[0], parts[1]
        else:
            label_name, source = parts[0], parts[1]
        tasks.append((path, root, dataset, role, source, label_name))
    if not tasks:
        raise ValueError(f"no supported images found beneath {root}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(_inspect, tasks))
    write_manifest(output, rows)
    return rows

