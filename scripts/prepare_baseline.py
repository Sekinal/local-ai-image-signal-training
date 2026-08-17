#!/usr/bin/env python3
"""Materialize a fast, generator-disjoint baseline benchmark.

Calibration: ADM, BigGAN, VQDM fakes + COCO val2017 reals.
Test: GLIDE, Midjourney, SD1.4, SD1.5, Wukong fakes + Tiny-GenImage reals.

Tiny-GenImage is only a compact redistribution of GenImage and is not treated as
the eventual final benchmark. This script exists to establish the stock-model
baseline quickly through the exact same release-grade validation machinery.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from aiblink_validation.hashing import image_metadata_and_phash
from aiblink_validation.io import sha256_file, write_manifest

HF_REVISION = "89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69"
HF_TEMPLATE = (
    "https://huggingface.co/datasets/TheKernel01/Tiny-GenImage/resolve/"
    + HF_REVISION
    + "/data/validation-{shard:05d}-of-00004.parquet"
)
COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
GENERATORS = {0: "Real", 1: "ADM", 2: "BigGAN", 3: "GLIDE", 4: "Midjourney", 5: "SD14", 6: "SD15", 7: "VQDM", 8: "Wukong"}
CALIBRATION_GENERATORS = {"ADM", "BigGAN", "VQDM"}
# Tiny-GenImage declares SD1.4 in its ClassLabel schema but its validation split
# currently contains no SD1.4 rows. Keep the baseline pinned to populations that
# physically exist rather than silently fabricating support.
TEST_GENERATORS = {"GLIDE", "Midjourney", "SD15", "Wukong"}


def download(url: str, output: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    print(f"downloading {url} -> {output}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=4 << 20)
    os.replace(temporary, output)


def stable_priority(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{key}".encode()).hexdigest()


def extract_image_bytes(value: dict) -> tuple[bytes, str]:
    data = value.get("bytes")
    if data is None:
        raise ValueError("parquet image row has no embedded bytes")
    path = value.get("path") or "image.jpg"
    return data, Path(path).suffix.lower() or ".jpg"


def write_genimage(cache: Path, data_root: Path, per_fake: int, seed: int, shards: int) -> list[dict]:
    candidates: dict[str, list[tuple[str, bytes, str]]] = {name: [] for name in GENERATORS.values()}
    for shard in range(shards):
        parquet = cache / f"validation-{shard:05d}-of-00004.parquet"
        download(HF_TEMPLATE.format(shard=shard), parquet)
        table = pq.read_table(parquet, columns=["image", "label", "generator"])
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        generators = table.column("generator").to_pylist()
        for row_index, (image, label, generator_index) in enumerate(zip(images, labels, generators, strict=True)):
            generator = GENERATORS[int(generator_index)]
            expected_label = 0 if generator == "Real" else 1
            if int(label) != expected_label:
                raise ValueError(f"label/generator mismatch in shard {shard}, row {row_index}")
            data, suffix = extract_image_bytes(image)
            key = f"{shard}:{row_index}:{hashlib.sha256(data).hexdigest()}"
            candidates[generator].append((stable_priority(key, seed), data, suffix))

    rows = []
    test_fake_count = per_fake * len(TEST_GENERATORS)
    active = {"Real", *CALIBRATION_GENERATORS, *TEST_GENERATORS}
    limits = {name: (test_fake_count if name == "Real" else per_fake) for name in active}
    for generator, entries in candidates.items():
        if generator not in active:
            continue
        entries.sort(key=lambda entry: entry[0])
        if len(entries) < limits[generator]:
            raise ValueError(f"not enough {generator} rows: {len(entries)} < {limits[generator]}")
        role = "test" if generator == "Real" or generator in TEST_GENERATORS else "calibration"
        label = int(generator != "Real")
        source = "ImageNet" if not label else generator
        directory = data_root / role / ("fake" if label else "real") / source
        directory.mkdir(parents=True, exist_ok=True)
        for index, (_, data, suffix) in enumerate(entries[: limits[generator]]):
            path = directory / f"{index:05d}{suffix}"
            path.write_bytes(data)
            content_hash = hashlib.sha256(data).hexdigest()
            width, height, image_format, phash = image_metadata_and_phash(path)
            rows.append(
                {
                    "sample_id": hashlib.sha256(f"tiny-genimage\0{generator}\0{content_hash}".encode()).hexdigest()[:24],
                    "path": str(path.resolve()),
                    "label": label,
                    "dataset": "Tiny-GenImage-validation",
                    "source": source,
                    "group_id": f"genimage:{'fake' if label else 'real'}:{source}",
                    "role": role,
                    "content_sha256": content_hash,
                    "phash": phash,
                    "width": width,
                    "height": height,
                    "mime": f"image/{image_format}",
                    "license": "GenImage dataset terms",
                }
            )
    return rows


def write_coco(cache: Path, data_root: Path, count: int, seed: int) -> list[dict]:
    archive = cache / "coco-val2017.zip"
    download(COCO_URL, archive)
    extracted = cache / "coco-val2017"
    if not extracted.is_dir():
        print(f"extracting {archive}", flush=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
    images = sorted(extracted.rglob("*.jpg"), key=lambda path: stable_priority(path.name, seed))[:count]
    target = data_root / "calibration" / "real" / "COCO-val2017"
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_path in images:
        path = target / source_path.name
        if not path.exists():
            os.link(source_path, path)
        content_hash = sha256_file(path)
        width, height, image_format, phash = image_metadata_and_phash(path)
        rows.append(
            {
                "sample_id": hashlib.sha256(f"coco-val2017\0{source_path.name}\0{content_hash}".encode()).hexdigest()[:24],
                "path": str(path.resolve()),
                "label": 0,
                "dataset": "COCO-val2017",
                "source": "COCO-val2017",
                "group_id": "coco:real:val2017",
                "role": "calibration",
                "content_sha256": content_hash,
                "phash": phash,
                "width": width,
                "height": height,
                "mime": f"image/{image_format}",
                "license": "COCO image-specific source terms",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/baseline"))
    parser.add_argument("--per-fake-generator", type=int, default=200)
    parser.add_argument("--shards", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--seed", type=int, default=323)
    args = parser.parse_args()
    root = args.root.resolve()
    cache = root / "downloads"
    data_root = root / "images"
    rows = write_genimage(cache, data_root, args.per_fake_generator, args.seed, args.shards)
    calibration_fake_count = args.per_fake_generator * len(CALIBRATION_GENERATORS)
    rows += write_coco(cache, data_root, calibration_fake_count, args.seed)
    manifest = root / "manifest.csv"
    write_manifest(manifest, rows)
    metadata = {
        "seed": args.seed,
        "per_fake_generator": args.per_fake_generator,
        "validation_shards": args.shards,
        "tiny_genimage_revision": HF_REVISION,
        "coco_url": COCO_URL,
        "counts": {
            role: {
                str(label): sum(1 for row in rows if row["role"] == role and row["label"] == label)
                for label in (0, 1)
            }
            for role in ("calibration", "test")
        },
    }
    (root / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
