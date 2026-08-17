#!/usr/bin/env python3
"""Build the source-balanced full-training and development manifests.

The script deliberately never reads OpenFake's test split.  It combines a
deterministic subset of OpenFake train, the prior GenImage pilot subset,
SynthBuster, COCO train2017, and WikiArt.  Original downloads are retained;
materialized training copies are bounded and web-tiered where appropriate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from PIL import Image, ImageOps

from aiblink_validation.hashing import image_metadata_and_phash
from aiblink_validation.io import read_manifest, sha256_file, write_manifest


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def _priority(seed: int, namespace: str, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).digest()[:8], "big")


def _select(rows: Iterable[dict[str, Any]], cap_by_source: dict[str, int], seed: int) -> list[dict[str, Any]]:
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        source = str(row["source"])
        cap = cap_by_source.get(source, 0)
        if cap <= 0:
            continue
        rank = _priority(seed, source, str(row["locator"]))
        entry = (-rank, str(row["locator"]), row)
        heap = heaps[source]
        if len(heap) < cap:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [entry[2] for source in sorted(heaps) for entry in sorted(heaps[source], reverse=True)]


def _metadata_row(
    path: Path,
    *,
    label: int,
    dataset: str,
    source: str,
    group_id: str,
    role: str,
    license_name: str,
) -> dict[str, Any]:
    width, height, image_format, phash = image_metadata_and_phash(path)
    digest = sha256_file(path)
    sample_id = hashlib.sha256(f"{dataset}\0{source}\0{digest}".encode()).hexdigest()[:24]
    return {
        "sample_id": sample_id,
        "path": str(path.resolve()),
        "label": label,
        "dataset": dataset,
        "source": source,
        "group_id": group_id,
        "role": role,
        "content_sha256": digest,
        "phash": phash,
        "width": width,
        "height": height,
        "mime": f"image/{(image_format or path.suffix.lstrip('.')).lower()}",
        "license": license_name,
    }


def _web_tier(data: bytes, max_short_edge: int = 768) -> bytes:
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        if min(image.size) > max_short_edge:
            scale = max_short_edge / min(image.size)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        output = io.BytesIO()
        image.save(output, "JPEG", quality=95, subsampling=0)
        return output.getvalue()


def _openfake_candidates(paths: list[Path], role: str) -> list[dict[str, Any]]:
    candidates = []
    for path in paths:
        table = pq.read_table(path, columns=["label", "model"])
        for index, (label, model) in enumerate(zip(table["label"].to_pylist(), table["model"].to_pylist(), strict=True)):
            model = str(model or "unknown").replace("/", "_")
            fake = str(label) == "fake"
            source = f"OpenFake:{model}" if fake else f"OpenFake-real:{model}"
            candidates.append(
                {
                    "locator": f"{path.name}:{index}",
                    "parquet": path,
                    "index": index,
                    "label": int(fake),
                    "source": source,
                    "model": model,
                    "role": role,
                }
            )
    return candidates


def materialize_openfake(
    paths: list[Path],
    out: Path,
    role: str,
    fake_cap: int,
    real_cap: int,
    seed: int,
    workers: int,
) -> list[dict[str, Any]]:
    candidates = _openfake_candidates(paths, role)
    sources = {str(row["source"]) for row in candidates}
    caps = {source: (fake_cap if source.startswith("OpenFake:") else real_cap) for source in sources}
    selected = _select(candidates, caps, seed)
    wanted: dict[Path, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        wanted[Path(row["parquet"])][int(row["index"])] = row

    def process(payload: tuple[bytes, dict[str, Any], Path, int]) -> dict[str, Any] | None:
        data, item, parquet_path, index = payload
        try:
            encoded = _web_tier(data) if role == "train" else data
            target = out / role / ("fake" if item["label"] else "real") / item["model"]
            target.mkdir(parents=True, exist_ok=True)
            image_path = target / f"{parquet_path.stem}-{index:04d}.jpg"
            image_path.write_bytes(encoded)
            return _metadata_row(
                image_path,
                label=int(item["label"]),
                dataset=f"OpenFake-core-{role}",
                source=str(item["source"]),
                group_id=f"OpenFake-core-{role}:{item['source']}",
                role=role,
                license_name="CC-BY-SA-4.0 / source-specific non-commercial terms",
            )
        except Exception as exc:
            print(f"warning: failed {parquet_path.name}:{index}: {type(exc).__name__}: {exc}", flush=True)
            return None

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for parquet_path in sorted(wanted):
            parquet = pq.ParquetFile(parquet_path)
            offset = 0
            for group_index in range(parquet.metadata.num_row_groups):
                count = parquet.metadata.row_group(group_index).num_rows
                indexes = [index for index in wanted[parquet_path] if offset <= index < offset + count]
                if indexes:
                    table = parquet.read_row_group(group_index, columns=["image"])
                    payloads = []
                    for index in indexes:
                        data = table["image"][index - offset].as_py()["bytes"]
                        if data:
                            payloads.append((data, wanted[parquet_path][index], parquet_path, index))
                    rows.extend(row for row in pool.map(process, payloads) if row is not None)
                offset += count
            print(f"OpenFake {role}: {parquet_path.name} complete ({len(rows)} rows total)", flush=True)
    return rows


def materialize_zip_images(
    zip_path: Path,
    out: Path,
    *,
    dataset: str,
    label: int,
    role: str,
    license_name: str,
    workers: int,
) -> list[dict[str, Any]]:
    def process(payload: tuple[str, bytes]) -> dict[str, Any] | None:
        name, data = payload
        parts = [part for part in Path(name).parts[:-1] if part not in {".", ".."}]
        source_name = parts[-1] if parts else dataset
        source = f"{dataset}:{source_name}"
        target_dir = out / dataset / source_name
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(name).suffix.lower()
        target = target_dir / f"{hashlib.sha256(data).hexdigest()[:24]}{suffix}"
        if not target.exists():
            target.write_bytes(data)
        try:
            return _metadata_row(
                target,
                label=label,
                dataset=dataset,
                source=source,
                group_id=source,
                role=role,
                license_name=license_name,
            )
        except Exception as exc:
            print(f"warning: failed {name}: {type(exc).__name__}: {exc}", flush=True)
            return None

    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(name for name in archive.namelist() if Path(name).suffix.lower() in IMAGE_SUFFIXES)
        batch_size = max(8, workers * 8)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for start in range(0, len(names), batch_size):
                batch_names = names[start:start + batch_size]
                payloads = [(name, archive.read(name)) for name in batch_names]
                rows.extend(row for row in pool.map(process, payloads) if row is not None)
                if len(rows) // 2000 != (len(rows) - len(batch_names)) // 2000:
                    print(f"{dataset}: {len(rows)}/{len(names)}", flush=True)
    return rows


def materialize_coco(zip_path: Path, out: Path, cap: int, seed: int, workers: int) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(name for name in archive.namelist() if Path(name).suffix.lower() == ".jpg")
        names.sort(key=lambda name: _priority(seed, "COCO-train2017", name))
        names = names[:cap]
        target_dir = out / "COCO-train2017"
        target_dir.mkdir(parents=True, exist_ok=True)

        def process(payload: tuple[str, bytes]) -> dict[str, Any]:
            name, data = payload
            target = target_dir / Path(name).name
            if not target.exists():
                target.write_bytes(data)
            return _metadata_row(
                target,
                label=0,
                dataset="COCO-train2017",
                source="COCO-train2017",
                group_id="COCO-train2017",
                role="train",
                license_name="COCO image source licenses",
            )
        rows = []
        batch_size = max(8, workers * 8)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for start in range(0, len(names), batch_size):
                batch_names = names[start:start + batch_size]
                payloads = [(name, archive.read(name)) for name in batch_names]
                rows.extend(pool.map(process, payloads))
                if len(rows) // 2000 != (len(rows) - len(batch_names)) // 2000:
                    print(f"COCO: {len(rows)}/{len(names)}", flush=True)
    return rows


def materialize_wikiart(paths: list[Path], out: Path, cap: int, seed: int) -> list[dict[str, Any]]:
    candidates = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        for index in range(parquet.metadata.num_rows):
            candidates.append({"locator": f"{path.name}:{index}", "source": "WikiArt", "parquet": path, "index": index})
    selected = _select(candidates, {"WikiArt": cap}, seed)
    wanted: dict[Path, set[int]] = defaultdict(set)
    for row in selected:
        wanted[Path(row["parquet"])].add(int(row["index"]))
    rows = []
    target_dir = out / "WikiArt"
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(wanted):
        parquet = pq.ParquetFile(path)
        offset = 0
        for group_index in range(parquet.metadata.num_row_groups):
            count = parquet.metadata.row_group(group_index).num_rows
            indexes = [index for index in wanted[path] if offset <= index < offset + count]
            if indexes:
                table = parquet.read_row_group(group_index, columns=["image"])
                for index in indexes:
                    image = table["image"][index - offset].as_py()
                    data = image.get("bytes") if image else None
                    if not data:
                        continue
                    try:
                        suffix = Path(image.get("path") or "image.jpg").suffix.lower()
                        if suffix not in IMAGE_SUFFIXES:
                            suffix = ".jpg"
                        target = target_dir / f"{path.stem}-{index:06d}{suffix}"
                        target.write_bytes(data)
                        rows.append(
                            _metadata_row(
                                target,
                                label=0,
                                dataset="WikiArt",
                                source="WikiArt",
                                group_id="WikiArt",
                                role="train",
                                license_name="WikiArt source artwork terms",
                            )
                        )
                    except Exception as exc:
                        print(f"warning: failed WikiArt {path.name}:{index}: {type(exc).__name__}: {exc}", flush=True)
            offset += count
    return rows


def _deduplicate(rows: list[dict[str, Any]], excluded: set[str]) -> tuple[list[dict[str, Any]], int]:
    seen = set(excluded)
    result = []
    dropped = 0
    for row in sorted(rows, key=lambda item: (str(item["role"]), str(item["sample_id"]))):
        digest = str(row["content_sha256"])
        if digest in seen:
            dropped += 1
            continue
        seen.add(digest)
        result.append(row)
    return result, dropped


def _drop_cross_role_phash(rows: list[dict[str, Any]], distance: int = 4) -> tuple[list[dict[str, Any]], int]:
    from aiblink_validation.audit import _near_cross_role

    dropped = 0
    while True:
        pairs = _near_cross_role(rows, distance, cap=10_000)
        blocked = {
            sample_id
            for pair in pairs
            for sample_id, role in (
                (str(pair["left"]), str(pair["left_role"])),
                (str(pair["right"]), str(pair["right_role"])),
            )
            if role == "calibration"
        }
        if not blocked:
            if pairs:
                raise ValueError("cross-role perceptual neighbors remain without a calibration row to drop")
            return rows, dropped
        rows = [row for row in rows if str(row["sample_id"]) not in blocked]
        dropped += len(blocked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=323)
    parser.add_argument("--openfake-train-fake-cap", type=int, default=600)
    parser.add_argument("--openfake-train-real-cap", type=int, default=12000)
    parser.add_argument("--openfake-validation-fake-cap", type=int, default=80)
    parser.add_argument("--openfake-validation-real-cap", type=int, default=1500)
    parser.add_argument("--coco-cap", type=int, default=15000)
    parser.add_argument("--wikiart-cap", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    required = [args.downloads / "synthbuster.zip", args.downloads / "train2017.zip"]
    for path in required:
        if not path.exists():
            raise SystemExit(f"missing required download {path}")
    train_parquets = sorted((args.downloads / "openfake_train").glob("*.parquet"))
    validation_parquets = sorted((args.downloads / "openfake_validation").glob("*.parquet"))
    wikiart_parquets = sorted((args.downloads / "wikiart").glob("*.parquet"))
    if not train_parquets or not validation_parquets or not wikiart_parquets:
        raise SystemExit("OpenFake train/validation and WikiArt parquet downloads are required")

    pilot_rows = [dict(row) for row in read_manifest(args.pilot_manifest) if row["role"] == "train"]
    rows = list(pilot_rows)
    rows += materialize_openfake(
        train_parquets, args.materialized / "openfake", "train",
        args.openfake_train_fake_cap, args.openfake_train_real_cap, args.seed, args.workers,
    )
    rows += materialize_openfake(
        validation_parquets, args.materialized / "openfake", "calibration",
        args.openfake_validation_fake_cap, args.openfake_validation_real_cap, args.seed + 1,
        args.workers,
    )
    rows += materialize_zip_images(
        args.downloads / "synthbuster.zip", args.materialized,
        dataset="SynthBuster", label=1, role="train", license_name="SynthBuster dataset terms",
        workers=args.workers,
    )
    rows += materialize_coco(
        args.downloads / "train2017.zip", args.materialized, args.coco_cap, args.seed, args.workers
    )
    rows += materialize_wikiart(wikiart_parquets, args.materialized, args.wikiart_cap, args.seed)

    excluded = {str(row["content_sha256"]) for row in read_manifest(args.exclude_manifest)}
    rows, dropped = _deduplicate(rows, excluded)
    rows, near_dropped = _drop_cross_role_phash(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.out, rows)
    counts = Counter((str(row["role"]), int(row["label"]), str(row["source"])) for row in rows)
    metadata = {
        "protocol": "aiblink-full-training-manifest/0.1.0",
        "seed": args.seed,
        "manifest_sha256": sha256_file(args.out),
        "rows": len(rows),
        "role_counts": Counter(str(row["role"]) for row in rows),
        "class_counts": Counter(f"{row['role']}:{row['label']}" for row in rows),
        "source_counts": {f"{role}:{label}:{source}": count for (role, label, source), count in sorted(counts.items())},
        "input_files": {
            "openfake_train": [path.name for path in train_parquets],
            "openfake_validation": [path.name for path in validation_parquets],
            "wikiart": [path.name for path in wikiart_parquets],
        },
        "excluded_exact_hashes": len(excluded),
        "deduplicated_rows": dropped,
        "cross_role_phash_rows_dropped_from_calibration": near_dropped,
        "test_role_opened": False,
    }
    (args.out.parent / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
