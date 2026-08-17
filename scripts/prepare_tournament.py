#!/usr/bin/env python3
"""Build a deterministic, generator-uniform pilot training subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from aiblink_validation.hashing import image_metadata_and_phash


GENERATORS = {
    0: "Real",
    1: "ADM",
    2: "BigGAN",
    3: "GLIDE",
    4: "Midjourney",
    5: "SD14",
    6: "SD15",
    7: "VQDM",
    8: "Wukong",
}
DEFAULT_FAKE_GENERATORS = ("GLIDE", "Midjourney", "SD14", "SD15", "Wukong")
FIELDS = (
    "sample_id", "path", "label", "dataset", "source", "group_id", "role",
    "content_sha256", "phash", "width", "height", "mime", "license",
)


def priority(seed: int, shard: str, row: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{shard}\0{row}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def keep_smallest(heap: list[tuple[int, str, int]], item: tuple[int, str, int], capacity: int) -> None:
    entry = (-item[0], item[1], item[2])
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def sniff_suffix(data: bytes, hint: str | None) -> tuple[str, str]:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg", "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png", "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp", "webp"
    suffix = Path(hint or "image.jpg").suffix.lower()
    return (suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"), "jpeg"


def excluded_hashes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["content_sha256"] for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path)
    parser.add_argument("--per-fake-generator", type=int, default=1500)
    parser.add_argument("--fake-generators", default=",".join(DEFAULT_FAKE_GENERATORS))
    parser.add_argument("--seed", type=int, default=323)
    parser.add_argument("--reserve-fraction", type=float, default=0.10)
    args = parser.parse_args()

    shards = sorted(args.parquet_dir.glob("train-*.parquet"))
    if not shards:
        raise SystemExit(f"no train parquet shards under {args.parquet_dir}")
    fake_generators = tuple(item.strip() for item in args.fake_generators.split(",") if item.strip())
    unknown = sorted(set(fake_generators) - set(GENERATORS.values()))
    if unknown:
        raise SystemExit(f"unknown fake generators: {unknown}")
    quotas = {generator: args.per_fake_generator for generator in fake_generators}
    quotas["Real"] = args.per_fake_generator * len(fake_generators)
    capacities = {key: max(quota, int(round(quota * (1 + args.reserve_fraction)))) for key, quota in quotas.items()}
    heaps: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    observed = Counter()

    for shard in shards:
        table = pq.read_table(shard, columns=["label", "generator"])
        labels = table["label"].to_numpy()
        generators = table["generator"].to_numpy()
        for index, (label, generator_index) in enumerate(zip(labels, generators, strict=True)):
            generator = GENERATORS[int(generator_index)]
            if generator not in quotas:
                continue
            expected = int(generator != "Real")
            if int(label) != expected:
                raise ValueError(f"label mismatch in {shard.name}:{index}")
            observed[generator] += 1
            keep_smallest(
                heaps[generator],
                (priority(args.seed, shard.name, index), shard.name, index),
                capacities[generator],
            )

    for generator, quota in quotas.items():
        if len(heaps[generator]) < quota:
            raise ValueError(f"insufficient {generator}: {len(heaps[generator])} < {quota}")

    selected_by_shard: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for generator, entries in heaps.items():
        for negative_priority, shard_name, index in entries:
            selected_by_shard[shard_name].append((-negative_priority, generator, index))

    args.out.mkdir(parents=True, exist_ok=True)
    image_root = args.out / "images"
    blocked = excluded_hashes(args.exclude_manifest)
    seen = set(blocked)
    candidates: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    shard_map = {path.name: path for path in shards}
    for shard_name, requests in sorted(selected_by_shard.items()):
        parquet = pq.ParquetFile(shard_map[shard_name])
        offsets = []
        running = 0
        for group_index in range(parquet.metadata.num_row_groups):
            count = parquet.metadata.row_group(group_index).num_rows
            offsets.append((running, running + count, group_index))
            running += count
        requests.sort(key=lambda item: item[2])
        for start, end, group_index in offsets:
            group_requests = [item for item in requests if start <= item[2] < end]
            if not group_requests:
                continue
            table = parquet.read_row_group(group_index, columns=["image", "label", "generator"])
            local_indices = pa.array([item[2] - start for item in group_requests], type=pa.int64())
            rows = table.take(local_indices).to_pylist()
            for (rank, generator, global_index), source_row in zip(group_requests, rows, strict=True):
                image = source_row["image"]
                data = image.get("bytes") if image else None
                if not data:
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                suffix, image_format = sniff_suffix(data, image.get("path"))
                target_dir = image_root / ("real" if generator == "Real" else "fake") / generator
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"{digest[:24]}{suffix}"
                target.write_bytes(data)
                width, height, decoded_format, phash = image_metadata_and_phash(target)
                row = {
                    "sample_id": hashlib.sha256(f"tiny-genimage-train\0{digest}".encode()).hexdigest()[:24],
                    "path": str(target.resolve()),
                    "label": int(generator != "Real"),
                    "dataset": "Tiny-GenImage-train",
                    "source": "ImageNet" if generator == "Real" else generator,
                    "group_id": f"tiny-genimage:train:{'real' if generator == 'Real' else 'fake'}:{generator}",
                    "role": "train",
                    "content_sha256": digest,
                    "phash": phash,
                    "width": width,
                    "height": height,
                    "mime": f"image/{decoded_format or image_format}",
                    "license": "GenImage dataset terms",
                }
                candidates[generator].append((rank, row))

    output_rows = []
    for generator, quota in quotas.items():
        rows = sorted(candidates[generator], key=lambda item: item[0])[:quota]
        if len(rows) != quota:
            raise ValueError(f"deduplication left {len(rows)} {generator} rows, expected {quota}")
        output_rows.extend(row for _, row in rows)
    output_rows.sort(key=lambda row: str(row["sample_id"]))
    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    metadata = {
        "protocol": "aiblink-generator-uniform-subset/0.1.0",
        "seed": args.seed,
        "source_shards": [path.name for path in shards],
        "source_shard_count": len(shards),
        "fake_generators": list(fake_generators),
        "heldout_fake_generators": sorted(set(GENERATORS.values()) - {"Real", *fake_generators}),
        "per_fake_generator": args.per_fake_generator,
        "real_count": quotas["Real"],
        "counts": Counter(str(row["source"]) for row in output_rows),
        "excluded_hash_count": len(blocked),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "observed_source_counts": observed,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
