#!/usr/bin/env python3
"""Select a balanced frontier subset, materialize it, and build a leak-audited training manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from aiblink_validation.audit import audit_manifest
from aiblink_validation.hashing import hamming_hex
from aiblink_validation.io import read_manifest, sha256_file, write_manifest


PROTECTED_MANIFESTS = (
    "/data/ai_blink/full/champion/frozen_manifest.csv",
    "/data/ai_blink/recent_openrouter/manifest.csv",
)


def phash_bands(value: str, distance: int = 4) -> list[tuple[int, int]]:
    number = int(value, 16)
    bands = distance + 1
    widths = [64 // bands + (index < 64 % bands) for index in range(bands)]
    result = []
    offset = 0
    for band, width in enumerate(widths):
        shift = 64 - offset - width
        result.append((band, (number >> shift) & ((1 << width) - 1)))
        offset += width
    return result


def protected_index(paths: list[Path]) -> tuple[set[str], dict[tuple[int, int], list[dict[str, Any]]], int]:
    exact = set()
    bands: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    for path in paths:
        if not path.is_file():
            continue
        for row in read_manifest(path):
            # OpenRouter is deliberately kept as a post-training diagnostic; for
            # other manifests, only calibration/test images are protected.
            protect = "recent_openrouter" in str(path) or row["role"] != "train"
            key = (row["content_sha256"], row["phash"])
            if not protect or key in seen:
                continue
            seen.add(key)
            exact.add(row["content_sha256"])
            for band in phash_bands(row["phash"]):
                bands[band].append(row)
    return exact, bands, len(seen)


def near_protected(phash: str, digest: str, exact: set[str], bands: dict[tuple[int, int], list[dict[str, Any]]]) -> tuple[bool, str]:
    if digest in exact:
        return True, "exact"
    candidates: dict[str, dict[str, Any]] = {}
    for band in phash_bands(phash):
        for row in bands.get(band, ()):
            candidates[row["sample_id"]] = row
    for row in candidates.values():
        if hamming_hex(phash, row["phash"]) <= 4:
            return True, "phash<=4"
    return False, ""


def select_ids(shards: list[Path], cap: int, recent_holdout_mod: int) -> tuple[set[str], set[str], dict[str, int]]:
    heaps: dict[str, list[tuple[int, str]]] = defaultdict(list)
    recent_holdout = set()
    counts = Counter()
    for shard in shards:
        table = pq.read_table(shard, columns=["sample_id", "generator", "source_dataset"])
        for sample_id, generator, source_dataset in zip(
            table["sample_id"].to_pylist(),
            table["generator"].to_pylist(),
            table["source_dataset"].to_pylist(),
            strict=True,
        ):
            counts[generator] += 1
            if source_dataset == "recent-image-models-synthetic-1k" and int(sample_id[:8], 16) % recent_holdout_mod == 0:
                recent_holdout.add(sample_id)
                continue
            rank = int.from_bytes(hashlib.sha256(f"323\0{generator}\0{sample_id}".encode()).digest()[:8], "big")
            entry = (-rank, sample_id)
            heap = heaps[generator]
            if len(heap) < cap:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)
    selected = {sample_id for heap in heaps.values() for _, sample_id in heap}
    return selected, recent_holdout, dict(counts)


def image_extension(path: str, mime: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in {"jpg", "jpeg", "png", "webp", "tif", "tiff"}:
        return "jpg" if suffix == "jpeg" else suffix
    return {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(mime, "img")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, default=Path("/data/ai_blink/full/manifest.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap-per-generator", type=int, default=12_000)
    parser.add_argument("--recent-holdout-mod", type=int, default=5)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        if not args.replace:
            raise RuntimeError(f"output exists: {args.output}; pass --replace")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    images_root = args.output / "materialized"
    diagnostics_root = args.output / "diagnostic_recent_holdout"
    shards = sorted((args.stage / "data").glob("*.parquet"))
    if not shards or not (args.stage / "READY").is_file():
        raise RuntimeError("public stage is not complete")

    selected, recent_holdout, available_by_generator = select_ids(
        shards, args.cap_per_generator, args.recent_holdout_mod
    )
    protected_paths = [Path(value) for value in PROTECTED_MANIFESTS]
    protected_exact, protected_bands, protected_count = protected_index(protected_paths)
    base_rows = read_manifest(args.base_manifest)
    existing_hashes = {row["content_sha256"] for row in base_rows}
    selected_rows = []
    diagnostic_rows = []
    filtered = Counter()

    for shard in shards:
        table = pq.read_table(shard)
        columns = {name: table[name].to_pylist() for name in table.column_names if name != "image"}
        images = table["image"].to_pylist()
        for index, sample_id in enumerate(columns["sample_id"]):
            is_diagnostic = sample_id in recent_holdout
            if sample_id not in selected and not is_diagnostic:
                continue
            digest = columns["content_sha256"][index]
            overlap, reason = near_protected(
                columns["phash"][index], digest, protected_exact, protected_bands
            )
            if overlap:
                filtered[reason] += 1
                continue
            if digest in existing_hashes:
                filtered["existing_train_exact"] += 1
                continue
            value = images[index]
            data = value.get("bytes")
            if not data or hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"embedded-byte hash mismatch in {shard}:{index}")
            generator = columns["generator"][index]
            extension = image_extension(value.get("path") or "", columns["mime"][index])
            root = diagnostics_root if is_diagnostic else images_root
            target = root / generator.replace("/", "_").replace(" ", "-") / f"{digest}.{extension}"
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(data)
            os.replace(temporary, target)
            row = {
                "sample_id": f"frontier-{digest[:24]}",
                "path": str(target.resolve()),
                "label": 1,
                "dataset": "FrontierSyntheticImages2026",
                "source": f"FrontierSynthetic:{generator}",
                "group_id": f"FrontierSynthetic:{columns['prompt_sha256'][index]}",
                "role": "train" if not is_diagnostic else "test",
                "content_sha256": digest,
                "phash": columns["phash"][index],
                "width": columns["width"][index],
                "height": columns["height"][index],
                "mime": columns["mime"][index],
                "license": columns["source_license"][index],
                "source_repo": columns["source_repo"][index],
                "source_revision": columns["source_revision"][index],
            }
            if is_diagnostic:
                diagnostic_rows.append(row)
            else:
                selected_rows.append(row)
                existing_hashes.add(digest)
        print(f"materialized through {shard.name}: train={len(selected_rows)} diagnostic={len(diagnostic_rows)}", flush=True)

    combined = base_rows + selected_rows
    manifest = args.output / "manifest.csv"
    write_manifest(manifest, combined)
    diagnostic_manifest = args.output / "recent_holdout_manifest.csv"
    if diagnostic_rows:
        write_manifest(diagnostic_manifest, diagnostic_rows)
    audit = audit_manifest(combined, phash_distance=4, check_files=True)
    if not audit["valid"]:
        raise RuntimeError(f"combined manifest audit failed: {audit['issues'][:3]}")
    report = {
        "protocol": "frontier-training-preparation/1.0",
        "public_stage_audit_sha256": (args.stage / "READY").read_text().strip(),
        "base_manifest": str(args.base_manifest.resolve()),
        "base_manifest_sha256": sha256_file(args.base_manifest),
        "combined_manifest": str(manifest.resolve()),
        "combined_manifest_sha256": sha256_file(manifest),
        "base_rows": len(base_rows),
        "new_train_rows": len(selected_rows),
        "diagnostic_recent_holdout_rows": len(diagnostic_rows),
        "combined_rows": len(combined),
        "cap_per_generator": args.cap_per_generator,
        "available_by_generator": dict(sorted(available_by_generator.items())),
        "selected_by_generator": dict(sorted(Counter(row["source"] for row in selected_rows).items())),
        "protected_images": protected_count,
        "protected_manifests": [str(path) for path in protected_paths],
        "filtered": dict(filtered),
        "audit": audit,
    }
    (args.output / "preparation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
