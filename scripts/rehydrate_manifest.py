#!/usr/bin/env python3
"""Map a path-free manifest inventory to reacquired image bytes by SHA-256."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from pathlib import Path


IMAGE_SUFFIXES = {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def open_inventory(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def rehydrate(inventory: Path, image_root: Path, output: Path, allow_missing: bool = False) -> dict[str, int]:
    by_hash: dict[str, Path] = {}
    duplicate_bytes = 0
    for path in sorted(candidate for candidate in image_root.rglob("*") if candidate.suffix.lower() in IMAGE_SUFFIXES):
        value = digest(path)
        if value in by_hash:
            duplicate_bytes += 1
            continue
        by_hash[value] = path.resolve()
    with open_inventory(inventory) as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "content_sha256" not in reader.fieldnames or "path" in reader.fieldnames:
            raise ValueError("expected a path-free inventory with content_sha256")
        fields = list(reader.fieldnames)
        fields.insert(1 if fields and fields[0] == "sample_id" else 0, "path")
        rows = list(reader)
    missing = [row["sample_id"] for row in rows if row["content_sha256"] not in by_hash]
    if missing and not allow_missing:
        raise RuntimeError(f"missing {len(missing)} of {len(rows)} inventory images; first={missing[:5]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            path = by_hash.get(row["content_sha256"])
            if path is None:
                continue
            writer.writerow({**row, "path": str(path)})
    return {"inventory_rows": len(rows), "written_rows": len(rows) - len(missing), "missing": len(missing), "duplicate_bytes": duplicate_bytes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    result = rehydrate(args.inventory, args.image_root, args.out, args.allow_missing)
    print(" ".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
