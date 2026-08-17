from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

REQUIRED_MANIFEST_COLUMNS = (
    "sample_id",
    "path",
    "label",
    "dataset",
    "source",
    "group_id",
    "role",
    "content_sha256",
    "phash",
)
VALID_ROLES = {"train", "calibration", "test"}


def sha256_file(path: str | Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    return sha256_file(path)


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path).resolve()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_MANIFEST_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest missing columns: {', '.join(sorted(missing))}")
        rows: list[dict[str, Any]] = []
        for line_number, raw in enumerate(reader, start=2):
            row = {key: (value.strip() if isinstance(value, str) else value) for key, value in raw.items()}
            try:
                row["label"] = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_number}: label must be 0 or 1") from exc
            if row["label"] not in (0, 1):
                raise ValueError(f"line {line_number}: label must be 0 or 1")
            if row["role"] not in VALID_ROLES:
                raise ValueError(f"line {line_number}: invalid role {row['role']!r}")
            image_path = Path(row["path"])
            if not image_path.is_absolute():
                row["path"] = str((manifest_path.parent / image_path).resolve())
            row["_line"] = line_number
            rows.append(row)
    return rows


def write_manifest(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("refusing to write an empty manifest")
    columns = list(REQUIRED_MANIFEST_COLUMNS)
    extras = sorted(set().union(*(row.keys() for row in rows)) - set(columns) - {"_line"})
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + extras, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc


def atomic_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, output)

