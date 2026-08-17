#!/usr/bin/env python3
"""Normalize, deduplicate, and shard the public frontier synthetic corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import itertools
import json
import os
import re
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Image as HFImage, Value
from PIL import Image, ImageOps


QWEN_REPO = "Qwen/Qwen-Image-Bench"
QWEN_REVISION = "d2493deb153b020cf169c7e3f57d15e4dd697038"
RAPIDATA = [
    (
        "flux2",
        "Rapidata/Flux-2-pro_t2i_human_preference",
        "ef2e83f43b41897a8e1a40dd8a5dac209d164c79",
        69,
    ),
    (
        "recraft3",
        "Rapidata/Recraft-v3-24-7-25_t2i_human_preference",
        "d7d47aab05cf8fcb5427e278d8817b9dbd65eafb",
        122,
    ),
    (
        "seedream3",
        "Rapidata/Seedream-3_t2i_human_preference",
        "a9cd84d2a93e82954fe69bd3097b53c48f51a486",
        101,
    ),
]


FEATURES = Features(
    {
        "image": HFImage(decode=False),
        "sample_id": Value("string"),
        "label": Value("int8"),
        "role": Value("string"),
        "generator": Value("string"),
        "prompt": Value("string"),
        "prompt_sha256": Value("string"),
        "group_id": Value("string"),
        "source_dataset": Value("string"),
        "source_repo": Value("string"),
        "source_revision": Value("string"),
        "source_license": Value("string"),
        "original_locator": Value("string"),
        "content_sha256": Value("string"),
        "phash": Value("string"),
        "width": Value("int32"),
        "height": Value("int32"),
        "mime": Value("string"),
        "preference_score": Value("float32"),
        "coherence_score": Value("float32"),
        "alignment_score": Value("float32"),
    }
)

_DCT_SIZE = 32
_DCT_X = np.arange(_DCT_SIZE, dtype=np.float64)
_DCT_K = _DCT_X[:, None]
_DCT = np.cos((np.pi / _DCT_SIZE) * (_DCT_X + 0.5) * _DCT_K)
_DCT[0] *= 1 / np.sqrt(2)
_DCT *= np.sqrt(2 / _DCT_SIZE)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    return cleaned[:120] or "unknown"


def prompt_hash(prompt: str) -> str:
    normalized = " ".join(prompt.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def image_facts(data: bytes) -> tuple[str, int, int, str, str, str]:
    digest = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as source:
        fmt = (source.format or "unknown").lower()
        width, height = source.size
        grayscale = ImageOps.exif_transpose(source).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(grayscale, dtype=np.float64)
    low = (_DCT @ pixels @ _DCT.T)[:8, :8]
    median = np.median(low.reshape(-1)[1:])
    value = 0
    for bit in (low >= median).reshape(-1):
        value = (value << 1) | int(bit)
    phash = f"{value:016x}"
    mime_format = "jpeg" if fmt in {"jpg", "jpeg"} else fmt
    extension = "jpg" if fmt in {"jpg", "jpeg"} else fmt
    return digest, width, height, f"image/{mime_format}", phash, extension


def database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS images (
            content_sha256 TEXT PRIMARY KEY,
            sample_id TEXT NOT NULL,
            generator TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            source_repo TEXT NOT NULL,
            original_locator TEXT NOT NULL,
            phash TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            mime TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS duplicates (
            content_sha256 TEXT NOT NULL,
            kept_generator TEXT NOT NULL,
            duplicate_generator TEXT NOT NULL,
            duplicate_source_repo TEXT NOT NULL,
            duplicate_locator TEXT NOT NULL
        );
        """
    )
    return connection


def normalized_row(
    connection: sqlite3.Connection,
    *,
    data: bytes,
    original_name: str,
    generator: str,
    prompt: str,
    source_dataset: str,
    source_repo: str,
    source_revision: str,
    source_license: str,
    locator: str,
    preference_score: float | None = None,
    coherence_score: float | None = None,
    alignment_score: float | None = None,
) -> dict[str, Any] | None:
    digest, width, height, mime, phash, extension = image_facts(data)
    existing = connection.execute(
        "SELECT generator FROM images WHERE content_sha256 = ?", (digest,)
    ).fetchone()
    if existing:
        connection.execute(
            "INSERT INTO duplicates VALUES (?, ?, ?, ?, ?)",
            (digest, existing[0], generator, source_repo, locator),
        )
        return None
    sample_id = digest
    connection.execute(
        "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (digest, sample_id, generator, source_dataset, source_repo, locator, phash, width, height, mime),
    )
    p_hash = prompt_hash(prompt)
    return {
        "image": {"bytes": data, "path": f"{slug(generator)}/{digest}.{extension}"},
        "sample_id": sample_id,
        "label": 1,
        "role": "train",
        "generator": generator,
        "prompt": prompt,
        "prompt_sha256": p_hash,
        "group_id": f"prompt:{p_hash}",
        "source_dataset": source_dataset,
        "source_repo": source_repo,
        "source_revision": source_revision,
        "source_license": source_license,
        "original_locator": locator or original_name,
        "content_sha256": digest,
        "phash": phash,
        "width": width,
        "height": height,
        "mime": mime,
        "preference_score": preference_score,
        "coherence_score": coherence_score,
        "alignment_score": alignment_score,
    }


def write_shard(output: Path, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = output / "data" / f"train-{name}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".parquet.tmp")
    table = pa.Table.from_pylist(rows, schema=FEATURES.arrow_schema)
    pq.write_table(table, temporary, compression="zstd", compression_level=3, row_group_size=128)
    os.replace(temporary, target)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "file": str(target.relative_to(output)),
        "rows": len(rows),
        "bytes": target.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def qwen_groups(root: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    metadata = root / "qwen_image_bench_hf_v0518.jsonl"
    prompt_rows = [json.loads(line) for line in metadata.open(encoding="utf-8")]
    model_keys = sorted(
        key for key, value in prompt_rows[0].items() if isinstance(value, str) and value.startswith("images/")
    )
    if len(prompt_rows) != 1000 or len(model_keys) != 18:
        raise RuntimeError(f"unexpected Qwen benchmark shape: {len(prompt_rows)} prompts, {len(model_keys)} models")
    for model in model_keys:
        rows = []
        for record in prompt_rows:
            relative = record[model]
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "data": path.read_bytes(),
                    "original_name": path.name,
                    "generator": model,
                    "prompt": record.get("prompt_en") or record["prompt_cn"],
                    "source_dataset": "Qwen-Image-Bench",
                    "source_repo": QWEN_REPO,
                    "source_revision": QWEN_REVISION,
                    "source_license": "Apache-2.0",
                    "locator": relative,
                }
            )
        yield f"qwen-{slug(model)}", rows


def recent_groups(root: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    ledger = root / "acquisition.jsonl"
    records = [json.loads(line) for line in ledger.open(encoding="utf-8") if line.strip()]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("error"):
            continue
        path = Path(record["path"])
        grouped.setdefault(record["model_id"], []).append(
            {
                "data": path.read_bytes(),
                "original_name": path.name,
                "generator": record["model_id"],
                "prompt": record["prompt"],
                "source_dataset": "recent-image-models-synthetic-1k",
                "source_repo": "Thermostatic/recent-image-models-synthetic-1k",
                "source_revision": record["revision"],
                "source_license": "Apache-2.0",
                "locator": record["job_id"],
            }
        )
    if sum(map(len, grouped.values())) != 1000:
        raise RuntimeError(f"expected 1,000 locally generated records, found {sum(map(len, grouped.values()))}")
    for model in sorted(grouped):
        yield f"recent-{slug(model)}", grouped[model]


def rapidata_groups(base: Path) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    for short, repo, revision, expected_shards in RAPIDATA:
        files = sorted((base / short / "source" / "data").glob("train-*.parquet"))
        if len(files) != expected_shards:
            raise RuntimeError(f"{repo}: expected {expected_shards} shards, found {len(files)}")
        for path in files:
            table = pq.read_table(path)
            rows = []
            for index in range(table.num_rows):
                prompt = table["prompt"][index].as_py()
                for slot in (1, 2):
                    image_value = table[f"image{slot}"][index].as_py()
                    data = image_value.get("bytes")
                    if not data:
                        raise RuntimeError(f"missing embedded image: {path}:{index}:image{slot}")
                    rows.append(
                        {
                            "data": data,
                            "original_name": image_value.get("path") or f"image{slot}",
                            "generator": table[f"model{slot}"][index].as_py(),
                            "prompt": prompt,
                            "source_dataset": repo.rsplit("/", 1)[1],
                            "source_repo": repo,
                            "source_revision": revision,
                            "source_license": "CDLA-Permissive-2.0",
                            "locator": f"{path.name}:{index}:image{slot}",
                            "preference_score": table[f"weighted_results_image{slot}_preference"][index].as_py(),
                            "coherence_score": table[f"weighted_results_image{slot}_coherence"][index].as_py(),
                            "alignment_score": table[f"weighted_results_image{slot}_alignment"][index].as_py(),
                        }
                    )
            yield f"rapidata-{short}-{path.stem.removeprefix('train-')}", rows


def card(audit: dict[str, Any]) -> str:
    count = audit["unique_images"]
    size_category = "100K<n<1M" if count >= 100_000 else "10K<n<100K"
    return f"""---
license: other
task_categories:
- image-classification
- text-to-image
language:
- en
- zh
tags:
- synthetic
- ai-generated
- image-forensics
- human-preferences
size_categories:
- {size_category}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
---

# Frontier Synthetic Images — Deduplicated Research Corpus

This is a training-only corpus of **{count:,} exact-deduplicated AI-generated images** from recent and frontier generators. It normalizes four provenance-pinned sources into one row-per-image schema for image-forensics research. It is not an evaluation benchmark and should not be used to report detector accuracy after training on it.

## Sources and licensing

- `Qwen/Qwen-Image-Bench` at `{QWEN_REVISION}`: upstream repository metadata declares Apache-2.0.
- Three Rapidata preference datasets at the exact revisions recorded per row: upstream repository metadata declares CDLA-Permissive-2.0.
- `Thermostatic/recent-image-models-synthetic-1k`: locally generated from pinned, public Apache-2.0 checkpoints; per-row checkpoint provenance is retained.

The combined dataset therefore uses `license: other`; each row carries `source_license`, `source_repo`, and `source_revision`. Upstream terms continue to apply to each component. Compilation metadata and normalization code may be reused under CC-BY-4.0. No participant detector weights or Sieve-generated content are included.

## Safety and leakage policy

All rows are labeled synthetic (`label=1`) and `role=train`. Exact SHA-256 duplicates were removed across all sources. `prompt_sha256` and `group_id` let users keep all renders of a prompt together. Before detector training, users should perceptually compare this corpus against their own locked validation/test sets and keep every match out of training. Public benchmarks can overlap other public evaluations, so do not assume independence.

## Fields

The dataset includes the image, generator, prompt, prompt grouping, dimensions, MIME type, exact and perceptual hashes, immutable source provenance, and (when available) aggregate Rapidata preference/coherence/alignment scores. Detailed individual voting records were intentionally omitted.

Build audit SHA-256: `{audit['audit_sha256']}`. Decode failures: {audit['decode_failures']}. Exact duplicate occurrences removed: {audit['duplicate_occurrences']}.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisitions", type=Path, default=Path("/data/ai_blink/acquisitions"))
    parser.add_argument("--recent", type=Path, default=Path("/data/ai_blink/recent_huggingface"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.replace:
            raise RuntimeError(f"output already exists: {args.output}; pass --replace")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    connection = database(args.output / "dedup.sqlite")
    shard_reports = []
    decode_failures = []
    started = time.time()

    groups: Iterable[tuple[str, list[dict[str, Any]]]] = itertools.chain(
        qwen_groups(args.acquisitions / "qwen_image_bench" / "source"),
        recent_groups(args.recent),
    )
    group_iterators = [groups, rapidata_groups(args.acquisitions / "rapidata")]
    for iterator in group_iterators:
        for name, candidates in iterator:
            rows = []
            for candidate in candidates:
                try:
                    row = normalized_row(connection, **candidate)
                except Exception as exc:
                    decode_failures.append({"locator": candidate["locator"], "error": f"{type(exc).__name__}: {exc}"})
                    continue
                if row is not None:
                    rows.append(row)
            connection.commit()
            if rows:
                report = write_shard(args.output, name, rows)
                shard_reports.append(report)
                print(
                    f"wrote {report['file']}: {report['rows']} rows, {report['bytes'] / 2**20:.1f} MiB; "
                    f"unique={connection.execute('SELECT COUNT(*) FROM images').fetchone()[0]}",
                    flush=True,
                )

    unique_images = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    duplicate_occurrences = connection.execute("SELECT COUNT(*) FROM duplicates").fetchone()[0]
    by_generator = dict(connection.execute("SELECT generator, COUNT(*) FROM images GROUP BY generator ORDER BY generator"))
    by_source = dict(connection.execute("SELECT source_repo, COUNT(*) FROM images GROUP BY source_repo ORDER BY source_repo"))
    with gzip.open(args.output / "duplicate_provenance.jsonl.gz", "wt", encoding="utf-8") as handle:
        for values in connection.execute("SELECT * FROM duplicates ORDER BY content_sha256, duplicate_source_repo, duplicate_locator"):
            handle.write(json.dumps(dict(zip(["content_sha256", "kept_generator", "duplicate_generator", "duplicate_source_repo", "duplicate_locator"], values)), sort_keys=True) + "\n")
    connection.close()
    audit = {
        "protocol": "frontier-synthetic-normalization/1.0",
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "unique_images": unique_images,
        "duplicate_occurrences": duplicate_occurrences,
        "decode_failures": len(decode_failures),
        "by_generator": by_generator,
        "by_source_repo": by_source,
        "shards": shard_reports,
        "source_revisions": {QWEN_REPO: QWEN_REVISION, **{repo: rev for _, repo, rev, _ in RAPIDATA}},
    }
    canonical = json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
    audit["audit_sha256"] = hashlib.sha256(canonical).hexdigest()
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (args.output / "decode_failures.json").write_text(json.dumps(decode_failures, indent=2, sort_keys=True) + "\n")
    (args.output / "README.md").write_text(card(audit))
    (args.output / "READY").write_text(audit["audit_sha256"] + "\n")
    print(json.dumps({"status": "ready", **audit}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
