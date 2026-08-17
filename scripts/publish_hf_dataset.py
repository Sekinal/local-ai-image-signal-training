#!/usr/bin/env python3
"""Audit, stage, publish, and verify the recent-model synthetic image dataset."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi
from PIL import Image


EXPECTED_MODELS = {
    "sana-sprint-1.6b": "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "z-image-turbo": "Tongyi-MAI/Z-Image-Turbo",
    "qwen-image-2512": "Qwen/Qwen-Image-2512",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def read_successes(ledger: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON at {ledger}:{line_number}: {exc}") from exc
        if not record.get("error"):
            records.append(record)
    return records


def audit(source: Path, expected_per_model: int) -> tuple[list[dict], dict]:
    ledger = source / "acquisition.jsonl"
    if not ledger.is_file():
        raise RuntimeError(f"missing ledger: {ledger}")
    records = read_successes(ledger)
    expected_total = expected_per_model * len(EXPECTED_MODELS)
    if len(records) != expected_total:
        raise RuntimeError(f"expected {expected_total} successful records, found {len(records)}")

    by_slug = collections.Counter(record["model_slug"] for record in records)
    expected_counts = {slug: expected_per_model for slug in EXPECTED_MODELS}
    if dict(by_slug) != expected_counts:
        raise RuntimeError(f"model counts mismatch: expected {expected_counts}, found {dict(by_slug)}")

    job_ids = [record["job_id"] for record in records]
    if len(set(job_ids)) != len(job_ids):
        raise RuntimeError("duplicate job_id values detected")
    digests = [record["content_sha256"] for record in records]
    if len(set(digests)) != len(digests):
        raise RuntimeError("duplicate exact image hashes detected")

    revisions: dict[str, set[str]] = collections.defaultdict(set)
    dimensions = collections.Counter()
    for index, record in enumerate(records, 1):
        slug = record["model_slug"]
        if slug not in EXPECTED_MODELS or record["model_id"] != EXPECTED_MODELS[slug]:
            raise RuntimeError(f"unexpected generator in job {record['job_id']}: {record.get('model_id')}")
        if record.get("label") != 1 or record.get("role") != "train":
            raise RuntimeError(f"invalid label/role in job {record['job_id']}")
        if record.get("license") != "Apache-2.0":
            raise RuntimeError(f"unexpected checkpoint license in job {record['job_id']}")
        path = Path(record["path"]).resolve()
        try:
            path.relative_to(source.resolve())
        except ValueError as exc:
            raise RuntimeError(f"image escapes source directory: {path}") from exc
        if not path.is_file():
            raise RuntimeError(f"missing image: {path}")
        actual_digest = sha256_file(path)
        if actual_digest != record["content_sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {path}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (record["width"], record["height"]):
                raise RuntimeError(f"image metadata mismatch: {path}")
        revisions[slug].add(record["revision"])
        dimensions[f"{record['width']}x{record['height']}"] += 1
        if index % 100 == 0:
            print(f"audited {index}/{expected_total}", flush=True)

    if any(len(values) != 1 for values in revisions.values()):
        raise RuntimeError(f"multiple revisions found for a generator: {dict(revisions)}")

    report = {
        "audited_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "successes": len(records),
        "failures": 0,
        "by_model_slug": dict(sorted(by_slug.items())),
        "revisions": {slug: next(iter(values)) for slug, values in sorted(revisions.items())},
        "dimensions": dict(sorted(dimensions.items())),
        "unique_job_ids": len(set(job_ids)),
        "unique_content_sha256": len(set(digests)),
        "ledger_sha256": sha256_file(ledger),
        "policy": "official public ungated Apache-2.0 foundation checkpoints; nonparticipant models",
    }
    return sorted(records, key=lambda item: (item["model_slug"], item["index"])), report


def dataset_card(report: dict) -> str:
    model_lines = []
    for slug, model_id in EXPECTED_MODELS.items():
        model_lines.append(
            f"- `{model_id}` at revision `{report['revisions'][slug]}`: "
            f"{report['by_model_slug'][slug]} images"
        )
    models = "\n".join(model_lines)
    dimensions = ", ".join(f"{key}: {value}" for key, value in report["dimensions"].items())
    return f"""---
license: apache-2.0
task_categories:
- image-classification
language:
- en
tags:
- synthetic
- ai-generated
- image-forensics
- text-to-image
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: train/**
---

# Recent Open Hugging Face Image Generators — Synthetic 1K

This dataset contains 1,000 locally generated synthetic PNG images intended as **training-only positive examples** for AI-image detection research. It is not a balanced detector benchmark and must not be used as validation or test data. Every row records its prompt, deterministic seed, exact generator revision, inference settings, dimensions, and content hashes.

## Generators

{models}

All four pinned checkpoints were checked as official, public, ungated, Apache-2.0 foundation-model releases. Participant-submitted detector or generator artifacts were excluded. In particular, no Sieve model or output is included.

## Composition and fields

- 250 images per generator; 1,000 total.
- Dimensions: {dimensions}.
- `file_name` points to the PNG decoded by the ImageFolder loader.
- `label=1` denotes synthetic/AI-generated content; every row has `role=train`.
- Provenance fields include `model_id`, `revision`, `model_created_at`, `prompt`, `seed`, `num_inference_steps`, `precision`, and generation dimensions.
- Integrity fields include `job_id`, SHA-256, perceptual hash, observed width/height, and creation timestamp.

## Generation and verification

Images were generated locally through Diffusers from pinned Hugging Face Hub revisions with a fixed base seed and a varied prompt/aspect-ratio matrix. Before upload, all 1,000 files were decoded, dimensions were checked against the ledger, every SHA-256 was recomputed, job IDs and exact hashes were confirmed unique, and the 250-per-model balance was enforced. Audit timestamp: `{report['audited_at_utc']}`. Ledger SHA-256: `{report['ledger_sha256']}`.

## Intended use and limitations

Use these samples only as one component of a broader, provenance-separated training set. They do not represent real photographs, do not provide negative examples, and should not be treated as evidence of performance on unseen generators. Avoid prompt or generator leakage across training and evaluation splits. Model-checkpoint licensing is recorded per row; downstream users remain responsible for assessing rights and compliance for their use case.
"""


def stage_dataset(source: Path, stage: Path, records: list[dict], report: dict) -> None:
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "train").mkdir(parents=True)
    metadata_lines = []
    for record in records:
        slug = record["model_slug"]
        source_image = Path(record["path"]).resolve()
        relative = Path("train") / slug / source_image.name
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_image, target)
        except OSError:
            shutil.copy2(source_image, target)
        parameters = record["generation_parameters"]
        row = {
            "file_name": str(relative.relative_to("train")),
            "sample_id": record["job_id"],
            "job_id": record["job_id"],
            "label": record["label"],
            "role": record["role"],
            "model_slug": slug,
            "model_id": record["model_id"],
            "revision": record["revision"],
            "model_created_at": record["model_created_at"],
            "checkpoint_license": record["license"],
            "prompt": record["prompt"],
            "seed": record["seed"],
            "num_inference_steps": parameters["num_inference_steps"],
            "requested_width": parameters["width"],
            "requested_height": parameters["height"],
            "width": record["width"],
            "height": record["height"],
            "precision": record["precision"],
            "content_sha256": record["content_sha256"],
            "phash": record["phash"],
            "mime": record["mime"],
            "provider": record["provider"],
            "created_at_utc": record["created_at_utc"],
        }
        metadata_lines.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
    atomic_text(stage / "train" / "metadata.jsonl", "\n".join(metadata_lines) + "\n")
    atomic_text(stage / "README.md", dataset_card(report))
    atomic_text(stage / "audit.json", json.dumps(report, indent=2, sort_keys=True) + "\n")


def validate_imagefolder(stage: Path, expected_rows: int) -> dict:
    from datasets import load_dataset

    dataset = load_dataset("imagefolder", data_dir=str(stage), split="train")
    if len(dataset) != expected_rows:
        raise RuntimeError(f"ImageFolder exposed {len(dataset)} rows, expected {expected_rows}")
    required_columns = {
        "image",
        "sample_id",
        "label",
        "role",
        "model_slug",
        "model_id",
        "revision",
        "prompt",
        "seed",
        "content_sha256",
    }
    missing = required_columns - set(dataset.column_names)
    if missing:
        raise RuntimeError(f"ImageFolder metadata columns missing: {sorted(missing)}")
    sample = dataset[0]
    if sample["label"] != 1 or sample["role"] != "train":
        raise RuntimeError("ImageFolder decoded an invalid label/role")
    sample["image"].load()
    return {
        "rows": len(dataset),
        "columns": sorted(dataset.column_names),
        "builder": "imagefolder",
    }


def publish(stage: Path, repo_name: str, token: str) -> dict:
    api = HfApi(token=token)
    identity = api.whoami()
    owner = identity.get("name")
    if not owner:
        raise RuntimeError("authenticated Hugging Face account has no username")
    repo_id = f"{owner}/{repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(stage),
        print_report=False,
    )
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    expected_pngs = sum(1 for _ in stage.glob("train/*/*.png"))
    remote_pngs = sum(name.endswith(".png") for name in files)
    required = {"README.md", "audit.json", "train/metadata.jsonl"}
    if remote_pngs != expected_pngs or not required.issubset(files):
        raise RuntimeError(
            f"remote verification failed: PNGs {remote_pngs}/{expected_pngs}, "
            f"missing {sorted(required - set(files))}"
        )
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "private": True,
        "remote_files": len(files),
        "remote_pngs": remote_pngs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--repo-name", default="recent-image-models-synthetic-1k")
    parser.add_argument("--expected-per-model", type=int, default=250)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    stage = args.stage.resolve()
    records, report = audit(source, args.expected_per_model)
    stage_dataset(source, stage, records, report)
    report["imagefolder_validation"] = validate_imagefolder(stage, len(records))
    atomic_text(stage / "README.md", dataset_card(report))
    atomic_text(stage / "audit.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit": report, "stage": str(stage)}, sort_keys=True), flush=True)
    if args.audit_only:
        return 0

    token = sys.stdin.readline().strip()
    if not token:
        raise RuntimeError("Hugging Face write token must be supplied on standard input")
    result = publish(stage, args.repo_name, token)
    print(json.dumps({"published": result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
