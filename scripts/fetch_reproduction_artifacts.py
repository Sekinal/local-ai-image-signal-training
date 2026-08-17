#!/usr/bin/env python3
"""Fetch immutable release artifacts and enforce their recorded SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_huggingface(item: dict[str, Any], target: Path) -> None:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=item["repo"],
        filename=item["file"],
        revision=item["revision"],
        repo_type=item["repo_type"],
        token=os.environ.get("HF_TOKEN") or None,
    )
    temporary = target.with_suffix(target.suffix + ".part")
    shutil.copyfile(cached, temporary)
    os.replace(temporary, target)


def fetch_url(item: dict[str, Any], target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(item["url"], headers={"User-Agent": "aiblink-reproduction/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
    os.replace(temporary, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).parents[1] / "reproduction" / "artifacts.json",
    )
    parser.add_argument("--profiles", default="deployment,data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    profiles = {value.strip() for value in args.profiles.split(",") if value.strip()}
    selected = [item for item in spec["artifacts"] if item["profile"] in profiles]
    if args.list:
        for item in selected:
            print(item["profile"], item["destination"], item["sha256"])
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    for item in selected:
        target = args.output / item["destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == item["bytes"] and sha256(target) == item["sha256"]:
            print(f"verified {target}")
            continue
        if item["kind"] == "huggingface":
            fetch_huggingface(item, target)
        elif item["kind"] == "url":
            fetch_url(item, target)
        else:
            raise ValueError(f"unsupported artifact kind: {item['kind']}")
        actual = sha256(target)
        if target.stat().st_size != item["bytes"] or actual != item["sha256"]:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"artifact verification failed: {item['destination']}")
        print(f"downloaded and verified {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
