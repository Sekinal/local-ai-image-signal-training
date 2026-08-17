#!/usr/bin/env python3
"""Resume pinned Hugging Face dataset files without Hub tree API calls."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import time
import urllib.parse
from pathlib import Path

import requests


def download_one(repo: str, revision: str, relative: str, destination: Path, retries: int) -> tuple[str, int, str]:
    target = destination / relative
    if target.is_file() and target.stat().st_size > 0:
        return relative, target.stat().st_size, "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    quoted = urllib.parse.quote(relative, safe="/")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quoted}?download=true"
    for attempt in range(retries):
        try:
            existing = partial.stat().st_size if partial.exists() else 0
            headers = {"Connection": "close", "User-Agent": "ai-blink-dataset-acquisition/1.0"}
            token = os.environ.get("HF_TOKEN", "")
            if token:
                headers["Authorization"] = f"Bearer {token}"
            if existing:
                headers["Range"] = f"bytes={existing}-"
            # Fresh connections avoid accumulating half-closed sockets when the
            # CDN rotates hosts across thousands of small files.
            with requests.get(url, headers=headers, stream=True, timeout=(20, 45), allow_redirects=True) as response:
                if existing and response.status_code == 200:
                    existing = 0
                response.raise_for_status()
                mode = "ab" if existing and response.status_code == 206 else "wb"
                with partial.open(mode) as handle:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(partial, target)
            return relative, target.stat().st_size, "downloaded"
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(60, (2**attempt) + random.random()))
    raise AssertionError("unreachable")


def qwen_paths(metadata: Path) -> list[str]:
    paths: set[str] = set()
    with metadata.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            paths.update(value for value in row.values() if isinstance(value, str) and value.startswith("images/"))
    if len(paths) != 18_000:
        raise RuntimeError(f"expected 18,000 Qwen benchmark image paths, found {len(paths)}")
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qwen-metadata", type=Path)
    group.add_argument("--parquet-shards", type=int)
    args = parser.parse_args()

    if args.qwen_metadata:
        paths = qwen_paths(args.qwen_metadata)
    else:
        assert args.parquet_shards is not None
        paths = [
            f"data/train-{index:05d}-of-{args.parquet_shards:05d}.parquet"
            for index in range(args.parquet_shards)
        ]
        paths.insert(0, "README.md")

    started = time.time()
    downloaded = existing = bytes_total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, args.repo, args.revision, path, args.destination, args.retries): path
            for path in paths
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            relative, size, status = future.result()
            bytes_total += size
            downloaded += status == "downloaded"
            existing += status == "existing"
            if completed % 25 == 0 or completed == len(paths):
                elapsed = max(time.time() - started, 0.001)
                print(
                    f"{completed}/{len(paths)} files; {bytes_total / 2**30:.2f} GiB; "
                    f"{bytes_total / elapsed / 2**20:.1f} MiB/s; last={relative}",
                    flush=True,
                )
    report = {
        "repo": args.repo,
        "revision": args.revision,
        "files": len(paths),
        "downloaded": downloaded,
        "existing": existing,
        "bytes": bytes_total,
        "completed_at_unix": time.time(),
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
