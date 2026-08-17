#!/usr/bin/env python3
"""Wait for a normalized corpus, then publish it publicly and verify it."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    token = os.environ.pop("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN must be supplied only in the process environment")
    api = HfApi(token=token)
    identity = api.whoami()
    owner = identity.get("name")
    if args.repo_id.split("/", 1)[0] != owner:
        raise RuntimeError(f"authenticated as {owner}, refusing target {args.repo_id}")
    api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    print(f"created/verified public dataset repository {args.repo_id}", flush=True)
    while not (args.folder / "READY").is_file():
        shards = list((args.folder / "data").glob("*.parquet")) if (args.folder / "data").exists() else []
        size = sum(path.stat().st_size for path in shards)
        print(f"waiting for READY: {len(shards)} shards, {size / 2**30:.2f} GiB staged", flush=True)
        time.sleep(args.poll_seconds)
    audit = json.loads((args.folder / "audit.json").read_text())
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.folder,
        private=False,
        allow_patterns=["README.md", "audit.json", "decode_failures.json", "duplicate_provenance.jsonl.gz", "data/*.parquet"],
        num_workers=8,
        print_report=True,
        print_report_every=60,
    )
    files = api.list_repo_files(args.repo_id, repo_type="dataset", revision="main")
    remote_shards = sorted(path for path in files if path.startswith("data/") and path.endswith(".parquet"))
    local_shards = sorted(path.name for path in (args.folder / "data").glob("*.parquet"))
    if len(remote_shards) != len(local_shards):
        raise RuntimeError(f"remote shard mismatch: {len(remote_shards)} != {len(local_shards)}")
    result = {
        "repo_id": args.repo_id,
        "url": f"https://huggingface.co/datasets/{args.repo_id}",
        "private": False,
        "remote_files": len(files),
        "remote_shards": len(remote_shards),
        "unique_images": audit["unique_images"],
        "audit_sha256": audit["audit_sha256"],
        "completed_at_unix": time.time(),
    }
    (args.folder / "upload-result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
