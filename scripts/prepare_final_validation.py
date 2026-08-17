#!/usr/bin/env python3
"""Prepare a calibration-derived development red-team manifest and frozen calibrator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--redteam-manifest", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    args = parser.parse_args()

    ranking = json.loads(args.ranking.read_text())
    selected = ranking["ranking"][0]
    if selected["candidate_id"] != "commfor_384":
        raise ValueError(f"unexpected selected candidate {selected['candidate_id']}")
    atomic_json(
        args.calibrator,
        {
            "protocol": "aiblink-frozen-final-calibrator/0.1.0",
            "selection_role": "calibration",
            "test_role_opened": False,
            "candidate_id": selected["candidate_id"],
            "final": selected["final_calibrator"],
        },
    )

    with args.manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            if row["role"] == "calibration":
                rows.append({**row, "role": "test"})
    if not rows or len({row["label"] for row in rows}) != 2:
        raise ValueError("development red-team manifest requires both calibration classes")
    args.redteam_manifest.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{args.redteam_manifest.name}.", dir=args.redteam_manifest.parent)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, args.redteam_manifest)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps({"redteam_rows": len(rows), "calibrator": selected["final_calibrator"]}, indent=2))


if __name__ == "__main__":
    main()
