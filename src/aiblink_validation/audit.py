from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .hashing import hamming_hex


def _near_cross_role(rows: list[dict[str, Any]], distance: int, cap: int = 200) -> list[dict[str, Any]]:
    """Find cross-role pHash neighbors with exact pigeonhole banding.

    Splitting the 64 bits into d+1 bands guarantees that hashes within Hamming
    distance d share at least one complete band. Candidate pairs are verified.
    """
    bands = distance + 1
    widths = [64 // bands + (i < 64 % bands) for i in range(bands)]
    offsets = []
    cursor = 0
    for width in widths:
        offsets.append((cursor, width))
        cursor += width
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    results: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index, row in enumerate(rows):
        value = int(row["phash"], 16)
        candidates: set[int] = set()
        for band, (offset, width) in enumerate(offsets):
            shift = 64 - offset - width
            key = (band, (value >> shift) & ((1 << width) - 1))
            candidates.update(buckets[key])
        for other_index in candidates:
            other = rows[other_index]
            if other["role"] == row["role"]:
                continue
            pair = (other_index, index)
            if pair in seen:
                continue
            seen.add(pair)
            actual = hamming_hex(other["phash"], row["phash"])
            if actual <= distance and other["content_sha256"] != row["content_sha256"]:
                results.append(
                    {
                        "left": other["sample_id"],
                        "left_role": other["role"],
                        "right": row["sample_id"],
                        "right_role": row["role"],
                        "distance": actual,
                    }
                )
                if len(results) >= cap:
                    return results
        for band, (offset, width) in enumerate(offsets):
            shift = 64 - offset - width
            buckets[(band, (value >> shift) & ((1 << width) - 1))].append(index)
    return results


def audit_manifest(rows: list[dict[str, Any]], phash_distance: int = 4, check_files: bool = True) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    sample_ids = Counter(row["sample_id"] for row in rows)
    for sample_id, count in sample_ids.items():
        if count > 1:
            issues.append({"severity": "error", "code": "duplicate_sample_id", "sample_id": sample_id, "count": count})

    group_roles: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_files = 0
    invalid_hashes = 0
    for row in rows:
        group_roles[row["group_id"]].add(row["role"])
        hashes[row["content_sha256"]].append(row)
        if len(row["content_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["content_sha256"].lower()):
            invalid_hashes += 1
        if len(row["phash"]) != 16 or any(c not in "0123456789abcdef" for c in row["phash"].lower()):
            invalid_hashes += 1
        if check_files and not Path(row["path"]).is_file():
            missing_files += 1
    for group, roles in group_roles.items():
        if len(roles) > 1:
            issues.append({"severity": "error", "code": "group_crosses_roles", "group_id": group, "roles": sorted(roles)})
    exact = []
    for content_hash, matches in hashes.items():
        roles = {row["role"] for row in matches}
        if len(roles) > 1:
            exact.append({"sha256": content_hash, "sample_ids": [row["sample_id"] for row in matches], "roles": sorted(roles)})
    near = []
    valid_phash_rows = [row for row in rows if len(row["phash"]) == 16]
    if phash_distance >= 0 and valid_phash_rows:
        near = _near_cross_role(valid_phash_rows, phash_distance)
    if exact:
        issues.append({"severity": "error", "code": "exact_cross_role_overlap", "count": len(exact), "examples": exact[:20]})
    if near:
        issues.append({"severity": "error", "code": "near_cross_role_overlap", "count_at_least": len(near), "examples": near[:20]})
    if missing_files:
        issues.append({"severity": "error", "code": "missing_files", "count": missing_files})
    if invalid_hashes:
        issues.append({"severity": "error", "code": "invalid_hashes", "count": invalid_hashes})

    role_class = Counter((row["role"], str(row["label"])) for row in rows)
    role_groups = Counter((row["role"], row["group_id"]) for row in rows)
    summary = {
        role: {
            "n": sum(count for (r, _), count in role_class.items() if r == role),
            "real": role_class[(role, "0")],
            "fake": role_class[(role, "1")],
            "groups": sum(1 for (r, _), _count in role_groups.items() if r == role),
        }
        for role in ("train", "calibration", "test")
    }
    return {
        "valid": not any(issue["severity"] == "error" for issue in issues),
        "summary": summary,
        "issues": issues,
        "near_duplicate_search": {"phash_distance": phash_distance, "method": "exact_banded_candidates"},
    }

