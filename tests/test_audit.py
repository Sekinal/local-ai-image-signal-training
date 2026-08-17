from aiblink_validation.audit import audit_manifest


def row(sample_id, role, group, content_hash, phash, label=0):
    return {
        "sample_id": sample_id,
        "path": __file__,
        "label": label,
        "dataset": "fixture",
        "source": group,
        "group_id": group,
        "role": role,
        "content_sha256": content_hash,
        "phash": phash,
    }


def test_audit_detects_exact_and_group_leakage():
    shared_hash = "a" * 64
    rows = [
        row("a", "train", "shared", shared_hash, "0000000000000000"),
        row("b", "test", "shared", shared_hash, "0000000000000000"),
    ]
    result = audit_manifest(rows, phash_distance=4)
    assert not result["valid"]
    codes = {issue["code"] for issue in result["issues"]}
    assert {"group_crosses_roles", "exact_cross_role_overlap"} <= codes


def test_audit_detects_near_duplicate():
    rows = [
        row("a", "calibration", "a", "a" * 64, "0000000000000000"),
        row("b", "test", "b", "b" * 64, "0000000000000003"),
    ]
    result = audit_manifest(rows, phash_distance=2)
    assert any(issue["code"] == "near_cross_role_overlap" for issue in result["issues"])

