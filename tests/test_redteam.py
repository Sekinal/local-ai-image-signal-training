from aiblink_validation.attacks import load_attack_profile
from aiblink_validation.io import atomic_json, atomic_jsonl, write_manifest
from aiblink_validation.redteam import generate_redteam_report


def test_redteam_report_measures_any_attack_evasion(tmp_path):
    attacks, _ = load_attack_profile("configs/redteam.yaml", "smoke")
    manifest_rows, predictions = [], []
    for label in (0, 1):
        for index in range(4):
            sample_id = f"{label}-{index}"
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "path": __file__,
                    "label": label,
                    "dataset": "fixture",
                    "source": f"source-{label}",
                    "group_id": f"group-{label}",
                    "role": "test",
                    "content_sha256": f"{label * 10 + index + 1:064x}",
                    "phash": f"{label * 10 + index + 1:016x}",
                }
            )
            for attack in attacks:
                logit = 2.0 if label else -2.0
                if label and attack.attack_id == "jpeg_q75":
                    logit = -2.0
                predictions.append(
                    {"sample_id": sample_id, "view": attack.attack_id, "raw_logit": logit, **attack.metadata()}
                )
    manifest = tmp_path / "manifest.csv"
    ledger = tmp_path / "predictions.jsonl"
    calibration = tmp_path / "calibration.json"
    write_manifest(manifest, manifest_rows)
    atomic_jsonl(ledger, predictions)
    atomic_json(
        calibration,
        {"method": "identity", "slope": 1.0, "intercept": 0.0, "raw_boundary": 0.619039, "target_threshold": 0.65},
    )
    result = generate_redteam_report(
        manifest,
        ledger,
        calibration,
        "configs/redteam.yaml",
        "smoke",
        tmp_path / "report",
        replicates=20,
    )
    assert result["attack_success"]["fake_any_attack"] == 1.0
    assert result["worst_variant_per_image"]["metrics"]["balanced_accuracy"] == 0.5
    assert not result["valid"]  # required physical/platform/generative cohorts are intentionally absent
    assert result["coverage"]["protocol_mismatch_count"] == 0


def test_redteam_report_rejects_unverified_attack_parameters(tmp_path):
    attacks, _ = load_attack_profile("configs/redteam.yaml", "adaptive")
    manifest_rows, predictions = [], []
    for label in (0, 1):
        sample_id = f"{label}-0"
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "path": __file__,
                "label": label,
                "dataset": "fixture",
                "source": f"source-{label}",
                "group_id": f"group-{label}",
                "role": "test",
                "content_sha256": f"{label + 1:064x}",
                "phash": f"{label + 1:016x}",
            }
        )
        for attack in attacks:
            metadata = attack.metadata()
            if attack.attack_id == "pgd_linf_1_255":
                metadata["attack_operations"] = [{"op": "whitebox_pgd", "epsilon": 1, "steps": 1, "restarts": 1}]
            predictions.append({"sample_id": sample_id, "view": attack.attack_id, "raw_logit": 2.0 if label else -2.0, **metadata})
    manifest = tmp_path / "manifest.csv"
    ledger = tmp_path / "predictions.jsonl"
    calibration = tmp_path / "calibration.json"
    write_manifest(manifest, manifest_rows)
    atomic_jsonl(ledger, predictions)
    atomic_json(calibration, {"method": "identity", "slope": 1.0, "intercept": 0.0, "raw_boundary": 0.619039, "target_threshold": 0.65})
    result = generate_redteam_report(
        manifest, ledger, calibration, "configs/redteam.yaml", "adaptive", tmp_path / "report", replicates=10
    )
    assert not result["valid"]
    assert result["coverage"]["protocol_mismatch_count"] == 2
    assert not next(gate for gate in result["gates"] if gate["name"] == "attack_protocol_match")["passed"]
