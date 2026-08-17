import numpy as np
import pytest

from aiblink_validation.calibration import fit_calibrator, select_and_fit


def test_bias_maps_optimal_boundary_to_target():
    logits = np.array([-4, -2, 1, 3], dtype=float)
    y = np.array([0, 0, 1, 1])
    calibrator = fit_calibrator(logits, y, "bias", 0.65)
    assert abs(float(calibrator.transform(np.array([calibrator.raw_boundary]))[0]) - 0.65) < 1e-12
    assert calibrator.slope > 0


def test_logo_selection_and_fit():
    rows = []
    for label, prefix, centers in ((0, "real", [-4, -3, -2]), (1, "fake", [2, 3, 4])):
        for group_index, center in enumerate(centers):
            for offset in (-0.1, 0.1):
                rows.append(
                    {
                        "sample_id": f"{prefix}-{group_index}-{offset}",
                        "group_id": f"{prefix}-{group_index}",
                        "label": label,
                        "raw_logit": center + offset,
                    }
                )
    calibrator, diagnostics = select_and_fit(rows, ["identity", "bias", "platt"], 0.65)
    assert calibrator.method in {"identity", "bias", "platt"}
    assert diagnostics["folds"] == 3
    # The weakest fake group is outside the training-fold fake-score range when
    # held out. LOGO must expose that miss instead of reporting pooled separability.
    assert diagnostics["candidates"][calibrator.method]["logo_macro_balanced_accuracy"] >= 0.8
