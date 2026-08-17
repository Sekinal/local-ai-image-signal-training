import numpy as np

from aiblink_validation.metrics import binary_metrics, clustered_bootstrap, optimal_balanced_threshold


def test_fixed_threshold_semantics_and_metrics():
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.65, 0.65, 0.9])
    result = binary_metrics(y, probability, 0.65)
    assert result["confusion"] == {"tp": 2, "tn": 1, "fp": 1, "fn": 0}
    assert result["balanced_accuracy"] == 0.75
    assert result["roc_auc"] == 0.875


def test_exact_optimal_threshold_not_grid_approximation():
    y = np.array([0, 0, 1, 1])
    score = np.array([-4.0, -2.00001, -2.0, 7.0])
    threshold, value = optimal_balanced_threshold(y, score)
    assert threshold == -2.0
    assert value == 1.0


def test_clustered_bootstrap_keeps_views_together():
    rows = []
    for label in (0, 1):
        for index in range(5):
            for view in ("clean", "web"):
                rows.append(
                    {
                        "sample_id": f"{label}-{index}",
                        "label": label,
                        "probability": 0.1 if label == 0 else 0.9,
                        "view": view,
                    }
                )
    intervals = clustered_bootstrap(rows, 0.65, replicates=50, seed=1, confidence=0.95)
    assert intervals["balanced_accuracy"] == {"low": 1.0, "high": 1.0}

