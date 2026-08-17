import numpy as np
import pytest
from PIL import Image

from aiblink_validation.dinov3 import calibration_fold, fit_probe, pool_backbone_output, preprocess_dinov3


def test_dinov3_preprocessing_is_fixed_size_and_finite():
    image = Image.new("RGB", (93, 51), (20, 80, 140))
    output = preprocess_dinov3(image, 256)
    assert output.shape == (3, 256, 256)
    assert np.isfinite(output).all()


def test_dinov3_calibration_folds_hold_out_generators_and_whole_real_images():
    groups = ["fake-a", "fake-b", "fake-c"]
    assert calibration_fold({"label": 1, "group_id": "fake-b", "sample_id": "x"}, groups) == 1
    row = {"label": 0, "group_id": "real", "sample_id": "real-7"}
    assert calibration_fold(row, groups) == calibration_fold(row, groups)
    assert 0 <= calibration_fold(row, groups) < 3


def test_dinov3_probe_separates_a_toy_problem():
    pytest.importorskip("torch")
    features = np.array([[-1.0, 0.0], [-0.8, 0.1], [0.8, -0.1], [1.0, 0.0]])
    labels = np.array([0, 0, 1, 1])
    probe, diagnostics = fit_probe(features, labels, ridge=1e-3)
    assert np.all(probe.logits(features[:2]) < 0)
    assert np.all(probe.logits(features[2:]) > 0)
    assert diagnostics["optimizer"] == "LBFGS"


def test_backbone_pooling_supports_tokens_and_native_heads():
    torch = pytest.importorskip("torch")

    class ToyModel:
        num_prefix_tokens = 1

        @staticmethod
        def forward_head(output, pre_logits=True):
            assert pre_logits
            return output[:, 1:].mean(dim=1)

    tokens = torch.tensor([[[3.0, 4.0], [1.0, 0.0], [0.0, 1.0]]])
    cls = pool_backbone_output(ToyModel(), tokens, "cls")
    cls_and_patches = pool_backbone_output(ToyModel(), tokens, "cls_mean_patch")
    native = pool_backbone_output(ToyModel(), tokens, "model_head")
    assert cls.shape == (1, 2)
    assert cls_and_patches.shape == (1, 4)
    assert native.shape == (1, 2)
    assert torch.allclose(torch.linalg.vector_norm(cls, dim=1), torch.ones(1))
    assert torch.allclose(torch.linalg.vector_norm(native, dim=1), torch.ones(1))


def test_backbone_pooling_rejects_incompatible_shapes():
    torch = pytest.importorskip("torch")

    class ToyModel:
        num_prefix_tokens = 1

    with pytest.raises(ValueError, match="CLS pooling requires token output"):
        pool_backbone_output(ToyModel(), torch.zeros(1, 2, 3, 3), "cls")
    with pytest.raises(ValueError, match="unknown backbone pooling mode"):
        pool_backbone_output(ToyModel(), torch.zeros(1, 2, 3), "mystery")
