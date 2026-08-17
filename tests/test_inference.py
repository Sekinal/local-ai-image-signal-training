import numpy as np
from PIL import Image

from aiblink_validation.inference import degrade, preprocess


def test_preprocess_shape_and_deterministic_views():
    image = Image.fromarray(np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3))
    clean = preprocess(degrade(image, "clean"))
    web_a = preprocess(degrade(image, "web"))
    web_b = preprocess(degrade(image, "web"))
    assert clean.shape == (3, 384, 384)
    assert clean.dtype == np.float32
    np.testing.assert_array_equal(web_a, web_b)
    assert not np.array_equal(clean, web_a)

