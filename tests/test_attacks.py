import numpy as np
from PIL import Image

from aiblink_validation.attacks import apply_attack, load_attack_profile


def test_attack_profile_is_deterministic():
    attacks, _ = load_attack_profile("configs/redteam.yaml", "smoke")
    image = Image.fromarray(np.arange(320 * 480 * 3, dtype=np.uint8).reshape(320, 480, 3))
    for attack in attacks:
        first = np.asarray(apply_attack(image, attack, "sample-1"))
        second = np.asarray(apply_attack(image, attack, "sample-1"))
        np.testing.assert_array_equal(first, second)
        assert first.shape[0] > 0 and first.shape[1] > 0


def test_core_profile_has_unique_ids_and_clean_control():
    attacks, _ = load_attack_profile("configs/redteam.yaml", "core")
    ids = [attack.attack_id for attack in attacks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "clean"
    assert len(ids) >= 25
    image = Image.fromarray(np.arange(180 * 240 * 3, dtype=np.uint8).reshape(180, 240, 3))
    background = Image.new("RGB", (300, 220), (120, 130, 140))
    for attack in attacks:
        output = apply_attack(image, attack, "core-fixture", background=background)
        assert output.mode == "RGB"
        assert output.width > 0 and output.height > 0
