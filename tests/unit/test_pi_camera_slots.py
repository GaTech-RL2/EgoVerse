"""Pi owns the mapping from dataset camera keys to openpi's fixed camera
slots (base_0_rgb, *_wrist_0_rgb). Datasets emit one naming for every algo."""

import pytest
import torch

from egomimic.models.preprocess_pi_obs import PI_CAMERA_SLOTS, gather_pi_images
from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human

CPU = torch.device("cpu")


def _img(seed: float) -> torch.Tensor:
    return torch.full((2, 3, 8, 8), seed)


def test_slot_map_covers_the_three_openpi_slots():
    assert tuple(PI_CAMERA_SLOTS) == (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    assert PI_CAMERA_SLOTS["base_0_rgb"] == Eva.VIZ_IMAGE_KEY == Human.VIZ_IMAGE_KEY


def test_dataset_names_map_onto_slots():
    batch = {
        "observations.images.front_img_1": _img(1.0),
        "observations.images.left_wrist_img": _img(2.0),
        "observations.images.right_wrist_img": _img(3.0),
        "actions_cartesian": torch.zeros(2, 100, 14),
    }
    images, present = gather_pi_images(batch, PI_CAMERA_SLOTS, CPU)
    assert tuple(images) == tuple(PI_CAMERA_SLOTS)
    assert images["base_0_rgb"][0, 0, 0, 0] == 1.0
    assert images["left_wrist_0_rgb"][0, 0, 0, 0] == 2.0
    assert images["right_wrist_0_rgb"][0, 0, 0, 0] == 3.0
    assert present == {k: True for k in PI_CAMERA_SLOTS}


def test_legacy_slot_names_still_accepted():
    # Robot rollout and pre-remap checkpoints hand the wrapper openpi names.
    batch = {"base_0_rgb": _img(1.0), "right_wrist_0_rgb": _img(3.0)}
    images, present = gather_pi_images(batch, PI_CAMERA_SLOTS, CPU)
    assert images["base_0_rgb"][0, 0, 0, 0] == 1.0
    assert images["right_wrist_0_rgb"][0, 0, 0, 0] == 3.0
    assert present == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": False,
        "right_wrist_0_rgb": True,
    }


def test_dataset_name_wins_over_legacy_alias():
    batch = {"observations.images.front_img_1": _img(1.0), "base_0_rgb": _img(9.0)}
    images, _ = gather_pi_images(batch, PI_CAMERA_SLOTS, CPU)
    assert images["base_0_rgb"][0, 0, 0, 0] == 1.0


def test_missing_wrists_are_duplicated_and_flagged_absent():
    # Human data has only the front camera; openpi still wants three slots.
    batch = {"observations.images.front_img_1": _img(1.0)}
    images, present = gather_pi_images(batch, PI_CAMERA_SLOTS, CPU)
    assert tuple(images) == tuple(PI_CAMERA_SLOTS)
    for k in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
        assert torch.equal(images[k], images["base_0_rgb"])
        assert present[k] is False
    assert present["base_0_rgb"] is True


def test_bhwc_images_are_normalised_to_bchw():
    batch = {"observations.images.front_img_1": torch.zeros(2, 8, 8, 3)}
    images, _ = gather_pi_images(batch, PI_CAMERA_SLOTS, CPU)
    assert images["base_0_rgb"].shape == (2, 3, 8, 8)


def test_no_camera_at_all_is_an_error():
    with pytest.raises(ValueError, match="base_0_rgb"):
        gather_pi_images(
            {"actions_cartesian": torch.zeros(2, 100, 14)}, PI_CAMERA_SLOTS, CPU
        )


@pytest.mark.parametrize("cls", [Eva, Human])
def test_pi_keymap_mode_is_a_deprecated_alias(cls):
    # Saved configs from pre-remap runs still say cartesian_pi; they must rebuild.
    with pytest.warns(FutureWarning, match="cartesian_pi"):
        km = cls.get_keymap(keymap_mode="cartesian_pi")
    assert km == cls.get_keymap(keymap_mode="cartesian")


@pytest.mark.parametrize("cls", [Eva, Human])
def test_cartesian_keymap_emits_dataset_front_key(cls):
    km = cls.get_keymap(keymap_mode="cartesian")
    assert cls.VIZ_IMAGE_KEY in km
    assert "base_0_rgb" not in km
