"""The synthetic episodes must load through each vendor's REAL keymap + transform list."""

import numpy as np
import pytest
from fixtures.synthetic_episodes import VENDORS, write_episode

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset

ACTION_DIM = {"eva_bimanual": 14, "human_bimanual": 12}


@pytest.mark.parametrize("vendor", sorted(VENDORS))
def test_episode_loads_with_vendor_recipe(tmp_path, vendor):
    v = VENDORS[vendor]
    for i in range(2):
        write_episode(tmp_path, vendor, seed=i)
    if v.embodiment == "eva_bimanual":
        key_map = Eva.get_keymap(keymap_mode="cartesian", annotation_key="annotations")
        transforms = Eva.get_transform_list(mode="cartesian")
    else:
        key_map = Human.get_keymap(
            keymap_mode="cartesian", annotation_key="annotations"
        )
        transforms = Human.get_transform_list(mode="cartesian", stride=v.stride)
    resolver = LocalEpisodeResolver(
        tmp_path, key_map=key_map, transform_list=transforms
    )
    ds = MultiDataset._from_resolver(resolver, mode="total")
    sample = ds[0]
    assert sample["actions_cartesian"].shape == (100, ACTION_DIM[v.embodiment])
    assert sample["observations.state.ee_pose"].shape == (ACTION_DIM[v.embodiment],)
    assert np.isfinite(np.asarray(sample["actions_cartesian"])).all()
    image_keys = [k for k in sample if k.startswith("observations.images.")]
    assert len(image_keys) == len(v.cameras), (v.cameras, image_keys)
