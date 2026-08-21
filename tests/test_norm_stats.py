import numpy as np
import torch

from egomimic.rldb.norm_stats import NormStats


def _state(mode):
    return {
        "norm_mode": mode,
        "embodiments": [8],
        "key_types": {8: {"action": "action_keys", "image": "camera_keys"}},
        "zarr_keys": {8: {"action": "actions_cartesian", "image": "front"}},
        "shapes": {8: {"action": (2,)}},
        "norm_stats": {
            8: {
                "action": {
                    "mean": np.array([2.0, 4.0]),
                    "std": np.array([2.0, 4.0]),
                    "min": np.array([0.0, 0.0]),
                    "max": np.array([4.0, 8.0]),
                    "quantile_1": np.array([0.0, 0.0]),
                    "quantile_99": np.array([4.0, 8.0]),
                }
            }
        },
    }


def test_checkpoint_norm_stats_round_trip_all_modes():
    action = torch.tensor([[1.0, 6.0]])
    for mode in ("zscore", "minmax", "quantile"):
        stats = NormStats(_state(mode))
        normalized = stats.normalize({"actions_cartesian": action}, 8)
        restored = stats.unnormalize(normalized, 8)
        torch.testing.assert_close(restored["actions_cartesian"], action)


def test_checkpoint_norm_stats_key_interface_and_numpy_input():
    stats = NormStats(_state("zscore"))
    normalized = stats.normalize(
        {"actions_cartesian": np.array([2.0, 4.0], dtype=np.float32)}, 8
    )

    torch.testing.assert_close(normalized["actions_cartesian"], torch.zeros(2))
    assert stats.keys_of_type("action_keys", 8) == ["action"]
    assert stats.keyname_to_zarr_key("action", 8) == "actions_cartesian"
    assert stats.zarr_key_to_keyname("actions_cartesian", 8) == "action"
    assert stats.key_shape("action", 8) == (2,)
