"""Keymap and transform helpers for the pushshapes_sim embodiment."""

from __future__ import annotations


def get_keymap(**kwargs) -> dict:
    """Return the key_map for pushshapes_sim ZarrDataset.

    Extra kwargs (e.g. norm_mode=True injected by trainHydra) are accepted
    and ignored so that norm-stat collection doesn't crash.
    """
    return {
        "front_img_1": {
            "key_type": "camera_keys",
            "zarr_key": "observations.images.front_img_1",
        },
        "state_agent_obj": {
            "key_type": "proprio_keys",
            "zarr_key": "observations.state",
        },
        "actions": {
            "key_type": "action_keys",
            "zarr_key": "actions",
        },
    }
