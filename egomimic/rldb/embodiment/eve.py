"""
Eve embodiment for ETH/CVG faive-bimanual hand data.

Unlike Eva (which has world-frame raw poses + extrinsics that need to be
transformed at load time) and Aria (which derives chunked actions from
stacked observations), the Eve pipeline assumes the upstream lerobot
converter (faive2lerobot) has already produced canonical, chunked,
frame-resolved trajectories. Zarr stores written by lerobot_to_zarr.py
contain those exact arrays with no further math required.

The keymap therefore maps post-transform batch keys directly to the same
zarr keys, and the transform list only converts numpy arrays to torch
tensors so downstream code receives the expected types.
"""

from __future__ import annotations

from typing import Literal

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import NumpyToTensor, Transform


class Eve(Embodiment):
    VIZ_INTRINSICS_KEY = "base"
    VIZ_IMAGE_KEY = "observations.images.front_img_1"

    @staticmethod
    def get_transform_list(
        mode: Literal["cartesian"],
    ) -> list[Transform]:
        if mode == "cartesian":
            return [
                NumpyToTensor(
                    keys=[
                        "observations.state.ee_pose",
                        "observations.state.cartesian_arm",
                        "observations.state.joints_hand",
                        "actions_cartesian",
                        "actions_joints",
                    ]
                ),
            ]
        raise ValueError(f"Unsupported Eve transform mode: {mode}")

    @classmethod
    def _get_keymap(cls, keymap_mode: Literal["cartesian"]):
        if keymap_mode != "cartesian":
            raise ValueError(f"Unsupported Eve keymap mode: {keymap_mode}")
        return {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "observations.images.front_img_1",
            },
            "observations.state.ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.ee_pose",
            },
            "observations.state.cartesian_arm": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.cartesian_arm",
            },
            "observations.state.joints_hand": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.joints_hand",
            },
            "actions_cartesian": {
                "key_type": "action_keys",
                "zarr_key": "actions_cartesian",
            },
            "actions_joints": {
                "key_type": "action_keys",
                "zarr_key": "actions_joints",
            },
        }
