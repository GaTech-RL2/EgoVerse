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

Validation viz: actions_cartesian is in robot base frame (faive2lerobot's
``--base-frame`` flag), so projecting to the front camera requires the
extra base->cam step that the generic ``_viz_traj`` skips. ``Eve.viz``
overrides the base class to apply ``inv(T_base_cam)`` per arm before the
intrinsics projection, using the ``ethOct14`` calibration.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import NumpyToTensor, Transform
from egomimic.utils.egomimicUtils import (
    EXTRINSICS,
    INTRINSICS,
    cam_frame_to_cam_pixels,
    draw_dot_on_frame,
    ee_pose_to_cam_frame,
)
from egomimic.utils.pose_utils import _split_action_pose
from egomimic.utils.viz_utils import (
    ColorPalette,
    _prepare_viz_image,
)


class Eve(Embodiment):
    VIZ_INTRINSICS_KEY = "base"
    VIZ_EXTRINSICS_KEY = "ethOct14"
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

    @classmethod
    def viz(
        cls,
        image,
        viz_data,
        mode: Literal["traj", "traj+rotation", "axes", "annotations"] = "traj",
        intrinsics_key: str | None = None,
        extrinsics_key: str | None = None,
        **kwargs,
    ):
        """Project base-frame action trajectories onto the front camera.

        Mirrors the OLD EgoVerse ``draw_actions`` base->cam pipeline: each
        per-arm xyz chunk is converted to camera frame via the inverse of the
        per-arm camera-to-base extrinsic, then projected with the front-camera
        intrinsics. Other modes fall back to the base implementation.
        """
        if mode != "traj":
            return super().viz(
                image,
                viz_data,
                mode=mode,
                intrinsics_key=intrinsics_key,
                **kwargs,
            )

        intrinsics_key = intrinsics_key or cls.VIZ_INTRINSICS_KEY
        extrinsics_key = extrinsics_key or cls.VIZ_EXTRINSICS_KEY
        color = kwargs.get("color", "Blues")
        alpha = kwargs.get("alpha", 1.0)
        if not ColorPalette.is_valid(color):
            raise ValueError(f"Invalid color palette: {color}")

        intrinsics = INTRINSICS[intrinsics_key]
        extr = EXTRINSICS[extrinsics_key]
        T_left_cam_base = np.linalg.inv(extr["left_cam"])
        T_right_cam_base = np.linalg.inv(extr["right_cam"])

        actions = np.asarray(viz_data)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        left_xyz, _, right_xyz, _ = _split_action_pose(actions)
        left_xyz = left_xyz.reshape(-1, 3)
        right_xyz = right_xyz.reshape(-1, 3)

        left_cam = ee_pose_to_cam_frame(left_xyz, T_left_cam_base)
        right_cam = ee_pose_to_cam_frame(right_xyz, T_right_cam_base)
        pts_cam = np.concatenate([left_cam, right_cam], axis=0)
        pts_pix = cam_frame_to_cam_pixels(pts_cam, intrinsics)

        base = _prepare_viz_image(image)
        overlay = draw_dot_on_frame(base.copy(), pts_pix, show=False, palette=color)
        if alpha < 1.0:
            return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)
        return overlay
