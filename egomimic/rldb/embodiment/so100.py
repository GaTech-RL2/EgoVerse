from __future__ import annotations

from typing import Literal

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    Transform,
    build_so100_singlearm_transform_list,
)


class So100SingleArm(Embodiment):
    """SO100 right-arm dataset with fixed-camera EE state and command actions."""

    VIZ_INTRINSICS_KEY = "base"
    ACTION_HORIZON_REAL = 30
    ACTION_CHUNK_LENGTH = 64
    ACTION_STRIDE = 1

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal["camera_frame_ypr"] = "camera_frame_ypr",
        chunk_length: int | None = None,
        stride: int | None = None,
    ) -> list[Transform]:
        if mode != "camera_frame_ypr":
            raise ValueError(
                f"Unsupported SO100 transform mode '{mode}'. Expected 'camera_frame_ypr'."
            )
        return build_so100_singlearm_transform_list(
            chunk_length=chunk_length or cls.ACTION_CHUNK_LENGTH,
            stride=stride or cls.ACTION_STRIDE,
        )

    @classmethod
    def _get_keymap(
        cls,
        keymap_mode: Literal["camera_frame_ypr"] = "camera_frame_ypr",
    ):
        if keymap_mode != "camera_frame_ypr":
            raise ValueError(
                f"Unsupported SO100 keymap mode '{keymap_mode}'. Expected 'camera_frame_ypr'."
            )
        return {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "obs_ee_pose_cam_rotvec": {
                "key_type": "proprio_keys",
                "zarr_key": "obs_ee_pose_cam_rotvec",
            },
            "cmd_ee_pose_cam_rotvec": {
                "key_type": "action_keys",
                "zarr_key": "cmd_ee_pose_cam_rotvec",
                "horizon": cls.ACTION_HORIZON_REAL,
            },
        }
