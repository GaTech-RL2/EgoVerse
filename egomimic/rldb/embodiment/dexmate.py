"""Consume delivered palm/keypoint arrays using the current egoview camera.

Targets are future observed hand configurations (as for human keypoints).
Commands and raw joints remain available in storage for separate action heads.
No model loading or forward kinematics occurs in this data path.
"""

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.embodiment.human import (
    Human,
    _build_human_keypoints_bimanual_transform_list,
)


class Dexmate(Embodiment):
    ACTION_HORIZON = 30

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        if keymap_mode not in ("keypoints", "keypoints_pi"):
            raise ValueError(
                f"Dexmate supports keypoints or keypoints_pi, got {keymap_mode!r}"
            )
        keymap = Human._get_keymap(keymap_mode, has_head_pose=True)
        for entry in keymap.values():
            entry["zarr_key"] = (
                entry["zarr_key"]
                .replace("obs_keypoints", "obs_hand_keypoints")
                .replace("obs_wrist_pose", "obs_ee_pose")
            )
        for side in ("left", "right"):
            key = (
                f"{side}_wrist_0_rgb"
                if keymap_mode.endswith("_pi")
                else f"observations.images.{side}_wrist"
            )
            keymap[key] = {
                "key_type": "camera_keys",
                "zarr_key": f"images.{side}_wrist",
            }
        return keymap

    @classmethod
    def get_transform_list(cls, mode="keypoints_headframe_quat", stride=1):
        if mode not in ("keypoints_headframe_quat", "keypoints_headframe_ypr"):
            raise ValueError(f"unsupported Dexmate transform mode {mode!r}")
        return _build_human_keypoints_bimanual_transform_list(
            stride=stride, is_quat=mode.endswith("quat")
        )
