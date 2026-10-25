"""Dexmate robot with two Sharpa Wave hands (dexmate_bimanual).

Episodes store the palm pose ({side}.obs_ee_pose), MANO-layout hand keypoints
({side}.obs_hand_keypoints) and the head link pose (obs_head_link_pose), all in
the robot base frame (egomimic.rldb.zarr.dexmate_rules). Training re-expresses
the hand tracks in the head link frame, so the keypoints mode reuses Human's
pipeline with obs_head_link_pose as the target frame. Targets are future
observed hand configurations; joints, commands and tactile stay in storage for
other heads. No forward kinematics runs here.
"""

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.embodiment.human import (
    Human,
    _build_human_keypoints_bimanual_transform_list,
)
from egomimic.rldb.zarr.dexmate_rules import HEAD_LINK_POSE, SIDES

# Human keymap zarr keys -> what a Dexmate episode stores in their place.
_ZARR_KEYS = {"obs_keypoints": "obs_hand_keypoints", "obs_wrist_pose": "obs_ee_pose"}


class Dexmate(Embodiment):
    ACTION_HORIZON = Human.ACTION_HORIZON
    WRIST_IMAGE_KEYS = {side: f"observations.images.{side}_wrist_img" for side in SIDES}

    @classmethod
    def get_keymap(
        cls,
        keymap_mode: str,
        wrist_cameras: bool = True,
        norm_mode: bool = False,
        annotation_key: str | None = None,
    ):
        """``wrist_cameras`` also loads images.{left,right}_wrist; set it False
        for episodes recorded without them. ``norm_mode``/``annotation_key``
        behave as in the base."""
        key_map = super().get_keymap(
            keymap_mode, norm_mode=norm_mode, annotation_key=annotation_key
        )
        if not wrist_cameras:
            for key in cls.WRIST_IMAGE_KEYS.values():
                key_map.pop(key, None)
        return key_map

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        if keymap_mode != "keypoints":
            raise ValueError(
                f"Unsupported keymap_mode {keymap_mode!r} for {cls.__name__}; "
                "expected 'keypoints'."
            )
        key_map = Human._get_keymap("keypoints", has_head_pose=False)
        for entry in key_map.values():
            prefix, dot, name = entry["zarr_key"].rpartition(".")
            entry["zarr_key"] = prefix + dot + _ZARR_KEYS.get(name, name)
        key_map[HEAD_LINK_POSE] = {
            "key_type": "proprio_keys",
            "zarr_key": HEAD_LINK_POSE,
        }
        for side, key in cls.WRIST_IMAGE_KEYS.items():
            key_map[key] = {
                "key_type": "camera_keys",
                "zarr_key": f"images.{side}_wrist",
            }
        return key_map

    @classmethod
    def get_transform_list(
        cls, mode: str = "keypoints_headframe_quat", stride: int = 1
    ):
        """Keypoints and palm poses in the head link frame: actions_keypoints and
        observations.state.keypoints, 140 wide with quaternion palms (138 YPR)."""
        if mode not in ("keypoints_headframe_quat", "keypoints_headframe_ypr"):
            raise ValueError(
                f"Unsupported transform_list mode {mode!r} for {cls.__name__}"
            )
        return _build_human_keypoints_bimanual_transform_list(
            target_world=HEAD_LINK_POSE,
            target_world_ypr=f"{HEAD_LINK_POSE}_ypr",
            stride=stride,
            is_quat=mode == "keypoints_headframe_quat",
        )
