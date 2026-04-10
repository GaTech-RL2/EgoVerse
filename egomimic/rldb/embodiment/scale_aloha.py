from __future__ import annotations

from typing import Literal

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolateLinear,
    InterpolatePose,
    NumpyToTensor,
    PoseCoordinateFrameTransform,
    Transform,
    XYZWXYZ_to_XYZYPR,
)
from egomimic.utils.egomimicUtils import EXTRINSICS
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz


class ScaleAloha(Embodiment):
    """Scale ALOHA bimanual robot embodiment.

    Uses a transform pipeline similar to Eva: base-frame wxyz quaternion data
    in zarr is transformed to camera frame, interpolated 45->100, then
    converted to ypr for the model.
    """

    VIZ_IMAGE_KEY = "observations.images.front_img_1"
    VIZ_INTRINSICS_KEY = "scale"

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal["cartesian"] = "cartesian",
    ) -> list[Transform]:
        if mode == "cartesian":
            return _build_scale_aloha_transform_list()
        raise ValueError(f"Unsupported mode: {mode}")

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        if keymap_mode == "cartesian":
            return {
                cls.VIZ_IMAGE_KEY: {
                    "key_type": "camera_keys",
                    "zarr_key": "observations.images.front_img_1",
                },
                "observations.images.cam_low": {
                    "key_type": "camera_keys",
                    "zarr_key": "observations.images.cam_low",
                },
                "observations.images.left_wrist_img": {
                    "key_type": "camera_keys",
                    "zarr_key": "observations.images.left_wrist_img",
                },
                "observations.images.right_wrist_img": {
                    "key_type": "camera_keys",
                    "zarr_key": "observations.images.right_wrist_img",
                },
                "right.obs_ee_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_ee_pose",
                },
                "right.obs_gripper": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.gripper",
                },
                "left.obs_ee_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_ee_pose",
                },
                "left.obs_gripper": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.gripper",
                },
                "right.gripper": {
                    "key_type": "action_keys",
                    "zarr_key": "right.gripper",
                    "horizon": 45,
                },
                "left.gripper": {
                    "key_type": "action_keys",
                    "zarr_key": "left.gripper",
                    "horizon": 45,
                },
                "right.cmd_ee_pose": {
                    "key_type": "action_keys",
                    "zarr_key": "right.cmd_ee_pose",
                    "horizon": 45,
                },
                "left.cmd_ee_pose": {
                    "key_type": "action_keys",
                    "zarr_key": "left.cmd_ee_pose",
                    "horizon": 45,
                },
            }
        raise ValueError(f"Unsupported keymap_mode: {keymap_mode}")


def _build_scale_aloha_transform_list(
    *,
    left_target_world: str = "left_extrinsics_pose",
    right_target_world: str = "right_extrinsics_pose",
    left_cmd_world: str = "left.cmd_ee_pose",
    right_cmd_world: str = "right.cmd_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_obs_gripper: str = "left.obs_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    left_gripper: str = "left.gripper",
    right_gripper: str = "right.gripper",
    left_cmd_camframe: str = "left.cmd_ee_pose_camframe",
    right_cmd_camframe: str = "right.cmd_ee_pose_camframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
    extrinsics_key: str = "scale_aloha",
) -> list[Transform]:
    """Transform pipeline for Scale ALOHA: base-frame quaternion -> camera-frame YPR."""
    extrinsics = EXTRINSICS[extrinsics_key]
    left_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["left"][None, :])[0]
    right_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["right"][None, :])[0]
    left_extra_batch_key = {"left_extrinsics_pose": left_extrinsics_pose}
    right_extra_batch_key = {"right_extrinsics_pose": right_extrinsics_pose}
    transform_list = [
        ActionChunkCoordinateFrameTransform(
            target_world=left_target_world,
            chunk_world=left_cmd_world,
            transformed_key_name=left_cmd_camframe,
            extra_batch_key=left_extra_batch_key,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_target_world,
            chunk_world=right_cmd_world,
            transformed_key_name=right_cmd_camframe,
            extra_batch_key=right_extra_batch_key,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=left_target_world,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_pose,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=right_target_world,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_pose,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_cmd_camframe,
            output_action_key=left_cmd_camframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_cmd_camframe,
            output_action_key=right_cmd_camframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=left_gripper,
            output_action_key=left_gripper,
            stride=stride,
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=right_gripper,
            output_action_key=right_gripper,
            stride=stride,
        ),
        XYZWXYZ_to_XYZYPR(
            keys=[
                left_cmd_camframe,
                right_cmd_camframe,
                left_obs_pose,
                right_obs_pose,
            ]
        ),
        ConcatKeys(
            key_list=[
                left_cmd_camframe,
                left_gripper,
                right_cmd_camframe,
                right_gripper,
            ],
            new_key_name=actions_key,
            delete_old_keys=True,
        ),
        ConcatKeys(
            key_list=[
                left_obs_pose,
                left_obs_gripper,
                right_obs_pose,
                right_obs_gripper,
            ],
            new_key_name=obs_key,
            delete_old_keys=True,
        ),
        DeleteKeys(
            keys_to_delete=[
                left_cmd_world,
                right_cmd_world,
                left_target_world,
                right_target_world,
            ]
        ),
        NumpyToTensor(
            keys=[
                actions_key,
                obs_key,
            ]
        ),
    ]
    return transform_list
