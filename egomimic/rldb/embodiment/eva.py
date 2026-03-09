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
from egomimic.utils.egomimicUtils import (
    EXTRINSICS,
)
from egomimic.utils.pose_utils import (
    _matrix_to_xyzwxyz,
)
from egomimic.utils.type_utils import _to_numpy
from egomimic.utils.viz_utils import (
    _viz_axes,
    _viz_traj,
)


class Eva(Embodiment):
    VIZ_INTRINSICS_KEY = "base"
    VIZ_IMAGE_KEY = "observations.images.front_img_1"

    @staticmethod
    def get_transform_list(
        arm_mode: Literal["bimanual", "left_arm", "right_arm"] = "bimanual",
        **kwargs,
    ) -> list[Transform]:
        return _build_eva_bimanual_transform_list(arm_mode=arm_mode, **kwargs)

    @classmethod
    def viz_transformed_batch(cls, batch, mode=""):
        """
        Visualize one transformed EVA batch sample.

        Modes:
            - traj: draw left/right trajectories from actions_cartesian.
            - axes: draw local xyz axes at each anchor using ypr.
        """
        image_key = cls.VIZ_IMAGE_KEY
        action_key = "actions_cartesian"
        intrinsics_key = "base"
        mode = (mode or "traj").lower()

        images = _to_numpy(batch[image_key][0])
        actions = _to_numpy(batch[action_key][0])

        return cls.viz(
            images=images, actions=actions, mode=mode, intrinsics_key=intrinsics_key
        )

    @classmethod
    def viz(cls, images, actions, mode=Literal["traj", "axes"], intrinsics_key=None):
        intrinsics_key = intrinsics_key or cls.VIZ_INTRINSICS_KEY
        if mode == "traj":
            return _viz_traj(
                images=images,
                actions=actions,
                intrinsics_key=intrinsics_key,
            )
        if mode == "axes":
            return _viz_axes(
                images=images,
                actions=actions,
                intrinsics_key=intrinsics_key,
            )
        raise ValueError(
            f"Unsupported mode '{mode}'. Expected one of: " f"('traj', 'axes')."
        )

    @classmethod
    def get_keymap(
        cls, arm_mode: Literal["bimanual", "left_arm", "right_arm"] = "bimanual"
    ):
        """Return key_map for zarr loading. For single-arm modes, omits the other arm's keys so
        datasets without right/left pose data load correctly."""
        base = {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "observations.images.right_wrist_img": {
                "key_type": "camera_keys",
                "zarr_key": "images.right_wrist",
            },
            "observations.images.left_wrist_img": {
                "key_type": "camera_keys",
                "zarr_key": "images.left_wrist",
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
        if arm_mode == "left_arm":
            drop = {
                "right.obs_ee_pose",
                "right.obs_gripper",
                "right.gripper",
                "right.cmd_ee_pose",
            }
            return {k: v for k, v in base.items() if k not in drop}
        if arm_mode == "right_arm":
            drop = {
                "left.obs_ee_pose",
                "left.obs_gripper",
                "left.gripper",
                "left.cmd_ee_pose",
            }
            return {k: v for k, v in base.items() if k not in drop}
        return base


def _build_eva_bimanual_transform_list(
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
    extrinsics_key: str = "x5Dec13_2",
    is_quat: bool = True,
    arm_mode: Literal["bimanual", "left_arm", "right_arm"] = "bimanual",
) -> list[Transform]:
    """Canonical EVA bimanual transform pipeline used by tests and notebooks."""
    extrinsics = EXTRINSICS[extrinsics_key]
    left_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["left"][None, :])[0]
    right_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["right"][None, :])[0]
    left_extra_batch_key = {"left_extrinsics_pose": left_extrinsics_pose}
    right_extra_batch_key = {"right_extrinsics_pose": right_extrinsics_pose}

    mode = "xyzwxyz" if is_quat else "xyzypr"

    action_transforms = [
        ActionChunkCoordinateFrameTransform(
            target_world=left_target_world,
            chunk_world=left_cmd_world,
            transformed_key_name=left_cmd_camframe,
            extra_batch_key=left_extra_batch_key,
            mode=mode,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_target_world,
            chunk_world=right_cmd_world,
            transformed_key_name=right_cmd_camframe,
            extra_batch_key=right_extra_batch_key,
            mode=mode,
        ),
    ]

    pose_transforms = [
        PoseCoordinateFrameTransform(
            target_world=left_target_world,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_pose,
            mode=mode,
        ),
        PoseCoordinateFrameTransform(
            target_world=right_target_world,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_pose,
            mode=mode,
        ),
    ]

    interpolate_transforms = [
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_cmd_camframe,
            output_action_key=left_cmd_camframe,
            stride=stride,
            mode=mode,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_cmd_camframe,
            output_action_key=right_cmd_camframe,
            stride=stride,
            mode=mode,
        ),
    ]

    gripper_transforms = [
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
    ]

    quat_keys = [left_cmd_camframe, left_obs_pose, right_cmd_camframe, right_obs_pose]
    grip_cam_concat_keys = [
        left_cmd_camframe,
        left_gripper,
        right_cmd_camframe,
        right_gripper,
    ]
    obs_concat_keys = [
        left_obs_pose,
        left_obs_gripper,
        right_obs_pose,
        right_obs_gripper,
    ]
    delete_keys = [
        left_cmd_world,
        left_target_world,
        right_cmd_world,
        right_target_world,
    ]

    if arm_mode == "right_arm":
        action_transforms = action_transforms[1:]
        pose_transforms = pose_transforms[1:]
        interpolate_transforms = interpolate_transforms[1:]
        gripper_transforms = gripper_transforms[1:]
        quat_keys = quat_keys[2:]
        grip_cam_concat_keys = grip_cam_concat_keys[2:]
        obs_concat_keys = obs_concat_keys[2:]
        delete_keys = delete_keys[2:]
    elif arm_mode == "left_arm":
        action_transforms = action_transforms[:1]
        pose_transforms = pose_transforms[:1]
        interpolate_transforms = interpolate_transforms[:1]
        gripper_transforms = gripper_transforms[:1]
        quat_keys = quat_keys[:2]
        grip_cam_concat_keys = grip_cam_concat_keys[:2]
        obs_concat_keys = obs_concat_keys[:2]
        delete_keys = delete_keys[:2]
    elif arm_mode == "bimanual":
        pass

    transform_list = (
        action_transforms
        + pose_transforms
        + interpolate_transforms
        + gripper_transforms
    )

    if is_quat:
        transform_list.append(XYZWXYZ_to_XYZYPR(keys=quat_keys))

    transform_list.extend(
        [
            ConcatKeys(
                key_list=grip_cam_concat_keys,
                new_key_name=actions_key,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=obs_concat_keys,
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(
                keys_to_delete=delete_keys,
            ),
            NumpyToTensor(
                keys=[
                    actions_key,
                    obs_key,
                ]
            ),
        ]
    )
    return transform_list
