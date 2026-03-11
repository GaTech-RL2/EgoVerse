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
    SplitKeys,
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
        mode: Literal["cartesian", "cartesian_wristframe"] = "cartesian",
    ) -> list[Transform]:
        if mode == "cartesian":
            return _build_eva_bimanual_transform_list()
        elif mode == "cartesian_wristframe":
            return _build_eva_bimanual_eef_frame_transform_list()
        else:
            raise ValueError(
                f"Unsupported mode '{mode}'. Expected one of: 'cartesian', 'cartesian_wristframe'."
            )

    @classmethod
    def viz_transformed_batch(
        cls,
        batch,
        mode=Literal["traj", "axes"],
        action_key="actions_cartesian",
        image_key=None,
        transform_list=None,
        **kwargs,
    ):
        if transform_list is not None:
            batch = cls.apply_transform(batch, transform_list)
        image_key = image_key or cls.VIZ_IMAGE_KEY
        action_key = action_key or "actions_cartesian"
        intrinsics_key = cls.VIZ_INTRINSICS_KEY
        mode = (mode or "traj").lower()

        images = _to_numpy(batch[image_key][0])
        actions = _to_numpy(batch[action_key][0])

        return cls.viz(
            images=images,
            actions=actions,
            mode=mode,
            intrinsics_key=intrinsics_key,
            **kwargs,
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
    def get_keymap(cls):
        return {
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


def _build_eva_bimanual_revert_eef_frame_transform_list(
    *,
    action_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    left_cmd_wristframe: str = "left.cmd_ee_pose_wristframe",
    right_cmd_wristframe: str = "right.cmd_ee_pose_wristframe",
    left_gripper: str = "left.gripper",
    right_gripper: str = "right.gripper",
    left_obs_camframe: str = "left.obs_ee_pose_camframe",
    right_obs_camframe: str = "right.obs_ee_pose_camframe",
    left_obs_gripper: str = "left.obs_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    left_cmd_camframe: str = "left.cmd_ee_pose_camframe",
    right_cmd_camframe: str = "right.cmd_ee_pose_camframe",
) -> list[Transform]:
    """Revert wrist-frame EVA actions back to camera frame for visualization."""
    transform_list = [
        # Extract obs camframe poses from the concatenated obs key
        SplitKeys(
            input_key=obs_key,
            output_key_list=[
                (left_obs_camframe, 6),
                (left_obs_gripper, 1),
                (right_obs_camframe, 6),
                (right_obs_gripper, 1),
            ],
        ),
        # Split wrist-frame actions into per-arm chunks
        SplitKeys(
            input_key=action_key,
            output_key_list=[
                (left_cmd_wristframe, 6),
                (left_gripper, 1),
                (right_cmd_wristframe, 6),
                (right_gripper, 1),
            ],
        ),
        # Revert wrist frame → camera frame (inverse=False: target_se3 @ chunk_se3)
        ActionChunkCoordinateFrameTransform(
            target_world=left_obs_camframe,
            chunk_world=left_cmd_wristframe,
            transformed_key_name=left_cmd_camframe,
            mode="xyzypr",
            inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_obs_camframe,
            chunk_world=right_cmd_wristframe,
            transformed_key_name=right_cmd_camframe,
            mode="xyzypr",
            inverse=False,
        ),
        ConcatKeys(
            key_list=[
                left_cmd_camframe,
                left_gripper,
                right_cmd_camframe,
                right_gripper,
            ],
            new_key_name=action_key,
            delete_old_keys=True,
        ),
    ]
    return transform_list


def _build_eva_bimanual_eef_frame_transform_list(
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
    left_obs_camframe: str = "left.obs_ee_pose_camframe",
    right_obs_camframe: str = "right.obs_ee_pose_camframe",
    left_cmd_wristframe: str = "left.cmd_ee_pose_wristframe",
    right_cmd_wristframe: str = "right.cmd_ee_pose_wristframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
    extrinsics_key: str = "x5Dec13_2",
    is_quat: bool = True,
) -> list[Transform]:
    """EVA bimanual transform pipeline with actions expressed relative to the
    current EEF pose (wrist frame), analogous to keypoints relative to wrist pose."""
    extrinsics = EXTRINSICS[extrinsics_key]
    left_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["left"][None, :])[0]
    right_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["right"][None, :])[0]
    left_extra_batch_key = {"left_extrinsics_pose": left_extrinsics_pose}
    right_extra_batch_key = {"right_extrinsics_pose": right_extrinsics_pose}

    mode = "xyzwxyz" if is_quat else "xyzypr"

    # Step 1: transform cmd and obs into camera frame using extrinsics
    transform_list = [
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
        PoseCoordinateFrameTransform(
            target_world=left_target_world,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_camframe,
            mode=mode,
        ),
        PoseCoordinateFrameTransform(
            target_world=right_target_world,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_camframe,
            mode=mode,
        ),
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
        # Step 2: transform camera-frame actions into EEF-relative (wrist) frame
        ActionChunkCoordinateFrameTransform(
            target_world=left_obs_camframe,
            chunk_world=left_cmd_camframe,
            transformed_key_name=left_cmd_wristframe,
            mode=mode,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_obs_camframe,
            chunk_world=right_cmd_camframe,
            transformed_key_name=right_cmd_wristframe,
            mode=mode,
        ),
    ]

    if is_quat:
        transform_list.append(
            XYZWXYZ_to_XYZYPR(
                keys=[
                    left_cmd_wristframe,
                    right_cmd_wristframe,
                    left_obs_camframe,
                    right_obs_camframe,
                ]
            )
        )

    transform_list.extend(
        [
            ConcatKeys(
                key_list=[
                    left_cmd_wristframe,
                    left_gripper,
                    right_cmd_wristframe,
                    right_gripper,
                ],
                new_key_name=actions_key,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[
                    left_obs_camframe,
                    left_obs_gripper,
                    right_obs_camframe,
                    right_obs_gripper,
                ],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(
                keys_to_delete=[
                    left_cmd_world,
                    right_cmd_world,
                    left_obs_pose,
                    right_obs_pose,
                    left_cmd_camframe,
                    right_cmd_camframe,
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
    )
    return transform_list


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
) -> list[Transform]:
    """Canonical EVA bimanual transform pipeline used by tests and notebooks."""
    extrinsics = EXTRINSICS[extrinsics_key]
    left_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["left"][None, :])[0]
    right_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["right"][None, :])[0]
    left_extra_batch_key = {"left_extrinsics_pose": left_extrinsics_pose}
    right_extra_batch_key = {"right_extrinsics_pose": right_extrinsics_pose}

    mode = "xyzwxyz" if is_quat else "xyzypr"
    transform_list = [
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

    if is_quat:
        transform_list.append(
            XYZWXYZ_to_XYZYPR(
                keys=[
                    left_cmd_camframe,
                    right_cmd_camframe,
                    left_obs_pose,
                    right_obs_pose,
                ]
            )
        )

    transform_list.extend(
        [
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
    )
    return transform_list
