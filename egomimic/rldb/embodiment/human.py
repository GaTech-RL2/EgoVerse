from __future__ import annotations

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.embodiment.eva import (
    _viz_batch_palm_axes,
    _viz_batch_palm_traj,
)
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolatePose,
    PoseCoordinateFrameTransform,
    Transform,
    XYZWXYZ_to_XYZYPR,
)


class Human(Embodiment):
    VIZ_INTRINSICS_KEY = "base"
    VIZ_IMAGE_KEY = "observations.images.front_img_1"
    ACTION_STRIDE = 3

    @classmethod
    def get_transform_list(cls) -> list[Transform]:
        return _build_aria_bimanual_transform_list(stride=cls.ACTION_STRIDE)

    @classmethod
    def viz_transformed_batch(cls, batch, mode=""):
        image_key = cls.VIZ_IMAGE_KEY
        action_key = "actions_cartesian"
        intrinsics_key = cls.VIZ_INTRINSICS_KEY
        mode = (mode or "palm_traj").lower()

        if mode == "palm_traj":
            return _viz_batch_palm_traj(
                batch=batch,
                image_key=image_key,
                action_key=action_key,
                intrinsics_key=intrinsics_key,
            )
        if mode == "palm_axes":
            return _viz_batch_palm_axes(
                batch=batch,
                image_key=image_key,
                action_key=action_key,
                intrinsics_key=intrinsics_key,
            )
        if mode == "keypoints":
            raise NotImplementedError(
                "mode='keypoints' is reserved and not implemented yet."
            )

        raise ValueError(
            f"Unsupported mode '{mode}'. Expected one of: "
            f"('palm_traj', 'palm_axes', 'keypoints')."
        )

    @classmethod
    def get_keymap(cls):
        return {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            "right.action_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "right.obs_ee_pose",
                "horizon": 30,
            },
            "left.action_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "left.obs_ee_pose",
                "horizon": 30,
            },
            "right.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_ee_pose",
            },
            "left.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "left.obs_ee_pose",
            },
            "obs_head_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "obs_head_pose",
            },
        }


class Aria(Human):
    VIZ_INTRINSICS_KEY = "base"
    ACTION_STRIDE = 3


class Scale(Human):
    VIZ_INTRINSICS_KEY = "scale"
    ACTION_STRIDE = 1


class Mecka(Human):
    VIZ_INTRINSICS_KEY = "mecka"
    ACTION_STRIDE = 1


def _build_aria_bimanual_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    left_action_world: str = "left.action_ee_pose",
    right_action_world: str = "right.action_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_action_headframe: str = "left.action_ee_pose_headframe",
    right_action_headframe: str = "right.action_ee_pose_headframe",
    left_obs_headframe: str = "left.obs_ee_pose_headframe",
    right_obs_headframe: str = "right.obs_ee_pose_headframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 3,
    delete_target_world: bool = True,
) -> list[Transform]:
    """Canonical ARIA bimanual transform pipeline used by tests and notebooks.

    Aria human data does not have commanded ee poses; action chunks are built
    from stacked observed ee poses (typically with a horizon on
    ``left/right.action_ee_pose`` mapped from ``left/right.obs_ee_pose``).
    """
    keys_to_delete = list(
        {
            left_action_world,
            right_action_world,
            left_obs_pose,
            right_obs_pose,
        }
    )
    target_pose_key = target_world
    if delete_target_world:
        keys_to_delete.append(target_world)
        if target_world_is_quat:
            keys_to_delete.append(target_world_ypr)

    transform_list: list[Transform] = [
        ActionChunkCoordinateFrameTransform(
            target_world=target_pose_key,
            chunk_world=left_action_world,
            transformed_key_name=left_action_headframe,
            is_quat=target_world_is_quat,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_pose_key,
            chunk_world=right_action_world,
            transformed_key_name=right_action_headframe,
            is_quat=target_world_is_quat,
        ),
        PoseCoordinateFrameTransform(
            target_world=target_pose_key,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_headframe,
            is_quat=target_world_is_quat,
        ),
        PoseCoordinateFrameTransform(
            target_world=target_pose_key,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_headframe,
            is_quat=target_world_is_quat,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_action_headframe,
            output_action_key=left_action_headframe,
            stride=stride,
            is_quat=target_world_is_quat,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_action_headframe,
            output_action_key=right_action_headframe,
            stride=stride,
            is_quat=target_world_is_quat,
        ),
    ]

    if target_world_is_quat:
        transform_list.append(
            XYZWXYZ_to_XYZYPR(
                keys=[
                    left_action_headframe,
                    right_action_headframe,
                    left_obs_headframe,
                    right_obs_headframe,
                ]
            )
        )

    transform_list.extend(
        [
            ConcatKeys(
                key_list=[left_action_headframe, right_action_headframe],
                new_key_name=actions_key,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[left_obs_headframe, right_obs_headframe],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(keys_to_delete=keys_to_delete),
        ]
    )
    return transform_list
