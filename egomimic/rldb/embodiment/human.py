from __future__ import annotations

from typing import Literal

import torch

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolatePose,
    PoseCoordinateFrameTransform,
    Reshape,
    Transform,
    XYZWXYZ_to_XYZYPR,
)

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, S3EpisodeResolver
from egomimic.utils.viz_utils import (
    _viz_batch_axes,
    _viz_batch_traj,
    _viz_batch_keypoints,
)


class Human(Embodiment):
    VIZ_INTRINSICS_KEY = "base"
    VIZ_IMAGE_KEY = "observations.images.front_img_1"
    ACTION_STRIDE = 3

    @classmethod
    def get_transform_list(cls, mode: Literal["cartesian", "keypoints"]) -> list[Transform]:
        if mode == "cartesian":
            return _build_aria_cartesian_bimanual_transform_list(stride=cls.ACTION_STRIDE)
        elif mode == "keypoints":
            return _build_aria_keypoints_bimanual_transform_list(stride=cls.ACTION_STRIDE)
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Expected one of: 'cartesian', 'keypoints'.")

    @classmethod
    def viz_transformed_batch(cls, batch, mode=""):
        image_key = cls.VIZ_IMAGE_KEY
        action_key = "actions_cartesian"
        intrinsics_key = cls.VIZ_INTRINSICS_KEY
        mode = (mode or "traj").lower()

        if mode == "traj":
            return _viz_batch_traj(
                batch=batch,
                image_key=image_key,
                action_key=action_key,
                intrinsics_key=intrinsics_key,
            )
        if mode == "axes":
            return _viz_batch_axes(
                batch=batch,
                image_key=image_key,
                action_key=action_key,
                intrinsics_key=intrinsics_key,
            )
        if mode == "keypoints":
            return _viz_batch_keypoints(
                batch=batch,
                image_key=image_key,
                action_key="actions_keypoints",
                intrinsics_key=intrinsics_key,
                edges=cls.FINGER_EDGES,
                colors=cls.FINGER_COLORS,
                edge_ranges=cls.FINGER_EDGE_RANGES,
            )

        raise ValueError(
            f"Unsupported mode '{mode}'. Expected one of: "
            f"('traj', 'axes', 'keypoints')."
        )

    @classmethod
    def get_keymap(cls, mode: Literal["cartesian", "keypoints"]):
        if mode == "cartesian":
            key_map = {
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
        elif mode == "keypoints":
            key_map = {
                cls.VIZ_IMAGE_KEY: {
                    "key_type": "camera_keys",
                    "zarr_key": "images.front_1",
                },
                "left.action_keypoints": {
                    "key_type": "action_keys",
                    "zarr_key": "left.obs_keypoints",
                    "horizon": 30,
                },
                "right.action_keypoints": {
                    "key_type": "action_keys",
                    "zarr_key": "right.obs_keypoints",
                    "horizon": 30,
                },
                "left.action_wrist_pose": {    
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_wrist_pose",
                    "horizon": 30,
                },
                "right.action_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_wrist_pose",
                    "horizon": 30,
                },
                "left.obs_keypoints": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_keypoints",
                },
                "right.obs_keypoints": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_keypoints",
                },
                "left.obs_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_wrist_pose",
                },
                "right.obs_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_wrist_pose",
                },
                "obs_head_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "obs_head_pose",
                },
            }
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Expected one of: 'cartesian', 'keypoints'.")
        return key_map


class Aria(Human):
    VIZ_INTRINSICS_KEY = "base"
    ACTION_STRIDE = 3
    FINGER_EDGES = [
        (5, 6,), (6, 7), (7, 0), # thumb
        (5, 8), (8, 9), (9, 10), (9, 1), # index
        (5, 11), (11, 12), (12, 13), (13, 2), # middle
        (5, 14), (14, 15), (15, 16), (16, 3), # ring
        (5, 17), (17, 18), (18, 19), (19, 4), # pinky
    ]
    FINGER_COLORS = {
        "thumb": (255, 100, 100),   # red
        "index": (100, 255, 100),   # green
        "middle": (100, 100, 255),  # blue
        "ring": (255, 255, 100),    # yellow
        "pinky": (255, 100, 255),   # magenta
    }
    FINGER_EDGE_RANGES = [
        ("thumb", 0, 3), ("index", 3, 6), ("middle", 6, 9),
        ("ring", 9, 12), ("pinky", 12, 15),
    ]


class Scale(Human):
    VIZ_INTRINSICS_KEY = "scale"
    ACTION_STRIDE = 1


class Mecka(Human):
    VIZ_INTRINSICS_KEY = "mecka"
    ACTION_STRIDE = 1


def _build_aria_keypoints_bimanual_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    left_keypoints_action_world: str = "left.action_keypoints",
    right_keypoints_action_world: str = "right.action_keypoints",
    left_keypoints_obs_pose: str = "left.obs_keypoints",
    right_keypoints_obs_pose: str = "right.obs_keypoints",
    left_keypoints_action_headframe: str = "left.action_keypoints_headframe",
    right_keypoints_action_headframe: str = "right.action_keypoints_headframe",
    left_keypoints_obs_headframe: str = "left.obs_keypoints_headframe",
    right_keypoints_obs_headframe: str = "right.obs_keypoints_headframe",
    left_wrist_action_world: str = "left.action_wrist_pose",
    right_wrist_action_world: str = "right.action_wrist_pose",
    left_wrist_obs_pose: str = "left.obs_wrist_pose",
    right_wrist_obs_pose: str = "right.obs_wrist_pose",
    left_wrist_action_headframe: str = "left.action_wrist_pose_headframe",
    right_wrist_action_headframe: str = "right.action_wrist_pose_headframe",
    left_wrist_obs_headframe: str = "left.obs_wrist_pose_headframe",
    right_wrist_obs_headframe: str = "right.obs_wrist_pose_headframe",
    delete_target_world: bool = True,
    chunk_length: int = 100,
    stride: int = 3,
) -> list[Transform]:
    keys_to_delete = list({
        left_keypoints_action_world,
        right_keypoints_action_world,
        left_keypoints_obs_pose,
        right_keypoints_obs_pose,
        left_wrist_action_world,
        right_wrist_action_world,
        left_wrist_obs_pose,
        right_wrist_obs_pose,
        left_keypoints_action_headframe,
        right_keypoints_action_headframe,
        left_keypoints_obs_headframe,
        right_keypoints_obs_headframe,
        left_wrist_action_headframe,
        right_wrist_action_headframe,
        left_wrist_obs_headframe,
        right_wrist_obs_headframe,
    })
    if delete_target_world:
        keys_to_delete.append(target_world)
        if target_world_is_quat:
            keys_to_delete.append(target_world_ypr)
    transform_list: list[Transform] = [
        Reshape(
            input_key=left_keypoints_action_world,
            output_key=left_keypoints_action_world,
            shape=(30, 21, 3),
        ),
        Reshape(
            input_key=right_keypoints_action_world,
            output_key=right_keypoints_action_world,
            shape=(30, 21, 3),
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=left_keypoints_action_world,
            transformed_key_name=left_keypoints_action_headframe,
            mode="xyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=right_keypoints_action_world,
            transformed_key_name=right_keypoints_action_headframe,
            mode="xyz",
        ),
        Reshape(
            input_key=left_keypoints_obs_pose,
            output_key=left_keypoints_obs_pose,
            shape=(21, 3),
        ),
        Reshape(
            input_key=right_keypoints_obs_pose,
            output_key=right_keypoints_obs_pose,
            shape=(21, 3),
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=left_keypoints_obs_pose,
            transformed_key_name=left_keypoints_obs_headframe,
            mode="xyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=right_keypoints_obs_pose,
            transformed_key_name=right_keypoints_obs_headframe,
            mode="xyz",
        ),
        Reshape(
            input_key=left_keypoints_obs_headframe,
            output_key=left_keypoints_obs_headframe,
            shape=(63,),
        ),
        Reshape(
            input_key=right_keypoints_obs_headframe,
            output_key=right_keypoints_obs_headframe,
            shape=(63,),
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_keypoints_action_headframe,
            output_action_key=left_keypoints_action_headframe,
            stride=stride,
            mode="xyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_keypoints_action_headframe,
            output_action_key=right_keypoints_action_headframe,
            stride=stride,
            mode="xyz",
        ),
        Reshape(
            input_key=left_keypoints_action_headframe,
            output_key=left_keypoints_action_headframe,
            shape=(chunk_length, 63),
        ),
        Reshape(
            input_key=right_keypoints_action_headframe,
            output_key=right_keypoints_action_headframe,
            shape=(chunk_length, 63),
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=left_wrist_action_world,
            transformed_key_name=left_wrist_action_headframe,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=right_wrist_action_world,
            transformed_key_name=right_wrist_action_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=left_wrist_obs_pose,
            transformed_key_name=left_wrist_obs_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=right_wrist_obs_pose,
            transformed_key_name=right_wrist_obs_headframe,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_wrist_action_headframe,
            output_action_key=left_wrist_action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_wrist_action_headframe,
            output_action_key=right_wrist_action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
    ]
    transform_list.extend([
        ConcatKeys(
            key_list=[left_wrist_action_headframe, left_keypoints_action_headframe, right_wrist_action_headframe, right_keypoints_action_headframe],
            new_key_name="actions_keypoints",
            delete_old_keys=True,
        ),
        ConcatKeys(
            key_list=[left_wrist_obs_headframe, left_keypoints_obs_headframe, right_wrist_obs_headframe, right_keypoints_obs_headframe],
            new_key_name="observations.state.keypoints",
            delete_old_keys=True,
        ),
        DeleteKeys(keys_to_delete=keys_to_delete),
    ])
    return transform_list

def _build_aria_cartesian_bimanual_transform_list(
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
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_pose_key,
            chunk_world=right_action_world,
            transformed_key_name=right_action_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_pose_key,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_pose_key,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_headframe,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_action_headframe,
            output_action_key=left_action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_action_headframe,
            output_action_key=right_action_headframe,
            stride=stride,
            mode="xyzwxyz",
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