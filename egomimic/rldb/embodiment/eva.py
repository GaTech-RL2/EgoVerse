from __future__ import annotations

from typing import Literal

import numpy as np

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.embodiment.human import ARIA_INTRINSICS
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
    transforms_for_rotation_mode,
)
from egomimic.utils.pose_utils import (
    _matrix_to_xyzwxyz,
)


class Eva(Embodiment):
    INTRINSICS = ARIA_INTRINSICS
    EXTRINSICS = {
        "left": np.array(
            [
                [0.01329544, -0.71757193, 0.69635749, -0.04409191],
                [-0.99959782, -0.02698416, -0.00872107, -0.23221381],
                [0.02504862, -0.69596148, -0.7176421, 0.57323278],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "right": np.array(
            [
                [-0.04733948, -0.76631195, 0.64072222, -0.01998031],
                [-0.9983006, 0.05811952, -0.00424732, 0.32539554],
                [-0.0339837, -0.63983444, -0.76776103, 0.64809634],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    }

    @staticmethod
    def get_transform_list(
        action_mode: Literal[
            "cartesian",
            "arc_tokenizer_cartesian",
        ] = "cartesian",
        coord_frame: Literal[
            "camframe",
            "world",
            "eef_frame",
        ] = "camframe",
        rotation_mode: Literal[
            "euler",
            "quat",
            "6D",
        ] = "euler",
        # Arc-tokenizer args, only consulted when
        # action_mode="arc_tokenizer_cartesian".
        # min_distance_unit = D in meters (per-arm arc length span of one token),
        # resampled_vector_length = M (number of waypoints per token). The
        # emitted sequence has M+1 rows: M waypoints followed by 1 velocity
        # token, each 14-dim (xyz + ypr + grip per arm) [L xyz(3), L ypr(3),
        # L grip(1), R xyz(3), R ypr(3), R grip(1)]. Rotation always supervised.
        min_distance_unit: float = 0.60,
        resampled_vector_length: int = 20,
    ) -> list[Transform]:
        """``action_mode`` is the action layout; ``coord_frame`` is where poses
        live; ``rotation_mode`` is how rotation is stored.

        Cam-frame actions are expressed in the wrist cameras via :attr:`EXTRINSICS`.
        ``world`` keeps poses in the raw robot base frame -- on Eva the base IS
        the front camera, so world-frame xyz still projects with the front K.
        EEF-frame actions are a delta from the current EEF pose. In all cases
        the geometric hops run in xyz+quat, then ``rotation_mode`` converts rotation
        to euler (xyz+ypr, 14D), quat (16D), or Zhou 6D (20D).

        ``arc_tokenizer_cartesian`` runs the cartesian pipeline for the chosen
        frame and then rewrites ``actions_cartesian`` to (M+1, 14) arc-length
        tokens -- xyz + ypr + grip per arm, rotation always included. Eva has a
        real gripper, so unlike Human it needs no padding step first; it is
        euler-only, because that is the layout the tokenizer consumes.
        """
        if action_mode not in ("cartesian", "arc_tokenizer_cartesian"):
            raise ValueError(f"unknown action_mode {action_mode!r}")
        builders = {
            "camframe": _build_eva_bimanual_transform_list,
            "world": _build_eva_bimanual_worldframe_transform_list,
            "eef_frame": _build_eva_bimanual_eef_frame_transform_list,
        }
        if coord_frame not in builders:
            raise ValueError(f"unknown coord_frame {coord_frame!r}")
        transform_list = builders[coord_frame](rotation_mode=rotation_mode)
        if action_mode == "arc_tokenizer_cartesian":
            return _append_arc_tokenizer(
                transform_list,
                min_distance_unit=min_distance_unit,
                resampled_vector_length=resampled_vector_length,
                rotation_mode=rotation_mode,
            )
        return transform_list


def _append_arc_tokenizer(
    transform_list: list[Transform],
    *,
    min_distance_unit: float,
    resampled_vector_length: int,
    rotation_mode: Literal["euler", "quat", "6D"] = "euler",
    dt: float | None = None,
    action_key: str = "actions_cartesian",
) -> list[Transform]:
    """Splice the arc-length tokenizer in before the final NumpyToTensor.

    The tokenizer works on numpy arrays, so it has to run before the cast;
    NumpyToTensor then converts the (M+1, 14) result to a torch tensor.

    ``rotation_mode`` must be ``euler``: the tokenizer's chunk layout is a
    hard-coded 14D ``[xyz(3), ypr(3), grip(1)] x 2``, and it SLERPs through
    the ypr slots. quat (16D) and 6D (20D) chunks are rejected here rather
    than at the first batch, where the shape check fires deep inside a run.
    """
    if rotation_mode != "euler":
        raise ValueError(
            "the arc-length tokenizer only supports rotation_mode='euler' "
            f"(its chunk layout is 14D [xyz, ypr, grip] x 2); got {rotation_mode!r}"
        )
    from egomimic.rldb.zarr.arc_length_tokenizer import (
        TokenizeBimanualArcLengthCartesian,
    )

    kwargs = {} if dt is None else {"dt": float(dt)}
    tokenize = TokenizeBimanualArcLengthCartesian(
        action_key=action_key,
        output_action_key=action_key,
        min_distance_unit=float(min_distance_unit),
        resampled_vector_length=int(resampled_vector_length),
        **kwargs,
    )
    for i in range(len(transform_list) - 1, -1, -1):
        if isinstance(transform_list[i], NumpyToTensor):
            return transform_list[:i] + [tokenize] + transform_list[i:]
    return transform_list + [tokenize]

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        # Camera key naming differs by algo:
        #   "cartesian"                -> dataset-style names (HPT and friends)
        #   "cartesian_pi"             -> PI/PaliGemma-style names (base_0_rgb, ...)
        #   "arc_tokenizer_cartesian"  -> same names as "cartesian" but with a
        #                                 wider action horizon so the arc-length
        #                                 integration has room to reach D.
        # Everything else (proprio + action) stays identical so the same
        # transform_list works either way.
        if keymap_mode == "cartesian_pi":
            front_key = "base_0_rgb"
            right_wrist_key = "right_wrist_0_rgb"
            left_wrist_key = "left_wrist_0_rgb"
        else:
            front_key = cls.VIZ_IMAGE_KEY
            right_wrist_key = "observations.images.right_wrist_img"
            left_wrist_key = "observations.images.left_wrist_img"

        # Arc-tokenizer mode needs a wider raw window so per-arm joint arc
        # length has room to reach ``min_distance_unit`` (D) before the
        # padded tail kicks in. 200 raw frames ≈ 6.7 s of eva motion at
        # 30 fps — plenty of budget for D up to ~2 m of per-arm travel at
        # a brisk 30 cm/s wrist speed.
        action_horizon = 200 if keymap_mode == "arc_tokenizer_cartesian" else 45

        key_map = {
            front_key: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
            },
            right_wrist_key: {
                "key_type": "camera_keys",
                "zarr_key": "images.right_wrist",
            },
            left_wrist_key: {
                "key_type": "camera_keys",
                "zarr_key": "images.left_wrist",
            },
            "right.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_ee_pose",
            },
            "right.obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_gripper",
            },
            "left.obs_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "left.obs_ee_pose",
            },
            "left.obs_gripper": {
                "key_type": "proprio_keys",
                "zarr_key": "left.obs_gripper",
            },
            "right.cmd_gripper": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_gripper",
                "horizon": action_horizon,
            },
            "left.cmd_gripper": {
                "key_type": "action_keys",
                "zarr_key": "left.cmd_gripper",
                "horizon": action_horizon,
            },
            "right.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_ee_pose",
                "horizon": action_horizon,
            },
            "left.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "left.cmd_ee_pose",
                "horizon": action_horizon,
            },
        }

        return key_map

    @classmethod
    def dinov3_keymap(cls):
        """
        Compact keymap for alignment training: cartesian action chunk, the
        DINOv3 image embedding produced by the embedding_process pipeline, and
        the language annotation track.
        """
        return {
            "actions_cartesian": {
                "key_type": "action_keys",
                "zarr_key": "actions_cartesian",
            },
            "dino_front_1": {
                "key_type": "proprio_keys",
                "zarr_key": "dino.front_img_1",
            },
            "annotations": {
                "key_type": "annotation_keys",
                "zarr_key": "annotations",
            },
        }


def _build_eva_bimanual_revert_eef_frame_transform_list(
    *,
    action_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    left_cmd_wristframe: str = "left.cmd_ee_pose_wristframe",
    right_cmd_wristframe: str = "right.cmd_ee_pose_wristframe",
    left_cmd_gripper: str = "left.cmd_gripper",
    right_cmd_gripper: str = "right.cmd_gripper",
    left_obs_camframe: str = "left.obs_ee_pose_camframe",
    right_obs_camframe: str = "right.obs_ee_pose_camframe",
    left_obs_gripper: str = "left.obs_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    left_cmd_camframe: str = "left.cmd_ee_pose_camframe",
    right_cmd_camframe: str = "right.cmd_ee_pose_camframe",
    is_quat: bool = True,
) -> list[Transform]:
    """Revert wrist-frame EVA actions back to camera frame for visualization."""
    if is_quat:
        pose_shape = 7
    else:
        pose_shape = 6
    transform_list = [
        # Extract obs camframe poses from the concatenated obs key
        SplitKeys(
            input_key=obs_key,
            output_key_list=[
                (left_obs_camframe, pose_shape),
                (left_obs_gripper, 1),
                (right_obs_camframe, pose_shape),
                (right_obs_gripper, 1),
            ],
        ),
        # Split wrist-frame actions into per-arm chunks
        SplitKeys(
            input_key=action_key,
            output_key_list=[
                (left_cmd_wristframe, pose_shape),
                (left_cmd_gripper, 1),
                (right_cmd_wristframe, pose_shape),
                (right_cmd_gripper, 1),
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
                left_cmd_gripper,
                right_cmd_camframe,
                right_cmd_gripper,
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
    left_cmd_gripper: str = "left.cmd_gripper",
    right_cmd_gripper: str = "right.cmd_gripper",
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
    rotation_mode: Literal["euler", "quat", "6D"] = "euler",
) -> list[Transform]:
    """EVA bimanual transform pipeline with actions expressed relative to the
    current EEF pose (wrist frame), analogous to keypoints relative to wrist pose."""
    extrinsics = Eva.EXTRINSICS
    left_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["left"][None, :])[0]
    right_extrinsics_pose = _matrix_to_xyzwxyz(extrinsics["right"][None, :])[0]
    left_extra_batch_key = {"left_extrinsics_pose": left_extrinsics_pose}
    right_extra_batch_key = {"right_extrinsics_pose": right_extrinsics_pose}

    # Step 1: transform cmd and obs into camera frame using extrinsics
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
            transformed_key_name=left_obs_camframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=right_target_world,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_camframe,
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
            action_key=left_cmd_gripper,
            output_action_key=left_cmd_gripper,
            stride=stride,
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=right_cmd_gripper,
            output_action_key=right_cmd_gripper,
            stride=stride,
        ),
        # Step 2: transform camera-frame actions into EEF-relative (wrist) frame
        ActionChunkCoordinateFrameTransform(
            target_world=left_obs_camframe,
            chunk_world=left_cmd_camframe,
            transformed_key_name=left_cmd_wristframe,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_obs_camframe,
            chunk_world=right_cmd_camframe,
            transformed_key_name=right_cmd_wristframe,
            mode="xyzwxyz",
        ),
    ]

    transform_list.extend(
        transforms_for_rotation_mode(
            keys=[
                left_cmd_wristframe,
                right_cmd_wristframe,
                left_obs_camframe,
                right_obs_camframe,
            ],
            rotation_mode=rotation_mode,
        )
    )

    transform_list.extend(
        [
            ConcatKeys(
                key_list=[
                    left_cmd_wristframe,
                    left_cmd_gripper,
                    right_cmd_wristframe,
                    right_cmd_gripper,
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
    left_cmd_gripper: str = "left.cmd_gripper",
    right_cmd_gripper: str = "right.cmd_gripper",
    left_cmd_camframe: str = "left.cmd_ee_pose_camframe",
    right_cmd_camframe: str = "right.cmd_ee_pose_camframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
    rotation_mode: Literal["euler", "quat", "6D"] = "euler",
) -> list[Transform]:
    """Canonical EVA bimanual transform pipeline used by tests and notebooks."""
    extrinsics = Eva.EXTRINSICS
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
            action_key=left_cmd_gripper,
            output_action_key=left_cmd_gripper,
            stride=stride,
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=right_cmd_gripper,
            output_action_key=right_cmd_gripper,
            stride=stride,
        ),
    ]

    transform_list.extend(
        transforms_for_rotation_mode(
            keys=[
                left_cmd_camframe,
                right_cmd_camframe,
                left_obs_pose,
                right_obs_pose,
            ],
            rotation_mode=rotation_mode,
        )
    )

    transform_list.extend(
        [
            ConcatKeys(
                key_list=[
                    left_cmd_camframe,
                    left_cmd_gripper,
                    right_cmd_camframe,
                    right_cmd_gripper,
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


def _build_eva_bimanual_worldframe_transform_list(
    *,
    left_cmd_world: str = "left.cmd_ee_pose",
    right_cmd_world: str = "right.cmd_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_obs_gripper: str = "left.obs_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    left_cmd_gripper: str = "left.cmd_gripper",
    right_cmd_gripper: str = "right.cmd_gripper",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
    rotation_mode: Literal["euler", "quat", "6D"] = "euler",
) -> list[Transform]:
    """World / front-camera frame EVA bimanual pipeline.

    Structurally mirrors ``_build_human_cartesian_bimanual_transform_list``
    (interpolate → ypr conversion → concat), but WITHOUT the
    ``ActionChunkCoordinateFrameTransform`` step because eva's raw
    ``cmd_ee_pose`` is already in the robot base frame, and the front camera
    on eva is static at the base (the zarr stores no separate front-camera
    extrinsic — this is the implicit convention). Human's pipeline needs the
    coord transform because aria stores poses in an aria-device frame that
    must be re-expressed relative to ``obs_head_pose`` (== front camera).

    Result: ``actions_cartesian`` shape ``(chunk_length, 14)`` with per-arm
    layout ``[xyz(3), ypr(3), grip(1)]`` in the base/front-cam frame — a
    single shared frame across both arms (analogous to how human actions
    live in the head frame across both arms). This makes eva and human
    action features directly comparable and lets the overlay project each
    xyz through the front-cam K without any additional frame math.
    """
    transform_list: list[Transform] = [
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_cmd_world,
            output_action_key=left_cmd_world,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_cmd_world,
            output_action_key=right_cmd_world,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=left_cmd_gripper,
            output_action_key=left_cmd_gripper,
            stride=stride,
        ),
        InterpolateLinear(
            new_chunk_length=chunk_length,
            action_key=right_cmd_gripper,
            output_action_key=right_cmd_gripper,
            stride=stride,
        ),
    ]
    transform_list.extend(
        transforms_for_rotation_mode(
            keys=[
                left_cmd_world,
                right_cmd_world,
                left_obs_pose,
                right_obs_pose,
            ],
            rotation_mode=rotation_mode,
        )
    )
    transform_list.extend(
        [
            ConcatKeys(
                key_list=[
                    left_cmd_world,
                    left_cmd_gripper,
                    right_cmd_world,
                    right_cmd_gripper,
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
            NumpyToTensor(keys=[actions_key, obs_key]),
        ]
    )
    return transform_list
