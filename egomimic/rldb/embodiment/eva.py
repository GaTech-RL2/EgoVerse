from __future__ import annotations

from typing import Literal

import numpy as np

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    BatchQuaternionPoseToYPR,
    CartesianRot6DToYPR,
    CartesianYPRToRot6D,
    ConcatKeys,
    DeleteKeys,
    InterpolateLinear,
    InterpolatePose,
    NumpyToTensor,
    PoseCoordinateFrameTransform,
    QuaternionPoseToYPR,
    SplitKeys,
    Transform,
    XYZWXYZ_to_XYZYPR,
)
from egomimic.utils.pose_utils import (
    _matrix_to_xyzwxyz,
)

# Per-rig extrinsics registry (camera pose; the frame transforms invert
# internally via target_world semantics). Mirrors the remote wristframe-6d
# repo's egomimicUtils EXTRINSICS registry.
#   x5Dec13_2:    per-arm hand-eye calib of the rl2 lab x5 eva rig (Dec 13) —
#                 the physical robot the rollout stack drives (its rollout
#                 configs bake this key). WRONG for ABC data: the right matrix
#                 throws the right arm off-screen on abc episodes.
#   abc_fold_viz: ABC-130k top camera — solvePnP least-squares fit over 112
#                 clicked EE points across 60 episodes / 7 tasks (~26px
#                 median, 41px RMSE, 99% in-frame). One SHARED world->cam
#                 matrix for both arms (single top camera over one world
#                 frame). ABC is multi-station, so this is a best-effort
#                 global average; crisp overlays need per-episode calibration.
EVA_EXTRINSICS = {
    "x5Dec13_2": {
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
    },
}
_ABC_FOLD_VIZ = np.array(
    [
        [-0.00540728648402, -0.91864395735192, 0.39504941573641, -0.07230219027034],
        [-0.99924104589933, -0.01027605228132, -0.03757306135427, -0.23692835817344],
        [0.03857581422213, -0.39495275966919, -0.91789118319482, 0.97783070025080],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
EVA_EXTRINSICS["abc_fold_viz"] = {"left": _ABC_FOLD_VIZ, "right": _ABC_FOLD_VIZ}

# ABC-130k RealSense top camera K (640x480 space). Replaces the old
# ARIA_INTRINSICS fallback (fx=266.5), which mis-scaled the eva eval overlay
# ~1.6x — same fix as the remote repo's Eva.VIZ_INTRINSICS_KEY="eva". Only
# consulted when an episode carries no per-episode K (NaN sentinel in
# batch["intrinsics"]).
EVA_INTRINSICS = np.array(
    [
        [436.26, 0.0, 310.10, 0.0],
        [0.0, 435.13, 241.93, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
)


class Eva(Embodiment):
    INTRINSICS = EVA_INTRINSICS
    # Legacy default (rollout rig calib); data configs for ABC data should
    # pass extrinsics_key="abc_fold_viz" to get_transform_list instead.
    EXTRINSICS = EVA_EXTRINSICS["x5Dec13_2"]

    @staticmethod
    def get_transform_list(
        mode: Literal[
            "cartesian",
            "cartesian_6d",
            "cartesian_wristframe_ypr",
            "cartesian_wristframe_6d",
            "cartesian_wristframe_quat",
        ],
        extrinsics_key: str = "x5Dec13_2",
    ) -> list[Transform]:
        # extrinsics_key selects the cam<-base transform baked into the
        # cartesian conversion — it sets ONLY the cam frame the proprio/obs
        # live in (and hence the eval-viz revert frame); wrist-relative action
        # targets are invariant to it. ABC data must use "abc_fold_viz".
        if mode == "cartesian":
            return _build_eva_bimanual_transform_list(
                is_quat=True, extrinsics_key=extrinsics_key
            )
        elif mode == "cartesian_6d":
            # Camera-frame cartesian (14D xyz+ypr+gripper per arm) with the
            # rotation re-expressed as the continuous 6D representation
            # (20D xyz+6d+gripper per arm) for pi0.5 normalized-rot6d encoding.
            # The proprio ee_pose is 6D-encoded too: normalized YPR proprio
            # saturates yaw/roll at ±π (wraparound), so per-dim normalization
            # is only meaningful on the continuous rep — same fix as actions.
            return _build_eva_bimanual_transform_list(
                is_quat=True, extrinsics_key=extrinsics_key
            ) + [
                CartesianYPRToRot6D(action_key="actions_cartesian"),
                CartesianYPRToRot6D(action_key="observations.state.ee_pose"),
            ]
        elif mode == "cartesian_wristframe_ypr":
            return _build_eva_bimanual_eef_frame_transform_list(
                is_quat=False, extrinsics_key=extrinsics_key
            )
        elif mode == "cartesian_wristframe_6d":
            # Wrist-frame cartesian (14D xyz+ypr+gripper per arm) with the
            # rotation re-expressed as the continuous 6D representation
            # (20D) for pi0.5 normalized-rot6d encoding. The cam-frame proprio
            # ee_pose is 6D-encoded too (see cartesian_6d) — extra important
            # here since the proprio is the only cam-frame signal the model
            # sees with wrist-relative action targets.
            return _build_eva_bimanual_eef_frame_transform_list(
                is_quat=False, extrinsics_key=extrinsics_key
            ) + [
                CartesianYPRToRot6D(action_key="actions_cartesian"),
                CartesianYPRToRot6D(action_key="observations.state.ee_pose"),
            ]
        elif mode == "cartesian_wristframe_quat":
            return _build_eva_bimanual_eef_frame_transform_list(
                is_quat=True, extrinsics_key=extrinsics_key
            )

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        # Camera key naming differs by algo:
        #   "cartesian"     -> dataset-style names (HPT and friends)
        #   "cartesian_pi"  -> PI/PaliGemma-style names (base_0_rgb, *_wrist_0_rgb)
        # Everything else (proprio + action) stays identical so the same
        # transform_list ("cartesian") works either way.
        if keymap_mode == "cartesian_pi":
            front_key = "base_0_rgb"
            right_wrist_key = "right_wrist_0_rgb"
            left_wrist_key = "left_wrist_0_rgb"
        else:
            front_key = cls.VIZ_IMAGE_KEY
            right_wrist_key = "observations.images.right_wrist_img"
            left_wrist_key = "observations.images.left_wrist_img"

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
                "horizon": 45,
            },
            "left.cmd_gripper": {
                "key_type": "action_keys",
                "zarr_key": "left.cmd_gripper",
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


def _build_eva_cartesian_revert_6d_transform_list(
    *,
    action_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
) -> list[Transform]:
    """Revert camera-frame 6D-rotation EVA cartesian actions back to ypr.

    Used by the cam-frame 6D evaluator: the action chunk is already in camera
    frame (produced by the ``cartesian_6d`` transform mode), so only the
    rotation representation is converted from xyz+6D (+gripper, 10/arm) back to
    xyz+ypr (+gripper, 7/arm) so cam-frame MSE and the viz video see the same
    ypr layout as the plain ``cartesian`` mode. The proprio ee_pose (also
    6D-encoded by the ``cartesian_6d`` mode) is reverted the same way.
    """
    return [
        CartesianRot6DToYPR(action_key=action_key),
        CartesianRot6DToYPR(action_key=obs_key),
    ]


def _build_eva_cartesian_revert_6d_wristframe_transform_list(
    *,
    action_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
) -> list[Transform]:
    """Revert wrist-frame 6D-rotation EVA actions back to camera-frame ypr.

    Three stages for the cam-frame 6D wristframe evaluator: (1) convert the
    action rotation from xyz+6D (+gripper) back to xyz+ypr (+gripper) via
    ``CartesianRot6DToYPR`` (Gram-Schmidt re-orthonormalizes the possibly
    non-orthonormal model prediction); (2) likewise revert the proprio
    ``observations.state.ee_pose`` (6D-encoded by the ``cartesian_wristframe_6d``
    mode) back to ypr; (3) project the wrist-frame ypr actions back into camera
    frame using the standard eef-frame revert, which reads that ypr proprio to
    define the frame.
    """
    return [
        CartesianRot6DToYPR(action_key=action_key),
        CartesianRot6DToYPR(action_key=obs_key),
        *_build_eva_bimanual_revert_eef_frame_transform_list(is_quat=False),
    ]


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
    is_quat: bool = True,
    extrinsics_key: str = "x5Dec13_2",
) -> list[Transform]:
    """EVA bimanual transform pipeline with actions expressed relative to the
    current EEF pose (wrist frame), analogous to keypoints relative to wrist pose."""
    extrinsics = EVA_EXTRINSICS[extrinsics_key]
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

    if not is_quat:
        transform_list.extend(
            [
                BatchQuaternionPoseToYPR(
                    pose_key=left_cmd_wristframe,
                    output_key=left_cmd_wristframe,
                ),
                BatchQuaternionPoseToYPR(
                    pose_key=right_cmd_wristframe,
                    output_key=right_cmd_wristframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=left_obs_camframe,
                    output_key=left_obs_camframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=right_obs_camframe,
                    output_key=right_obs_camframe,
                ),
            ]
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
    is_quat: bool = True,
    extrinsics_key: str = "x5Dec13_2",
) -> list[Transform]:
    """Canonical EVA bimanual transform pipeline used by tests and notebooks."""
    extrinsics = EVA_EXTRINSICS[extrinsics_key]
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
