from __future__ import annotations

from typing import Literal

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    BatchQuaternionPoseToYPR,
    ConcatKeys,
    DeleteKeys,
    InterpolateLinear,
    InterpolatePose,
    NumpyToTensor,
    QuaternionPoseToYPR,
    SplitKeys,
    Transform,
)


class Yam(Embodiment):
    """The two-arm YAM teleoperation station behind ABC-130k.

    Episodes are produced by ``egomimic/scripts/abc_process/abc_to_zarr.py``.
    Structurally this is close to :class:`~egomimic.rldb.embodiment.eva.Eva` --
    bimanual, parallel-jaw, per-arm ``obs/cmd_ee_pose`` (XYZWXYZ) plus
    ``obs/cmd_gripper``, a front camera and two wrist cameras -- with three
    differences that matter downstream:

    1. **No extrinsics, and none needed.** ABC's MCAP carries camera
       *intrinsics* only. That does not block the wrist-frame pipeline: actions
       are a delta relative to the current EEF pose, so a rigid frame change
       applied to both operands cancels
       (``(E^-1 T_obs)^-1 (E^-1 T_cmd) == T_obs^-1 T_cmd``). Eva's camera-frame
       hop is therefore a no-op for the action chunk, and :meth:`get_transform_list`
       simply skips it and takes the delta in the station world frame.
       The one real consequence is proprio: Eva feeds a *camera-frame* EEF pose
       into ``observations.state.ee_pose`` while YAM feeds a station-world-frame
       one. Both are a fixed frame, but they are not the same frame, which
       matters when cotraining against Eva.
    2. **Per-episode intrinsics.** Two station types (RealSense 640x480 and
       ZED-X 1920x1200) with different calibration appear in the dataset, so K
       is read from each episode's ``zarr.attrs["intrinsics"]`` rather than
       being a class constant. ``INTRINSICS`` stays ``None`` and
       :meth:`Embodiment.viz` falls back to the per-batch value.
    3. **Station-anchored world frame**, right-handed and Z-up (Z up, X forward,
       Y left), fixed per station -- not an egocentric SLAM frame. There is no
       ``obs_head_pose``, ``obs_wrist_pose`` or ``obs_keypoints``.

    Joint-space (``{side}.obs_joints`` / ``{side}.cmd_joints``, 6 DoF, radians)
    is also carried by the converter; the YAM MJCF needed to do FK on it ships
    with the dataset's own release rather than with EgoVerse.
    """

    # Per-episode; read from zarr.attrs["intrinsics"]["front_1"] (see above).
    INTRINSICS = None
    # ABC records no camera poses, so there is nothing to put here.
    EXTRINSICS = None

    @staticmethod
    def get_transform_list(
        mode: Literal["cartesian_wristframe_quat", "cartesian_wristframe_ypr"] = (
            "cartesian_wristframe_quat"
        ),
    ) -> list[Transform]:
        if mode == "cartesian_wristframe_quat":
            return _build_yam_bimanual_eef_frame_transform_list(is_quat=True)
        if mode == "cartesian_wristframe_ypr":
            return _build_yam_bimanual_eef_frame_transform_list(is_quat=False)
        raise ValueError(
            f"Yam supports the wrist-frame modes only, got {mode!r}. There is no "
            "camera-frame mode: ABC records no extrinsics, so a camera-frame "
            "proprio/action representation cannot be built for this embodiment."
        )

    @staticmethod
    def get_revert_transform_list(is_quat: bool = False) -> list[Transform]:
        """Undo the wrist-frame step, recovering absolute poses for visualization.

        Mirrors ``_build_eva_bimanual_revert_eef_frame_transform_list``. Eva's
        version lands in the camera frame because Eva's proprio is camera-frame;
        YAM proprio is station-world-frame, so this lands in the station world
        frame. Those absolute poses are what the 3D trajectory plots use.

        Projecting them into the top-camera image needs a world<-cam transform,
        which ABC does not record (see :attr:`EXTRINSICS`). Set
        ``Yam.EXTRINSICS = {"front_1": T}`` if you obtain one and the image-space
        overlay becomes available.
        """
        return _build_yam_bimanual_revert_eef_frame_transform_list(is_quat=is_quat)

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        """Mirrors Eva's keymap: the zarr keys the converter writes are the same."""
        if keymap_mode == "cartesian_pi":
            front_key = "base_0_rgb"
            right_wrist_key = "right_wrist_0_rgb"
            left_wrist_key = "left_wrist_0_rgb"
        else:
            front_key = cls.VIZ_IMAGE_KEY
            right_wrist_key = "observations.images.right_wrist_img"
            left_wrist_key = "observations.images.left_wrist_img"

        return {
            front_key: {"key_type": "camera_keys", "zarr_key": "images.front_1"},
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


def _build_yam_bimanual_eef_frame_transform_list(
    *,
    left_cmd_world: str = "left.cmd_ee_pose",
    right_cmd_world: str = "right.cmd_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_obs_gripper: str = "left.obs_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    left_cmd_gripper: str = "left.cmd_gripper",
    right_cmd_gripper: str = "right.cmd_gripper",
    left_cmd_wristframe: str = "left.cmd_ee_pose_wristframe",
    right_cmd_wristframe: str = "right.cmd_ee_pose_wristframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
    is_quat: bool = True,
) -> list[Transform]:
    """YAM bimanual pipeline: actions relative to the current EEF pose (wrist frame).

    Mirrors ``_build_eva_bimanual_eef_frame_transform_list`` with Eva's step 1
    (world -> camera via ``Eva.EXTRINSICS``) dropped. That step is a no-op for
    the action chunk: step 2 takes the delta between two poses that were both
    mapped by the same rigid ``E^-1``, so ``E`` cancels and taking the delta
    directly in the station world frame is equivalent. Dropping it also means
    proprio stays in the station world frame rather than a camera frame.
    """
    transform_list = [
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
        # Actions relative to the current EEF pose, taken in the world frame.
        ActionChunkCoordinateFrameTransform(
            target_world=left_obs_pose,
            chunk_world=left_cmd_world,
            transformed_key_name=left_cmd_wristframe,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_obs_pose,
            chunk_world=right_cmd_world,
            transformed_key_name=right_cmd_wristframe,
            mode="xyzwxyz",
        ),
    ]

    if not is_quat:
        transform_list.extend(
            [
                BatchQuaternionPoseToYPR(
                    pose_key=left_cmd_wristframe, output_key=left_cmd_wristframe
                ),
                BatchQuaternionPoseToYPR(
                    pose_key=right_cmd_wristframe, output_key=right_cmd_wristframe
                ),
                QuaternionPoseToYPR(pose_key=left_obs_pose, output_key=left_obs_pose),
                QuaternionPoseToYPR(pose_key=right_obs_pose, output_key=right_obs_pose),
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
                    left_obs_pose,
                    left_obs_gripper,
                    right_obs_pose,
                    right_obs_gripper,
                ],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(keys_to_delete=[left_cmd_world, right_cmd_world]),
            NumpyToTensor(keys=[actions_key, obs_key]),
        ]
    )
    return transform_list


def _build_yam_bimanual_revert_eef_frame_transform_list(
    *,
    action_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    left_cmd_wristframe: str = "left.cmd_ee_pose_wristframe",
    right_cmd_wristframe: str = "right.cmd_ee_pose_wristframe",
    left_cmd_gripper: str = "left.cmd_gripper",
    right_cmd_gripper: str = "right.cmd_gripper",
    left_obs_world: str = "left.obs_ee_pose_world",
    right_obs_world: str = "right.obs_ee_pose_world",
    left_obs_gripper: str = "left.obs_gripper",
    right_obs_gripper: str = "right.obs_gripper",
    left_cmd_world: str = "left.cmd_ee_pose_world",
    right_cmd_world: str = "right.cmd_ee_pose_world",
    is_quat: bool = False,
) -> list[Transform]:
    """Revert wrist-frame YAM actions back to the station world frame."""
    pose_shape = 7 if is_quat else 6
    mode = "xyzwxyz" if is_quat else "xyzypr"
    return [
        SplitKeys(
            input_key=obs_key,
            output_key_list=[
                (left_obs_world, pose_shape),
                (left_obs_gripper, 1),
                (right_obs_world, pose_shape),
                (right_obs_gripper, 1),
            ],
        ),
        SplitKeys(
            input_key=action_key,
            output_key_list=[
                (left_cmd_wristframe, pose_shape),
                (left_cmd_gripper, 1),
                (right_cmd_wristframe, pose_shape),
                (right_cmd_gripper, 1),
            ],
        ),
        # inverse=False: target_se3 @ chunk_se3, i.e. re-compose onto the obs pose
        ActionChunkCoordinateFrameTransform(
            target_world=left_obs_world,
            chunk_world=left_cmd_wristframe,
            transformed_key_name=left_cmd_world,
            mode=mode,
            inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_obs_world,
            chunk_world=right_cmd_wristframe,
            transformed_key_name=right_cmd_world,
            mode=mode,
            inverse=False,
        ),
        ConcatKeys(
            key_list=[
                left_cmd_world,
                left_cmd_gripper,
                right_cmd_world,
                right_cmd_gripper,
            ],
            new_key_name=action_key,
            delete_old_keys=True,
        ),
    ]
