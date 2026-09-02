from __future__ import annotations

from typing import Literal

import numpy as np

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
    transforms_for_rotation_mode,
)
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz
from egomimic.utils.viz_utils import _viz_annotations


def _flatten_annotations(raw) -> list[str]:
    """Batch annotation entries -> flat list of non-empty strings."""
    if raw is None:
        return []
    out: list[str] = []
    stack = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
        elif isinstance(item, str) and item.strip():
            out.append(item)
    return out


class Yam(Embodiment):
    """The two-arm YAM teleoperation station behind ABC-130k.

    Episodes are produced by ``egomimic/scripts/abc_process/abc_to_zarr.py``.
    Structurally this is close to :class:`~egomimic.rldb.embodiment.eva.Eva` --
    bimanual, parallel-jaw, per-arm ``obs/cmd_ee_pose`` (XYZWXYZ) plus
    ``obs/cmd_gripper``, a front camera and two wrist cameras -- with three
    differences that matter downstream:

    1. **Extrinsics come from the station model, not the data.** ABC's MCAP
       carries camera *intrinsics* only; :attr:`EXTRINSICS` is recovered from the
       published rig (see that attribute) and covers the RealSense D405 station
       only. So the ``camframe`` modes, and every overlay that projects
       through K, work exactly as they do for Eva -- but only on those episodes.
       The ``eef_frame`` modes need no extrinsics at all: actions
       are a delta relative to the current EEF pose, and a rigid frame change
       applied to both operands cancels
       (``(E^-1 T_obs)^-1 (E^-1 T_cmd) == T_obs^-1 T_cmd``), so Eva's
       camera-frame hop is a no-op for the action chunk. Those are the modes to
       use on a ZED-X episode, which has no published rig. Note their proprio
       then differs from Eva's: a station-world-frame EEF pose rather than a
       camera-frame one, which matters when cotraining against Eva.
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

    # world(left_base) <- top_camera, for the RealSense D405 station.
    #
    # The translation's y and z are REFINED from the published nominal by
    # +0.0413 m and +0.0133 m (4.3 cm total). Projecting with the nominal values
    # put the EE about 18 px left of the gripper in the top-camera image,
    # consistently for both arms and at every arm pose -- the signature of a
    # small rigid mount offset, which is expected: the bracket is bolted to a
    # gantry crossbar and the model gives design intent, not per-station
    # calibration. The refinement was fit to two hand-marked gripper positions
    # and cuts the reprojection residual from ~18 px to ~4 px. Rotation is
    # untouched; it reproduces the published values exactly.
    #
    # This is therefore a per-station correction. A different D405 station may
    # need a different one, and re-fitting is the way to get it.
    #
    # ABC's own MCAP records no extrinsics, but the station this data was
    # collected on is published: i2rt-robotics/i2rt, robot_models/station/
    # yam_station_{crank,linear}_4310_d405. Composing that MJCF's
    # top_camera_bracket -> top_camera_body -> top_camera chain reproduces the
    # extrinsics its README documents, including the stated sanity check (the
    # optical axis meets the base plane at (0.384, -0.305, 0), 60 degrees below
    # horizontal). The chain is byte-identical between the crank and linear
    # stations, so the gripper variant does not matter.
    #
    # ABC's world frame is that model's `left_base`: its documented
    # left_base -> right_base offset of 0.61m matches the arm separation in the
    # data, and projecting ABC's EE poses through this transform lands them on
    # the grippers in the top-camera image.
    #
    # ONLY VALID FOR THE REALSENSE D405 STATION. ZED-X episodes use a different
    # camera and rig, and i2rt publishes no model for it; the converter attaches
    # this only when the episode's metadata reports a D405 top camera.
    TOP_CAMERA_D405 = np.array(
        [
            [-0.000003673, -0.866026618,  0.499997896, -0.166494880],
            [-1.000000000,  0.000003181, -0.000001837, -0.263749126],
            [ 0.000000000, -0.499997896, -0.866026618,  0.967579819],
            [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]
        ]
    )
    EXTRINSICS = {"front_1": TOP_CAMERA_D405}

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
        # action_mode="arc_tokenizer_cartesian". See eva.py for the layout.
        min_distance_unit: float = 0.60,
        resampled_vector_length: int = 20,
    ) -> list[Transform]:
        """``action_mode`` is the action layout; ``coord_frame`` is where poses
        live; ``rotation_mode`` is how rotation is stored.

        ``camframe`` puts poses in the TOP-CAMERA frame via :attr:`EXTRINSICS`,
        the analogue of Eva's cam-frame mode. Use this when you want
        projectable poses (the ``mode="traj"`` overlays).

        ``world`` keeps poses in the raw station base frame. NOTE the
        difference from Eva: on Eva the base IS the front camera, so cam-frame
        is still projectable with the front K. On YAM the top camera sits
        ~0.95m above the base and 60 degrees off horizontal, so world-frame
        poses are NOT projectable without going through EXTRINSICS -- use
        ``camframe`` for anything that draws on the image.

        ``eef_frame`` expresses actions as a delta from the current EEF pose.
        That is extrinsics-independent, so it works on any station, including
        the wide-camera episodes that carry no published rig.

        Geometric hops always run in xyz+quat; ``rotation_mode`` then converts
        rotation to euler (xyz+ypr, 14D), quat (16D), or Zhou 6D (20D).

        ``arc_tokenizer_cartesian`` runs the cartesian pipeline for the chosen
        frame and then rewrites ``actions_cartesian`` to (M+1, 14) arc-length
        tokens, the same layout Eva and Human emit -- Yam is already 14D with a
        real gripper, so no padding step is needed.
        """
        if action_mode not in ("cartesian", "arc_tokenizer_cartesian"):
            raise ValueError(f"unknown action_mode {action_mode!r}")
        if coord_frame == "camframe":
            transform_list = _build_yam_bimanual_camframe_transform_list(
                rotation_mode=rotation_mode, to_camera_frame=True
            )
        elif coord_frame == "world":
            transform_list = _build_yam_bimanual_camframe_transform_list(
                rotation_mode=rotation_mode, to_camera_frame=False
            )
        elif coord_frame == "eef_frame":
            transform_list = _build_yam_bimanual_eef_frame_transform_list(
                rotation_mode=rotation_mode
            )
        else:
            raise ValueError(f"unknown coord_frame {coord_frame!r}")
        if action_mode == "arc_tokenizer_cartesian":
            from egomimic.rldb.embodiment.eva import _append_arc_tokenizer

            return _append_arc_tokenizer(
                transform_list,
                min_distance_unit=min_distance_unit,
                resampled_vector_length=resampled_vector_length,
            )
        return transform_list

    @classmethod
    def viz_wristframe_batch(
        cls,
        batch,
        mode: str = "traj+rotation",
        annotation_key: str | None = "annotations",
        is_quat: bool = False,
        **kwargs,
    ):
        """Overlay WRIST-FRAME actions, plus their annotation, on the frame.

        This is the visual check on the training representation itself: it takes
        the wrist-frame action chunk a policy actually predicts, reverts it onto
        the current EEF pose, maps it into the top-camera frame and projects it.
        The ordinary :meth:`viz_transformed_batch` expects camera-frame actions
        and so only works with the ``cartesian`` modes.

        The base ``viz`` modes are mutually exclusive (``"annotations"``
        *replaces* the trajectory), so the annotation is drawn as a second pass
        on top of the rendered trajectory rather than instead of it.
        """
        vis = cls.viz_transformed_batch(
            batch,
            mode=mode,
            transform_list=cls.get_revert_transform_list(
                is_quat=is_quat, to_camera_frame=True
            ),
            **kwargs,
        )
        texts = _flatten_annotations(batch.get(annotation_key) if annotation_key else None)
        if texts:
            vis = _viz_annotations(image=vis, annotations=texts)
        return vis

    @staticmethod
    def get_revert_transform_list(
        is_quat: bool = False, to_camera_frame: bool = False
    ) -> list[Transform]:
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
        return _build_yam_bimanual_revert_eef_frame_transform_list(
            is_quat=is_quat, to_camera_frame=to_camera_frame
        )

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

    @classmethod
    def _get_keymap(cls, keymap_mode: str):
        """Mirrors Eva's keymap: the zarr keys the converter writes are the same.

        Camera key naming differs by algo:
          "cartesian"    -> dataset-style names (HPT and friends)
          "cartesian_pi" -> PI/PaliGemma-style names (base_0_rgb, ...)
        """
        if keymap_mode == "cartesian_pi":
            front_key = "base_0_rgb"
            right_wrist_key = "right_wrist_0_rgb"
            left_wrist_key = "left_wrist_0_rgb"
        else:
            front_key = cls.VIZ_IMAGE_KEY
            right_wrist_key = "observations.images.right_wrist_img"
            left_wrist_key = "observations.images.left_wrist_img"

        horizon = 45

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
                "horizon": horizon,
            },
            "left.cmd_gripper": {
                "key_type": "action_keys",
                "zarr_key": "left.cmd_gripper",
                "horizon": horizon,
            },
            "right.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "right.cmd_ee_pose",
                "horizon": horizon,
            },
            "left.cmd_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "left.cmd_ee_pose",
                "horizon": horizon,
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
    rotation_mode: Literal["euler", "quat", "6D"] = "euler",
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

    transform_list.extend(
        transforms_for_rotation_mode(
            keys=[
                left_cmd_wristframe,
                right_cmd_wristframe,
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
    to_camera_frame: bool = False,
    top_camera_pose: str = "top_camera_pose",
) -> list[Transform]:
    """Revert wrist-frame YAM actions back to absolute poses.

    Lands in the station world frame; with ``to_camera_frame`` it is carried on
    into the top-camera frame via :attr:`Yam.EXTRINSICS` so it can be projected.
    """
    pose_shape = 7 if is_quat else 6
    mode = "xyzwxyz" if is_quat else "xyzypr"
    out = [
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
    ]

    left_out, right_out = left_cmd_world, right_cmd_world
    if to_camera_frame:
        cam_pose = _matrix_to_xyzwxyz(Yam.EXTRINSICS["front_1"][None, :])[0]
        extra = {top_camera_pose: cam_pose}
        left_out, right_out = "left.cmd_ee_pose_camframe", "right.cmd_ee_pose_camframe"
        out += [
            ActionChunkCoordinateFrameTransform(
                target_world=top_camera_pose,
                chunk_world=left_cmd_world,
                transformed_key_name=left_out,
                extra_batch_key=extra,
                mode=mode,
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=top_camera_pose,
                chunk_world=right_cmd_world,
                transformed_key_name=right_out,
                extra_batch_key=extra,
                mode=mode,
            ),
        ]

    out.append(
        ConcatKeys(
            key_list=[left_out, left_cmd_gripper, right_out, right_cmd_gripper],
            new_key_name=action_key,
            delete_old_keys=True,
        )
    )
    return out


def _build_yam_bimanual_camframe_transform_list(
    *,
    top_camera_pose: str = "top_camera_pose",
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
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 1,
    rotation_mode: Literal["euler", "quat", "6D"] = "euler",
    to_camera_frame: bool = True,
) -> list[Transform]:
    """YAM bimanual pipeline with poses in the top-camera frame.

    With ``to_camera_frame=False`` the extrinsic step is skipped and poses stay
    in the raw station base frame (``coord_frame="world"``).
    Unlike Eva, that output is NOT projectable with the front K, because YAM's
    top camera is not co-located with the base.

    Mirrors ``_build_eva_bimanual_transform_list``. The one structural
    difference: Eva has a wrist camera per arm and so keys EXTRINSICS per arm,
    while YAM's is a single fixed overhead camera, so BOTH arms are referred to
    the same ``top_camera_pose``. Output poses are in that camera's frame
    (+X right, +Y down, +Z along the optical axis), so projecting with the
    episode's K is meaningful.
    """
    if to_camera_frame:
        cam_pose = _matrix_to_xyzwxyz(Yam.EXTRINSICS["front_1"][None, :])[0]
        extra = {top_camera_pose: cam_pose}
    else:
        # Stay in the base frame: no extrinsic hop, so the "camframe" keys are
        # just the raw ones and ConcatKeys consumes them directly.
        left_cmd_camframe, right_cmd_camframe = left_cmd_world, right_cmd_world
        left_obs_camframe, right_obs_camframe = left_obs_pose, right_obs_pose

    frame_steps = (
        [
            ActionChunkCoordinateFrameTransform(
                target_world=top_camera_pose,
                chunk_world=left_cmd_world,
                transformed_key_name=left_cmd_camframe,
                extra_batch_key=extra,
                mode="xyzwxyz",
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=top_camera_pose,
                chunk_world=right_cmd_world,
                transformed_key_name=right_cmd_camframe,
                extra_batch_key=extra,
                mode="xyzwxyz",
            ),
            PoseCoordinateFrameTransform(
                target_world=top_camera_pose,
                pose_world=left_obs_pose,
                transformed_key_name=left_obs_camframe,
                mode="xyzwxyz",
            ),
            PoseCoordinateFrameTransform(
                target_world=top_camera_pose,
                pose_world=right_obs_pose,
                transformed_key_name=right_obs_camframe,
                mode="xyzwxyz",
            ),
        ]
        if to_camera_frame
        else []
    )

    transform_list = frame_steps + [
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
                    left_obs_camframe,
                    left_obs_gripper,
                    right_obs_camframe,
                    right_obs_gripper,
                ],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(
                keys_to_delete=(
                    [
                        left_cmd_world,
                        right_cmd_world,
                        left_obs_pose,
                        right_obs_pose,
                        top_camera_pose,
                    ]
                    if to_camera_frame
                    # In world mode those ARE the concatenated keys, already
                    # consumed by ConcatKeys(delete_old_keys=True), and there is
                    # no top_camera_pose in the batch to remove.
                    else []
                )
            ),
            NumpyToTensor(keys=[actions_key, obs_key]),
        ]
    )
    return transform_list
