from __future__ import annotations

from abc import abstractmethod
from typing import Literal

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    BatchQuaternionPoseToYPR,
    ConcatKeys,
    DeleteKeys,
    InterpolatePose,
    PadGripperZeros,
    PoseCoordinateFrameTransform,
    QuaternionPoseToYPR,
    Reshape,
    SplitKeys,
    Transform,
    XYZ6D_to_XYZYPR,
    XYZWXYZ_to_XYZ6D,
    XYZWXYZ_to_XYZYPR,
)
from egomimic.utils.viz_utils import (
    ColorPalette,
    _viz_keypoints,
)


class Human(Embodiment):
    ACTION_STRIDE = 3
    # MANO 21-keypoint topology: 0=wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
    # Subclasses with non-MANO conventions (e.g. Aria) override these.
    FINGER_EDGES = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # thumb
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),  # index
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),  # middle
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),  # ring
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),  # pinky
    ]
    FINGER_COLORS = {
        "thumb": (255, 100, 100),
        "index": (100, 255, 100),
        "middle": (100, 100, 255),
        "ring": (255, 255, 100),
        "pinky": (255, 100, 255),
    }
    FINGER_EDGE_RANGES = [
        ("thumb", 0, 4),
        ("index", 4, 8),
        ("middle", 8, 12),
        ("ring", 12, 16),
        ("pinky", 16, 20),
    ]
    DOT_COLOR = (255, 165, 0)

    @classmethod
    def viz(
        cls,
        image,
        viz_data,
        mode=Literal["traj", "traj+rotation", "axes", "annotations", "keypoints"],
        intrinsics_key=None,
        **kwargs,
    ):
        if mode == "keypoints":
            intrinsics_key = intrinsics_key or cls.VIZ_INTRINSICS_KEY
            color = kwargs.get("color", None)
            if color is not None and ColorPalette.is_valid(color):
                n = len(cls.FINGER_COLORS)
                colors = {
                    finger: ColorPalette.to_rgb(color, value=(i + 1) / (n + 1))
                    for i, finger in enumerate(cls.FINGER_COLORS)
                }
                dot_color = ColorPalette.to_rgb(color, value=0.7)
            else:
                colors = cls.FINGER_COLORS
                dot_color = cls.DOT_COLOR
            return _viz_keypoints(
                image=image,
                actions=viz_data,
                intrinsics_key=intrinsics_key,
                edges=cls.FINGER_EDGES,
                edge_ranges=cls.FINGER_EDGE_RANGES,
                colors=colors,
                dot_color=dot_color,
                **kwargs,
            )
        return super().viz(
            image, viz_data, mode=mode, intrinsics_key=intrinsics_key, **kwargs
        )

    @abstractmethod
    def _get_keymap(
        cls, mode: Literal["cartesian", "keypoints"], annotation_key: str = None
    ):
        pass

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal[
            "cartesian",
            "cartesian_padded",
            "cartesian_6d",
            "cartesian_wristframe_ypr",
            "cartesian_wristframe_6d",
            "keypoints_headframe_ypr",
            "keypoints_headframe_quat",
            "keypoints_wristframe_ypr",
            "keypoints_wristframe_quat",
        ],
    ) -> list[Transform]:
        if mode == "cartesian":
            return _build_human_cartesian_bimanual_transform_list(
                stride=cls.ACTION_STRIDE
            )
        if mode == "cartesian_padded":
            return _build_human_cartesian_bimanual_transform_list(
                stride=cls.ACTION_STRIDE
            ) + [PadGripperZeros(action_key="actions_cartesian")]
        if mode == "cartesian_6d":
            return _build_human_cartesian_bimanual_transform_list(
                stride=cls.ACTION_STRIDE, rot_repr="6d"
            )
        if mode == "cartesian_wristframe_ypr":
            return _build_human_cartesian_eef_frame_transform_list(
                stride=cls.ACTION_STRIDE
            )
        if mode == "cartesian_wristframe_6d":
            return _build_human_cartesian_eef_frame_transform_list(
                stride=cls.ACTION_STRIDE, rot_repr="6d"
            )
        if mode == "keypoints_headframe_ypr":
            return _build_human_keypoints_bimanual_transform_list(
                stride=cls.ACTION_STRIDE, is_quat=False
            )
        if mode == "keypoints_headframe_quat":
            return _build_human_keypoints_bimanual_transform_list(
                stride=cls.ACTION_STRIDE, is_quat=True
            )
        if mode == "keypoints_wristframe_ypr":
            return _build_human_keypoints_eef_frame_transform_list(
                stride=cls.ACTION_STRIDE, is_quat=False
            )
        if mode == "keypoints_wristframe_quat":
            return _build_human_keypoints_eef_frame_transform_list(
                stride=cls.ACTION_STRIDE, is_quat=True
            )
        raise ValueError(f"Unsupported transform_list mode '{mode}' for {cls.__name__}")


class Aria(Human):
    VIZ_INTRINSICS_KEY = "base"
    ACTION_STRIDE = 3
    # Aria's 21-keypoint layout is NOT MANO: 0-4 are fingertips, 5 is the palm root.
    FINGER_EDGES = [
        (5, 6),
        (6, 7),
        (7, 0),  # thumb
        (5, 8),
        (8, 9),
        (9, 10),
        (10, 1),  # index
        (5, 11),
        (11, 12),
        (12, 13),
        (13, 2),  # middle
        (5, 14),
        (14, 15),
        (15, 16),
        (16, 3),  # ring
        (5, 17),
        (17, 18),
        (18, 19),
        (19, 4),  # pinky
    ]
    FINGER_EDGE_RANGES = [
        ("thumb", 0, 3),
        ("index", 3, 7),
        ("middle", 7, 11),
        ("ring", 11, 15),
        ("pinky", 15, 19),
    ]

    @classmethod
    def _get_keymap(
        cls,
        keymap_mode: Literal["cartesian", "cartesian_pi", "keypoints"],
    ):
        if keymap_mode in ("cartesian", "cartesian_pi"):
            # cartesian_pi uses Pi/PaliGemma-style image key naming so a Pi
            # batch's pi_cam_keys (base_0_rgb / *_wrist_0_rgb) can find aria's
            # front image. aria has no wrist cameras; the missing wrist keys
            # are auto-duplicated by Pi's _fill_missing_images.
            front_key = (
                "base_0_rgb" if keymap_mode == "cartesian_pi" else cls.VIZ_IMAGE_KEY
            )
            return {
                front_key: {
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
        elif keymap_mode == "keypoints":
            return {
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


class Scale(Human):
    VIZ_INTRINSICS_KEY = "scale"
    ACTION_STRIDE = 1

    @classmethod
    def _get_keymap(
        cls,
        keymap_mode: Literal["cartesian", "keypoints"],
    ):
        if keymap_mode == "cartesian":
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
            }
        elif keymap_mode == "keypoints":
            return {
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
            }


class Mecka(Human):
    VIZ_INTRINSICS_KEY = "mecka"
    ACTION_STRIDE = 1

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal["cartesian", "cartesian_6d"] = "cartesian",
        chunk_length: int = 100,
    ) -> list[Transform]:
        if mode == "cartesian":
            return _build_human_cartesian_bimanual_transform_list(
                stride=cls.ACTION_STRIDE,
                chunk_length=chunk_length,
            )
        elif mode == "cartesian_6d":
            return _build_human_cartesian_bimanual_transform_list(
                stride=cls.ACTION_STRIDE,
                chunk_length=chunk_length,
                rot_repr="6d",
            )

    @classmethod
    def get_keymap(
        cls,
        mode: Literal["cartesian", "cartesian_pi", "keypoints"] | None = None,
        keymap_mode: Literal["cartesian", "cartesian_pi", "keypoints"] | None = None,
        annotations: bool = False,
        norm_mode: bool = False,
        annotation_key: str | None = None,
        proprio_history: int = 1,
    ):
        if mode is None:
            mode = keymap_mode or "cartesian"
        elif keymap_mode is not None and keymap_mode != mode:
            raise ValueError(
                f"Conflicting Mecka keymap modes: mode='{mode}', keymap_mode='{keymap_mode}'."
            )

        if mode in ("cartesian", "cartesian_pi"):
            front_key = "base_0_rgb" if mode == "cartesian_pi" else cls.VIZ_IMAGE_KEY
            key_map = {
                front_key: {
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
            raise ValueError(
                f"Unsupported mode '{mode}'. Expected one of: 'cartesian', 'cartesian_pi', 'keypoints'."
            )

        requested_annotation_key = annotation_key or (
            "annotations" if annotations else None
        )
        if requested_annotation_key is not None:
            key_map[requested_annotation_key] = {
                "key_type": "annotation_keys",
                "zarr_key": requested_annotation_key,
            }
        if norm_mode:
            for key in list(key_map):
                if key_map[key].get("key_type") in ("camera_keys", "annotation_keys"):
                    del key_map[key]
        # Short-term memory: when proprio_history>1, read a backward window of
        # that many frames for the EE/keypoint observation poses (current frame
        # last). proprio_history=1 is a byte-identical no-op.
        #
        # `obs_head_pose` is EXCLUDED: it is the coordinate REFERENCE frame
        # (target_world for the head-frame transforms), not a model observation.
        # It must stay single-frame so each windowed EE pose is re-expressed in
        # the *current* head frame; giving it a window makes target_world (K, 7)
        # and breaks the (B, 7)-only frame math. The reader's `horizon` branch
        # also takes precedence, so keypoints-mode keys with a horizon are
        # unaffected.
        history_exclude = {"obs_head_pose"}
        if proprio_history and proprio_history > 1:
            for name, entry in key_map.items():
                if (
                    entry.get("key_type") == "proprio_keys"
                    and name not in history_exclude
                    and entry.get("zarr_key") not in history_exclude
                ):
                    entry["history"] = proprio_history
        return key_map


# this works for quat and ypr since actionChunkCoordinateFrameTransform works for both
def _build_human_keypoints_revert_eef_frame_transform_list(
    *,
    action_key: str = "actions_keypoints",
    obs_key: str = "observations.state.keypoints",
    left_keypoints_action_wristframe: str = "left.action_keypoints_wristframe",
    right_keypoints_action_wristframe: str = "right.action_keypoints_wristframe",
    left_wrist_obs_headframe: str = "left.obs_wrist_pose_headframe",
    right_wrist_obs_headframe: str = "right.obs_wrist_pose_headframe",
    left_wrist_action_headframe: str = "left.action_wrist_pose_headframe",
    right_wrist_action_headframe: str = "right.action_wrist_pose_headframe",
    left_wrist_action_wristframe: str = "left.action_wrist_pose_wristframe",
    right_wrist_action_wristframe: str = "right.action_wrist_pose_wristframe",
    left_keypoints_action_headframe: str = "left.action_keypoints_headframe",
    right_keypoints_action_headframe: str = "right.action_keypoints_headframe",
    left_keypoints_obs_wristframe: str = "left.obs_keypoints_wristframe",
    right_keypoints_obs_wristframe: str = "right.obs_keypoints_wristframe",
    is_quat: bool = True,
) -> list[Transform]:
    if is_quat:
        pose_shape = 7
    else:
        pose_shape = 6
    transform_list = [
        SplitKeys(
            input_key=obs_key,
            output_key_list=[
                (left_wrist_obs_headframe, pose_shape),
                (left_keypoints_obs_wristframe, 63),
                (right_wrist_obs_headframe, pose_shape),
                (right_keypoints_obs_wristframe, 63),
            ],
        ),
        SplitKeys(
            input_key=action_key,
            output_key_list=[
                (left_wrist_action_wristframe, pose_shape),
                (left_keypoints_action_wristframe, 63),
                (right_wrist_action_wristframe, pose_shape),
                (right_keypoints_action_wristframe, 63),
            ],
        ),
        Reshape(
            input_key=left_keypoints_action_wristframe,
            output_key=left_keypoints_action_wristframe,
            shape=(100, 21, 3),
        ),
        Reshape(
            input_key=right_keypoints_action_wristframe,
            output_key=right_keypoints_action_wristframe,
            shape=(100, 21, 3),
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=left_wrist_obs_headframe,
            chunk_world=left_keypoints_action_wristframe,
            transformed_key_name=left_keypoints_action_headframe,
            mode="xyz",
            inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_wrist_obs_headframe,
            chunk_world=right_keypoints_action_wristframe,
            transformed_key_name=right_keypoints_action_headframe,
            mode="xyz",
            inverse=False,
        ),
        Reshape(
            input_key=left_keypoints_action_headframe,
            output_key=left_keypoints_action_headframe,
            shape=(100, 63),
        ),
        Reshape(
            input_key=right_keypoints_action_headframe,
            output_key=right_keypoints_action_headframe,
            shape=(100, 63),
        ),
        ConcatKeys(
            key_list=[
                left_keypoints_action_headframe,
                right_keypoints_action_headframe,
            ],
            new_key_name=action_key,
            delete_old_keys=True,
        ),
    ]
    return transform_list


def _build_human_keypoints_eef_frame_transform_list(
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
    left_keypoints_action_wristframe: str = "left.action_keypoints_wristframe",
    right_keypoints_action_wristframe: str = "right.action_keypoints_wristframe",
    left_wrist_action_wristframe: str = "left.action_wrist_pose_wristframe",
    right_wrist_action_wristframe: str = "right.action_wrist_pose_wristframe",
    left_wrist_obs_pose: str = "left.obs_wrist_pose",
    right_wrist_obs_pose: str = "right.obs_wrist_pose",
    left_wrist_action_headframe: str = "left.action_wrist_pose_headframe",
    right_wrist_action_headframe: str = "right.action_wrist_pose_headframe",
    left_wrist_obs_headframe: str = "left.obs_wrist_pose_headframe",
    right_wrist_obs_headframe: str = "right.obs_wrist_pose_headframe",
    left_keypoints_obs_wristframe: str = "left.obs_keypoints_wristframe",
    right_keypoints_obs_wristframe: str = "right.obs_keypoints_wristframe",
    delete_target_world: bool = True,
    chunk_length: int = 100,
    stride: int = 3,
    is_quat: bool = True,
) -> list[Transform]:
    transform_list = _build_human_keypoints_bimanual_transform_list(
        target_world=target_world,
        target_world_ypr=target_world_ypr,
        target_world_is_quat=target_world_is_quat,
        delete_target_world=delete_target_world,
        chunk_length=chunk_length,
        stride=stride,
        concat_keys=False,
        is_quat=True,
    )
    delete_keys = [
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
    ]
    if delete_target_world:
        delete_keys.append(target_world)
        if target_world_is_quat:
            delete_keys.append(target_world_ypr)
    transform_list.extend(
        [
            Reshape(
                input_key=left_keypoints_action_headframe,
                output_key=left_keypoints_action_headframe,
                shape=(chunk_length, 21, 3),
            ),
            Reshape(
                input_key=right_keypoints_action_headframe,
                output_key=right_keypoints_action_headframe,
                shape=(chunk_length, 21, 3),
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=left_wrist_obs_headframe,
                chunk_world=left_keypoints_action_headframe,
                transformed_key_name=left_keypoints_action_wristframe,
                mode="xyz",
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=right_wrist_obs_headframe,
                chunk_world=right_keypoints_action_headframe,
                transformed_key_name=right_keypoints_action_wristframe,
                mode="xyz",
            ),
            Reshape(
                input_key=left_keypoints_action_wristframe,
                output_key=left_keypoints_action_wristframe,
                shape=(chunk_length, 63),
            ),
            Reshape(
                input_key=right_keypoints_action_wristframe,
                output_key=right_keypoints_action_wristframe,
                shape=(chunk_length, 63),
            ),
            Reshape(
                input_key=left_keypoints_obs_headframe,
                output_key=left_keypoints_obs_headframe,
                shape=(21, 3),
            ),
            Reshape(
                input_key=right_keypoints_obs_headframe,
                output_key=right_keypoints_obs_headframe,
                shape=(21, 3),
            ),
            PoseCoordinateFrameTransform(
                target_world=left_wrist_obs_headframe,
                pose_world=left_keypoints_obs_headframe,
                transformed_key_name=left_keypoints_obs_wristframe,
                mode="xyz",
            ),
            PoseCoordinateFrameTransform(
                target_world=right_wrist_obs_headframe,
                pose_world=right_keypoints_obs_headframe,
                transformed_key_name=right_keypoints_obs_wristframe,
                mode="xyz",
            ),
            Reshape(
                input_key=left_keypoints_obs_wristframe,
                output_key=left_keypoints_obs_wristframe,
                shape=(63,),
            ),
            Reshape(
                input_key=right_keypoints_obs_wristframe,
                output_key=right_keypoints_obs_wristframe,
                shape=(63,),
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=left_wrist_obs_headframe,
                chunk_world=left_wrist_action_headframe,
                transformed_key_name=left_wrist_action_wristframe,
                mode="xyzwxyz",
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=right_wrist_obs_headframe,
                chunk_world=right_wrist_action_headframe,
                transformed_key_name=right_wrist_action_wristframe,
                mode="xyzwxyz",
            ),
        ]
    )
    if not is_quat:
        transform_list.extend(
            [
                BatchQuaternionPoseToYPR(
                    pose_key=left_wrist_action_wristframe,
                    output_key=left_wrist_action_wristframe,
                ),
                BatchQuaternionPoseToYPR(
                    pose_key=right_wrist_action_wristframe,
                    output_key=right_wrist_action_wristframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=left_wrist_obs_headframe,
                    output_key=left_wrist_obs_headframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=right_wrist_obs_headframe,
                    output_key=right_wrist_obs_headframe,
                ),
            ]
        )
    transform_list.extend(
        [
            ConcatKeys(
                key_list=[
                    left_wrist_action_wristframe,
                    left_keypoints_action_wristframe,
                    right_wrist_action_wristframe,
                    right_keypoints_action_wristframe,
                ],
                new_key_name="actions_keypoints",
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[
                    left_wrist_obs_headframe,
                    left_keypoints_obs_wristframe,
                    right_wrist_obs_headframe,
                    right_keypoints_obs_wristframe,
                ],
                new_key_name="observations.state.keypoints",
                delete_old_keys=True,
            ),
            DeleteKeys(keys_to_delete=delete_keys),
        ]
    )
    return transform_list


def _build_human_keypoints_bimanual_transform_list(
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
    concat_keys: bool = True,
    is_quat: bool = True,
) -> list[Transform]:
    keys_to_delete = list(
        {
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
        }
    )
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
    if not is_quat:
        transform_list.extend(
            [
                BatchQuaternionPoseToYPR(
                    pose_key=left_wrist_action_headframe,
                    output_key=left_wrist_action_headframe,
                ),
                BatchQuaternionPoseToYPR(
                    pose_key=right_wrist_action_headframe,
                    output_key=right_wrist_action_headframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=left_wrist_obs_headframe,
                    output_key=left_wrist_obs_headframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=right_wrist_obs_headframe,
                    output_key=right_wrist_obs_headframe,
                ),
            ]
        )
    if concat_keys:
        transform_list.extend(
            [
                ConcatKeys(
                    key_list=[
                        left_wrist_action_headframe,
                        left_keypoints_action_headframe,
                        right_wrist_action_headframe,
                        right_keypoints_action_headframe,
                    ],
                    new_key_name="actions_keypoints",
                    delete_old_keys=True,
                ),
                ConcatKeys(
                    key_list=[
                        left_wrist_obs_headframe,
                        left_keypoints_obs_headframe,
                        right_wrist_obs_headframe,
                        right_keypoints_obs_headframe,
                    ],
                    new_key_name="observations.state.keypoints",
                    delete_old_keys=True,
                ),
                DeleteKeys(keys_to_delete=keys_to_delete),
            ]
        )
    return transform_list


def _build_human_cartesian_revert_eef_frame_transform_list(
    *,
    action_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    left_action_wristframe: str = "left.action_ee_pose_wristframe",
    right_action_wristframe: str = "right.action_ee_pose_wristframe",
    left_obs_headframe: str = "left.obs_ee_pose_headframe",
    right_obs_headframe: str = "right.obs_ee_pose_headframe",
    left_action_headframe: str = "left.action_ee_pose_headframe",
    right_action_headframe: str = "right.action_ee_pose_headframe",
    is_quat: bool = False,
    rot_repr: str = "ypr",
) -> list[Transform]:
    """Revert wrist-frame ARIA cartesian actions back to head (camera) frame.

    Inverse of ``_build_human_cartesian_eef_frame_transform_list`` for viz: the
    action chunks live in each side's wrist frame, the proprio ee-poses live in
    headframe (= Aria camera frame). Re-composes ``target_headframe @ chunk_wristframe``
    so action chunks are back in headframe / camera frame.

    ``rot_repr="6d"`` reverts a model 6D prediction (per-arm width 9, ``xyz6d``
    coordinate transform with Gram-Schmidt), then collapses back to ypr.
    """
    if rot_repr == "6d":
        pose_shape = 9
        mode = "xyz6d"
    else:
        pose_shape = 7 if is_quat else 6
        mode = "xyzwxyz" if is_quat else "xyzypr"
    transform_list = [
        SplitKeys(
            input_key=obs_key,
            output_key_list=[
                (left_obs_headframe, pose_shape),
                (right_obs_headframe, pose_shape),
            ],
        ),
        SplitKeys(
            input_key=action_key,
            output_key_list=[
                (left_action_wristframe, pose_shape),
                (right_action_wristframe, pose_shape),
            ],
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=left_obs_headframe,
            chunk_world=left_action_wristframe,
            transformed_key_name=left_action_headframe,
            mode=mode,
            inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_obs_headframe,
            chunk_world=right_action_wristframe,
            transformed_key_name=right_action_headframe,
            mode=mode,
            inverse=False,
        ),
    ]
    if rot_repr == "6d":
        transform_list.append(
            XYZ6D_to_XYZYPR(keys=[left_action_headframe, right_action_headframe])
        )
    transform_list.append(
        ConcatKeys(
            key_list=[left_action_headframe, right_action_headframe],
            new_key_name=action_key,
            delete_old_keys=True,
        ),
    )
    return transform_list


def _build_human_cartesian_eef_frame_transform_list(
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
    left_action_wristframe: str = "left.action_ee_pose_wristframe",
    right_action_wristframe: str = "right.action_ee_pose_wristframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 3,
    delete_target_world: bool = True,
    rot_repr: str = "ypr",
) -> list[Transform]:
    """ARIA bimanual cartesian pipeline expressed in the current wrist frame.

    Action ee-pose chunks are first transformed world → headframe (via
    ``obs_head_pose``), then headframe → wristframe (via the proprio
    ``*.obs_ee_pose_headframe`` for each side). Proprio ee-poses remain in
    headframe (wristframe of the wrist itself is identity). All retained poses
    are converted to xyz-ypr, or to continuous 6D columns when ``rot_repr="6d"``.
    """
    keys_to_delete = list(
        {
            left_action_world,
            right_action_world,
            left_obs_pose,
            right_obs_pose,
            left_action_headframe,
            right_action_headframe,
        }
    )
    if delete_target_world:
        keys_to_delete.append(target_world)
        if target_world_is_quat:
            keys_to_delete.append(target_world_ypr)

    transform_list: list[Transform] = [
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=left_action_world,
            transformed_key_name=left_action_headframe,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=right_action_world,
            transformed_key_name=right_action_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
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
        ActionChunkCoordinateFrameTransform(
            target_world=left_obs_headframe,
            chunk_world=left_action_headframe,
            transformed_key_name=left_action_wristframe,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_obs_headframe,
            chunk_world=right_action_headframe,
            transformed_key_name=right_action_wristframe,
            mode="xyzwxyz",
        ),
        (XYZWXYZ_to_XYZ6D if rot_repr == "6d" else XYZWXYZ_to_XYZYPR)(
            keys=[
                left_action_wristframe,
                right_action_wristframe,
                left_obs_headframe,
                right_obs_headframe,
            ]
        ),
        ConcatKeys(
            key_list=[left_action_wristframe, right_action_wristframe],
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
    return transform_list


def _build_human_cartesian_bimanual_transform_list(
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
    rot_repr: str = "ypr",
) -> list[Transform]:
    """Canonical ARIA bimanual transform pipeline used by tests and notebooks.

    Aria human data does not have commanded ee poses; action chunks are built
    from stacked observed ee poses (typically with a horizon on
    ``left/right.action_ee_pose`` mapped from ``left/right.obs_ee_pose``).
    ``rot_repr="6d"`` emits continuous 6D rotation columns instead of ypr.
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
            (XYZWXYZ_to_XYZ6D if rot_repr == "6d" else XYZWXYZ_to_XYZYPR)(
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
