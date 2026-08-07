from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import numpy as np

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
    XYZWXYZ_to_XYZYPR,
)
from egomimic.utils.viz_utils import (
    ColorPalette,
    _viz_gaze,
    _viz_keypoints,
)


ARIA_INTRINSICS = np.array(
    [
        [133.25430222 * 2, 0.0, 320, 0],
        [0.0, 133.25430222 * 2, 240, 0],
        [0.0, 0.0, 1.0, 0],
    ]
)

ARIA_INTRINSICS_HALF = np.array(
    [
        [133.25430222, 0.0, 320 / 2, 0],
        [0.0, 133.25430222, 240 / 2, 0],
        [0.0, 0.0, 1.0, 0],
    ]
)

SCALE_INTRINSICS = np.array(
    [[214.134, 0.0, 324.593, 0], [0.0, 256.968, 260.146, 0], [0.0, 0.0, 1.0, 0]]
)

_w0, _h0 = float(1920), float(1080)
_fx0, _fy0 = float(752.4707352849115), float(753.0015979987369)
_cx0, _cy0 = float(961.8249427694457), float(553.245895705989)
_sx = 640 / _w0
_sy = 360 / _h0
_fx, _fy = _fx0 * _sx, _fy0 * _sy
_cx, _cy = _cx0 * _sx, _cy0 * _sy

MECKA_INTRINSICS = np.array(
    [[_fx, 0.0, _cx, 0], [0.0, _fy, _cy, 0], [0.0, 0.0, 1.0, 0]], dtype=np.float64
)

LIGHTWHEEL_INTRINSICS = np.array(
    [
        [786.6216072, 0.0, 960.0, 0],
        [0.0, 786.6216072, 728.0, 0],
        [0.0, 0.0, 1.0, 0],
    ]
)

ARIA_T_RGB_CPF = np.array(
    [
        [-0.99989084, 0.01251132, -0.00786028, 0.05686918],
        [-0.01132842, -0.99067146, -0.13580032, 0.00922798],
        [-0.009486, -0.13569645, 0.99070505, -0.01147902],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# Aria's raw 21-keypoint layout (0-4 fingertips, 5 palm root) — NOT MANO. Used
# only for the opt-in raw-Aria-keypoint viz; the canonical keypoints are MANO.
ARIA_FINGER_EDGES = [
    (5, 6), (6, 7), (7, 0),                # thumb
    (5, 8), (8, 9), (9, 10), (10, 1),      # index
    (5, 11), (11, 12), (12, 13), (13, 2),  # middle
    (5, 14), (14, 15), (15, 16), (16, 3),  # ring
    (5, 17), (17, 18), (18, 19), (19, 4),  # pinky
]
ARIA_FINGER_EDGE_RANGES = [
    ("thumb", 0, 3), ("index", 3, 7), ("middle", 7, 11), ("ring", 11, 15), ("pinky", 15, 19),
]


class Human(Embodiment):
    """Single data-driven human embodiment shared by every vendor's human data.

    Per-vendor structural differences are explicit classmethod arguments supplied
    by the data config, because get_keymap / get_transform_list resolve at hydra
    config time (before any episode is read):
      - get_keymap(keymap_mode, has_head_pose=True, include_aria_keypoints=False)
      - get_transform_list(mode, stride=3)
    Per-episode camera intrinsics travel in ``batch["intrinsics"]`` (from
    zarr.json); ``cls.INTRINSICS`` is only a fallback for legacy episodes that
    lack them. The canonical keypoints are MANO for every vendor.
    """
    INTRINSICS = ARIA_INTRINSICS  # fallback only — real value comes from the batch
    ACTION_HORIZON = 30
    # Front-image key for Pi/PaliGemma-style naming (any "_pi"-suffixed mode);
    # Pi's _fill_missing_images auto-duplicates the absent wrist keys.
    PI_FRONT_KEY = "base_0_rgb"
    T_RGB_CPF = ARIA_T_RGB_CPF  # for the opt-in aria gaze viz
    # Canonical MANO 21-keypoint topology: 0=wrist, 1-4 thumb, 5-8 index, ...
    FINGER_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),         # index
        (0, 9), (9, 10), (10, 11), (11, 12),    # middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
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
        mode=Literal[
            "traj", "traj+rotation", "axes", "annotations", "keypoints", "gaze"
        ],
        intrinsics=None,
        finger_edges=None,
        finger_edge_ranges=None,
        **kwargs,
    ):
        K = intrinsics if intrinsics is not None else cls.INTRINSICS
        if mode == "gaze":
            return _viz_gaze(
                image=image,
                gaze_data=viz_data,
                intrinsics=K,
                t_rgb_cpf=cls.T_RGB_CPF,
                **kwargs,
            )
        if mode == "keypoints":
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
                intrinsics=K,
                edges=finger_edges if finger_edges is not None else cls.FINGER_EDGES,
                edge_ranges=(
                    finger_edge_ranges
                    if finger_edge_ranges is not None
                    else cls.FINGER_EDGE_RANGES
                ),
                colors=colors,
                dot_color=dot_color,
                **kwargs,
            )
        return super().viz(image, viz_data, mode=mode, intrinsics=intrinsics, **kwargs)

    @classmethod
    def get_keymap(
        cls,
        keymap_mode: str,
        has_head_pose: bool = True,
        include_aria_keypoints: bool = False,
        norm_mode: bool = False,
        annotation_key: str = None,
    ):
        """Build the keymap. Per-vendor knobs are explicit args from the data
        config: ``has_head_pose`` (Scale=False) and ``include_aria_keypoints``
        (Aria=True). ``norm_mode``/``annotation_key`` behave as in the base.
        """
        key_map = cls._get_keymap(
            keymap_mode,
            has_head_pose=has_head_pose,
            include_aria_keypoints=include_aria_keypoints,
        )
        if annotation_key is not None and not norm_mode:
            key_map[annotation_key] = {
                "key_type": "annotation_keys",
                "zarr_key": annotation_key,
            }
        if norm_mode:
            to_delete = [
                k
                for k, v in key_map.items()
                if v.get("key_type") in ("camera_keys", "annotation_keys")
            ]
            for k in to_delete:
                del key_map[k]
        return key_map

    @classmethod
    def _get_keymap(
        cls,
        keymap_mode: str,
        has_head_pose: bool = True,
        include_aria_keypoints: bool = False,
    ):
        """Canonical MANO keymap. A ``_pi`` suffix swaps the front image key to
        ``PI_FRONT_KEY``; ``include_aria_keypoints`` additionally exposes the raw
        Aria-layout proprio keypoints alongside the MANO ones.
        """
        is_pi = keymap_mode.endswith("_pi")
        base_mode = keymap_mode[: -len("_pi")] if is_pi else keymap_mode
        front_key = cls.PI_FRONT_KEY if is_pi else cls.VIZ_IMAGE_KEY
        horizon = cls.ACTION_HORIZON

        if base_mode == "cartesian":
            key_map = {
                front_key: {
                    "key_type": "camera_keys",
                    "zarr_key": "images.front_1",
                },
                "right.action_ee_pose": {
                    "key_type": "action_keys",
                    "zarr_key": "right.obs_ee_pose",
                    "horizon": horizon,
                },
                "left.action_ee_pose": {
                    "key_type": "action_keys",
                    "zarr_key": "left.obs_ee_pose",
                    "horizon": horizon,
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
        elif base_mode == "keypoints":
            kp = "obs_keypoints"  # canonical MANO keypoints for every vendor
            key_map = {
                front_key: {
                    "key_type": "camera_keys",
                    "zarr_key": "images.front_1",
                },
                "left.action_keypoints": {
                    "key_type": "action_keys",
                    "zarr_key": f"left.{kp}",
                    "horizon": horizon,
                },
                "right.action_keypoints": {
                    "key_type": "action_keys",
                    "zarr_key": f"right.{kp}",
                    "horizon": horizon,
                },
                "left.action_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_wrist_pose",
                    "horizon": horizon,
                },
                "right.action_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_wrist_pose",
                    "horizon": horizon,
                },
                "left.obs_keypoints": {
                    "key_type": "proprio_keys",
                    "zarr_key": f"left.{kp}",
                },
                "right.obs_keypoints": {
                    "key_type": "proprio_keys",
                    "zarr_key": f"right.{kp}",
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
            if include_aria_keypoints:
                # Raw Aria-layout keypoints exposed alongside the canonical MANO
                # ones (proprio, no horizon: no transform consumes them).
                for side in ("left", "right"):
                    key_map[f"{side}.obs_aria_keypoints"] = {
                        "key_type": "proprio_keys",
                        "zarr_key": f"{side}.obs_aria_keypoints",
                    }
        else:
            raise ValueError(
                f"Unsupported keymap_mode '{keymap_mode}' for {cls.__name__}. "
                "Expected 'cartesian' or 'keypoints' (optionally with a '_pi' suffix)."
            )

        if has_head_pose:
            key_map["obs_head_pose"] = {
                "key_type": "proprio_keys",
                "zarr_key": "obs_head_pose",
            }
        return key_map

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal[
            "cartesian",
            "cartesian_padded",
            "cartesian_wristframe_ypr",
            "keypoints_headframe_ypr",
            "keypoints_headframe_quat",
            "keypoints_wristframe_ypr",
            "keypoints_wristframe_quat",
        ],
        stride: int = 3,
    ) -> list[Transform]:
        """Transform pipeline. ``stride`` is the per-vendor action stride
        (Aria/LightWheel=3, Scale/Mecka=1), supplied by the data config.
        """
        if mode == "cartesian":
            return _build_human_cartesian_bimanual_transform_list(stride=stride)
        if mode == "cartesian_padded":
            return _build_human_cartesian_bimanual_transform_list(
                stride=stride
            ) + [PadGripperZeros(action_key="actions_cartesian")]
        if mode == "cartesian_wristframe_ypr":
            return _build_human_cartesian_eef_frame_transform_list(stride=stride)
        if mode == "keypoints_headframe_ypr":
            return _build_human_keypoints_bimanual_transform_list(
                stride=stride, is_quat=False
            )
        if mode == "keypoints_headframe_quat":
            return _build_human_keypoints_bimanual_transform_list(
                stride=stride, is_quat=True
            )
        if mode == "keypoints_wristframe_ypr":
            return _build_human_keypoints_eef_frame_transform_list(
                stride=stride, is_quat=False
            )
        if mode == "keypoints_wristframe_quat":
            return _build_human_keypoints_eef_frame_transform_list(
                stride=stride, is_quat=True
            )
        raise ValueError(f"Unsupported transform_list mode '{mode}' for {cls.__name__}")


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
) -> list[Transform]:
    """Revert wrist-frame ARIA cartesian actions back to head (camera) frame.

    Inverse of ``_build_human_cartesian_eef_frame_transform_list`` for viz: the
    action chunks live in each side's wrist frame, the proprio ee-poses live in
    headframe (= Aria camera frame). Re-composes ``target_headframe @ chunk_wristframe``
    so action chunks are back in headframe / camera frame.
    """
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
        ConcatKeys(
            key_list=[left_action_headframe, right_action_headframe],
            new_key_name=action_key,
            delete_old_keys=True,
        ),
    ]
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
) -> list[Transform]:
    """ARIA bimanual cartesian pipeline expressed in the current wrist frame.

    Action ee-pose chunks are first transformed world → headframe (via
    ``obs_head_pose``), then headframe → wristframe (via the proprio
    ``*.obs_ee_pose_headframe`` for each side). Proprio ee-poses remain in
    headframe (wristframe of the wrist itself is identity). All retained poses
    are converted to xyz-ypr.
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
        XYZWXYZ_to_XYZYPR(
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
