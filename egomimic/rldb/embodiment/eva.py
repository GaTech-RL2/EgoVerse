from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    ConcatKeys,
    DeleteKeys,
    InterpolateLinear,
    InterpolatePose,
    PoseCoordinateFrameTransform,
    Transform,
    XYZWXYZ_to_XYZYPR,
)
from egomimic.utils.egomimicUtils import (
    EXTRINSICS,
    INTRINSICS,
    cam_frame_to_cam_pixels,
    draw_actions,
)
from egomimic.utils.pose_utils import (
    _matrix_to_xyzwxyz,
)


class Eva(Embodiment):
    VIZ_IMAGE_KEY = "observations.images.front_img_1"

    @staticmethod
    def get_transform_list() -> list[Transform]:
        return _build_eva_bimanual_transform_list()

    @classmethod
    def viz_transformed_batch(cls, batch, mode=""):
        """
        Visualize one transformed EVA batch sample.

        Modes:
            - palm_traj: draw left/right palm trajectories from actions_cartesian.
            - palm_axes: draw local xyz axes at each palm anchor using ypr.
        """
        image_key = cls.VIZ_IMAGE_KEY
        action_key = "actions_cartesian"
        intrinsics_key = "base"
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
    transform_list = [
        ActionChunkCoordinateFrameTransform(
            target_world=left_target_world,
            chunk_world=left_cmd_world,
            transformed_key_name=left_cmd_camframe,
            extra_batch_key=left_extra_batch_key,
            is_quat=is_quat,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_target_world,
            chunk_world=right_cmd_world,
            transformed_key_name=right_cmd_camframe,
            extra_batch_key=right_extra_batch_key,
            is_quat=is_quat,
        ),
        PoseCoordinateFrameTransform(
            target_world=left_target_world,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_pose,
            is_quat=is_quat,
        ),
        PoseCoordinateFrameTransform(
            target_world=right_target_world,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_pose,
            is_quat=is_quat,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_cmd_camframe,
            output_action_key=left_cmd_camframe,
            stride=stride,
            is_quat=is_quat,
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_cmd_camframe,
            output_action_key=right_cmd_camframe,
            stride=stride,
            is_quat=is_quat,
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
        ]
    )
    return transform_list


def _to_numpy(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        return arr.numpy()
    return np.asarray(arr)


def _split_action_pose(actions):
    # 14D layout: [L xyz ypr g, R xyz ypr g]
    # 12D layout: [L xyz ypr, R xyz ypr]
    if actions.shape[-1] == 14:
        left_xyz = actions[:, :3]
        left_ypr = actions[:, 3:6]
        right_xyz = actions[:, 7:10]
        right_ypr = actions[:, 10:13]
    elif actions.shape[-1] == 12:
        left_xyz = actions[:, :3]
        left_ypr = actions[:, 3:6]
        right_xyz = actions[:, 6:9]
        right_ypr = actions[:, 9:12]
    else:
        raise ValueError(f"Unsupported action dim {actions.shape[-1]}")
    return left_xyz, left_ypr, right_xyz, right_ypr


def _prepare_viz_image(batch, image_key):
    img = _to_numpy(batch[image_key][0])
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    return img


def _viz_batch_palm_traj(batch, image_key, action_key, intrinsics_key):
    img_np = _prepare_viz_image(batch, image_key)
    intrinsics = INTRINSICS[intrinsics_key]
    actions = _to_numpy(batch[action_key][0])
    left_xyz, _, right_xyz, _ = _split_action_pose(actions)

    vis = draw_actions(
        img_np.copy(),
        type="xyz",
        color="Blues",
        actions=left_xyz,
        extrinsics=None,
        intrinsics=intrinsics,
        arm="left",
    )
    vis = draw_actions(
        vis,
        type="xyz",
        color="Reds",
        actions=right_xyz,
        extrinsics=None,
        intrinsics=intrinsics,
        arm="right",
    )
    return vis


def _viz_batch_palm_axes(batch, image_key, action_key, intrinsics_key, axis_len_m=0.04):
    img_np = _prepare_viz_image(batch, image_key)
    intrinsics = INTRINSICS[intrinsics_key]
    actions = _to_numpy(batch[action_key][0])
    left_xyz, left_ypr, right_xyz, right_ypr = _split_action_pose(actions)
    vis = img_np.copy()

    def _draw_axis_color_legend(frame):
        _, w = frame.shape[:2]
        x_right = w - 12
        y_start = 14
        y_step = 12
        line_len = 24
        axis_legend = [
            ("x", (255, 0, 0)),
            ("y", (0, 255, 0)),
            ("z", (0, 0, 255)),
        ]
        for i, (name, color) in enumerate(axis_legend):
            y = y_start + i * y_step
            x0 = x_right - line_len
            x1 = x_right
            cv2.line(frame, (x0, y), (x1, y), color, 3)
            cv2.putText(
                frame,
                name,
                (x0 - 12, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        return frame

    def _draw_rotation_at_palm(frame, xyz_seq, ypr_seq, label, anchor_color):
        if len(xyz_seq) == 0 or len(ypr_seq) == 0:
            return frame

        palm_xyz = xyz_seq[0]
        palm_ypr = ypr_seq[0]
        rot = R.from_euler("ZYX", palm_ypr, degrees=False).as_matrix()

        axis_points_cam = np.vstack(
            [
                palm_xyz,
                palm_xyz + rot[:, 0] * axis_len_m,
                palm_xyz + rot[:, 1] * axis_len_m,
                palm_xyz + rot[:, 2] * axis_len_m,
            ]
        )

        px = cam_frame_to_cam_pixels(axis_points_cam, intrinsics)[:, :2]
        if not np.isfinite(px).all():
            return frame
        pts = np.round(px).astype(np.int32)

        h, w = frame.shape[:2]
        x0, y0 = pts[0]
        if not (0 <= x0 < w and 0 <= y0 < h):
            return frame

        cv2.circle(frame, (x0, y0), 4, anchor_color, -1)
        axis_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, color in enumerate(axis_colors, start=1):
            x1, y1 = pts[i]
            if 0 <= x1 < w and 0 <= y1 < h:
                cv2.line(frame, (x0, y0), (x1, y1), color, 2)
                cv2.circle(frame, (x1, y1), 2, color, -1)

        cv2.putText(
            frame,
            label,
            (x0 + 6, max(12, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            anchor_color,
            1,
            cv2.LINE_AA,
        )
        return frame

    vis = _draw_rotation_at_palm(vis, left_xyz, left_ypr, "L rot", (255, 180, 80))
    vis = _draw_rotation_at_palm(vis, right_xyz, right_ypr, "R rot", (80, 180, 255))
    vis = _draw_axis_color_legend(vis)
    return vis
