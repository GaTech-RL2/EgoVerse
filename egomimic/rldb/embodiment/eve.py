"""
Eve embodiment for ETH/CVG faive-bimanual hand data.

Unlike Eva (which has world-frame raw poses + extrinsics that need to be
transformed at load time) and Aria (which derives chunked actions from
stacked observations), the Eve pipeline assumes the upstream lerobot
converter (faive2lerobot) has already produced canonical, chunked,
frame-resolved trajectories. Zarr stores written by lerobot_to_zarr.py
contain those exact arrays with no further math required.

The keymap therefore maps post-transform batch keys directly to the same
zarr keys, and the transform list only converts numpy arrays to torch
tensors so downstream code receives the expected types.

Validation viz: actions_cartesian is in robot base frame (faive2lerobot's
``--base-frame`` flag), so projecting to the front camera requires the
extra base->cam step that the generic ``_viz_traj`` skips. ``Eve.viz``
overrides the base class to apply ``inv(T_base_cam)`` per arm before the
intrinsics projection, using the ``ethOct14`` calibration.
"""

from __future__ import annotations

import copy
from typing import Literal

import cv2
import numpy as np

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import NumpyToTensor, Transform
from egomimic.utils.egomimicUtils import (
    EXTRINSICS,
    INTRINSICS,
    cam_frame_to_cam_pixels,
    draw_dot_on_frame,
    ee_pose_to_cam_frame,
)
from egomimic.utils.pose_utils import _split_action_pose
from egomimic.utils.type_utils import _to_numpy
from egomimic.utils.viz_utils import (
    ColorPalette,
    _prepare_viz_image,
)


# Solid bright colors for cv2.circle. The frames handed to Eve.viz are RGB
# (decoded JPEG -> RGB, see zarr_dataset_multi), and cv2 draws the tuple as
# given, so these are RGB tuples (previously BGR: "Reds" rendered blue).
# Bypasses the matplotlib gradient that paints the first half of a long
# trajectory in near-white.
_PALETTE_TO_RGB = {
    "Greens":  (40, 220, 40),    # bright green
    "Reds":    (220, 40, 40),    # bright red
    "Blues":   (40, 40, 220),    # bright blue
    "Purples": (180, 60, 180),   # bright purple
}


def _solid_color_for_palette(palette: str) -> tuple[int, int, int]:
    return _PALETTE_TO_RGB.get(palette, (255, 255, 255))


class Eve(Embodiment):
    VIZ_INTRINSICS_KEY = "base"
    VIZ_EXTRINSICS_KEY = "ethOct14"
    VIZ_IMAGE_KEY = "observations.images.front_img_1"

    @staticmethod
    def get_transform_list(
        mode: Literal["cartesian", "cartesian_wrist", "cartesian_right_wrist"],
    ) -> list[Transform]:
        if mode in ("cartesian", "cartesian_wrist", "cartesian_right_wrist"):
            return [
                NumpyToTensor(
                    keys=[
                        "observations.state.ee_pose",
                        "observations.state.cartesian_arm",
                        "observations.state.joints_hand",
                        "actions_cartesian",
                        "actions_joints",
                    ]
                ),
            ]
        raise ValueError(f"Unsupported Eve transform mode: {mode}")

    @classmethod
    def _get_keymap(
        cls,
        keymap_mode: Literal["cartesian", "cartesian_wrist", "cartesian_right_wrist"],
    ):
        base = {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "observations.images.front_img_1",
            },
            "observations.state.ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.ee_pose",
            },
            "observations.state.cartesian_arm": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.cartesian_arm",
            },
            "observations.state.joints_hand": {
                "key_type": "proprio_keys",
                "zarr_key": "observations.state.joints_hand",
            },
            "actions_cartesian": {
                "key_type": "action_keys",
                "zarr_key": "actions_cartesian",
            },
            "actions_joints": {
                "key_type": "action_keys",
                "zarr_key": "actions_joints",
            },
        }
        if keymap_mode == "cartesian":
            return base
        if keymap_mode == "cartesian_right_wrist":
            # Single-arm rig (Franka + ORCA right hand) with one wrist camera.
            base["observations.images.right_wrist_img"] = {
                "key_type": "camera_keys",
                "zarr_key": "observations.images.right_wrist_img",
            }
            return base
        if keymap_mode == "cartesian_wrist":
            base["observations.images.left_wrist_img"] = {
                "key_type": "camera_keys",
                "zarr_key": "observations.images.left_wrist_img",
            }
            base["observations.images.right_wrist_img"] = {
                "key_type": "camera_keys",
                "zarr_key": "observations.images.right_wrist_img",
            }
            return base
        raise ValueError(f"Unsupported Eve keymap mode: {keymap_mode}")

    @classmethod
    def viz(
        cls,
        image,
        viz_data,
        mode: Literal["traj", "traj+rotation", "axes", "annotations"] = "traj",
        intrinsics_key: str | None = None,
        extrinsics_key: str | None = None,
        **kwargs,
    ):
        """Project base-frame action trajectories onto the front camera.

        Mirrors the OLD EgoVerse ``draw_actions`` base->cam pipeline: each
        per-arm xyz chunk is converted to camera frame via the inverse of the
        per-arm camera-to-base extrinsic, then projected with the front-camera
        intrinsics. Dots are painted opaquely (no alpha blend) — the calling
        ``viz_gt_preds`` controls draw order so both GT and pred remain visible.
        Other modes fall back to the base implementation.
        """
        if mode != "traj":
            return super().viz(
                image,
                viz_data,
                mode=mode,
                intrinsics_key=intrinsics_key,
                **kwargs,
            )

        intrinsics_key = intrinsics_key or cls.VIZ_INTRINSICS_KEY
        extrinsics_key = extrinsics_key or cls.VIZ_EXTRINSICS_KEY
        color = kwargs.get("color", "Blues")
        if not ColorPalette.is_valid(color):
            raise ValueError(f"Invalid color palette: {color}")

        intrinsics = INTRINSICS[intrinsics_key]
        extr = EXTRINSICS[extrinsics_key]

        actions = np.asarray(viz_data)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        left_xyz, _, right_xyz, _ = _split_action_pose(actions)
        # NOTE: NEW egomimicUtils.ee_pose_to_cam_frame inverts its 2nd arg
        # internally (line 715), so to get p_cam = T_cam_base @ p_base the
        # caller must pass T_base_cam (= the stored extrinsics matrix
        # "left_cam" / "right_cam", whose translation column is the camera
        # origin in robot base frame). Passing the inverted matrix would
        # double-invert and project off-image (top-right corner).
        # For right_arm-only embodiments, left_xyz is None and we only
        # project the right side.
        per_arm_cam = []
        if left_xyz is not None:
            left_xyz = left_xyz.reshape(-1, 3)
            per_arm_cam.append(ee_pose_to_cam_frame(left_xyz, extr["left_cam"]))
        right_xyz = right_xyz.reshape(-1, 3)
        per_arm_cam.append(ee_pose_to_cam_frame(right_xyz, extr["right_cam"]))
        pts_cam = np.concatenate(per_arm_cam, axis=0)
        pts_pix = cam_frame_to_cam_pixels(pts_cam, intrinsics)

        base = _prepare_viz_image(image)
        # NOTE: draw_dot_on_frame applies a matplotlib cmap gradient over
        # np.linspace(0, 1, N). For palettes like "Greens"/"Reds" with ~200
        # dots, the first half are near-white and visually disappear against
        # the camera image. To keep both GT and pred clearly visible we draw
        # solid saturated colors instead, derived from the palette name.
        rgb = _solid_color_for_palette(color)
        out = base.copy()
        for px in pts_pix:
            try:
                out = cv2.circle(out, (int(px[0]), int(px[1])), 5, rgb, -1)
            except Exception:
                pass
        return out

    @classmethod
    def viz_gt_preds(
        cls,
        predictions,
        batch,
        image_key,
        action_key,
        annotation_key=None,
        transform_list=None,
        mode: Literal["traj", "traj+rotation", "axes", "annotations"] = "traj",
        gt_alpha=1.0,
        pred_alpha=1.0,
        joints_key: str | None = None,
        hand_mount_key: str | None = None,
        hand_steps=(0, 50, 99),
        tool_flip_x180: bool = False,
        hand_show_arm: bool = True,
        **kwargs,
    ):
        """Match OLD EgoVerse draw order: predictions first, ground truth on
        top. The opposite (GT first, pred on top — the base-class default)
        causes well-trained predictions to occlude GT, leaving the user with
        a single-color overlay.

        joints_key: optional auxiliary action key (e.g. ``actions_joints``).
        When set, a panel with the GT (green) vs predicted (red) hand-joint
        chunk is appended to the right of each frame so the future hand
        trajectory is visible alongside the projected EE trajectory.
        hand_mount_key: key into ``orca_hand_fk.HAND_MOUNTS`` (T_tcp_hand).
        When set together with joints_key, the ORCA hand wireframe is drawn
        in the image at ``hand_steps`` of the chunk, placed by the GT (green)
        / predicted (red) EE pose and hand joints of that step.
        tool_flip_x180: the stored EE orientation is R_tcp @ Rx(180 deg)
        (real_mug adapter's seam fix); undo it before applying the mount."""
        embodiment_id = batch["embodiment"][0].item()
        from egomimic.rldb.embodiment.embodiment import get_embodiment
        embodiment_name = get_embodiment(embodiment_id).lower()
        pred_actions = predictions[f"{embodiment_name}_{action_key}"]
        gt_joints = pred_joints = None
        if joints_key is not None:
            pred_joints_key = f"{embodiment_name}_{joints_key}"
            if joints_key in batch and pred_joints_key in predictions:
                gt_joints = _to_numpy(batch[joints_key])
                pred_joints = _to_numpy(predictions[pred_joints_key])
        if transform_list is not None:
            pred_batch = copy.deepcopy(batch)
            pred_batch[action_key] = pred_actions
            batch = cls.apply_transform(batch, transform_list)
            pred_batch = cls.apply_transform(pred_batch, transform_list)
            pred_actions = pred_batch[action_key]

        images = _to_numpy(batch[image_key])
        actions = _to_numpy(batch[action_key])
        pred_actions = _to_numpy(pred_actions)
        if annotation_key is not None:
            annotations = batch[annotation_key]

        ims_list = []
        for i in range(images.shape[0]):
            image = images[i]
            action = actions[i]
            pred_action = pred_actions[i]
            # OLD order: pred first, GT on top, both opaque.
            ims = cls.viz(
                image, pred_action, mode=mode, color="Reds", alpha=pred_alpha, **kwargs
            )
            ims = cls.viz(
                ims, action, mode=mode, color="Greens", alpha=gt_alpha, **kwargs
            )
            if annotation_key is not None:
                ims = cls.viz(ims, [annotations[i]], mode="annotations", **kwargs)
            if gt_joints is not None and hand_mount_key is not None:
                ims = _prepare_viz_image(ims).copy()
                # same order as the trajectory dots: pred first, GT drawn on top
                for pose_chunk, joint_chunk, color in (
                    (pred_action, pred_joints[i], (220, 40, 40)),
                    (action, gt_joints[i], (40, 220, 40)),
                ):
                    ims = cls.draw_hand_chunk(
                        ims, pose_chunk, joint_chunk, hand_mount_key, hand_steps, color,
                        tool_flip_x180=tool_flip_x180,
                        intrinsics_key=kwargs.get("intrinsics_key"),
                        extrinsics_key=kwargs.get("extrinsics_key"),
                        skip=() if hand_show_arm else ("tower",),
                    )
            if gt_joints is not None:
                ims = cls.joints_panel(ims, gt_joints[i], pred_joints[i])
            ims_list.append(ims)
        return np.stack(ims_list, axis=0)

    @classmethod
    def draw_hand_chunk(
        cls, image, pose_chunk, joint_chunk, hand_mount_key, steps, color,
        tool_flip_x180=False, intrinsics_key=None, extrinsics_key=None, skip=(),
    ):
        """Project the ORCA hand wireframe(s) for a few steps of an action chunk.

        pose_chunk: (T, 6) single right arm or (T, 12) bimanual [L xyz ypr, R xyz
        ypr] base-frame EE poses (ZYX, the converter's layout, each arm in its
        own base frame); joint_chunk: (T, 17) or (T, 34) = [left 17, right 17]
        hand joints (orca_core v1 order). Per arm the hand frame is
        T_base_tcp(step) @ mount, projected with that arm's extrinsics."""
        from scipy.spatial.transform import Rotation as R
        from egomimic.utils import orca_hand_fk as ofk

        intrinsics = INTRINSICS[intrinsics_key or cls.VIZ_INTRINSICS_KEY]
        extr_all = EXTRINSICS[extrinsics_key or cls.VIZ_EXTRINSICS_KEY]
        pose_chunk = np.asarray(pose_chunk).reshape(-1, pose_chunk.shape[-1])
        joint_chunk = np.asarray(joint_chunk).reshape(-1, joint_chunk.shape[-1])
        T = min(len(pose_chunk), len(joint_chunk))
        flip = R.from_euler("x", 180, degrees=True).as_matrix()
        left_xyz, left_ypr, right_xyz, right_ypr = _split_action_pose(pose_chunk)
        arms = [("right", right_xyz, right_ypr, joint_chunk[:, -17:], extr_all["right_cam"])]
        if left_xyz is not None and joint_chunk.shape[1] >= 34:
            arms.insert(0, ("left", left_xyz, left_ypr, joint_chunk[:, :17], extr_all["left_cam"]))
        for side, xyz_all, ypr_all, joints, extr in arms:
            fk = ofk.get_fk(side)
            T_tcp_hand = ofk.mount_for(side, hand_mount_key)
            for k, step in enumerate(steps):
                s = min(int(step), T - 1)
                R_tcp = R.from_euler("ZYX", ypr_all[s]).as_matrix()
                if tool_flip_x180:
                    R_tcp = R_tcp @ flip
                T_base_tcp = np.eye(4); T_base_tcp[:3, :3] = R_tcp; T_base_tcp[:3, 3] = xyz_all[s]
                P = fk.keypoints(joints[s], T_ref_world=T_base_tcp @ T_tcp_hand)
                pix = cam_frame_to_cam_pixels(ee_pose_to_cam_frame(P, extr), intrinsics)
                # later steps drawn thinner/dimmer so the current pose stands out
                fade = 1.0 - 0.45 * k / max(1, len(steps) - 1)
                c = tuple(int(v * fade) for v in color)
                image = ofk.draw_wireframe(image, pix[:, :2], fk.edges, fk.finger_of_edge, color=c,
                                           thickness=2 if k == 0 else 1, radius=2 if k == 0 else 1, skip=skip)
        return image

    @staticmethod
    def joints_panel(image, gt, pred, panel_w=200, pad=4):
        """Append a panel of per-joint sparklines (GT green, pred red) to the
        right of ``image``. gt/pred: (T, J) chunks of joint targets. Each row
        is scaled to the joint's GT range (+-10% margin) so small motions stay
        visible; the current (t=0) value is the left edge of each row."""
        base = _prepare_viz_image(image)
        H = base.shape[0]
        J = gt.shape[1]
        # bimanual (34 = left 17 | right 17) -> two columns so rows stay legible
        ncol = 2 if J >= 34 else 1
        rows_per_col = int(np.ceil(J / ncol))
        col_w = panel_w if ncol == 1 else int(panel_w * 0.8)
        panel = np.full((H, col_w * ncol, 3), 24, dtype=np.uint8)
        row_h = (H - 2 * pad) / rows_per_col
        xs = np.linspace(pad, col_w - pad - 1, gt.shape[0])
        for j in range(J):
            cx = (j // rows_per_col) * col_w
            y0 = pad + (j % rows_per_col) * row_h
            lo = min(gt[:, j].min(), pred[:, j].min())
            hi = max(gt[:, j].max(), pred[:, j].max())
            m = 0.1 * (hi - lo) + 1e-3
            lo, hi = lo - m, hi + m
            cv2.line(panel, (cx + pad, int(y0 + row_h)), (cx + col_w - pad, int(y0 + row_h)), (60, 60, 60), 1)
            for series, color in ((pred, (220, 40, 40)), (gt, (40, 220, 40))):  # RGB: pred red, GT green
                ys = y0 + row_h - 1 - (series[:, j] - lo) / (hi - lo) * (row_h - 2)
                pts = np.stack([cx + xs, ys], 1).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(panel, [pts], False, color, 1, cv2.LINE_AA)
        label = "hand joints: GT / pred" if ncol == 1 else "hand joints L | R: GT / pred"
        cv2.putText(panel, label, (pad, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)
        return np.concatenate([base, panel], axis=1)
