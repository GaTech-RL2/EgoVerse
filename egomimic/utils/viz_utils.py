import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.utils.egomimicUtils import (
    cam_frame_to_cam_pixels,
    draw_actions,
    draw_dot_on_frame,
    get_gaze_endpoint,
)
from egomimic.utils.pose_utils import _split_action_pose, _split_keypoints


class ColorPalette:
    Blues = "Blues"
    Greens = "Greens"
    Reds = "Reds"
    Oranges = "Oranges"
    Purples = "Purples"
    Greys = "Greys"

    @classmethod
    def is_valid(cls, name: str) -> bool:
        return name in vars(cls).values()

    @classmethod
    def to_rgb(cls, cmap_name: str, value: float = 0.7) -> tuple[int, int, int]:
        """Convert a ColorPalette cmap name to an RGB tuple (0-255).
        value: 0-1, where higher = darker shade."""
        rgba = plt.get_cmap(cmap_name)(value)
        return tuple(int(c * 255) for c in rgba[:3])


def _prepare_viz_image(img):
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


def _format_rotation_values(rot):
    rot = np.asarray(rot).reshape(-1)
    return ", ".join(f"{value:.2f}" for value in rot)


def _extract_rotation_for_txt(actions):
    actions = np.asarray(actions)
    while actions.ndim > 1:
        actions = actions[0]

    _, left_ypr, _, right_ypr = _split_action_pose(actions)
    return np.asarray(left_ypr).reshape(-1), np.asarray(right_ypr).reshape(-1)


def _viz_rotation_txt(image, actions, **kwargs):
    vis = _prepare_viz_image(image).copy()
    left_rot, right_rot = _extract_rotation_for_txt(actions)

    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = kwargs.get("rotation_font_scale")
    if font_scale is None:
        font_scale = max(0.4, h / 900)
    color = kwargs.get("rotation_text_color", (255, 255, 255))
    thickness = kwargs.get("rotation_text_thickness")
    if thickness is None:
        thickness = max(1, int(h / 450))
    margin = kwargs.get("rotation_text_margin")
    if margin is None:
        margin = max(10, int(h * 0.03))
    line_spacing = (
        kwargs.get("rotation_text_line_spacing")
        if kwargs.get("rotation_text_line_spacing") is not None
        else max(6, int(h * 0.012))
    )
    role = kwargs.get("rotation_text_role")
    if role is None:
        color_name = kwargs.get("color")
        if color_name == ColorPalette.Greens:
            role = "gt"
        elif color_name == ColorPalette.Reds:
            role = "pred"
    prefix = kwargs.get("rotation_text_prefix")
    start_line = kwargs.get("rotation_text_start_line")
    if prefix is None and role in ("gt", "pred"):
        prefix = role.upper()
    if start_line is None and role in ("gt", "pred"):
        start_line = 0 if role == "gt" else 2
    prefix = (prefix or "").strip()
    start_line = max(0, int(0 if start_line is None else start_line))
    label_prefix = f"{prefix} " if prefix else ""

    lines = [
        f"{label_prefix}L rot: [{_format_rotation_values(left_rot)}]",
        f"{label_prefix}R rot: [{_format_rotation_values(right_rot)}]",
    ]
    line_metrics = [
        cv2.getTextSize(line, font, font_scale, thickness) for line in lines
    ]
    line_height = max(text_h + baseline for (_, text_h), baseline in line_metrics)
    y = margin + start_line * (line_height + line_spacing)

    for line, ((text_w, text_h), baseline) in zip(lines, line_metrics, strict=True):
        x = max(margin, w - margin - text_w)
        y += text_h
        cv2.putText(
            vis,
            line,
            (x, y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            line,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += baseline + line_spacing

    return vis


def _viz_traj(image, actions, intrinsics, **kwargs):
    color = kwargs.get("color", "Blues")
    alpha = kwargs.get("alpha", 1.0)
    if not ColorPalette.is_valid(color):
        raise ValueError(f"Invalid color palette: {color}")

    image = _prepare_viz_image(image)
    left_xyz, _, right_xyz, _ = _split_action_pose(actions)

    base = image.copy()
    overlay = draw_actions(
        base.copy(),
        type="xyz",
        color=color,
        actions=left_xyz,
        extrinsics=None,
        intrinsics=intrinsics,
        arm="left",
    )
    overlay = draw_actions(
        overlay,
        type="xyz",
        color=color,
        actions=right_xyz,
        extrinsics=None,
        intrinsics=intrinsics,
        arm="right",
    )
    if alpha < 1.0:
        vis = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)
    else:
        vis = overlay
    return vis


def _viz_axes(image, actions, intrinsics, axis_len_m=0.04, **kwargs):
    alpha = kwargs.get("alpha", 1.0)
    image = _prepare_viz_image(image)
    left_xyz, left_ypr, right_xyz, right_ypr = _split_action_pose(actions)
    base = image.copy()
    vis = base.copy()

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

    def _draw_rotation_at_anchor(
        frame, xyz_seq, ypr_seq, label, anchor_color, **kwargs
    ):
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
        # Painter's algorithm: draw the axes far->near (by each tip's camera-z
        # depth) so the axis closest to the camera ends up on top, instead of a
        # fixed x->y->z order that can hide a near axis behind a far one.
        axis_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        draw_order = sorted((1, 2, 3), key=lambda i: -float(axis_points_cam[i][2]))
        for i in draw_order:
            x1, y1 = pts[i]
            if 0 <= x1 < w and 0 <= y1 < h:
                cv2.line(frame, (x0, y0), (x1, y1), axis_colors[i - 1], 2)
                cv2.circle(frame, (x1, y1), 2, axis_colors[i - 1], -1)

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

    vis = _draw_rotation_at_anchor(vis, left_xyz, left_ypr, "L rot", (255, 180, 80))
    vis = _draw_rotation_at_anchor(vis, right_xyz, right_ypr, "R rot", (80, 180, 255))
    vis = _draw_axis_color_legend(vis)
    if alpha < 1.0:
        vis = cv2.addWeighted(vis, alpha, base, 1.0 - alpha, 0)
    return vis


def _viz_gaze(
    image,
    gaze_data,
    intrinsics,
    t_rgb_cpf,
    palette="Purples",
    dot_size=8,
    no_gaze_sentinel=-100,
    **kwargs,
):
    """Project the gaze endpoint (yaw, pitch, depth in CPF) onto the image."""
    image = _prepare_viz_image(image)
    gaze = np.asarray(gaze_data).reshape(-1)
    if gaze.size < 3 or float(gaze[0]) == float(no_gaze_sentinel):
        return image.copy()

    yaw, pitch, depth = float(gaze[0]), float(gaze[1]), float(gaze[2])
    endpoint_cam = get_gaze_endpoint(yaw, pitch, depth, t_rgb_cpf)[None, :]
    pixel = cam_frame_to_cam_pixels(endpoint_cam, intrinsics)
    return draw_dot_on_frame(
        image.copy(), pixel, show=False, palette=palette, dot_size=dot_size
    )


def _viz_keypoints(
    image,
    actions,
    intrinsics,
    edges,
    colors,
    edge_ranges,
    dot_color=None,
    **kwargs,
):
    """Visualize all 21 MANO keypoints per hand, projected onto the image."""
    alpha = kwargs.get("alpha", 1.0)
    image = _prepare_viz_image(image)

    base = image.copy()
    vis = base.copy()
    h, w = vis.shape[:2]

    if actions.shape[-1] == 140:
        _, _, left_keypoints, _, _, right_keypoints = _split_keypoints(
            actions, wrist_in_data=True
        )
    elif actions.shape[-1] == 138:
        _, _, left_keypoints, _, _, right_keypoints = _split_keypoints(
            actions, wrist_in_data=True, is_quat=False
        )
    else:
        left_keypoints, right_keypoints = _split_keypoints(actions, wrist_in_data=False)
    keypoints = {}
    keypoints["left"] = left_keypoints.reshape(-1, 3)
    keypoints["right"] = right_keypoints.reshape(-1, 3)
    _default_dot_colors = {"left": (0, 120, 255), "right": (255, 80, 0)}
    for hand in ("left", "right"):
        hand_dot_color = (
            dot_color if dot_color is not None else _default_dot_colors[hand]
        )
        kps_cam = keypoints[hand]
        # Camera frame -> pixels
        kps_px = cam_frame_to_cam_pixels(kps_cam, intrinsics)  # (42, 3+) 21 per arm

        # Identify valid keypoints (z > 0 and in image bounds)
        valid = kps_cam[:, 2] > 0.01
        valid &= (kps_px[:, 0] >= 0) & (kps_px[:, 0] < w)
        valid &= (kps_px[:, 1] >= 0) & (kps_px[:, 1] < h)

        # Draw skeleton edges (colored by finger)
        for finger, start, end in edge_ranges:
            color = colors[finger]
            for edge_idx in range(start, end):
                i, j = edges[edge_idx]
                if valid[i] and valid[j]:
                    p1 = (int(kps_px[i, 0]), int(kps_px[i, 1]))
                    p2 = (int(kps_px[j, 0]), int(kps_px[j, 1]))
                    cv2.line(vis, p1, p2, color, 2)

        # Draw keypoint dots on top
        for k in range(21):
            if valid[k]:
                center = (int(kps_px[k, 0]), int(kps_px[k, 1]))
                cv2.circle(vis, center, 4, hand_dot_color, -1)
                cv2.circle(vis, center, 4, (255, 255, 255), 1)  # white border

        # Label wrist
        if valid[0]:
            wrist_px = (int(kps_px[0, 0]) + 6, int(kps_px[0, 1]) - 6)
            cv2.putText(
                vis,
                f"{hand[0].upper()}",
                wrist_px,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                hand_dot_color,
                2,
            )

    if alpha < 1.0:
        vis = cv2.addWeighted(vis, alpha, base, 1.0 - alpha, 0)
    return vis


def _viz_keypoints_horizon_trace(
    image,
    actions,
    intrinsics,
    colors,
    wrist_color=None,
    fingertip_indices=(4, 8, 12, 16, 20),
    finger_names=("thumb", "index", "middle", "ring", "pinky"),
    **kwargs,
):
    """Visualize the horizon TRACE of the 5 fingertips + wrist per hand.

    Actions layout (per timestep, per hand):
      [wrist_xyz(3), wrist_ypr(3), keypoints(21*3=63)]
    Concatenated bimanual = 138-D: [L wrist(6), L kps(63), R wrist(6), R kps(63)].

    For each hand, for each fingertip index in ``fingertip_indices`` (default is
    MANO tips [4, 8, 12, 16, 20]) and for the wrist (kps index 0 OR the wrist
    xyz slot — we use the wrist xyz slot which is the commanded wrist position
    at every timestep of the horizon), we project each timestep into pixel
    space and draw a polyline connecting them. Start (t=0) and end (t=H-1)
    markers show the trace direction.

    Args:
        actions: shape (H, 138). Other cases fall through gracefully.
        colors: dict[finger_name -> BGR tuple], same shape as Human.FINGER_COLORS.
        wrist_color: BGR tuple for the wrist trace (defaults to Human.DOT_COLOR).
    """
    alpha = kwargs.get("alpha", 1.0)
    image = _prepare_viz_image(image)
    base = image.copy()
    vis = base.copy()
    h, w = vis.shape[:2]

    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[-1] != 138:
        # Fall through cleanly for unsupported layouts; nothing to draw.
        if alpha < 1.0:
            vis = cv2.addWeighted(vis, alpha, base, 1.0 - alpha, 0)
        return vis

    # 138-D layout: [L wrist(6), L kps(63), R wrist(6), R kps(63)]
    left_wrist_xyz = actions[:, 0:3]
    left_kps = actions[:, 6:69].reshape(-1, 21, 3)
    right_wrist_xyz = actions[:, 69:72]
    right_kps = actions[:, 75:138].reshape(-1, 21, 3)

    if wrist_color is None:
        wrist_color = (255, 165, 0)

    hands = [
        ("left", left_wrist_xyz, left_kps),
        ("right", right_wrist_xyz, right_kps),
    ]

    def _project_and_draw(points_cam, color, thickness=2, marker_size=4):
        """points_cam: (H, 3). Project + polyline + start/end markers."""
        if points_cam.shape[0] == 0:
            return
        px = cam_frame_to_cam_pixels(points_cam, intrinsics)  # (H, 3+)
        valid = points_cam[:, 2] > 0.01
        valid &= (px[:, 0] >= 0) & (px[:, 0] < w)
        valid &= (px[:, 1] >= 0) & (px[:, 1] < h)
        pts = np.round(px[:, :2]).astype(np.int32)

        # Polyline: connect consecutive valid points.
        prev = None
        for t in range(pts.shape[0]):
            if not valid[t]:
                prev = None
                continue
            cur = (int(pts[t, 0]), int(pts[t, 1]))
            if prev is not None:
                cv2.line(vis, prev, cur, color, thickness, cv2.LINE_AA)
            prev = cur

        # Start marker (filled circle) at first valid, end marker (open square) at last valid.
        first_t = next((t for t in range(pts.shape[0]) if valid[t]), None)
        last_t = next((t for t in range(pts.shape[0] - 1, -1, -1) if valid[t]), None)
        if first_t is not None:
            cv2.circle(
                vis,
                (int(pts[first_t, 0]), int(pts[first_t, 1])),
                marker_size,
                color,
                -1,
                cv2.LINE_AA,
            )
        if last_t is not None and last_t != first_t:
            ex, ey = int(pts[last_t, 0]), int(pts[last_t, 1])
            r = marker_size + 1
            cv2.rectangle(
                vis, (ex - r, ey - r), (ex + r, ey + r), color, 2, cv2.LINE_AA
            )

    for hand_name, wrist_xyz, kps in hands:
        # Wrist trace (from wrist xyz slot in the action).
        _project_and_draw(wrist_xyz, wrist_color, thickness=2, marker_size=5)
        # Fingertip traces, one per finger.
        for tip_idx, finger_name in zip(fingertip_indices, finger_names, strict=True):
            color = colors.get(finger_name, (200, 200, 200))
            tip_traj = kps[:, tip_idx, :]  # (H, 3)
            _project_and_draw(tip_traj, color, thickness=2, marker_size=4)

    if alpha < 1.0:
        vis = cv2.addWeighted(vis, alpha, base, 1.0 - alpha, 0)
    return vis


# ---------------------------------------------------------------------------
# 140-D shared-head cross-embodiment viz helpers.
#
# The 140-D layout (see cotrain_mecka_eva_fold_clothes_140d.yaml and
# HumanKeypointsTo140D / EvaCartesianTo140D transforms) is:
#   [0:3]   left wrist/EE XYZ
#   [3:6]   left wrist/EE YPR
#   [6:7]   left gripper
#   [7:70]  left MANO keypoints  (21 * 3 = 63)
#   [70:73] right wrist/EE XYZ
#   [73:76] right wrist/EE YPR
#   [76:77] right gripper
#   [77:140] right MANO keypoints (21 * 3 = 63)
#
# The two functions below let us render the *cross-embodiment* prediction:
# on eva samples we can still visualize the model's predicted MANO keypoints,
# and on human samples we can still show the model's predicted gripper value.
# ---------------------------------------------------------------------------
_140D_KPS_IDX = list(range(7, 70)) + list(range(77, 140))
_140D_GRIP_IDX = [6, 76]


def _split_140d_to_138d_keypoints(actions_140d):
    """Slice a (H, 140) action chunk down to the (H, 138) keypoint layout
    ``[L wrist(6), L kps(63), R wrist(6), R kps(63)]`` consumed by
    ``_viz_keypoints_horizon_trace``.
    """
    actions_140d = np.asarray(actions_140d)
    if actions_140d.ndim != 2 or actions_140d.shape[-1] != 140:
        return None
    # 138-D "actions_keypoints" reconstruction slice (drops the two gripper
    # slots at 6 and 76).
    idx = (
        list(range(0, 6))
        + list(range(7, 70))
        + list(range(70, 76))
        + list(range(77, 140))
    )
    return actions_140d[:, idx]


def _viz_cartesian_with_keypoint_overlay(
    image, actions, intrinsics, colors=None, wrist_color=None, **kwargs
):
    """Eva cross-embodiment viz: draw the 12-D EE trajectory *and* overlay the
    predicted MANO keypoint fingertip/wrist traces from the same 140-D action.

    ``actions`` is (H, 140). We build a 14-D actions_cartesian chunk for the
    standard ``_viz_traj`` (using EE slots [0:6] ⊕ [70:76] with a zero
    gripper slot inserted) and reuse ``_viz_keypoints_horizon_trace`` for the
    fingertip/wrist overlay from the keypoint slice.
    """
    alpha = kwargs.get("alpha", 1.0)
    color = kwargs.get("color", "Blues")

    actions_arr = np.asarray(actions)
    image_arr = _prepare_viz_image(image)

    # Fall through cleanly if the 140-D layout is not what we expect.
    if actions_arr.ndim != 2 or actions_arr.shape[-1] != 140:
        base = image_arr.copy()
        return base

    # Assemble a 14-D chunk (matches the actions_cartesian layout used by
    # _split_action_pose inside _viz_traj: [L xyz+ypr+gripper, R xyz+ypr+gripper]).
    H = actions_arr.shape[0]
    actions_14d = np.zeros((H, 14), dtype=actions_arr.dtype)
    actions_14d[:, 0:6] = actions_arr[:, 0:6]  # left EE xyz+ypr
    actions_14d[:, 6:7] = actions_arr[:, 6:7]  # left gripper
    actions_14d[:, 7:13] = actions_arr[:, 70:76]  # right EE xyz+ypr
    actions_14d[:, 13:14] = actions_arr[:, 76:77]  # right gripper

    # Base = the existing eva cartesian trajectory viz.
    vis = _viz_traj(image_arr, actions_14d, intrinsics, **kwargs)

    # Overlay = the horizon fingertip/wrist trace on the 138-D keypoint slice.
    actions_138d = _split_140d_to_138d_keypoints(actions_arr)
    if actions_138d is None:
        return vis

    # Lighter fingertip/wrist palette so the keypoint overlay stays visually
    # distinct from the EE trajectory (which already owns the "color" palette).
    # Use complementary hues, thinner polylines.
    if colors is None:
        if color == "Greens":
            # gt-pass: pale greens for keypoints (so EE greens still dominate)
            colors = {
                "thumb": (180, 255, 180),
                "index": (150, 255, 150),
                "middle": (120, 255, 120),
                "ring": (100, 240, 100),
                "pinky": (80, 220, 80),
            }
            _wrist = (200, 255, 200)
        elif color == "Reds":
            # pred-pass: pale reds
            colors = {
                "thumb": (180, 180, 255),
                "index": (150, 150, 255),
                "middle": (120, 120, 255),
                "ring": (100, 100, 240),
                "pinky": (80, 80, 220),
            }
            _wrist = (200, 200, 255)
        else:
            colors = {
                "thumb": (255, 100, 100),
                "index": (100, 255, 100),
                "middle": (100, 100, 255),
                "ring": (255, 255, 100),
                "pinky": (255, 100, 255),
            }
            _wrist = (255, 165, 0)
    else:
        _wrist = wrist_color if wrist_color is not None else (255, 165, 0)

    # Route the same alpha through, but draw thin (thickness=1) via kwargs.
    # ``_viz_keypoints_horizon_trace`` handles its own alpha blending.
    vis = _viz_keypoints_horizon_trace(
        image=vis,
        actions=actions_138d,
        intrinsics=intrinsics,
        colors=colors,
        wrist_color=_wrist,
        alpha=alpha if alpha < 1.0 else 0.85,
    )
    return vis


def _viz_keypoints_traj_with_gripper_tag(
    image, actions, intrinsics, colors, wrist_color=None, **kwargs
):
    """Human cross-embodiment viz: draw the wrist/fingertip horizon trace
    (identical to ``_viz_keypoints_horizon_trace``) with a top-right text tag
    showing the model's predicted (or gt) gripper state at t=0.

    ``actions`` is (H, 140). We slice out the 138-D keypoint layout for the
    base viz and read the 2-D gripper slice ``[6, 76]`` for the tag.
    """
    actions_arr = np.asarray(actions)
    image_arr = _prepare_viz_image(image)

    if actions_arr.ndim != 2 or actions_arr.shape[-1] != 140:
        return image_arr.copy()

    actions_138d = _split_140d_to_138d_keypoints(actions_arr)
    if actions_138d is None:
        return image_arr.copy()

    # Base = the existing wrist + fingertip polyline viz.
    vis = _viz_keypoints_horizon_trace(
        image=image_arr,
        actions=actions_138d,
        intrinsics=intrinsics,
        colors=colors,
        wrist_color=wrist_color,
        **kwargs,
    )

    # Gripper tag (top-right). Read the t=0 gripper for both hands.
    left_grip = float(actions_arr[0, 6])
    right_grip = float(actions_arr[0, 76])

    color = kwargs.get("color")
    if color == "Greens":
        text_color = (60, 220, 60)
        role = "GT"
    elif color == "Reds":
        text_color = (60, 60, 220)
        role = "PR"
    else:
        text_color = (255, 255, 255)
        role = ""

    h, w = vis.shape[:2]
    # HERSHEY_SIMPLEX is the closest to a monospaced OpenCV font — pair it
    # with fixed-width numeric formatting for a stable columnar look.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, h / 900)
    thickness = max(1, int(h / 450))
    margin = max(10, int(h * 0.03))
    line_spacing = max(6, int(h * 0.012))
    label = (
        f"{role} L:{left_grip:+.2f} R:{right_grip:+.2f}"
        if role
        else f"L:{left_grip:+.2f} R:{right_grip:+.2f}"
    )

    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    # Position tag in the top-right, staggered vertically for GT vs PR so both
    # passes are readable when Embodiment.viz_gt_preds composes both.
    if role == "GT":
        y = margin + text_h
    elif role == "PR":
        y = margin + 2 * (text_h + line_spacing)
    else:
        y = margin + text_h
    x = max(margin, w - margin - text_w)

    cv2.putText(
        vis, label, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    cv2.putText(
        vis, label, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA
    )
    return vis


def _wrap_text(text, font, font_scale, thickness, max_width):
    """Word-wrap *text* so each line fits within *max_width* pixels."""
    words = text.split()
    if not words:
        return [""]
    lines, current = [], words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        tw, _ = cv2.getTextSize(candidate, font, font_scale, thickness)[0]
        if tw <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _viz_annotations(image, annotations: list[str], **kwargs):
    """Render a list of text annotations onto the image."""
    image = _prepare_viz_image(image)
    vis = image.copy()
    h, w = vis.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, h / 800)
    thickness = max(1, int(h / 400))
    line_height = int(font_scale * 30)
    margin = int(h * 0.02)
    max_text_width = w - 2 * margin

    flat = []
    for item in annotations:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)

    wrapped_lines = []
    for text in flat:
        wrapped_lines.extend(
            _wrap_text(text, font, font_scale, thickness, max_text_width)
        )

    y = h - margin - len(wrapped_lines) * line_height
    for line in wrapped_lines:
        y += line_height
        cv2.putText(
            vis,
            line,
            (margin, y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            line,
            (margin, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return vis


def save_image(image: np.ndarray, path: str) -> None:
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
