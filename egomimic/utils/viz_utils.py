import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torchvision.transforms.functional as TF

from egomimic.utils.egomimicUtils import (cam_frame_to_cam_pixels, ee_pose_to_cam_frame, get_gaze_endpoint)
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


# ---------------------------------------------------------------------------
# Drawing helpers relocated here from egomimic/utils/egomimicUtils.py
# (hierarchy pass: utils/ junk-drawer split). Bodies byte-identical to the
# originals; dependency flipped so viz_utils owns the drawing fns and pulls
# only geometry/constants (INTRINSICS, cam_frame_to_cam_pixels,
# ee_pose_to_cam_frame) from egomimicUtils.
# ---------------------------------------------------------------------------

def fmt(v):
    # Convert to flat list of floats no matter the input shape/type
    if isinstance(v, torch.Tensor):
        v = v.flatten().tolist()
    elif isinstance(v, np.ndarray):
        v = v.flatten().tolist()
    return ", ".join(f"{f:.2f}" for f in v)


def draw_annotation_text(
    image: np.ndarray,
    annotation: str,
    font_scale: float = 0.45,
    color: tuple = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """
    Draws annotation text on an image.

    Args:
        image (np.ndarray): Image of shape (H, W, 3) in uint8 format.
        annotation (str): Annotation text to draw.
        font_scale (float): Font size.
        color (tuple): Text color (B, G, R).
        thickness (int): Line thickness.

    Returns:
        image (np.ndarray): Annotated image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Annotation: {annotation}"
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 10, image.shape[0] - baseline - 10

    image = np.ascontiguousarray(image.copy())

    image = cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
    )

    return image


def draw_rotation_text(
    image: np.ndarray,
    gt_rot: torch.Tensor,
    pred_rot: torch.Tensor,
    position: tuple = (340, 20),
    font_scale: float = 0.45,
    color: tuple = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """
    Draws ground truth and predicted rotation vectors on an image.

    Args:
        image (np.ndarray): Image of shape (H, W, 3) in uint8 format.
        gt_rot (torch.Tensor): Rotation vector, shape (3,) or (6,) (dual arm).
        pred_rot (torch.Tensor): Same shape as gt_rot.
        position (tuple): Top-left corner (x, y) for drawing text.
        font_scale (float): Font size.
        color (tuple): Text color (B, G, R).
        thickness (int): Line thickness.

    Returns:
        image (np.ndarray): Annotated image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position

    image = np.ascontiguousarray(image.copy())

    if gt_rot.shape[-1] == 3:
        image = cv2.putText(
            image,
            f"GT rot:    [{fmt(gt_rot)}]",
            (x, y),
            font,
            font_scale,
            color,
            thickness,
        )
        image = cv2.putText(
            image,
            f"Pred rot:  [{fmt(pred_rot)}]",
            (x, y + 20),
            font,
            font_scale,
            color,
            thickness,
        )
    elif gt_rot.shape[-1] == 6:
        image = cv2.putText(
            image,
            f"L GT rot:  [{fmt(gt_rot[0:3])}]",
            (x, y),
            font,
            font_scale,
            color,
            thickness,
        )
        image = cv2.putText(
            image,
            f"L Pred rot:[{fmt(pred_rot[0:3])}]",
            (x, y + 20),
            font,
            font_scale,
            color,
            thickness,
        )
        image = cv2.putText(
            image,
            f"R GT rot:  [{fmt(gt_rot[3:6])}]",
            (x, y + 40),
            font,
            font_scale,
            color,
            thickness,
        )
        image = cv2.putText(
            image,
            f"R Pred rot:[{fmt(pred_rot[3:6])}]",
            (x, y + 60),
            font,
            font_scale,
            color,
            thickness,
        )
    else:
        raise ValueError(f"Unsupported rotation shape: {gt_rot.shape}")

    return image


def draw_actions(
    im, type, color, actions, extrinsics, intrinsics, arm="both", kinematics_solver=None
):
    """
    args:
        im: (H, W, C)
        type: "joints" or "xyz"
        color: ex) "Purples", "Blues", "Greens"
        actions: (N, 6) or (N, 3) if type is "xyz" or (N, 7) or (N, 14) if type is "joints"
        extrinsics: dict with keys "left" and "right" with values (4, 4)
        intrinsics: (3, 4)
        arm: "both", "left", "right"
    returns
        im: (H, W, C)
    """
    if type == "joints" and kinematics_solver is None:
        raise ValueError("kinematics_solver is required for joints actions")
    if type == "joints":
        if arm == "both":
            right_actions = kinematics_solver.fk_pos(actions[:, 7:13])
            right_actions_drawable = ee_pose_to_cam_frame(
                right_actions, extrinsics["right"]
            )
            left_actions = kinematics_solver.fk_pos(actions[:, :6])
            left_actions_drawable = ee_pose_to_cam_frame(
                left_actions, extrinsics["left"]
            )
            actions_drawable = np.concatenate(
                (left_actions_drawable, right_actions_drawable), axis=0
            )
        elif arm == "right":
            right_actions = kinematics_solver.fk_pos(actions[:, 7:13])
            right_actions_drawable = ee_pose_to_cam_frame(
                right_actions, extrinsics["right"]
            )
            actions_drawable = right_actions_drawable
        elif arm == "left":
            left_actions = kinematics_solver.fk_pos(actions[:, :6])
            left_actions_drawable = ee_pose_to_cam_frame(
                left_actions, extrinsics["left"]
            )
            actions_drawable = left_actions_drawable
    else:
        actions = actions.reshape(-1, 3)
        actions_drawable = actions

    actions_drawable = cam_frame_to_cam_pixels(actions_drawable, intrinsics)
    im = draw_dot_on_frame(im, actions_drawable, show=False, palette=color)

    return im


def draw_dot_on_frame(frame, pixel_vals, show=True, palette="Purples", dot_size=5):
    """
    frame: (H, W, C) numpy array
    pixel_vals: (N, 2) numpy array of pixel values to draw on frame
    Drawn in light to dark order
    """
    frame = frame.astype(np.uint8).copy()
    if isinstance(pixel_vals, tuple):
        pixel_vals = [pixel_vals]

    # get purples color palette, and color the circles accordingly
    color_palette = plt.get_cmap(palette)
    color_palette = color_palette(np.linspace(0, 1, len(pixel_vals)))
    color_palette = (color_palette[:, :3] * 255).astype(np.uint8)
    color_palette = color_palette.tolist()

    for i, pixel_val in enumerate(pixel_vals):
        try:
            frame = cv2.circle(
                frame,
                (int(pixel_val[0]), int(pixel_val[1])),
                dot_size,
                color_palette[i],
                -1,
            )
        except Exception:
            print("Got bad pixel_val: ", pixel_val)
        if show:
            plt.imshow(frame)
            plt.show()

    return frame


def miniviewer(frame, goal_frame, location="top_right"):
    """
    overlay goal_frame in a corner of frame
    frame: (H, W, C) numpy array
    goal_frame: (H, W, C) numpy array
    location: "top_right", "top_left", "bottom_left", "bottom_right"

    return frame with goal_frame in top right corner (1/4 original size)

    resize using TF
    """
    frame = frame.copy()
    goal_frame = goal_frame.copy()
    if isinstance(frame, np.ndarray):
        frame = torch.from_numpy(frame)
    if isinstance(goal_frame, np.ndarray):
        goal_frame = torch.from_numpy(goal_frame)

    goal_frame = goal_frame.permute((2, 0, 1))
    frame = frame.permute((2, 0, 1))

    goal_frame = TF.resize(goal_frame, (frame.shape[1] // 4, frame.shape[2] // 4))
    if location == "top_right":
        frame[:, : goal_frame.shape[1], -goal_frame.shape[2] :] = goal_frame
    elif location == "top_left":
        frame[:, : goal_frame.shape[1], : goal_frame.shape[2]] = goal_frame
    elif location == "bottom_left":
        frame[:, -goal_frame.shape[1] :, : goal_frame.shape[2]] = goal_frame
    elif location == "bottom_right":
        frame[:, -goal_frame.shape[1] :, -goal_frame.shape[2] :] = goal_frame
    # frame[:, :goal_frame.shape[1], -goal_frame.shape[2]:] = goal_frame
    return frame.permute((1, 2, 0)).numpy()
