#!/usr/bin/env python3
"""
Visualize LeRobot Dataset with Camera Projection
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "lerobot"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]

LEFT_HAND_COLOR = (0, 255, 0)
RIGHT_HAND_COLOR = (0, 0, 255)


def project_3d_to_2d(x, y, z, focal_length, cx, cy):
    """
    Project 3D point to 2D using perspective projection.

    Args:
        x, y, z: 3D coordinates in camera frame
        focal_length: Camera focal length in pixels
        cx, cy: Principal point (image center)

    Returns:
        (pixel_x, pixel_y): 2D pixel coordinates
    """
    if z <= 0:
        return float("nan"), float("nan")

    pixel_x = (x * focal_length) / z + cx
    pixel_y = (y * focal_length) / z + cy

    return pixel_x, pixel_y


def project_keypoints_to_2d(keypoints_3d, focal_length, cx, cy):
    """
    Project 3D keypoints to 2D pixels.

    Args:
        keypoints_3d: (21, 3) array of 3D keypoints in camera frame
        focal_length: Camera focal length
        cx, cy: Principal point

    Returns:
        (21, 2) array of 2D pixel coordinates
    """
    keypoints_2d = np.zeros((keypoints_3d.shape[0], 2))

    for i, point_3d in enumerate(keypoints_3d):
        x, y = project_3d_to_2d(
            point_3d[0], point_3d[1], point_3d[2], focal_length, cx, cy
        )
        keypoints_2d[i] = [x, y]

    return keypoints_2d


def draw_hand_skeleton(img, keypoints_2d, color, img_width, img_height):
    """
    Draw hand skeleton on image.

    Args:
        img: Image to draw on
        keypoints_2d: (21, 2) array of 2D pixel coordinates
        color: BGR color tuple
        img_width, img_height: Image dimensions for bounds checking
    """
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = keypoints_2d[start_idx]
        end = keypoints_2d[end_idx]

        if (
            not (
                np.isnan(start[0])
                or np.isnan(start[1])
                or np.isnan(end[0])
                or np.isnan(end[1])
            )
            and 0 <= start[0] < img_width
            and 0 <= start[1] < img_height
            and 0 <= end[0] < img_width
            and 0 <= end[1] < img_height
        ):
            cv2.line(
                img,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                color,
                2,
            )

    for i, (x, y) in enumerate(keypoints_2d):
        if (
            not (np.isnan(x) or np.isnan(y))
            and 0 <= x < img_width
            and 0 <= y < img_height
        ):
            radius = 8 if i == 0 else 5
            cv2.circle(img, (int(x), int(y)), radius, color, -1)


def load_camera_intrinsics(dataset_path):
    """
    Load camera intrinsics from dataset metadata.

    Returns:
        dict with keys: w, h, fl_x, fl_y, cx, cy, k1, k2, p1, p2
    """
    info_path = Path(dataset_path) / "meta" / "info.json"

    if not info_path.exists():
        print(f"Warning: {info_path} not found, using default intrinsics")
        return {
            "w": 1920,
            "h": 1080,
            "fl_x": 756.0,
            "fl_y": 756.0,
            "cx": 960.0,
            "cy": 540.0,
        }

    with open(info_path, "r") as f:
        info = json.load(f)

    intrinsics = info.get("intrinsics", {})

    if not intrinsics:
        print("Warning: No intrinsics in info.json, using defaults")
        intrinsics = {
            "w": 1920,
            "h": 1080,
            "fl_x": 756.0,
            "fl_y": 756.0,
            "cx": 960.0,
            "cy": 540.0,
        }

    return intrinsics


def visualize_frame_with_hands(dataset, frame_idx, intrinsics, output_path=None):
    """
    Visualize a single frame with projected hand keypoints.

    Args:
        dataset: LeRobotDataset
        frame_idx: Frame index to visualize
        intrinsics: Camera intrinsics dict
        output_path: Path to save image (optional)

    Returns:
        Annotated image
    """
    sample = dataset[frame_idx]

    img_tensor = sample["observations.images.front_img_1"]
    img = img_tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img_height, img_width = img.shape[:2]

    focal_length = intrinsics.get("fl_x", 756.0)
    cx = intrinsics.get("cx", img_width / 2.0)
    cy = intrinsics.get("cy", img_height / 2.0)

    keypoints_world_flat = sample["actions_ee_keypoints_world"][0].cpu().numpy()

    keypoints_world = keypoints_world_flat.reshape(2, 21, 3)

    left_hand_world = keypoints_world[0]
    right_hand_world = keypoints_world[1]

    def mecka_to_opencv(keypoints):
        """Convert from Mecka coords to OpenCV coords"""
        opencv_kp = np.zeros_like(keypoints)
        opencv_kp[:, 0] = -keypoints[:, 1]
        opencv_kp[:, 1] = -keypoints[:, 2]
        opencv_kp[:, 2] = keypoints[:, 0]
        return opencv_kp

    left_hand_opencv = mecka_to_opencv(left_hand_world)
    right_hand_opencv = mecka_to_opencv(right_hand_world)

    left_hand_2d = project_keypoints_to_2d(left_hand_opencv, focal_length, cx, cy)
    right_hand_2d = project_keypoints_to_2d(right_hand_opencv, focal_length, cx, cy)

    draw_hand_skeleton(img, left_hand_2d, LEFT_HAND_COLOR, img_width, img_height)
    draw_hand_skeleton(img, right_hand_2d, RIGHT_HAND_COLOR, img_width, img_height)

    cv2.putText(
        img,
        f"Frame: {frame_idx}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        img, "Left Hand", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, LEFT_HAND_COLOR, 2
    )

    cv2.putText(
        img, "Right Hand", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RIGHT_HAND_COLOR, 2
    )

    cv2.putText(
        img,
        "21 Hand Keypoints",
        (20, img_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if output_path:
        cv2.imwrite(output_path, img)

    return img


def main():
    print("=" * 60)
    print("VISUALIZING DATASET WITH CAMERA PROJECTION")
    print("=" * 60)
    print()

    print("Loading dataset...")
    dataset = LeRobotDataset(
        repo_id="folding_shirt", root="./lerobot_output", local_files_only=True
    )

    print(f"✅ Dataset loaded: {len(dataset)} frames")

    intrinsics = load_camera_intrinsics("./lerobot_output")
    print("\nCamera intrinsics:")
    print(f"  Resolution: {intrinsics.get('w', 1920)}x{intrinsics.get('h', 1080)}")
    print(f"  Focal length: {intrinsics.get('fl_x', 756.0)} px")
    print(
        f"  Principal point: ({intrinsics.get('cx', 960.0)}, {intrinsics.get('cy', 540.0)})"
    )
    print()

    output_dir = "./visualization_projected"
    os.makedirs(output_dir, exist_ok=True)

    sample_indices = [0, 100, 300, 500, 700, 900, 1100, 1300, 1452]

    print("Generating visualizations...")
    for idx in sample_indices:
        output_path = os.path.join(output_dir, f"frame_{idx:04d}_projected.jpg")
        visualize_frame_with_hands(dataset, idx, intrinsics, output_path)
        print(f"  ✓ Frame {idx} -> {output_path}")

    print()
    print("=" * 60)
    print("✅ DONE!")
    print("=" * 60)
    print(f"Visualizations saved to: {output_dir}/")
    print()
    print("Legend:")
    print("  🟢 Green = Left hand (21 keypoints)")
    print("  🔴 Red = Right hand (21 keypoints)")
    print("  Lines = Hand skeleton connections")


if __name__ == "__main__":
    main()
