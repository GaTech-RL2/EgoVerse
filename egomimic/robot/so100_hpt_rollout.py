from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_CONVERSION_METADATA = Path(
    "/Users/zxwang/.cache/huggingface/lerobot/zxwang/so100-put-apricot-fps-20-ee-cam/meta/ee_pose_conversion.json"
)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {transform.shape}")
    out = np.eye(4, dtype=np.float64)
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def pose7_ypr_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected pose shape (7,), got {pose.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_euler("ZYX", pose[3:6], degrees=False).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose7_rotvec(transform: np.ndarray, gripper: float) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got {transform.shape}")
    rotvec = R.from_matrix(transform[:3, :3]).as_rotvec()
    return np.concatenate(
        [transform[:3, 3], rotvec, np.array([float(gripper)], dtype=np.float64)],
        axis=0,
    ).astype(np.float32)


def load_base_T_camera(conversion_metadata_path: str | Path) -> np.ndarray:
    payload = json.loads(Path(conversion_metadata_path).read_text(encoding="utf-8"))
    calibration = payload.get("calibration", {})
    if "T_cam_base" not in calibration:
        raise KeyError(
            f"{conversion_metadata_path} does not contain calibration.T_cam_base"
        )
    cam_T_base = np.asarray(calibration["T_cam_base"], dtype=np.float64)
    return invert_transform(cam_T_base)


def camera_chunk_ypr_to_base_rotvec(
    actions_camera_ypr: np.ndarray,
    *,
    base_T_camera: np.ndarray,
) -> np.ndarray:
    actions = np.asarray(actions_camera_ypr, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected action chunk shape (H, 7), got {actions.shape}")
    base_targets = []
    for pose in actions:
        camera_T_ee = pose7_ypr_to_matrix(pose)
        base_T_ee = base_T_camera @ camera_T_ee
        base_targets.append(matrix_to_pose7_rotvec(base_T_ee, gripper=float(pose[6])))
    return np.stack(base_targets, axis=0)


def select_receding_horizon_steps(action_chunk: np.ndarray, execute_steps: int) -> np.ndarray:
    if execute_steps <= 0:
        raise ValueError(f"execute_steps must be positive, got {execute_steps}")
    return np.asarray(action_chunk)[: int(execute_steps)]


def write_targets_json(path: str | Path, targets: np.ndarray, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "targets_base_rotvec": np.asarray(targets, dtype=np.float32).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert SO100 HPT camera-frame YPR predictions into base-frame rotvec "
            "targets for manual receding-horizon rollout."
        )
    )
    parser.add_argument(
        "--prediction_npy",
        required=True,
        help="Path to a .npy action chunk with shape (H, 7): xyz+yaw/pitch/roll+gripper in camera frame.",
    )
    parser.add_argument(
        "--conversion_metadata",
        default=str(DEFAULT_CONVERSION_METADATA),
        help="LeRobot EE conversion metadata containing calibration.T_cam_base.",
    )
    parser.add_argument(
        "--execute_steps",
        type=int,
        default=3,
        help="Number of leading predicted steps to execute before replanning.",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Where to write base-frame rotvec targets for the manual robot runner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    action_chunk = np.load(args.prediction_npy)
    selected = select_receding_horizon_steps(action_chunk, args.execute_steps)
    base_T_camera = load_base_T_camera(args.conversion_metadata)
    base_targets = camera_chunk_ypr_to_base_rotvec(
        selected,
        base_T_camera=base_T_camera,
    )
    write_targets_json(
        args.output_json,
        base_targets,
        metadata={
            "prediction_npy": str(args.prediction_npy),
            "conversion_metadata": str(args.conversion_metadata),
            "execute_steps": int(args.execute_steps),
            "target_layout": ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"],
            "target_frame": "so100_base",
        },
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "targets_shape": list(base_targets.shape),
                "execute_steps": int(args.execute_steps),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
