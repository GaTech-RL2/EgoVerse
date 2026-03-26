"""
Convert Eva HDF5 episodes to Zarr format.

Mirrors the main(args) interface of eva_to_lerobot.py so that
run_eva_conversion.py can swap between LeRobot and Zarr backends.
"""

import argparse
import logging
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.eva_process.eva_utils import EvaHD5Extractor
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash
from egomimic.utils.egomimicUtils import str2bool
from egomimic.utils.pose_utils import xyzw_to_wxyz
from egomimic.utils.video_utils import resize_video_thwc, save_preview_mp4

logger = logging.getLogger(__name__)

_SPLIT_KEYS = {"obs_ee_pose", "cmd_ee_pose", "obs_joints", "cmd_joints"}

DATASET_KEY_MAPPINGS = {
    "obs_eepose": "obs_ee_pose",
    "obs_joints": "obs_joints",
    "cmd_eepose": "cmd_ee_pose",
    "cmd_joints": "cmd_joints",
    "images.front_img_1": "images.front_1",
    "images.right_wrist_img": "images.right_wrist",
    "images.left_wrist_img": "images.left_wrist",
}

R_t_e = np.array(
    [
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ],
    dtype=float,
)


def rot_orientation(quat: np.ndarray) -> np.ndarray:
    rotation = R.from_quat(quat).as_matrix()
    rotation = R_t_e @ rotation
    return R.from_matrix(rotation).as_quat()


def _arm_to_embodiment(arm: str) -> str:
    """Map arm string to embodiment identifier."""
    return {
        "left": "eva_left_arm",
        "right": "eva_right_arm",
        "both": "eva_bimanual",
    }.get(arm, "eva_bimanual")


def _separate_numeric_and_image(episode_feats: dict):
    """Split process_episode() output into numeric and image dicts.

    * Keys containing "images" are treated as image data
    * Images are transposed from (T,C,H,W) to (T,H,W,C) because
      process_episode() stores (T,C,H,W) while ZarrWriter
      expects (T,H,W,3) for JPEG encoding.
    * metadata.* keys are skipped (they are per-timestep constants
      like embodiment id that are stored in zarr attrs instead).
    """
    numeric_data: dict[str, np.ndarray] = {}
    image_data: dict[str, np.ndarray] = {}
    allowed_keys = set(DATASET_KEY_MAPPINGS.keys())

    for key, value in episode_feats.items():
        if key.startswith("metadata."):
            continue

        if key not in allowed_keys:
            continue

        zarr_key = DATASET_KEY_MAPPINGS[key]
        arr = np.asarray(value)

        if "images" in key:
            # Transpose (T,C,H,W) -> (T,H,W,C) when needed
            if (
                arr.ndim == 4
                and arr.shape[1] in (1, 3, 4)
                and arr.shape[2] > arr.shape[1]
            ):
                arr = arr.transpose(0, 2, 3, 1)
            arr = resize_video_thwc(arr)
            image_data[zarr_key] = arr

        else:
            numeric_data[zarr_key] = arr

    return numeric_data, image_data


def _split_per_arm(numeric_data: dict, arm: str) -> dict:
    """Split combined arm arrays into per-arm keys with gripper separated.

    Bimanual layout (T, 14):
        [0:6]  left xyz+ypr, [6]  left gripper,
        [7:13] right xyz+ypr, [13] right gripper.
    Single-arm layout (T, 7):
        [0:6]  xyz+ypr, [6] gripper.

    Produces keys like ``left.obs_eepose`` (T,6), ``right.gripper`` (T,1), etc.
    Gripper is taken from ``cmd_joints`` only (commanded state).
    """
    out = {k: v for k, v in numeric_data.items() if k not in _SPLIT_KEYS}

    side = {"left": "left", "right": "right"}.get(arm)  # None for "both"

    for base_key in _SPLIT_KEYS:
        arr = numeric_data.get(base_key)
        if arr is None:
            continue

        if arm == "both":
            if "joints" in base_key:
                left_joints = arr[:, 0:6]
                right_joints = arr[:, 7:13]
                out[f"left.{base_key}"] = left_joints
                out[f"right.{base_key}"] = right_joints
                out["left.gripper"] = arr[:, 6:7]
                out["right.gripper"] = arr[:, 13:14]
            else:
                left_ypr = arr[:, 3:6]
                right_ypr = arr[:, 10:13]
                left_quat = rot_orientation(
                    R.from_euler("ZYX", left_ypr, degrees=False).as_quat()
                )
                right_quat = rot_orientation(
                    R.from_euler("ZYX", right_ypr, degrees=False).as_quat()
                )
                left_quat = xyzw_to_wxyz(left_quat)
                right_quat = xyzw_to_wxyz(right_quat)
                left_translation = arr[:, 0:3]
                right_translation = arr[:, 7:10]
                left_translation_quat = np.concatenate(
                    [left_translation, left_quat], axis=-1
                )
                right_translation_quat = np.concatenate(
                    [right_translation, right_quat], axis=-1
                )
                out[f"left.{base_key}"] = left_translation_quat
                out[f"right.{base_key}"] = right_translation_quat
                out["left.gripper"] = arr[:, 6:7]
                out["right.gripper"] = arr[:, 13:14]
        else:
            if "joints" in base_key:
                out[f"{side}.{base_key}"] = arr[:, :6]
                out[f"{side}.gripper"] = arr[:, 6:7]
            else:
                translation = arr[:, 0:3]
                quat = rot_orientation(
                    R.from_euler("ZYX", arr[:, 3:6], degrees=False).as_quat()
                )
                quat = xyzw_to_wxyz(quat)
                translation_quat = np.concatenate([translation, quat], axis=-1)
                out[f"{side}.{base_key}"] = translation_quat
                gripper = arr[:, 6:7]
                out[f"{side}.gripper"] = gripper

    return out


def _infer_total_frames(
    numeric_data: dict[str, np.ndarray], image_data: dict[str, np.ndarray]
) -> int:
    """Infer episode length from numeric/image arrays."""
    for arr in numeric_data.values():
        return int(len(arr))
    for arr in image_data.values():
        return int(len(arr))
    return 0


def convert_episode(
    raw_path: Path,
    output_dir: Path,
    dataset_name: str,
    arm: str,
    fps: int,
    task_name: str = "",
    task_description: str = "",
    save_mp4: bool = False,
    chunk_timesteps: int = 100,
) -> tuple[Path, Path]:
    """Process one HDF5 file and write a .zarr episode.

    Returns the zarr episode path on success.
    """

    episode_feats = EvaHD5Extractor.process_episode(
        episode_path=raw_path,
        arm=arm,
    )

    front_key = "images.front_img_1"
    images_tchw = None
    if save_mp4 and front_key in episode_feats:
        images_tchw = np.asarray(episode_feats[front_key])

    numeric_data, image_data = _separate_numeric_and_image(episode_feats)
    numeric_data = _split_per_arm(numeric_data, arm)

    embodiment = _arm_to_embodiment(arm)

    zarr_path = ZarrWriter.create_and_write(
        episode_path=output_dir / f"{dataset_name}.zarr",
        numeric_data=numeric_data or None,
        image_data=image_data or None,
        embodiment=embodiment,
        fps=fps,
        task_name=task_name,
        task_description=task_description,
        chunk_timesteps=chunk_timesteps,
    )

    logger.info("Wrote zarr episode: %s", zarr_path)

    del episode_feats
    mp4_path = None
    if save_mp4 and images_tchw is not None:
        mp4_path = output_dir / f"{dataset_name}.mp4"
        try:
            logger.info("Saving preview MP4 to: %s", mp4_path)
            save_preview_mp4(images_tchw, mp4_path, fps, half_res=False)
            logger.info("Saved preview MP4: %s", mp4_path)
        except Exception:
            logger.warning(
                "Failed to save preview MP4 at %s:\n%s",
                mp4_path,
                traceback.format_exc(),
            )

    return zarr_path, mp4_path


def main(args) -> None:
    """Convert Eva HDF5 dataset to Zarr episodes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (same shape as eva_to_lerobot).
    """

    try:
        episode_hash = timestamp_ms_to_episode_hash(Path(args.raw_path).stem)
        zarr_path, mp4_path = convert_episode(
            raw_path=Path(args.raw_path),
            output_dir=Path(args.output_dir),
            dataset_name=episode_hash,
            arm=args.arm,
            fps=args.fps,
            task_name=args.task_name,
            task_description=args.task_description,
            save_mp4=args.save_mp4,
            chunk_timesteps=args.chunk_timesteps,
        )
        return zarr_path, mp4_path
    except Exception:
        logger.error("Error converting %s:\n%s", args.raw_path, traceback.format_exc())
        return None, None


def argument_parse():
    parser = argparse.ArgumentParser(
        description="Convert Eva HDF5 dataset to Zarr episodes."
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        required=True,
        help="Directory containing raw HDF5 files.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Root output directory."
    )
    parser.add_argument(
        "--arm", type=str, choices=["left", "right", "both"], default="both"
    )
    parser.add_argument("--save-mp4", type=str2bool, default=False)
    parser.add_argument(
        "--chunk-timesteps",
        type=int,
        default=100,
        help="Timesteps per zarr chunk for numeric arrays.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--task-name", type=str, default="")
    parser.add_argument("--task-description", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(main(argument_parse()))
