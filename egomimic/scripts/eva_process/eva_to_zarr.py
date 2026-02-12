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
import torch
from scipy.spatial.transform import Rotation as R

from egomimic.scripts.eva_process.zarr_utils import EvaHD5Extractor
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.utils.egomimicUtils import EXTRINSICS, str2bool, xyzw_to_wxyz
""
logger = logging.getLogger(__name__)

DATASET_KEY_MAPPINGS = {
    "observations.state.eepose": "obs_ee_pose",
    "observations.state.joint_positions": "obs_joints",
    "actions_base_cartesian": "cmd_ee_pose",
    "actions_joints": "cmd_joints",
    "observations.images.front_img_1": "images.front_1",
    "observations.images.right_wrist_img": "images.right_wrist",
    "observations.images.left_wrist_img": "images.left_wrist",
}

# ---------------------------------------------------------------------------
# Preview MP4
# ---------------------------------------------------------------------------


R_t_e = np.array([
  [0, 0,  1],
  [-1,  0, 0],
  [0,  -1,  0],
], dtype=float)

def eva_reorientation(quat: np.ndarray) -> np.ndarray:
    rotation = R.from_quat(quat).as_matrix()
    rotation = R_t_e @ rotation
    return R.from_matrix(rotation).as_quat()
    

def save_preview_mp4(
    images_tchw: np.ndarray, output_path: Path, fps: int
) -> None:
    """Save a half-resolution, web-compatible MP4 from a (T,C,H,W) uint8 array.

    Tries torchvision.io.write_video first, then falls back to ffmpeg CLI.

    Parameters
    ----------
    images_tchw : np.ndarray
        Image array with shape (T, C, H, W), uint8.
    output_path : Path
        Destination path for the .mp4 file.
    fps : int
        Frames per second.
    """
    import torch.nn.functional as F

    imgs = torch.from_numpy(np.asarray(images_tchw))  # (T,C,H,W)
    if imgs.ndim != 4 or len(imgs) == 0:
        logger.warning("No valid frames for preview MP4 — skipping.")
        return

    _, _, H, W = imgs.shape

    # Half-resolution with even dims for yuv420p
    outW, outH = W // 2, H // 2
    outW -= outW % 2
    outH -= outH % 2
    if outW <= 0 or outH <= 0:
        raise ValueError(f"Invalid output size: {outW}x{outH}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_frames = []
    for chw in imgs:
        t = chw.detach().cpu().to(torch.uint8)
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        t_resized = F.interpolate(
            t.unsqueeze(0).float(),
            size=(outH, outW),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).to(torch.uint8)
        rgb_frames.append(t_resized.permute(1, 2, 0).contiguous())  # (H,W,3)

    video_tensor = torch.stack(rgb_frames, dim=0)  # (T,H,W,3) uint8

    # Try torchvision first
    try:
        import torchvision.io

        torchvision.io.write_video(
            str(output_path), video_tensor, fps=fps,
            video_codec="libx264",
            options={"crf": "23", "pix_fmt": "yuv420p"},
        )
        return
    except Exception:
        logger.debug("torchvision.io.write_video failed, trying ffmpeg CLI.")

    # Fallback: pipe raw frames to ffmpeg
    import subprocess, shutil

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Neither torchvision.io.write_video nor ffmpeg CLI available.")

    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{outW}x{outH}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "fast",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stdin.write(video_tensor.numpy().tobytes())
    proc.stdin.close()
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr.decode()}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

        # Flatten one nested level (e.g., observations -> observations.images.front_img_1)
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                full_key = f"{key}.{nested_key}"
                if full_key not in allowed_keys:
                    continue

                zarr_key = DATASET_KEY_MAPPINGS[full_key]
                arr = np.asarray(nested_value)

                if "images" in full_key:
                    # Transpose (T,C,H,W) -> (T,H,W,C) when needed
                    if arr.ndim == 4 and arr.shape[1] in (1, 3, 4) and arr.shape[2] > arr.shape[1]:
                        arr = arr.transpose(0, 2, 3, 1)
                    image_data[zarr_key] = arr
                else:
                    numeric_data[zarr_key] = arr
        else:
            if key not in allowed_keys:
                continue

            zarr_key = DATASET_KEY_MAPPINGS[key]
            numeric_data[zarr_key] = np.asarray(value)

    return numeric_data, image_data


_SPLIT_KEYS = {"obs_ee_pose", "cmd_ee_pose", "obs_joints", "cmd_joints"}


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

        # (TODO) aria to zarr change to use both instead of bimanual
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
                left_quat = eva_reorientation(R.from_euler("ZYX", left_ypr, degrees=False).as_quat())
                right_quat = eva_reorientation(R.from_euler("ZYX", right_ypr, degrees=False).as_quat())
                left_quat = xyzw_to_wxyz(left_quat)
                right_quat = xyzw_to_wxyz(right_quat)
                left_translation = arr[:, 0:3]
                right_translation = arr[:, 7:10]
                left_translation_quat = np.concatenate([left_translation, left_quat], axis=-1)
                right_translation_quat = np.concatenate([right_translation, right_quat], axis=-1)
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
                quat = eva_reorientation(R.from_euler("ZYX", arr[:, 3:6], degrees=False).as_quat())
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


def _build_example_annotations(
    total_frames: int, description: str
) -> list[tuple[str, int, int]]:
    """Create simple example language spans for demo/testing."""
    if total_frames <= 0:
        return []

    end_idx = total_frames - 1
    mid_idx = end_idx // 2
    task_text = description.strip() or "perform the demonstrated task"

    annotations: list[tuple[str, int, int]] = [
        (task_text, 0, end_idx),
    ]

    if end_idx >= 3:
        annotations.append(("reach and grasp", 0, mid_idx))
        if mid_idx + 1 <= end_idx:
            annotations.append(("lift and place", mid_idx + 1, end_idx))

    return annotations


# ---------------------------------------------------------------------------
# Single-episode conversion
# ---------------------------------------------------------------------------

def convert_episode(
    hdf5_path: Path,
    zarr_episode_path: Path,
    arm: str,
    extrinsics_key: str,
    fps: int,
    description: str = "",
    mp4_dir: Path | None = None,
    chunk_timesteps: int = 100,

    example_annotations: bool = False,
) -> Path:
    """Process one HDF5 file and write a .zarr episode.

    Returns the zarr episode path on success.
    """
    hdf5_path = Path(hdf5_path)
    zarr_episode_path = Path(zarr_episode_path)

    extrinsics = EXTRINSICS[extrinsics_key]

    episode_feats = EvaHD5Extractor.process_episode(
        episode_path=hdf5_path,
        arm=arm,
        extrinsics=extrinsics,
    )

    numeric_data, image_data = _separate_numeric_and_image(episode_feats)
    numeric_data = _split_per_arm(numeric_data, arm)
    total_frames = _infer_total_frames(numeric_data, image_data)

    embodiment = _arm_to_embodiment(arm)
    annotations = (
        _build_example_annotations(total_frames, description)
        if example_annotations
        else []
    )

    ZarrWriter.create_and_write(
        episode_path=zarr_episode_path,
        numeric_data=numeric_data or None,
        image_data=image_data or None,
        embodiment=embodiment,
        fps=fps,
        task=description,
        annotations=annotations,
        chunk_timesteps=chunk_timesteps,
        enable_sharding=True,
    )

    logger.info("Wrote zarr episode: %s", zarr_episode_path)

    # Optional preview MP4
    front_key = "images.front_img_1"
    obs = episode_feats.get("observations") or {}
    if mp4_dir is not None and front_key in obs:
        images_tchw = np.asarray(obs[front_key])
        mp4_path = Path(mp4_dir) / f"{hdf5_path.stem}_video.mp4"
        try:
            logger.info("Saving preview MP4 to: %s", mp4_path)
            save_preview_mp4(images_tchw, mp4_path, fps)
            logger.info("Saved preview MP4: %s", mp4_path)
        except Exception:
            logger.warning(
                "Failed to save preview MP4 at %s:\n%s", mp4_path, traceback.format_exc()
            )

    return zarr_episode_path


# ---------------------------------------------------------------------------
# main(args) – matches eva_to_lerobot.main(args) signature
# ---------------------------------------------------------------------------

def main(args) -> None:
    """Convert Eva HDF5 dataset to Zarr episodes.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (same shape as eva_to_lerobot).
    """
    raw_path = Path(args.raw_path)
    output_base = Path(args.output_dir)

    episode_list = sorted(raw_path.glob("*.hdf5"))
    if not episode_list:
        raise ValueError(f"No .hdf5 files found in {raw_path}")

    EvaHD5Extractor.check_format(episode_list, image_compressed=getattr(args, "image_compressed", False))

    mp4_dir = output_base if getattr(args, "save_mp4", False) else None
    chunk_timesteps = getattr(args, "chunk_timesteps", 100)
    example_annotations = getattr(
        args,
        "example_language_annotations",
        getattr(args, "example_annotations", False),
    )

    for hdf5_path in episode_list:
        stem = hdf5_path.stem
        zarr_episode_path = output_base / f"{stem}.zarr"
        try:
            convert_episode(
                hdf5_path=hdf5_path,
                zarr_episode_path=zarr_episode_path,
                arm=args.arm,
                extrinsics_key=getattr(args, "extrinsics_key", "x5Dec13_2"),
                fps=getattr(args, "fps", 30),
                description=getattr(args, "description", ""),
                mp4_dir=mp4_dir,
                chunk_timesteps=chunk_timesteps,
                example_annotations=example_annotations,
            )
        except Exception:
            logger.error("Error converting %s:\n%s", hdf5_path, traceback.format_exc())
            continue


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def argument_parse():
    parser = argparse.ArgumentParser(
        description="Convert Eva HDF5 dataset to Zarr episodes."
    )
    parser.add_argument("--raw-path", type=Path, required=True, help="Directory containing raw HDF5 files.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Root output directory.")
    parser.add_argument("--arm", type=str, choices=["left", "right", "both"], default="both")
    parser.add_argument("--extrinsics-key", type=str, default="x5Dec13_2")
    parser.add_argument("--image-compressed", type=str2bool, default=False)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--save-mp4", type=str2bool, default=False)
    parser.add_argument("--chunk-timesteps", type=int, default=100, help="Timesteps per zarr chunk for numeric arrays.")
    parser.add_argument(
        "--example-language-annotations",
        type=str2bool,
        default=False,
        help="If true, write simple example language spans into each episode.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nproc", type=int, default=12)
    parser.add_argument("--nthreads", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(argument_parse())
