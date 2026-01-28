"""
Verify Zarr dataset (from eva_to_zarr.py) against LeRobot dataset (from eva_to_lerobot.py).

This script validates that the Zarr conversion produces equivalent data
to the LeRobot dataset format.

Usage:
    # Compare single zarr dir with single lerobot dataset:
    python verify_zarr.py \
        --zarr-path zarr/comp \
        --lerobot-path lerobot/comp/1762544076731_processed \
        --num-episodes 5

    # Compare zarr dir with multiple lerobot datasets (each in separate folder):
    python verify_zarr.py \
        --zarr-path zarr/comp \
        --lerobot-path lerobot/comp \
        --multi-dataset
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import zarr

from egomimic.rldb.compression_utils import decode_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_zarr_episode(zarr_path: Path) -> dict:
    """Load all data from a Zarr episode store."""
    store = zarr.open(str(zarr_path), mode="r")

    data = {}

    # Load attributes/metadata
    data["metadata"] = dict(store.attrs)

    # Load all arrays
    for key in store.array_keys():
        arr = store[key][:]
        data[key] = arr

    return data


def decode_jpeg_array(encoded_arr: np.ndarray) -> np.ndarray:
    """Decode array of JPEG bytes to RGB images."""
    decoded = []
    for i in range(len(encoded_arr)):
        jpeg_bytes = encoded_arr[i]
        if isinstance(jpeg_bytes, bytes):
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
        else:
            nparr = np.frombuffer(bytes(jpeg_bytes), np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        decoded.append(img_rgb)
    return np.stack(decoded, axis=0)


def decode_video_file(video_path: Path) -> np.ndarray:
    """Decode video file on disk into (N, H, W, C) uint8."""
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return np.stack(frames, axis=0)


def decode_lerobot_images(
    lerobot_images: np.ndarray,
    key: str,
    lerobot_info: dict,
    lerobot_path: Path,
    episode_idx: int = 0,
) -> np.ndarray:
    """Decode LeRobot image column into (N, H, W, C) uint8 when needed."""
    feature = lerobot_info.get("features", {}).get(key, {})
    dtype = feature.get("dtype", "")

    # LeRobot v2: Check for video file in videos/ directory
    if dtype == "video":
        # Try standard LeRobot v2 video path pattern
        video_path = feature.get("video_path", f"videos/{key}/episode_{{episode_index:06d}}.mp4")
        video_file = lerobot_path / video_path.format(episode_index=episode_idx)
        if video_file.exists():
            return decode_video_file(video_file)
        # Also try without the leading videos/ in case of different structure
        alt_video_file = lerobot_path / "videos" / key / f"episode_{episode_idx:06d}.mp4"
        if alt_video_file.exists():
            return decode_video_file(alt_video_file)

    if lerobot_images.ndim == 1:
        sample = lerobot_images[0]

        if isinstance(sample, dict):
            if "data" in sample and "metadata" in sample:
                return decode_video(sample["data"], sample["metadata"])
            if "path" in sample:
                candidate = lerobot_path / sample["path"]
                if candidate.exists():
                    return decode_video_file(candidate)

        if isinstance(sample, (list, tuple)) and len(sample) == 2:
            data, metadata = sample
            if isinstance(data, (bytes, bytearray)):
                return decode_video(data, metadata)

        if isinstance(sample, str):
            candidate = lerobot_path / sample
            if candidate.exists():
                return decode_video_file(candidate)

        if dtype == "video":
            # Fall back to JPEG decode if video metadata isn't available.
            return decode_jpeg_array(lerobot_images)

        # Check if it's actually bytes data
        if isinstance(sample, (bytes, bytearray)):
            return decode_jpeg_array(lerobot_images)

        raise ValueError(f"Cannot decode images for key {key}: sample type={type(sample)}, dtype={dtype}")

    if lerobot_images.ndim == 4 and lerobot_images.shape[1] in [1, 3]:
        return np.transpose(lerobot_images, (0, 2, 3, 1))

    return lerobot_images


def load_lerobot_episode(lerobot_path: Path, episode_idx: int) -> dict:
    """Load episode data from LeRobot dataset format."""
    import pyarrow.parquet as pq

    # Load metadata
    import json
    meta_path = lerobot_path / "meta" / "info.json"
    with open(meta_path) as f:
        info = json.load(f)

    # Find parquet file for this episode
    data_pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunk_size = info.get("chunks_size", 1000)
    episode_chunk = episode_idx // chunk_size

    parquet_path = lerobot_path / data_pattern.format(
        episode_chunk=episode_chunk,
        episode_index=episode_idx
    )

    data = {}
    data["metadata"] = {
        "episode_index": episode_idx,
        "fps": info.get("fps", 30),
        "total_frames": info.get("total_frames", 0),
    }

    # Load parquet data directly
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # Extract each feature
    features = info.get("features", {})
    for key, feat_info in features.items():
        if key in df.columns:
            col_data = df[key].values

            # Handle nested arrays - need to recursively convert
            def to_numpy_recursive(x):
                """Convert nested lists/arrays to numpy array."""
                if isinstance(x, np.ndarray):
                    if x.dtype == object:
                        # Recursively convert object arrays
                        return np.stack([to_numpy_recursive(item) for item in x])
                    return x
                elif isinstance(x, (list, tuple)):
                    # Check if elements are also lists/arrays
                    if len(x) > 0 and isinstance(x[0], (list, tuple, np.ndarray)):
                        return np.stack([to_numpy_recursive(item) for item in x])
                    return np.array(x)
                return np.array(x)

            # Convert the column data
            try:
                arr = np.stack([to_numpy_recursive(x) for x in col_data], axis=0)
                data[key] = arr
            except Exception as e:
                logger.warning(f"Failed to convert {key}: {e}")
                data[key] = col_data

    return data, info


def compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    key: str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict:
    """Compare two numpy arrays and return comparison results."""
    result = {
        "key": key,
        "match": False,
        "shape_match": arr1.shape == arr2.shape,
        "dtype_match": arr1.dtype == arr2.dtype,
        "arr1_shape": arr1.shape,
        "arr2_shape": arr2.shape,
        "arr1_dtype": str(arr1.dtype),
        "arr2_dtype": str(arr2.dtype),
    }

    if not result["shape_match"]:
        result["error"] = f"Shape mismatch: {arr1.shape} vs {arr2.shape}"
        return result

    # For numeric arrays, use allclose
    if np.issubdtype(arr1.dtype, np.number) and np.issubdtype(arr2.dtype, np.number):
        arr1_float = arr1.astype(np.float64)
        arr2_float = arr2.astype(np.float64)

        result["match"] = np.allclose(arr1_float, arr2_float, atol=atol, rtol=rtol)

        if not result["match"]:
            diff = np.abs(arr1_float - arr2_float)
            result["max_diff"] = float(np.max(diff))
            result["mean_diff"] = float(np.mean(diff))
            result["num_mismatches"] = int(np.sum(~np.isclose(arr1_float, arr2_float, atol=atol, rtol=rtol)))
            result["total_elements"] = int(arr1.size)

            # Debug: find where max diff occurs
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            result["max_diff_index"] = list(max_idx)
            result["zarr_value_at_max"] = float(arr1_float[max_idx])
            result["lerobot_value_at_max"] = float(arr2_float[max_idx])

            # Check first few elements
            result["zarr_first_5"] = arr1_float.flat[:5].tolist()
            result["lerobot_first_5"] = arr2_float.flat[:5].tolist()

            # For angle-containing arrays, check if difference is due to euler representation
            if "cartesian" in key or "ee_pose" in key:
                # Check dimensions 3,4,5 (yaw, pitch, roll) - compare rotations properly
                from scipy.spatial.transform import Rotation as R
                try:
                    # Get first frame's angles
                    if arr1_float.ndim == 3:  # prestacked (T, S, D)
                        z_angles = arr1_float[0, 0, 3:6]
                        l_angles = arr2_float[0, 0, 3:6]
                    else:  # (T, D)
                        z_angles = arr1_float[0, 3:6]
                        l_angles = arr2_float[0, 3:6]

                    # Convert to rotation matrices and compare
                    R_zarr = R.from_euler("ZYX", z_angles).as_matrix()
                    R_lerobot = R.from_euler("ZYX", l_angles).as_matrix()
                    rot_diff = np.abs(R_zarr - R_lerobot).max()
                    result["rotation_matrix_diff_frame0"] = float(rot_diff)
                    result["zarr_euler_frame0"] = z_angles.tolist()
                    result["lerobot_euler_frame0"] = l_angles.tolist()
                except Exception as e:
                    result["rotation_check_error"] = str(e)
    else:
        # For non-numeric (e.g., object arrays with bytes), check equality
        result["match"] = np.array_equal(arr1, arr2)

    return result


def compare_images(
    zarr_images: np.ndarray,
    lerobot_images: np.ndarray,
    key: str,
    lerobot_info: dict,
    lerobot_path: Path,
    episode_idx: int = 0,
    jpeg_quality_tolerance: int = 10,
) -> dict:
    """Compare JPEG-encoded Zarr images with LeRobot images.

    Both may have gone through JPEG compression, so we compare with tolerance.
    """
    result = {
        "key": key,
        "match": False,
        "num_frames": len(zarr_images),
    }

    # Decode JPEG images from Zarr
    try:
        decoded_zarr = decode_jpeg_array(zarr_images)
    except Exception as e:
        result["error"] = f"Failed to decode JPEG images from Zarr: {e}"
        return result

    # Decode or reshape LeRobot images if needed
    try:
        lerobot_images = decode_lerobot_images(lerobot_images, key, lerobot_info, lerobot_path, episode_idx)
    except Exception as e:
        result["error"] = f"Failed to decode LeRobot images: {e}"
        return result

    # Check shapes
    if decoded_zarr.shape != lerobot_images.shape:
        result["error"] = f"Shape mismatch after decode: {decoded_zarr.shape} vs {lerobot_images.shape}"
        return result

    # Compare with tolerance for JPEG artifacts
    diff = np.abs(decoded_zarr.astype(np.int16) - lerobot_images.astype(np.int16))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Allow for JPEG compression differences (both datasets use JPEG)
    tolerance = jpeg_quality_tolerance * 3
    result["match"] = max_diff <= tolerance
    result["max_pixel_diff"] = int(max_diff)
    result["mean_pixel_diff"] = float(mean_diff)

    return result


def compare_zarr_with_lerobot(
    zarr_path: Path,
    lerobot_path: Path,
    episode_idx: int = 0,
    verbose: bool = True,
) -> dict:
    """Compare a single Zarr episode with corresponding LeRobot episode."""
    results = {
        "zarr_path": str(zarr_path),
        "lerobot_path": str(lerobot_path),
        "episode_idx": episode_idx,
        "all_match": True,
        "comparisons": [],
    }

    # Load Zarr data
    logger.info(f"Loading Zarr episode: {zarr_path}")
    zarr_data = load_zarr_episode(zarr_path)

    # Load LeRobot data
    logger.info(f"Loading LeRobot episode {episode_idx}: {lerobot_path}")
    lerobot_data, lerobot_info = load_lerobot_episode(lerobot_path, episode_idx)

    # Get keys (excluding metadata)
    zarr_keys = set(k for k in zarr_data.keys() if k != "metadata")
    lerobot_keys = set(k for k in lerobot_data.keys() if k != "metadata")

    # Map keys for comparison
    common_keys = zarr_keys & lerobot_keys
    zarr_only = zarr_keys - lerobot_keys
    lerobot_only = lerobot_keys - zarr_keys

    results["common_keys"] = list(common_keys)
    results["zarr_only_keys"] = list(zarr_only)
    results["lerobot_only_keys"] = list(lerobot_only)

    if zarr_only:
        logger.warning(f"Keys only in Zarr: {zarr_only}")
    if lerobot_only:
        logger.warning(f"Keys only in LeRobot: {lerobot_only}")

    # Compare each common key
    for key in sorted(common_keys):
        zarr_arr = zarr_data[key]
        lerobot_arr = lerobot_data[key]

        if "images" in key:
            comparison = compare_images(zarr_arr, lerobot_arr, key, lerobot_info, lerobot_path, episode_idx)
        else:
            comparison = compare_arrays(zarr_arr, lerobot_arr, key)

        results["comparisons"].append(comparison)

        if not comparison["match"]:
            results["all_match"] = False
            if verbose:
                logger.warning(f"Mismatch in {key}: {comparison}")
        elif verbose:
            logger.info(f"Match: {key} (shape: {zarr_arr.shape})")

    return results


def find_lerobot_datasets(lerobot_path: Path) -> list[Path]:
    """Find all LeRobot dataset directories (those with meta/info.json)."""
    datasets = []

    # Check if this path itself is a dataset
    if (lerobot_path / "meta" / "info.json").exists():
        return [lerobot_path]

    # Otherwise, look for subdirectories that are datasets
    for subdir in sorted(lerobot_path.iterdir()):
        if subdir.is_dir() and (subdir / "meta" / "info.json").exists():
            datasets.append(subdir)

    return datasets


def compare_datasets(
    zarr_dir: Path,
    lerobot_path: Path,
    multi_dataset: bool = False,
    num_episodes: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Compare all episodes between Zarr dataset and LeRobot dataset(s)."""
    results = {
        "zarr_dir": str(zarr_dir),
        "lerobot_path": str(lerobot_path),
        "episodes": [],
        "summary": {
            "total_episodes": 0,
            "matching_episodes": 0,
            "failed_episodes": 0,
        },
    }

    # Find Zarr episodes
    zarr_episodes = sorted(zarr_dir.glob("episode_*.zarr"))

    if multi_dataset:
        # Multiple LeRobot datasets, each with 1 episode
        lerobot_datasets = find_lerobot_datasets(lerobot_path)

        if not lerobot_datasets:
            logger.error(f"No LeRobot datasets found in {lerobot_path}")
            return results

        if num_episodes:
            zarr_episodes = zarr_episodes[:num_episodes]
            lerobot_datasets = lerobot_datasets[:num_episodes]

        if len(zarr_episodes) != len(lerobot_datasets):
            logger.warning(
                f"Count mismatch: {len(zarr_episodes)} Zarr episodes vs {len(lerobot_datasets)} LeRobot datasets"
            )

        results["summary"]["total_episodes"] = min(len(zarr_episodes), len(lerobot_datasets))

        for i, (zarr_ep, lerobot_ds) in enumerate(zip(zarr_episodes, lerobot_datasets)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Comparing episode {i}: {zarr_ep.name} vs {lerobot_ds.name}")
            logger.info(f"{'='*60}")

            try:
                ep_result = compare_zarr_with_lerobot(
                    zarr_ep,
                    lerobot_ds,
                    episode_idx=0,  # Each dataset has only 1 episode
                    verbose=verbose,
                )
                results["episodes"].append(ep_result)

                if ep_result["all_match"]:
                    results["summary"]["matching_episodes"] += 1
                    logger.info(f"Episode {i}: ALL MATCH")
                else:
                    results["summary"]["failed_episodes"] += 1
                    logger.warning(f"Episode {i}: MISMATCH DETECTED")

            except Exception as e:
                logger.error(f"Episode {i}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                results["episodes"].append({
                    "zarr_path": str(zarr_ep),
                    "lerobot_path": str(lerobot_ds),
                    "episode_idx": i,
                    "error": str(e),
                })
                results["summary"]["failed_episodes"] += 1
    else:
        # Single LeRobot dataset with multiple episodes
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(repo_id="local/dataset", root=lerobot_path, local_files_only=True)
        lerobot_num_episodes = dataset.num_episodes

        if num_episodes:
            zarr_episodes = zarr_episodes[:num_episodes]
            lerobot_num_episodes = min(lerobot_num_episodes, num_episodes)

        if len(zarr_episodes) != lerobot_num_episodes:
            logger.warning(
                f"Episode count mismatch: {len(zarr_episodes)} Zarr vs {lerobot_num_episodes} LeRobot"
            )

        results["summary"]["total_episodes"] = min(len(zarr_episodes), lerobot_num_episodes)

        for i, zarr_ep in enumerate(zarr_episodes[:lerobot_num_episodes]):
            logger.info(f"\n{'='*60}")
            logger.info(f"Comparing episode {i}: {zarr_ep.name}")
            logger.info(f"{'='*60}")

            try:
                ep_result = compare_zarr_with_lerobot(
                    zarr_ep,
                    lerobot_path,
                    episode_idx=i,
                    verbose=verbose,
                )
                results["episodes"].append(ep_result)

                if ep_result["all_match"]:
                    results["summary"]["matching_episodes"] += 1
                    logger.info(f"Episode {i}: ALL MATCH")
                else:
                    results["summary"]["failed_episodes"] += 1
                    logger.warning(f"Episode {i}: MISMATCH DETECTED")

            except Exception as e:
                logger.error(f"Episode {i}: ERROR - {e}")
                import traceback
                traceback.print_exc()
                results["episodes"].append({
                    "zarr_path": str(zarr_ep),
                    "episode_idx": i,
                    "error": str(e),
                })
                results["summary"]["failed_episodes"] += 1

    return results


def print_summary(results: dict):
    """Print a summary of comparison results."""
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    summary = results["summary"]
    print(f"Total episodes compared: {summary['total_episodes']}")
    print(f"Matching episodes: {summary['matching_episodes']}")
    print(f"Failed/mismatched episodes: {summary['failed_episodes']}")

    if summary["failed_episodes"] > 0:
        print("\nFailed episodes:")
        for ep in results["episodes"]:
            if "error" in ep or not ep.get("all_match", True):
                print(f"  - Episode {ep.get('episode_idx', 'unknown')}: {ep.get('zarr_path', 'unknown')}")
                if "error" in ep:
                    print(f"    Error: {ep['error']}")
                else:
                    # Find mismatched keys
                    mismatched = [c["key"] for c in ep.get("comparisons", []) if not c["match"]]
                    print(f"    Mismatched keys: {mismatched}")

    match_rate = summary["matching_episodes"] / max(summary["total_episodes"], 1) * 100
    print(f"\nMatch rate: {match_rate:.1f}%")

    if match_rate == 100:
        print("\nAll data matches between Zarr and LeRobot datasets.")
    else:
        print("\nSome differences detected. Review logs for details.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify Zarr dataset against LeRobot dataset"
    )

    parser.add_argument(
        "--zarr-path",
        type=Path,
        required=True,
        help="Path to Zarr dataset directory (containing episode_*.zarr folders)",
    )
    parser.add_argument(
        "--lerobot-path",
        type=Path,
        required=True,
        help="Path to LeRobot dataset directory",
    )
    parser.add_argument(
        "--multi-dataset",
        action="store_true",
        help="LeRobot path contains multiple dataset folders (one per episode)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to compare (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed comparison info",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Save results to JSON file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    results = compare_datasets(
        zarr_dir=args.zarr_path,
        lerobot_path=args.lerobot_path,
        multi_dataset=args.multi_dataset,
        num_episodes=args.num_episodes,
        verbose=args.verbose,
    )

    print_summary(results)

    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            def convert(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            json.dump(results, f, indent=2, default=convert)
        logger.info(f"Results saved to {args.output_json}")

    # Return exit code based on results
    if results["summary"]["failed_episodes"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
