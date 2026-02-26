#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert HDF5 episodes to Zarr format using ZarrWriter and EvaHD5Extractor.

This module provides a utility for converting Eva HDF5 episode files
to the Zarr v3 format compatible with ZarrEpisode reader, with full
forward kinematics, action preprocessing, and camera frame transformations.

Usage:
    # Convert a single episode
    python zarr_write_test.py --hdf5-path /path/to/episode.hdf5 --output-dir /output/zarr --arm both --extrinsics-key x5Dec13_2

    # Convert all episodes in a directory
    python zarr_write_test.py --hdf5-dir /path/to/episodes --output-dir /output/zarr --arm both --extrinsics-key x5Dec13_2
"""

import argparse
from pathlib import Path

import numpy as np

from egomimic.rldb.zarr import ZarrWriter
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset, ZarrEpisode
from egomimic.scripts.eva_process.zarr_utils import EvaHD5Extractor
from egomimic.utils.egomimicUtils import EXTRINSICS


def is_image_array(arr) -> bool:
    """
    Check if array is an image based on shape and dtype.

    Images are expected to be:
    - 4D arrays with shape (T, H, W, C) or (T, C, H, W)
    - C (channels) is typically 1, 3, or 4
    - H and W (spatial) are larger than C
    - dtype is uint8

    Args:
        arr: Array to check

    Returns:
        True if array appears to be an image sequence
    """
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
        return False

    if arr.ndim != 4:
        return False

    # Check dtype (images should be uint8)
    if arr.dtype != np.uint8:
        return False

    # Check if it looks like (T, H, W, C) or (T, C, H, W)
    T, dim1, dim2, dim3 = arr.shape

    # Check for channels-last (T, H, W, C)
    if dim3 in [1, 3, 4] and dim1 > dim3 and dim2 > dim3:
        return True

    # Check for channels-first (T, C, H, W)
    if dim1 in [1, 3, 4] and dim2 > dim1 and dim3 > dim1:
        return True

    return False


def needs_transpose_to_hwc(arr) -> bool:
    """
    Check if image array needs transpose from (T, C, H, W) to (T, H, W, C).

    Args:
        arr: Image array

    Returns:
        True if array is in channels-first format and needs transpose
    """
    if arr.ndim != 4:
        return False

    T, dim1, dim2, dim3 = arr.shape

    # If first spatial dim is small (1-4) and others are large, it's channels-first
    return dim1 in [1, 3, 4] and dim2 > dim1 and dim3 > dim1


def convert_hdf5_to_zarr(
    hdf5_path: str | Path,
    zarr_episode_path: str | Path,
    arm: str = "both",
    extrinsics_key: str = "x5Dec13_2",
    prestack: bool = False,
    low_res: bool = False,
    no_rot: bool = False,
    fps: int = 30,
    incremental: bool = False,
    batch_size: int = 64,
):
    """
    Convert an Eva HDF5 episode file to Zarr format using EvaHD5Extractor.

    This function processes the HDF5 file with full forward kinematics,
    camera frame transformations, and action preprocessing before writing to Zarr.

    Args:
        hdf5_path: Path to the input HDF5 file
        zarr_episode_path: Output path for the Zarr episode (e.g., /data/episode_000000.zarr)
        arm: Which arm to process - "left", "right", or "both"
        extrinsics_key: Key to lookup camera extrinsics from EXTRINSICS dict
        prestack: Whether to prestack future actions into chunks
        low_res: Whether to downsample images to low resolution
        no_rot: Whether to skip rotation processing (position only)
        fps: Frames per second for the episode

    Returns:
        Path to created Zarr episode

    Example:
        >>> convert_hdf5_to_zarr(
        ...     hdf5_path="/data/episodes/episode_000000.hdf5",
        ...     zarr_episode_path="/data/zarr_episodes/episode_000000.zarr",
        ...     arm="both",
        ...     extrinsics_key="x5Dec13_2",
        ... )
    """
    print("\n" + "=" * 60)
    print(f"Converting HDF5 to Zarr: {hdf5_path}")
    print(f"  Arm: {arm}")
    print(f"  Extrinsics: {extrinsics_key}")
    print(f"  Prestack: {prestack}")
    print(f"  Low res: {low_res}")
    print(f"  No rotation: {no_rot}")
    print("=" * 60)

    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        print(f"Error: HDF5 file not found at {hdf5_path}")
        return None

    # Get camera extrinsics
    if extrinsics_key not in EXTRINSICS:
        print(f"Error: Unknown extrinsics key '{extrinsics_key}'")
        print(f"Available keys: {list(EXTRINSICS.keys())}")
        return None

    extrinsics = EXTRINSICS[extrinsics_key]

    # Process episode using EvaHD5Extractor
    print("\nProcessing episode with EvaHD5Extractor...")
    episode_feats = EvaHD5Extractor.process_episode(
        episode_path=hdf5_path,
        arm=arm,
        extrinsics=extrinsics,
        prestack=prestack,
        low_res=low_res,
        no_rot=no_rot,
    )

    # Flatten nested observations dictionary and separate numeric/image data and metadata
    numeric_data = {}
    image_data = {}
    metadata_dict = {}

    for key, value in episode_feats.items():
        if key == "observations" and isinstance(value, dict):
            # Preserve observations prefix: observations.images.camera
            for nested_key, nested_value in value.items():
                full_key = f"observations.{nested_key}"

                # Auto-detect and convert images from T×C×H×W to T×H×W×C for JPEG compression
                if is_image_array(nested_value):
                    if needs_transpose_to_hwc(nested_value):
                        # Transpose from (T, C, H, W) to (T, H, W, C)
                        nested_value = nested_value.transpose(0, 2, 3, 1)
                        print(
                            f"  Image: {nested_key} -> {nested_value.shape} (T×H×W×C) [transposed]"
                        )
                    else:
                        print(
                            f"  Image: {nested_key} -> {nested_value.shape} (T×H×W×C)"
                        )
                    image_data[full_key] = nested_value
                else:
                    numeric_data[full_key] = nested_value
        elif key.startswith("metadata."):
            # Extract metadata separately (don't include as features)
            metadata_key = key.replace("metadata.", "")
            metadata_dict[metadata_key] = value
            print(f"  Metadata: {key} (excluded from features)")
        else:
            # Actions and other top-level data are numeric
            numeric_data[key] = value

    print(f"\n{'=' * 60}")
    print("Data Summary:")
    print(f"{'=' * 60}")
    print(f"Numeric data: {len(numeric_data)} arrays")
    for key, value in numeric_data.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

    print(f"\nImage data: {len(image_data)} arrays")
    for key, value in image_data.items():
        print(f"  {key}: shape {value.shape}, dtype {value.dtype}")

    if metadata_dict:
        print(f"\nMetadata: {len(metadata_dict)} items (excluded from features)")
        for key, value in metadata_dict.items():
            if hasattr(value, "shape"):
                print(
                    f"  {key}: shape {value.shape}, constant value {value[0] if len(value) > 0 else 'N/A'}"
                )
            else:
                print(f"  {key}: {value}")
    print(f"{'=' * 60}")

    # Validate we have data to write
    if not numeric_data and not image_data:
        print(
            "\n❌ ERROR: No data to write! Both numeric_data and image_data are empty."
        )
        return None

    # Determine robot type from arm
    if arm == "both":
        embodiment = "eva_bimanual"
    elif arm == "left":
        embodiment = "eva_left_arm"
    elif arm == "right":
        embodiment = "eva_right_arm"
    else:
        embodiment = "eva"

    # Ensure output directory exists
    zarr_episode_path = Path(zarr_episode_path)
    zarr_episode_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {zarr_episode_path.parent}")
    print(f"Output file: {zarr_episode_path.name}")

    # Write to Zarr with explicit numeric/image separation
    print(f"\nWriting to Zarr (incremental={incremental})...")
    try:
        if incremental:
            # Determine total_frames from data
            all_arrays = list(numeric_data.values()) + list(image_data.values())
            total_frames = len(all_arrays[0])

            writer = ZarrWriter(
                episode_path=zarr_episode_path,
                fps=fps,
                embodiment=embodiment,
                enable_sharding=False,
                task="",
            )
            with writer.write_incremental(total_frames=total_frames) as inc:
                for start in range(0, total_frames, batch_size):
                    end = min(start + batch_size, total_frames)
                    batch_numeric = (
                        {k: v[start:end] for k, v in numeric_data.items()}
                        if numeric_data
                        else {}
                    )
                    batch_images = (
                        {k: v[start:end] for k, v in image_data.items()}
                        if image_data
                        else {}
                    )
                    inc.add_frames(numeric=batch_numeric, images=batch_images)
                    print(f"  Wrote frames {start}:{end} / {total_frames}")

            zarr_path = writer.episode_path
        else:
            zarr_path = ZarrWriter.create_and_write(
                episode_path=zarr_episode_path,
                numeric_data=numeric_data if numeric_data else None,
                image_data=image_data if image_data else None,
                fps=fps,
                embodiment=embodiment,
                enable_sharding=False,
                task="",
            )
    except Exception as e:
        print(f"\n❌ ERROR writing Zarr file: {e}")
        import traceback

        traceback.print_exc()
        return None

    print(f"\n✅ Successfully converted to: {zarr_path}")
    print(f"   File exists: {zarr_path.exists()}")

    if zarr_path.exists():
        # Get size of zarr directory
        import os

        total_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(zarr_path)
            for filename in filenames
        )
        print(f"   Size: {total_size / (1024 * 1024):.2f} MB")

    # Verify the conversion
    print("\nVerifying Zarr episode...")
    try:
        episode = ZarrEpisode(zarr_path)
        print(f"  ✅ Total frames: {len(episode)}")
        print(f"  ✅ Available keys: {list(episode.keys.keys())}")

        # Sample first frame using ZarrDataset
        if len(episode) > 0:
            print("\n  First frame sample:")
            dataset = ZarrDataset(zarr_path)
            frame = dataset[0]
            for key, value in frame.items():
                if hasattr(value, "shape"):
                    print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"    {key}: {type(value)}")
    except Exception as e:
        print(f"  ❌ ERROR verifying Zarr episode: {e}")
        import traceback

        traceback.print_exc()

    return zarr_path


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert Eva HDF5 episodes to Zarr format with full processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single episode with both arms
  %(prog)s --hdf5-path /data/episode_000000.hdf5 --output-dir /data/zarr --arm both --extrinsics-key x5Dec13_2

  # Convert all episodes in a directory with prestacking
  %(prog)s --hdf5-dir /data/episodes --output-dir /data/zarr --arm both --extrinsics-key x5Dec13_2 --prestack

Expected HDF5 structure:
  /observations/joint_positions      -> (T, 14) joint angles
  /observations/images/front_img_1   -> (T, H, W, 3) or compressed
  /observations/images/right_wrist_img -> (T, H, W, 3) or compressed
  /observations/images/left_wrist_img  -> (T, H, W, 3) or compressed
  /action                            -> (T, 14) joint actions

The converter will:
  - Compute forward kinematics for end-effector poses
  - Transform poses to camera frame using extrinsics
  - Compute Cartesian actions in base and camera frames
  - Compute relative EEF actions
  - Add embodiment metadata
  - Optionally prestack future actions into chunks

Image arrays will be automatically detected and JPEG-compressed.
        """,
    )

    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--hdf5-path", type=Path, help="Path to a single HDF5 episode file to convert"
    )
    group.add_argument(
        "--hdf5-dir",
        type=Path,
        help="Directory containing HDF5 episode files (*.hdf5) to convert",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for Zarr episodes",
    )

    # Eva-specific processing arguments
    parser.add_argument(
        "--arm",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="Which arm(s) to process (default: both)",
    )
    parser.add_argument(
        "--extrinsics-key",
        type=str,
        default="x5Dec13_2",
        help="Camera extrinsics key from EXTRINSICS dict (default: x5Dec13_2)",
    )
    parser.add_argument(
        "--prestack", action="store_true", help="Prestack future actions into chunks"
    )
    parser.add_argument(
        "--low-res",
        action="store_true",
        help="Downsample images to low resolution (240x320)",
    )
    parser.add_argument(
        "--no-rot", action="store_true", help="Skip rotation processing (position only)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the episode (default: 30)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Use write_incremental with add_frames instead of bulk write",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Frames per batch when using --incremental (default: 64)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir.absolute()}")
    print(f"Output directory exists: {args.output_dir.exists()}")

    # Convert single file or batch
    if args.hdf5_path:
        # Single file conversion
        if not args.hdf5_path.exists():
            print(f"Error: HDF5 file not found at {args.hdf5_path}")
            return 1

        # Generate output path with same base name
        zarr_filename = args.hdf5_path.stem + ".zarr"
        zarr_episode_path = args.output_dir / zarr_filename

        convert_hdf5_to_zarr(
            hdf5_path=args.hdf5_path,
            zarr_episode_path=zarr_episode_path,
            arm=args.arm,
            extrinsics_key=args.extrinsics_key,
            prestack=args.prestack,
            low_res=args.low_res,
            no_rot=args.no_rot,
            fps=args.fps,
            incremental=args.incremental,
            batch_size=args.batch_size,
        )

    else:
        # Batch conversion
        if not args.hdf5_dir.exists():
            print(f"Error: Directory not found at {args.hdf5_dir}")
            return 1

        hdf5_files = sorted(args.hdf5_dir.glob("*.hdf5"))
        if not hdf5_files:
            print(f"Error: No HDF5 files found in {args.hdf5_dir}")
            return 1

        print(f"\nFound {len(hdf5_files)} HDF5 files to convert")

        for hdf5_path in hdf5_files:
            # Generate output path with same base name
            zarr_filename = hdf5_path.stem + ".zarr"
            zarr_episode_path = args.output_dir / zarr_filename

            try:
                convert_hdf5_to_zarr(
                    hdf5_path=hdf5_path,
                    zarr_episode_path=zarr_episode_path,
                    arm=args.arm,
                    extrinsics_key=args.extrinsics_key,
                    prestack=args.prestack,
                    low_res=args.low_res,
                    no_rot=args.no_rot,
                    fps=args.fps,
                    incremental=args.incremental,
                    batch_size=args.batch_size,
                )
            except Exception as e:
                print(f"Error converting {hdf5_path}: {e}")
                import traceback

                traceback.print_exc()
                continue

    print("\n" + "=" * 60)
    print("✅ Conversion completed successfully!")
    print("=" * 60)
    print(f"\nOutput location: {args.output_dir.absolute()}")

    # List created files
    zarr_files = sorted(args.output_dir.glob("*.zarr"))
    if zarr_files:
        print(f"\nCreated {len(zarr_files)} Zarr episodes:")
        for zarr_file in zarr_files:
            print(f"  - {zarr_file.name}")
    else:
        print("\n⚠️  Warning: No .zarr files found in output directory")

    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    exit(main())
