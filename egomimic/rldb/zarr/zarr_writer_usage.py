#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert HDF5 episodes to Zarr format using ZarrWriter.

This module provides a simple utility for converting HDF5 episode files
to the Zarr v3 format compatible with ZarrEpisode reader.

Usage:
    # Convert a single episode
    python zarr_writer_usage.py --hdf5-path /path/to/episode.hdf5 --output-dir /output/zarr --episode-index 0

    # Convert all episodes in a directory
    python zarr_writer_usage.py --hdf5-dir /path/to/episodes --output-dir /output/zarr
"""

import argparse
import h5py
from pathlib import Path
from egomimic.rldb.zarr import ZarrWriter, ZarrEpisode


def convert_hdf5_to_zarr(hdf5_path: str | Path, zarr_episode_path: str | Path):
    """
    Convert an HDF5 episode file to Zarr format.

    Args:
        hdf5_path: Path to the input HDF5 file
        zarr_episode_path: Output path for the Zarr episode (e.g., /data/episode_000000.zarr)

    Returns:
        Path to created Zarr episode

    Example:
        >>> convert_hdf5_to_zarr(
        ...     hdf5_path="/data/episodes/episode_000000.hdf5",
        ...     zarr_episode_path="/data/zarr_episodes/episode_000000.zarr",
        ... )
    """
    print("\n" + "=" * 60)
    print(f"Converting HDF5 to Zarr: {hdf5_path}")
    print("=" * 60)

    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        print(f"Error: HDF5 file not found at {hdf5_path}")
        return None

    with h5py.File(hdf5_path, "r") as f:
        # Read all datasets from HDF5
        all_data = {}

        def collect_datasets(name, obj):
            """Recursively collect all datasets from HDF5 file."""
            if isinstance(obj, h5py.Dataset):
                # Convert HDF5 path separators to dots for Zarr keys
                key = name.replace("/", ".")
                all_data[key] = obj[:]
                print(f"  Found dataset: {name} -> shape {obj.shape}")

        f.visititems(collect_datasets)

        # Extract metadata from HDF5 attributes
        metadata = {
            "fps": f.attrs.get("fps", 30),
            "robot_type": f.attrs.get("robot_type", ""),
            "task": f.attrs.get("task", ""),
        }

        # Write to Zarr (images are auto-detected as 4D arrays with shape T×H×W×3)
        zarr_path = ZarrWriter.create_and_write(
            data=all_data,
            episode_path=zarr_episode_path,
            fps=metadata["fps"],
            robot_type=metadata["robot_type"],
            task=metadata["task"],
        )

    print(f"\nSuccessfully converted to: {zarr_path}")

    # Verify the conversion
    episode = ZarrEpisode(zarr_path)
    print(f"Zarr episode has {len(episode)} frames")
    print(f"Available keys: {episode.keys}")

    return zarr_path


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert HDF5 episodes to Zarr format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single episode
  %(prog)s --hdf5-path /data/episode_000000.hdf5 --output-dir /data/zarr

  # Convert all episodes in a directory
  %(prog)s --hdf5-dir /data/episodes --output-dir /data/zarr

Expected HDF5 structure:
  /observations/state          -> (T, state_dim)
  /observations/images/camera  -> (T, H, W, 3) for images
  /action                      -> (T, action_dim)
  attributes: fps, robot_type, task

Image arrays (4D with shape T×H×W×3) will be automatically detected and JPEG-compressed.
        """
    )

    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--hdf5-path",
        type=Path,
        help="Path to a single HDF5 episode file to convert"
    )
    group.add_argument(
        "--hdf5-dir",
        type=Path,
        help="Directory containing HDF5 episode files (*.hdf5) to convert"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for Zarr episodes"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

            convert_hdf5_to_zarr(
                hdf5_path=hdf5_path,
                zarr_episode_path=zarr_episode_path,
            )

    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    exit(main())
