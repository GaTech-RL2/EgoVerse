"""
Duplicate Zarr or LeRobot episode files.

Creates copies of existing episode files with new episode indices,
useful for testing with larger datasets without needing additional raw data.

Supports:
- Zarr format: episode_*.zarr directories
- LeRobot format: data/chunk-*/episode_*.parquet + videos/
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def detect_format(input_path: Path) -> tuple[str, Path]:
    """Detect whether the dataset is Zarr or LeRobot format.

    Returns:
        Tuple of (format_name, dataset_root) where dataset_root is the
        actual path containing the data (for LeRobot, this is the
        processed/ subdirectory).
    """
    input_path = Path(input_path)

    # Check for Zarr format
    zarr_files = list(input_path.glob("episode_*.zarr"))
    if zarr_files:
        return "zarr", input_path

    # Check for LeRobot format - look for processed/ subdirectory
    processed_dir = input_path / "processed"
    if processed_dir.exists():
        meta_dir = processed_dir / "meta"
        data_dir = processed_dir / "data"
        if meta_dir.exists() and data_dir.exists():
            return "lerobot", input_path

    # Also check if input_path itself is a LeRobot dataset (meta/ and data/ directly)
    meta_dir = input_path / "meta"
    data_dir = input_path / "data"
    if meta_dir.exists() and data_dir.exists():
        return "lerobot", input_path.parent

    raise ValueError(
        f"Could not detect dataset format in {input_path}. "
        "Expected either episode_*.zarr files or a directory with processed/meta/ and processed/data/."
    )


def duplicate_zarr_episodes(
    input_path: Path,
    output_path: Path,
    num_duplicates: int = 1,
):
    """Duplicate .zarr episode files."""
    import zarr

    # Find existing episodes
    episode_files = sorted(input_path.glob("episode_*.zarr"))
    if not episode_files:
        raise ValueError(f"No episode_*.zarr files found in {input_path}")

    num_original = len(episode_files)
    print(f"Found {num_original} original Zarr episodes")
    print(f"Creating {num_duplicates} duplicate(s) of the dataset")

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy original episodes to output if different from input
    if output_path != input_path:
        for ep_file in episode_files:
            dest = output_path / ep_file.name
            if not dest.exists():
                print(f"Copying {ep_file.name} to output...")
                shutil.copytree(ep_file, dest)

    # Create duplicates
    for dup_idx in range(num_duplicates):
        for orig_idx, orig_file in enumerate(episode_files):
            new_idx = num_original * (dup_idx + 1) + orig_idx
            new_name = f"episode_{new_idx:06d}.zarr"
            new_path = output_path / new_name

            if new_path.exists():
                print(f"Skipping {new_name} (already exists)")
                continue

            print(f"Creating {new_name} from {orig_file.name}...")
            shutil.copytree(orig_file, new_path)

            # Update episode_index in metadata
            store = zarr.open(str(new_path), mode="r+")
            store.attrs["episode_index"] = new_idx

    total_episodes = num_original * (num_duplicates + 1)
    print(f"\nDone! Total Zarr episodes: {total_episodes}")


def duplicate_lerobot_episodes(
    input_path: Path,
    output_path: Path,
    num_duplicates: int = 1,
):
    """Duplicate LeRobot dataset by copying the entire processed/ folder.

    Creates multiple copies of the processed folder:
        output_path/
        ├── processed/      (original, if output != input)
        ├── processed_1/
        ├── processed_2/
        └── ...

    Args:
        input_path: Parent directory containing processed/ subdirectory
        output_path: Parent directory where duplicated processed_N/ folders will be created
        num_duplicates: Number of copies to create
    """
    # LeRobot datasets have structure: parent/processed/{meta,data,videos}
    processed_dir = input_path / "processed"
    if not processed_dir.exists():
        raise ValueError(f"Expected processed/ directory in {input_path}")

    print(f"Source: {processed_dir}")
    print(f"Creating {num_duplicates} duplicate(s) of processed/ folder")

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy original if output != input
    if output_path != input_path:
        dest = output_path / "processed"
        if not dest.exists():
            print(f"Copying processed/ to {dest}...")
            shutil.copytree(processed_dir, dest)

    # Create duplicates
    for dup_idx in range(1, num_duplicates + 1):
        dest = output_path / f"processed_{dup_idx}"
        if dest.exists():
            print(f"Skipping processed_{dup_idx}/ (already exists)")
            continue

        print(f"Creating processed_{dup_idx}/...")
        shutil.copytree(processed_dir, dest)

    total_folders = num_duplicates + 1
    print(f"\nDone! Created {total_folders} processed folders")


def duplicate_episodes(
    input_path: Path,
    output_path: Path | None = None,
    num_duplicates: int = 1,
    format: str | None = None,
):
    """Duplicate episodes in Zarr or LeRobot format.

    Args:
        input_path: Directory containing the dataset
            - For Zarr: directory with episode_*.zarr files
            - For LeRobot: parent directory containing processed/ subdirectory
        output_path: Output directory (defaults to input_path)
        num_duplicates: Number of times to duplicate the dataset
        format: Force format ("zarr" or "lerobot"), auto-detected if None
    """
    input_path = Path(input_path)

    # Detect or use specified format
    if format is None:
        detected_format, dataset_root = detect_format(input_path)
        format = detected_format
        # For LeRobot, use the detected parent directory
        if format == "lerobot":
            input_path = dataset_root
    else:
        # If format is forced, still need to find the right root for LeRobot
        if format == "lerobot":
            # Check if input_path has processed/ or is a processed/ dir itself
            if (input_path / "processed").exists():
                pass  # input_path is correct (parent)
            elif (input_path / "meta").exists():
                input_path = input_path.parent  # input_path is processed/, go up

    output_path = Path(output_path) if output_path else input_path

    print(f"Detected format: {format}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")

    if format == "zarr":
        duplicate_zarr_episodes(input_path, output_path, num_duplicates)
    elif format == "lerobot":
        duplicate_lerobot_episodes(input_path, output_path, num_duplicates)
    else:
        raise ValueError(f"Unknown format: {format}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Duplicate Zarr or LeRobot episode files"
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Directory containing the dataset. "
             "For Zarr: directory with episode_*.zarr files. "
             "For LeRobot: parent directory containing processed/ subdirectory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output directory (defaults to input-path for in-place duplication). "
             "For LeRobot, this is the parent where processed/ will be created.",
    )
    parser.add_argument(
        "-n", "--num-duplicates",
        type=int,
        default=1,
        help="Number of times to duplicate the dataset (default: 1)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["zarr", "lerobot"],
        default=None,
        help="Force dataset format (auto-detected if not specified)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    duplicate_episodes(
        input_path=args.input_path,
        output_path=args.output_path,
        num_duplicates=args.num_duplicates,
        format=args.format,
    )


if __name__ == "__main__":
    main()
