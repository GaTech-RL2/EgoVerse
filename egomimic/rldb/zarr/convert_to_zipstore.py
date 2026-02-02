#!/usr/bin/env python3
"""
Convert Zarr DirectoryStore episodes to ZipStore format for better NFS performance.

This reduces the number of file operations from ~20 per episode to 1,
significantly improving performance on network filesystems.
"""

import argparse
import shutil
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def convert_episode_to_zip(episode_path: Path, output_dir: Path, overwrite: bool = False) -> tuple[Path, bool, str]:
    """Convert a single episode DirectoryStore to ZipStore.

    Args:
        episode_path: Path to episode_*.zarr directory
        output_dir: Directory to write the .zip file
        overwrite: If True, overwrite existing zip files

    Returns:
        Tuple of (output_path, success, message)
    """
    output_path = output_dir / f"{episode_path.name}.zip"

    if output_path.exists() and not overwrite:
        return output_path, True, "already exists, skipped"

    try:
        # Create zip file with no compression (faster reads, zarr data is often already compressed)
        with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_STORED) as zf:
            for file_path in episode_path.rglob('*'):
                if file_path.is_file():
                    # Archive name is relative to episode directory
                    arcname = file_path.relative_to(episode_path)
                    zf.write(file_path, arcname)

        return output_path, True, "converted"
    except Exception as e:
        # Clean up partial file
        if output_path.exists():
            output_path.unlink()
        return output_path, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Zarr DirectoryStore episodes to ZipStore format"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Directory containing episode_*.zarr directories",
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir_flag",
        type=Path,
        default=None,
        help="Directory containing episode_*.zarr directories (alternative to positional)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: input_dir_zipstore)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing zip files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--copy-metadata",
        action="store_true",
        default=True,
        help="Copy meta/ directory and root files to output (default: True)",
    )
    args = parser.parse_args()

    # Support both positional and flag argument
    input_dir = args.input_dir or args.input_dir_flag
    if input_dir is None:
        print("Error: input_dir is required (positional or --input-dir)")
        parser.print_help()
        return 1

    input_dir = input_dir.resolve()
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    # Default output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_zipstore"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Find all episode directories
    episode_dirs = sorted([
        p for p in input_dir.iterdir()
        if p.is_dir() and p.name.startswith("episode_") and p.name.endswith(".zarr")
    ])

    if not episode_dirs:
        print("No episode_*.zarr directories found!")
        return 1

    print(f"Found {len(episode_dirs)} episodes to convert")

    # Copy metadata files (meta/, zarr.json, etc.)
    if args.copy_metadata:
        for item in input_dir.iterdir():
            if item.name.startswith("episode_"):
                continue
            dest = output_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
                print(f"Copied directory: {item.name}")
            elif item.is_file():
                shutil.copy2(item, dest)
                print(f"Copied file: {item.name}")

    # Convert episodes in parallel
    print(f"\nConverting episodes with {args.workers} workers...")

    success_count = 0
    skip_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert_episode_to_zip, ep, output_dir, args.overwrite): ep
            for ep in episode_dirs
        }

        for i, future in enumerate(as_completed(futures), 1):
            episode_path = futures[future]
            _, success, message = future.result()

            status = "✓" if success else "✗"
            print(f"[{i}/{len(episode_dirs)}] {status} {episode_path.name} -> {message}")

            if success:
                if "skipped" in message:
                    skip_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1

    print(f"\nDone! Converted: {success_count}, Skipped: {skip_count}, Failed: {fail_count}")

    # Show size comparison
    input_size = sum(f.stat().st_size for f in input_dir.rglob('*') if f.is_file())
    output_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    print(f"Input size:  {input_size / 1e9:.2f} GB")
    print(f"Output size: {output_size / 1e9:.2f} GB")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
