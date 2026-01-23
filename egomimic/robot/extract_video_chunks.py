#!/usr/bin/env python3
"""
Extract frame segments and save to HDF5 format.

Takes a directory of PNG/JPEG frames and a segments file, extracts the specified
frame ranges, and saves each as an HDF5 file compatible with the demo format.

Segments file format (one segment per line):
    start_frame end_frame [demo_id]
    
demo_id is optional - if omitted, auto-increments from --start-id.

Example segments.txt:
    # Comments start with #
    0 150
    200 400
    500 750 100

Usage:
    python extract_video_chunks.py --frames-dir ./recording --segments segments.txt --output-dir ./demos --camera-key front_img_1
"""

import os
import sys
import argparse
import json
import numpy as np
import cv2
import h5py
from pathlib import Path
from typing import List, Tuple, Optional
from glob import glob


def parse_segments_file(filepath: str) -> List[Tuple[int, int, Optional[int]]]:
    """
    Parse segments file.
    
    Returns list of (start_frame, end_frame, demo_id) tuples.
    demo_id is None if not specified.
    """
    segments = []
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            
            if len(parts) < 2:
                print(f"Warning: Line {line_num} has fewer than 2 values, skipping: {line}")
                continue
            
            try:
                start_frame = int(parts[0])
                end_frame = int(parts[1])
                demo_id = int(parts[2]) if len(parts) > 2 else None
                
                if end_frame <= start_frame:
                    print(f"Warning: Line {line_num} has end <= start, skipping: {line}")
                    continue
                
                segments.append((start_frame, end_frame, demo_id))
                
            except ValueError as e:
                print(f"Warning: Line {line_num} parse error ({e}), skipping: {line}")
                continue
    
    return segments


def find_frame_files(frames_dir: Path) -> Tuple[List[Path], str]:
    """
    Find all frame files in directory.
    
    Returns:
        Tuple of (sorted list of frame paths, file extension)
    """
    # Try PNG first, then JPEG
    for ext in ['png', 'jpg', 'jpeg']:
        pattern = frames_dir / f"frame_*.{ext}"
        files = sorted(glob(str(pattern)))
        if files:
            return [Path(f) for f in files], ext
    
    raise FileNotFoundError(f"No frame files found in {frames_dir}")


def load_frames(
    frames_dir: Path,
    start_frame: int,
    end_frame: int,
) -> List[np.ndarray]:
    """
    Load frames from directory for the given range.
    
    Args:
        frames_dir: Directory containing frame files
        start_frame: First frame index (inclusive)
        end_frame: Last frame index (exclusive)
    
    Returns:
        List of RGB frames
    """
    frame_files, ext = find_frame_files(frames_dir)
    
    frames = []
    for frame_idx in range(start_frame, end_frame):
        frame_path = frames_dir / f"frame_{frame_idx:06d}.{ext}"
        
        if not frame_path.exists():
            print(f"  Warning: Frame {frame_idx} not found, stopping at {len(frames)} frames")
            break
        
        # Read and convert BGR to RGB
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  Warning: Failed to read frame {frame_idx}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)
    
    return frames


def save_to_hdf5(
    frames: List[np.ndarray],
    output_path: Path,
    camera_key: str,
    frame_rate: float,
    save_resolution: Optional[Tuple[int, int]] = None,
) -> bool:
    """
    Save frames to HDF5 file.
    
    Args:
        frames: List of RGB frames
        output_path: Output HDF5 file path
        camera_key: Camera key for the dataset
        frame_rate: Frame rate metadata
        save_resolution: Optional (width, height) to resize frames
    """
    if not frames:
        print("No frames to save!")
        return False
    
    # Process frames (optionally resize)
    processed_frames = []
    for frame in frames:
        if save_resolution is not None:
            frame = cv2.resize(frame, save_resolution, interpolation=cv2.INTER_CUBIC)
        processed_frames.append(frame)
    
    image_data = np.array(processed_frames)
    max_timesteps = len(frames)
    img_height, img_width = image_data.shape[1], image_data.shape[2]
    
    with h5py.File(str(output_path), "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["frame_rate"] = frame_rate
        root.attrs["camera_key"] = camera_key
        root.attrs["num_frames"] = max_timesteps
        
        obs = root.create_group("observations")
        image_group = obs.create_group("images")
        
        dset = image_group.create_dataset(
            camera_key,
            (max_timesteps, img_height, img_width, 3),
            dtype="uint8",
            chunks=(1, img_height, img_width, 3),
        )
        dset[...] = image_data
    
    return True


def extract_chunks(
    frames_dir: str,
    segments_path: str,
    output_dir: str,
    camera_key: str,
    start_id: int = 0,
    frame_rate: Optional[float] = None,
    save_width: Optional[int] = None,
    save_height: Optional[int] = None,
):
    """
    Extract frame segments and save to HDF5 files.
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    # Try to load metadata for frame rate
    metadata_path = frames_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        if frame_rate is None:
            frame_rate = metadata.get("frame_rate", 30)
        print(f"Loaded metadata: {metadata.get('frame_count', '?')} frames at {frame_rate} FPS")
    else:
        if frame_rate is None:
            frame_rate = 30
        print(f"No metadata found, using frame rate: {frame_rate}")
    
    # Parse segments
    segments = parse_segments_file(segments_path)
    
    if not segments:
        print("No valid segments found in segments file!")
        return
    
    print(f"Found {len(segments)} segments to extract")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine save resolution
    save_resolution = None
    if save_width is not None and save_height is not None:
        save_resolution = (save_width, save_height)
    
    # Process each segment
    current_id = start_id
    
    for i, (start_frame, end_frame, demo_id) in enumerate(segments):
        # Use specified demo_id or auto-increment
        if demo_id is not None:
            current_id = demo_id
        
        print(f"\nSegment {i+1}/{len(segments)}: frames {start_frame}-{end_frame} -> demo_{current_id}")
        
        # Load frames
        frames = load_frames(frames_dir, start_frame, end_frame)
        
        if not frames:
            print(f"  Warning: No frames loaded!")
            continue
        
        print(f"  Loaded {len(frames)} frames")
        
        # Save to HDF5
        output_path = output_dir / f"demo_{current_id}.hdf5"
        
        if save_to_hdf5(frames, output_path, camera_key, frame_rate, save_resolution):
            print(f"  Saved to {output_path}")
        
        # Auto-increment if demo_id wasn't specified
        if demo_id is None:
            current_id += 1
    
    print(f"\nDone! Extracted {len(segments)} segments to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frame segments and save to HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Segments file format (one segment per line):
    start_frame end_frame [demo_id]
    
Frame indices are 0-based. end_frame is exclusive (like Python ranges).
demo_id is optional - auto-increments from --start-id if omitted.

Example segments.txt:
    # Comment lines start with #
    0 150
    200 400
    500 750 100

Examples:
    python extract_video_chunks.py --frames-dir ./recording --segments segments.txt --output-dir ./demos --camera-key front_img_1
    
    # With resize
    python extract_video_chunks.py --frames-dir ./recording --segments segments.txt --output-dir ./demos --camera-key front_img_1 --save-width 320 --save-height 240
""",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory containing frame PNG/JPEG files",
    )
    parser.add_argument(
        "--segments",
        type=str,
        required=True,
        help="Segments file with frame ranges",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        required=True,
        help="Camera key for HDF5 dataset (e.g., 'front_img_1')",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Starting demo ID for auto-increment (default: 0)",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Frame rate for metadata (default: read from metadata.json or 30)",
    )
    parser.add_argument(
        "--save-width",
        type=int,
        default=None,
        help="Width to resize frames (optional)",
    )
    parser.add_argument(
        "--save-height",
        type=int,
        default=None,
        help="Height to resize frames (optional)",
    )
    
    args = parser.parse_args()
    
    # Validate resize args
    if (args.save_width is None) != (args.save_height is None):
        parser.error("Both --save-width and --save-height must be specified together")
    
    try:
        extract_chunks(
            frames_dir=args.frames_dir,
            segments_path=args.segments,
            output_dir=args.output_dir,
            camera_key=args.camera_key,
            start_id=args.start_id,
            frame_rate=args.frame_rate,
            save_width=args.save_width,
            save_height=args.save_height,
        )
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
