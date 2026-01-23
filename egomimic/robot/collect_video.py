#!/usr/bin/env python3
"""
Video recorder - records camera feed to both raw frames and video file.

Usage:
    # Record both raw frames (for HDF5 extraction) and video (for viewing)
    python collect_video.py --config configs_yam.yaml --camera-key front_img_1 --output-dir ./recording

    # Video only (for quick viewing/scrubbing)
    python collect_video.py --config configs_yam.yaml --camera-key front_img_1 --output-dir ./recording --no-frames

    # Frames only (for lossless quality)
    python collect_video.py --config configs_yam.yaml --camera-key front_img_1 --output-dir ./recording --no-video

Outputs:
- Raw frames in tmp/ subdirectory (lossless PNG for HDF5 extraction)
- Video file for easy viewing and frame navigation

To extract segments to HDF5:
    python extract_video_chunks.py --frames-dir ./recording/tmp --segments segments.txt --output-dir ./demos
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Optional

# Add path for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))

from robot_utils import RateLoop


DEFAULT_FRAME_RATE = 30


def create_camera_recorder(cam_cfg: dict):
    """Create a camera recorder based on camera configuration."""
    cam_type = cam_cfg.get("type", "").lower()
    
    if not cam_cfg.get("enabled", True):
        raise ValueError("Camera is disabled in config")
    
    if cam_type == "d405":
        from stream_d405 import RealSenseRecorder
        serial = str(cam_cfg.get("serial_number", ""))
        width = cam_cfg.get("width", 640)
        height = cam_cfg.get("height", 480)
        return RealSenseRecorder(serial, width=width, height=height)
    elif cam_type == "aria":
        from stream_aria import AriaRecorder
        recorder = AriaRecorder(show_image=False, save_location=None)
        recorder.start()
        return recorder
    else:
        raise ValueError(f"Unknown camera type '{cam_type}'. Supported: d405, aria")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def wait_for_camera(recorder, timeout: float = 5.0) -> bool:
    """Wait for camera to produce valid frames."""
    start = time.time()
    while time.time() - start < timeout:
        img = recorder.get_image()
        if img is not None:
            return True
        time.sleep(0.05)
    return False


def record(
    config_path: str,
    camera_key: str,
    output_dir: str,
    frame_rate: int = DEFAULT_FRAME_RATE,
    save_frames: bool = True,
    save_video: bool = True,
    jpeg_quality: Optional[int] = None,
    video_codec: str = "mp4v",
):
    """
    Record from camera to raw frames and/or video file.
    
    Press Ctrl+C to stop recording.
    
    Args:
        config_path: Path to camera config YAML
        camera_key: Camera key from config
        output_dir: Base output directory
        frame_rate: Target frame rate
        save_frames: Save raw frames to tmp/ subdirectory
        save_video: Save video file
        jpeg_quality: If set (1-100), save frames as JPEG instead of PNG
        video_codec: Video codec fourcc (default: mp4v)
    """
    if not save_frames and not save_video:
        raise ValueError("At least one of --no-frames or --no-video must be omitted")
    
    # Load config
    config = load_config(config_path)
    
    if "cameras" not in config:
        raise ValueError(f"No 'cameras' section in config: {config_path}")
    if camera_key not in config["cameras"]:
        available = list(config["cameras"].keys())
        raise ValueError(f"Camera key '{camera_key}' not found. Available: {available}")
    
    cam_cfg = config["cameras"][camera_key]
    
    # Initialize camera
    print(f"Initializing camera '{camera_key}'...")
    recorder = create_camera_recorder(cam_cfg)
    
    print("Waiting for camera to be ready...")
    if not wait_for_camera(recorder):
        print("ERROR: Camera not producing frames!")
        return
    
    # Get first frame to determine dimensions
    first_frame = recorder.get_image()
    height, width = first_frame.shape[:2]
    
    # Setup output directories and files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Frames directory (tmp subdirectory)
    frames_dir = None
    frame_ext = None
    encode_params = None
    if save_frames:
        frames_dir = output_dir / "tmp"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        if jpeg_quality is not None:
            frame_ext = "jpg"
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            frame_format_str = f"JPEG (quality={jpeg_quality})"
        else:
            frame_ext = "png"
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # Fast compression
            frame_format_str = "PNG (lossless)"
    
    # Video writer
    video_writer = None
    video_path = None
    if save_video:
        video_ext = "avi" if video_codec.upper() == "XVID" else "mp4"
        video_path = output_dir / f"recording.{video_ext}"
        fourcc = cv2.VideoWriter_fourcc(*video_codec)
        video_writer = cv2.VideoWriter(str(video_path), fourcc, frame_rate, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")
    
    # Print configuration
    print("\n" + "=" * 60)
    print("VIDEO RECORDER")
    print("=" * 60)
    print(f"Camera:      {camera_key}")
    print(f"Resolution:  {width}x{height}")
    print(f"Frame rate:  {frame_rate} Hz")
    print("-" * 60)
    if save_frames:
        print(f"Frames:      {frames_dir}")
        print(f"Format:      {frame_format_str}")
    else:
        print("Frames:      DISABLED")
    print("-" * 60)
    if save_video:
        print(f"Video:       {video_path}")
        print(f"Codec:       {video_codec}")
        print(f"Overlay:     Frame numbers (top-left)")
    else:
        print("Video:       DISABLED")
    print("=" * 60)
    print("\nRecording... Press Ctrl+C to stop.\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        with RateLoop(frequency=frame_rate, verbose=False) as loop:
            for _ in loop:
                frame = recorder.get_image()
                if frame is None:
                    continue
                
                # Save raw frame (BGR from RealSense) - unmodified
                if save_frames:
                    frame_path = frames_dir / f"frame_{frame_count:06d}.{frame_ext}"
                    cv2.imwrite(str(frame_path), frame, encode_params)
                
                # Write to video with frame number overlay
                if save_video:
                    video_frame = frame.copy()
                    # Draw frame number with background for visibility
                    text = f"Frame: {frame_count}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    # Black background rectangle
                    cv2.rectangle(video_frame, (5, 5), (15 + text_w, 15 + text_h + baseline), (0, 0, 0), -1)
                    # White text
                    cv2.putText(video_frame, text, (10, 10 + text_h), font, font_scale, (255, 255, 255), thickness)
                    video_writer.write(video_frame)
                
                frame_count += 1
                
                # Print progress every second
                elapsed = time.time() - start_time
                if frame_count % frame_rate == 0:
                    print(f"\rFrames: {frame_count} | Time: {elapsed:.1f}s | FPS: {frame_count/elapsed:.1f}", end="", flush=True)
    
    except KeyboardInterrupt:
        print("\n\nStopping recording...")
    
    finally:
        if video_writer is not None:
            video_writer.release()
        try:
            recorder.stop()
        except:
            pass
    
    elapsed = time.time() - start_time
    
    # Save metadata
    metadata = {
        "frame_count": frame_count,
        "frame_rate": frame_rate,
        "width": width,
        "height": height,
        "camera_key": camera_key,
        "duration_seconds": elapsed,
        "has_frames": save_frames,
        "has_video": save_video,
    }
    if save_frames:
        metadata["frames_dir"] = "tmp"
        metadata["frame_format"] = frame_ext
    if save_video:
        metadata["video_file"] = video_path.name
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print(f"\n" + "=" * 60)
    print("RECORDING COMPLETE")
    print("=" * 60)
    print(f"Frames:      {frame_count}")
    print(f"Duration:    {elapsed:.1f} seconds")
    print(f"Actual FPS:  {frame_count/elapsed:.1f}")
    print("-" * 60)
    if save_frames:
        print(f"Raw frames:  {frames_dir}")
    if save_video:
        print(f"Video file:  {video_path}")
    print(f"Metadata:    {metadata_path}")
    print("=" * 60)
    
    print(f"\nFrame range: 0 - {frame_count - 1}")
    
    if save_video:
        print(f"\nTo find frame numbers, just play the video - frame numbers are overlaid!")
        print(f"  mpv {video_path}")
        print(f"  vlc {video_path}")
    
    if save_frames:
        print(f"\nTo extract segments, create segments.txt:")
        print(f"  # start_frame end_frame [demo_id]")
        print(f"  0 150")
        print(f"  200 400")
        print(f"\nThen run:")
        print(f"  python extract_video_chunks.py --frames-dir {frames_dir} --segments segments.txt --output-dir ./demos --camera-key {camera_key}")


def main():
    parser = argparse.ArgumentParser(
        description="Record camera to raw frames and/or video file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record both frames (lossless) and video (for viewing)
  python collect_video.py --config configs_yam.yaml --camera-key front_img_1 --output-dir ./recording
  
  # Video only (smaller, for quick viewing)
  python collect_video.py --config configs_yam.yaml --camera-key front_img_1 --output-dir ./recording --no-frames
  
  # Frames only (lossless, for ML training)
  python collect_video.py --config configs_yam.yaml --camera-key front_img_1 --output-dir ./recording --no-video
  
  # Frames as JPEG (smaller but slight quality loss)
  python collect_video.py --config configs_yam.yaml --camera-key front_img_1 --output-dir ./recording --jpeg-quality 95

After recording:
  1. Open the video in mpv/VLC to find frame numbers
  2. Create segments.txt with frame ranges
  3. Run extract_video_chunks.py to create HDF5 files
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        required=True,
        help="Camera key from config (e.g., 'front_img_1')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=DEFAULT_FRAME_RATE,
        help=f"Target frame rate (default: {DEFAULT_FRAME_RATE})",
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Don't save raw frames (video only)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Don't save video file (frames only)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="Save frames as JPEG with this quality (1-100). Default: PNG (lossless)",
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="mp4v",
        help="Video codec fourcc (default: mp4v). Use 'XVID' for .avi",
    )
    
    args = parser.parse_args()
    
    if args.jpeg_quality is not None and not (1 <= args.jpeg_quality <= 100):
        parser.error("--jpeg-quality must be between 1 and 100")
    
    if args.no_frames and args.no_video:
        parser.error("Cannot use both --no-frames and --no-video")
    
    try:
        record(
            config_path=args.config,
            camera_key=args.camera_key,
            output_dir=args.output_dir,
            frame_rate=args.frame_rate,
            save_frames=not args.no_frames,
            save_video=not args.no_video,
            jpeg_quality=args.jpeg_quality,
            video_codec=args.video_codec,
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
