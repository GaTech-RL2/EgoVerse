"""
Convert HDF5 datasets to Zarr format.

This script reads HDF5 episode files and converts them to the Zarr format
expected by ZarrDataset, maintaining compatibility with the LeRobotDataset API.

Directory structure created:
    dataset_root/
    └── episode_{ep_idx}.zarr/
        ├── .zattrs (metadata)
        ├── observation.images.{cam}  (JPEG-XL compressed)
        ├── observation.state
        ├── actions_joints
        └── ...

Each episode is self-contained with its own metadata, enabling:
- Independent episode uploads to S3
- Parallel processing without global coordination
- Easy episode-level data management
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import imagecodecs
import numpy as np
import warnings
import zarr
from zarr.core.dtype import VariableLengthBytes

from egomimic.scripts.eva_process.eva_to_lerobot import (
    EvaHD5Extractor,
    EXTRINSICS,
)
from egomimic.utils.egomimicUtils import str2bool

logger = logging.getLogger(__name__)


def encode_jxl(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode image to JPEG-XL format.

    Args:
        image: RGB image array of shape (H, W, C) with dtype uint8
        quality: Compression quality (0-100), higher is better quality

    Returns:
        Compressed image bytes with padding byte to prevent null-byte stripping
    """
    # Add padding byte (0x01) to prevent zarr VariableLengthBytes from stripping trailing nulls
    return imagecodecs.jpegxl_encode(image, level=quality) + b'\x01'


def decode_jxl(data: bytes) -> np.ndarray:
    """Decode JPEG-XL image to numpy array.

    Args:
        data: Compressed image bytes

    Returns:
        RGB image array of shape (H, W, C) with dtype uint8
    """
    return imagecodecs.jpegxl_decode(data)


class HDF5ToZarrConverter:
    """Convert HDF5 episodes to Zarr format."""

    def __init__(
        self,
        raw_path: Path,
        output_path: Path,
        dataset_repo_id: str,
        fps: int,
        arm: str = "both",
        extrinsics_key: str = "x5Dec13_2",
        image_compressed: bool = True,
        jxl_quality: int = 50,
        prestack: bool = False,
        chunk_size_mb: float = 2.0,
        chunk_timesteps: int | None = None,
        debug: bool = False,
    ):
        self.raw_path = Path(raw_path)
        self.output_path = Path(output_path)
        self.dataset_repo_id = dataset_repo_id
        self.fps = fps
        self.arm = arm
        self.extrinsics_key = extrinsics_key
        self.image_compressed = image_compressed
        self.jxl_quality = jxl_quality
        self.prestack = prestack
        self.chunk_size_mb = chunk_size_mb
        self.chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
        self.chunk_timesteps = chunk_timesteps
        self.debug = debug

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Get episode list
        self.episode_list = sorted(self.raw_path.glob("*.hdf5"))
        if debug:
            self.episode_list = self.episode_list[:2]

        if not self.episode_list:
            raise ValueError(f"No HDF5 files found in {self.raw_path}")

        # Validate format
        EvaHD5Extractor.check_format(self.episode_list, image_compressed=image_compressed)

        # Robot type
        if self.arm == "both":
            self.robot_type = "eva_bimanual"
        elif self.arm == "right":
            self.robot_type = "eva_right_arm"
        elif self.arm == "left":
            self.robot_type = "eva_left_arm"
        else:
            self.robot_type = "eva"

        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Episode metadata (for summary logging)
        self.episodes_data = []
        self.total_frames = 0

        self.logger.info(f"{'-' * 10} HDF5 to Zarr Converter {'-' * 10}")
        self.logger.info(f"Input: {self.raw_path}")
        self.logger.info(f"Output: {self.output_path}")
        self.logger.info(f"Episodes to process: {len(self.episode_list)}")
        self.logger.info(f"JPEG-XL quality: {self.jxl_quality}")
        if self.chunk_timesteps is not None:
            self.logger.info(f"Chunk size: {self.chunk_timesteps} timesteps")
        else:
            self.logger.info(f"Chunk size: {self.chunk_size_mb} MB")

    def _write_zarr_episode(
        self,
        episode_path: Path,
        episode_idx: int,
        num_frames: int,
        frames_data: dict[str, list],
        image_data: dict[str, list[bytes]],
        image_shapes: dict[str, tuple],
        task: str = "",
    ):
        """Write episode frames and metadata to a Zarr store."""
        store = zarr.open(str(episode_path), mode="w")

        # Write numeric arrays with target chunk size
        for key, values in frames_data.items():
            arr = np.stack(values, axis=0)
            # Calculate frames per chunk
            if self.chunk_timesteps is not None:
                # Use explicit timestep-based chunking
                frames_per_chunk = self.chunk_timesteps
            else:
                # Calculate frames per chunk based on target MB
                bytes_per_frame = arr[0].nbytes
                if bytes_per_frame > 0:
                    frames_per_chunk = max(1, self.chunk_size_bytes // bytes_per_frame)
                else:
                    frames_per_chunk = num_frames
            # Cap to episode length
            frames_per_chunk = min(frames_per_chunk, num_frames)
            frames_per_chunk = max(1, frames_per_chunk)
            # Chunk shape: (frames, ...) - keep other dimensions intact
            chunk_shape = (frames_per_chunk,) + arr.shape[1:]
            store.create_array(
                key,
                data=arr,
                chunks=chunk_shape,
            )

        # Write compressed images as variable-length byte arrays
        with warnings.catch_warnings():
            # Suppress unstable spec warning for VariableLengthBytes
            warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification.*")
            for key, compressed_frames in image_data.items():
                arr = store.create_array(
                    key,
                    shape=(len(compressed_frames),),
                    dtype=VariableLengthBytes(),
                    chunks=(1,),  # One image per chunk for efficient random access
                )
                arr[:] = compressed_frames

        # Build features dict for this episode
        features = {}
        for key, shape in image_shapes.items():
            features[key] = {
                "dtype": "jxl",
                "shape": list(shape),
                "names": ["height", "width", "channel"],
            }

        info = {
            "episode_index": episode_idx,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "total_frames": num_frames,
            "task": task,
            "features": features,
        }

        # Attach metadata to zarr group attrs (Zarr v3-compatible)
        store.attrs.update(info)

        self.logger.info(f"Wrote {num_frames} frames to {episode_path}")

    def convert_episode(self, episode_path: Path, episode_idx: int, task: str = ""):
        """Convert a single HDF5 episode to Zarr format."""
        self.logger.info(f"Converting episode {episode_idx}: {episode_path}")
        t0 = time.time()

        # Get extrinsics
        extrinsics = EXTRINSICS[self.extrinsics_key]

        # Process episode using existing extractor
        episode_feats = EvaHD5Extractor.process_episode(
            episode_path,
            arm=self.arm,
            extrinsics=extrinsics,
            prestack=self.prestack,
            low_res=False,
            no_rot=False,
        )

        # Episode zarr path (directly under output_path)
        episode_zarr_path = self.output_path / f"episode_{episode_idx:06d}.zarr"

        # Get number of frames
        num_frames = next(iter(episode_feats["observations"].values())).shape[0]
        if self.debug:
            num_frames = min(num_frames, 10)  # Only process 10 frames in debug mode
        self.logger.info(f"Processing {num_frames} frames...")

        # Prepare data dictionaries
        frames_data = {}
        image_data = {}
        image_shapes = {}  # Track shapes for this episode

        # Extract frame-by-frame data
        for frame_idx in range(num_frames):
            if frame_idx % 25 == 0:
                self.logger.info(f"  Frame {frame_idx}/{num_frames}")
            for obs_key, obs_value in episode_feats["observations"].items():
                full_key = f"observation.{obs_key}"

                if "images" in obs_key:
                    # Compress image with JPEG-XL
                    if full_key not in image_data:
                        image_data[full_key] = []

                    img_raw = obs_value[frame_idx]
                    if self.image_compressed:
                        # Decompress from JPEG/PNG first
                        img_rgb = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
                        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                    else:
                        # EvaHD5Extractor returns (C, H, W), convert to (H, W, C) for encoding
                        img_rgb = np.transpose(img_raw, (1, 2, 0))

                    # Track shape for features (on first frame)
                    if full_key not in image_shapes:
                        image_shapes[full_key] = img_rgb.shape

                    # Compress with JPEG-XL
                    compressed = encode_jxl(img_rgb, quality=self.jxl_quality)
                    image_data[full_key].append(compressed)
                else:
                    # Store state data
                    if full_key not in frames_data:
                        frames_data[full_key] = []
                    frames_data[full_key].append(obs_value[frame_idx])

            # Add actions
            for action_key in ["actions_joints", "actions_cartesian", "actions_base_cartesian", "actions_eef_cartesian"]:
                if action_key in episode_feats:
                    if action_key not in frames_data:
                        frames_data[action_key] = []
                    frames_data[action_key].append(episode_feats[action_key][frame_idx])

        # Write zarr episode with metadata
        self._write_zarr_episode(
            episode_zarr_path,
            episode_idx=episode_idx,
            num_frames=num_frames,
            frames_data=frames_data,
            image_data=image_data,
            image_shapes=image_shapes,
            task=task,
        )

        # Update summary metadata
        self.episodes_data.append({
            "episode_index": episode_idx,
            "length": num_frames,
            "task": task,
        })
        self.total_frames += num_frames

        elapsed = time.time() - t0
        self.logger.info(f"Converted episode {episode_idx} in {elapsed:.2f}s ({num_frames} frames)")

    def convert_all(self, task_description: str = ""):
        """Convert all episodes."""
        for episode_idx, episode_path in enumerate(self.episode_list):
            try:
                self.convert_episode(episode_path, episode_idx, task=task_description)
            except Exception as e:
                self.logger.error(f"Failed to convert episode {episode_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.logger.info(f"Conversion complete: {len(self.episodes_data)} episodes, {self.total_frames} total frames")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert HDF5 episodes to Zarr format")

    parser.add_argument(
        "--raw-path",
        type=Path,
        required=True,
        help="Directory containing HDF5 files",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output directory for Zarr dataset",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=0,
        help="Dataset repository ID",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second",
    )
    parser.add_argument(
        "--arm",
        type=str,
        choices=["left", "right", "both"],
        default="both",
        help="Which arm(s) to process",
    )
    parser.add_argument(
        "--extrinsics-key",
        type=str,
        default="x5Dec13_2",
        help="Key for camera extrinsics",
    )
    parser.add_argument(
        "--image-compressed",
        type=str2bool,
        default=False,
        help="Whether images in HDF5 are compressed",
    )
    parser.add_argument(
        "--jxl-quality",
        type=int,
        default=50,
        help="JPEG-XL compression quality (0-100)",
    )
    parser.add_argument(
        "--prestack",
        type=str2bool,
        default=True,
        help="Prestack future actions",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default="",
        help="Task description for all episodes",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=float,
        default=2.0,
        help="Target chunk size in MB for numeric arrays (default: 2)",
    )
    parser.add_argument(
        "--chunk-timesteps",
        type=int,
        default=None,
        help="Number of timesteps per chunk (overrides --chunk-size-mb if specified)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only first 2 episodes",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    converter = HDF5ToZarrConverter(
        raw_path=args.raw_path,
        output_path=args.output_path,
        dataset_repo_id=args.dataset_repo_id,
        fps=args.fps,
        arm=args.arm,
        extrinsics_key=args.extrinsics_key,
        image_compressed=args.image_compressed,
        jxl_quality=args.jxl_quality,
        prestack=args.prestack,
        chunk_size_mb=args.chunk_size_mb,
        chunk_timesteps=args.chunk_timesteps,
        debug=args.debug,
    )

    converter.convert_all(task_description=args.task_description)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Output: {args.output_path}")
    print(f"Episodes: {len(converter.episodes_data)}")
    print(f"Total frames: {converter.total_frames}")
    print(f"{'='*60}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
