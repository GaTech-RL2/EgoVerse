"""
Zarr-based dataset utilities for EgoVerse.

This module provides a drop-in replacement for LeRobot-based datasets using Zarr
as the storage backend. Key features:
- JPEG-compressed images with per-frame random access
- On-the-fly action chunking (no 100x storage blowup)
- Self-contained annotations
- Compatible with existing DataSchematic and training pipeline
"""

import json
import logging
import random
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import simplejpeg
import torch
import zarr
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray
import numcodecs


# ============================================================================
# Embodiment definitions (copied from utils.py to avoid lerobot dependency)
# ============================================================================

class EMBODIMENT(Enum):
    EVE_RIGHT_ARM = 0
    EVE_LEFT_ARM = 1
    EVE_BIMANUAL = 2
    ARIA_RIGHT_ARM = 3
    ARIA_LEFT_ARM = 4
    ARIA_BIMANUAL = 5
    EVA_RIGHT_ARM = 6
    EVA_LEFT_ARM = 7
    EVA_BIMANUAL = 8
    MECKA_BIMANUAL = 9
    MECKA_RIGHT_ARM = 10
    MECKA_LEFT_ARM = 11


EMBODIMENT_ID_TO_KEY = {
    member.value: key for key, member in EMBODIMENT.__members__.items()
}

SEED = 42


def get_embodiment(index: int) -> Optional[str]:
    """Get embodiment name from ID."""
    return EMBODIMENT_ID_TO_KEY.get(index, None)


def get_embodiment_id(embodiment_name: str) -> int:
    """Get embodiment ID from name."""
    return EMBODIMENT[embodiment_name.upper()].value


def split_dataset_names(dataset_names, valid_ratio=0.2, seed=SEED):
    """Split dataset names into train/valid sets."""
    names = sorted(dataset_names)
    if not names:
        return set(), set()

    rng = random.Random(seed)
    rng.shuffle(names)

    if not (0.0 <= valid_ratio <= 1.0):
        raise ValueError(f"valid_ratio must be in [0,1], got {valid_ratio}")

    n_valid = int(len(names) * valid_ratio)
    if valid_ratio > 0.0:
        n_valid = max(1, n_valid)

    valid = set(names[:n_valid])
    train = set(names[n_valid:])
    return train, valid


class SimplejpegCodec(Codec):
    """
    Zarr codec for JPEG compression using simplejpeg.

    Designed for per-frame image chunks with shape (1, H, W, 3) uint8 RGB.
    Zarr passes flattened bytes to encode/decode; we reshape accordingly.

    Usage:
        codec = SimplejpegCodec(quality=85, height=360, width=640)
        arr = zarr.open_array(
            path, mode='w',
            shape=(T, H, W, 3),
            chunks=(1, H, W, 3),
            dtype=np.uint8,
            compressor=codec,
        )

    Why dimensions in codec?
        Zarr passes flattened bytes to the compressor. To reshape for JPEG
        encoding, we need to know the image dimensions. These are stored in
        the codec config and persisted with the Zarr array metadata.

    Compression:
        Typical 10-20x compression ratio for natural images at quality=85.
        JPEG is lossy; mean pixel difference ~5-10 for quality=85.
    """

    codec_id = "simplejpeg"

    def __init__(self, quality: int = 85, height: int = 360, width: int = 640):
        """
        Initialize JPEG codec.

        Args:
            quality: JPEG quality (1-100). Higher = better quality, larger files.
                     85 is a good balance. 95+ for near-lossless.
            height: Image height in pixels.
            width: Image width in pixels.

        Raises:
            ValueError: If quality not in [1, 100] or dimensions not positive.
        """
        if not 1 <= quality <= 100:
            raise ValueError(f"quality must be in [1, 100], got {quality}")
        if height <= 0 or width <= 0:
            raise ValueError(f"dimensions must be positive, got ({height}, {width})")

        self.quality = quality
        self.height = height
        self.width = width
        self._expected_size = height * width * 3

    def encode(self, buf) -> bytes:
        """
        Encode flattened RGB image chunk to JPEG bytes.

        Args:
            buf: Flattened uint8 array of size (1 * H * W * 3).

        Returns:
            JPEG-compressed bytes.

        Raises:
            ValueError: If buffer size doesn't match expected dimensions.
        """
        arr = ensure_contiguous_ndarray(buf)

        if arr.size != self._expected_size:
            raise ValueError(
                f"Buffer size {arr.size} doesn't match expected {self._expected_size} "
                f"for image ({self.height}, {self.width}, 3). "
                f"Did you use chunks=(1, H, W, 3)?"
            )

        # Reshape to (H, W, 3)
        img = arr.reshape(self.height, self.width, 3)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # Ensure C-contiguous for simplejpeg
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)

        return simplejpeg.encode_jpeg(img, quality=self.quality, colorspace='RGB')

    def decode(self, buf, out=None) -> np.ndarray:
        """
        Decode JPEG bytes back to flattened RGB array.

        Args:
            buf: JPEG-compressed bytes.
            out: Optional pre-allocated output array.

        Returns:
            Flattened uint8 array of size (H * W * 3).
        """
        if isinstance(buf, np.ndarray):
            buf = buf.tobytes()

        # Decode JPEG -> (H, W, 3)
        img = simplejpeg.decode_jpeg(buf, colorspace='RGB')

        # Flatten for Zarr
        result = img.ravel()

        if out is not None:
            np.copyto(out.ravel(), result)
            return out
        return result

    def get_config(self) -> dict:
        """Return codec configuration for serialization."""
        return {
            'id': self.codec_id,
            'quality': self.quality,
            'height': self.height,
            'width': self.width,
        }

    @classmethod
    def from_config(cls, config: dict) -> 'SimplejpegCodec':
        """Reconstruct codec from configuration."""
        return cls(
            quality=config.get('quality', 85),
            height=config.get('height', 360),
            width=config.get('width', 640),
        )

    def __repr__(self) -> str:
        return f"SimplejpegCodec(quality={self.quality}, height={self.height}, width={self.width})"


# Register the codec with numcodecs
try:
    numcodecs.register_codec(SimplejpegCodec)
except ValueError:
    pass  # Already registered

logger = logging.getLogger(__name__)


# Key mapping from Zarr structure to LeRobot-compatible keys
ZARR_TO_LEROBOT_KEYS = {
    "rgb/front_img_1": "observations.images.front_img_1",
    "state/ee_pose_cam": "observations.state.ee_pose_cam",
    "actions/ee_cartesian_cam": "actions_ee_cartesian_cam",
    "keypoints/hand_keypoints_world": "actions_ee_keypoints_world",
    "head/pose_world": "actions_head_cartesian_world",
}

LEROBOT_TO_ZARR_KEYS = {v: k for k, v in ZARR_TO_LEROBOT_KEYS.items()}


class ZarrDatasetMetadata:
    """Metadata container for Zarr datasets, compatible with LeRobotDatasetMetadata."""

    def __init__(self, zarr_root: Union[str, Path], local_files_only: bool = True):
        self.root = Path(zarr_root)
        self.local_files_only = local_files_only

        # Load consolidated metadata
        self.store = zarr.open(str(self.root), mode="r")

        # Load dataset-level attrs
        self.attrs = dict(self.store.attrs)

        # Load meta files
        self.info = self._load_json("meta/info.json")
        self.stats = self._load_json("meta/stats.json")
        self.tasks = self._load_json("meta/tasks.json")
        self.episodes = self._load_episodes()

    def _load_json(self, path: str) -> dict:
        """Load JSON from meta directory."""
        full_path = self.root / path
        if full_path.exists():
            with open(full_path, "r") as f:
                return json.load(f)
        return {}

    def _load_episodes(self) -> List[dict]:
        """Load episode metadata from episodes.jsonl."""
        episodes_path = self.root / "meta" / "episodes.jsonl"
        episodes = []
        if episodes_path.exists():
            with open(episodes_path, "r") as f:
                for line in f:
                    if line.strip():
                        episodes.append(json.loads(line))
        return episodes

    @property
    def robot_type(self) -> str:
        return self.info.get("robot_type", "UNKNOWN")

    @property
    def fps(self) -> int:
        return self.info.get("fps", 30)

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    @property
    def total_frames(self) -> int:
        return sum(ep.get("length", 0) for ep in self.episodes)

    def _update_splits(self, valid_ratio: float = 0.2):
        """Update train/valid splits in info dict."""
        episode_indices = list(range(len(self.episodes)))
        train_set, valid_set = split_dataset_names(
            [str(i) for i in episode_indices], valid_ratio=valid_ratio, seed=SEED
        )
        self.info["splits"] = {
            "train": [int(i) for i in train_set],
            "valid": [int(i) for i in valid_set],
        }


class ZarrDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset backed by Zarr storage.

    Features:
    - JPEG-compressed images with per-frame random access
    - On-the-fly action chunking (computes 100-step chunks from raw actions)
    - Compatible with LeRobot key naming conventions
    - Supports embodiment tracking for multi-embodiment training
    """

    def __init__(
        self,
        zarr_root: Union[str, Path],
        chunk_size: int = 100,
        episodes: Optional[List[int]] = None,
        image_transforms=None,
        fps: int = 30,
    ):
        """
        Initialize ZarrDataset.

        Args:
            zarr_root: Path to zarr store root
            chunk_size: Number of future timesteps for action chunking
            episodes: Optional list of episode indices to include
            image_transforms: Optional torchvision transforms for images
            fps: Frames per second (for timestamp computation)
        """
        self.zarr_root = Path(zarr_root)
        self.chunk_size = chunk_size
        self.image_transforms = image_transforms
        self.fps = fps

        # Open zarr store
        self.store = zarr.open(str(self.zarr_root), mode="r")

        # Load metadata
        self.meta = ZarrDatasetMetadata(zarr_root)

        # Get embodiment from metadata
        self.embodiment = get_embodiment_id(self.meta.robot_type)

        # Build frame index
        self._build_index(episodes)

        logger.info(
            f"ZarrDataset initialized: {len(self)} frames, "
            f"{len(self.episode_hashes)} episodes, embodiment={self.embodiment}"
        )

    def _build_index(self, episodes: Optional[List[int]] = None):
        """Build global frame index mapping idx -> (episode_hash, local_frame_idx)."""
        self.index_map = []
        self.episode_hashes = []
        self.episode_lengths = {}
        self.episode_hash_to_idx = {}  # O(1) lookup for episode index

        for ep_idx, ep_meta in enumerate(self.meta.episodes):
            if episodes is not None and ep_idx not in episodes:
                continue

            ep_hash = ep_meta.get("hash", ep_meta.get("episode_id", f"ep_{ep_idx:06d}"))
            length = ep_meta.get("length", 0)

            self.episode_hash_to_idx[ep_hash] = len(self.episode_hashes)
            self.episode_hashes.append(ep_hash)
            self.episode_lengths[ep_hash] = length

            for local_idx in range(length):
                self.index_map.append((ep_hash, local_idx))

    @lru_cache(maxsize=32)
    def _get_episode_group(self, episode_hash: str) -> zarr.Group:
        """Get episode group from zarr store (cached)."""
        return self.store[f"episodes/{episode_hash}"]

    def _get_action_chunk(
        self, actions: zarr.Array, t: int, chunk_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute action chunk on-the-fly (vectorized).

        Args:
            actions: Zarr array of shape (T, action_dim)
            t: Current timestep
            chunk_size: Override chunk size (default: self.chunk_size)

        Returns:
            Action chunk of shape (chunk_size, action_dim)
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        T = actions.shape[0]
        end_t = min(t + chunk_size, T)

        # Load only needed frames
        chunk = actions[t:end_t]  # (actual_len, action_dim)

        # Pad if near episode end
        if len(chunk) < chunk_size:
            pad_len = chunk_size - len(chunk)
            last_frame = chunk[-1:]  # (1, action_dim)
            padding = np.repeat(last_frame, pad_len, axis=0)
            chunk = np.concatenate([chunk, padding], axis=0)

        return chunk.astype(np.float32)

    def _load_image(self, ep_group: zarr.Group, local_idx: int, key: str = "rgb/front_img_1") -> torch.Tensor:
        """Load and convert image to tensor."""
        # JPEG is auto-decoded by numcodecs
        img = ep_group[key][local_idx]  # (H, W, 3), uint8

        # Convert to tensor: (H, W, 3) -> (3, H, W), float [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.image_transforms is not None:
            img_tensor = self.image_transforms(img_tensor)

        return img_tensor

    def _get_annotation_for_frame(self, ep_group: zarr.Group, local_idx: int) -> str:
        """Get annotation label for a specific frame."""
        try:
            annotations = ep_group["annotations"]
            label_ids = annotations["label_id"][:]
            start_frames = annotations["start_frame"][:]
            end_frames = annotations["end_frame"][:]

            # Get label vocabulary from attrs
            label_vocab = ep_group.attrs.get("label_vocab", [])

            # Find matching annotation
            for i, (label_id, start, end) in enumerate(zip(label_ids, start_frames, end_frames)):
                if start <= local_idx <= end:
                    if label_id < len(label_vocab):
                        return label_vocab[label_id]
                    return f"label_{label_id}"

            return ""
        except (KeyError, IndexError):
            return ""

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single frame with all observations and actions.

        Returns dict with LeRobot-compatible keys:
        - observations.images.front_img_1: (3, H, W) float tensor
        - observations.state.ee_pose_cam: (12,) float tensor
        - actions_ee_cartesian_cam: (chunk_size, 12) float tensor
        - actions_ee_keypoints_world: (chunk_size, 126) float tensor
        - actions_head_cartesian_world: (10,) float tensor
        - metadata.embodiment: (1,) int tensor
        - timestamp: float
        - frame_index: int
        - episode_index: int
        """
        episode_hash, local_idx = self.index_map[idx]
        ep = self._get_episode_group(episode_hash)

        # Get episode-level metadata
        ep_attrs = dict(ep.attrs)
        embodiment_id = ep_attrs.get("embodiment_id", self.embodiment)
        fps = ep_attrs.get("fps", self.fps)

        item = {}

        # Load image
        try:
            item["observations.images.front_img_1"] = self._load_image(ep, local_idx)
        except KeyError:
            # Image key might not exist
            pass

        # Load state (single frame)
        try:
            state = ep["state/ee_pose_cam"][local_idx]
            item["observations.state.ee_pose_cam"] = torch.from_numpy(state).float()
        except KeyError:
            pass

        # Compute action chunks on-the-fly
        try:
            actions = self._get_action_chunk(ep["actions/ee_cartesian_cam"], local_idx)
            item["actions_ee_cartesian_cam"] = torch.from_numpy(actions).float()
        except KeyError:
            pass

        try:
            keypoints = self._get_action_chunk(ep["keypoints/hand_keypoints_world"], local_idx)
            item["actions_ee_keypoints_world"] = torch.from_numpy(keypoints).float()
        except KeyError:
            pass

        # Head pose (single frame, not chunked)
        try:
            head_pose = ep["head/pose_world"][local_idx]
            item["actions_head_cartesian_world"] = torch.from_numpy(head_pose).float()
        except KeyError:
            pass

        # Metadata
        item["metadata.embodiment"] = torch.tensor([embodiment_id], dtype=torch.int32)
        item["timestamp"] = torch.tensor([local_idx / fps], dtype=torch.float32)
        item["frame_index"] = torch.tensor([local_idx], dtype=torch.int64)
        item["episode_index"] = torch.tensor(
            [self.episode_hash_to_idx[episode_hash]], dtype=torch.int64
        )

        # Annotation (for language models)
        item["annotations"] = self._get_annotation_for_frame(ep, local_idx)

        return item

    @property
    def num_episodes(self) -> int:
        return len(self.episode_hashes)

    @property
    def num_frames(self) -> int:
        return len(self.index_map)


class RLDBZarrDataset(torch.utils.data.Dataset):
    """
    Drop-in replacement for RLDBDataset using Zarr backend.

    Provides train/valid splitting, sampling modes, and compatibility
    with existing DataSchematic and training pipeline.
    """

    def __init__(
        self,
        zarr_root: Union[str, Path],
        mode: str = "train",
        valid_ratio: float = 0.2,
        percent: float = 0.1,
        chunk_size: int = 100,
        use_task_string: bool = False,
        task_string: str = "",
        **kwargs,
    ):
        """
        Initialize RLDBZarrDataset.

        Args:
            zarr_root: Path to zarr store
            mode: One of "train", "valid", "sample", "percent", "total"
            valid_ratio: Fraction of episodes for validation
            percent: Fraction of frames to sample (for mode="percent")
            chunk_size: Action chunk size
            use_task_string: Whether to add task string to items
            task_string: Task string to add
        """
        self.zarr_root = Path(zarr_root)
        self.mode = mode
        self.valid_ratio = valid_ratio
        self.percent = percent
        self.chunk_size = chunk_size
        self.use_task_string = use_task_string
        self.task_string = task_string

        # Load metadata
        self.meta = ZarrDatasetMetadata(zarr_root)
        self.meta._update_splits(valid_ratio)

        self.embodiment = get_embodiment_id(self.meta.robot_type)
        self.fps = self.meta.fps

        # Determine which episodes to load
        episodes = self._get_episode_indices()

        # Create underlying ZarrDataset
        self._zarr_dataset = ZarrDataset(
            zarr_root=zarr_root,
            chunk_size=chunk_size,
            episodes=episodes,
            fps=self.fps,
        )

        # Handle percent mode (sample frames)
        self.sampled_indices = None
        if mode == "percent" and 0 < percent < 1:
            total_frames = len(self._zarr_dataset)
            num_sampled = max(1, int(percent * total_frames))
            self.sampled_indices = sorted(random.sample(range(total_frames), num_sampled))

        logger.info(
            f"RLDBZarrDataset: mode={mode}, {len(self)} frames, "
            f"embodiment={get_embodiment(self.embodiment)}"
        )

    def _get_episode_indices(self) -> Optional[List[int]]:
        """Get episode indices based on mode."""
        if self.mode == "total":
            return None  # All episodes

        splits = self.meta.info.get("splits", {})

        if self.mode == "train":
            return splits.get("train")
        elif self.mode == "valid":
            return splits.get("valid")
        elif self.mode in ("sample", "percent"):
            # Sample/percent modes use train split to avoid data leakage
            return splits.get("train")
        else:
            return None

    def __len__(self) -> int:
        if self.sampled_indices is not None:
            return len(self.sampled_indices)
        return len(self._zarr_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.sampled_indices is not None:
            idx = self.sampled_indices[idx]

        item = self._zarr_dataset[idx]

        if self.use_task_string:
            item["high_level_language_prompt"] = self.task_string

        return item

    @property
    def hf_dataset(self):
        """Compatibility shim for DataSchematic.infer_norm_from_dataset()."""
        return ZarrHFShim(self._zarr_dataset)

    @property
    def num_episodes(self) -> int:
        return self._zarr_dataset.num_episodes

    @property
    def num_frames(self) -> int:
        return len(self)


class ZarrHFShim:
    """
    Shim to make ZarrDataset compatible with DataSchematic normalization.

    DataSchematic calls:
        dataset.hf_dataset.with_format("numpy", columns=[col_name])[:][col_name]

    This shim emulates that interface. Note: Only [:] slicing is supported,
    not individual item indexing (dataset[i]).
    """

    def __init__(self, zarr_dataset: ZarrDataset):
        self._zarr_dataset = zarr_dataset
        self._format = None
        self._columns = None

    def with_format(self, format_type: str, columns: List[str] = None):
        """Set format and columns, return self for chaining."""
        self._format = format_type
        self._columns = columns
        return self

    def __getitem__(self, key):
        """Support [:] indexing to load all data."""
        if key == slice(None):
            return self._load_all_columns()
        raise NotImplementedError(f"ZarrHFShim only supports [:] indexing, got {key}")

    def _load_all_columns(self) -> Dict[str, np.ndarray]:
        """Load all requested columns as numpy arrays."""
        if self._columns is None:
            return {}

        result = {}
        for col_name in self._columns:
            zarr_key = LEROBOT_TO_ZARR_KEYS.get(col_name, col_name)
            result[col_name] = self._load_column(zarr_key)

        return result

    def _load_column(self, zarr_key: str) -> np.ndarray:
        """Load a single column from all episodes."""
        all_data = []

        for ep_hash in self._zarr_dataset.episode_hashes:
            ep = self._zarr_dataset._get_episode_group(ep_hash)

            try:
                # Navigate to the array
                parts = zarr_key.split("/")
                arr = ep
                for part in parts:
                    arr = arr[part]

                data = arr[:]
                all_data.append(data)
            except KeyError:
                logger.warning(f"Key {zarr_key} not found in episode {ep_hash}")
                continue

        if not all_data:
            raise KeyError(f"No data found for key {zarr_key}")

        return np.concatenate(all_data, axis=0)


class MultiRLDBZarrDataset(torch.utils.data.Dataset):
    """
    Combines multiple RLDBZarrDataset instances into a single dataset.

    Compatible with MultiRLDBDataset from rldb/utils.py.
    """

    def __init__(
        self,
        datasets: Dict[str, RLDBZarrDataset],
        embodiment: str,
        key_map: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """
        Initialize MultiRLDBZarrDataset.

        Args:
            datasets: Dict mapping dataset names to RLDBZarrDataset instances
            embodiment: Expected embodiment (e.g., "MECKA_BIMANUAL")
            key_map: Optional per-dataset key remapping
        """
        self.datasets = datasets
        self.key_map = key_map

        self.embodiment = get_embodiment_id(embodiment)

        # Validate all datasets have same embodiment
        for name, ds in self.datasets.items():
            if ds.embodiment != self.embodiment:
                raise ValueError(
                    f"Dataset {name} has embodiment {ds.embodiment}, "
                    f"expected {self.embodiment}"
                )

        # Build index map
        self.index_map = []
        for name, ds in self.datasets.items():
            for local_idx in range(len(ds)):
                self.index_map.append((name, local_idx))

        logger.info(
            f"MultiRLDBZarrDataset: {len(self.datasets)} datasets, "
            f"{len(self)} total frames"
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_name, local_idx = self.index_map[idx]
        item = self.datasets[dataset_name][local_idx]

        # Apply key remapping if specified
        if self.key_map and dataset_name in self.key_map:
            key_remap = self.key_map[dataset_name]
            item = {key_remap.get(k, k): v for k, v in item.items()}

        return item

    @property
    def hf_dataset(self):
        """Compatibility shim for DataSchematic."""
        # Return first dataset's hf_dataset (for normalization)
        first_ds = next(iter(self.datasets.values()))
        return first_ds.hf_dataset


class FolderRLDBZarrDataset(MultiRLDBZarrDataset):
    """
    Load multiple ZarrDatasets from a folder.

    Compatible with FolderRLDBDataset from rldb/utils.py.
    """

    def __init__(
        self,
        folder_path: Union[str, Path],
        embodiment: str,
        mode: str = "train",
        valid_ratio: float = 0.2,
        **kwargs,
    ):
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError(f"{folder_path} is not a valid directory")

        datasets = {}

        # Look for zarr stores in folder
        for subdir in sorted(folder_path.iterdir()):
            if subdir.is_dir():
                meta_path = subdir / "meta" / "info.json"
                if meta_path.exists() or (subdir / ".zattrs").exists():
                    try:
                        ds = RLDBZarrDataset(
                            zarr_root=subdir,
                            mode=mode,
                            valid_ratio=valid_ratio,
                            **kwargs,
                        )
                        if ds.embodiment == get_embodiment_id(embodiment):
                            datasets[subdir.name] = ds
                            logger.info(f"Loaded: {subdir.name}")
                        else:
                            logger.warning(
                                f"Skipping {subdir.name}: embodiment mismatch"
                            )
                    except Exception as e:
                        logger.error(f"Failed to load {subdir.name}: {e}")

        if not datasets:
            raise ValueError("No valid ZarrDatasets found in folder")

        super().__init__(datasets=datasets, embodiment=embodiment)
