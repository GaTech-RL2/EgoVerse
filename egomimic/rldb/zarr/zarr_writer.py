"""
ZarrWriter: General-purpose Zarr episode writer.

This module provides a reusable writer for creating Zarr v3 episode stores
compatible with the ZarrEpisode reader.
"""

from pathlib import Path
from typing import Any

import numpy as np
import simplejpeg
import zarr
from zarr.core.dtype import VariableLengthBytes


class ZarrWriter:
    """
    General-purpose writer for Zarr v3 episode stores.

    Creates episodes compatible with the ZarrEpisode reader, handling both
    numeric and image data with intelligent chunking and optional sharding.
    """

    JPEG_QUALITY = 85  # Fixed JPEG quality for image compression

    def __init__(
        self,
        episode_path: str | Path,
        robot_type: str = "",
        total_frames: int | None = None,
        fps: int = 30,
        task: str = "",
        language_annotations: list[tuple[str, int, int]] | None = None,
        chunk_timesteps: int = 100,
        enable_sharding: bool = True,
    ):
        """
        Initialize ZarrWriter.

        Args:
            episode_path: Path to episode .zarr directory.
            robot_type: Robot type identifier (e.g., "eva_bimanual").
            total_frames: Total number of frames. If None, will be inferred from data.
            fps: Frames per second for playback (default: 30).
            task: Task description.
            language_annotations: List of (text, start_idx, end_idx) tuples describing language annotations.
            chunk_timesteps: Number of timesteps per chunk for numeric arrays (default: 100).
            enable_sharding: Enable Zarr sharding for better cloud performance (default: True).
        """
        self.episode_path = Path(episode_path)

        # Store parameters
        self.total_frames = total_frames
        self.fps = fps
        self.robot_type = robot_type
        self.task = task
        self.language_annotations = language_annotations or []
        self.chunk_timesteps = chunk_timesteps
        self.enable_sharding = enable_sharding

        # Track image shapes for metadata
        self._features: dict[str, dict[str, Any]] = {}

    def write(
        self,
        numeric_data: dict[str, np.ndarray] | None = None,
        image_data: dict[str, np.ndarray] | None = None,
        metadata_override: dict[str, Any] | None = None,
    ) -> None:
        """
        Write episode data to Zarr store.

        Args:
            numeric_data: Dictionary of numeric arrays (state, actions, etc.).
                All arrays must have same length along axis 0.
            image_data: Dictionary of image arrays with shape (T, H, W, 3).
                Images will be JPEG-compressed.
            metadata_override: Optional metadata overrides to apply after building metadata.

        Raises:
            ValueError: If arrays have inconsistent frame counts.
            ValueError: If total_frames was not set and cannot be inferred.
        """
        numeric_data = numeric_data or {}
        image_data = image_data or {}

        if not numeric_data and not image_data:
            raise ValueError("Must provide at least one of numeric_data or image_data")

        # Validate and infer total_frames
        all_lengths = []
        for key, arr in {**numeric_data, **image_data}.items():
            all_lengths.append(len(arr))

        if len(set(all_lengths)) > 1:
            raise ValueError(
                f"Inconsistent frame counts across arrays: {dict(zip(numeric_data.keys() | image_data.keys(), all_lengths))}"
            )

        inferred_frames = all_lengths[0] if all_lengths else 0
        if self.total_frames is None:
            if inferred_frames == 0:
                raise ValueError("total_frames not set and cannot be inferred from empty data")
            self.total_frames = inferred_frames
        elif self.total_frames != inferred_frames:
            raise ValueError(
                f"total_frames={self.total_frames} but data has {inferred_frames} frames"
            )

        # Calculate padded frame count if sharding is enabled
        padded_frames = self.total_frames
        if self.enable_sharding and self.total_frames % self.chunk_timesteps != 0:
            padded_frames = ((self.total_frames + self.chunk_timesteps - 1) // self.chunk_timesteps) * self.chunk_timesteps

        # Create parent directory
        self.episode_path.parent.mkdir(parents=True, exist_ok=True)

        # Open Zarr v3 store
        mode = "w" if self.episode_path.exists() else "w"
        store = zarr.open(str(self.episode_path), mode=mode, zarr_format=3)

        # Write numeric arrays
        for key, arr in numeric_data.items():
            self._write_numeric_array(store, key, arr, padded_frames)

        # Write image arrays
        for key, arr in image_data.items():
            self._write_image_array(store, key, arr, padded_frames)

        # Write language annotations if provided
        if self.language_annotations:
            self._write_language_annotations(store, self.language_annotations)

        # Build and attach metadata
        metadata = self._build_metadata(metadata_override)
        store.attrs.update(metadata)

    def _write_numeric_array(self, store: zarr.Group, key: str, arr: np.ndarray, padded_frames: int) -> None:
        """
        Write a numeric array to the Zarr store.

        Args:
            store: Zarr group to write to.
            key: Array key name.
            arr: Numeric array with shape (T, ...).
            padded_frames: Target frame count after padding (for sharding alignment).
        """
        num_frames = len(arr)

        # Store original shape and dtype before padding for metadata
        original_shape = arr.shape[1:]  # Shape excluding time dimension
        dtype_str = str(arr.dtype)

        # Pad array if needed
        if padded_frames > num_frames:
            pad_len = padded_frames - num_frames
            pad_shape = (pad_len,) + arr.shape[1:]
            arr = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)

        # Use chunk_timesteps for frames per chunk
        frames_per_chunk = min(self.chunk_timesteps, padded_frames)
        frames_per_chunk = max(1, frames_per_chunk)

        # Chunk shape: (frames, ...) - keep other dimensions intact
        chunk_shape = (frames_per_chunk,) + arr.shape[1:]

        # Create array with or without sharding
        if self.enable_sharding:
            shard_shape = arr.shape
            store.create_array(
                key,
                data=arr,
                chunks=chunk_shape,
                shards=shard_shape,
            )
        else:
            store.create_array(
                key,
                data=arr,
                chunks=chunk_shape,
            )

        # Track shape and dtype for metadata
        dimension_names = [f"dim_{i}" for i in range(len(original_shape))]
        self._features[key] = {
            "dtype": dtype_str,
            "shape": list(original_shape),
            "names": dimension_names,
        }

    def _write_image_array(self, store: zarr.Group, key: str, image_arr: np.ndarray, padded_frames: int) -> None:
        """
        Write an image array to the Zarr store with JPEG compression.

        Images are always chunked 1 per timestep for efficient random access,
        regardless of chunk_timesteps setting.

        Args:
            store: Zarr group to write to.
            key: Array key name.
            image_arr: Image array with shape (T, H, W, 3).
            padded_frames: Target frame count after padding (for sharding alignment).
        """
        # Validate shape
        if image_arr.ndim != 4 or image_arr.shape[-1] != 3:
            raise ValueError(
                f"Image array '{key}' must have shape (T, H, W, 3), got {image_arr.shape}"
            )

        # Encode each frame as JPEG
        num_frames = len(image_arr)

        # Encode to padded_frames length (pad with duplicate of last frame if needed)
        encoded = np.empty((padded_frames,), dtype=object)
        for i in range(padded_frames):
            # Use last frame for padding
            frame_idx = min(i, num_frames - 1)
            img = image_arr[frame_idx]
            jpeg_bytes = simplejpeg.encode_jpeg(img, quality=self.JPEG_QUALITY, colorspace='RGB')
            encoded[i] = jpeg_bytes

        # Images are always chunked 1 per timestep, regardless of chunk_timesteps
        chunk_shape = (1,)

        # Create array with VariableLengthBytes dtype
        if self.enable_sharding:
            shard_shape = encoded.shape
            store.create_array(
                key,
                shape=encoded.shape,
                chunks=chunk_shape,
                shards=shard_shape,
                dtype=VariableLengthBytes(),
            )
        else:
            store.create_array(
                key,
                shape=encoded.shape,
                chunks=chunk_shape,
                dtype=VariableLengthBytes(),
            )

        # Assign data after creation (required for VariableLengthBytes)
        store[key][:] = encoded

        # Track shape for metadata
        self._features[key] = {
            "dtype": "jpeg",
            "shape": list(image_arr.shape[1:]),  # [H, W, 3]
            "names": ["height", "width", "channel"],
        }

    def _write_language_annotations(
        self, store: zarr.Group, annotations: list[tuple[str, int, int]]
    ) -> None:
        """
        Write language annotations as a structured Zarr array.

        Args:
            store: Zarr group to write to.
            annotations: List of (text, start_idx, end_idx) tuples.
        """
        # Convert to structured numpy array
        dtype = [("text", object), ("start_idx", np.int32), ("end_idx", np.int32)]
        arr = np.array(annotations, dtype=dtype)

        # Create Zarr array with VariableLengthBytes for text field
        zarr_dtype = np.dtype(
            [("text", VariableLengthBytes()), ("start_idx", np.int32), ("end_idx", np.int32)]
        )

        # Store as a single structured array
        store.create_array(
            "language_annotations",
            shape=arr.shape,
            dtype=zarr_dtype,
        )
        store["language_annotations"][:] = arr

        # Track in features
        self._features["language_annotations"] = {
            "dtype": "structured",
            "shape": [len(annotations)],
            "names": ["text", "start_idx", "end_idx"],
        }

    def _build_metadata(self, metadata_override: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Build episode metadata dictionary.

        Args:
            metadata_override: Optional overrides to apply.

        Returns:
            Metadata dictionary.
        """
        metadata = {
            "robot_type": self.robot_type,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "task": self.task,
            "features": self._features,
        }

        # Apply overrides
        if metadata_override:
            metadata.update(metadata_override)

        return metadata

    @staticmethod
    def create_and_write(
        episode_path: str | Path,
        numeric_data: dict[str, np.ndarray] | None = None,
        image_data: dict[str, np.ndarray] | None = None,
        robot_type: str = "",
        fps: int = 30,
        task: str = "",
        language_annotations: list[tuple[str, int, int]] | None = None,
        chunk_timesteps: int = 100,
        enable_sharding: bool = True,
        metadata_override: dict[str, Any] | None = None,
    ) -> Path:
        """
        Convenience method: create writer and write in one call.

        Args:
            episode_path: Path to episode .zarr directory.
            numeric_data: Dictionary of numeric arrays (state, actions, etc.).
            image_data: Dictionary of image arrays with shape (T, H, W, 3).
            robot_type: Robot type identifier.
            fps: Frames per second (default: 30).
            task: Task description.
            language_annotations: List of (text, start_idx, end_idx) tuples describing language annotations.
            chunk_timesteps: Number of timesteps per chunk for numeric arrays (default: 100).
            enable_sharding: Enable Zarr sharding (default: True).
            metadata_override: Optional metadata overrides.

        Returns:
            Path to created episode.

        Raises:
            ValueError: If neither numeric_data nor image_data are provided.
        """
        # Validate we have at least some data
        if not numeric_data and not image_data:
            raise ValueError(
                "Must provide at least one of: 'numeric_data' or 'image_data'"
            )

        # Create writer
        writer = ZarrWriter(
            episode_path=episode_path,
            robot_type=robot_type,
            fps=fps,
            task=task,
            language_annotations=language_annotations,
            chunk_timesteps=chunk_timesteps,
            enable_sharding=enable_sharding,
        )

        # Write data
        writer.write(
            numeric_data=numeric_data,
            image_data=image_data,
            metadata_override=metadata_override,
        )

        return writer.episode_path
