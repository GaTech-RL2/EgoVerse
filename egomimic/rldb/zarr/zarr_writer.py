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

    Example:
        >>> import numpy as np
        >>> from egomimic.rldb.zarr import ZarrWriter
        >>>
        >>> # Create test data
        >>> numeric_data = {
        ...     "observations.state": np.random.randn(100, 10),
        ...     "actions": np.random.randn(100, 7),
        ... }
        >>> image_data = {
        ...     "observations.images.cam1": np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8),
        ... }
        >>>
        >>> # Write episode
        >>> writer = ZarrWriter(
        ...     root_path="/tmp/episodes",
        ...     episode_index=0,
        ...     total_frames=100,
        ...     fps=30,
        ... )
        >>> writer.write(numeric_data=numeric_data, image_data=image_data)
        >>>
        >>> # One-shot convenience method
        >>> all_data = {**numeric_data, **image_data}
        >>> path = ZarrWriter.create_and_write(
        ...     data=all_data,
        ...     root_path="/tmp/episodes",
        ...     episode_index=1,
        ...     auto_detect_images=True,
        ... )
    """

    def __init__(
        self,
        episode_path: str | Path | None = None,
        root_path: str | Path | None = None,
        episode_index: int | None = None,
        total_frames: int | None = None,
        fps: int = 30,
        robot_type: str = "",
        task: str = "",
        jpeg_quality: int = 95,
        chunk_size_mb: float = 2.0,
        chunk_timesteps: int | None = None,
        enable_sharding: bool = True,
        auto_pad_for_sharding: bool = True,
        overwrite: bool = False,
    ):
        """
        Initialize ZarrWriter.

        Args:
            episode_path: Explicit path to episode .zarr directory.
                If provided, root_path and episode_index are ignored.
            root_path: Root directory for episodes. Episode will be created as
                {root_path}/episode_{episode_index:06d}.zarr
            episode_index: Episode index number (required if using root_path).
            total_frames: Total number of frames. If None, will be inferred from data.
            fps: Frames per second for playback (default: 30).
            robot_type: Robot type identifier (e.g., "eva_bimanual").
            task: Task description.
            jpeg_quality: JPEG quality 0-100 for image compression (default: 95).
            chunk_size_mb: Target chunk size in MB for numeric arrays (default: 2.0).
                Ignored if chunk_timesteps is specified.
            chunk_timesteps: Explicit number of timesteps per chunk.
                Overrides chunk_size_mb if provided.
            enable_sharding: Enable Zarr sharding for better cloud performance (default: True).
            auto_pad_for_sharding: Automatically pad arrays to be divisible by chunk size (default: True).
            overwrite: If True, overwrite existing episode. If False, raise error if exists (default: False).

        Raises:
            ValueError: If neither episode_path nor (root_path, episode_index) is provided.
            FileExistsError: If episode exists and overwrite=False.
        """
        # Determine episode path
        if episode_path is not None:
            self.episode_path = Path(episode_path)
        elif root_path is not None and episode_index is not None:
            self.episode_path = Path(root_path) / f"episode_{episode_index:06d}.zarr"
        else:
            raise ValueError("Must provide either episode_path or (root_path, episode_index)")

        # Check for existing episode
        if self.episode_path.exists() and not overwrite:
            raise FileExistsError(
                f"Episode already exists at {self.episode_path}. "
                "Set overwrite=True to replace it."
            )

        # Store parameters
        self.episode_index = episode_index
        self.total_frames = total_frames
        self.fps = fps
        self.robot_type = robot_type
        self.task = task
        self.jpeg_quality = jpeg_quality
        self.chunk_size_mb = chunk_size_mb
        self.chunk_size_bytes = int(chunk_size_mb * 1024 * 1024)
        self.chunk_timesteps = chunk_timesteps
        self.enable_sharding = enable_sharding
        self.auto_pad_for_sharding = auto_pad_for_sharding

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

        # Create parent directory
        self.episode_path.parent.mkdir(parents=True, exist_ok=True)

        # Open Zarr v3 store
        mode = "w" if self.episode_path.exists() else "w"
        store = zarr.open(str(self.episode_path), mode=mode, zarr_format=3)

        # Write numeric arrays
        for key, arr in numeric_data.items():
            self._write_numeric_array(store, key, arr)

        # Write image arrays
        for key, arr in image_data.items():
            self._write_image_array(store, key, arr)

        # Build and attach metadata
        metadata = self._build_metadata(metadata_override)
        store.attrs.update(metadata)

        # Consolidate metadata
        zarr.consolidate_metadata(str(self.episode_path))

    def _write_numeric_array(self, store: zarr.Group, key: str, arr: np.ndarray) -> None:
        """
        Write a numeric array to the Zarr store.

        Args:
            store: Zarr group to write to.
            key: Array key name.
            arr: Numeric array with shape (T, ...).
        """
        num_frames = len(arr)

        # Calculate frames per chunk
        if self.chunk_timesteps is not None:
            frames_per_chunk = self.chunk_timesteps
        else:
            bytes_per_frame = arr[0].nbytes
            if bytes_per_frame > 0:
                frames_per_chunk = max(1, self.chunk_size_bytes // bytes_per_frame)
            else:
                frames_per_chunk = num_frames

        # Cap to episode length
        frames_per_chunk = min(frames_per_chunk, num_frames)
        frames_per_chunk = max(1, frames_per_chunk)

        # Pad array for shard alignment if needed
        if self.enable_sharding and self.auto_pad_for_sharding:
            if num_frames % frames_per_chunk != 0:
                padded_frames = ((num_frames + frames_per_chunk - 1) // frames_per_chunk) * frames_per_chunk
                pad_len = padded_frames - num_frames
                pad_shape = (pad_len,) + arr.shape[1:]
                arr = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)

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

    def _write_image_array(self, store: zarr.Group, key: str, image_arr: np.ndarray) -> None:
        """
        Write an image array to the Zarr store with JPEG compression.

        Args:
            store: Zarr group to write to.
            key: Array key name.
            image_arr: Image array with shape (T, H, W, 3).
        """
        # Validate shape
        if image_arr.ndim != 4 or image_arr.shape[-1] != 3:
            raise ValueError(
                f"Image array '{key}' must have shape (T, H, W, 3), got {image_arr.shape}"
            )

        # Encode each frame as JPEG
        num_frames = len(image_arr)
        encoded = np.empty((num_frames,), dtype=object)
        for i in range(num_frames):
            img = image_arr[i]
            jpeg_bytes = simplejpeg.encode_jpeg(img, quality=self.jpeg_quality, colorspace='RGB')
            encoded[i] = jpeg_bytes

        # Chunk per image for efficient random access
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

    def _build_metadata(self, metadata_override: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Build episode metadata dictionary.

        Args:
            metadata_override: Optional overrides to apply.

        Returns:
            Metadata dictionary.
        """
        metadata = {
            "episode_index": self.episode_index,
            "fps": self.fps,
            "robot_type": self.robot_type,
            "total_frames": self.total_frames,
            "task": self.task,
            "features": self._features,
        }

        # Apply overrides
        if metadata_override:
            metadata.update(metadata_override)

        return metadata

    def write_incremental(self, key: str, data: np.ndarray, is_image: bool) -> None:
        """
        Write data incrementally (for building episodes step-by-step).

        This method allows building episodes by adding arrays one at a time.
        You must call finalize() after all incremental writes.

        Args:
            key: Array key name.
            data: Array data.
            is_image: If True, treat as image and apply JPEG compression.

        Example:
            >>> writer = ZarrWriter(episode_path="/tmp/ep.zarr", total_frames=30)
            >>> writer.write_incremental("state", np.random.randn(30, 10), is_image=False)
            >>> writer.write_incremental("image", np.random.randint(0, 255, (30, 480, 640, 3), dtype=np.uint8), is_image=True)
            >>> writer.finalize()
        """
        # Create parent directory
        self.episode_path.parent.mkdir(parents=True, exist_ok=True)

        # Open store (append mode if exists, otherwise create)
        mode = "a" if self.episode_path.exists() else "w"
        store = zarr.open(str(self.episode_path), mode=mode, zarr_format=3)

        # Write array
        if is_image:
            self._write_image_array(store, key, data)
        else:
            self._write_numeric_array(store, key, data)

    def finalize(self, metadata_override: dict[str, Any] | None = None) -> None:
        """
        Finalize incremental writing by adding metadata and consolidating.

        Must be called after all write_incremental() calls.

        Args:
            metadata_override: Optional metadata overrides.
        """
        # Open store
        store = zarr.open(str(self.episode_path), mode="a", zarr_format=3)

        # Build and attach metadata
        metadata = self._build_metadata(metadata_override)
        store.attrs.update(metadata)

        # Consolidate metadata
        zarr.consolidate_metadata(str(self.episode_path))

    @staticmethod
    def detect_image_keys(data: dict[str, np.ndarray]) -> set[str]:
        """
        Auto-detect which keys contain image data.

        Image arrays are detected as 4D arrays with shape (T, H, W, 3).

        Args:
            data: Dictionary of arrays.

        Returns:
            Set of keys that appear to contain images.

        Example:
            >>> data = {
            ...     "state": np.random.randn(100, 10),
            ...     "image": np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8),
            ... }
            >>> ZarrWriter.detect_image_keys(data)
            {'image'}
        """
        image_keys = set()
        for key, arr in data.items():
            if arr.ndim == 4 and arr.shape[-1] == 3:
                image_keys.add(key)
        return image_keys

    @staticmethod
    def create_and_write(
        data: dict[str, np.ndarray],
        episode_path: str | Path | None = None,
        root_path: str | Path | None = None,
        episode_index: int | None = None,
        auto_detect_images: bool = True,
        image_keys: set[str] | None = None,
        fps: int = 30,
        robot_type: str = "",
        task: str = "",
        jpeg_quality: int = 95,
        chunk_size_mb: float = 2.0,
        chunk_timesteps: int | None = None,
        enable_sharding: bool = True,
        auto_pad_for_sharding: bool = True,
        overwrite: bool = False,
        metadata_override: dict[str, Any] | None = None,
    ) -> Path:
        """
        Convenience method: create writer, separate data, and write in one call.

        Args:
            data: Combined dictionary of all arrays.
            episode_path: Explicit path to episode .zarr directory.
            root_path: Root directory for episodes.
            episode_index: Episode index number.
            auto_detect_images: If True, automatically detect image arrays (default: True).
                Ignored if image_keys is provided.
            image_keys: Explicit set of keys to treat as images.
                If provided, overrides auto_detect_images.
            fps: Frames per second (default: 30).
            robot_type: Robot type identifier.
            task: Task description.
            jpeg_quality: JPEG quality 0-100 (default: 95).
            chunk_size_mb: Target chunk size in MB (default: 2.0).
            chunk_timesteps: Explicit timesteps per chunk (overrides chunk_size_mb).
            enable_sharding: Enable Zarr sharding (default: True).
            auto_pad_for_sharding: Auto-pad for sharding (default: True).
            overwrite: Overwrite if exists (default: False).
            metadata_override: Optional metadata overrides.

        Returns:
            Path to created episode.

        Example:
            >>> data = {
            ...     "state": np.random.randn(100, 10),
            ...     "image": np.random.randint(0, 255, (100, 480, 640, 3), dtype=np.uint8),
            ... }
            >>> path = ZarrWriter.create_and_write(
            ...     data=data,
            ...     root_path="/tmp/episodes",
            ...     episode_index=0,
            ...     auto_detect_images=True,
            ... )
        """
        # Detect or use explicit image keys
        if image_keys is None and auto_detect_images:
            image_keys = ZarrWriter.detect_image_keys(data)
        elif image_keys is None:
            image_keys = set()

        # Separate data
        numeric_data = {k: v for k, v in data.items() if k not in image_keys}
        image_data = {k: v for k, v in data.items() if k in image_keys}

        # Create writer
        writer = ZarrWriter(
            episode_path=episode_path,
            root_path=root_path,
            episode_index=episode_index,
            fps=fps,
            robot_type=robot_type,
            task=task,
            jpeg_quality=jpeg_quality,
            chunk_size_mb=chunk_size_mb,
            chunk_timesteps=chunk_timesteps,
            enable_sharding=enable_sharding,
            auto_pad_for_sharding=auto_pad_for_sharding,
            overwrite=overwrite,
        )

        # Write data
        writer.write(
            numeric_data=numeric_data,
            image_data=image_data,
            metadata_override=metadata_override,
        )

        return writer.episode_path
