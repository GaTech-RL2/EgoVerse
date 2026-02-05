"""
ZarrEpisode: Efficient wrapper for Zarr episode stores.
"""

from pathlib import Path
from typing import Any

import numpy as np
import simplejpeg
import torch
import zarr
from torch.utils.data import Dataset


class ZarrEpisode:
    """
    Lightweight wrapper around a single Zarr episode store.

    Designed for efficient PyTorch DataLoader usage with direct store access.
    """

    __slots__ = (
        "_path",
        "_store",
        "metadata",
        "keys",
    )

    def __init__(self, path: str | Path):
        """
        Initialize ZarrEpisode wrapper.

        Args:
            path: Path to the .zarr episode directory
        """
        self._path = Path(path)
        self._store = zarr.open_group(str(self._path), mode='r')
        self.metadata = dict(self._store.attrs)
        # self.keys = self._collect_keys()
        self.keys = self.metadata["features"]

    def read(self, keys_with_ranges: dict[str, tuple[int, int | None]]) -> dict[str, np.ndarray]:
        """
        Read data for specified keys, each with their own index or range.

        Args:
            keys_with_ranges: Dictionary mapping keys to (start, end) tuples.
                - start: Starting frame index
                - end: Ending frame index (exclusive). If None, reads single frame at start.

        Returns:
            Dictionary mapping keys to numpy arrays

        Example:
            >>> episode.read({
            ...     "obs/image": (0, 10),      # Read frames 0-10
            ...     "actions": (5, 15),        # Read frames 5-15
            ...     "rewards": (20, None),     # Read single frame at index 20
            ... })
        """
        result = {}
        for key, (start, end) in keys_with_ranges.items():
            arr = self._store[key]
            if end is not None:
                data = arr[start:end]
            else:
                # Single frame read - use slicing to avoid 0D array issues with VariableLengthBytes
                # arr[start:start+1] gives us a 1D array, then [0] extracts the actual object
                data = arr[start:start+1][0]
            result[key] = data

        return result

    def _collect_keys(self) -> list[str]:
        """
        Collect all array keys from the store.

        Returns:
            List of array keys (flat structure with dot-separated names)
        """
        keys = []
        for name in self._store.array_keys():
            keys.append(name)
        return keys

    def __len__(self) -> int:
        """
        Get total number of frames in the episode.

        Returns:
            Number of frames
        """
        return self.metadata['total_frames']

    def __repr__(self) -> str:
        """String representation of the episode."""
        return f"ZarrEpisode(path={self._path}, frames={len(self)})"


class ZarrDataset(Dataset):
    """
    PyTorch Dataset wrapper for a single ZarrEpisode.

    Handles JPEG decoding with simplejpeg for efficient image loading.
    """

    __slots__ = (
        "_episode",
        "_keys",
        "_image_keys",
    )

    def __init__(self, episode_path: str | Path):
        """
        Initialize ZarrDataset for a single episode.

        Args:
            episode_path: Path to .zarr episode directory
        """
        self._episode = ZarrEpisode(episode_path)
        self._keys = self._episode.keys
        self._image_keys = self._detect_image_keys()

    def _detect_image_keys(self) -> set[str]:
        """
        Detect which keys contain JPEG-encoded image data from metadata.

        Returns:
            Set of keys containing JPEG data
        """
        features = self._episode.metadata.get("features", {})
        return {key for key, info in features.items() if info.get("dtype") == "jpeg"}

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Get episode metadata.

        Returns:
            Dictionary of metadata from the episode
        """
        return self._episode.metadata

    def __len__(self) -> int:
        """
        Get total number of frames in the episode.

        Returns:
            Frame count
        """
        return len(self._episode)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray | torch.Tensor]:
        """
        Get a single frame by index.

        Args:
            idx: Frame index

        Returns:
            Dictionary mapping keys to decoded numpy arrays or torch tensors
        """
        # Read raw data - create dict with (start, end) tuples for each key
        keys_with_ranges = {key: (idx, None) for key in self._keys}
        data = self._episode.read(keys_with_ranges)

        # Decode JPEG-encoded image data
        for key in self._image_keys:
            if key in data:
                jpeg_bytes = data[key]
                decoded = simplejpeg.decode_jpeg(jpeg_bytes, colorspace='RGB')
                data[key] = decoded

        return data
