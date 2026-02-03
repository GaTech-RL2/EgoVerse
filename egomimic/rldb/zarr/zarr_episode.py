"""
ZarrEpisode: Efficient wrapper for Zarr episode stores.
"""

from pathlib import Path
from typing import Any

import numpy as np
import zarr


class ZarrEpisode:
    """
    Lightweight wrapper around a single Zarr episode store.

    Designed for efficient PyTorch DataLoader usage with direct store access.
    """

    __slots__ = (
        "_path",
        "_store",
        "_metadata",
        "_keys",
    )

    def __init__(self, path: str | Path):
        """
        Initialize ZarrEpisode wrapper.

        Args:
            path: Path to the .zarr episode directory
        """
        self._path = Path(path)
        self._store = zarr.open_group(str(self._path), mode='r')
        self._metadata = dict(self._store.attrs)
        self._keys = self._collect_keys()

    def read(self, keys: list[str], start: int, end: int | None = None) -> dict[str, np.ndarray]:
        """
        Read data for specified keys at a given index or range.

        Args:
            keys: List of dot-separated array paths
            start: Starting frame index
            end: Ending frame index (exclusive). If None, reads single frame at start.

        Returns:
            Dictionary mapping keys to numpy arrays
        """
        result = {}
        for key in keys:
            arr = self._store[key]
            data = arr[start:end] if end is not None else arr[start]
            result[key] = data

        return result

    def _collect_keys(self) -> list[str]:
        """
        Collect all array keys from the store.

        Returns:
            List of array keys (flat structure with dot-separated names)
        """
        return [name for name, item in self._store.items() if isinstance(item, zarr.Array)]

    def keys(self) -> list[str]:
        """
        Get all available array keys in the episode.

        Returns:
            List of dot-separated paths to all arrays
        """
        return self._keys

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Get episode metadata from zarr attributes.

        Returns:
            Dictionary of metadata (fps, robot_type, total_frames, features, etc.)
        """
        return self._metadata

    def __len__(self) -> int:
        """
        Get total number of frames in the episode.

        Returns:
            Number of frames
        """
        return self.metadata['total_frames']
