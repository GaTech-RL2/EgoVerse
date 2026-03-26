"""
Zarr-based dataset implementations for EgoVerse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from egomimic.rldb.zarr.zarr_dataset_multi import (
    EpisodeResolver,
    LocalEpisodeResolver,
    MultiDataset,
    S3EpisodeResolver,
    ZarrDataset,
    ZarrEpisode,
)

if TYPE_CHECKING:
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter

__all__ = [
    "EpisodeResolver",
    "MultiDataset",
    "ZarrDataset",
    "ZarrEpisode",
    "ZarrWriter",
    "LocalEpisodeResolver",
    "S3EpisodeResolver",
]


def __getattr__(name: str) -> Any:
    # ZarrWriter needs zarr>=3.1 (Python>=3.11 per upstream wheels). Importing it
    # eagerly breaks rollout/training entrypoints that only need action transforms.
    if name == "ZarrWriter":
        from egomimic.rldb.zarr.zarr_writer import ZarrWriter as _ZarrWriter

        return _ZarrWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
