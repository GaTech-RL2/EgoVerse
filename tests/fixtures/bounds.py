"""Stats-only MultiDataset shells for exercising ``_check_bounds``."""

from __future__ import annotations


def bounds_dataset(stats_by_key: dict, norm_mode: str = "quantile", embodiment=0):
    """A MultiDataset with only the fields ``_check_bounds`` reads: ``stats_by_key``
    maps each zarr key (also its key name) to its norm stats."""
    from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset

    md = MultiDataset.__new__(MultiDataset)
    md.norm_mode = norm_mode
    md.norm_stats = {embodiment: dict(stats_by_key)}
    md.zarr_keys = {embodiment: {key: key for key in stats_by_key}}
    md._warned_violations = set()
    return md
