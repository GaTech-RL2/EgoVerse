"""Shared names and normalization helpers for episode attributes.

The writer and readers depend on this module without depending on each other.
"""

from __future__ import annotations

from collections.abc import Mapping

#: A finished recording eligible for staging and resolver-based loading.
DATA_STATUS_COMPLETE = "complete"

#: A delivery-format example that staging and dataset resolvers must exclude.
DATA_STATUS_STRUCTURAL_SAMPLE = "structural_sample"

DATA_STATUS_VALUES = (DATA_STATUS_COMPLETE, DATA_STATUS_STRUCTURAL_SAMPLE)


def data_status(attrs: Mapping) -> str:
    """Return the stored status, defaulting a missing or falsey value to complete.

    The default preserves the behavior of episodes written before
    ``data_status`` was introduced. This helper does not validate a non-empty
    stored value; the writer and schema validator do that separately.

    Args:
        attrs: The episode's Zarr attributes.
    """
    value = attrs.get("data_status")
    return DATA_STATUS_COMPLETE if not value else str(value)


def is_complete(attrs: Mapping) -> bool:
    """Return whether the normalized status is exactly ``complete``."""
    return data_status(attrs) == DATA_STATUS_COMPLETE


__all__ = [
    "DATA_STATUS_COMPLETE",
    "DATA_STATUS_STRUCTURAL_SAMPLE",
    "DATA_STATUS_VALUES",
    "data_status",
    "is_complete",
]
