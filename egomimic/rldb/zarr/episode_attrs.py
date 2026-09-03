"""Episode attribute vocabulary shared by the writer, the reader and the validator.

Keeping the vocabulary here rather than in any one of them means the writer
does not import the reader, and nobody restates a literal.
"""

from __future__ import annotations

from collections.abc import Mapping

#: A finished recording. Only these episodes get a database row and reach a
#: dataset.
DATA_STATUS_COMPLETE = "complete"

#: A sample sent to show the shape of a delivery. It is real data in the schema
#: sense and not real data in the training sense, and nothing downstream could
#: previously tell the two apart.
DATA_STATUS_STRUCTURAL_SAMPLE = "structural_sample"

DATA_STATUS_VALUES = (DATA_STATUS_COMPLETE, DATA_STATUS_STRUCTURAL_SAMPLE)


def data_status(attrs: Mapping) -> str:
    """Return one episode's data status.

    An episode written before the attribute existed is ``complete``: the
    corpus predates the distinction and every episode in it was delivered as
    finished data.

    Args:
        attrs: The episode's Zarr attributes.
    """
    value = attrs.get("data_status")
    return DATA_STATUS_COMPLETE if not value else str(value)


def is_complete(attrs: Mapping) -> bool:
    """Return whether one episode is a finished recording."""
    return data_status(attrs) == DATA_STATUS_COMPLETE


__all__ = [
    "DATA_STATUS_COMPLETE",
    "DATA_STATUS_STRUCTURAL_SAMPLE",
    "DATA_STATUS_VALUES",
    "data_status",
    "is_complete",
]
