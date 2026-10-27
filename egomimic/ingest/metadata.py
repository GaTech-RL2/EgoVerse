"""The metadata every ingested delivery carries, whatever its format.

MCAP carries it in its session-metadata record; EVA and Aria uploads carry it in
`<id>_metadata.json`. Each format maps its own keys onto these snake_case ones; the rules
are main's: the zarr required metadata (egomimic.rldb.zarr.required_metadata: task_name,
operator, lab, scene, objects; only scene and objects can be waived, through
`missing_metadata`) plus the recording keys below. Values come from the upload, never
from SQL. The vendor is never a metadata key: it is the bucket the upload came through.
"""

from __future__ import annotations

from typing import Any

from egomimic.rldb.zarr.required_metadata import (
    REQUIRED_METADATA_FIELDS,
    check_stored_metadata,
    is_placeholder,
    vendor_recording_id_error,
    waiver_errors,
)

MISSING_METADATA_KEY = "missing_metadata"
# Beyond the required metadata: what the recordings row needs from the upload.
RECORDING_KEYS = ("embodiment", "rig_name", "device_serial", "license")


def waived_fields(meta: dict[str, Any]) -> set[str]:
    """Fields the upload lists in missing_metadata (comma text or a list)."""
    raw = meta.get(MISSING_METADATA_KEY) or []
    names = raw.split(",") if isinstance(raw, str) else raw
    return {str(n).strip() for n in names if str(n).strip()}


def check_metadata(
    meta: dict[str, Any],
    embodiments,
    *,
    allow_missing: bool = True,
    hints: dict[str, str] | None = None,
    optional: frozenset[str] = frozenset(),
) -> tuple[list[str], list[str]]:
    """(problems, waiver warnings) for an upload's metadata in the shared keys.
    `embodiments` are the ones this format can convert; `hints` says, per recording key,
    where the format gets it, added to its problem; `optional` recording keys may be
    left out (but must be real values when given). Waived fields are warnings with
    `allow_missing` (ingest: staff decide at review), problems without."""
    hints = hints or {}
    waived = waived_fields(meta)
    problems = waiver_errors(meta, waived)
    required = {}
    for name in REQUIRED_METADATA_FIELDS:
        value = meta.get(name)
        if name in waived and not value:
            required[name] = None
        elif name in meta and value is not None:
            required[name] = value
    errors, warnings, _ = check_stored_metadata(required, allow_missing=allow_missing)
    problems += errors
    if err := vendor_recording_id_error(meta.get("vendor_recording_id")):
        problems.append(err)
    for key in RECORDING_KEYS:
        value = meta.get(key)
        if key in optional and value is None:
            continue
        if not isinstance(value, str) or not value.strip() or is_placeholder(value):
            hint = f" ({hints[key]})" if key in hints else ""
            problems.append(f"{key} is required, got {value!r}{hint}")
    if meta.get("embodiment") is not None and meta["embodiment"] not in embodiments:
        problems.append(
            f"embodiment {meta['embodiment']!r} is not one of {sorted(embodiments)}"
        )
    return problems, warnings
