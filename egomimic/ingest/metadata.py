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


def read_metadata_file(path) -> tuple[dict | None, str | None]:
    """(the upload's metadata, None) or (None, why it cannot be read)."""
    import json
    from pathlib import Path

    path = Path(path)
    try:
        md = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        return None, f"{path.name}: {e}"
    if not isinstance(md, dict):
        return None, f"{path.name}: not a JSON object"
    return md, None


def rig_metadata(
    md: dict[str, Any],
    vid: str,
    layouts: dict[str, str],
    default_license: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """A lab rig's metadata file (as add_raw_data_to_table reads them) in the shared
    keys, plus what the recording gives. The rigs write `task` for task_name, the device
    or robot as `embodiment` (aria, eva) and the arm layout as `robot_name` (sometimes
    "robot name", or spelled with spaces); `layouts` maps those layouts to embodiments.
    The device (or `device`, when the format implies one) stands in for a missing
    rig_name, the vendor's default for a missing license, a numeric scene ("scene": 2)
    is its text, and objects ["None"] (or []), how the rigs say "no objects", becomes a
    missing_metadata waiver for review. Files already in the shared keys pass through."""
    meta = dict(md)
    if "task_name" not in meta and "task" in meta:
        meta["task_name"] = meta.pop("task")
    written = meta.get("embodiment")
    if written not in layouts.values():
        layout = meta.pop("robot_name", None) or meta.pop("robot name", None)
        if isinstance(layout, str):
            layout = layouts.get(layout.strip().lower().replace(" ", "_"), layout)
        meta["embodiment"] = layout
        device = written or device
    if not meta.get("rig_name") and device:
        meta["rig_name"] = device
    scene = meta.get("scene")
    if isinstance(scene, (int, float)) and not isinstance(scene, bool):
        meta["scene"] = str(scene)
    if not meta.get("license") and default_license:
        meta["license"] = default_license
    objects = meta.get("objects")
    if isinstance(objects, list) and all(
        str(o).strip().lower() in ("", "none") for o in objects
    ):
        meta.pop("objects")
        meta[MISSING_METADATA_KEY] = sorted(waived_fields(meta) | {"objects"})
    meta["vendor_recording_id"] = vid
    meta["start_time_unix"] = vid
    return meta


def start_time_problem(vid: str, file: str) -> str | None:
    """Why `vid`, a recording's start in epoch ms named by `file`, is implausible."""
    from egomimic.rldb.zarr.episode_rules import start_time_error

    if err := start_time_error(int(vid) if vid.isdigit() else vid):
        return f"{file}: the file name is the recording's start; {err}"
    return None
