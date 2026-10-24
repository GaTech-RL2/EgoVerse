"""SharpaDex manifest records and the episode ID contract.

A SharpaDex release names an episode by where it sits::

    <release_root>/<task>/<season>/lerobot_v2.1/..., episode_index

Two IDs are derived from that triple:

- the **source ID** ``<task>/<season>/<episode_index>``, the manifest's
  ``recording_id``. It selects and locates source episodes, nothing else.
- the **canonical ID** ``vendor_recording_id(task, season, episode_index)``,
  ``<task>__<season>__<episode_index:06d>``. It is the episode's
  ``vendor_recording_id`` (one path component, CONTRIBUTING_DATA.md §6.6), and
  resume and ownership checks compare it.

Converted stores and sidecars hold both, as ``vendor_recording_id`` and
``source_episode_id``; ``ids_correspond`` checks that they agree. A task or
season containing ``/`` or ``__`` is rejected, so the canonical ID parses back
to one triple and two source IDs never share one.

The manifest itself is written by ``sharpadex_manifest``; this module only
reads it, with the stdlib, so the ingest driver can use it without the
converter's dependencies.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from egomimic.rldb.zarr.required_metadata import vendor_recording_id_error

#: Joins the parts of a canonical ID.
SEPARATOR = "__"
#: Digits of the zero-padded episode index in a canonical ID.
INDEX_DIGITS = 6
CLOCK_SYNTHESIZED = "synthesized_from_season_name_and_frame_index"
_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_INDEX = re.compile(r"\d{%d,}" % INDEX_DIGITS)


class ManifestError(ValueError):
    """A manifest is unreadable, or its records break the ID contract."""


@dataclass
class EpisodeRecord:
    """One release episode and what the ingestion does with it.

    Field names are the manifest's JSON keys; ``recording_id`` is the source ID.
    """

    recording_id: str
    task: str
    season: str
    episode_index: int
    rig: str | None
    length: int
    fps: int
    source_uid: str | None
    fingerprint: str
    tags: dict
    start_local: str | None = None
    start_source: str | None = None
    clock: str = CLOCK_SYNTHESIZED
    start_ns: int | None = None
    episode_hash: str | None = None
    hash_offset_ms: int = 0
    decision: str = "keep"
    reasons: list[str] = field(default_factory=list)
    duplicate_of: str | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def vendor_recording_id(self) -> str:
        """The canonical ID of this episode."""
        return vendor_recording_id(self.task, self.season, self.episode_index)


# ---------------------------------------------------------------------------
# ID contract
# ---------------------------------------------------------------------------


def part_error(name: str, value) -> str | None:
    """Why ``value`` cannot be the task or season part of an ID, or None."""
    if not isinstance(value, str) or not _PART.fullmatch(value):
        return (
            f"{name} {value!r} must be letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    if SEPARATOR in value:
        return f"{name} {value!r} contains {SEPARATOR!r}, the canonical ID separator"
    return None


def _check_parts(task: str, season: str, episode_index: int) -> None:
    problems = [
        p
        for p in (part_error("task", task), part_error("season", season))
        if p is not None
    ]
    if type(episode_index) is not int or episode_index < 0:
        problems.append(f"episode_index {episode_index!r} must be an int >= 0")
    if problems:
        raise ValueError("; ".join(problems))


def vendor_recording_id(task_folder: str, season: str, episode_index: int) -> str:
    """Return the canonical ID ``<task>__<season>__<episode_index:06d>``.

    Raises:
        ValueError: If a part could make the ID ambiguous or not a single
            path component.
    """
    _check_parts(task_folder, season, episode_index)
    vid = SEPARATOR.join((task_folder, season, f"{episode_index:0{INDEX_DIGITS}d}"))
    if (err := vendor_recording_id_error(vid)) is not None:
        raise ValueError(err)
    return vid


def parse_vendor_recording_id(vid: str) -> tuple[str, str, int]:
    """Return ``(task, season, episode_index)`` from a canonical ID.

    Raises:
        ValueError: If ``vid`` is not exactly what ``vendor_recording_id``
            returns for some triple.
    """
    parts = vid.split(SEPARATOR) if isinstance(vid, str) else []
    if len(parts) != 3 or not _INDEX.fullmatch(parts[2]):
        raise ValueError(
            f"{vid!r} is not a SharpaDex vendor_recording_id "
            f"(<task>{SEPARATOR}<season>{SEPARATOR}<index:06d>)"
        )
    triple = parts[0], parts[1], int(parts[2])
    if vendor_recording_id(*triple) != vid:
        raise ValueError(f"{vid!r} is not in canonical form")
    return triple


def source_episode_id(task_folder: str, season: str, episode_index: int) -> str:
    """Return the source ID ``<task>/<season>/<episode_index>``."""
    _check_parts(task_folder, season, episode_index)
    return f"{task_folder}/{season}/{episode_index}"


def parse_source_episode_id(sid: str) -> tuple[str, str, int]:
    """Return ``(task, season, episode_index)`` from a source ID.

    Raises:
        ValueError: If ``sid`` is not exactly what ``source_episode_id``
            returns for some triple.
    """
    parts = sid.split("/") if isinstance(sid, str) else []
    if len(parts) != 3 or not parts[2].isdigit():
        raise ValueError(
            f"{sid!r} is not a SharpaDex source id (<task>/<season>/<index>)"
        )
    triple = parts[0], parts[1], int(parts[2])
    if source_episode_id(*triple) != sid:
        raise ValueError(f"{sid!r} is not in canonical form")
    return triple


def canonical_id(sid: str) -> str:
    """Map a source ID to its canonical ID."""
    return vendor_recording_id(*parse_source_episode_id(sid))


def ids_correspond(vid, sid) -> bool:
    """Whether a stored ``vendor_recording_id`` and ``source_episode_id`` name
    the same episode."""
    try:
        return parse_vendor_recording_id(vid) == parse_source_episode_id(sid)
    except ValueError:
        return False


def id_problems(records: Iterable[EpisodeRecord]) -> list[str]:
    """Every ID-contract violation in ``records``; empty when all is well.

    Each record's ``recording_id`` must be the source ID of its own
    ``(task, season, episode_index)``, every canonical ID must be valid, and
    no two records may share a source or a canonical ID.
    """
    problems = []
    by_canonical: dict[str, list[str]] = defaultdict(list)
    for record in records:
        try:
            expected = source_episode_id(
                record.task, record.season, record.episode_index
            )
            vid = record.vendor_recording_id
        except ValueError as exc:
            problems.append(f"{record.recording_id}: {exc}")
            continue
        if record.recording_id != expected:
            problems.append(
                f"{record.recording_id}: recording_id does not match its "
                f"task/season/episode_index ({expected})"
            )
            continue
        by_canonical[vid].append(record.recording_id)
    for vid, sids in sorted(by_canonical.items()):
        if len(sids) > 1:
            problems.append(f"{vid} is the canonical id of {sorted(sids)}")
    return problems


# ---------------------------------------------------------------------------
# Manifest file
# ---------------------------------------------------------------------------


def read_manifest(path: Path) -> tuple[dict, dict[str, EpisodeRecord]]:
    """Return ``(header, {recording_id: record})`` from a manifest file.

    The first line is ``{"manifest": header}``; every other non-blank line is
    one ``EpisodeRecord``.

    Raises:
        ManifestError: If the file is not a manifest, a record is malformed,
            or the records break the ID contract (``id_problems``).
        OSError: If the file cannot be read.
    """
    lines = Path(path).read_text().splitlines()
    try:
        first = json.loads(lines[0]) if lines else None
    except json.JSONDecodeError:
        first = None
    if not isinstance(first, dict) or not isinstance(first.get("manifest"), dict):
        raise ManifestError(f"{path}: not a SharpaDex manifest")
    records: dict[str, EpisodeRecord] = {}
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            record = EpisodeRecord(**json.loads(line))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ManifestError(f"{path}:{number}: malformed record ({exc})") from exc
        if record.recording_id in records:
            raise ManifestError(f"{path}:{number}: {record.recording_id} is repeated")
        records[record.recording_id] = record
    if problems := id_problems(records.values()):
        more = f" and {len(problems) - 5} more" if len(problems) > 5 else ""
        raise ManifestError(
            f"{path}: {len(problems)} id problem(s): " + "; ".join(problems[:5]) + more
        )
    return first["manifest"], records


__all__ = [
    "EpisodeRecord",
    "ManifestError",
    "canonical_id",
    "id_problems",
    "ids_correspond",
    "parse_source_episode_id",
    "parse_vendor_recording_id",
    "read_manifest",
    "source_episode_id",
    "vendor_recording_id",
]
