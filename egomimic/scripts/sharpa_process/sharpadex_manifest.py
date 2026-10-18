"""Plan a SharpaDex release ingestion: deduplicate episodes and name them.

SharpaDex releases nest one LeRobot dataset per recording session under its
task folder::

    <release_root>/<task>/season_POC<rig>_<YYYY_MM_DD_HH_MM_SS>_train/
        lerobot_v2.1/meta/{info,episodes,episodes_stats,annotations}.jsonl ...

This tool reads metadata only. For every episode it records:

- the recording start, from the per-episode name Sharpa stores in
  ``annotations.jsonl`` (``episode_id``, the ``anno_path`` folder or the
  ``source_hdf5`` name; one-second resolution, site-local time), falling back
  to the season start plus the episode's frame offset;
- a content fingerprint from ``episodes_stats.jsonl``;
- a decision: ``keep``, ``drop_duplicate`` (an identical copy is kept),
  ``hold_uid_collision`` (one ``source_uid`` names different data),
  ``hold_gross_excursion`` (observed hand joints far outside the registry
  limits) or ``exclude`` (listed in ``--exclude-file``);
- for kept episodes, the EgoVerse episode hash, unique within the release and
  against ``--existing-hashes`` (and the SQL episode table with
  ``--check-sql``). A clash moves the start by whole milliseconds, recorded as
  ``hash_offset_ms``.

The converter reads the manifest with ``--manifest``. A typical run::

    python -m egomimic.scripts.sharpa_process.sharpadex_manifest \\
        --release-root /data/SharpaDex-v1.0 --output manifest.jsonl \\
        --exclude-file sharpadex_v1_exclusions.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from egomimic.scripts.sharpa_process.lerobot_reader import parse_season_name
from egomimic.scripts.sharpa_process.sharpa_to_zarr import (
    episode_hash,
    parse_utc_offset,
    utc_ns,
)

logger = logging.getLogger(__name__)

#: Season subdirectory spellings, in preference order.
V21_DIRS = ("lerobot_v2.1", "lerobotv2.1")
CLOCK_VENDOR = "vendor_episode_start+frame_index"
CLOCK_SYNTHESIZED = "synthesized_from_season_name_and_frame_index"
_EPISODE_STAMP = re.compile(r"(POC[0-9A-Za-z]+)_(\d{4}(?:_\d{2}){5})")
_HDF5_STAMP = re.compile(r"@(\d{4}(?:_\d{2}){5})")
#: Observed hand joints this far past a registry limit are flagged (5 deg).
FLAG_MARGIN = float(np.radians(5.0))
SIDES = {"left": 7, "right": 36}
HAND_DOF = 22
FINGERS = ("thumb", "index", "middle", "ring", "little")


class ManifestError(ValueError):
    """The release or an option prevents building a manifest."""


@dataclass
class EpisodeRecord:
    """One release episode and what the ingestion does with it."""

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


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def _meta_dir(season_dir: Path) -> Path | None:
    for name in V21_DIRS:
        meta = season_dir / name / "meta"
        if (meta / "info.json").is_file():
            return meta
    return None


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def episode_start(
    annotation: dict, rig: str | None
) -> tuple[datetime | None, str | None]:
    """Return the per-episode local start and the field it came from.

    Sharpa names each recording ``POC<rig>_<YYYY_MM_DD_HH_MM_SS>_train``. A
    name from another rig is ignored.
    """
    candidates = (
        ("episode_id", annotation.get("episode_id") or "", _EPISODE_STAMP),
        (
            "anno_path",
            os.path.basename(os.path.dirname(annotation.get("anno_path") or "")),
            _EPISODE_STAMP,
        ),
        (
            "source_hdf5",
            os.path.basename(annotation.get("source_hdf5") or ""),
            _HDF5_STAMP,
        ),
    )
    for source, text, pattern in candidates:
        match = pattern.search(text)
        if match is None:
            continue
        if pattern is _EPISODE_STAMP and rig is not None and match[1] != rig:
            continue
        return datetime.strptime(match[match.lastindex], "%Y_%m_%d_%H_%M_%S"), source
    return None, None


def content_fingerprint(stats: dict, length: int) -> str:
    """Hash an episode's length and joint statistics; equal data, equal hash."""
    payload = {"length": length}
    for name in ("observation.state", "action"):
        block = stats.get(name) or {}
        payload[name] = {k: block.get(k) for k in ("min", "max", "mean", "std")}
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _hand_limits() -> dict[str, np.ndarray]:
    from egomimic.rldb.embodiment.registry import load_end_effectors

    specs = load_end_effectors()
    return {
        side: np.asarray(specs[f"sharpa_wave_v1_{side}"].joint_limits, dtype=float)
        for side in SIDES
    }


def stats_flags(stats: dict, limits: dict[str, np.ndarray], gross: float) -> tuple:
    """Return ``(flags, gross_problems)`` derived from one episode's statistics."""
    flags, gross_problems = [], []
    state = stats.get("observation.state") or {}
    if state.get("min") and state.get("max"):
        low, high = np.asarray(state["min"], float), np.asarray(state["max"], float)
        for side, offset in SIDES.items():
            columns = slice(offset, offset + HAND_DOF)
            excess = np.maximum(
                limits[side][:, 0] - low[columns], high[columns] - limits[side][:, 1]
            )
            if excess.max() > FLAG_MARGIN:
                flags.append(f"obs_hand_excursion_over_5deg:{side}")
            if excess.max() > gross:
                gross_problems.append(
                    f"{side} hand joint {int(excess.argmax())} observed "
                    f"{excess.max():.3f} rad outside the registry limits"
                )
    tactile = stats.get("observation.tactile") or {}
    if tactile.get("min") and tactile.get("max"):
        span = np.asarray(tactile["max"], float) - np.asarray(tactile["min"], float)
        for s, side in enumerate(SIDES):
            for f, finger in enumerate(FINGERS):
                start = (s * len(FINGERS) + f) * 6
                if span[start : start + 6].max() < 1e-6:
                    flags.append(f"tactile_flat:{side}_{finger}")
    return flags, gross_problems


def scan_release(
    root: Path, *, gross_margin: float | None = None
) -> list[EpisodeRecord]:
    """Read every season's metadata under ``root``; one record per episode."""
    root = Path(root)
    if gross_margin is None:
        from egomimic.rldb.zarr.validate import limit_margins

        gross_margin = limit_margins()["joint_limit_gross"]
    limits = _hand_limits()
    records = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for season_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            meta = _meta_dir(season_dir)
            if meta is None:
                continue
            info = json.loads((meta / "info.json").read_text())
            fps = int(info["fps"])
            rig, _ = parse_season_name(season_dir.name)
            episodes = {
                int(r["episode_index"]): r for r in _jsonl(meta / "episodes.jsonl")
            }
            stats = {
                int(r["episode_index"]): r.get("stats", {})
                for r in _jsonl(meta / "episodes_stats.jsonl")
            }
            annotations = {
                int(r["episode_index"]): r
                for r in _jsonl(meta / "annotations.jsonl")
                if "episode_index" in r
            }
            for index in sorted(episodes):
                length = int(episodes[index]["length"])
                annotation = annotations.get(index, {})
                start, source = episode_start(annotation, rig)
                flags, gross = stats_flags(stats.get(index, {}), limits, gross_margin)
                tags = dict(annotation.get("tags") or {})
                if not tags.get("score"):
                    flags.append("no_score")
                record = EpisodeRecord(
                    recording_id=f"{task_dir.name}/{season_dir.name}/{index}",
                    task=task_dir.name,
                    season=season_dir.name,
                    episode_index=index,
                    rig=rig,
                    length=length,
                    fps=fps,
                    source_uid=annotation.get("source_uid"),
                    fingerprint=content_fingerprint(stats.get(index, {}), length),
                    tags=tags,
                    start_local=None if start is None else start.isoformat(),
                    start_source=source,
                    flags=flags,
                )
                if gross:
                    record.decision = "hold_gross_excursion"
                    record.reasons.extend(gross)
                records.append(record)
    if not records:
        raise ManifestError(f"{root}: no <task>/<season>/lerobot_v2.1 metadata found")
    return records


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


def _folder_match(record: EpisodeRecord) -> int:
    """Return how many words the task folder shares with Sharpa's task tag."""
    return len(_tokens(record.task) & _tokens(record.tags.get("task_name") or ""))


def dedup(records: list[EpisodeRecord]) -> None:
    """Mark duplicate copies and conflicting ``source_uid`` groups in place.

    Identical copies (same ``source_uid`` and fingerprint) keep one record:
    the copy whose task folder best matches Sharpa's ``task_name`` tag, then
    the earliest season, then the lowest episode index. A ``source_uid`` that
    names different data holds every record that uses it.
    """
    groups: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        if record.source_uid:
            groups[record.source_uid].append(record)
    for uid, group in groups.items():
        if len(group) < 2:
            continue
        if len({r.fingerprint for r in group}) > 1:
            ids = sorted(r.recording_id for r in group)
            for record in group:
                record.decision = "hold_uid_collision"
                record.reasons.append(f"source_uid {uid} also names {ids}")
            continue
        ranked = sorted(
            group,
            key=lambda r: (
                r.decision != "keep",
                -_folder_match(r),
                r.season,
                r.task,
                r.episode_index,
            ),
        )
        kept = ranked[0]
        for record in ranked[1:]:
            if record.decision == "keep":
                record.decision = "drop_duplicate"
            record.duplicate_of = kept.recording_id
            record.reasons.append(f"identical copy of {kept.recording_id}")


def load_exclusions(path: Path) -> dict[str, str]:
    """Read ``{recording_id: reason}`` from a YAML list of exclusions."""
    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"{path}: cannot read exclusions ({exc})") from exc
    out = {}
    for entry in data.get("exclude", []):
        try:
            key = f"{entry['task']}/{entry['season']}/{int(entry['episode_index'])}"
            out[key] = str(entry["reason"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"{path}: malformed exclusion {entry!r}") from exc
    return out


def apply_exclusions(records: list[EpisodeRecord], exclusions: dict[str, str]) -> None:
    known = {r.recording_id for r in records}
    unknown = sorted(set(exclusions) - known)
    if unknown:
        raise ManifestError(f"exclusions name unknown episodes: {unknown[:5]}")
    for record in records:
        if record.recording_id in exclusions:
            record.decision = "exclude"
            record.reasons.append(exclusions[record.recording_id])


# ---------------------------------------------------------------------------
# Clock and names
# ---------------------------------------------------------------------------


def _season_offsets(records: list[EpisodeRecord]) -> dict[str, int]:
    """Frames recorded in each season before each episode, by recording id."""
    by_season: dict[tuple[str, str], list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        by_season[(record.task, record.season)].append(record)
    offsets = {}
    for group in by_season.values():
        total = 0
        for record in sorted(group, key=lambda r: r.episode_index):
            offsets[record.recording_id] = total
            total += record.length
    return offsets


def assign_clocks(records: list[EpisodeRecord], zone) -> None:
    """Set each record's UTC start and clock source in place.

    A per-episode start must not precede the end of the season's previous
    episode by more than one second (the stamp's resolution); otherwise the
    season-synthesized clock is used and the record is flagged.
    """
    offsets = _season_offsets(records)
    by_season: dict[tuple[str, str], list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        by_season[(record.task, record.season)].append(record)
    for group in by_season.values():
        group.sort(key=lambda r: r.episode_index)
        previous_end = None
        for record in group:
            _, season_start = parse_season_name(record.season)
            vendor = None
            if record.start_local is not None:
                vendor = utc_ns(datetime.fromisoformat(record.start_local), zone)
                if previous_end is not None and vendor < previous_end - 10**9:
                    record.flags.append("start_before_previous_episode_end")
                    vendor = None
            if vendor is not None:
                record.start_ns, record.clock = vendor, CLOCK_VENDOR
            elif season_start is not None:
                offset = offsets[record.recording_id]
                record.start_ns = (
                    utc_ns(season_start, zone) + (offset * 10**9) // record.fps
                )
                record.clock = CLOCK_SYNTHESIZED
            else:
                record.decision = "exclude"
                record.reasons.append("no recording start in metadata or season name")
                continue
            previous_end = record.start_ns + (record.length * 10**9) // record.fps


def assign_hashes(records: list[EpisodeRecord], existing: set[str] = frozenset()):
    """Give every kept record a unique episode hash, in place.

    A hash already used by an earlier kept record, or listed in ``existing``,
    moves the start forward by one millisecond until the name is free.
    Records are visited in start order, so the result is deterministic.
    """
    taken = set(existing)
    kept = [r for r in records if r.decision == "keep" and r.start_ns is not None]
    kept.sort(
        key=lambda r: (r.start_ns, r.rig or "", r.task, r.season, r.episode_index)
    )
    for record in kept:
        name = episode_hash(record.start_ns)
        while name in taken:
            record.start_ns += 10**6
            record.hash_offset_ms += 1
            name = episode_hash(record.start_ns)
        record.episode_hash = name
        taken.add(name)


def sql_existing_hashes(candidates: list[str]) -> set[str]:
    """Return the candidates already present in ``app.episodes`` (read-only)."""
    from sqlalchemy import text

    from egomimic.utils.aws.aws_sql import create_default_engine

    engine = create_default_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT episode_hash FROM app.episodes "
                    "WHERE episode_hash = ANY(:hashes)"
                ),
                {"hashes": list(candidates)},
            ).all()
    finally:
        engine.dispose()
    return {row[0] for row in rows}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_manifest(path: Path, records: list[EpisodeRecord], header: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"manifest": header}, sort_keys=True) + "\n")
        for record in records:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_manifest(path: Path) -> tuple[dict, dict[str, EpisodeRecord]]:
    """Return ``(header, {recording_id: record})`` from a manifest file."""
    lines = Path(path).read_text().splitlines()
    if not lines or "manifest" not in json.loads(lines[0]):
        raise ManifestError(f"{path}: not a SharpaDex manifest")
    header = json.loads(lines[0])["manifest"]
    records = {}
    for line in lines[1:]:
        if line.strip():
            record = EpisodeRecord(**json.loads(line))
            records[record.recording_id] = record
    return header, records


def summarize(records: list[EpisodeRecord]) -> dict:
    kept = [r for r in records if r.decision == "keep"]
    flags = Counter(f.split(":")[0] for r in records for f in r.flags)
    return {
        "episodes": len(records),
        "decisions": dict(Counter(r.decision for r in records)),
        "kept_frames": sum(r.length for r in kept),
        "kept_clock": dict(Counter(r.clock for r in kept)),
        "kept_with_hash_offset": sum(1 for r in kept if r.hash_offset_ms),
        "flags": dict(flags),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m egomimic.scripts.sharpa_process.sharpadex_manifest",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-revision", default="", help="Release revision (e.g. the HF sha)."
    )
    parser.add_argument("--source-utc-offset", default="+08:00")
    parser.add_argument(
        "--exclude-file", type=Path, help="YAML `exclude:` list (task, season, ...)."
    )
    parser.add_argument(
        "--existing-hashes",
        type=Path,
        help="File of episode hashes already in use, one per line.",
    )
    parser.add_argument(
        "--check-sql",
        action="store_true",
        help="Also avoid hashes present in app.episodes (read-only query).",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        zone = parse_utc_offset(args.source_utc_offset)
        records = scan_release(args.release_root)
        if args.exclude_file:
            apply_exclusions(records, load_exclusions(args.exclude_file))
        dedup(records)
        assign_clocks(records, zone)
        existing = set()
        if args.existing_hashes:
            existing |= {
                line.strip()
                for line in args.existing_hashes.read_text().splitlines()
                if line.strip()
            }
        assign_hashes(records, existing)
        sql_checked = False
        if args.check_sql:
            # Re-plan the clashing names until none is in the episode table.
            for _ in range(10):
                planned = {r.episode_hash for r in records if r.episode_hash}
                clash = sql_existing_hashes(sorted(planned))
                if not clash:
                    sql_checked = True
                    break
                redo = [r for r in records if r.episode_hash in clash]
                for record in redo:
                    record.episode_hash = None
                existing |= planned
                assign_hashes(redo, existing)
            else:
                raise ManifestError("hashes still clash with app.episodes")
    except (ManifestError, OSError, ValueError) as exc:
        raise SystemExit(f"sharpadex_manifest: {exc}") from exc
    summary = summarize(records)
    header = {
        "release_root": str(args.release_root),
        "source_revision": args.source_revision,
        "source_utc_offset": args.source_utc_offset,
        "existing_hashes_checked": len(existing),
        "sql_checked": sql_checked,
        "summary": summary,
    }
    write_manifest(args.output, records, header)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
