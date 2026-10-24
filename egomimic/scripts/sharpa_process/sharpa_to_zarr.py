"""Convert SharpaDex LeRobot v2.1 episodes into dexmate_bimanual Zarr episodes.

``README.md`` in this directory is the runbook. The manifest (see
``sharpadex_records``) names, clocks and selects the episodes::

    python -m egomimic.scripts.sharpa_process.sharpa_to_zarr \\
        --dataset-root /data/SharpaDex-v1.0 --manifest manifest.jsonl \\
        --tasks iron_garment --episodes 0-1 --output-dir converted/

Each kept episode becomes, in ``--output-dir``:

- ``<episode_hash>.zarr``, the hash being the manifest's UTC start;
- ``<episode_hash>.mp4``, an H.264 preview of ``images.front_1``;
- ``<episode_hash>.validation.json``, the verification report: main's
  validator (including the dexmate_bimanual rules) run with
  ``--allow-missing``, plus forward-kinematics residuals of the stored hand
  keypoints and the arm TCP side check;
- ``<episode_hash>.conversion.json``, the sidecar, written last.

``--recording-ids FILE`` converts exactly the listed source ids, and
``--skip-done`` passes over episodes whose sidecar names them, so a killed
run resumes where it stopped. Stores and sidecars hold the canonical
``vendor_recording_id`` and the source ``source_episode_id``; ownership checks
compare the canonical one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import simplejpeg
import yaml
import zarr

from egomimic.rldb.zarr import dexmate_rules
from egomimic.rldb.zarr.required_metadata import MissingMetadata
from egomimic.rldb.zarr.validate import validate_episode
from egomimic.rldb.zarr.zarr_writer import ZarrWriter, write_annotations
from egomimic.scripts.sharpa_process.kinematics import load_spec
from egomimic.scripts.sharpa_process.lerobot_reader import (
    Episode,
    LeRobotError,
    Season,
    discover_task_seasons,
    episode_annotations,
    load_episode,
    open_season,
)
from egomimic.scripts.sharpa_process.sharpa_tracks import (
    EMBODIMENT,
    TCP_LINK,
    SharpaKinematics,
    derive_tracks,
    fk_residuals,
    joint_limit_excursions,
    sample_frames,
    tcp_side_check,
)
from egomimic.scripts.sharpa_process.sharpadex_records import (
    CLOCK_SYNTHESIZED,
    ManifestError,
    ids_correspond,
    read_manifest,
    source_episode_id,
    vendor_recording_id,
)
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz

logger = logging.getLogger(__name__)

TOOL = "egomimic.scripts.sharpa_process.sharpa_to_zarr"
#: The episode ``operator``. SharpaDex does not say who teleoperated an
#: episode, so this names the vendor, not a person: a documented surrogate.
#: When Sharpa supplies operator ids, replace it in SQL over lab = LAB.
OPERATOR = "sharpa"
#: The episode ``lab``; it must equal the S3 folder the episodes go to.
LAB = "sharpa_dex"
ACTION_SEMANTICS = "absolute_joint_position_setpoint"
#: Sharpa video stream -> EgoVerse camera; ``head_right`` needs a flag.
CAMERAS = {
    "head_left": "front_1",
    "wrist_left": "left_wrist",
    "wrist_right": "right_wrist",
}
OPTIONAL_CAMERAS = {"head_right": "head_right"}
#: Vendor columns kept beside the dexmate_bimanual arrays.
VENDOR_ARRAYS = {
    "observation.state.joint_torque": "sharpa.joint_torque",
    "observation.state.tcp": "sharpa.tcp",
    "observation.state.tcp_pose": "sharpa.obs_tcp_pose",
    "action.tcp_pose": "sharpa.cmd_tcp_pose",
    "subtask_index": "sharpa.subtask_index",
}
NOT_STORED = {
    "timestamp": "episode-relative frame_index / fps, not a capture clock",
}
FINGERS = ("thumb", "index", "middle", "ring", "little")
TACTILE_CHANNELS = ("fx", "fy", "fz", "tx", "ty", "tz")
#: The deformation mosaic: one row per hand (left first), one tile per finger.
TACTILE_TILE = 240
#: Largest accepted channel difference in a decoded grayscale mosaic.
GRAY_TOLERANCE = 4
#: Separator between a subtask's text and its skill label in SharpaDex.
SKILL_SEPARATOR = " | Skill: "
#: Not allowed in annotation text: it encodes metadata in free text.
BANNED_DELIMITER = " | "
#: ``format`` of ``<hash>.validation.json``.
REPORT_FORMAT = "sharpadex_validation_v1"
#: Fields main's validator may waive here; SharpaDex has none of them.
WAIVED_FIELDS = ("scene", "objects", "intrinsics")
_WAIVER = re.compile(r"^(\w+): marked as not available \(MissingMetadata\)$")
_HASH_FORMAT = "%Y-%m-%d-%H-%M-%S-%f"


class ConversionError(ValueError):
    """An input or option prevents converting an episode."""


def episode_hash(start_ns: int) -> str:
    """Return ``YYYY-MM-DD-HH-MM-SS-ffffff`` for a UTC start, at millisecond
    resolution, so the name round-trips through
    ``episode_rules.episode_hash_start_ms``."""
    seconds, milliseconds = divmod(int(start_ns) // 10**6, 1000)
    start = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return start.replace(microsecond=milliseconds * 1000).strftime(_HASH_FORMAT)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """One requested episode and its planned output name."""

    season: Season
    index: int
    #: ``<task>/<season>/<index>``; selects and locates the source episode.
    source_id: str
    #: The canonical id, stored as the episode's ``vendor_recording_id``.
    vendor_recording_id: str
    start_ns: int | None = None
    episode_hash: str | None = None
    error: str | None = None
    #: The episode's manifest record.
    record: object | None = None
    #: Why the manifest leaves this episode out, if it does.
    skipped: str | None = None
    #: ``--skip-done``: an earlier run's ``conversion.json`` names this episode.
    done: bool = False


def parse_indices(text: str | None) -> list[int] | None:
    """Parse ``0,3,5-7`` into sorted unique indices; ``None`` selects all."""
    if text is None:
        return None
    indices = set()
    for part in text.split(","):
        part = part.strip()
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if match is None:
            raise argparse.ArgumentTypeError(f"bad episode selection {part!r}")
        first = int(match[1])
        last = int(match[2]) if match[2] else first
        indices.update(range(first, last + 1))
    return sorted(indices)


def read_recording_ids(path: Path) -> list[str]:
    """Read one source id (``task/season/index``) per line, skipping blanks and
    ``#`` lines.

    A repeated id is dropped; the first keeps its place.
    """
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return list(dict.fromkeys(ids))


def select_recordings(ids, manifest) -> dict[tuple[str, str], list[int]]:
    """Group source ids the manifest keeps by ``(task, season)``.

    Returns:
        ``{(task, season): sorted indices}`` in sorted key order.

    Raises:
        ConversionError: If an id is not in the manifest or not kept.
    """
    selection: dict[tuple[str, str], list[int]] = {}
    bad = []
    for recording_id in ids:
        record = manifest.get(recording_id)
        if record is None or record.decision != "keep":
            why = "not in the manifest" if record is None else record.decision
            bad.append(f"{recording_id} ({why})")
            continue
        key = (record.task, record.season)
        selection.setdefault(key, []).append(record.episode_index)
    if bad:
        more = f" and {len(bad) - 10} more" if len(bad) > 10 else ""
        raise ConversionError(
            f"{len(bad)} recording id(s) are not kept manifest episodes: "
            + ", ".join(bad[:10])
            + more
        )
    return {key: sorted(selection[key]) for key in sorted(selection)}


def sidecar_ids(path: Path) -> tuple[str, str | None] | None:
    """Return ``(vendor_recording_id, source_episode_id)`` a ``conversion.json``
    names, or ``None`` if it is absent or cut off."""
    try:
        sidecar = json.loads(path.read_text())
        vid = sidecar["vendor_recording_id"]
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return vid, sidecar.get("source_episode_id")


def _owner_problem(where: Path, job: Job, vid, sid) -> str | None:
    """Why ``where`` (holding ``vid``/``sid``) is not ``job``'s to reuse, or None."""
    if vid != job.vendor_recording_id:
        return (
            f"{where} already holds {vid or 'an unidentified episode'}; refusing "
            f"to replace it with {job.vendor_recording_id}"
        )
    if sid != job.source_id or not ids_correspond(vid, sid):
        return (
            f"{where} names {vid} with source_episode_id {sid!r}, expected "
            f"{job.source_id!r}; inspect it before converting again"
        )
    return None


def plan_jobs(seasons, indices, args, manifest) -> list[Job]:
    """Name every requested episode before decoding anything.

    ``indices`` lists the episodes taken from every season (``None``: all), or
    maps a season's ``(task folder, season name)`` to its own list. The
    manifest (``{source id: record}``) supplies each name and start, and an
    episode it does not keep is skipped. A name shared by two requested
    episodes, or held by an existing output of another recording, fails those
    jobs up front. With ``--skip-done``, an episode whose
    ``<hash>.conversion.json`` names it is marked done.
    """
    jobs = []
    for season in seasons:
        task = season.path.parent.name
        if isinstance(indices, dict):
            selected = indices[(task, season.name)]
        else:
            selected = sorted(season.episodes) if indices is None else indices
        for index in selected:
            try:
                ids = (
                    source_episode_id(task, season.name, index),
                    vendor_recording_id(task, season.name, index),
                )
            except ValueError as exc:
                raise ConversionError(f"{season.path}: {exc}") from exc
            job = Job(season, index, *ids)
            jobs.append(job)
            if index not in season.episodes:
                job.error = f"episode {index} is not declared in {season.name}"
                continue
            _plan_from_manifest(job, manifest)
    by_hash: dict[str, list[Job]] = {}
    for job in jobs:
        if job.episode_hash is not None and job.error is None and not job.skipped:
            by_hash.setdefault(job.episode_hash, []).append(job)
    for name, group in by_hash.items():
        if len(group) > 1:
            ids = [job.source_id for job in group]
            for job in group:
                job.error = f"episode name {name} is shared by {ids}"
            continue
        job = group[0]
        if args.skip_done:
            # conversion.json is written last: it marks a finished episode.
            sidecar = args.output_dir / f"{name}.conversion.json"
            owner = sidecar_ids(sidecar)
            if owner is not None:
                job.error = _owner_problem(sidecar, job, *owner)
                job.done = job.error is None
                continue
        existing = args.output_dir / f"{name}.zarr"
        if existing.exists():
            try:
                attrs = zarr.open_group(existing, mode="r").attrs
                owner = attrs.get("vendor_recording_id"), attrs.get("source_episode_id")
            except Exception:  # an unreadable store is never ours to replace
                owner = None, None
            job.error = _owner_problem(existing, job, *owner)
            if job.error is None and not args.overwrite:
                job.error = f"{existing} exists; pass --overwrite to replace it"
    return jobs


def _plan_from_manifest(job: Job, manifest) -> None:
    """Take a job's name and start from its manifest record."""
    record = manifest.get(job.source_id)
    if record is None:
        job.error = f"{job.source_id} is not in the manifest"
        return
    job.record = record
    if record.decision != "keep":
        job.skipped = f"{record.decision}: {'; '.join(record.reasons)}"
        return
    info = job.season.episodes[job.index]
    if record.length != info.length or record.fps != job.season.fps:
        job.error = (
            f"manifest describes {record.length} frames at {record.fps} fps, the "
            f"season declares {info.length} at {job.season.fps}; rebuild the manifest"
        )
        return
    if record.start_ns is None or record.episode_hash != episode_hash(record.start_ns):
        job.error = f"manifest hash {record.episode_hash} does not match its start"
        return
    job.start_ns, job.episode_hash = record.start_ns, record.episode_hash


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


class _Preview:
    """Stream RGB frames into an H.264 ``<hash>.mp4`` preview."""

    def __init__(self, path: Path, fps: int, threads: int | None = None):
        self.path = path
        self.container = av.open(str(path), mode="w", format="mp4")
        codec = "libx264" if "libx264" in av.codecs_available else "h264"
        self.stream = self.container.add_stream(codec, rate=Fraction(fps, 1))
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "23", "preset": "fast"}
        if threads:
            self.stream.thread_count = threads
        self.started = False

    def add(self, image: np.ndarray) -> None:
        if not self.started:
            height, width = image.shape[:2]
            self.stream.width, self.stream.height = (
                width - width % 2,
                height - height % 2,
            )
            self.started = True
        image = image[: self.stream.height, : self.stream.width]
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self.started:
            for packet in self.stream.encode():
                self.container.mux(packet)
        self.container.close()


def available_cpus() -> int:
    """Return the CPUs this process may run on (a Slurm job's share)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def encode_camera(
    episode: Episode, camera: str, quality: int, preview=None, threads=None
):
    """Decode one stream and JPEG-encode every frame.

    Returns:
        ``(jpeg_bytes, [H, W, 3])`` with one object entry per retained frame.
    """
    encoded = np.empty(episode.length, dtype=object)
    shape = None
    for i, image in enumerate(episode.frames(camera, threads=threads)):
        if shape is None:
            shape = list(image.shape)
        elif list(image.shape) != shape:
            raise ConversionError(f"{camera}: frame {i} is {image.shape}, not {shape}")
        encoded[i] = simplejpeg.encode_jpeg(image, quality=quality, colorspace="RGB")
        if preview is not None:
            preview.add(image)
    return encoded, shape


# ---------------------------------------------------------------------------
# SharpaDex layout
# ---------------------------------------------------------------------------


def expected_vendor_names() -> dict[str, list[str]]:
    """Return the column names SharpaDex declares for the vectors it converts."""
    joints = []
    for side in dexmate_rules.SIDES:
        joints += [f"{side}_arm_j{j}" for j in range(dexmate_rules.ARM_DOF)]
        joints += [f"{side}_hand_j{j}" for j in range(dexmate_rules.HAND_DOF)]
    joints += [f"motor_j{j}" for j in range(dexmate_rules.AUX_DOF)]
    tactile = [
        f"{side}_{finger}_{channel}"
        for side in dexmate_rules.SIDES
        for finger in FINGERS
        for channel in TACTILE_CHANNELS
    ]
    tcp = [f"{side}_arm_tcp_pose_{j}" for side in dexmate_rules.SIDES for j in range(6)]
    return {
        "observation.state": joints,
        "action": joints,
        "observation.tactile": tactile,
        "observation.state.tcp_pose": tcp,
        "action.tcp_pose": tcp,
    }


def check_vendor_layout(season: Season) -> None:
    """Fail unless the season declares the column names and shapes we map.

    The block widths and order of the joint vectors, the tactile channel
    order and the deformation mosaic size are what the converter's slicing
    assumes; the hand-joint order inside a block is checked by forward
    kinematics instead (the names are generic).
    """
    features = season.info.get("features", {})
    problems = []
    for name, expected in expected_vendor_names().items():
        names = (features.get(name) or {}).get("names")
        if names != expected:
            problems.append(f"{name} names {_short(names)} != {_short(expected)}")
    deform = features.get("observation.images.tactile_deform") or {}
    if deform and list(deform.get("shape") or []) != [
        2 * TACTILE_TILE,
        len(FINGERS) * TACTILE_TILE,
        3,
    ]:
        problems.append(f"tactile_deform shape {deform.get('shape')}")
    if problems:
        raise ConversionError(
            f"{season.path}: unexpected SharpaDex layout: " + "; ".join(problems)
        )


def _short(names) -> str:
    if not isinstance(names, list):
        return repr(names)
    return f"[{len(names)}: {names[:2]}...{names[-1:]}]"


def split_tactile(values: np.ndarray) -> dict[str, np.ndarray]:
    """Split the 60-column tactile vector into ``{side}.obs_tactile`` arrays.

    Columns are left then right; within a hand, fingers thumb to little, each
    ``fx, fy, fz, tx, ty, tz``.
    """
    values = np.asarray(values)
    width = dexmate_rules.TACTILE_DIM
    if values.ndim != 2 or values.shape[1] != 2 * width:
        raise ConversionError(f"observation.tactile has shape {values.shape}")
    return {
        "left.obs_tactile": np.ascontiguousarray(values[:, :width]),
        "right.obs_tactile": np.ascontiguousarray(values[:, width:]),
    }


def encode_tactile_images(episode: Episode, quality: int, threads=None):
    """Decode ``tactile_deform`` and JPEG-encode one strip per hand.

    The mosaic is grayscale: row 0 is the left hand, row 1 the right, and each
    row holds one 240 x 240 tile per finger, thumb to little. A strip is the
    rounded channel mean repeated over RGB, so main's 3-channel image check
    and decoders accept it.

    Returns:
        ``{"left.obs_tactile_image": (jpegs, [240, 1200, 3]), "right...": ...}``.

    Raises:
        ConversionError: If a frame is not the 2 x 5 grayscale mosaic.
    """
    sides = dexmate_rules.SIDES
    out = {side: np.empty(episode.length, dtype=object) for side in sides}
    shape = [TACTILE_TILE, len(FINGERS) * TACTILE_TILE, 3]
    for i, image in enumerate(episode.frames("tactile_deform", threads=threads)):
        if image.shape != (2 * TACTILE_TILE, len(FINGERS) * TACTILE_TILE, 3):
            raise ConversionError(f"tactile_deform frame {i} is {image.shape}")
        wide = image.astype(np.int16)
        spread = int(np.abs(wide - wide[..., :1]).max())
        if spread > GRAY_TOLERANCE:
            raise ConversionError(
                f"tactile_deform frame {i} is not grayscale (channels differ by "
                f"{spread})"
            )
        gray = np.round(wide.mean(axis=2)).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        for row, side in enumerate(sides):
            strip = np.ascontiguousarray(
                rgb[row * TACTILE_TILE : (row + 1) * TACTILE_TILE]
            )
            out[side][i] = simplejpeg.encode_jpeg(
                strip, quality=quality, colorspace="RGB"
            )
    return {f"{side}.obs_tactile_image": (jpegs, shape) for side, jpegs in out.items()}


def _clean_text(text: str) -> tuple[str, bool]:
    """Drop the banned delimiter's metadata tail; report whether any was cut."""
    text = str(text).strip()
    if BANNED_DELIMITER not in text:
        return text, False
    return text.split(BANNED_DELIMITER)[0].strip(), True


def _join_parts(text: str) -> tuple[str, bool]:
    """Join text parts split by the banned delimiter with ``"; "``."""
    parts = [part.strip() for part in str(text).split(BANNED_DELIMITER)]
    return "; ".join(part for part in parts if part), len(parts) > 1


def episode_segments(annotation: dict, length: int):
    """Return subtask and skill spans from one SharpaDex annotation row.

    SharpaDex spans are ``[start_step, end_step]`` inclusive and the text
    ends in `` | Skill: <skill>``. EgoVerse spans are ``[start, end)`` and the
    text carries no delimiter-encoded metadata, so the skill moves to its own
    span list. Some tasks describe each hand in one span (``Left hand: … |
    Right hand: …``); those parts are joined with ``"; "`` so no text is
    lost. Spans are clipped to the retained ``length``.

    Returns:
        ``(subtasks, skills, notes)``; each span list holds
        ``(text, start, end)`` tuples.
    """
    subtasks, skills, notes = [], [], []
    joined = 0
    for segment in annotation.get("segments") or []:
        if segment.get("label") != "subtasks":
            continue
        start, last = segment.get("start_step"), segment.get("end_step")
        if not isinstance(start, int) or not isinstance(last, int) or last < start:
            notes.append(f"skipped malformed span {segment.get('id')}")
            continue
        end = min(last + 1, length)
        if start >= end:
            continue
        text, _, skill = str(segment.get("text") or "").partition(SKILL_SEPARATOR)
        text, several = _join_parts(text)
        joined += several
        skill = (segment.get("task_skill") or skill or "").strip()
        if text:
            subtasks.append((text, start, end))
        if skill:
            skills.append((_clean_text(skill)[0], start, end))
    if joined:
        notes.append(f"{joined} span(s) joined at {BANNED_DELIMITER!r} with '; '")
    return subtasks, skills, notes


def dataset_license(season: Season) -> str | None:
    """Return ``license`` from a README's YAML front matter.

    The task folder's README is read first, then the season's, then the
    release README above the task folders.
    """
    for folder in (season.path.parent, season.path, season.path.parent.parent):
        readme = folder / "README.md"
        if not readme.is_file():
            continue
        text = readme.read_text(errors="replace")
        match = re.match(r"---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
        if match is None:
            continue
        try:
            front = yaml.safe_load(match[1]) or {}
        except yaml.YAMLError:
            continue
        if isinstance(front, dict) and isinstance(front.get("license"), str):
            return front["license"]
    return None


def array_names() -> dict[str, list[str]]:
    """Column names of the stored arrays, for the ``sharpa_array_names`` attr.

    The writer's ``features`` names columns ``dim_0``; this converter-owned
    attr gives the joint and taxel-channel names, and the finger of each
    tactile-image tile, instead.
    """
    spec = load_spec()
    names: dict[str, list[str]] = {}
    for prefix in ("obs", "cmd"):
        names[f"{prefix}_aux_joints"] = list(dexmate_rules.AUX_JOINT_NAMES)
        for side in dexmate_rules.SIDES:
            names[f"{side}.{prefix}_joints"] = list(dexmate_rules.ARM_JOINT_NAMES[side])
            names[f"{side}.{prefix}_hand_joints"] = list(
                dexmate_rules.HAND_JOINT_NAMES[side]
            )
    for side in dexmate_rules.SIDES:
        tactile = spec["hands"][side]["tactile"]
        names[f"{side}.obs_tactile"] = [
            f"{f}_{c}" for f in tactile["fingers"] for c in tactile["channel_names"]
        ]
        names[f"{side}.obs_tactile_image"] = list(tactile["fingers"])
    return names


def tactile_units() -> dict[str, str]:
    """Unit of each tactile channel, for the ``sharpa_tactile_units`` attr.

    Inferred from value ranges; Sharpa has not confirmed them.
    """
    tactile = load_spec()["hands"]["left"]["tactile"]
    return dict(zip(tactile["channel_names"], tactile["channel_units"], strict=True))


# ---------------------------------------------------------------------------
# Verification report
# ---------------------------------------------------------------------------


def _fk_check(store, total_frames: int, kinematics) -> tuple[dict, list[str]]:
    """FK residuals of the stored keypoints, on up to 64 evenly spaced frames."""
    results, problems = {}, []
    frames = sample_frames(total_frames)
    for prefix in ("obs", "cmd"):
        for side in dexmate_rules.SIDES:
            keys = [
                f"{side}.{prefix}_hand_joints",
                f"{side}.{prefix}_hand_keypoints",
                f"{side}.{prefix}_ee_pose",
            ]
            if not all(key in store for key in keys):
                if prefix == "obs":
                    problems.append(f"{side}: no {prefix} joints, keypoints and pose")
                continue
            tracks = [np.asarray(store[key][:total_frames])[frames] for key in keys]
            residuals = fk_residuals(side, *tracks, kinematics)
            worst = float(residuals.max()) if residuals.size else 0.0
            tolerance = float(load_spec()["hands"][side]["fk_tolerance_m"])
            results[f"{side}.{prefix}"] = worst
            if not np.isfinite(residuals).all() or worst > tolerance:
                slot = int(np.unravel_index(residuals.argmax(), residuals.shape)[1])
                problems.append(
                    f"{side}.{prefix}_hand_keypoints disagree with forward "
                    f"kinematics of {side}.{prefix}_hand_joints by {worst:.4f} m at "
                    f"slot {slot} (tolerance {tolerance:g} m)"
                )
    return {"max_m": results, "frames": len(frames)}, problems


def verification_report(path: Path, kinematics, tcp: dict | None) -> dict:
    """Return one report for a written episode.

    It merges main's ``validate_episode(path, allow_missing=True)``, which
    already runs ``dexmate_rules.check_dexmate``, with the checks that need
    the URDFs: FK residuals of the stored observed and commanded keypoints,
    and the arm TCP side check computed during conversion. It also checks
    that the stored ids correspond and that only scene, objects and
    intrinsics are waived. ``ok`` is ``not errors``.
    """
    validation = validate_episode(path, allow_missing=True)
    errors = [f"validator: {e}" for e in validation.errors]
    warnings = [f"validator: {w}" for w in validation.warnings]
    waived = [m[1] for w in validation.warnings if (m := _WAIVER.match(w))]
    for field in sorted(set(waived) - set(WAIVED_FIELDS)):
        errors.append(f"waiver: {field} is not expected to be missing")
    vid = sid = None
    fk: dict = {}
    try:
        store = zarr.open_group(str(path), mode="r", zarr_format=3)
        vid = store.attrs.get("vendor_recording_id")
        sid = store.attrs.get("source_episode_id")
        if not ids_correspond(vid, sid):
            errors.append(
                f"ids: vendor_recording_id {vid!r} and source_episode_id {sid!r} "
                "do not name the same episode"
            )
        total_frames = int(store.attrs["total_frames"])
        fk, problems = _fk_check(store, total_frames, kinematics)
        errors += [f"fk_residual: {p}" for p in problems]
    except Exception as exc:  # the validator has reported what is unreadable
        errors.append(f"fk_residual: could not run: {type(exc).__name__}: {exc}")
    if not isinstance(tcp, dict) or not isinstance(tcp.get("passed"), bool):
        errors.append("tcp_check: missing")
    elif not tcp["passed"]:
        errors += [f"tcp_check: {p}" for p in tcp.get("problems") or ["failed"]]
    return {
        "format": REPORT_FORMAT,
        "episode": Path(path).name,
        "vendor_recording_id": vid,
        "source_episode_id": sid,
        "allow_missing": True,
        "waived": sorted(set(waived)),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "validator": {
                "ok": validation.ok,
                "errors": len(validation.errors),
                "warnings": len(validation.warnings),
            },
            "fk_residual": fk,
            "tcp_check": tcp,
        },
    }


def read_report(path: Path, vendor_recording_id: str | None = None):
    """Read a ``validation.json`` strictly; return ``(report, problems)``.

    A missing, unreadable or malformed report is a problem, as is any error
    it lists or, given ``vendor_recording_id``, a report for another episode.
    ``ok`` is recomputed from ``errors`` rather than trusted. An empty
    ``problems`` means the episode passed.
    """
    path = Path(path)
    try:
        report = json.loads(path.read_text())
    except FileNotFoundError:
        return None, [f"no {path.name}"]
    except (OSError, ValueError) as exc:
        return None, [f"unreadable {path.name}: {exc}"]
    if not isinstance(report, dict) or report.get("format") != REPORT_FORMAT:
        return None, [f"{path.name} is not a {REPORT_FORMAT} report"]
    lists = [report.get(key) for key in ("errors", "warnings")]
    if not all(
        isinstance(x, list) and all(isinstance(s, str) for s in x) for x in lists
    ):
        return None, [f"{path.name}: errors and warnings must be lists of strings"]
    errors = lists[0]
    if report.get("ok") is not (not errors):
        return None, [f"{path.name}: ok={report.get('ok')!r} disagrees with errors"]
    problems = []
    if vendor_recording_id is not None and (
        report.get("vendor_recording_id") != vendor_recording_id
    ):
        problems.append(
            f"{path.name} is for {report.get('vendor_recording_id')!r}, "
            f"not {vendor_recording_id!r}"
        )
    if errors:
        problems.append(f"{len(errors)} validation error(s): {errors[0][:200]}")
    return report, problems


# ---------------------------------------------------------------------------
# Episode conversion
# ---------------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).parent),
                "describe",
                "--always",
                "--dirty",
                "--abbrev=40",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _jsonable(value):
    """Convert NumPy scalars and arrays inside attribute values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _discard_partials(output_dir: Path, name: str) -> None:
    """Remove a failed conversion's unfinished outputs."""
    for path in (
        output_dir / f".{name}.zarr.partial",
        output_dir / f".{name}.mp4.partial",
    ):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _warn_box(lines: list[str]) -> None:
    width = max(len(line) for line in lines) + 4
    print("!" * width, file=sys.stderr)
    for line in lines:
        print(f"! {line}", file=sys.stderr)
    print("!" * width, file=sys.stderr)


def status_reasons(
    *,
    has_front: bool,
    synthesized: bool,
    excursions: bool,
    gross: bool,
    truncated: tuple[int, int] | None,
    tcp_problem: str | None,
) -> tuple[list[str], list[str]]:
    """Return ``(reasons, limitations)`` for an episode's data status.

    Any reason makes the episode a ``structural_sample``; only ``complete``
    episodes are ingested. Reasons are defects in the stored data (a missing
    ego video, a gross observed joint excursion, truncation, a failed TCP
    check). Limitations list what the delivery does not provide (camera
    calibration, a head camera pose, a measured clock, joint excursions
    inside the validator's error margin).
    """
    reasons = []
    limitations = [
        "no measured front_1 calibration",
        "no obs_head_pose: no measured head camera mount",
    ]
    if not has_front:
        reasons.append("no head_left (front_1) video")
    if synthesized:
        limitations.append(
            "capture clock synthesized from the season name and frame index"
        )
    if excursions and gross:
        reasons.append(
            "observed hand joints exceed the validator's error margin "
            f"({dexmate_rules.ERROR_MARGIN_RAD} rad)"
        )
    elif excursions:
        limitations.append(
            "joint values outside the URDF limits, inside the validator's error margin"
        )
    if truncated:
        reasons.append(f"truncated to {truncated[0]} of {truncated[1]} frames")
    if tcp_problem:
        reasons.append(tcp_problem)
    return reasons, limitations


_ANNOTATIONS: dict[Path, dict[int, dict]] = {}


def _season_annotations(season: Season) -> dict[int, dict]:
    """Read a season's annotations once per process."""
    if season.root not in _ANNOTATIONS:
        _ANNOTATIONS[season.root] = episode_annotations(season)
    return _ANNOTATIONS[season.root]


def convert_episode(job: Job, args, kinematics: SharpaKinematics) -> dict:
    """Write one episode and its sidecars; return the conversion summary."""
    season, record = job.season, job.record
    started = time.monotonic()
    check_vendor_layout(season)
    license_name = args.license or dataset_license(season)
    if license_name is None:
        raise ConversionError(
            f"{season.path}: no license in a README above the season; pass --license"
        )
    episode = load_episode(season, job.index, args.max_frames)
    columns = episode.columns
    for name in (
        "observation.state",
        "action",
        "observation.state.tcp_pose",
        "observation.tactile",
    ):
        if name not in columns:
            raise ConversionError(f"{season.name}: column {name!r} is missing")

    # The manifest's start, advanced by the frame index.
    notes = []
    steps = (np.arange(episode.length, dtype=np.int64) * 10**9) // season.fps
    stamps = record.start_ns + steps
    if record.hash_offset_ms:
        notes.append(
            f"start moved {record.hash_offset_ms} ms to keep the episode name unique"
        )
    if episode_hash(int(stamps[0])) != job.episode_hash:
        raise ConversionError("the episode clock changed after planning")

    # Joint values are stored unchanged; excursions are reported.
    source = {"obs": columns["observation.state"], "cmd": columns["action"]}
    excursions = {
        prefix: joint_limit_excursions(raw, prefix, kinematics)
        for prefix, raw in source.items()
    }
    excursion_count = sum(len(r) for r in excursions.values())
    checked = [
        r for records in excursions.values() for r in records if r["validator_checked"]
    ]

    # Cameras: stored, absent from this season, or not requested.
    wanted = dict(CAMERAS)
    if args.include_head_right:
        wanted.update(OPTIONAL_CAMERAS)
    present = {s: name for s, name in wanted.items() if s in season.cameras}
    camera_report = {
        "stored": {name: s for s, name in present.items()},
        "absent_in_source": sorted(set(wanted) - set(present)),
        "not_requested": sorted(
            c for c in season.cameras if c not in wanted and not c.startswith("tactile")
        ),
        "never_stored": sorted(
            c
            for c in season.cameras
            if c.startswith("tactile") and c != "tactile_deform"
        ),
    }
    tactile_images = "tactile_deform" in season.cameras
    if tactile_images:
        camera_report["tactile_images"] = {
            "tactile_deform": ["left.obs_tactile_image", "right.obs_tactile_image"]
        }

    tcp_links = [TCP_LINK.format(side=side) for side in dexmate_rules.SIDES]
    tracks, link_poses = derive_tracks(
        source["obs"],
        source["cmd"],
        kinematics=kinematics,
        links=(kinematics.head_link, *tcp_links),
    )
    numeric = dict(tracks)
    numeric["obs_rgb_timestamps_ns"] = stamps
    numeric.update(split_tactile(columns["observation.tactile"]))
    numeric[dexmate_rules.HEAD_LINK_POSE] = _matrix_to_xyzwxyz(
        link_poses[kinematics.head_link]
    )
    for name, key in VENDOR_ARRAYS.items():
        if name in columns:
            numeric[key] = columns[name]
    tcp = tcp_side_check(columns["observation.state.tcp_pose"], link_poses)

    # Decode and JPEG-encode all streams; streams decode concurrently.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial = args.output_dir / f".{job.episode_hash}.zarr.partial"
    preview_path = args.output_dir / f".{job.episode_hash}.mp4.partial"
    # Every stream decodes at once; split this process's CPUs between them so
    # FFmpeg does not start a thread per host CPU in each decoder.
    streams = max(1, len(present) + tactile_images)
    threads = max(1, available_cpus() // streams)
    preview = (
        _Preview(preview_path, season.fps, threads=threads)
        if "head_left" in present
        else None
    )
    try:
        with ThreadPoolExecutor(max_workers=streams) as pool:
            futures = {
                name: pool.submit(
                    encode_camera,
                    episode,
                    s,
                    args.jpeg_quality,
                    preview if s == "head_left" else None,
                    threads,
                )
                for s, name in present.items()
            }
            tactile = (
                pool.submit(encode_tactile_images, episode, args.jpeg_quality, threads)
                if tactile_images
                else None
            )
            images = {f"images.{name}": f.result() for name, f in futures.items()}
            tactile = {} if tactile is None else tactile.result()
    finally:
        if preview is not None:
            preview.close()

    tcp_problem = None
    if not tcp["passed"]:
        tcp_problem = "arm TCP check failed: " + "; ".join(tcp["problems"])
        logger.warning("%s (%s): %s", job.episode_hash, job.source_id, tcp_problem)
    reasons, limitations = status_reasons(
        has_front="images.front_1" in images,
        synthesized=record.clock == CLOCK_SYNTHESIZED,
        excursions=bool(excursion_count),
        gross=bool(checked),
        truncated=(episode.length, episode.info.length) if episode.truncated else None,
        tcp_problem=tcp_problem,
    )
    status = "structural_sample" if reasons else "complete"

    row = _season_annotations(season).get(job.index, {})
    annotations, skills, annotation_notes = episode_segments(row, episode.length)
    report = {
        "tool": TOOL,
        "source": {
            "dataset": season.path.parent.name,
            "season": season.name,
            "episode_index": job.index,
            "lerobot_version": season.version,
            "source_frames": episode.info.length,
            "retained_frames": episode.length,
            "truncated": episode.truncated,
        },
        "timestamps": {"source": record.clock, "notes": notes},
        "cameras": camera_report,
        "joint_limits": {
            "tolerance": 1e-6,
            "source_excursions": excursions,
            "stored_excursions_checked_by_validator": len(checked),
        },
        "not_stored": {k: v for k, v in NOT_STORED.items() if k in columns},
        "data_status": status,
        "data_status_reasons": reasons,
        "data_limitations": limitations,
        "tcp_check": tcp,
        "annotations": {
            "subtasks": len(annotations),
            "skills": len(skills),
            "notes": annotation_notes,
        },
    }
    tags = dict(record.tags)
    header = args.manifest_header
    metadata = _jsonable(
        {
            "source_episode_id": job.source_id,
            "license": license_name,
            "action_semantics": ACTION_SEMANTICS,
            "source_dataset": Path(header.get("release_root") or "").name
            or season.path.parent.parent.name,
            "source_task_folder": season.path.parent.name,
            "source_season": season.name,
            "source_episode_index": job.index,
            "source_lerobot_version": season.version,
            "source_robot_type": season.robot_type,
            "source_task": list(episode.info.tasks),
            "source_revision": header.get("source_revision") or "unknown",
            "source_uid": record.source_uid,
            "source_score": tags.get("score"),
            "source_task_name": tags.get("task_name"),
            "source_tags": tags,
            "rig_name": season.rig,
            "timestamp_source": record.clock,
            "start_source": record.start_source,
            "source_utc_offset": header.get("source_utc_offset"),
            "hash_offset_ms": record.hash_offset_ms,
            "quality_flags": list(record.flags),
            "data_limitations": limitations,
            "conversion_tool": TOOL,
            "conversion_tool_version": _git_sha() or "unknown",
            "conversion_report": report,
            "sharpa_array_names": array_names(),
            "sharpa_tactile_units": tactile_units(),
        }
    )
    task_description = _clean_text(tags.get("task_instruction") or "")[0]

    if partial.exists():
        shutil.rmtree(partial)
    ZarrWriter(
        episode_path=partial,
        embodiment=EMBODIMENT,
        # SharpaDex ships no camera calibration (see dexmate_rules); scene
        # and objects have no source field.
        intrinsics=MissingMetadata(),
        task_name=season.path.parent.name,
        operator=OPERATOR,
        lab=LAB,
        scene=MissingMetadata(),
        objects=MissingMetadata(),
        vendor_recording_id=job.vendor_recording_id,
        fps=season.fps,
        task_description=task_description,
        annotations=annotations,
        chunk_timesteps=args.chunk_timesteps,
    ).write(
        numeric_data=numeric,
        pre_encoded_image_data={**images, **tactile},
        metadata_override=metadata,
    )
    del images, tactile
    if skills:
        write_annotations(partial, skills, "annotations_skill")

    final = args.output_dir / f"{job.episode_hash}.zarr"
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)
    if preview is not None:
        preview_path.rename(args.output_dir / f"{job.episode_hash}.mp4")

    summary = {
        "episode": str(final),
        "episode_hash": job.episode_hash,
        "vendor_recording_id": job.vendor_recording_id,
        "source_episode_id": job.source_id,
        "total_frames": episode.length,
        "data_status": status,
        "seconds": round(time.monotonic() - started, 1),
    }
    if checked:
        joints = sorted({f"{r['array']}:{r['joint']}" for r in checked})
        worst = max(max(r["max_above"], r["max_below"]) for r in checked)
        _warn_box(
            [
                f"{job.episode_hash} ({job.source_id}): {len(joints)} observed "
                f"hand joint column(s) exceed the URDF limits by up to "
                f"{worst:.4f} rad:",
                *[f"  {j}" for j in joints],
                "This episode FAILS the validator's joint-limit check: an "
                "excursion this large points at",
                "units, column order or corrupt frames. Raise it with Sharpa. "
                f"Details: {job.episode_hash}.conversion.json",
            ]
        )
    verification = verification_report(final, kinematics, tcp)
    (args.output_dir / f"{job.episode_hash}.validation.json").write_text(
        json.dumps(_jsonable(verification), indent=2) + "\n"
    )
    summary["validation"] = {
        "ok": verification["ok"],
        "errors": verification["errors"],
        "warnings": len(verification["warnings"]),
    }
    # Written last: its presence marks a finished episode.
    sidecar = {**summary, "attributes": metadata}
    (args.output_dir / f"{job.episode_hash}.conversion.json").write_text(
        json.dumps(sidecar, indent=2) + "\n"
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m egomimic.scripts.sharpa_process.sharpa_to_zarr",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="SharpaDex release root (<root>/<task>/<season>/lerobot_v2.1).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="SharpaDex manifest; names, clocks and selects episodes.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks", default="*", help="Glob over task folder names (default: all)."
    )
    parser.add_argument(
        "--seasons",
        default="*",
        help="Glob over season directory names (default: all).",
    )
    parser.add_argument(
        "--episodes",
        type=parse_indices,
        help="Episode indices such as 0,3,5-7 (default: all).",
    )
    parser.add_argument(
        "--recording-ids",
        type=Path,
        help="File of kept source ids (task/season/index), one per line; "
        "convert exactly those (replaces --tasks, --seasons and --episodes).",
    )
    parser.add_argument(
        "--include-head-right",
        action="store_true",
        help="Also store head_right as images.head_right.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Keep only the first N frames (smoke runs; makes a structural sample).",
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--chunk-timesteps", type=int, default=100)
    parser.add_argument(
        "--license", help="Override the license read from the release README."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output of the same recording.",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip an episode whose <hash>.conversion.json names the same "
        "recording (finished by an earlier run).",
    )
    return parser


def main(argv=None) -> int:
    """Convert the selected episodes.

    Returns:
        ``0`` on success, ``1`` if any episode failed to convert, ``2`` if all
        converted but a verification report has errors.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be in 1..100")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be positive")
    if args.chunk_timesteps < 1:
        raise SystemExit("--chunk-timesteps must be positive")
    if args.recording_ids is not None:
        mixed = [
            flag
            for flag, given in (
                ("--tasks", args.tasks != "*"),
                ("--seasons", args.seasons != "*"),
                ("--episodes", args.episodes is not None),
            )
            if given
        ]
        if mixed:
            raise SystemExit(
                f"--recording-ids selects the episodes; drop {', '.join(mixed)}"
            )
    try:
        args.manifest_header, manifest = read_manifest(args.manifest)
    except (ManifestError, OSError) as exc:
        raise SystemExit(f"sharpa_to_zarr: {exc}") from exc
    selection = None
    if args.recording_ids is not None:
        try:
            ids = read_recording_ids(args.recording_ids)
            selection = select_recordings(ids, manifest)
        except (ConversionError, OSError) as exc:
            raise SystemExit(f"sharpa_to_zarr: {args.recording_ids}: {exc}") from exc
        if not selection:
            raise SystemExit(
                f"sharpa_to_zarr: {args.recording_ids} lists no recording ids"
            )
    import torch

    # Small 4x4 batches gain nothing from many threads; on a shared host,
    # oversubscription can slow forward kinematics by orders of magnitude.
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    try:
        kinematics = SharpaKinematics()
        if selection is not None:
            # Open only the listed seasons, by exact name.
            seasons = [
                open_season(args.dataset_root / task / name) for task, name in selection
            ]
        else:
            seasons = discover_task_seasons(args.dataset_root, args.seasons, args.tasks)
        if not seasons:
            raise ConversionError(
                f"no season matches --tasks {args.tasks!r} --seasons "
                f"{args.seasons!r} under {args.dataset_root}"
            )
        indices = args.episodes if selection is None else selection
        jobs = plan_jobs(seasons, indices, args, manifest)
    except (ConversionError, LeRobotError, OSError, ValueError) as exc:
        raise SystemExit(f"sharpa_to_zarr: {exc}") from exc
    logger.info("%d episode(s) from %d season(s)", len(jobs), len(seasons))
    converted, failed, skipped, done = [], [], [], []
    for job in jobs:
        if job.skipped and job.error is None:
            skipped.append(job)
            logger.info("skipped %s: %s", job.source_id, job.skipped)
            continue
        if job.done:
            done.append(job)
            logger.info(
                "already done %s: %s.conversion.json", job.source_id, job.episode_hash
            )
            continue
        if job.error is None:
            try:
                summary = convert_episode(job, args, kinematics)
                converted.append(summary)
                logger.info(
                    "%s <- %s: %d frames, %s, %.1f s",
                    summary["episode_hash"],
                    job.source_id,
                    summary["total_frames"],
                    summary["data_status"],
                    summary["seconds"],
                )
                for error in summary["validation"]["errors"]:
                    logger.warning("  validation: %s", error)
                continue
            except (ConversionError, LeRobotError, OSError) as exc:
                job.error = str(exc)
            except Exception as exc:  # report and continue with other episodes
                logger.exception("unexpected failure converting %s", job.source_id)
                job.error = f"{type(exc).__name__}: {exc}"
            _discard_partials(args.output_dir, job.episode_hash)
        failed.append(job)
        logger.error("FAILED %s: %s", job.source_id, job.error)
    invalid = [s for s in converted if not s["validation"]["ok"]]
    print(
        f"Converted {len(converted)} episode(s), {len(failed)} failed"
        + (f", {len(done)} already done" if args.skip_done else "")
        + f", {len(skipped)} skipped by the manifest"
        + f", {len(invalid)} with validation errors"
        + f"; output in {args.output_dir}"
    )
    if failed:
        return 1
    return 2 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
