"""Convert Sharpa LeRobot episodes into EgoVerse Zarr v3.1 episodes.

The vendor guide, ``docs/SHARPA_LEROBOT_CONVERSION.md``, gives the field
mapping, the inputs a ``complete`` delivery needs, and the verification
commands. A typical run::

    python -m egomimic.scripts.sharpa_process.sharpa_to_zarr \\
        --dataset-root /data/RoCo_TaskBoardAssembly \\
        --seasons 'season_POC22002_2026_08_15_*' --episodes 0 \\
        --task-name task_board_assembly --output-dir converted/ --validate

Each episode becomes ``<output-dir>/<episode_hash>.zarr``, where the hash is
the episode's UTC start time, plus a ``<episode_hash>.conversion.json``
sidecar. Joint values are written unchanged. Values outside the registry or
URDF limits are reported; observed hand joints far outside them fail the
validator's integrity check unless ``--clip-joint-limits`` is given; clipping
is for local experiments only.

``--profile sharpadex`` converts a SharpaDex release
(``<release>/<task>/<season>/lerobot_v2.1``) planned by ``sharpadex_manifest``::

    python -m egomimic.scripts.sharpa_process.sharpa_to_zarr --profile sharpadex \
        --dataset-root /data/SharpaDex-v1.0 --manifest manifest.jsonl \
        --tasks iron_garment --episodes 0-1 --output-dir converted/ \
        --status-policy provenance --validate

The profile names, clocks and selects episodes from the manifest, takes the
task name from the task folder and the description from Sharpa's tags, stores
subtask and skill annotations, both hands' tactile vectors and deformation
images, commanded keypoints, the head-link pose and provenance attributes, and
checks the vendor column layout and the arm TCPs against forward kinematics.
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
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import simplejpeg
import yaml
import zarr

from egomimic.rldb.zarr.calibration import (
    CalibrationError,
    parse_calibration,
)
from egomimic.rldb.zarr.episode_attrs import (
    DATA_STATUS_COMPLETE,
    DATA_STATUS_STRUCTURAL_SAMPLE,
)
from egomimic.rldb.zarr.zarr_writer import ZarrWriter
from egomimic.scripts.sharpa_process.lerobot_reader import (
    Episode,
    LeRobotError,
    Season,
    discover_seasons,
    discover_task_seasons,
    episode_annotations,
    load_episode,
)
from egomimic.scripts.sharpa_process.sharpa_tracks import (
    EMBODIMENT,
    HEAD_LINK,
    TCP_LINK,
    SharpaKinematics,
    arm_joint_names,
    clip_to_limits,
    derive_tracks,
    joint_limit_excursions,
    tcp_side_check,
)
from egomimic.utils.aws.aws_sql import timestamp_ms_to_episode_hash
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "v3.1"
TOOL = "egomimic.scripts.sharpa_process.sharpa_to_zarr"
#: Sharpa video stream → EgoVerse camera; ``head_right`` needs a flag.
CAMERAS = {
    "head_left": "front_1",
    "wrist_left": "left_wrist",
    "wrist_right": "right_wrist",
}
OPTIONAL_CAMERAS = {"head_right": "head_right"}
#: Raw vendor columns kept by ``--keep-vendor-arrays``, outside the contract.
VENDOR_ARRAYS = {
    "observation.state": "sharpa.observation_state",
    "action": "sharpa.action",
    "observation.state.joint_torque": "sharpa.joint_torque",
    "observation.state.tcp": "sharpa.tcp",
    "observation.tactile": "sharpa.observation_tactile",
}
#: Vendor columns the SharpaDex profile keeps beside the contract arrays.
SHARPADEX_VENDOR_ARRAYS = {
    "observation.state.joint_torque": "sharpa.joint_torque",
    "observation.state.tcp": "sharpa.tcp",
    "observation.state.tcp_pose": "sharpa.obs_tcp_pose",
    "action.tcp_pose": "sharpa.cmd_tcp_pose",
    "subtask_index": "sharpa.subtask_index",
}
NOT_STORED = {
    "observation.tactile": "stored as {side}.obs_tactile only with --store-tactile",
    "observation.state.joint_torque": "no contract array; --keep-vendor-arrays",
    "observation.state.tcp": "no contract array; --keep-vendor-arrays",
    "timestamp": "episode-relative frame_index / fps, not a capture clock",
}
SYNTHESIZED = "synthesized_from_season_name_and_frame_index"
FINGERS = ("thumb", "index", "middle", "ring", "little")
TACTILE_CHANNELS = ("fx", "fy", "fz", "tx", "ty", "tz")
#: The deformation mosaic: one row per hand (left first), one tile per finger.
TACTILE_TILE = 240
#: Largest accepted channel difference in a decoded grayscale mosaic.
GRAY_TOLERANCE = 4
#: Separator between a subtask's text and its skill label in SharpaDex.
SKILL_SEPARATOR = " | Skill: "
BANNED_DELIMITER = " | "
_CALIBRATION_FIELDS = frozenset(
    {
        "K",
        "model",
        "distortion",
        "resolution",
        "rectified",
        "parent_link",
        "parent_link_T_cam",
    }
)
_REQUIRED_CALIBRATION_FIELDS = _CALIBRATION_FIELDS - {"distortion"}
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class ConversionError(ValueError):
    """An input or option prevents converting an episode."""


# ---------------------------------------------------------------------------
# Calibration file
# ---------------------------------------------------------------------------


def _placeholders(value) -> list[str]:
    """Return string leaves; the template marks unmeasured values as strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [s for item in value for s in _placeholders(item)]
    return []


def _rigid(value, where: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConversionError(f"{where}: not a numeric 4x4 matrix ({exc})") from exc
    rotation = matrix[:3, :3] if matrix.shape == (4, 4) else None
    if (
        rotation is None
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ConversionError(
            f"{where}: expected a 4x4 rigid transform with a proper rotation"
        )
    return matrix


def load_calibration_file(path: Path, kinematics: SharpaKinematics) -> dict:
    """Read and strictly validate a rig calibration YAML.

    The file maps a rig name (``POC22002``) or ``default`` to Sharpa camera
    names (``head_left``, ``wrist_left``, ``wrist_right``, ``head_right``),
    each with ``K``, ``model``, ``distortion``, ``resolution``, ``rectified``,
    ``parent_link`` and ``parent_link_T_cam``. See ``calibration_template.yaml``.

    Returns:
        ``{rig: {camera: entry}}`` with ``parent_link_T_cam`` as an array.

    Raises:
        ConversionError: For an unknown field or camera, a placeholder, a
            malformed camera model, an unknown link, or a non-rigid transform.
    """
    try:
        data = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConversionError(f"{path}: cannot read calibration ({exc})") from exc
    if not isinstance(data, dict) or not data:
        raise ConversionError(f"{path}: expected a mapping of rig name to cameras")
    known = {**CAMERAS, **OPTIONAL_CAMERAS}
    rigs = {}
    for rig, cameras in data.items():
        where = f"{path}[{rig!r}]"
        if not isinstance(rig, str) or not isinstance(cameras, dict) or not cameras:
            raise ConversionError(f"{where}: expected a non-empty camera mapping")
        rigs[rig] = {}
        for camera, entry in cameras.items():
            here = f"{where}[{camera!r}]"
            if camera not in known:
                raise ConversionError(
                    f"{here}: unknown camera; expected one of {sorted(known)}"
                )
            if not isinstance(entry, dict):
                raise ConversionError(f"{here}: expected a mapping")
            unknown = sorted(set(entry) - _CALIBRATION_FIELDS)
            missing = sorted(_REQUIRED_CALIBRATION_FIELDS - set(entry))
            if unknown or missing:
                raise ConversionError(
                    f"{here}: unknown field(s) {unknown}, missing field(s) {missing}"
                )
            filler = _placeholders(
                [entry["K"], entry.get("distortion") or [], entry["resolution"]]
                + [entry["parent_link_T_cam"]]
            )
            if filler:
                raise ConversionError(
                    f"{here}: placeholder value(s) {filler[:4]}; replace every "
                    "template value with the rig's measured calibration"
                )
            block = {k: entry[k] for k in ("K", "model", "resolution", "rectified")}
            block["distortion"] = entry.get("distortion") or []
            try:
                parsed = parse_calibration(
                    {"reference_frame": "robot_base", "cameras": {camera: block}},
                    where=here,
                ).cameras[camera]
            except CalibrationError as exc:
                raise ConversionError(str(exc)) from exc
            K = parsed.K
            if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1, 0]):
                raise ConversionError(
                    f"{here}.K: focal lengths must be positive and the last row "
                    "[0, 0, 1]"
                )
            if parsed.resolution is None:
                raise ConversionError(f"{here}.resolution: required [width, height]")
            if entry["parent_link"] not in kinematics.links:
                raise ConversionError(
                    f"{here}.parent_link: {entry['parent_link']!r} is not a link of "
                    "the platform or hand URDFs"
                )
            rigs[rig][camera] = {
                "camera": parsed,
                "parent_link": entry["parent_link"],
                "parent_link_T_cam": _rigid(
                    entry["parent_link_T_cam"], f"{here}.parent_link_T_cam"
                ),
            }
    return rigs


def build_calibration(entries, stored, link_poses, kinematics, image_shapes):
    """Return the calibration block, mount records and ``obs_head_pose``.

    Args:
        entries: One rig's validated entries keyed by Sharpa camera name.
        stored: ``{sharpa camera: egoverse camera}`` for the stored streams.
        link_poses: Observed ``robot_base_T_link`` tracks of the parent links.
        kinematics: Loaded models.
        image_shapes: ``{egoverse camera: (H, W, 3)}`` of the stored frames.

    Returns:
        ``(block | None, camera_mounts, obs_head_pose | None)``.
    """
    cameras, mounts, head_pose = {}, {}, None
    for source, entry in entries.items():
        if source not in stored:
            continue
        name, camera = stored[source], entry["camera"]
        height, width = image_shapes[name][:2]
        if camera.resolution != (width, height):
            raise ConversionError(
                f"calibration for {source} declares resolution "
                f"{list(camera.resolution)}, stored frames are [{width}, {height}]; "
                "K must describe the stored frames"
            )
        parent, mount = entry["parent_link"], entry["parent_link_T_cam"]
        fixed = kinematics.is_fixed_in_base(parent)
        block = camera.to_jsonable()
        if fixed:
            block["ref_T_cam"] = (link_poses[parent][0] @ mount).tolist()
        elif name == "front_1":
            # The optical pose rides the observed head chain at every frame.
            head_pose = _matrix_to_xyzwxyz(link_poses[parent] @ mount)
        cameras[name] = block
        mounts[name] = {
            "source_camera": source,
            "parent_link": parent,
            "parent_link_T_cam": mount.tolist(),
            "fixed_in_reference_frame": fixed,
        }
    if not cameras:
        return None, {}, None
    block = {"reference_frame": "robot_base", "cameras": cameras}
    parse_calibration(block)
    return block, mounts, head_pose


# ---------------------------------------------------------------------------
# Clock and naming
# ---------------------------------------------------------------------------


def parse_utc_offset(text: str) -> timezone:
    """Parse ``+08:00``, ``-0530`` or ``Z`` into a fixed time zone."""
    if text in ("Z", "UTC"):
        return timezone.utc
    match = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", text)
    if match is None or int(match[2]) > 23 or int(match[3]) > 59:
        raise ConversionError(f"--source-utc-offset {text!r} is not ±HH:MM")
    sign = 1 if match[1] == "+" else -1
    return timezone(sign * timedelta(hours=int(match[2]), minutes=int(match[3])))


def utc_ns(local: datetime, zone: timezone) -> int:
    """Return integer UTC nanoseconds for a naive local time in ``zone``."""
    delta = local.replace(tzinfo=zone) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 10**9 + delta.microseconds * 1000


def synthesized_start_ns(season: Season, index: int, zone: timezone) -> int:
    """Season start plus the episode's frame offset, in integer nanoseconds."""
    if season.start_local is None:
        raise ConversionError(
            f"{season.name}: the directory name carries no start time, so a "
            "clock cannot be synthesized; pass --utc-timestamp-column"
        )
    offset = season.episodes[index].frame_offset
    return utc_ns(season.start_local, zone) + (offset * 10**9) // season.fps


def synthesized_clock(season: Season, index: int, frames: int, zone) -> np.ndarray:
    """Return ``start + (i * 1e9) // fps`` as int64, computed in integers."""
    steps = (np.arange(frames, dtype=np.int64) * 10**9) // season.fps
    return synthesized_start_ns(season, index, zone) + steps


def vendor_clock(episode: Episode, column: str) -> np.ndarray:
    """Return a vendor UTC-nanosecond column after checking type and order."""
    values = np.asarray(episode.columns[column])
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ConversionError(
            f"{column} must be a 1-D integer UTC-nanosecond column, got "
            f"{values.dtype} with shape {values.shape}"
        )
    values = values.astype(np.int64)
    stalled = int(np.count_nonzero(np.diff(values) <= 0))
    if stalled:
        raise ConversionError(f"{column} does not increase on {stalled} step(s)")
    return values


def episode_hash(start_ns: int) -> str:
    """Return ``YYYY-MM-DD-HH-MM-SS-ffffff`` for a UTC start, at millisecond
    resolution so the name round-trips through the episode registry helpers."""
    return timestamp_ms_to_episode_hash(start_ns // 10**6)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """One requested episode and its planned output name."""

    season: Season
    index: int
    recording_id: str
    start_ns: int | None = None
    episode_hash: str | None = None
    error: str | None = None
    #: The episode's ``sharpadex_manifest`` record, under ``--manifest``.
    record: object | None = None
    #: Why the manifest leaves this episode out, if it does.
    skipped: str | None = None


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


def plan_jobs(seasons, indices, args, zone, manifest=None) -> list[Job]:
    """Name every requested episode before decoding anything.

    A name shared by two requested episodes, or held by an existing output
    from another recording, fails those jobs up front. With a ``manifest``
    (``{recording_id: record}``), the record supplies the name and start, and
    an episode the manifest does not keep is skipped.
    """
    jobs = []
    for season in seasons:
        selected = sorted(season.episodes) if indices is None else indices
        for index in selected:
            job = Job(season, index, f"{season.path.parent.name}/{season.name}/{index}")
            jobs.append(job)
            if index not in season.episodes:
                job.error = f"episode {index} is not declared in {season.name}"
                continue
            if manifest is not None:
                _plan_from_manifest(job, manifest)
                continue
            try:
                job.start_ns = planned_start_ns(season, index, args, zone)
                job.episode_hash = episode_hash(job.start_ns)
            except (ConversionError, LeRobotError, OSError) as exc:
                job.error = str(exc)
    by_hash: dict[str, list[Job]] = {}
    for job in jobs:
        if job.episode_hash is not None and job.error is None and not job.skipped:
            by_hash.setdefault(job.episode_hash, []).append(job)
    for name, group in by_hash.items():
        if len(group) > 1:
            ids = [job.recording_id for job in group]
            for job in group:
                job.error = (
                    f"episode name {name} is shared by {ids}; the start times "
                    "coincide at millisecond resolution"
                )
            continue
        job = group[0]
        existing = args.output_dir / f"{name}.zarr"
        if existing.exists():
            try:
                owner = zarr.open_group(existing, mode="r").attrs.get(
                    "vendor_recording_id"
                )
            except Exception:  # an unreadable store is never ours to replace
                owner = None
            if owner != job.recording_id:
                job.error = (
                    f"{existing} already holds {owner or 'an unidentified episode'}; "
                    f"refusing to replace it with {job.recording_id}"
                )
            elif not args.overwrite:
                job.error = f"{existing} exists; pass --overwrite to replace it"
    return jobs


def _plan_from_manifest(job: Job, manifest) -> None:
    """Take a job's name and start from its manifest record."""
    record = manifest.get(job.recording_id)
    if record is None:
        job.error = f"{job.recording_id} is not in the manifest"
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
    if record.episode_hash != episode_hash(record.start_ns):
        job.error = f"manifest hash {record.episode_hash} does not match its start"
        return
    job.start_ns, job.episode_hash = record.start_ns, record.episode_hash


def planned_start_ns(season: Season, index: int, args, zone) -> int:
    """Return the episode's first-frame UTC time from metadata or the clock."""
    column = args.utc_timestamp_column
    if column:
        episode = load_episode(season, index, max_frames=1, columns=[column])
        if column in episode.columns:
            return int(vendor_clock(episode, column)[0])
    return synthesized_start_ns(season, index, zone)


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
# SharpaDex profile
# ---------------------------------------------------------------------------


def expected_vendor_names() -> dict[str, list[str]]:
    """Return the column names SharpaDex declares for the vectors it converts."""
    joints = []
    for side in ("left", "right"):
        joints += [f"{side}_arm_j{j}" for j in range(7)]
        joints += [f"{side}_hand_j{j}" for j in range(22)]
    joints += [f"motor_j{j}" for j in range(7)]
    tactile = [
        f"{side}_{finger}_{channel}"
        for side in ("left", "right")
        for finger in FINGERS
        for channel in TACTILE_CHANNELS
    ]
    tcp = [f"{side}_arm_tcp_pose_{j}" for side in ("left", "right") for j in range(6)]
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
    """Split the 60-column tactile vector into ``{side}.obs_tactile`` arrays."""
    values = np.asarray(values)
    width = len(FINGERS) * len(TACTILE_CHANNELS)
    if values.ndim != 2 or values.shape[1] != 2 * width:
        raise ConversionError(f"observation.tactile has shape {values.shape}")
    return {
        "left.obs_tactile": np.ascontiguousarray(values[:, :width]),
        "right.obs_tactile": np.ascontiguousarray(values[:, width:]),
    }


def encode_tactile_images(episode: Episode, quality: int, threads=None):
    """Decode ``tactile_deform`` and JPEG-encode one grayscale strip per hand.

    Returns:
        ``{"left.obs_tactile_image": (jpegs, [H, W, 1]), "right...": ...}``.

    Raises:
        ConversionError: If a frame is not the 2 x 5 grayscale mosaic.
    """
    out = {side: np.empty(episode.length, dtype=object) for side in ("left", "right")}
    shape = [TACTILE_TILE, len(FINGERS) * TACTILE_TILE, 1]
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
        gray = np.round(wide.mean(axis=2)).astype(np.uint8)[..., None]
        for row, side in enumerate(("left", "right")):
            strip = np.ascontiguousarray(
                gray[row * TACTILE_TILE : (row + 1) * TACTILE_TILE]
            )
            out[side][i] = simplejpeg.encode_jpeg(
                strip, quality=quality, colorspace="GRAY"
            )
    return {f"{side}.obs_tactile_image": (jpegs, shape) for side, jpegs in out.items()}


def _clean_text(text: str) -> tuple[str, bool]:
    """Drop the banned delimiter's metadata tail; report whether any was cut."""
    text = str(text).strip()
    if BANNED_DELIMITER not in text:
        return text, False
    return text.split(BANNED_DELIMITER)[0].strip(), True


def episode_segments(annotation: dict, length: int):
    """Return subtask and skill spans from one SharpaDex annotation row.

    SharpaDex spans are ``[start_step, end_step]`` inclusive and the text
    ends in `` | Skill: <skill>``. EgoVerse spans are ``[start, end)`` and the
    text carries no delimiter-encoded metadata, so the skill moves to its own
    span list. Spans are clipped to the retained ``length``.

    Returns:
        ``(subtasks, skills, notes)``; each span list holds
        ``(text, start, end)`` tuples.
    """
    subtasks, skills, notes = [], [], []
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
        text, cut = _clean_text(text)
        if cut:
            notes.append(
                f"{segment.get('id')}: dropped text after {BANNED_DELIMITER!r}"
            )
        skill = (segment.get("task_skill") or skill or "").strip()
        if text:
            subtasks.append((text, start, end))
        if skill:
            skills.append((_clean_text(skill)[0], start, end))
    return subtasks, skills, notes


def dataset_license(season: Season, levels: int = 1) -> str | None:
    """Return ``license`` from a dataset README's YAML front matter.

    The README beside the seasons is read first, then one in the season
    itself. ``levels=2`` also reads a release README above task folders.
    """
    folders = [season.path.parent, season.path, *list(season.path.parents)[1:levels]]
    for folder in folders:
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
    policy: str,
    *,
    has_front: bool,
    front_calibrated: bool,
    head_pose: bool,
    synthesized: bool,
    excursions: bool,
    gross: bool,
    clipped: bool,
    truncated: tuple[int, int] | None,
    tcp_problem: str | None,
) -> tuple[list[str], list[str] | None]:
    """Return ``(reasons, limitations)`` for an episode's data status.

    Any reason makes the episode a ``structural_sample``. ``strict`` treats
    every gap as a reason and returns no limitations. ``provenance`` keeps
    only defects in the stored data as reasons (a missing ego video, a gross
    observed joint excursion, clipping, truncation, a failed TCP check) and
    lists what the delivery does not provide (calibration, a measured clock,
    a head camera pose, within-tier joint excursions) as limitations.
    """
    if policy == "strict":
        reasons = []
        if not has_front:
            reasons.append("no head_left (front_1) video")
        elif not front_calibrated:
            reasons.append("no measured front_1 calibration (--calibration)")
        if synthesized:
            reasons.append(
                "capture clock synthesized from the season name and frame index"
            )
        if excursions and not clipped:
            reasons.append("joint values outside the registry/URDF limits")
        if clipped and excursions:
            reasons.append("joint values clipped to limits (--clip-joint-limits)")
        if truncated:
            reasons.append(f"truncated to {truncated[0]} of {truncated[1]} frames")
        if tcp_problem:
            reasons.append(tcp_problem)
        return reasons, None
    if policy != "provenance":
        raise ValueError(f"unknown status policy {policy!r}")
    reasons, limitations = [], []
    if not has_front:
        reasons.append("no head_left (front_1) video")
    elif not front_calibrated:
        limitations.append("no measured front_1 calibration")
    if not head_pose:
        limitations.append("no obs_head_pose: no measured head camera mount")
    if synthesized:
        limitations.append(
            "capture clock synthesized from the season name and frame index"
        )
    if excursions and not clipped:
        if gross:
            reasons.append(
                "observed hand joints exceed the validator's joint_limit_gross margin"
            )
        else:
            limitations.append(
                "joint values outside the registry/URDF limits, inside the "
                "validator's error margin"
            )
    if clipped and excursions:
        reasons.append("joint values clipped to limits (--clip-joint-limits)")
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


def convert_episode(job: Job, args, kinematics, calibration_rigs, zone) -> dict:
    """Write one episode and its sidecars; return the conversion summary."""
    season = job.season
    started = time.monotonic()
    sharpadex = args.profile == "sharpadex"
    record = job.record
    if sharpadex:
        check_vendor_layout(season)
    episode = load_episode(season, job.index, args.max_frames)
    columns = episode.columns
    required = ["observation.state", "action"]
    if sharpadex:
        required.append("observation.state.tcp_pose")
    if args.store_tactile:
        required.append("observation.tactile")
    for name in required:
        if name not in columns:
            raise ConversionError(f"{season.name}: column {name!r} is missing")

    # Clock: the manifest's start, a vendor UTC column, or the season synthesis.
    notes = []
    column = args.utc_timestamp_column
    if record is not None:
        steps = (np.arange(episode.length, dtype=np.int64) * 10**9) // season.fps
        stamps = record.start_ns + steps
        timestamp_source = record.clock
        if record.hash_offset_ms:
            notes.append(
                f"start moved {record.hash_offset_ms} ms to keep the episode name unique"
            )
    elif column and column in columns:
        stamps = vendor_clock(episode, column)
        timestamp_source = f"vendor_column:{column}"
    else:
        if column:
            notes.append(f"requested column {column!r} is absent; clock synthesized")
        stamps = synthesized_clock(season, job.index, episode.length, zone)
        timestamp_source = SYNTHESIZED
    if episode_hash(int(stamps[0])) != job.episode_hash:
        raise ConversionError("the episode clock changed after planning")

    # Report excursions on the source values, then store exactly what FK uses.
    source = {"obs": columns["observation.state"], "cmd": columns["action"]}
    source_excursions = {
        prefix: joint_limit_excursions(raw, prefix, kinematics)
        for prefix, raw in source.items()
    }
    stored = {
        prefix: clip_to_limits(raw, kinematics) if args.clip_joint_limits else raw
        for prefix, raw in source.items()
    }
    stored_excursions = {
        prefix: joint_limit_excursions(raw, prefix, kinematics)
        for prefix, raw in stored.items()
    }
    excursion_count = sum(len(r) for r in source_excursions.values())
    checked = [
        record
        for records in stored_excursions.values()
        for record in records
        if record["validator_checked"]
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
            if c.startswith("tactile")
            and not (args.store_tactile and c == "tactile_deform")
        ),
    }
    tactile_images = args.store_tactile and "tactile_deform" in season.cameras
    if tactile_images:
        camera_report["tactile_images"] = {
            "tactile_deform": ["left.obs_tactile_image", "right.obs_tactile_image"]
        }

    entries = {}
    if calibration_rigs is not None:
        rig = season.rig or ""
        entries = calibration_rigs.get(rig, calibration_rigs.get("default"))
        if entries is None:
            raise ConversionError(
                f"--calibration has no entry for rig {rig!r} and no 'default'"
            )
    for s in set(entries) & set(present):
        # Fail before decoding when info.json already shows a mismatch.
        declared = season.info["features"][f"observation.images.{s}"].get("shape")
        resolution = entries[s]["camera"].resolution
        if declared and resolution != (declared[1], declared[0]):
            raise ConversionError(
                f"calibration for {s} declares resolution {list(resolution)}, "
                f"info.json declares [{declared[1]}, {declared[0]}]"
            )
    parents = {entry["parent_link"] for s, entry in entries.items() if s in present}
    tcp_links = [TCP_LINK.format(side=side) for side in ("left", "right")]
    tracks, link_poses = derive_tracks(
        stored["obs"],
        stored["cmd"],
        kinematics=kinematics,
        links=(HEAD_LINK, *sorted(parents), *(tcp_links if sharpadex else ())),
        command_keypoints=args.command_keypoints,
    )
    numeric = dict(tracks)
    numeric["obs_rgb_timestamps_ns"] = stamps
    if args.keep_vendor_arrays:
        for name, key in VENDOR_ARRAYS.items():
            if name in columns:
                numeric[key] = columns[name]
    if args.store_tactile:
        numeric.update(split_tactile(columns["observation.tactile"]))
    stored_sources = set(VENDOR_ARRAYS) if args.keep_vendor_arrays else set()
    if args.store_tactile:
        stored_sources.add("observation.tactile")
    tcp = None
    if sharpadex:
        stored_sources |= set(SHARPADEX_VENDOR_ARRAYS)
        numeric["obs_head_link_pose"] = _matrix_to_xyzwxyz(link_poses[HEAD_LINK])
        for name, key in SHARPADEX_VENDOR_ARRAYS.items():
            if name in columns:
                numeric[key] = columns[name]
        tcp = tcp_side_check(columns["observation.state.tcp_pose"], link_poses)

    # Decode and JPEG-encode all streams; cameras decode concurrently.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial = args.output_dir / f".{job.episode_hash}.zarr.partial"
    preview_path = args.output_dir / f".{job.episode_hash}.mp4.partial"
    # Every stream decodes at once; split this process's CPUs between them so
    # FFmpeg does not start a thread per host CPU in each decoder.
    streams = max(1, len(present) + bool(tactile_images))
    threads = max(1, available_cpus() // streams)
    preview = (
        _Preview(preview_path, season.fps, threads=threads)
        if args.save_mp4 and "head_left" in present
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
    shapes = {key.removeprefix("images."): shape for key, (_, shape) in images.items()}

    calibration, mounts, head_pose = build_calibration(
        entries, present, link_poses, kinematics, shapes
    )
    if head_pose is not None:
        numeric["obs_head_pose"] = head_pose
    front_calibrated = calibration is not None and "front_1" in calibration["cameras"]

    tcp_problem = None
    if tcp is not None and not tcp["passed"]:
        tcp_problem = "arm TCP check failed: " + "; ".join(tcp["problems"])
        logger.warning("%s (%s): %s", job.episode_hash, job.recording_id, tcp_problem)
    reasons, limitations = status_reasons(
        args.status_policy,
        has_front=("front_1" in shapes),
        front_calibrated=front_calibrated,
        head_pose=head_pose is not None,
        synthesized=timestamp_source == SYNTHESIZED,
        excursions=bool(excursion_count),
        gross=bool(checked),
        clipped=bool(args.clip_joint_limits),
        truncated=(episode.length, episode.info.length) if episode.truncated else None,
        tcp_problem=tcp_problem,
    )
    status = DATA_STATUS_STRUCTURAL_SAMPLE if reasons else DATA_STATUS_COMPLETE

    annotations, skills, annotation_notes = [], [], []
    if sharpadex:
        row = _season_annotations(season).get(job.index, {})
        annotations, skills, annotation_notes = episode_segments(row, episode.length)

    resolved = kinematics.resolved
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
        "timestamps": {"source": timestamp_source, "notes": notes},
        "cameras": camera_report,
        "joint_limits": {
            "tolerance": 1e-6,
            "clipped": bool(args.clip_joint_limits and excursion_count),
            "source_excursions": source_excursions,
            "stored_excursions_checked_by_validator": len(checked),
        },
        "not_stored": {
            k: v
            for k, v in NOT_STORED.items()
            if k in columns and k not in stored_sources
        },
        "data_status": status,
        "data_status_reasons": reasons,
    }
    if limitations is not None:
        report["status_policy"] = args.status_policy
        report["data_limitations"] = limitations
    if tcp is not None:
        report["tcp_check"] = tcp
    if sharpadex:
        report["annotations"] = {
            "subtasks": len(annotations),
            "skills": len(skills),
            "notes": annotation_notes,
        }
    license_name = args.license or dataset_license(season, levels=2 if sharpadex else 1)
    if license_name is None:
        if sharpadex:
            raise ConversionError(
                f"{season.path}: no license in a README above the season; pass --license"
            )
        logger.warning("%s: no README license found; recording CC-BY-4.0", season.path)
        license_name = "CC-BY-4.0"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "morphology": {
            "platform": resolved.platform.name,
            "end_effector": {s: ee.name for s, ee in resolved.end_effectors.items()},
        },
        "action_semantics": "absolute_joint_position_setpoint",
        "vendor_recording_id": job.recording_id,
        "source_dataset": season.path.parent.name,
        "source_season": season.name,
        "source_episode_index": job.index,
        "source_lerobot_version": season.version,
        "source_robot_type": season.robot_type,
        "source_task": list(episode.info.tasks),
        "rig_name": season.rig,
        "license": license_name,
        "timestamp_source": timestamp_source,
        "source_utc_offset": args.source_utc_offset,
        "conversion_tool": TOOL,
        "conversion_tool_version": _git_sha() or "unknown",
        "conversion_report": report,
    }
    if mounts:
        metadata["camera_mounts"] = mounts
    if limitations is not None:
        metadata["status_policy"] = args.status_policy
        metadata["data_limitations"] = limitations
    if sharpadex:
        tags = dict(record.tags) if record is not None else {}
        header = getattr(args, "manifest_header", {}) or {}
        metadata.update(
            {
                "source_dataset": Path(header.get("release_root") or "").name
                or season.path.parent.parent.name,
                "source_task_folder": season.path.parent.name,
                "source_revision": header.get("source_revision") or "unknown",
                "source_uid": None if record is None else record.source_uid,
                "source_score": tags.get("score"),
                "source_task_name": tags.get("task_name"),
                "source_tags": tags,
                "start_source": None if record is None else record.start_source,
                "hash_offset_ms": 0 if record is None else record.hash_offset_ms,
                "quality_flags": [] if record is None else list(record.flags),
                "calibration_source": "none" if calibration is None else "measured",
            }
        )
    metadata = _jsonable(metadata)
    task_name, task_description = args.task_name, args.task_description
    if sharpadex:
        task_name = args.task_name or season.path.parent.name
        task_description = (
            args.task_description
            or _clean_text(
                (record.tags if record else {}).get("task_instruction") or ""
            )[0]
        )

    if partial.exists():
        shutil.rmtree(partial)
    writer = ZarrWriter(
        episode_path=partial,
        embodiment=EMBODIMENT,
        fps=season.fps,
        task_name=task_name,
        task_description=task_description,
        annotations=annotations,
        chunk_timesteps=args.chunk_timesteps,
        intrinsics=None
        if calibration is None
        else parse_calibration(calibration).intrinsics(),
        calibration=calibration,
        data_status=status,
    )
    writer.write(
        numeric_data=numeric,
        pre_encoded_image_data={**images, **tactile},
        metadata_override=metadata,
    )
    del images, tactile
    if skills:
        writer.append_annotations("annotations_skill", skills)

    # Declare joint order beside the writer's dtype/shape descriptions.
    store = zarr.open_group(partial, mode="a")
    features = dict(store.attrs["features"])
    names = {"aux_joints": list(resolved.platform.aux.joint_names)}
    for side, spec in resolved.end_effectors.items():
        names[f"{side}.hand_joints"] = list(spec.joint_names)
        names[f"{side}.joints"] = list(arm_joint_names(side))
    for prefix in ("obs", "cmd"):
        for key, joint_names in names.items():
            side, _, suffix = key.rpartition(".")
            array = f"{side}.{prefix}_{suffix}" if side else f"{prefix}_{suffix}"
            features[array] = {**features[array], "joint_names": joint_names}
    for name, key in {**VENDOR_ARRAYS, **SHARPADEX_VENDOR_ARRAYS}.items():
        if key in features:
            features[key] = {
                **features[key],
                "source_column": name,
                "contract": "vendor extension, not part of the EgoVerse contract",
            }
    for side, spec in resolved.end_effectors.items():
        key = f"{side}.obs_tactile"
        if key in features and spec.tactile is not None:
            features[key] = {
                **features[key],
                "source_column": "observation.tactile",
                "names": [
                    f"{f}_{c}" for f in FINGERS for c in spec.tactile.channel_names
                ],
                "units": [spec.tactile.unit(c) for c in range(spec.tactile.channels)]
                * spec.tactile.taxels,
            }
        key = f"{side}.obs_tactile_image"
        if key in features:
            features[key] = {
                **features[key],
                "source_column": "observation.images.tactile_deform",
                "colorspace": "GRAY",
                "tiles": list(FINGERS),
            }
    store.attrs["features"] = features

    final = args.output_dir / f"{job.episode_hash}.zarr"
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)
    if preview is not None:
        preview_path.rename(args.output_dir / f"{job.episode_hash}.mp4")

    summary = {
        "episode": str(final),
        "episode_hash": job.episode_hash,
        "vendor_recording_id": job.recording_id,
        "total_frames": episode.length,
        "data_status": status,
        "seconds": round(time.monotonic() - started, 1),
    }
    if checked and not args.clip_joint_limits:
        joints = sorted({f"{r['array']}:{r['joint']}" for r in checked})
        worst = max(max(r["max_above"], r["max_below"]) for r in checked)
        _warn_box(
            [
                f"{job.episode_hash} ({job.recording_id}): {len(joints)} observed "
                f"hand joint column(s) exceed the registry limits by up to "
                f"{worst:.4f} rad:",
                *[f"  {j}" for j in joints],
                "This episode FAILS the validator's joint_limit_gross integrity "
                "check: an excursion this",
                "large points at units, column order or corrupt frames. Raise it "
                "with Sharpa; --clip-joint-limits",
                "is for local experiments only. Details: "
                f"{job.episode_hash}.conversion.json",
            ]
        )
    if args.validate:
        from egomimic.rldb.zarr.validate import validate_episode

        validation = validate_episode(final, ego_overlay=front_calibrated)
        (args.output_dir / f"{job.episode_hash}.validation.json").write_text(
            json.dumps(validation.to_jsonable(), indent=2) + "\n"
        )
        summary["validation"] = {
            "ok": validation.ok,
            "errors": [str(f) for f in validation.errors],
            "warnings": len(validation.warnings),
        }
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
        help="Folder of season_* directories, or one season.",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lerobot-version", choices=("auto", "v2.1", "v3.0"), default="auto"
    )
    parser.add_argument(
        "--profile",
        choices=("default", "sharpadex"),
        default="default",
        help="sharpadex: a SharpaDex release planned by sharpadex_manifest "
        "(implies --store-tactile --command-keypoints; needs --manifest).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="sharpadex_manifest output; names, clocks and selects episodes.",
    )
    parser.add_argument(
        "--tasks",
        default="*",
        help="Glob over task folder names (sharpadex profile; default: all).",
    )
    parser.add_argument(
        "--status-policy",
        choices=("strict", "provenance"),
        default="strict",
        help="strict: any gap makes a structural_sample. provenance: only "
        "defects do; missing calibration, a synthesized clock and in-tier "
        "joint excursions are recorded as data_limitations.",
    )
    parser.add_argument(
        "--store-tactile",
        action="store_true",
        help="Store {side}.obs_tactile and, from tactile_deform, "
        "{side}.obs_tactile_image.",
    )
    parser.add_argument(
        "--command-keypoints",
        action="store_true",
        help="Also store {side}.cmd_hand_keypoints (FK of the commands).",
    )
    parser.add_argument(
        "--task-name",
        help="EgoVerse task name; the source task is a placeholder. Required "
        "unless --profile sharpadex, which uses the task folder.",
    )
    parser.add_argument("--task-description", default="")
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Rig calibration YAML (see calibration_template.yaml).",
    )
    parser.add_argument(
        "--source-utc-offset",
        default="+08:00",
        help="UTC offset of the local season-folder times.",
    )
    parser.add_argument(
        "--utc-timestamp-column",
        help="Parquet column of int64 UTC capture nanoseconds.",
    )
    parser.add_argument(
        "--include-head-right",
        action="store_true",
        help="Also store head_right as images.head_right.",
    )
    parser.add_argument(
        "--keep-vendor-arrays",
        action="store_true",
        help="Store raw state/action/torque/TCP/tactile as sharpa.*.",
    )
    parser.add_argument(
        "--clip-joint-limits",
        action="store_true",
        help="Clip joints to limits before FK (local experiments only).",
    )
    parser.add_argument(
        "--max-frames", type=int, help="Keep only the first N frames (smoke runs)."
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--chunk-timesteps", type=int, default=100)
    parser.add_argument(
        "--license", help="Override the license read from the dataset README."
    )
    parser.add_argument(
        "--save-mp4", action="store_true", help="Write a <hash>.mp4 preview of front_1."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each output; write <hash>.validation.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output of the same recording.",
    )
    return parser


def main(argv=None) -> int:
    """Convert the selected episodes.

    Returns:
        ``0`` on success, ``1`` if any episode failed to convert, ``2`` if all
        converted but ``--validate`` reported errors.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be in 1..100")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("--max-frames must be positive")
    if args.chunk_timesteps < 1:
        raise SystemExit("--chunk-timesteps must be positive")
    manifest = None
    if args.profile == "sharpadex":
        args.store_tactile = args.command_keypoints = True
        if args.manifest is None:
            raise SystemExit("--profile sharpadex needs --manifest")
        from egomimic.scripts.sharpa_process.sharpadex_manifest import (
            ManifestError,
            read_manifest,
        )

        try:
            args.manifest_header, manifest = read_manifest(args.manifest)
        except (ManifestError, OSError, ValueError) as exc:
            raise SystemExit(f"sharpa_to_zarr: {exc}") from exc
    elif args.task_name is None:
        raise SystemExit("--task-name is required")
    elif args.manifest is not None:
        raise SystemExit("--manifest needs --profile sharpadex")
    import torch

    # Small 4x4 batches gain nothing from many threads; on a shared host,
    # oversubscription can slow forward kinematics by orders of magnitude.
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    try:
        zone = parse_utc_offset(args.source_utc_offset)
        kinematics = SharpaKinematics()
        rigs = (
            None
            if args.calibration is None
            else load_calibration_file(args.calibration, kinematics)
        )
        if args.profile == "sharpadex":
            seasons = discover_task_seasons(
                args.dataset_root, args.seasons, args.lerobot_version, args.tasks
            )
        else:
            seasons = discover_seasons(
                args.dataset_root, args.seasons, args.lerobot_version
            )
    except (ConversionError, LeRobotError, OSError) as exc:
        raise SystemExit(f"sharpa_to_zarr: {exc}") from exc
    if not seasons:
        raise SystemExit(
            f"sharpa_to_zarr: no {args.lerobot_version} season matches "
            f"{args.seasons!r} under {args.dataset_root}"
        )
    jobs = plan_jobs(seasons, args.episodes, args, zone, manifest)
    logger.info("%d episode(s) from %d season(s)", len(jobs), len(seasons))
    converted, failed, skipped = [], [], []
    for job in jobs:
        if job.skipped and job.error is None:
            skipped.append(job)
            logger.info("skipped %s: %s", job.recording_id, job.skipped)
            continue
        if job.error is None:
            try:
                summary = convert_episode(job, args, kinematics, rigs, zone)
                converted.append(summary)
                logger.info(
                    "%s <- %s: %d frames, %s, %.1f s",
                    summary["episode_hash"],
                    job.recording_id,
                    summary["total_frames"],
                    summary["data_status"],
                    summary["seconds"],
                )
                for error in summary.get("validation", {}).get("errors", []):
                    logger.warning("  validation: %s", error)
                continue
            except (ConversionError, LeRobotError, CalibrationError, OSError) as exc:
                job.error = str(exc)
            except Exception as exc:  # report and continue with other episodes
                logger.exception("unexpected failure converting %s", job.recording_id)
                job.error = f"{type(exc).__name__}: {exc}"
            _discard_partials(args.output_dir, job.episode_hash)
        failed.append(job)
        logger.error("FAILED %s: %s", job.recording_id, job.error)
    invalid = [s for s in converted if not s.get("validation", {}).get("ok", True)]
    print(
        f"Converted {len(converted)} episode(s), {len(failed)} failed"
        + (f", {len(skipped)} skipped by the manifest" if manifest is not None else "")
        + (f", {len(invalid)} with validation errors" if args.validate else "")
        + f"; output in {args.output_dir}"
    )
    if failed:
        return 1
    return 2 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
