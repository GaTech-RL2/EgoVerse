"""Read Sharpa LeRobot seasons (v2.1 and v3.0) as NumPy arrays and RGB frames.

Sharpa publishes each recording session ("season") as its own LeRobot dataset::

    <dataset_root>/season_POC<rig>_<YYYY_MM_DD_HH_MM_SS>_train/
        lerobotv2.1/   meta/info.json, data/chunk-000/episode_000000.parquet,
                       videos/chunk-000/<video_key>/episode_000000.mp4
        lerobot3.0/    meta/info.json, data/chunk-000/file-000.parquet,
                       videos/<video_key>/chunk-000/file-000.mp4

A season directory may also be a LeRobot root itself. ``meta/info.json``
``codebase_version`` selects the layout. A v3.0 data or video file holds
several episodes; ``meta/episodes`` locates each episode's rows and its
``[from_timestamp, to_timestamp)`` video span.

This module has no Zarr dependency. Decoded frame counts must equal the parquet
row counts; any disagreement raises ``LeRobotError``.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

VERSIONS = ("v2.1", "v3.0")
#: Season subdirectories holding each layout, in ``auto`` preference order.
VERSION_DIRS = {"v2.1": "lerobotv2.1", "v3.0": "lerobot3.0"}
VIDEO_PREFIX = "observation.images."
SEASON_NAME = re.compile(
    r"^season_(?P<rig>POC[0-9A-Za-z]+)_(?P<start>\d{4}(?:_\d{2}){5})(?:_.*)?$"
)


class LeRobotError(ValueError):
    """A LeRobot dataset is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class VideoSpan:
    """Where one episode's frames sit in a shared v3.0 video file."""

    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class EpisodeInfo:
    """One row of a season's episode metadata.

    Attributes:
        index: ``episode_index``.
        length: Frame count declared by the metadata.
        tasks: Task strings declared for the episode.
        frame_offset: Frames recorded in this season before this episode,
            from the declared lengths of all lower episode indices.
        data_file: v3.0 ``(chunk_index, file_index)`` of the episode's rows.
        dataset_from_index: v3.0 global index of the episode's first row.
        videos: v3.0 video spans keyed by short camera name.
    """

    index: int
    length: int
    tasks: tuple[str, ...]
    frame_offset: int
    data_file: tuple[int, int] | None = None
    dataset_from_index: int | None = None
    videos: Mapping[str, VideoSpan] = field(default_factory=dict)


@dataclass(frozen=True)
class Season:
    """A LeRobot dataset for one Sharpa recording session.

    Attributes:
        name: The season directory name.
        path: The season directory.
        root: The LeRobot root holding ``meta/info.json``.
        version: ``v2.1`` or ``v3.0``.
        info: Parsed ``meta/info.json``.
        episodes: Episode metadata keyed by ``episode_index``.
        tasks: Task strings keyed by ``task_index``.
        rig: Rig name parsed from the directory name, e.g. ``POC22002``.
        start_local: Session start parsed from the directory name, in the
            recording site's local time (no zone).
    """

    name: str
    path: Path
    root: Path
    version: str
    info: Mapping
    episodes: Mapping[int, EpisodeInfo]
    tasks: Mapping[int, str]
    rig: str | None
    start_local: datetime | None

    @property
    def fps(self) -> int:
        fps = self.info.get("fps")
        if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
            raise LeRobotError(f"{self.root}: info.json fps must be a positive int")
        return fps

    @property
    def robot_type(self) -> str | None:
        return self.info.get("robot_type")

    @property
    def cameras(self) -> tuple[str, ...]:
        """Short names of the declared video streams, e.g. ``head_left``."""
        return tuple(
            key.removeprefix(VIDEO_PREFIX)
            for key, feature in self.info.get("features", {}).items()
            if key.startswith(VIDEO_PREFIX) and feature.get("dtype") == "video"
        )


@dataclass
class Episode:
    """One episode's parquet columns and lazily decoded video frames.

    Attributes:
        season: The season holding the episode.
        info: The episode's metadata row.
        columns: Parquet columns. List columns are ``(T, width)`` arrays and
            scalar columns are ``(T,)`` arrays, truncated to ``length`` rows.
        length: Retained frames (``info.length`` unless truncated).
    """

    season: Season
    info: EpisodeInfo
    columns: dict[str, np.ndarray]
    length: int

    @property
    def truncated(self) -> bool:
        return self.length < self.info.length

    def video_path(self, camera: str) -> Path:
        """Return the episode's video file for a declared camera."""
        season, key = self.season, VIDEO_PREFIX + camera
        if camera not in season.cameras:
            raise LeRobotError(f"{season.name}: camera {camera!r} is not declared")
        template = season.info["video_path"]
        if season.version == "v2.1":
            relative = template.format(
                episode_chunk=self.info.index // _chunks_size(season),
                video_key=key,
                episode_index=self.info.index,
            )
        else:
            span = self.info.videos.get(camera)
            if span is None:
                raise LeRobotError(
                    f"{season.name}: episode {self.info.index} has no {camera} span"
                )
            relative = template.format(
                video_key=key,
                chunk_index=span.chunk_index,
                file_index=span.file_index,
            )
        path = season.root / relative
        if not path.is_file():
            raise LeRobotError(f"{path}: declared video file is missing")
        return path

    def frames(self, camera: str) -> Iterator[np.ndarray]:
        """Yield exactly ``length`` ``(H, W, 3)`` uint8 RGB frames.

        A full v2.1 episode must decode to exactly its parquet row count. A
        v3.0 episode must start at ``from_timestamp`` and contain its row
        count before ``to_timestamp``. Truncated reads compare the container's
        declared frame count instead, when it states one.
        """
        path = self.video_path(camera)
        span = self.info.videos.get(camera) if self.season.version == "v3.0" else None
        return decode_frames(
            path,
            count=self.length,
            fps=self.season.fps,
            expected=self.info.length,
            start=None if span is None else span.from_timestamp,
            end=None if span is None else span.to_timestamp,
        )


def decode_frames(
    path: Path,
    *,
    count: int,
    fps: int,
    expected: int,
    start: float | None = None,
    end: float | None = None,
) -> Iterator[np.ndarray]:
    """Decode ``count`` RGB frames, checking them against ``expected`` rows.

    Args:
        path: The MP4 file (H.264, AV1, or any codec PyAV decodes).
        count: Frames to yield; at most ``expected``.
        fps: Nominal frame rate, used for half-frame timestamp tolerance.
        expected: The episode's full parquet row count.
        start: v3.0 span start in seconds; ``None`` reads from the beginning.
        end: v3.0 span end in seconds, exclusive.

    Raises:
        LeRobotError: If decoding yields a different number of frames than
            the parquet rows, or the first frame is not at ``start``.
    """
    half = 0.5 / fps
    if end is not None and round((end - (start or 0.0)) * fps) != expected:
        raise LeRobotError(
            f"{path}: span [{start}, {end}) s holds "
            f"{(end - (start or 0.0)) * fps:.2f} frames at {fps} fps, parquet has "
            f"{expected} rows"
        )
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if start is None and count < expected and stream.frames not in (0, expected):
            raise LeRobotError(
                f"{path}: container declares {stream.frames} frames, parquet has "
                f"{expected} rows"
            )
        if start:
            container.seek(int(start / stream.time_base), stream=stream, backward=True)
        produced = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise LeRobotError(f"{path}: decoded a frame without a timestamp")
            seconds = float(frame.pts * stream.time_base)
            if start is not None:
                if seconds < start - half:
                    continue
                if produced == 0 and seconds > start + half:
                    raise LeRobotError(
                        f"{path}: first frame at {seconds:.4f} s, span starts at "
                        f"{start:.4f} s"
                    )
            if produced == count:
                # A truncated read stops here; a full read must end exactly.
                if count == expected and (end is None or seconds < end - half):
                    raise LeRobotError(
                        f"{path}: more than {expected} frames decode for an episode "
                        f"with {expected} parquet rows"
                    )
                return
            yield frame.to_ndarray(format="rgb24")
            produced += 1
    if produced < count:
        raise LeRobotError(
            f"{path}: decoded {produced} frames, parquet has {expected} rows"
        )


def parse_season_name(name: str) -> tuple[str | None, datetime | None]:
    """Return ``(rig, local start)`` from ``season_POC<rig>_<Y_m_d_H_M_S>...``."""
    match = SEASON_NAME.match(name)
    if match is None:
        return None, None
    return match["rig"], datetime.strptime(match["start"], "%Y_%m_%d_%H_%M_%S")


def _read_info(root: Path) -> dict:
    try:
        return json.loads((root / "meta" / "info.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LeRobotError(f"{root}: cannot read meta/info.json ({exc})") from exc


def _chunks_size(season: Season) -> int:
    size = season.info.get("chunks_size", 1000)
    if not isinstance(size, int) or size <= 0:
        raise LeRobotError(f"{season.root}: chunks_size must be a positive int")
    return size


def find_root(season_dir: Path, version: str = "auto") -> Path | None:
    """Return the LeRobot root to read for a season directory, if any.

    ``auto`` prefers a root at the season directory itself, then the v2.1
    subdirectory, then the v3.0 subdirectory.
    """
    if version not in ("auto", *VERSIONS):
        raise ValueError(f"version must be auto, v2.1 or v3.0, got {version!r}")
    candidates = [season_dir] + [
        season_dir / VERSION_DIRS[v]
        for v in (VERSIONS if version == "auto" else (version,))
    ]
    for root in candidates:
        if not (root / "meta" / "info.json").is_file():
            continue
        if version == "auto" or _read_info(root).get("codebase_version") == version:
            return root
    return None


def _v21_episodes(root: Path) -> list[dict]:
    path = root / "meta" / "episodes.jsonl"
    try:
        lines = path.read_text().splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise LeRobotError(f"{path}: cannot read episode metadata ({exc})") from exc


def _v21_tasks(root: Path) -> dict[int, str]:
    path = root / "meta" / "tasks.jsonl"
    if not path.is_file():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {int(row["task_index"]): str(row["task"]) for row in rows}


def _v30_episodes(root: Path) -> list[dict]:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise LeRobotError(f"{root}: no meta/episodes/chunk-*/file-*.parquet")
    rows = []
    for path in files:
        names = [n for n in pq.read_schema(path).names if not n.startswith("stats/")]
        rows.extend(pq.read_table(path, columns=names).to_pylist())
    for row in rows:
        row["videos"] = {
            key.removeprefix(f"videos/{VIDEO_PREFIX}").removesuffix("/chunk_index"): (
                VideoSpan(
                    int(row[key]),
                    int(row[key.replace("/chunk_index", "/file_index")]),
                    float(row[key.replace("/chunk_index", "/from_timestamp")]),
                    float(row[key.replace("/chunk_index", "/to_timestamp")]),
                )
            )
            for key in list(row)
            if key.startswith("videos/") and key.endswith("/chunk_index")
        }
    return rows


def _v30_tasks(root: Path) -> dict[int, str]:
    path = root / "meta" / "tasks.parquet"
    if not path.is_file():
        return {}
    table = pq.read_table(path).to_pydict()
    return dict(zip(map(int, table["task_index"]), map(str, table["task"])))


def open_season(season_dir: Path, version: str = "auto") -> Season:
    """Read one season's metadata.

    Args:
        season_dir: A season directory, or a LeRobot root.
        version: ``auto``, ``v2.1`` or ``v3.0``.

    Raises:
        LeRobotError: If no matching LeRobot root exists or its metadata is
            malformed.
    """
    season_dir = Path(season_dir)
    root = find_root(season_dir, version)
    if root is None:
        raise LeRobotError(f"{season_dir}: no LeRobot {version} dataset found")
    info = _read_info(root)
    found = info.get("codebase_version")
    if found not in VERSIONS:
        raise LeRobotError(f"{root}: unsupported codebase_version {found!r}")
    rows = _v21_episodes(root) if found == "v2.1" else _v30_episodes(root)
    tasks = _v21_tasks(root) if found == "v2.1" else _v30_tasks(root)
    episodes = {}
    offset = 0
    for row in sorted(rows, key=lambda r: int(r["episode_index"])):
        index, length = int(row["episode_index"]), int(row["length"])
        if index in episodes or length <= 0:
            raise LeRobotError(f"{root}: duplicate or empty episode {index}")
        data_file = None
        if found == "v3.0":
            data_file = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        episodes[index] = EpisodeInfo(
            index=index,
            length=length,
            tasks=tuple(row.get("tasks") or ()),
            frame_offset=offset,
            data_file=data_file,
            dataset_from_index=row.get("dataset_from_index"),
            videos=row.get("videos", {}),
        )
        offset += length
    name = season_dir.name
    rig, start = parse_season_name(name)
    return Season(name, season_dir, root, found, info, episodes, tasks, rig, start)


def _is_season(path: Path) -> bool:
    return find_root(path) is not None


def discover_seasons(
    dataset_root: Path, pattern: str = "*", version: str = "auto"
) -> list[Season]:
    """Open every season under ``dataset_root`` whose name matches ``pattern``.

    ``dataset_root`` may itself be one season. Directories without a matching
    LeRobot root are skipped.
    """
    dataset_root = Path(dataset_root)
    if _is_season(dataset_root):
        candidates = [dataset_root]
    else:
        candidates = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    seasons = []
    for path in candidates:
        if fnmatch.fnmatchcase(path.name, pattern) and find_root(path, version):
            seasons.append(open_season(path, version))
    return seasons


def _column_array(table: pa.Table, name: str) -> np.ndarray:
    """Return a fixed-width list column as ``(T, width)`` or a scalar as ``(T,)``."""
    column = table.column(name)
    if column.null_count:
        raise LeRobotError(f"column {name!r} has {column.null_count} null rows")
    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        lengths = pc.list_value_length(column)
        width = pc.min_max(lengths).as_py()
        if width["min"] != width["max"]:
            raise LeRobotError(f"column {name!r} has rows of different widths")
        values = pc.list_flatten(column).to_numpy(zero_copy_only=False)
        return values.reshape(table.num_rows, width["min"] or 0)
    if pa.types.is_fixed_size_list(column.type):
        values = pc.list_flatten(column).to_numpy(zero_copy_only=False)
        return values.reshape(table.num_rows, column.type.list_size)
    return column.to_numpy()


#: Columns read with any subset, for the row-consistency checks.
_BOOKKEEPING = ("frame_index", "episode_index", "index")


def load_episode(
    season: Season,
    index: int,
    max_frames: int | None = None,
    columns: Sequence[str] | None = None,
) -> Episode:
    """Load one episode's parquet rows, optionally keeping the first frames.

    Args:
        season: The season to read.
        index: The ``episode_index``.
        max_frames: Keep at most this many leading frames.
        columns: Read only these columns (when present) and the bookkeeping
            columns; ``None`` reads every column.

    Raises:
        LeRobotError: If the episode is undeclared, its file is missing, or its
            rows disagree with the metadata (count, order, or global index).
    """
    info = season.episodes.get(index)
    if info is None:
        raise LeRobotError(f"{season.name}: episode {index} is not declared")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    template = season.info["data_path"]
    if season.version == "v2.1":
        relative = template.format(
            episode_chunk=index // _chunks_size(season), episode_index=index
        )
        filters = None
    else:
        chunk, file = info.data_file
        relative = template.format(chunk_index=chunk, file_index=file)
        filters = [("episode_index", "==", index)]
    path = season.root / relative
    if not path.is_file():
        raise LeRobotError(f"{path}: episode data file is missing")
    if columns is not None:
        available = pq.read_schema(path).names
        wanted = set(columns) | set(_BOOKKEEPING)
        columns = [name for name in available if name in wanted]
    table = pq.read_table(path, columns=columns, filters=filters)
    if "frame_index" in table.column_names:
        table = table.sort_by("frame_index")
    if table.num_rows != info.length:
        raise LeRobotError(
            f"{path}: episode {index} has {table.num_rows} rows, metadata declares "
            f"{info.length}"
        )
    arrays = {name: _column_array(table, name) for name in table.column_names}
    if "frame_index" in arrays and not np.array_equal(
        arrays["frame_index"], np.arange(info.length)
    ):
        raise LeRobotError(f"{path}: frame_index is not 0..{info.length - 1}")
    if "episode_index" in arrays and (arrays["episode_index"] != index).any():
        raise LeRobotError(f"{path}: rows from another episode_index")
    # The synthesized clock places episodes by their frame offset; the stored
    # global index and v3.0 metadata must agree with it.
    for name, first in (
        ("index", arrays["index"][0] if "index" in arrays else None),
        ("dataset_from_index", info.dataset_from_index),
    ):
        if first is not None and int(first) != info.frame_offset:
            raise LeRobotError(
                f"{path}: episode {index} {name} starts at {int(first)}, but "
                f"{info.frame_offset} frames precede it in the episode metadata"
            )
    length = info.length if max_frames is None else min(info.length, max_frames)
    arrays = {name: values[:length] for name, values in arrays.items()}
    return Episode(season=season, info=info, columns=arrays, length=length)


__all__ = [
    "Episode",
    "EpisodeInfo",
    "LeRobotError",
    "Season",
    "VideoSpan",
    "decode_frames",
    "discover_seasons",
    "find_root",
    "load_episode",
    "open_season",
    "parse_season_name",
]
