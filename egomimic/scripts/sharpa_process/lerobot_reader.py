"""Read SharpaDex LeRobot v2.1 seasons as NumPy arrays and RGB frames.

A SharpaDex release nests one LeRobot v2.1 dataset per recording session
("season") under its task folder::

    <release_root>/<task>/season_POC<rig>_<YYYY_MM_DD_HH_MM_SS>_train/
        lerobot_v2.1/   meta/{info.json,episodes.jsonl,annotations.jsonl,...}
                        data/chunk-000/episode_000000.parquet
                        videos/chunk-000/<video_key>/episode_000000.mp4

Releases also carry a ``lerobot_v3.0`` export of the same episodes, without
annotations; it is not read.

This module has no Zarr dependency. Decoded frame counts must equal the parquet
row counts; any disagreement raises ``LeRobotError``.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

VERSION = "v2.1"
#: Season subdirectory spellings of a v2.1 export, in preference order.
VERSION_DIRS = ("lerobot_v2.1", "lerobotv2.1")
VIDEO_PREFIX = "observation.images."
SEASON_NAME = re.compile(
    r"^season_(?P<rig>POC[0-9A-Za-z]+)_(?P<start>\d{4}(?:_\d{2}){5})(?:_.*)?$"
)


class LeRobotError(ValueError):
    """A LeRobot dataset is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class EpisodeInfo:
    """One row of a season's episode metadata.

    Attributes:
        index: ``episode_index``.
        length: Frame count declared by the metadata.
        tasks: Task strings declared for the episode.
        frame_offset: Frames recorded in this season before this episode,
            from the declared lengths of all lower episode indices.
    """

    index: int
    length: int
    tasks: tuple[str, ...]
    frame_offset: int


@dataclass(frozen=True)
class Season:
    """A LeRobot v2.1 dataset for one Sharpa recording session.

    Attributes:
        name: The season directory name.
        path: The season directory.
        root: The LeRobot root holding ``meta/info.json``.
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
    info: Mapping
    episodes: Mapping[int, EpisodeInfo]
    tasks: Mapping[int, str]
    rig: str | None
    start_local: datetime | None

    version = VERSION

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
        season = self.season
        if camera not in season.cameras:
            raise LeRobotError(f"{season.name}: camera {camera!r} is not declared")
        relative = season.info["video_path"].format(
            episode_chunk=self.info.index // _chunks_size(season),
            video_key=VIDEO_PREFIX + camera,
            episode_index=self.info.index,
        )
        path = season.root / relative
        if not path.is_file():
            raise LeRobotError(f"{path}: declared video file is missing")
        return path

    def frames(self, camera: str, threads: int | None = None) -> Iterator[np.ndarray]:
        """Yield exactly ``length`` ``(H, W, 3)`` uint8 RGB frames.

        ``threads`` caps the decoder's threads; ``None`` lets FFmpeg choose,
        which sizes to every CPU on the host rather than to a job's share.

        A full episode must decode to exactly its parquet row count. Truncated
        reads compare the container's declared frame count instead, when it
        states one.
        """
        return decode_frames(
            self.video_path(camera),
            count=self.length,
            expected=self.info.length,
            threads=threads,
        )


def decode_frames(
    path: Path, *, count: int, expected: int, threads: int | None = None
) -> Iterator[np.ndarray]:
    """Decode ``count`` RGB frames, checking them against ``expected`` rows.

    Args:
        path: The MP4 file (H.264, AV1, or any codec PyAV decodes).
        count: Frames to yield; at most ``expected``.
        expected: The episode's full parquet row count.
        threads: Decoder thread count; ``None`` lets FFmpeg choose.

    Raises:
        LeRobotError: If decoding yields a different number of frames than
            the parquet rows.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if threads:
            stream.thread_count = threads
        if count < expected and stream.frames not in (0, expected):
            raise LeRobotError(
                f"{path}: container declares {stream.frames} frames, parquet has "
                f"{expected} rows"
            )
        produced = 0
        for frame in container.decode(stream):
            if produced == count:
                # A truncated read stops here; a full read must end exactly.
                if count == expected:
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


def _jsonl(path: Path, what: str) -> list[dict]:
    try:
        lines = path.read_text().splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise LeRobotError(f"{path}: cannot read {what} ({exc})") from exc


def _chunks_size(season: Season) -> int:
    size = season.info.get("chunks_size", 1000)
    if not isinstance(size, int) or size <= 0:
        raise LeRobotError(f"{season.root}: chunks_size must be a positive int")
    return size


def find_root(season_dir: Path) -> Path | None:
    """Return the season's LeRobot v2.1 root, if it has one."""
    for name in VERSION_DIRS:
        root = Path(season_dir) / name
        if (root / "meta" / "info.json").is_file():
            return root
    return None


def open_season(season_dir: Path) -> Season:
    """Read one season's metadata.

    Raises:
        LeRobotError: If the season has no v2.1 export or its metadata is
            malformed.
    """
    season_dir = Path(season_dir)
    root = find_root(season_dir)
    if root is None:
        raise LeRobotError(f"{season_dir}: no LeRobot {VERSION} dataset found")
    info = _read_info(root)
    found = info.get("codebase_version")
    if found != VERSION:
        raise LeRobotError(f"{root}: codebase_version {found!r}, expected {VERSION}")
    rows = _jsonl(root / "meta" / "episodes.jsonl", "episode metadata")
    tasks_path = root / "meta" / "tasks.jsonl"
    tasks = {}
    if tasks_path.is_file():
        tasks = {
            int(row["task_index"]): str(row["task"])
            for row in _jsonl(tasks_path, "tasks")
        }
    episodes = {}
    offset = 0
    for row in sorted(rows, key=lambda r: int(r["episode_index"])):
        index, length = int(row["episode_index"]), int(row["length"])
        if index in episodes or length <= 0:
            raise LeRobotError(f"{root}: duplicate or empty episode {index}")
        episodes[index] = EpisodeInfo(
            index=index,
            length=length,
            tasks=tuple(row.get("tasks") or ()),
            frame_offset=offset,
        )
        offset += length
    rig, start = parse_season_name(season_dir.name)
    return Season(season_dir.name, season_dir, root, info, episodes, tasks, rig, start)


def discover_task_seasons(
    release_root: Path, pattern: str = "*", tasks: str = "*"
) -> list[Season]:
    """Open every season under ``<release_root>/<task>/`` matching the globs.

    A season's task is ``season.path.parent.name``. The same season name may
    appear under several task folders; each is opened separately.
    """
    seasons = []
    for task_dir in sorted(p for p in Path(release_root).iterdir() if p.is_dir()):
        if not fnmatch.fnmatchcase(task_dir.name, tasks):
            continue
        for path in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            if fnmatch.fnmatchcase(path.name, pattern) and find_root(path):
                seasons.append(open_season(path))
    return seasons


def episode_annotations(season: Season) -> dict[int, dict]:
    """Return SharpaDex ``meta/annotations.jsonl`` rows keyed by episode.

    A season without the file returns an empty mapping.
    """
    path = season.root / "meta" / "annotations.jsonl"
    if not path.is_file():
        return {}
    rows = _jsonl(path, "annotations")
    return {int(r["episode_index"]): r for r in rows if "episode_index" in r}


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
    relative = season.info["data_path"].format(
        episode_chunk=index // _chunks_size(season), episode_index=index
    )
    path = season.root / relative
    if not path.is_file():
        raise LeRobotError(f"{path}: episode data file is missing")
    if columns is not None:
        available = pq.read_schema(path).names
        wanted = set(columns) | set(_BOOKKEEPING)
        columns = [name for name in available if name in wanted]
    table = pq.read_table(path, columns=columns)
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
    # A season clock places episodes by their frame offset; the stored global
    # index must agree with it.
    if "index" in arrays and int(arrays["index"][0]) != info.frame_offset:
        raise LeRobotError(
            f"{path}: episode {index} index starts at {int(arrays['index'][0])}, "
            f"but {info.frame_offset} frames precede it in the episode metadata"
        )
    length = info.length if max_frames is None else min(info.length, max_frames)
    arrays = {name: values[:length] for name, values in arrays.items()}
    return Episode(season=season, info=info, columns=arrays, length=length)


__all__ = [
    "Episode",
    "EpisodeInfo",
    "LeRobotError",
    "Season",
    "decode_frames",
    "discover_task_seasons",
    "episode_annotations",
    "find_root",
    "load_episode",
    "open_season",
    "parse_season_name",
]
