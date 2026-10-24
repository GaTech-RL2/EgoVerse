"""Tiny synthetic SharpaDex releases and manifests for the converter tests.

A release is ``<root>/<task>/<season>/lerobot_v2.1`` with the column names,
streams and annotations SharpaDex ships, lossless H.264 videos and in-limit
joint vectors whose TCP column agrees with forward kinematics. No real data.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from egomimic.scripts.sharpa_process.sharpa_to_zarr import (
    episode_hash,
    expected_vendor_names,
)
from egomimic.scripts.sharpa_process.sharpa_tracks import (
    TCP_LINK,
    TCP_MODEL,
    derive_tracks,
)
from egomimic.scripts.sharpa_process.sharpadex_records import EpisodeRecord

FPS = 30
SIZE = 32
LENGTHS = (24, 20)
RIG = "POC99001"
TASK = "iron_garment"
SEASON = f"season_{RIG}_2026_01_02_03_04_05_train"
#: 2026-01-02 03:04:05 at +08:00, as UTC nanoseconds.
SEASON_START_NS = (
    int(datetime(2026, 1, 1, 19, 4, 5, tzinfo=timezone.utc).timestamp()) * 10**9
)
#: Each later recording starts 55 s after the one before.
STEP_NS = 55 * 10**9
TAGS = {
    "task_name": "Iron_Clothes",
    "task_instruction": "Grip the iron head and iron the garment.",
    "score": "S",
    "SOP": "V1.0",
}
CAMERAS = ("head_left", "wrist_left", "wrist_right", "head_right")
DEFORM = (480, 1200)
LICENSE = "cc-by-nc-4.0"
PER_HAND_TEXT = "Left hand: idle: Stays still. | Right hand: reach: Moves down."


def joints(kinematics, length: int, phase: float) -> np.ndarray:
    """Deterministic in-limit 65-column vectors that move every joint."""
    lower, upper = kinematics.limits.T
    t = np.arange(length)[:, None]
    wave = np.sin(0.2 * t + np.arange(65)[None] + phase)
    return ((lower + upper) / 2 + 0.25 * (upper - lower) * wave).astype(np.float32)


def frame(global_index: int, camera_index: int) -> np.ndarray:
    image = np.empty((SIZE, SIZE, 3), np.uint8)
    image[...] = (global_index * 7 + camera_index * 50) % 200 + 20
    image[:, : SIZE // 2, 1] = 255 - image[:, : SIZE // 2, 0]
    return image


def deform(global_index: int) -> np.ndarray:
    """A grayscale 2 x 5 mosaic whose tiles differ by hand and finger."""
    rows, cols = DEFORM
    image = np.empty((rows, cols, 3), np.uint8)
    for row in range(2):
        for col in range(5):
            value = (global_index * 3 + row * 90 + col * 30) % 200 + 20
            image[row * 240 : (row + 1) * 240, col * 240 : (col + 1) * 240] = value
    return image


def write_video(path: Path, frames) -> None:
    frames = list(frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"qp": "0", "preset": "ultrafast"}
        for index, image in enumerate(frames):
            video_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            video_frame.pts, video_frame.time_base = index, Fraction(1, FPS)
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _list_column(values):
    values = np.asarray(values)
    offsets = np.arange(0, values.size + 1, values.shape[1], dtype=np.int32)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(values.ravel()))


def tcp_pose(kinematics, observed) -> np.ndarray:
    """Sharpa's TCP for ``observed`` joints, from the fitted TCP model."""
    from scipy.spatial.transform import Rotation

    links = [TCP_LINK.format(side=side) for side in ("left", "right")]
    _, poses = derive_tracks(observed, observed, kinematics=kinematics, links=links)
    out = []
    for side in ("left", "right"):
        model, link = TCP_MODEL[side], poses[TCP_LINK.format(side=side)]
        rotation = Rotation.from_matrix(link[:, :3, :3])
        position = (
            link[:, :3, 3] + rotation.apply(model["tool_m"]) + model["base_shift_m"]
        )
        tool = Rotation.from_rotvec(np.radians(model["tool_rotvec_deg"]))
        out += [position, (rotation * tool).as_rotvec()]
    return np.concatenate(out, axis=1).astype(np.float32)


def segments(episode: int, length: int) -> list[dict]:
    """Two subtask spans; episode 1 describes both hands in its first span."""
    first = PER_HAND_TEXT if episode == 1 else "Grip the iron head. | Skill: grip"
    return [
        {
            "label": "subtasks",
            "start_step": 0,
            "end_step": 9,
            "id": "subtask-001",
            "text": first,
            **({} if episode == 1 else {"task_skill": "grip"}),
        },
        {
            "label": "subtasks",
            "start_step": 10,
            "end_step": length - 1,
            "id": "subtask-002",
            "text": "Iron the garment. | Skill: press",
            "task_skill": "press",
        },
    ]


def build_season(
    root: Path,
    kinematics,
    *,
    task: str = TASK,
    season: str = SEASON,
    lengths=LENGTHS,
    cameras=CAMERAS,
    swap_tcp: bool = False,
    excursion: tuple[str, float] | None = None,
) -> Path:
    """Write one SharpaDex task/season (v2.1) under the release ``root``.

    ``excursion`` puts observed joint ``name`` that far above its upper limit
    on frames 2..4 of every episode.
    """
    names = expected_vendor_names()
    v21 = root / task / season / "lerobot_v2.1"
    (v21 / "meta").mkdir(parents=True)
    features = {}
    for name, width in (
        ("observation.state", 65),
        ("action", 65),
        ("observation.state.joint_torque", 65),
        ("observation.state.tcp", 24),
        ("observation.state.tcp_pose", 12),
        ("action.tcp_pose", 12),
        ("observation.tactile", 60),
    ):
        feature = {"dtype": "float32", "shape": [width]}
        if name in names:
            feature["names"] = names[name]
        features[name] = feature
    for camera in cameras:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": [SIZE, SIZE, 3],
        }
    features["observation.images.tactile_deform"] = {
        "dtype": "video",
        "shape": [*DEFORM, 3],
    }
    for name in (
        "frame_index",
        "episode_index",
        "index",
        "task_index",
        "subtask_index",
    ):
        features[name] = {"dtype": "int64", "shape": [1]}
    info = {
        "codebase_version": "v2.1",
        "robot_type": "robot",
        "fps": FPS,
        "chunks_size": 1000,
        "total_episodes": len(lengths),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (v21 / "meta/info.json").write_text(json.dumps(info))
    task_text = f"Task: {TAGS['task_name']} | Instruction: {TAGS['task_instruction']}"
    (v21 / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task_text}) + "\n"
    )
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    episodes, annotations = [], []
    for e, (n, o) in enumerate(zip(lengths, offsets, strict=True)):
        observed = joints(kinematics, n, e)
        commanded = joints(kinematics, n, e + 0.5)
        tcp = tcp_pose(kinematics, observed)
        if swap_tcp:
            tcp = np.concatenate([tcp[:, 6:], tcp[:, :6]], axis=1)
        if excursion is not None:
            column = kinematics.column_names.index(excursion[0])
            observed[2:5, column] = kinematics.limits[column, 1] + excursion[1]
        rng = np.random.default_rng(e)
        frames = np.arange(n)
        table = pa.table(
            {
                "observation.state": _list_column(observed),
                "action": _list_column(commanded),
                "observation.state.joint_torque": _list_column(
                    rng.normal(size=(n, 65)).astype(np.float32)
                ),
                "observation.state.tcp": _list_column(
                    rng.normal(size=(n, 24)).astype(np.float32)
                ),
                "observation.state.tcp_pose": _list_column(tcp),
                "action.tcp_pose": _list_column(tcp),
                "observation.tactile": _list_column(
                    rng.normal(size=(n, 60)).astype(np.float32)
                ),
                "subtask_index": pa.array((frames >= 10).astype(np.int64)),
                "timestamp": pa.array((frames / FPS).astype(np.float32)),
                "frame_index": pa.array(frames),
                "episode_index": pa.array(np.full(n, e)),
                "index": pa.array(o + frames),
                "task_index": pa.array(np.zeros(n, np.int64)),
            }
        )
        data = v21 / f"data/chunk-000/episode_{e:06d}.parquet"
        data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data)
        videos = v21 / "videos/chunk-000"
        for c, camera in enumerate(cameras):
            write_video(
                videos / f"observation.images.{camera}/episode_{e:06d}.mp4",
                [frame(o + i, c) for i in range(n)],
            )
        write_video(
            videos / f"observation.images.tactile_deform/episode_{e:06d}.mp4",
            [deform(o + i) for i in range(n)],
        )
        episodes.append({"episode_index": e, "tasks": [task_text], "length": n})
        annotations.append(
            {
                "episode_index": e,
                "source_uid": f"{task}/{season}/uid{e}",
                "tags": TAGS,
                "segments": segments(e, n),
            }
        )
    for name, rows in (("episodes", episodes), ("annotations", annotations)):
        (v21 / f"meta/{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    return root


def write_readme(root: Path, license: str | None = LICENSE) -> None:
    root.mkdir(parents=True, exist_ok=True)
    front = f"---\nlicense: {license}\n---\n" if license else ""
    (root / "README.md").write_text(f"{front}# SharpaDex\n")


def records(
    task: str = TASK,
    season: str = SEASON,
    lengths=LENGTHS,
    start_ns: int = SEASON_START_NS,
) -> list[EpisodeRecord]:
    """Kept manifest records for one season, ``STEP_NS`` apart."""
    out = []
    for e, n in enumerate(lengths):
        start = start_ns + e * STEP_NS
        out.append(
            EpisodeRecord(
                recording_id=f"{task}/{season}/{e}",
                task=task,
                season=season,
                episode_index=e,
                rig=RIG,
                length=n,
                fps=FPS,
                source_uid=f"{task}/{season}/uid{e}",
                fingerprint=f"{e:016d}",
                tags=dict(TAGS),
                start_source="episode_id",
                clock="vendor_episode_start+frame_index",
                start_ns=start,
                episode_hash=episode_hash(start),
            )
        )
    return out


def write_manifest(path: Path, release: Path, rows: list[EpisodeRecord]) -> Path:
    header = {
        "release_root": str(release),
        "source_revision": "abc123",
        "source_utc_offset": "+08:00",
    }
    lines = [json.dumps({"manifest": header})]
    lines += [json.dumps(asdict(r), sort_keys=True) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path
