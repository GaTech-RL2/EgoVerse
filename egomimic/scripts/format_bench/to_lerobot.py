"""Lance pilot tables -> LeRobot v3.0 dataset, without re-encoding video.

Runs in the LeRobot venv (no egomimic imports). Numerics come from the Lance steps table, so all
three formats hold the same values. The Lance MP4s already use LeRobot's default encode
(libsvtav1, g=2, crf=30, preset 12, yuv420p), so each episode's MP4 is appended to the current
video file with LeRobot's own `concatenate_video_files`, exactly as its writer does after encoding.
Data and video files roll over at LeRobot's size limits; episode metadata, tasks, info and stats
go through `LeRobotDatasetMetadata.save_episode`.
"""

import argparse
import shutil
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import (
    auto_downsample_height_width,
    compute_episode_stats,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import get_file_size_in_mb
from lerobot.datasets.utils import update_chunk_file_indices
from lerobot.datasets.video_utils import (
    concatenate_video_files,
    get_video_duration_in_s,
)

from egomimic.rldb.lance.schema import only_steps_path

VIDEO_KEY = "observation.images.front_1"
VECTORS = {
    "head_pose": 7,
    "eye_gaze": 3,
    **{
        f"{s}.{f}": d
        for s in ("left", "right")
        for f, d in (
            ("wrist_pose", 7),
            ("ee_pose", 7),
            ("keypoints", 63),
            ("aria_keypoints", 63),
        )
    },
}
SCALARS = {"left.confidence": "float32", "right.confidence": "float32"}
BOOLS = ("left_valid", "right_valid", "head_pose_valid")
# Packed layout, as LeRobot datasets ship: one action and one state vector (the keymap's horizon
# keys); everything else stays its own observation.* column.
PACKED = [
    f"{s}.{f}"
    for s in ("left", "right")
    for f in ("keypoints", "ee_pose", "wrist_pose")
]
PACKED_DIM = sum(VECTORS[k] for k in PACKED)


def features(h: int, w: int, packed: bool) -> dict:
    vec = {k: d for k, d in VECTORS.items() if not (packed and k in PACKED)}
    f = {k: {"dtype": "float32", "shape": (d,), "names": None} for k, d in vec.items()}
    if packed:
        for k in ("action", "observation.state"):
            f[k] = {"dtype": "float32", "shape": (PACKED_DIM,), "names": None}
    f.update(
        {k: {"dtype": t, "shape": (1,), "names": None} for k, t in SCALARS.items()}
    )
    f.update({k: {"dtype": "bool", "shape": (1,), "names": None} for k in BOOLS})
    f["timestamp_ns"] = {"dtype": "int64", "shape": (1,), "names": None}
    f[VIDEO_KEY] = {
        "dtype": "video",
        "shape": (h, w, 3),
        "names": ["height", "width", "channels"],
    }
    return f


def column(tbl: pa.Table, dotted: str) -> np.ndarray:
    path = dotted.split(".")
    col = tbl.column(path[0]).combine_chunks()
    for p in path[1:]:
        col = col.field(p)
    return col


def episode_table(
    steps: pa.Table, ep: int, task_index: int, start: int, fps: float, packed: bool
) -> pa.Table:
    n = steps.num_rows
    cols = {}
    if packed:
        vec = np.concatenate(
            [
                np.asarray(column(steps, k).flatten(), np.float32).reshape(n, -1)
                for k in PACKED
            ],
            axis=1,
        )
        arr = pa.FixedSizeListArray.from_arrays(pa.array(vec.reshape(-1)), PACKED_DIM)
        cols["action"] = arr
        cols["observation.state"] = arr
    for k, d in VECTORS.items():
        if packed and k in PACKED:
            continue
        cols[k] = column(steps, k).cast(pa.list_(pa.float32(), d))
    for k in SCALARS:
        cols[k] = column(steps, k).cast(pa.float32())
    for k in BOOLS:
        cols[k] = column(steps, k)
    cols["timestamp_ns"] = column(steps, "timestamp_ns")
    cols["timestamp"] = pa.array(np.arange(n, dtype=np.float32) / np.float32(fps))
    cols["frame_index"] = pa.array(np.arange(n, dtype=np.int64))
    cols["episode_index"] = pa.array(np.full(n, ep, dtype=np.int64))
    cols["index"] = pa.array(np.arange(start, start + n, dtype=np.int64))
    cols["task_index"] = pa.array(np.full(n, task_index, dtype=np.int64))
    return pa.table(cols)


def numeric_stats(tbl: pa.Table, feats: dict) -> dict:
    data = {}
    for k in tbl.column_names:
        if k not in feats or feats[k]["dtype"] in ("video", "string"):
            continue
        col = tbl.column(k).combine_chunks()
        if pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type):
            arr = np.asarray(col.flatten()).reshape(len(tbl), -1)
        else:
            arr = np.asarray(col)
        data[k] = arr.astype(np.float64) if arr.dtype == bool else arr
    return compute_episode_stats(data, {k: feats[k] for k in data})


def video_stats(mp4: Path, n: int) -> dict:
    """LeRobot's image stats (sample_images + per-channel stats / 255) on frames decoded from the MP4."""
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(str(mp4), dimension_order="NCHW")
    idx = sample_indices(min(n, dec.metadata.num_frames))
    imgs = [
        auto_downsample_height_width(f.numpy())
        for f in dec.get_frames_at(indices=idx).data
    ]
    st = get_feature_stats(np.stack(imgs), axis=(0, 2, 3), keepdims=True)
    return {
        k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in st.items()
    }


class DataWriter:
    """Mirrors `DatasetWriter._save_episode_data`: append episodes to one parquet file until the
    next would push it past `data_files_size_in_mb`."""

    def __init__(self, meta: LeRobotDatasetMetadata):
        self.meta, self.chunk, self.file, self.writer = meta, 0, 0, None
        self.file_start, self.next_index = 0, 0

    def path(self) -> Path:
        return self.meta.root / self.meta.data_path.format(
            chunk_index=self.chunk, file_index=self.file
        )

    def write(self, tbl: pa.Table) -> dict:
        if self.writer is not None:
            size = get_file_size_in_mb(self.path())
            in_file = self.next_index - self.file_start
            per_frame = size / in_file if in_file else 0
            if size + per_frame * tbl.num_rows >= self.meta.data_files_size_in_mb:
                self.writer.close()
                self.writer = None
                self.chunk, self.file = update_chunk_file_indices(
                    self.chunk, self.file, self.meta.chunks_size
                )
                self.file_start = self.next_index
        if self.writer is None:
            self.path().parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(
                self.path(), tbl.schema, compression="snappy", use_dictionary=True
            )
        self.writer.write_table(tbl)
        out = {
            "data/chunk_index": self.chunk,
            "data/file_index": self.file,
            "dataset_from_index": self.next_index,
            "dataset_to_index": self.next_index + tbl.num_rows,
        }
        self.next_index += tbl.num_rows
        return out

    def close(self):
        if self.writer is not None:
            self.writer.close()


class VideoWriter:
    """Mirrors `DatasetWriter._save_episode_video` for an already-encoded episode MP4."""

    def __init__(self, meta: LeRobotDatasetMetadata, key: str, tmp: Path):
        self.meta, self.key, self.tmp = meta, key, tmp
        self.chunk, self.file, self.duration = 0, 0, None

    def path(self) -> Path:
        return self.meta.root / self.meta.video_path.format(
            video_key=self.key, chunk_index=self.chunk, file_index=self.file
        )

    def write(self, mp4: Path) -> dict:
        ep_s = get_video_duration_in_s(mp4)
        ep_mb = get_file_size_in_mb(mp4)
        if (
            self.duration is not None
            and get_file_size_in_mb(self.path()) + ep_mb
            >= self.meta.video_files_size_in_mb
        ):
            self.chunk, self.file = update_chunk_file_indices(
                self.chunk, self.file, self.meta.chunks_size
            )
            self.duration = None
        if self.duration is None:
            self.path().parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(mp4, self.path())
            start = 0.0
        else:
            start = self.duration
            # concatenate_video_files reads its inputs before writing, but stage a copy anyway
            # so the source table's MP4 is never an output.
            staged = self.tmp / mp4.name
            shutil.copyfile(mp4, staged)
            concatenate_video_files([self.path(), staged], self.path())
            staged.unlink()
        self.duration = start + ep_s
        return {
            f"videos/{self.key}/chunk_index": self.chunk,
            f"videos/{self.key}/file_index": self.file,
            f"videos/{self.key}/from_timestamp": start,
            f"videos/{self.key}/to_timestamp": start + ep_s,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lance-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--camera", default="front_1_480")
    ap.add_argument("--repo-id", default="egoverse/aria_pilot")
    ap.add_argument(
        "--limit", type=int, default=0, help="first N recordings only (smoke tests)"
    )
    ap.add_argument(
        "--packed",
        action="store_true",
        help="one action + one observation.state vector",
    )
    args = ap.parse_args()

    root = args.lance_root
    rec = lance.dataset(str(root / "recordings.lance")).to_table().to_pandas()
    tasks = (
        lance.dataset(str(root / "tasks.lance"))
        .to_table(columns=["recording_id", "task_name", "task_description"])
        .to_pandas()
    )
    rec = rec.merge(tasks, on="recording_id")
    steps_ds = lance.dataset(str(root / only_steps_path(rec)))
    ids = (
        steps_ds.to_table(columns=["recording_id"])
        .column("recording_id")
        .to_numpy(zero_copy_only=False)
    )
    first_row = {}
    for i, r in enumerate(ids):
        first_row.setdefault(r, i)
    rec["row_start"] = rec.recording_id.map(first_row)
    rec = rec.sort_values("row_start").reset_index(drop=True)
    if args.limit:
        rec = rec.head(args.limit)

    from torchcodec.decoders import VideoDecoder

    first_mp4 = root / dict(rec.video_path[0])[args.camera]
    m = VideoDecoder(str(first_mp4)).metadata
    fps = int(round(float(rec.fps[0])))
    feats = features(m.height, m.width, args.packed)

    if args.out.exists():
        shutil.rmtree(args.out)
    meta = LeRobotDatasetMetadata.create(
        repo_id=args.repo_id, fps=fps, features=feats, root=args.out
    )
    tmp = args.out.parent / f".{args.out.name}_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    data_w, video_w = DataWriter(meta), VideoWriter(meta, VIDEO_KEY, tmp)
    t0 = time.perf_counter()
    for ep, row in rec.iterrows():
        n = int(row.num_frames)
        steps = steps_ds.take(list(range(int(row.row_start), int(row.row_start) + n)))
        mp4 = root / dict(row.video_path)[args.camera]
        nv = VideoDecoder(str(mp4)).metadata.num_frames
        if nv != n:
            raise ValueError(f"{row.recording_id}: {nv} video frames, {n} steps rows")
        task = row.task_description or row.task_name
        meta.save_episode_tasks([task])
        task_index = int(meta.get_task_index(task))
        tbl = episode_table(steps, ep, task_index, data_w.next_index, fps, args.packed)
        md = data_w.write(tbl)
        md.update(video_w.write(mp4))
        stats = numeric_stats(tbl, meta.features)
        stats[VIDEO_KEY] = video_stats(mp4, n)
        meta.save_episode(ep, n, [task], stats, md)
        if ep == 0:
            meta.update_video_info(VIDEO_KEY)
        print(
            f"ep {ep} {row.recording_id} {n} frames {time.perf_counter() - t0:.0f}s",
            flush=True,
        )
    data_w.close()
    meta.finalize()
    shutil.rmtree(tmp)
    print(
        f"done: {len(rec)} episodes, {data_w.next_index} frames, {time.perf_counter() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
