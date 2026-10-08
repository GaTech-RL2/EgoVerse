"""Lance-backed episode datasets: one leaf per recording window, drop-in for ZarrDataset.

Leaves reuse ZarrDataset's keymap semantics (horizons, history, lag, pad masks,
annotations, transforms); only storage access differs. A leaf's index runs over
eligible anchors, not every frame. Lance and torchcodec handles are opened lazily per
process: Lance is not fork-safe (https://lance.org/integrations/pytorch/), so loaders
over these datasets must use a forkserver or spawn worker context.
"""

import concurrent.futures as cf
import functools
import hashlib
import logging
import os
from collections import OrderedDict
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import torch

from egomimic.rldb.embodiment.embodiment import HISTORY_MASK_SUFFIX, get_embodiment_id
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import (
    PinError,
    ZarrDataset,
    _ensure_dataset_filter,
)

logger = logging.getLogger(__name__)

# zarr key -> Lance steps column (nested fields as dotted paths).
ZARR_TO_LANCE = {
    "obs_head_pose": "head_pose",
    "obs_eye_gaze": "eye_gaze",
    **{
        f"{s}.{z}": f"{s}.{c}"
        for s in ("left", "right")
        for z, c in (
            ("obs_wrist_pose", "wrist_pose"),
            ("obs_ee_pose", "ee_pose"),
            ("obs_keypoints", "keypoints"),
            ("obs_aria_keypoints", "aria_keypoints"),
        )
    },
}
VALIDITY = {"left": "left_valid", "right": "right_valid", "head": "head_pose_valid"}
HANDS_OF = {
    "human_bimanual": ("left", "right"),
    "human_left_arm": ("left",),
    "human_right_arm": ("right",),
}

_TABLES: dict[tuple[str, int], "lance.LanceDataset"] = {}
_DECODERS: "OrderedDict[str, object]" = OrderedDict()
_MAX_DECODERS = 16


def _is_remote(uri: str) -> bool:
    return uri.startswith("s3://")


def storage_options(uri: str) -> dict | None:
    """R2 credentials for an s3:// URI, read from the environment in each process so
    they never travel in pickled datasets, configs or cache keys."""
    if not _is_remote(uri):
        return None
    return {
        "access_key_id": os.environ["R2_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["R2_SECRET_ACCESS_KEY"],
        "endpoint": os.environ["R2_ENDPOINT_URL"],
        "region": "auto",
    }


def open_table(uri: str, version: int | str | None = None):
    return lance.dataset(uri, version=version, storage_options=storage_options(uri))


def _table(uri: str, version: int):
    key = (uri, version)
    if key not in _TABLES:
        _TABLES[key] = open_table(uri, version)
    return _TABLES[key]


@functools.cache
def _s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def _video_source(path: str) -> str:
    """Local path as is; s3:// as a presigned HTTPS URL that FFmpeg reads with range requests."""
    if not _is_remote(path):
        return path
    bucket, key = path[len("s3://") :].split("/", 1)
    return _s3_client().generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=7 * 86400
    )


def _decoder(path: str):
    from torchcodec.decoders import VideoDecoder

    dec = _DECODERS.pop(path, None)
    if dec is None:
        # Our MP4s are constant-rate encodes, so header metadata indexes frames exactly.
        dec = VideoDecoder(
            _video_source(path), seek_mode="approximate", dimension_order="NHWC"
        )
    _DECODERS[path] = dec
    while len(_DECODERS) > _MAX_DECODERS:
        _DECODERS.popitem(last=False)
    return dec


def _runs(ids: pa.ChunkedArray) -> tuple[list, np.ndarray]:
    """(value, start row) of each run of equal values; a value can recur in later runs."""
    # A numpy object array of every row's id costs seconds at 50M rows; run-end encoding stays in Arrow.
    values, starts, offset = [], [], 0
    for chunk in pc.run_end_encode(ids).chunks:
        ends = chunk.run_ends.to_numpy()
        for s, v in zip(np.r_[0, ends[:-1]] + offset, chunk.values.to_pylist()):
            if not values or v != values[-1]:
                values.append(v)
                starts.append(int(s))
        offset += len(chunk)
    return values, np.asarray(starts, dtype=np.int64)


def eligible_intervals(valid: dict[str, np.ndarray], hands, horizon: int) -> np.ndarray:
    """(K, 2) half-open frame intervals of anchors whose action chunk is valid for every hand.

    The chunk is [t, min(t + horizon, n)); the head pose must be valid at t because
    actions are expressed in the anchor's head frame.
    """
    n = len(valid["head"])
    ok = valid["head"].copy()
    for h in hands:
        v = valid[h].astype(np.int64)
        c = np.r_[0, np.cumsum(v)]
        end = np.minimum(np.arange(n) + horizon, n)
        ok &= (c[end] - c[:n]) == (end - np.arange(n))
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    return edges.reshape(-1, 2)


def _frames_of(window) -> np.ndarray:
    if isinstance(window, np.ndarray):
        return window.astype(np.int64)
    start, end = window
    return np.arange(start, start + 1 if end is None else end)


def _column(tbl: pa.Table, dotted: str) -> np.ndarray:
    path = dotted.split(".")
    col = tbl.column(path[0]).combine_chunks()
    for p in path[1:]:
        col = col.field(p)
    arr = np.asarray(col.flatten() if pa.types.is_fixed_size_list(col.type) else col)
    return arr.reshape(len(tbl), -1)


def _fetch_numeric(items: list) -> list[dict]:
    """One `take` per steps table over every item's rows and columns."""
    out = [{} for _ in items]
    groups: dict[tuple, list[int]] = {}
    for j, (leaf, plan) in enumerate(items):
        if plan["needs"]:
            groups.setdefault((leaf.steps_uri, leaf.steps_version), []).append(j)
    for (uri, version), js in groups.items():
        rows = np.unique(
            np.concatenate(
                [
                    f + items[j][0].row_start
                    for j in js
                    for f in items[j][1]["needs"].values()
                ]
            )
        )
        cols = sorted({ZARR_TO_LANCE[zk] for j in js for zk in items[j][1]["needs"]})
        tbl = _table(uri, version).take(rows.tolist(), columns=cols)
        arrays = {c: _column(tbl, c) for c in cols}
        for j in js:
            leaf, plan = items[j]
            for zk, frames in plan["needs"].items():
                at = np.searchsorted(rows, frames + leaf.row_start)
                out[j][zk] = arrays[ZARR_TO_LANCE[zk]][at]
    return out


def _fetch_chunk_frames(items: list) -> list[dict]:
    """Frames per image key from video chunk rows: one `read_blobs` per chunk table,
    each chunk decoded once."""
    out = [{} for _ in items]
    want: dict[tuple, dict[int, list]] = {}
    for j, (leaf, plan) in enumerate(items):
        for k, (zk, frames) in plan["images"].items():
            uri, version, c0, kf = leaf.chunks[zk]
            per_chunk = want.setdefault((uri, version), {})
            for n, f in enumerate(frames):
                per_chunk.setdefault(c0 + int(f) // kf, []).append(
                    (j, k, n, int(f) % kf, len(frames))
                )
    if want:
        from torchcodec.decoders import VideoDecoder
    for (uri, version), per_chunk in want.items():
        rows = sorted(per_chunk)
        blobs = _table(uri, version).read_blobs(
            "data", indices=rows, preserve_order=True
        )
        for row, (_, data) in zip(rows, blobs, strict=True):
            uses = per_chunk[row]
            offsets = sorted({u[3] for u in uses})
            # torchcodec does not hold a reference to `data`; it stays alive in this scope.
            dec = VideoDecoder(data, seek_mode="exact", dimension_order="NHWC")
            got = dict(zip(offsets, dec.get_frames_at(indices=offsets).data.numpy()))
            del dec
            for j, k, n, off, total in uses:
                if k not in out[j]:
                    out[j][k] = np.empty((total, *got[off].shape), got[off].dtype)
                out[j][k][n] = got[off]
    return out


@functools.cache
def _read_pool() -> cf.ThreadPoolExecutor:
    """Per-process pool for the chunk reads that overlap a batch's steps `take`."""
    return cf.ThreadPoolExecutor(max_workers=8, thread_name_prefix="lance-chunks")


def _chunks_of(table: dict, recording_id: str, num_frames: int, name: str) -> tuple:
    if table["frames"].get(recording_id) != num_frames:
        raise ValueError(
            f"{name}: {table['frames'].get(recording_id, 0)} frames in {table['uri']}, num_frames {num_frames}"
        )
    return (table["uri"], table["version"], table["rows"][recording_id], table["k"])


class LanceRecordingDataset(ZarrDataset):
    """One recording from local Lance tables: steps rows plus its MP4s on disk, read
    one sample at a time. `LanceStreamingDataset` is the object-storage variant."""

    fork_safe = False

    def __init__(
        self,
        *,
        name: str,
        steps_uri: str,
        steps_version: int,
        row_start: int,
        num_frames: int,
        anchors: np.ndarray,
        embodiment: str,
        fps: float,
        intrinsics: np.ndarray | None,
        videos: dict[str, str],
        task_text: str,
        key_map: dict,
        transform_list: list | None = None,
        content_digest: str | None = None,
    ):
        self.episode_path = name
        self.content_digest = content_digest
        self.steps_uri = steps_uri
        self.steps_version = steps_version
        self.row_start = row_start
        self.total_frames = num_frames
        self.embodiment = embodiment
        self.metadata = {
            "fps": fps,
            "total_frames": num_frames,
            "embodiment": embodiment,
        }
        self._intrinsics = intrinsics
        self.videos = videos
        self.task_text = task_text
        self.key_map = key_map
        self.transform = transform_list
        self._image_keys = {k for k in videos}
        self._json_keys = set()
        self._annotations = []
        self.anchors = np.asarray(anchors, dtype=np.int64).reshape(-1, 2)
        self._cum = np.r_[0, np.cumsum(self.anchors[:, 1] - self.anchors[:, 0])]
        torch.utils.data.Dataset.__init__(self)

    def cache_fingerprint(self) -> str:
        """Keys on the recording's content digest, so commits to other recordings and
        compaction keep it; without a digest, falls back to the table version and row
        position. Never on a tag name."""
        h = hashlib.sha256()
        if self.content_digest:
            h.update(f"digest\0{self.content_digest}\0{self.total_frames}\0".encode())
            for k in sorted(self.videos):
                h.update(f"{k}\0{Path(self.videos[k]).name}\0".encode())
        else:
            h.update(
                f"{self.steps_uri}\0{self.steps_version}\0{self.row_start}\0{self.total_frames}\0".encode()
            )
            for k in sorted(self.videos):
                h.update(f"{k}\0{self.videos[k]}\0".encode())
        h.update(self.anchors.tobytes())
        return h.hexdigest()

    @property
    def intrinsics(self):
        return self._intrinsics

    def __len__(self) -> int:
        return int(self._cum[-1])

    def anchor(self, i: int) -> int:
        j = int(np.searchsorted(self._cum, i, side="right")) - 1
        return int(self.anchors[j, 0] + (i - self._cum[j]))

    def _annotation_text_for_frame(self, frame_idx: int):
        return [self.task_text]

    def _read_numeric(self, needs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """One `take` for every numeric column; returns {zarr key: rows at the requested frames}."""
        return _fetch_numeric([(self, {"needs": needs})])[0]

    def plan_sample(self, i: int) -> dict:
        """What sample `i` reads: numeric frames per zarr key and frames per image key."""
        idx = self.anchor(i)
        windows = self._read_windows(idx)
        needs, images = {}, {}
        for k, (zk, window) in windows.items():
            frames = _frames_of(window)
            if zk in self._image_keys:
                images[k] = (zk, frames)
                continue
            needs[zk] = np.unique(np.r_[needs.get(zk, np.empty(0, np.int64)), frames])
        return {"idx": idx, "windows": windows, "needs": needs, "images": images}

    def assemble_sample(self, plan: dict, fetched: tuple[dict, dict]):
        idx, windows, needs = plan["idx"], plan["windows"], plan["needs"]
        numeric, frames_by_key = fetched
        data = {}
        for k, spec in self.key_map.items():
            zk = spec["zarr_key"]
            if spec.get("key_type") == "annotation_keys":
                data[k] = self._annotation_text_for_frame(idx)
                continue
            window = windows[k][1]
            frames = _frames_of(window)
            single = not isinstance(window, np.ndarray) and window[1] is None
            if zk in self._image_keys:
                img = np.moveaxis(frames_by_key[k], -1, -3).astype(
                    np.float32
                ) / np.float32(255.0)
                data[k] = img[0] if single else img
                continue
            rows = numeric[zk][np.searchsorted(needs[zk], frames)].astype(np.float32)
            data[k] = self._pad_to_horizon(
                rows[0] if single else rows, spec.get("horizon")
            )
        return self._finish(idx, data)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        plan = self.plan_sample(i)
        numeric = _fetch_numeric([(self, plan)])[0]
        frames = {
            k: _decoder(self.videos[zk]).get_frames_at(indices=f.tolist()).data.numpy()
            for k, (zk, f) in plan["images"].items()
        }
        return self.assemble_sample(plan, (numeric, frames))

    def _finish(self, idx: int, data: dict) -> dict:
        pad_mask = self._action_pad_mask(idx)
        if pad_mask is not None:
            data["action_pad_mask"] = pad_mask
        for k, spec in self.key_map.items():
            if spec.get("key_type") == "camera_keys" and spec.get("history"):
                data[f"{k}{HISTORY_MASK_SUFFIX}"] = self._history_indices(idx, spec)[1]
        history_mask = self._proprio_history_mask(idx)
        if history_mask is not None:
            data["proprio_history_mask"] = history_mask

        for transform in self.transform or []:
            data = transform.transform(data)
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).to(torch.float32)

        data["embodiment"] = get_embodiment_id(self.embodiment)
        data["fps"] = torch.tensor(float(self.metadata["fps"]), dtype=torch.float32)
        K = self._intrinsics
        K = (
            np.full((3, 4), np.nan, np.float32)
            if K is None
            else np.c_[np.asarray(K, np.float32).reshape(3, 3), np.zeros(3, np.float32)]
        )
        data["intrinsics"] = torch.from_numpy(np.ascontiguousarray(K))
        data["episode_hash"] = self.episode_path
        return data


class LanceStreamingDataset(LanceRecordingDataset):
    """One recording read straight from object storage (or any Lance URI): frames come
    from K-frame video chunk rows (`video_chunks/<name>.lance`, `chunk_videos.py`)
    instead of whole MP4s, and `MultiDataset.__getitems__` fetches a DataLoader batch
    through `fetch_planned`, so a batch costs about two round trips."""

    def __init__(
        self,
        *,
        chunks: dict[str, tuple[str, int, int, int]],
        fetch_split: int | None = None,
        overlap_reads: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # {zarr image key: (chunk table uri, version, first chunk row, frames per chunk)}
        self.chunks = dict(chunks)
        # MultiDataset.__getitems__ fetches a batch in parts of this many samples,
        # concurrently, and assembles each part as it lands; None = one fetch.
        self.fetch_split = fetch_split
        self.overlap_reads = overlap_reads

    @staticmethod
    def fetch_planned(items: list) -> list[tuple[dict, dict]]:
        """Reads for many planned samples at once: one `take` per steps table and one
        `read_blobs` per chunk table, issued concurrently. `items` is [(leaf, plan)];
        returns [(numeric {zarr key: rows}, frames {key: uint8 NHWC})] in order."""
        if items and items[0][0].overlap_reads:
            frames = _read_pool().submit(_fetch_chunk_frames, items)
            numeric = _fetch_numeric(items)
            frames = frames.result()
        else:
            numeric, frames = _fetch_numeric(items), _fetch_chunk_frames(items)
        return list(zip(numeric, frames, strict=True))

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        plan = self.plan_sample(i)
        return self.assemble_sample(plan, self.fetch_planned([(self, plan)])[0])


class LanceEpisodeResolver:
    """Resolves recordings from local Lance tables into LanceRecordingDataset leaves.

    Filter lambdas see each recording joined with its task, with SQL-style field names
    (`episode_hash`, `lab` is the site, `task`, ...), so existing filter configs apply.

    Each table is read at `versions[name]` (a number or tag), else at `tag`, else at its
    latest version. Tags are resolved to numbers once, here, so a tag moved later does
    not change what an existing resolver reads.
    """

    def __init__(
        self,
        lance_root: str | Path,
        lab: str = "aria",
        key_map: dict | None = None,
        transform_list: list | None = None,
        cameras: dict[str, str] | None = None,
        versions: dict[str, int | str] | None = None,
        tag: str | None = None,
    ):
        self.root = str(lance_root).rstrip("/")
        self.lab = lab
        self.key_map = key_map
        self.transform_list = transform_list
        self.cameras = cameras or {"images.front_1": "front_1"}
        self.tag = tag
        self.versions = dict(versions or {})

    def _open(self, name: str):
        uri = f"{self.root}/{name}"
        want = self.versions.get(name, self.tag)
        if not isinstance(want, int):
            want = open_table(uri, want).version
            self.versions[name] = want
        return uri, want, _table(uri, want)

    def catalog(self) -> pd.DataFrame:
        _, _, rec = self._open("recordings.lance")
        _, _, tasks = self._open("tasks.lance")
        _, _, aliases = self._open("aliases.lance")
        r = rec.to_table().to_pandas()
        t = tasks.to_table(
            columns=["recording_id", "task_name", "task_description", "objects"]
        ).to_pandas()
        a = aliases.to_table(columns=["recording_id", "episode_hash"]).to_pandas()
        df = r.merge(t, on="recording_id").merge(a, on="recording_id")
        df = df[df.lab == self.lab].rename(
            columns={"lab": "vendor", "site": "lab", "task_name": "task"}
        )
        return df.reset_index(drop=True)

    def _horizon(self) -> int:
        hs = [
            int(s["horizon"])
            for s in self.key_map.values()
            if s.get("key_type") == "action_keys" and s.get("horizon")
        ]
        return max(hs) if hs else 1

    def resolve(
        self,
        filters: DatasetFilter | None = None,
        expected_embodiment: str | None = None,
    ):
        filters = _ensure_dataset_filter(filters)
        cat = self.catalog()
        keep = cat[[filters.matches(row) for row in cat.to_dict("records")]]
        if expected_embodiment is not None:
            keep = keep[keep.embodiment.str.lower() == expected_embodiment.lower()]
        missing = sorted(set(filters.episode_hashes) - set(keep.episode_hash))
        if missing:
            raise PinError(f"pinned episodes not in the Lance catalog: {missing}")

        uri, version, steps = self._open(f"steps/{self.lab}.lance")
        cols = steps.to_table(columns=["recording_id", *VALIDITY.values()])
        n_rows = cols.num_rows
        first, change = _runs(cols.column("recording_id"))
        starts = dict(zip(first, change))
        if len(starts) != len(change):
            raise ValueError(
                f"{uri}@{version}: a recording's steps rows are not contiguous"
            )
        counts = dict(zip(first, np.diff(np.r_[change, n_rows])))
        all_valid = {
            h: cols.column(c).to_numpy(zero_copy_only=False)
            for h, c in VALIDITY.items()
        }
        horizon = self._horizon()
        tables = self._leaf_tables()

        out = {}
        for row in keep.itertuples():
            r0, n = int(starts.get(row.recording_id, -1)), int(row.num_frames)
            if counts.get(row.recording_id) != n:
                raise ValueError(
                    f"{row.episode_hash}: {counts.get(row.recording_id, 0)} steps rows at {uri}@{version}, num_frames {n}"
                )
            valid = {h: a[r0 : r0 + n] for h, a in all_valid.items()}
            anchors = eligible_intervals(
                valid, HANDS_OF.get(row.embodiment, ("left", "right")), horizon
            )
            if not len(anchors):
                logger.warning("recording %s has no eligible anchors", row.episode_hash)
                continue
            intr = dict(row.intrinsics) if row.intrinsics is not None else {}
            vids = dict(row.video_path)
            cam = next(iter(self.cameras.values()))
            out[row.episode_hash] = self._leaf(
                row,
                tables,
                name=row.episode_hash,
                steps_uri=uri,
                steps_version=version,
                row_start=r0,
                num_frames=n,
                anchors=anchors,
                embodiment=row.embodiment,
                fps=float(row.fps),
                intrinsics=np.asarray(intr[cam]) if cam in intr else None,
                videos={zk: f"{self.root}/{vids[c]}" for zk, c in self.cameras.items()},
                task_text=row.task_description or row.task,
                key_map=self.key_map,
                transform_list=self.transform_list,
                content_digest=getattr(row, "content_digest", None),
            )
        return out

    def _leaf_tables(self) -> dict:
        return {}

    def _leaf(self, row, tables: dict, **kwargs) -> LanceRecordingDataset:
        return LanceRecordingDataset(**kwargs)


class LanceStreamingResolver(LanceEpisodeResolver):
    """Resolves recordings into LanceStreamingDataset leaves, for training straight from
    object storage (`lance_root` may be s3://). `video_chunks` names the chunk table
    for each camera key: {"images.front_1": "front_1_k8"}."""

    def __init__(
        self,
        lance_root: str | Path,
        video_chunks: dict[str, str],
        fetch_split: int | None = None,
        overlap_reads: bool = True,
        **kwargs,
    ):
        super().__init__(lance_root, **kwargs)
        self.video_chunks = dict(video_chunks)
        self.fetch_split = fetch_split
        self.overlap_reads = overlap_reads
        used = {
            s["zarr_key"]
            for s in (self.key_map or {}).values()
            if s.get("key_type") == "camera_keys"
        }
        missing = sorted(used & set(self.cameras) - set(self.video_chunks))
        if missing:
            raise ValueError(f"no video_chunks table for camera keys {missing}")

    def _leaf_tables(self) -> dict:
        return {zk: self._chunk_table(name) for zk, name in self.video_chunks.items()}

    def _leaf(self, row, tables: dict, **kwargs) -> LanceStreamingDataset:
        chunks = {
            zk: _chunks_of(t, row.recording_id, kwargs["num_frames"], row.episode_hash)
            for zk, t in tables.items()
        }
        return LanceStreamingDataset(
            chunks=chunks,
            fetch_split=self.fetch_split,
            overlap_reads=self.overlap_reads,
            **kwargs,
        )

    def _chunk_table(self, name: str) -> dict:
        uri, version, tbl = self._open(f"video_chunks/{name}.lance")
        cols = tbl.to_table(columns=["recording_id", "num_frames"])
        ids, starts = _runs(cols.column("recording_id"))
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"{uri}@{version}: a recording's chunks are not contiguous"
            )
        nf = cols.column("num_frames").to_numpy()
        frames = np.add.reduceat(nf, starts) if len(starts) else np.zeros(0, np.int64)
        return {
            "uri": uri,
            "version": version,
            "k": int(nf.max()) if len(nf) else 0,
            "rows": dict(zip(ids, starts.tolist())),
            "frames": dict(zip(ids, frames.tolist())),
        }
