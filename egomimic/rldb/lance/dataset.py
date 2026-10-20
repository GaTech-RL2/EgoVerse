"""Lance-backed episode datasets: one leaf per recording window, drop-in for ZarrDataset.

Leaves reuse ZarrDataset's keymap semantics (horizons, history, lag, pad masks,
annotations, transforms); only storage access differs. A leaf's index runs over
eligible anchors, not every frame. Lance and torchcodec handles are opened lazily per
process: Lance is not fork-safe (https://lance.org/integrations/pytorch/), so loaders
over these datasets must use a forkserver or spawn worker context.
"""

import concurrent.futures as cf
import copy
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

from egomimic.rldb.filters import DatasetFilter, PinError, _ensure_dataset_filter
from egomimic.rldb.lance.schema import task_instance_id
from egomimic.rldb.leaf import EpisodeLeaf

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


def _mask(spans, n: int) -> np.ndarray:
    """Frames [0, n) inside any half-open (start, end) span."""
    on = np.zeros(n + 1, np.int64)
    for a, b in spans:
        if a < b:
            on[a] += 1
            on[b] -= 1
    return np.cumsum(on)[:n] > 0


def _intervals(mask: np.ndarray) -> np.ndarray:
    """(K, 2) half-open runs of True in `mask`."""
    return np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0])).reshape(-1, 2)


def _spans_of(df: pd.DataFrame | None) -> list[tuple[int, int]]:
    if df is None:
        return []
    return list(zip(df.t_start_ns.tolist(), df.t_end_ns.tolist()))


def _spans_array(spans, texts) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(start frames, end frames, texts) of (start, end) frame `spans`, empty ones dropped."""
    se = np.asarray(spans, np.int64).reshape(-1, 2)
    s, e = se[:, 0], se[:, 1]
    keep = s < e
    return s[keep], e[keep], [t for t, k in zip(texts, keep) if k]


class _Clock:
    """Recording time to frames: frame f covers [ts[f], ts[f + 1]). `ts` is None when
    every range asked of it covers the whole recording."""

    def __init__(self, n: int, first: int, last: int, ts: np.ndarray | None):
        self.n, self.first, self.last, self.ts = n, int(first), int(last), ts

    def frames(self, span: tuple[int, int] | None) -> tuple[int, int]:
        """Half-open frames whose timestamps fall in the half-open `span`; None is all."""
        if span is None or (span[0] <= self.first and span[1] > self.last):
            return 0, self.n
        a, b = np.searchsorted(self.ts, span, "left")
        return int(a), int(b)

    def time(self, frame: int) -> int:
        if frame >= self.n:
            return self.last + 1
        return self.first if frame == 0 else int(self.ts[frame])


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


class LanceRecordingDataset(EpisodeLeaf):
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
        clips: dict[str, tuple[int, int]] | None = None,
        transform_list: list | None = None,
        content_digest: str | None = None,
        annotations: tuple[np.ndarray, np.ndarray, list[str]] | None = None,
        clamp_actions: bool = False,
        segments: np.ndarray | None = None,
        action_clamp: str = "task",
        history_floor: str = "recording",
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
        self._image_keys = {k for k in videos}
        self.anchors = np.asarray(anchors, dtype=np.int64).reshape(-1, 2)
        self._cum = np.r_[0, np.cumsum(self.anchors[:, 1] - self.anchors[:, 0])]
        # Old clip episode_hashes aliased to this recording: {hash: (start, end frame)}.
        self.clips = dict(clips or {})
        # (start frames, end frames, texts) sorted by start: the tasks rows, or the
        # annotations rows under the resolver's `annotations` config.
        self._ann = annotations
        self._clamp_actions = clamp_actions
        # (S, 2) frame spans of the segments sampled, sorted by start.
        self.segments = (
            np.asarray(segments, np.int64).reshape(-1, 2)
            if segments is not None
            else np.array([[0, num_frames]], np.int64)
        )
        self.action_clamp = action_clamp
        self.history_floor = history_floor
        super().__init__(key_map=key_map, transform_list=transform_list)

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
        h.update(f"{self.action_clamp}\0{self.history_floor}\0".encode())
        h.update(self.segments.tobytes())
        if self._clamp_actions and self._ann is not None:
            h.update(b"clamp\0" + self._ann[0].tobytes() + self._ann[1].tobytes())
        return h.hexdigest()

    @property
    def intrinsics(self):
        return self._intrinsics

    def __len__(self) -> int:
        return int(self._cum[-1])

    def anchor(self, i: int) -> int:
        j = int(np.searchsorted(self._cum, i, side="right")) - 1
        return int(self.anchors[j, 0] + (i - self._cum[j]))

    def _covering(self, frame_idx: int) -> np.ndarray:
        s, e, _ = self._ann
        return np.flatnonzero((s <= frame_idx) & (frame_idx < e))

    def clip(self, episode_hash: str) -> "LanceRecordingDataset":
        """This leaf sampling only the frames of one aliased clip."""
        a, b = self.clips[episode_hash]
        leaf = copy.copy(self)
        leaf.anchors = _intervals(
            _mask(self.anchors, self.total_frames) & _mask([(a, b)], self.total_frames)
        )
        leaf._cum = np.r_[0, np.cumsum(leaf.anchors[:, 1] - leaf.anchors[:, 0])]
        return leaf

    def _annotation_text_for_frame(self, frame_idx: int):
        if self._ann is None:
            return []
        return [self._ann[2][j] for j in self._covering(frame_idx)]

    def _segment(self, frame: int) -> tuple[int, int]:
        """The innermost sampled segment holding `frame` (the recording when none does)."""
        s, e = self.segments[:, 0], self.segments[:, 1]
        on = (s <= frame) & (frame < e)
        if not on.any():
            return 0, self.total_frames
        return int(s[on].max()), int(e[on].min())

    def _history_floor(self, idx: int) -> int:
        return self._segment(idx)[0] if self.history_floor == "task" else 0

    def _chunk_end_idx(self, start_idx: int, horizon: int, key_type: str | None) -> int:
        end = super()._chunk_end_idx(start_idx, horizon, key_type)
        if key_type != "action_keys":
            return end
        if self.action_clamp == "task":
            end = min(end, self._segment(start_idx)[1])
        if not self._clamp_actions or self._ann is None:
            return end
        cover = self._covering(start_idx)
        return min(end, int(self._ann[1][cover].min())) if len(cover) else end

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


class LanceStreamingDataset(LanceRecordingDataset):
    """One recording read straight from object storage (or any Lance URI): frames come
    from K-frame video chunk rows (`video_chunks/<name>.lance`, `chunk_videos.py`)
    instead of whole MP4s, and `MultiDataset.__getitems__` fetches a DataLoader batch
    through `fetch_planned`, so a batch costs about two round trips."""

    batched_fetch = True

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

    Leaves are keyed by recording_id. Filter lambdas run once per segment of a
    recording: each tasks row, plus each untasked gap (`task` None) when
    `include_taskless_frames`. They see the recording joined with the segment, with
    SQL-style field names (`episode_hash` is the recording_id, `lab` is the site,
    `task`, `task_instance_id`, ...), so existing filter configs apply. Pins and
    excludes name a recording_id or an old clip episode_hash, meaning that clip's time
    range.

    `action_clamp` "task" ends action chunks at the end of the anchor's segment
    (repeat-last padding, flagged in `action_pad_mask`); "recording" runs them on.
    `history_floor` "recording" lets history reach into earlier segments; "task" pads
    and masks it at the segment start.

    Each table is read at `versions[name]` (a number or tag), else at `tag`, else at its
    latest version. Tags are resolved to numbers once, here, so a tag moved later does
    not change what an existing resolver reads.

    `annotations` switches annotation keys from the task description to the rows of
    `annotations.lance` covering the frame, in start order. Its keys (all optional):
    `types` (default ["language"]), `vendors`, `min_quality` (ungraded rows fail it),
    `clamp_actions` (stop action chunks at the end of the covering annotation) and
    `require` (sample only anchors some annotation covers; drops leaves with none).
    """

    ANNOTATION_DEFAULTS = {
        "types": ["language"],
        "vendors": None,
        "min_quality": None,
        "clamp_actions": False,
        "require": False,
    }

    def __init__(
        self,
        lance_root: str | Path,
        lab: str = "aria",
        key_map: dict | None = None,
        transform_list: list | None = None,
        cameras: dict[str, str] | None = None,
        versions: dict[str, int | str] | None = None,
        tag: str | None = None,
        annotations: dict | None = None,
        include_taskless_frames: bool = False,
        action_clamp: str = "task",
        history_floor: str = "recording",
    ):
        if action_clamp not in ("task", "recording"):
            raise ValueError(f"action_clamp {action_clamp!r}: 'task' or 'recording'")
        if history_floor not in ("task", "recording"):
            raise ValueError(f"history_floor {history_floor!r}: 'task' or 'recording'")
        self.root = str(lance_root).rstrip("/")
        self.lab = lab
        self.key_map = key_map
        self.transform_list = transform_list
        self.cameras = cameras or {"images.front_1": "front_1"}
        self.tag = tag
        self.versions = dict(versions or {})
        self.include_taskless_frames = include_taskless_frames
        self.action_clamp = action_clamp
        self.history_floor = history_floor
        self.annotations = None
        if annotations is not None:
            unknown = set(annotations) - set(self.ANNOTATION_DEFAULTS)
            if unknown:
                raise ValueError(f"unknown annotations options {sorted(unknown)}")
            self.annotations = {**self.ANNOTATION_DEFAULTS, **dict(annotations)}

    def _open(self, name: str):
        uri = f"{self.root}/{name}"
        want = self.versions.get(name, self.tag)
        if not isinstance(want, int):
            want = open_table(uri, want).version
            self.versions[name] = want
        return uri, want, _table(uri, want)

    def recordings(self) -> pd.DataFrame:
        """One row per recording of `lab`, with SQL-style names (`lab` is the site)."""
        _, _, rec = self._open("recordings.lance")
        r = rec.to_table().to_pandas()
        r = r[r.lab == self.lab].rename(columns={"lab": "vendor", "site": "lab"})
        return r.reset_index(drop=True)

    def _table_df(self, name: str, columns: list[str]) -> pd.DataFrame:
        _, _, tbl = self._open(name)
        return tbl.to_table(columns=columns).to_pandas()

    def dataset_keys(self, hashes) -> dict[str, str]:
        """{pin: dataset key} for pins naming a recording_id or an aliased clip hash."""
        hashes = set(hashes)
        if not hashes:
            return {}
        rids = set(self._table_df("recordings.lance", ["recording_id"]).recording_id)
        a = self._table_df("aliases.lance", ["episode_hash", "recording_id"])
        out = dict(zip(a.episode_hash, a.recording_id))
        out.update({r: r for r in rids})
        return {h: out[h] for h in hashes if h in out}

    def _annotation_rows(self) -> dict[str, pd.DataFrame]:
        """{recording_id: rows passing the `annotations` filters, sorted by start}."""
        cfg = self.annotations
        _, _, tbl = self._open("annotations.lance")
        df = tbl.to_table().to_pandas()
        keep = df.annotation_type.isin(cfg["types"])
        if cfg["vendors"] is not None:
            keep &= df.annotation_vendor.isin(cfg["vendors"])
        if cfg["min_quality"] is not None:
            keep &= df.quality_score.notna() & (df.quality_score >= cfg["min_quality"])
        df = df[keep].sort_values(["recording_id", "t_start_ns", "t_end_ns"])
        return {rid: g for rid, g in df.groupby("recording_id")}

    def _horizon(self) -> int:
        hs = [
            int(s["horizon"])
            for s in self.key_map.values()
            if s.get("key_type") == "action_keys" and s.get("horizon")
        ]
        return max(hs) if hs else 1

    def _ranges(self, hashes, recs: pd.DataFrame, what: str) -> dict[str, list]:
        """{recording_id: [(t_start_ns, t_end_ns) | None]} for pins or excludes; None
        is the whole recording. A hash is a recording_id or an aliased clip hash."""
        out: dict[str, list] = {}
        if not hashes:
            return out
        rids = set(recs.recording_id)
        a = self._table_df(
            "aliases.lance", ["episode_hash", "recording_id", "t_start_ns", "t_end_ns"]
        )
        a = a[a.episode_hash.isin(hashes) & a.recording_id.isin(rids)]
        for row in a.itertuples():
            out.setdefault(row.recording_id, []).append(
                (int(row.t_start_ns), int(row.t_end_ns))
            )
        for h in hashes & rids:
            out.setdefault(h, []).append(None)
        if what == "pins":
            missing = sorted(hashes - rids - set(a.episode_hash))
            if missing:
                raise PinError(f"pinned episodes not in the Lance catalog: {missing}")
        return out

    def resolve(
        self,
        filters: DatasetFilter | None = None,
        expected_embodiment: str | None = None,
    ):
        """{recording_id: leaf}. A recording's samples are the eligible anchors inside
        its segments (tasks rows, plus the untasked gaps with `include_taskless_frames`)
        that pass the filter lambdas, narrowed to pinned and minus excluded ranges."""
        filters = _ensure_dataset_filter(filters)
        recs = self.recordings()
        if expected_embodiment is not None:
            recs = recs[recs.embodiment.str.lower() == expected_embodiment.lower()]
        pins = self._ranges(set(filters.episode_hashes), recs, "pins")
        excludes = self._ranges(set(filters.exclude_hashes), recs, "excludes")
        if filters.episode_hashes:
            recs = recs[recs.recording_id.isin(set(pins))]
        tasks = {
            rid: g.sort_values("t_start_ns")
            for rid, g in self._table_df(
                "tasks.lance",
                [
                    "task_instance_id",
                    "recording_id",
                    "t_start_ns",
                    "t_end_ns",
                    "task_name",
                    "task_description",
                    "objects",
                ],
            ).groupby("recording_id")
        }
        clips = {
            rid: g
            for rid, g in self._table_df(
                "aliases.lance",
                ["episode_hash", "recording_id", "t_start_ns", "t_end_ns"],
            ).groupby("recording_id")
        }

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
        for row in recs.itertuples():
            if counts.get(row.recording_id) != row.num_frames:
                raise ValueError(
                    f"{row.recording_id}: {counts.get(row.recording_id, 0)} steps rows at {uri}@{version}, num_frames {row.num_frames}"
                )
        all_valid = {
            h: cols.column(c).to_numpy(zero_copy_only=False)
            for h, c in VALIDITY.items()
        }
        ann_rows = self._annotation_rows() if self.annotations is not None else {}
        clocks = self._clocks(
            steps,
            {rid: (int(starts[rid]), int(counts[rid])) for rid in recs.recording_id},
            {
                rid: [
                    *_spans_of(tasks.get(rid)),
                    *_spans_of(clips.get(rid)),
                    *_spans_of(ann_rows.get(rid)),
                    *(r for r in pins.get(rid, []) + excludes.get(rid, []) if r),
                ]
                for rid in recs.recording_id
            },
        )
        horizon = self._horizon()
        tables = self._leaf_tables()

        out, no_ann, no_anchor = {}, [], []
        for row in recs.itertuples():
            rid, clock = row.recording_id, clocks[row.recording_id]
            r0, n = int(starts[rid]), int(row.num_frames)
            segs = self._segments(row, tasks.get(rid), clock)
            passing = [
                (a, b)
                for seg, (a, b) in segs
                if a < b and filters.passes_lambdas({**row._asdict(), **seg})
            ]
            mask = _mask(passing, n)
            if filters.episode_hashes:
                mask &= _mask([clock.frames(r) for r in pins[rid]], n)
            mask &= ~_mask([clock.frames(r) for r in excludes.get(rid, [])], n)
            valid = {h: a[r0 : r0 + n] for h, a in all_valid.items()}
            eligible = _mask(
                eligible_intervals(
                    valid, HANDS_OF.get(row.embodiment, ("left", "right")), horizon
                ),
                n,
            )
            anchors = _intervals(mask & eligible)
            task_text = [
                t.task_description or t.task_name
                for t in (tasks[rid].itertuples() if rid in tasks else ())
            ]
            ann_kw = dict(
                annotations=_spans_array(
                    [clock.frames(r) for r in _spans_of(tasks.get(rid))], task_text
                )
            )
            if self.annotations is not None:
                g = ann_rows.get(rid)
                spans = _spans_array(
                    [clock.frames(r) for r in _spans_of(g)],
                    g.text.tolist() if g is not None else [],
                )
                if self.annotations["require"]:
                    anchors = _intervals(_mask(anchors, n) & _mask(zip(*spans[:2]), n))
                    if not len(anchors):
                        no_ann.append(rid)
                        continue
                ann_kw = dict(
                    annotations=spans, clamp_actions=self.annotations["clamp_actions"]
                )
            if not len(anchors):
                no_anchor.append(rid)
                continue
            intr = dict(row.intrinsics) if row.intrinsics is not None else {}
            vids = dict(row.video_path)
            cam = next(iter(self.cameras.values()))
            out[rid] = self._leaf(
                row,
                tables,
                name=rid,
                steps_uri=uri,
                steps_version=version,
                row_start=r0,
                num_frames=n,
                anchors=anchors,
                embodiment=row.embodiment,
                fps=float(row.fps),
                intrinsics=np.asarray(intr[cam]) if cam in intr else None,
                videos={zk: f"{self.root}/{vids[c]}" for zk, c in self.cameras.items()},
                task_text=task_text[0] if task_text else "",
                clips={
                    c.episode_hash: clock.frames((c.t_start_ns, c.t_end_ns))
                    for c in (clips[rid].itertuples() if rid in clips else ())
                },
                key_map=self.key_map,
                transform_list=self.transform_list,
                content_digest=getattr(row, "content_digest", None),
                segments=sorted(passing),
                action_clamp=self.action_clamp,
                history_floor=self.history_floor,
                **ann_kw,
            )
        if no_anchor:
            logger.warning(
                "%d recordings have no sampleable anchors, e.g. %s",
                len(no_anchor),
                no_anchor[:5],
            )
        if self.annotations is not None and self.annotations["require"]:
            logger.info(
                "kept %d/%d recordings with annotated anchors; dropped e.g. %s",
                len(out),
                len(out) + len(no_ann),
                no_ann[:5],
            )
        return out

    def _segments(self, row, tasks: pd.DataFrame | None, clock: "_Clock") -> list:
        """[(filter fields, (frame start, frame end))]: one per tasks row, plus one per
        untasked gap when `include_taskless_frames`."""
        segs, covered = [], np.zeros(clock.n, bool)
        for t in tasks.itertuples() if tasks is not None else ():
            a, b = clock.frames((t.t_start_ns, t.t_end_ns))
            covered[a:b] = True
            seg = dict(
                task_instance_id=t.task_instance_id,
                t_start_ns=t.t_start_ns,
                t_end_ns=t.t_end_ns,
                task=t.task_name,
                task_description=t.task_description,
                objects=t.objects,
            )
            segs.append((seg, (a, b)))
        if self.include_taskless_frames:
            for a, b in _intervals(~covered):
                t0 = clock.time(a)
                seg = dict(
                    task_instance_id=task_instance_id(row.recording_id, None, t0),
                    t_start_ns=t0,
                    t_end_ns=clock.time(b),
                    task=None,
                    task_description=None,
                    objects=[],
                )
                segs.append((seg, (int(a), int(b))))
        for seg, _ in segs:
            seg["episode_hash"] = row.recording_id
        return segs

    def _clocks(self, steps, rows: dict, ranges: dict) -> dict[str, "_Clock"]:
        """{recording_id: _Clock}. Reads every timestamp only of recordings with a range
        that starts or ends inside them; the rest map time to frames from their ends."""
        rids = list(rows)
        ends = (
            steps.take(
                [
                    r
                    for rid in rids
                    for r in (rows[rid][0], rows[rid][0] + rows[rid][1] - 1)
                ],
                columns=["timestamp_ns"],
            )
            .column(0)
            .to_numpy()
            .reshape(-1, 2)
        )
        bounds = dict(zip(rids, ends))
        need = [
            rid
            for rid in rids
            if any(
                not (t0 <= bounds[rid][0] and t1 > bounds[rid][1])
                for t0, t1 in ranges[rid]
            )
        ]
        ts = {}
        if need:
            take = np.concatenate([np.arange(rows[r][0], sum(rows[r])) for r in need])
            col = (
                steps.take(take.tolist(), columns=["timestamp_ns"]).column(0).to_numpy()
            )
            at = np.cumsum([0] + [rows[r][1] for r in need])
            ts = {r: col[at[j] : at[j + 1]] for j, r in enumerate(need)}
        return {rid: _Clock(rows[rid][1], *bounds[rid], ts.get(rid)) for rid in rids}

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
