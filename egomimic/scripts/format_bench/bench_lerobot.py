"""LeRobot v3.0 loader bench (LeRobot venv): same JSON fields as bench_ego.py.

`LeRobotDataset` as shipped, filtered to the bimanual episodes with its own `episodes=` argument.
Phases: resolve = everything in __init__ except the HF dataset load (metadata, episode table,
cache checks); construct = `load_nested_dataset` (Arrow cache build when cold, memory-map when
warm). `--cold` empties HF_DATASETS_CACHE first.

s3:// roots go through `S3LeRobotDataset`: LeRobot 0.6.1 only reads a local root or the HF Hub,
so the shim mirrors `meta/` locally (counted in resolve), loads data parquet from s3:// via fsspec,
and hands torchcodec s3:// video paths (LeRobot's decoder cache already opens through fsspec).
The endpoint comes from FSSPEC_S3_ENDPOINT_URL and AWS_* credentials.
"""

import argparse
import os
import shutil
import time
from pathlib import Path

from egomimic.scripts.format_bench import hooks
from egomimic.scripts.format_bench.common import emit, evict, run_loader

HORIZON_KEYS = [
    f"{s}.{k}"
    for s in ("left", "right")
    for k in ("keypoints", "ee_pose", "wrist_pose")
]
CAM = "observation.images.front_1"
FPS = 30


def delta_timestamps(pattern: str, packed: bool) -> dict:
    keys = ["action"] if packed else HORIZON_KEYS
    d = {k: [i / FPS for i in range(30)] for k in keys}
    if pattern == "frames8":
        d[CAM] = [i / FPS for i in range(-7, 1)]
    if pattern == "rdt":
        d[CAM] = [-3 / FPS, 0.0]
    return d


def ds_features(root: Path) -> dict:
    import json

    return json.loads((Path(root) / "meta/info.json").read_text())["features"]


def timed_hf_load(reader_cls, box: dict):
    orig = reader_cls._load_hf_dataset

    def wrapped(self):
        t = time.time()
        out = orig(self)
        box["construct_s"] = box.get("construct_s", 0.0) + time.time() - t
        return out

    reader_cls._load_hf_dataset = wrapped


class S3:
    root = None
    local = None


def patch_s3():
    """Patches DatasetReader in place (LeRobotDataset imports the class directly), so forked
    workers inherit it."""
    import datasets
    import fsspec
    import pyarrow.dataset as pa_ds
    from lerobot.datasets import dataset_reader
    from lerobot.datasets.feature_utils import get_hf_features_from_features
    from lerobot.datasets.io_utils import hf_transform_to_torch

    # LeRobot's decoder cache opens videos with a bare fsspec.open, so s3fs defaults apply: a 50 MB
    # readahead block per random seek, ~30 MB fetched for each ~70 KB frame.
    fsspec.config.conf.setdefault("s3", {}).update(
        default_cache_type="readahead", default_block_size=256 * 2**10
    )

    def _load_hf_dataset(self):
        fs = fsspec.filesystem("s3")
        paths = sorted(fs.glob(f"{S3.root}/data/*/*.parquet"))
        ds = datasets.Dataset.from_parquet(
            [f"s3://{p}" for p in paths],
            features=get_hf_features_from_features(self._meta.features),
            # The parquet download keeps s3fs's default 50 MB block.
            storage_options={**fs.storage_options, "default_block_size": 50 * 2**20},
            filters=None
            if self.episodes is None
            else pa_ds.field("episode_index").isin(self.episodes),
        )
        ds.set_transform(hf_transform_to_torch)
        return ds

    orig_decode = dataset_reader.decode_video_frames

    def decode_s3(video_path, *a, **kw):
        rel = Path(video_path).relative_to(S3.local)
        return orig_decode(f"{S3.root}/{rel}", *a, **kw)

    dataset_reader.DatasetReader._load_hf_dataset = _load_hf_dataset
    dataset_reader.DatasetReader._check_cached_episodes_sufficient = (
        lambda self: self.hf_dataset is not None and len(self.hf_dataset) > 0
    )
    dataset_reader.decode_video_frames = decode_s3


STREAM_COLUMNS = [
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]


class S3ParquetRows:
    """Stand-in for `DatasetReader.hf_dataset` that reads rows from s3:// parquet per access,
    so numerics stream like zarr and Lance instead of being downloaded at construct.

    The row -> (file, row group) map comes from meta/episodes, so the parent reads no parquet.
    Each worker opens files lazily through pyarrow's S3 client (footer kept, column chunks
    pre-buffered concurrently) and keeps the last FB_LR_RG_CACHE row groups (default 1, enough
    for the item row, its action window and frame timestamps to share one fetch). Parquet's
    smallest unit is a row group's column chunk: one episode, ~4 MB for `action`."""

    def __init__(self, meta, episodes, s3_root: str, columns=STREAM_COLUMNS):
        import numpy as np

        eps = meta.episodes
        file_start = {}
        for ep in eps:
            f = (ep["data/chunk_index"], ep["data/file_index"])
            file_start[f] = min(file_start.get(f, 2**62), ep["dataset_from_index"])
        sel = sorted(range(len(eps)) if episodes is None else episodes)
        abs_idx, row_of, self.file = [], [], []
        for e in sel:
            ep = eps[e]
            f = (ep["data/chunk_index"], ep["data/file_index"])
            r = np.arange(ep["dataset_from_index"], ep["dataset_to_index"])
            abs_idx.append(r)
            row_of.append(r - file_start[f])
            self.file += [f] * len(r)
        self.abs = np.concatenate(abs_idx)
        self.row = np.concatenate(row_of)
        self.root = s3_root
        self.columns = columns
        self.cache_size = int(os.environ.get("FB_LR_RG_CACHE", "1"))
        self._fs = None
        self._pf = {}
        self._rg = {}

    def __len__(self):
        return len(self.abs)

    @property
    def data(self):
        import pyarrow as pa

        return pa.table({"index": self.abs})

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items() if k not in ("_fs", "_pf", "_rg")
        } | {
            "_fs": None,
            "_pf": {},
            "_rg": {},
        }

    def _file(self, f):
        import pyarrow.fs as pafs
        import pyarrow.parquet as pq

        if self._fs is None:
            ep = os.environ["FSSPEC_S3_ENDPOINT_URL"]
            self._fs = pafs.S3FileSystem(
                endpoint_override=ep.split("://")[1],
                scheme=ep.split("://")[0],
                access_key=os.environ["AWS_ACCESS_KEY_ID"],
                secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                region=os.environ.get("AWS_REGION", "us-east-1"),
            )
        if f not in self._pf:
            path = f"{self.root[len('s3://'):]}/data/chunk-{f[0]:03d}/file-{f[1]:03d}.parquet"
            pf = pq.ParquetFile(self._fs.open_input_file(path), pre_buffer=True)
            md = pf.metadata
            starts = [0]
            for i in range(md.num_row_groups):
                starts.append(starts[-1] + md.row_group(i).num_rows)
            self._pf[f] = (pf, starts)
        return self._pf[f]

    def _row_group(self, f, rg):
        key = (f, rg)
        if key in self._rg:
            self._rg[key] = self._rg.pop(key)
            return self._rg[key]
        pf, _ = self._file(f)
        t = pf.read_row_group(rg, columns=self.columns)
        self._rg[key] = t
        while len(self._rg) > self.cache_size:
            self._rg.pop(next(iter(self._rg)))
        return t

    def _rows(self, rel: list[int]):
        import bisect

        import pyarrow as pa

        parts = []
        for i in rel:
            f, r = self.file[i], int(self.row[i])
            _, starts = self._file(f)
            rg = bisect.bisect_right(starts, r) - 1
            parts.append(self._row_group(f, rg).slice(r - starts[rg], 1))
        return pa.concat_tables(parts)

    def __getitem__(self, idx):
        from lerobot.datasets.io_utils import hf_transform_to_torch

        if isinstance(idx, str):
            raise TypeError("column access is not streamed; index rows first")
        single = not isinstance(idx, list | tuple)
        rel = [int(idx)] if single else [int(i) for i in idx]
        batch = hf_transform_to_torch(self._rows(rel).to_pydict())
        return {k: v[0] for k, v in batch.items()} if single else batch


def patch_stream_parquet():
    """After patch_s3: numerics read from s3:// parquet per sample (S3ParquetRows)."""
    from lerobot.datasets import dataset_reader

    def _load_hf_dataset(self):
        return S3ParquetRows(self._meta, self.episodes, S3.root)

    dataset_reader.DatasetReader._load_hf_dataset = _load_hf_dataset


def mirror_meta(s3_root: str, local: Path):
    import fsspec

    fs = fsspec.filesystem("s3")
    if local.exists():
        shutil.rmtree(local)
    fs.get(f"{s3_root}/meta", str(local / "meta"), recursive=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--pattern", choices=["chunk1", "numeric", "frames8", "rdt"], default="chunk1"
    )
    ap.add_argument(
        "--episodes", type=Path, help="episode indices to load (whitespace separated)"
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--context", default="fork")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--real", type=Path, help="first batch samples only these episode indices"
    )
    ap.add_argument("--cold", action="store_true", help="empty HF_DATASETS_CACHE first")
    ap.add_argument(
        "--stream-parquet",
        action="store_true",
        help="s3:// only: read parquet rows per sample instead of downloading them at construct",
    )
    ap.add_argument("--evict", nargs="*", default=[])
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out")
    args = ap.parse_args()

    from lerobot.datasets import dataset_reader

    cache = Path(
        os.environ.get("HF_DATASETS_CACHE", Path(os.environ["HF_HOME"]) / "datasets")
    )
    if args.cold and cache.exists():
        shutil.rmtree(cache)
    evicted = evict(*args.evict) if args.evict else 0
    remote = args.root.startswith("s3://")
    box = {}
    episodes = (
        [int(x) for x in args.episodes.read_text().split()] if args.episodes else None
    )

    t0 = time.time()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if remote:
        patch_s3()
        if args.stream_parquet:
            patch_stream_parquet()
        S3.root, S3.local = args.root.rstrip("/"), Path(os.environ["FB_META_MIRROR"])
        mirror_meta(S3.root, S3.local)
        root = S3.local
    else:
        root = Path(args.root)
    timed_hf_load(dataset_reader.DatasetReader, box)
    ds = LeRobotDataset(
        "egoverse/aria_pilot",
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps(args.pattern, "action" in ds_features(root)),
        video_backend="torchcodec",
        # float32 timestamps ~20 min into a 200 MB concatenated file are only good to ~1e-4 s,
        # which trips the 1e-4 default.
        tolerance_s=1e-3,
    )
    if args.pattern == "numeric":
        # LeRobot has no switch to skip video keys; hide them so get_item never decodes.
        type(ds.meta).video_keys = property(lambda self: [])
    init_s = time.time() - t0
    construct_s = box.get("construct_s", 0.0)

    indices = None
    if args.real:
        real = set(int(x) for x in args.real.read_text().split())
        ep_col = ds.hf_dataset.data.column("episode_index").to_numpy()
        import numpy as np

        indices = np.flatnonzero(np.isin(ep_col, list(real))).tolist()
    res = run_loader(
        ds,
        workers=args.workers,
        context=args.context,
        batch_size=args.batch_size,
        batches=args.batches,
        warmup=args.warmup,
        hook=None if args.pattern == "numeric" or remote else hooks.lerobot_decoders,
        indices=indices,
    )
    emit(
        dict(
            label=args.label,
            stream_parquet=args.stream_parquet,
            format="lerobot" if "action" in ds_features(root) else "lerobot_unpacked",
            location="s3" if remote else "local",
            root=args.root,
            pattern=args.pattern,
            workers=args.workers,
            context=args.context,
            batch_size=args.batch_size,
            repeat=args.repeat,
            cold=args.cold,
            episodes=ds.num_episodes,
            samples=len(ds),
            evicted_gb=round(evicted / 1e9, 2),
            resolve_s=round(init_s - construct_s, 3),
            construct_s=round(construct_s, 3),
            **res,
            startup_total_s=round(init_s + res["first_batch_s"], 3),
        ),
        args.out,
    )
    os._exit(0)


if __name__ == "__main__":
    main()
