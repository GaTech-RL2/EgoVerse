"""Time whole-dataset edits in zarr vs LeRobot v3.1 vs Lance: replace head_pose, append a new
per-row column, rewrite every episode's annotations.

`make` builds the same annotated rows in every format (no video, which none of these edits
touch); `update --op` applies one edit and times it end to end, then verifies it; `fit` projects
to a target row count. Roots are local paths or s3://bucket/prefix; s3 goes to FB_S3_ENDPOINT
(plain http, MinIO) with the R2_* credentials, as in s3patch.py.

Rows are real: fake episodes are contiguous slices of the pilot recordings (Lance steps table),
with lengths cycling through the pilot zarr manifest's, so every format compresses like real
data and zarr has the real object count. The global row index g (`index` in LeRobot,
timestamp_ns // DT_NS in Lance, episode start + frame in zarr) maps back to the source row; edit
generation k writes source head_pose + k * 1e-3 and annotation text tagged with k.

Annotations are one span per SPAN_FRAMES frames, stored as each format does:
  zarr     `annotations` array of JSON bytes per episode (ZarrWriter._write_annotations)
  lerobot  `language_persistent` (the episode's spans, broadcast to every frame) and
           `language_events` columns in the data parquet (LeRobot 0.6 annotation writer)
  lance    the `annotations` column of tasks.lance, one row per episode

What each edit writes:
  zarr     objects built in memory and PUT, nothing read back except the group zarr.json on
           append (its `features` attr lists every array): replace = 1 shard per episode,
           append = array zarr.json + shard + group zarr.json, annotations = zarr.json + shard
  lerobot  every data parquet file in full (one row group per episode), plus meta/episodes
           stats, stats.json and info.json where the edit changes them
  lance    replace: one new data file per fragment (merge_columns + Merge commit), then
           drop_columns + alter_columns, both metadata only; append: merge_columns + Merge;
           annotations: the same column replace on tasks.lance
"""

import argparse
import csv
import functools
import json
import os
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pyarrow.parquet as pq

DT_NS = 33_333_333
FPS = 30
CHUNK = 100
SPAN_FRAMES = 300
PARQUET_MB = 100
CHUNKS_SIZE = 1000
LANCE_ROWS_PER_FILE = 1024 * 1024
STATS = ("min", "max", "mean", "std")
HANDS = ("wrist_pose", "ee_pose", "keypoints", "aria_keypoints")
# Flattened HUMAN_STEPS, as to_lerobot.py lays it out (copied: to_lerobot imports lerobot).
VECTORS = {
    "head_pose": 7,
    "eye_gaze": 3,
    **{f"{s}.{f}": d for s in ("left", "right") for f, d in zip(HANDS, (7, 7, 63, 63))},
}
SCALARS = ("left.confidence", "right.confidence")
BOOLS = ("left_valid", "right_valid", "head_pose_valid")
FORMATS = ("zarr", "lerobot", "lerobot_zstd", "lance")
OPS = ("replace", "append", "annotations")
COMPRESSION = {"lerobot": "snappy", "lerobot_zstd": "zstd"}
LANGUAGE_COLS = ["language_persistent", "language_events"]
# lerobot/datasets/language.py row types; tool_calls is its pa.string() fallback for pa.json_().
LANG_PERSISTENT = pa.list_(
    pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string()),
            pa.field("style", pa.string()),
            pa.field("timestamp", pa.float32(), nullable=False),
            pa.field("camera", pa.string()),
            pa.field("tool_calls", pa.list_(pa.string())),
        ]
    )
)
LANG_EVENTS = pa.list_(
    pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string()),
            pa.field("style", pa.string()),
            pa.field("camera", pa.string()),
            pa.field("tool_calls", pa.list_(pa.string())),
        ]
    )
)


# --------------------------------------------------------------------------- rows
def nested(t: pa.Table, dotted: str) -> pa.Array:
    head, *rest = dotted.split(".")
    col = t.column(head).combine_chunks()
    for p in rest:
        col = col.field(p)
    return col


def vec(arr: pa.FixedSizeListArray, d: int) -> np.ndarray:
    # .values keeps null-backed slots (flatten() drops them) but ignores the array offset.
    v = np.asarray(arr.values)
    return v[arr.offset * d : (arr.offset + len(arr)) * d].reshape(-1, d)


def fsl(a: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1)), a.shape[1])


def feat_name(k: int) -> str:
    return f"feat_{k}"


def spans(ep: int, n: int, k: int) -> list[tuple[str, int, int]]:
    """(text, start frame, end frame inclusive) per span, as ZarrWriter takes them."""
    return [
        (
            f"ep {ep} subtask {i} v{k}: pick up the folded shirt and place it on the stack",
            i * SPAN_FRAMES,
            min(n, (i + 1) * SPAN_FRAMES) - 1,
        )
        for i in range(n // SPAN_FRAMES)
    ]


class Layout:
    """Fake episodes cut from source recordings: episode i has length ep_lens[i % len] (clipped
    to its source recording, the last one to `rows`) and starts at a fixed offset inside
    recording i % R."""

    def __init__(self, steps: pa.Table, ep_lens: list[int], rows: int):
        self.pool = {k: vec(nested(steps, k), d) for k, d in VECTORS.items()}
        self.pool.update(
            {k: pc.fill_null(nested(steps, k), 0.0).to_numpy() for k in SCALARS}
        )
        self.pool.update(
            {
                k: pc.fill_null(nested(steps, k), False).to_numpy(zero_copy_only=False)
                for k in BOOLS
            }
        )
        ids = steps.column("recording_id").to_numpy(zero_copy_only=False)
        rec_starts = np.concatenate([[0], np.flatnonzero(ids[1:] != ids[:-1]) + 1])
        rec_lens = np.diff(np.append(rec_starts, len(ids)))
        lens, src, total = [], [], 0
        while total < rows:
            ep = len(lens)
            r = ep % len(rec_lens)
            n = int(min(ep_lens[ep % len(ep_lens)], rec_lens[r], rows - total))
            off = rec_starts[r] + (ep * 7919) % (rec_lens[r] - n + 1)
            src.append(np.arange(off, off + n))
            lens.append(n)
            total += n
        self.lens = np.asarray(lens)
        self.starts = np.concatenate([[0], np.cumsum(self.lens)[:-1]])
        self.src = np.concatenate(src)

    @property
    def n_eps(self) -> int:
        return len(self.lens)

    def g(self, ep: int) -> np.ndarray:
        return np.arange(self.starts[ep], self.starts[ep] + self.lens[ep])

    def episode(self, ep: int) -> dict[str, np.ndarray]:
        g = self.g(ep)
        cols = {k: v[self.src[g]] for k, v in self.pool.items()}
        cols["g"] = g
        return cols

    def new_head_pose(self, g: np.ndarray, k: int) -> np.ndarray:
        return self.pool["head_pose"][self.src[g]] + np.float32(k * 1e-3)

    def spans(self, ep: int, k: int) -> list[tuple[str, int, int]]:
        return spans(ep, int(self.lens[ep]), k)


def pilot_layout(pilot: Path, rows: int) -> Layout:
    import lance

    steps = lance.dataset(str(pilot / "lance/steps/aria.lance")).to_table()
    with open(pilot / "manifest.csv") as f:
        ep_lens = [int(r["num_frames"]) for r in csv.DictReader(f)]
    return Layout(steps, ep_lens, rows)


# --------------------------------------------------------------------------- storage
def is_s3(root: str) -> bool:
    return root.startswith("s3://")


def creds() -> tuple[str, str, str]:
    return (
        os.environ["FB_S3_ENDPOINT"],
        os.environ["R2_ACCESS_KEY_ID"],
        os.environ["R2_SECRET_ACCESS_KEY"],
    )


def arrow_fs(root: str) -> tuple[pafs.FileSystem, str]:
    if not is_s3(root):
        return pafs.LocalFileSystem(), str(Path(root).absolute())
    ep, key, secret = creds()
    scheme, host = ep.split("://")
    fs = pafs.S3FileSystem(
        access_key=key,
        secret_key=secret,
        endpoint_override=host,
        scheme=scheme,
        region="us-east-1",
    )
    return fs, root[len("s3://") :]


def lance_opts(root: str) -> dict | None:
    if not is_s3(root):
        return None
    ep, key, secret = creds()
    return {
        "access_key_id": key,
        "secret_access_key": secret,
        "endpoint": ep,
        "region": "us-east-1",
        "allow_http": "true",
    }


def obj_store(root: str):
    from obstore.store import LocalStore, S3Store

    if not is_s3(root):
        return LocalStore(prefix=Path(root).absolute(), mkdir=True)
    ep, key, secret = creds()
    bucket, prefix = root[len("s3://") :].split("/", 1)
    return S3Store(
        bucket,
        prefix=prefix.rstrip("/"),
        endpoint=ep,
        access_key_id=key,
        secret_access_key=secret,
        region="us-east-1",
        client_options={"allow_http": True},
    )


def mkparent(fs, path: str) -> None:
    if isinstance(fs, pafs.LocalFileSystem):
        fs.create_dir(path.rsplit("/", 1)[0], recursive=True)


def write_json(fs, path: str, obj) -> None:
    mkparent(fs, path)
    with fs.open_output_stream(path) as f:
        f.write(json.dumps(obj, indent=2).encode())


def read_json(fs, path: str):
    with fs.open_input_stream(path) as f:
        return json.loads(f.read())


def emit(rec: dict, out: str | None) -> None:
    # common.emit imports torch, which takes minutes on a loaded node.
    line = json.dumps(
        {"node": socket.gethostname().split(".")[0], "t": time.time(), **rec}
    )
    print(line, flush=True)
    if out:
        with open(out, "a") as f:
            f.write(line + "\n")


def pmap(fn, items, workers: int) -> list:
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


# --------------------------------------------------------------------------- zarr
def ep_name(ep: int) -> str:
    return f"ep-{ep:07d}.zarr"


def padded(hp: np.ndarray) -> np.ndarray:
    """float64, zero-padded to a whole inner chunk, as ZarrWriter stores it."""
    out = np.zeros((-(-len(hp) // CHUNK) * CHUNK, hp.shape[1]))
    out[: len(hp)] = hp
    return out


def shard(chunks: list[bytes]) -> bytes:
    """zarr 3 sharding_indexed with the default index codecs (bytes + crc32c, at the end):
    zstd(0)-compressed inner chunks, then (offset, nbytes) uint64 pairs and their crc32c."""
    import google_crc32c
    from numcodecs import Zstd

    z = Zstd(level=0)
    enc = [z.encode(c) for c in chunks]
    offsets = np.cumsum([0] + [len(e) for e in enc[:-1]])
    index = np.stack([offsets, [len(e) for e in enc]], 1).astype("<u8").tobytes()
    return b"".join(enc) + index + struct.pack("<I", google_crc32c.value(index))


def numeric_shard(a: np.ndarray) -> bytes:
    a = np.ascontiguousarray(a, "<f8")
    return shard([a[i : i + CHUNK].tobytes() for i in range(0, len(a), CHUNK)])


def annotation_items(sp: list[tuple[str, int, int]]) -> list[bytes]:
    return [
        json.dumps(
            {"text": t, "start_idx": s, "end_idx": e},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        for t, s, e in sp
    ]


def vlen_shard(items: list[bytes]) -> bytes:
    """One inner chunk holding every item (ZarrWriter's annotations layout), vlen-bytes encoded."""
    body = struct.pack("<I", len(items)) + b"".join(
        struct.pack("<I", len(b)) + b for b in items
    )
    return shard([body])


@functools.cache
def array_meta(shape: tuple[int, ...], vlen: bool) -> bytes:
    """zarr.json of an array as ZarrWriter creates it, rendered by zarr once per shape."""
    import zarr
    from zarr.core.dtype import VariableLengthBytes

    d = {}
    edge = (max(1, shape[0]),)
    zarr.create_array(
        zarr.storage.MemoryStore(d),
        shape=shape,
        dtype=VariableLengthBytes() if vlen else "float64",
        chunks=edge if vlen else (CHUNK, *shape[1:]),
        shards=edge if vlen else shape,
    )
    return d["zarr.json"].to_bytes()


def feature(arr: np.ndarray | None) -> dict:
    if arr is None:
        return {
            "dtype": "json",
            "shape": [0],
            "names": ["json"],
            "format": "annotation_v1",
        }
    return {"dtype": "float64", "shape": [arr.shape[1]], "names": ["dim_0"]}


def put_array(store, path: str, meta: bytes, data: bytes | None) -> None:
    import obstore

    obstore.put(store, f"{path}/zarr.json", meta)
    key = f"{path}/c/0" if path.endswith("annotations") else f"{path}/c/0/0"
    if data is None:
        # An empty array has no chunk object; S3 deletes are idempotent, local ones are not.
        try:
            obstore.delete(store, key)
        except FileNotFoundError:
            pass
    else:
        obstore.put(store, key, data)


def put_annotations(store, ep: int, sp: list[tuple[str, int, int]]) -> None:
    items = annotation_items(sp)
    put_array(
        store,
        f"{ep_name(ep)}/annotations",
        array_meta((len(items),), True),
        vlen_shard(items) if items else None,
    )


def make_zarr(root: str, lay: Layout, workers: int) -> None:
    import zarr

    store = obj_store(root)
    zstore = zarr.storage.ObjectStore(store)

    def one(ep):
        hp = padded(lay.pool["head_pose"][lay.src[lay.g(ep)]])
        sp = lay.spans(ep, 0)
        features = {"obs_head_pose": feature(hp), "annotations": feature(None)}
        features["annotations"]["shape"] = [len(sp)]
        zarr.open_group(
            zstore,
            path=ep_name(ep),
            mode="w",
            attributes={
                "embodiment": "human_bimanual",
                "total_frames": int(lay.lens[ep]),
                "fps": FPS,
                "features": features,
            },
        )
        put_array(
            store,
            f"{ep_name(ep)}/obs_head_pose",
            array_meta(hp.shape, False),
            numeric_shard(hp),
        )
        put_annotations(store, ep, sp)

    pmap(one, range(lay.n_eps), workers)


def update_zarr(root: str, lay: Layout, op: str, k: int, workers: int) -> dict:
    import obstore

    store = obj_store(root)

    def one(ep):
        name = ep_name(ep)
        if op == "annotations":
            put_annotations(store, ep, lay.spans(ep, k))
            return 2
        hp = padded(lay.new_head_pose(lay.g(ep), k))
        if op == "replace":
            obstore.put(store, f"{name}/obs_head_pose/c/0/0", numeric_shard(hp))
            return 1
        put_array(
            store,
            f"{name}/{feat_name(k)}",
            array_meta(hp.shape, False),
            numeric_shard(hp),
        )
        group = json.loads(obstore.get(store, f"{name}/zarr.json").bytes().to_bytes())
        group["attributes"]["features"][feat_name(k)] = feature(hp)
        obstore.put(store, f"{name}/zarr.json", json.dumps(group).encode())
        return 3

    puts = pmap(one, range(lay.n_eps), workers)
    return {"objects_written": sum(puts), "gets": lay.n_eps if op == "append" else 0}


def verify_zarr(root: str, lay: Layout, op: str, k: int) -> None:
    import zarr

    store = zarr.storage.ObjectStore(obj_store(root))
    for ep in {0, lay.n_eps // 2, lay.n_eps - 1}:
        g = zarr.open_group(store, path=ep_name(ep), mode="r")
        if op == "annotations":
            got = [json.loads(b) for b in g["annotations"][:]]
            want = [json.loads(b) for b in annotation_items(lay.spans(ep, k))]
            assert got == want, (ep, got[:1], want[:1])
            continue
        key = "obs_head_pose" if op == "replace" else feat_name(k)
        assert key in g.attrs["features"]
        np.testing.assert_array_equal(
            g[key][:], padded(lay.new_head_pose(lay.g(ep), k))
        )


# --------------------------------------------------------------------------- lerobot
def language_columns(
    sp: list[tuple[str, int, int]], n: int
) -> tuple[pa.Array, pa.Array]:
    """The episode's spans broadcast to all n frames, and no events, as LeRobot's writer does."""
    rows = pa.array(
        [
            {
                "role": "assistant",
                "content": t,
                "style": "subtask",
                "timestamp": s / FPS,
                "camera": None,
                "tool_calls": None,
            }
            for t, s, _ in sp
        ],
        LANG_PERSISTENT.value_type,
    )
    m = len(rows)
    persistent = pa.ListArray.from_arrays(
        pa.array(np.arange(n + 1, dtype=np.int32) * m),
        rows.take(pa.array(np.tile(np.arange(m), n))),
        type=LANG_PERSISTENT,
    )
    events = pa.ListArray.from_arrays(
        pa.array(np.zeros(n + 1, np.int32)),
        pa.array([], LANG_EVENTS.value_type),
        type=LANG_EVENTS,
    )
    return persistent, events


def lerobot_table(cols: dict, ep: int, sp: list) -> pa.Table:
    n = len(cols["g"])
    t = {k: fsl(cols[k]) for k in VECTORS}
    t.update({k: pa.array(cols[k]) for k in SCALARS + BOOLS})
    t["timestamp_ns"] = pa.array(cols["g"] * DT_NS)
    t["timestamp"] = pa.array(np.arange(n, dtype=np.float32) / np.float32(FPS))
    t["frame_index"] = pa.array(np.arange(n, dtype=np.int64))
    t["episode_index"] = pa.array(np.full(n, ep, np.int64))
    t["index"] = pa.array(cols["g"])
    t["task_index"] = pa.array(np.zeros(n, np.int64))
    t["language_persistent"], t["language_events"] = language_columns(sp, n)
    return pa.table(t)


def vec_stats(a: np.ndarray) -> dict[str, list[float]]:
    a = a.astype(np.float64)
    return {
        "min": a.min(0).tolist(),
        "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(),
        "std": a.std(0).tolist(),
    }


def data_path(file_no: int) -> str:
    c, f = divmod(file_no, CHUNKS_SIZE)
    return f"data/chunk-{c:03d}/file-{f:03d}.parquet"


def episodes_meta(stats: list[dict], lens: np.ndarray, file_no: np.ndarray) -> pa.Table:
    """meta/episodes rows as LeRobot v3 stores them: one stats/<key>/<stat> column per stat."""
    ends = np.cumsum(lens)
    cols = {
        "episode_index": np.arange(len(lens)),
        "length": lens,
        "dataset_from_index": ends - lens,
        "dataset_to_index": ends,
        "data/chunk_index": file_no // CHUNKS_SIZE,
        "data/file_index": file_no % CHUNKS_SIZE,
    }
    t = {k: pa.array(np.asarray(v, np.int64)) for k, v in cols.items()}
    for key in stats[0]:
        for s in STATS:
            t[f"stats/{key}/{s}"] = pa.array(
                [st[key][s] for st in stats], pa.list_(pa.float64())
            )
        t[f"stats/{key}/count"] = pa.array(
            [[int(n)] for n in lens], pa.list_(pa.int64())
        )
    return pa.table(t)


def aggregate(stats: list[dict], lens: np.ndarray, key: str) -> dict:
    w = lens / lens.sum()
    mean = w @ np.array([s[key]["mean"] for s in stats])
    m2 = w @ np.array(
        [np.square(s[key]["std"]) + np.square(s[key]["mean"]) for s in stats]
    )
    return {
        "min": np.min([s[key]["min"] for s in stats], 0).tolist(),
        "max": np.max([s[key]["max"] for s in stats], 0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(m2 - mean**2, 0)).tolist(),
    }


def write_meta(fs, base: str, stats: list[dict], lens: np.ndarray, file_no: np.ndarray):
    meta = episodes_meta(stats, lens, file_no)
    rows_per_meta = 50_000
    written = 0
    for j in range(0, meta.num_rows, rows_per_meta):
        p = f"{base}/meta/episodes/chunk-000/file-{j // rows_per_meta:03d}.parquet"
        mkparent(fs, p)
        pq.write_table(
            meta.slice(j, rows_per_meta), p, filesystem=fs, compression="snappy"
        )
        written += fs.get_file_info(p).size
    write_json(
        fs, f"{base}/meta/stats.json", {k: aggregate(stats, lens, k) for k in stats[0]}
    )
    return written


def read_meta(fs, base: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    meta = pq.read_table(f"{base}/meta/episodes/chunk-000", filesystem=fs)
    keys = sorted(
        {c.split("/")[1] for c in meta.column_names if c.startswith("stats/")}
    )
    cols = {
        (k, s): meta.column(f"stats/{k}/{s}").to_pylist() for k in keys for s in STATS
    }
    stats = [
        {k: {s: cols[k, s][i] for s in STATS} for k in keys}
        for i in range(meta.num_rows)
    ]
    file_no = (
        meta.column("data/chunk_index").to_numpy() * CHUNKS_SIZE
        + meta.column("data/file_index").to_numpy()
    )
    return stats, meta.column("length").to_numpy(), file_no


def write_parquet(fs, path: str, tables, compression: str) -> None:
    """One row group per episode, as LeRobot's writer leaves it."""
    mkparent(fs, path)
    w = None
    for t in tables:
        if w is None:
            w = pq.ParquetWriter(
                path,
                t.schema,
                filesystem=fs,
                compression=compression,
                use_dictionary=True,
            )
        w.write_table(t)
    w.close()


def make_lerobot(root: str, lay: Layout, workers: int, compression: str) -> None:
    fs, base = arrow_fs(root)
    probe = pa.BufferOutputStream()
    pq.write_table(
        lerobot_table(lay.episode(0), 0, lay.spans(0, 0)),
        probe,
        compression=compression,
    )
    rows_per_file = PARQUET_MB * 2**20 * lay.lens[0] / probe.getvalue().size
    file_no, f, in_file = [], 0, 0
    for n in lay.lens:
        if in_file and in_file + n > rows_per_file:
            f, in_file = f + 1, 0
        file_no.append(f)
        in_file += n
    file_no = np.asarray(file_no)

    def one(f):
        eps = np.flatnonzero(file_no == f)
        cols = [lay.episode(ep) for ep in eps]
        write_parquet(
            fs,
            f"{base}/{data_path(f)}",
            (lerobot_table(c, ep, lay.spans(ep, 0)) for c, ep in zip(cols, eps)),
            compression,
        )
        return [{k: vec_stats(c[k]) for k in VECTORS} for c in cols]

    stats = [s for per in pmap(one, range(file_no[-1] + 1), workers) for s in per]
    write_meta(fs, base, stats, lay.lens, file_no)
    features = {k: {"dtype": "float32", "shape": [d]} for k, d in VECTORS.items()}
    features.update({k: {"dtype": "language"} for k in LANGUAGE_COLS})
    write_json(
        fs,
        f"{base}/meta/info.json",
        {
            "codebase_version": "v3.1",
            "total_episodes": lay.n_eps,
            "total_frames": int(lay.lens.sum()),
            "chunks_size": CHUNKS_SIZE,
            "data_files_size_in_mb": PARQUET_MB,
            "n_files": int(file_no[-1] + 1),
            "compression": compression,
            "features": features,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        },
    )


def update_lerobot(root: str, lay: Layout, op: str, k: int, workers: int) -> dict:
    fs, base = arrow_fs(root)
    info = read_json(fs, f"{base}/meta/info.json")
    key = {"replace": "head_pose", "append": feat_name(k)}.get(op)

    def one(f):
        p = f"{base}/{data_path(f)}"
        size_in = fs.get_file_info(p).size
        t = pq.read_table(p, filesystem=fs)
        eps = t.column("episode_index").to_numpy()
        cuts = np.flatnonzero(np.diff(eps)) + 1
        bounds = list(zip(np.r_[0, cuts], np.r_[cuts, len(eps)]))
        per_ep = []
        if op == "annotations":
            parts = [
                language_columns(lay.spans(int(eps[a]), k), b - a) for a, b in bounds
            ]
            for i, name in enumerate(LANGUAGE_COLS):
                col = pa.chunked_array([part[i] for part in parts])
                t = t.set_column(t.schema.get_field_index(name), name, col)
        else:
            new = lay.new_head_pose(t.column("index").to_numpy(), k)
            if op == "replace":
                t = t.set_column(t.schema.get_field_index(key), key, fsl(new))
            else:
                t = t.append_column(key, fsl(new))
            per_ep = [(int(eps[a]), vec_stats(new[a:b])) for a, b in bounds]
        write_parquet(
            fs, p, (t.slice(a, b - a) for a, b in bounds), info["compression"]
        )
        return size_in, fs.get_file_info(p).size, per_ep

    res = pmap(one, range(info["n_files"]), workers)
    rec = {
        "objects_written": len(res),
        "bytes_read": sum(r[0] for r in res),
        "bytes_written": sum(r[1] for r in res),
    }
    if key is None:
        return rec
    stats, lens, file_no = read_meta(fs, base)
    for _, _, per_ep in res:
        for ep, st in per_ep:
            stats[ep][key] = st
    rec["bytes_written"] += write_meta(fs, base, stats, lens, file_no)
    rec["objects_written"] += 2
    if op == "append":
        info["features"][key] = {"dtype": "float32", "shape": [7]}
        write_json(fs, f"{base}/meta/info.json", info)
        rec["objects_written"] += 1
    return rec


def verify_lerobot(root: str, lay: Layout, op: str, k: int) -> None:
    fs, base = arrow_fs(root)
    info = read_json(fs, f"{base}/meta/info.json")
    for f in {0, info["n_files"] - 1}:
        t = pq.read_table(f"{base}/{data_path(f)}", filesystem=fs)
        if op == "annotations":
            eps = t.column("episode_index").to_numpy()
            for ep in np.unique(eps):
                first = int(np.flatnonzero(eps == ep)[0])
                rows = t.column("language_persistent")[first].as_py()
                assert [r["content"] for r in rows] == [
                    s[0] for s in lay.spans(int(ep), k)
                ]
            continue
        key = "head_pose" if op == "replace" else feat_name(k)
        np.testing.assert_array_equal(
            vec(t.column(key).combine_chunks(), 7),
            lay.new_head_pose(t.column("index").to_numpy(), k),
        )
    if op != "annotations":
        stats, _, _ = read_meta(fs, base)
        want = vec_stats(lay.new_head_pose(lay.g(0), k))
        for s in STATS:
            np.testing.assert_allclose(stats[0][key][s], want[s])


# --------------------------------------------------------------------------- lance
def steps_uri(root: str) -> str:
    return f"{root.rstrip('/')}/steps.lance"


def tasks_uri(root: str) -> str:
    return f"{root.rstrip('/')}/tasks.lance"


def rec_id(ep: int) -> str:
    return f"rec-{ep:07d}"


def steps_batch(cols: dict, recording_id: str) -> pa.RecordBatch:
    """One HUMAN_STEPS batch from flattened columns (VECTORS/SCALARS/BOOLS keys plus g)."""
    from egomimic.rldb.lance.schema import HUMAN_STEPS

    def hand(s):
        arrs = [fsl(cols[f"{s}.{f}"]) for f in HANDS]
        arrs.append(pa.array(cols[f"{s}.confidence"]))
        return pa.StructArray.from_arrays(arrs, fields=list(HUMAN_STEPS.field(s).type))

    n = len(cols["g"])
    return pa.RecordBatch.from_arrays(
        [
            pa.array([recording_id] * n),
            pa.array(cols["g"] * DT_NS),
            fsl(cols["head_pose"]),
            hand("left"),
            hand("right"),
            pa.array(cols["left_valid"]),
            pa.array(cols["right_valid"]),
            pa.array(cols["head_pose_valid"]),
            fsl(cols["eye_gaze"]),
        ],
        schema=HUMAN_STEPS,
    )


def lance_annotations(lay: Layout, eps: np.ndarray, k: int) -> pa.Array:
    from egomimic.rldb.lance.schema import ANNOTATION

    t0 = lay.starts[eps] * DT_NS
    return pa.array(
        [
            [
                {
                    "t_start_ns": int(t + s * DT_NS),
                    "t_end_ns": int(t + e * DT_NS),
                    "text": txt,
                    "annotator": "bench",
                }
                for txt, s, e in lay.spans(int(ep), k)
            ]
            for ep, t in zip(eps, t0)
        ],
        pa.list_(ANNOTATION),
    )


def make_lance(root: str, lay: Layout, rows_per_frag: int) -> None:
    import lance

    from egomimic.rldb.lance.schema import HUMAN_STEPS, TASKS

    opts = lance_opts(root)
    batches = (steps_batch(lay.episode(ep), rec_id(ep)) for ep in range(lay.n_eps))
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(HUMAN_STEPS, batches),
        steps_uri(root),
        schema=HUMAN_STEPS,
        max_rows_per_file=rows_per_frag,
        storage_options=opts,
    )
    eps = np.arange(lay.n_eps)
    t0 = lay.starts * DT_NS
    tasks = pa.table(
        {
            "task_instance_id": [f"task-{ep:07d}" for ep in eps],
            "recording_id": [rec_id(ep) for ep in eps],
            "t_start_ns": t0,
            "t_end_ns": t0 + (lay.lens - 1) * DT_NS,
            "task_name": ["fold_clothes"] * lay.n_eps,
            "task_description": ["folding clothes"] * lay.n_eps,
            "objects": pa.array([["shirt"]] * lay.n_eps, pa.list_(pa.string())),
            "annotations": lance_annotations(lay, eps, 0),
        },
        schema=TASKS,
    )
    lance.write_dataset(tasks, tasks_uri(root), storage_options=opts)


def lance_add(uri: str, opts, name: str, read: str, fn, workers: int):
    """New column `name` = fn(batch of `read`) via one data file per fragment and one Merge."""
    import lance

    ds = lance.dataset(uri, storage_options=opts)

    def udf(batch: pa.RecordBatch) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays([fn(batch.column(read))], names=[name])

    res = pmap(
        lambda frag: frag.merge_columns(udf, columns=[read]),
        ds.get_fragments(),
        workers,
    )
    ds = lance.LanceDataset.commit(
        uri,
        lance.LanceOperation.Merge([f for f, _ in res], res[0][1]),
        read_version=ds.version,
        storage_options=opts,
    )
    return ds, len(res)


def lance_replace(uri: str, opts, col: str, read: str, fn, workers: int) -> int:
    ds, n = lance_add(uri, opts, f"{col}_new", read, fn, workers)
    ds.drop_columns([col])
    ds.alter_columns({"path": f"{col}_new", "name": col})
    return n


def update_lance(root: str, lay: Layout, op: str, k: int, workers: int) -> dict:
    opts = lance_opts(root)
    if op == "annotations":

        def anns(rids):
            eps = np.array([int(r[4:]) for r in rids.to_pylist()])
            return lance_annotations(lay, eps, k)

        n = lance_replace(
            tasks_uri(root), opts, "annotations", "recording_id", anns, workers
        )
        return {"objects_written": n}

    def hp(ts):
        return fsl(lay.new_head_pose(ts.to_numpy() // DT_NS, k))

    if op == "replace":
        n = lance_replace(
            steps_uri(root), opts, "head_pose", "timestamp_ns", hp, workers
        )
    else:
        _, n = lance_add(
            steps_uri(root), opts, feat_name(k), "timestamp_ns", hp, workers
        )
    return {"objects_written": n}


def verify_lance(root: str, lay: Layout, op: str, k: int) -> None:
    import lance

    from egomimic.rldb.lance.schema import HUMAN_STEPS, TASKS

    opts = lance_opts(root)
    if op == "annotations":
        ds = lance.dataset(tasks_uri(root), storage_options=opts)
        assert {f.name: f.type for f in ds.schema} == {f.name: f.type for f in TASKS}
        eps = sorted({0, lay.n_eps // 2, lay.n_eps - 1})
        t = ds.take(eps, columns=["recording_id", "annotations"])
        got_eps = np.array([int(r[4:]) for r in t.column("recording_id").to_pylist()])
        assert (
            t.column("annotations").to_pylist()
            == lance_annotations(lay, got_eps, k).to_pylist()
        )
        return
    ds = lance.dataset(steps_uri(root), storage_options=opts)
    schema = {f.name: f.type for f in ds.schema}
    assert all(schema[f.name] == f.type for f in HUMAN_STEPS)
    key = "head_pose" if op == "replace" else feat_name(k)
    n = ds.count_rows()
    t = ds.take(sorted({0, n // 2, n - 1}), columns=["timestamp_ns", key])
    np.testing.assert_array_equal(
        vec(t.column(key).combine_chunks(), 7),
        lay.new_head_pose(t.column("timestamp_ns").to_numpy() // DT_NS, k),
    )


# --------------------------------------------------------------------------- cli
def fmt_root(root: str, fmt: str) -> str:
    return f"{root.rstrip('/')}/{fmt}"


def make(fmt: str, root: str, lay: Layout, workers: int, rows_per_frag: int) -> None:
    if fmt == "zarr":
        make_zarr(root, lay, workers)
    elif fmt == "lance":
        make_lance(root, lay, rows_per_frag)
    else:
        make_lerobot(root, lay, workers, COMPRESSION[fmt])


def update(fmt: str, root: str, lay: Layout, op: str, k: int, workers: int) -> dict:
    fn = {"zarr": update_zarr, "lance": update_lance}.get(fmt, update_lerobot)
    return fn(root, lay, op, k, workers)


def verify(fmt: str, root: str, lay: Layout, op: str, k: int) -> None:
    fn = {"zarr": verify_zarr, "lance": verify_lance}.get(fmt, verify_lerobot)
    fn(root, lay, op, k)


def cmd_make(a) -> None:
    lay = pilot_layout(a.pilot, a.rows)
    print(f"{lay.n_eps} episodes, median {int(np.median(lay.lens))} frames", flush=True)
    for fmt in a.formats:
        t = time.time()
        make(fmt, fmt_root(a.root, fmt), lay, a.workers, a.rows_per_frag)
        print(f"make {fmt}: {time.time() - t:.0f}s", flush=True)


def cmd_update(a) -> None:
    lay = pilot_layout(a.pilot, a.rows)
    r = fmt_root(a.root, a.fmt)
    t = time.perf_counter()
    rec = update(a.fmt, r, lay, a.op, a.k, a.workers)
    sec = time.perf_counter() - t
    verify(a.fmt, r, lay, a.op, a.k)
    emit(
        {
            "bench": "update",
            "op": a.op,
            "format": a.fmt,
            "rows": int(lay.lens.sum()),
            "episodes": lay.n_eps,
            "workers": a.workers,
            "latency_ms": a.latency_ms,
            "seconds": sec,
            "rows_per_s": lay.lens.sum() / sec,
            **rec,
        },
        a.out,
    )


def cmd_fit(a) -> None:
    """Rate-based projection from the largest run per (op, format, latency, workers)."""
    runs = {}
    for line in Path(a.results).read_text().splitlines():
        r = json.loads(line)
        if r.get("bench") == "update":
            key = (r.get("op", "replace"), r["format"], r["latency_ms"], r["workers"])
            runs.setdefault(key, []).append(r)
    print(
        f"| op | format | latency ms | workers | rows/s by size | hours at {a.target:.0e} rows |"
    )
    print("|---|---|---|---|---|---|")
    for key in sorted(runs):
        rs = sorted(runs[key], key=lambda r: r["rows"])
        rates = ", ".join(f"{r['rows'] / 1e6:g}M: {r['rows_per_s']:,.0f}" for r in rs)
        hours = a.target / rs[-1]["rows_per_s"] / 3600
        print(f"| {' | '.join(map(str, key))} | {rates} | {hours:,.2f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("make", "update"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        p.add_argument("--rows", type=int, required=True)
        p.add_argument(
            "--pilot", type=Path, default=Path("/coc/flash7/scratch/lance_pilot")
        )
        p.add_argument("--workers", type=int, default=16)
    m = sub.choices["make"]
    m.add_argument("--rows-per-frag", type=int, default=LANCE_ROWS_PER_FILE)
    m.add_argument("--formats", nargs="+", default=list(FORMATS))
    u = sub.choices["update"]
    u.add_argument("--fmt", required=True, choices=FORMATS)
    u.add_argument("--op", required=True, choices=OPS)
    u.add_argument(
        "--k", type=int, required=True, help="edit generation; values encode it"
    )
    u.add_argument("--latency-ms", type=float, default=0.0, help="label only")
    u.add_argument("--out")
    f = sub.add_parser("fit")
    f.add_argument("results")
    f.add_argument("--target", type=float, default=2e9)
    a = ap.parse_args()
    {"make": cmd_make, "update": cmd_update, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
