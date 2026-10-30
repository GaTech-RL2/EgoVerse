"""Startup-scale datasets: the 48 real pilot episodes plus N fake bimanual episodes, per format.

Each fake is `--frames` real rows sliced from a bimanual pilot episode and points at that
episode's real video, so every loader runs unmodified:
  zarr     a symlink to the source store (startup reads only attrs, so length does not matter)
  lance    recordings/aliases/tasks rows + contiguous steps rows; the 480p k8 chunk table gets
           one row per 8 frames with an empty blob (the resolver only scans recording_id and
           num_frames; first batches sample real episodes only); videos/ is a symlink
  lerobot  data parquet rows + episodes rows whose video from/to timestamps fall inside the
           source episode's span of a real video file, rolled at LeRobot's file-size limits;
           videos/ is a symlink
Also writes `real_<fmt>.txt` (what the first batch may sample) and `lerobot_episodes.txt`.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lance.blob import blob_array

from egomimic.rldb.lance.schema import only_steps_path
from egomimic.scripts.lance_pilot import PILOT

BIMANUAL = "human_bimanual"


def fake_name(i: int) -> str:
    return f"fake-{i:07d}"


def sources(
    n_fakes: int, src_lengths: list[int], frames: int, seed: int = 0
) -> list[tuple[int, int]]:
    """(source index, start frame) per fake, deterministic."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_fakes):
        s = i % len(src_lengths)
        out.append((s, int(rng.integers(0, src_lengths[s] - frames))))
    return out


# --------------------------------------------------------------------------- zarr
def make_zarr(src: Path, out: Path, n: int, manifest: list[dict]) -> None:
    out.mkdir(parents=True)
    bi = [m["episode_hash"] for m in manifest if m["embodiment"] == BIMANUAL]
    for m in manifest:
        os.symlink(src / m["episode_hash"], out / m["episode_hash"])
    for i in range(n):
        os.symlink(src / bi[i % len(bi)], out / fake_name(i))
    (out.parent / "real_zarr.txt").write_text(" ".join(bi) + "\n")


# --------------------------------------------------------------------------- lance
def make_lance(src: Path, chunks_src: Path, out: Path, n: int, frames: int) -> None:
    out.mkdir(parents=True)
    os.symlink(src / "videos", out / "videos")
    rec = lance.dataset(str(src / "recordings.lance")).to_table()
    ali = lance.dataset(str(src / "aliases.lance")).to_table()
    tasks = lance.dataset(str(src / "tasks.lance")).to_table()
    steps_rel = only_steps_path(rec)
    steps = lance.dataset(str(src / steps_rel)).to_table()
    chunks_ds = lance.dataset(str(chunks_src))
    chunks = chunks_ds.to_table(columns=["num_frames"])

    ids = steps.column("recording_id").to_numpy(zero_copy_only=False)
    first = {}
    for i, r in enumerate(ids):
        first.setdefault(r, i)
    recs = rec.to_pylist()
    bi = sorted(
        (r for r in recs if r["embodiment"] == BIMANUAL),
        key=lambda r: first[r["recording_id"]],
    )
    plan = sources(n, [r["num_frames"] for r in bi], frames)
    fake_ids = [fake_name(i) for i in range(n)]

    def rows_like(tbl, key, src_ids, new_ids, overrides=None):
        by_id = {r[key]: r for r in tbl.to_pylist()}
        rows = []
        for s, f in zip(src_ids, new_ids):
            r = dict(by_id[s])
            r[key] = f
            r.update((overrides or {}).get(f, {}))
            rows.append(r)
        return pa.concat_tables([tbl, pa.Table.from_pylist(rows, schema=tbl.schema)])

    src_of = [bi[s]["recording_id"] for s, _ in plan]
    lance.write_dataset(
        rows_like(
            rec,
            "recording_id",
            src_of,
            fake_ids,
            {f: {"num_frames": frames} for f in fake_ids},
        ),
        str(out / "recordings.lance"),
    )
    a = rows_like(ali, "recording_id", src_of, fake_ids)
    eh = a.column("episode_hash").to_pylist()
    eh[len(ali) :] = fake_ids
    a = a.set_column(
        a.schema.get_field_index("episode_hash"), "episode_hash", pa.array(eh)
    )
    lance.write_dataset(a, str(out / "aliases.lance"))
    lance.write_dataset(
        rows_like(tasks, "recording_id", src_of, fake_ids), str(out / "tasks.lance")
    )

    rid_col = steps.schema.get_field_index("recording_id")

    def step_batches():
        yield from steps.to_batches(max_chunksize=1 << 20)
        per = max(1, (1 << 20) // frames)
        for b0 in range(0, n, per):
            idx = np.concatenate(
                [
                    np.arange(
                        first[bi[s]["recording_id"]] + f0,
                        first[bi[s]["recording_id"]] + f0 + frames,
                    )
                    for s, f0 in plan[b0 : b0 + per]
                ]
            )
            t = steps.take(pa.array(idx))
            new = pa.array(
                np.repeat(np.asarray(fake_ids[b0 : b0 + per], dtype=object), frames),
                pa.string(),
            )
            yield from t.set_column(rid_col, "recording_id", new).to_batches()

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(steps.schema, step_batches()),
        str(out / steps_rel),
        max_rows_per_file=1 << 23,
    )

    k = int(pc.max(chunks.column("num_frames")).as_py())
    n_chunks = -(-frames // k)
    nf = [min(k, frames - j * k) for j in range(n_chunks)]

    def chunk_batches():
        per = max(1, 65536 // n_chunks)
        for b0 in range(0, n, per):
            ids_ = fake_ids[b0 : b0 + per]
            m = len(ids_)
            yield pa.record_batch(
                [
                    pa.array(
                        np.repeat(np.asarray(ids_, dtype=object), n_chunks), pa.string()
                    ),
                    pa.array(np.tile(np.arange(n_chunks, dtype=np.int32), m)),
                    pa.array(
                        np.tile(np.arange(0, frames, k, dtype=np.int32)[:n_chunks], m)
                    ),
                    pa.array(np.tile(np.asarray(nf, np.int32), m)),
                    blob_array([b""] * (m * n_chunks)),
                ],
                schema=chunks_ds.schema,
            )

    # Blob columns do not round-trip through Arrow: copy the real table, append fake rows.
    dst = out / "video_chunks" / chunks_src.name
    shutil.copytree(chunks_src, dst)
    lance.write_dataset(
        pa.RecordBatchReader.from_batches(chunks_ds.schema, chunk_batches()),
        str(dst),
        mode="append",
        max_rows_per_file=1 << 20,
    )
    a_all = ali.to_pylist()
    real_bi = {r["recording_id"] for r in bi}
    (out.parent / "real_lance.txt").write_text(
        " ".join(r["episode_hash"] for r in a_all if r["recording_id"] in real_bi)
        + "\n"
    )


# --------------------------------------------------------------------------- lerobot
def make_lerobot(
    src: Path, out: Path, n: int, frames: int, bimanual_eps: list[int]
) -> None:
    info = json.loads((src / "meta/info.json").read_text())
    fps = info["fps"]
    (out / "meta").mkdir(parents=True)
    os.symlink(src / "videos", out / "videos")
    for f in ("tasks.parquet", "stats.json"):
        shutil.copy(src / "meta" / f, out / "meta" / f)
    data = pq.read_table(sorted((src / "data").glob("*/*.parquet")))
    eps = pq.read_table(sorted((src / "meta/episodes").glob("*/*.parquet")))
    n_real_eps, n_real_frames = eps.num_rows, data.num_rows
    lengths = eps.column("length").to_pylist()
    starts = eps.column("dataset_from_index").to_pylist()
    plan = sources(n, [lengths[e] for e in bimanual_eps], frames)
    data_mb, chunks_size = info["data_files_size_in_mb"], info["chunks_size"]

    # Data: real rows first, then fakes, rolled like DatasetWriter._save_episode_data.
    chunk, file, writer, file_start, nxt = 0, 0, None, 0, 0
    loc = []  # (chunk, file) per episode, real then fake

    def path(c, f):
        return out / info["data_path"].format(chunk_index=c, file_index=f)

    def write_episode(tbl):
        nonlocal chunk, file, writer, file_start, nxt
        if writer is not None:
            size = os.path.getsize(path(chunk, file)) / 2**20
            in_file = nxt - file_start
            if size + (size / in_file if in_file else 0) * tbl.num_rows >= data_mb:
                writer.close()
                writer = None
                file += 1
                if file >= chunks_size:
                    chunk, file = chunk + 1, 0
                file_start = nxt
        if writer is None:
            path(chunk, file).parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(
                path(chunk, file), tbl.schema, compression="snappy", use_dictionary=True
            )
        writer.write_table(tbl)
        loc.append((chunk, file))
        nxt += tbl.num_rows

    for e in range(n_real_eps):
        write_episode(data.slice(starts[e], lengths[e]))
    ei, ii, fi, ti = (
        data.schema.get_field_index(c)
        for c in ("episode_index", "index", "frame_index", "timestamp")
    )
    for i, (s, f0) in enumerate(plan):
        e = bimanual_eps[s]
        t = data.slice(starts[e] + f0, frames)
        t = t.set_column(
            ei, "episode_index", pa.array(np.full(frames, n_real_eps + i, np.int64))
        )
        t = t.set_column(
            ii, "index", pa.array(np.arange(nxt, nxt + frames, dtype=np.int64))
        )
        t = t.set_column(fi, "frame_index", pa.array(np.arange(frames, dtype=np.int64)))
        t = t.set_column(
            ti,
            "timestamp",
            pa.array(np.arange(frames, dtype=np.float32) / np.float32(fps)),
        )
        write_episode(t)
    writer.close()

    # Episodes metadata: copy each source row (stats included), rolled at the same limit.
    vk = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    src_idx = np.asarray([bimanual_eps[s] for s, _ in plan], dtype=np.int64)
    f0s = np.asarray([f0 for _, f0 in plan], dtype=np.float64)
    fake = eps.take(pa.array(src_idx))

    def put(tbl, name, values):
        i = tbl.schema.get_field_index(name)
        return tbl.set_column(
            i, tbl.schema.field(i), pa.array(values, tbl.schema.field(i).type)
        )

    fake = put(fake, "episode_index", np.arange(n_real_eps, n_real_eps + n))
    fake = put(fake, "length", np.full(n, frames))
    fake = put(fake, "dataset_from_index", n_real_frames + np.arange(n) * frames)
    fake = put(fake, "dataset_to_index", n_real_frames + (np.arange(n) + 1) * frames)
    for key in vk:
        start_ts = fake.column(f"videos/{key}/from_timestamp").to_numpy() + f0s / fps
        fake = put(fake, f"videos/{key}/from_timestamp", start_ts)
        fake = put(fake, f"videos/{key}/to_timestamp", start_ts + frames / fps)
    allt = pa.concat_tables([eps, fake])
    allt = put(allt, "data/chunk_index", [c for c, _ in loc])
    allt = put(allt, "data/file_index", [f for _, f in loc])
    # Same rule as LeRobotDatasetMetadata._save_episode_metadata: roll at data_files_size_in_mb.
    sink = pa.BufferOutputStream()
    pq.write_table(allt.slice(0, 1000), sink, compression="snappy")
    per_row_mb = sink.getvalue().size / 2**20 / min(1000, allt.num_rows)
    rows_per_file = max(1, int(data_mb / per_row_mb))
    file_no = np.arange(allt.num_rows) // rows_per_file
    allt = put(allt, "meta/episodes/chunk_index", file_no // chunks_size)
    allt = put(allt, "meta/episodes/file_index", file_no % chunks_size)
    for j in range(0, allt.num_rows, rows_per_file):
        c, f = divmod(j // rows_per_file, chunks_size)
        p = out / f"meta/episodes/chunk-{c:03d}/file-{f:03d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(allt.slice(j, rows_per_file), p, compression="snappy")

    info["total_episodes"] = allt.num_rows
    info["total_frames"] = n_real_frames + n * frames
    info["splits"] = {"train": f"0:{allt.num_rows}"}
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    eps_list = bimanual_eps + list(range(n_real_eps, n_real_eps + n))
    (out.parent / "lerobot_episodes.txt").write_text(
        " ".join(map(str, eps_list)) + "\n"
    )
    (out.parent / "real_lerobot.txt").write_text(
        " ".join(map(str, bimanual_eps)) + "\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", type=Path, default=PILOT)
    ap.add_argument(
        "--chunks", type=Path, required=True, help="480p k8 chunk table of the pilot"
    )
    ap.add_argument("--lerobot", type=Path, required=True)
    ap.add_argument("--lerobot-bimanual", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--formats", nargs="+", default=["zarr", "lance", "lerobot"])
    args = ap.parse_args()

    import csv

    manifest = list(csv.DictReader(open(args.pilot / "manifest.csv")))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        t = time.time()
        if (out / fmt).exists():
            shutil.rmtree(out / fmt)
        if fmt == "zarr":
            make_zarr(args.pilot / "zarr", out / "zarr", args.n, manifest)
        elif fmt == "lance":
            make_lance(
                args.pilot / "lance", args.chunks, out / "lance", args.n, args.frames
            )
        else:
            bi = [int(x) for x in args.lerobot_bimanual.read_text().split()]
            make_lerobot(args.lerobot, out / "lerobot", args.n, args.frames, bi)
        print(
            json.dumps(
                dict(
                    format=fmt,
                    n=args.n,
                    frames=args.frames,
                    build_s=round(time.time() - t, 1),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
