"""Split each recording's MP4 into K-frame self-contained MP4s, one blob row each.

The pilot encodes use GOP 2 (`aria_to_lance.py`), so every even frame is a keyframe
and an even K needs no re-encode: packets are stream-copied. Writes
`video_chunks/<camera>_k<K>.lance` under the Lance root with rows ordered by
recording, then chunk, so chunk row = first row of the recording + frame // K.
"""

import argparse
import io
import json
import random
from pathlib import Path

import av
import lance
import numpy as np
import pyarrow as pa
from lance.blob import blob_array, blob_field

from egomimic.scripts.lance_pilot import PILOT

SCHEMA = pa.schema(
    [
        ("recording_id", pa.string()),
        ("chunk_index", pa.int32()),
        ("frame_start", pa.int32()),
        ("num_frames", pa.int32()),
        blob_field("data", nullable=False),
    ]
)

CHUNK_ROWS_PER_FILE = 1 << 20


def chunk_table(
    recording_id: str, chunks: list[bytes], num_frames: int, k: int
) -> pa.Table:
    """One recording's chunk rows; raises unless the chunks cover exactly `num_frames`."""
    starts = list(range(0, len(chunks) * k, k))
    sizes = [min(k, num_frames - s) for s in starts]
    if sum(sizes) != num_frames:
        raise ValueError(
            f"{recording_id}: chunks cover {len(chunks)} x {k}, num_frames {num_frames}"
        )
    return pa.table(
        {
            "recording_id": [recording_id] * len(chunks),
            "chunk_index": list(range(len(chunks))),
            "frame_start": starts,
            "num_frames": sizes,
            "data": blob_array(chunks),
        },
        schema=SCHEMA,
    )


def chunk_mp4(path: Path, k: int) -> list[bytes]:
    """Stream-copy `path` into consecutive k-frame MP4s."""
    src = av.open(str(path))
    vin = src.streams.video[0]
    packets = [p for p in src.demux(vin) if p.size]
    out = []
    for start in range(0, len(packets), k):
        group = packets[start : start + k]
        if not group[0].is_keyframe:
            raise ValueError(
                f"{path}: frame {start} is not a keyframe; K must be a multiple of the GOP"
            )
        buf = io.BytesIO()
        dst = av.open(buf, "w", format="mp4")
        vout = dst.add_stream(template=vin)
        t0 = group[0].dts
        for p in group:
            p.stream = vout
            p.dts -= t0
            p.pts -= t0
            dst.mux(p)
        dst.close()
        out.append(buf.getvalue())
    src.close()
    return out


def frames(source, indices):
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(source, seek_mode="exact", dimension_order="NHWC")
    return dec.get_frames_at(
        indices=list(indices)
    ).data.numpy(), dec.metadata.num_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lance-root", type=Path, default=PILOT / "lance")
    ap.add_argument(
        "--out-root",
        type=Path,
        help="defaults to --lance-root; the resolver looks for video_chunks/ there",
    )
    ap.add_argument("--hashes", type=Path, required=True)
    ap.add_argument("--camera", default="front_1")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 8, 32])
    ap.add_argument("--check-frames", type=int, default=20)
    args = ap.parse_args()
    args.out_root = args.out_root or args.lance_root

    hashes = set(args.hashes.read_text().split())
    ali = (
        lance.dataset(str(args.lance_root / "aliases.lance"))
        .to_table(columns=["recording_id", "episode_hash"])
        .to_pylist()
    )
    rids = sorted(a["recording_id"] for a in ali if a["episode_hash"] in hashes)
    recs = {
        r["recording_id"]: r
        for r in lance.dataset(str(args.lance_root / "recordings.lance"))
        .to_table()
        .to_pylist()
    }
    rng = random.Random(0)
    for k in args.k:
        uri = str(args.out_root / "video_chunks" / f"{args.camera}_k{k}.lance")
        mp4_bytes = chunk_bytes = n_chunks = 0
        for i, rid in enumerate(rids):
            path = args.lance_root / dict(recs[rid]["video_path"])[args.camera]
            chunks = chunk_mp4(path, k)
            nf = recs[rid]["num_frames"]
            tbl = chunk_table(rid, chunks, nf, k)
            want = sorted(rng.sample(range(nf), min(args.check_frames, nf)))
            full, _ = frames(str(path), want)
            for j, f in enumerate(want):
                got, _ = frames(chunks[f // k], [f % k])
                if not np.array_equal(got[0], full[j]):
                    raise ValueError(
                        f"{rid}: frame {f} differs between chunk and full MP4"
                    )
            lance.write_dataset(
                tbl,
                uri,
                mode="append" if i else "overwrite",
                max_rows_per_file=CHUNK_ROWS_PER_FILE,
            )
            mp4_bytes += path.stat().st_size
            chunk_bytes += sum(map(len, chunks))
            n_chunks += len(chunks)
        print(
            json.dumps(
                dict(
                    k=k,
                    recordings=len(rids),
                    chunks=n_chunks,
                    mp4_gb=round(mp4_bytes / 1e9, 3),
                    chunk_gb=round(chunk_bytes / 1e9, 3),
                    overhead=round(chunk_bytes / mp4_bytes - 1, 4),
                    kb_per_chunk=round(chunk_bytes / n_chunks / 1e3, 1),
                    frames_checked=len(rids) * args.check_frames,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
