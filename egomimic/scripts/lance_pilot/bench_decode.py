"""Per-frame decode cost, one thread, files preloaded into memory: zarr JPEG vs the Lance AV1 MP4s.

jpeg            simplejpeg on random zarr frames
av1_random      get_frame_at on random indices (what the shuffle buffer does)
av1_sorted      get_frames_at on the same indices, sorted
av1_sequential  every frame of a contiguous range, in order
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from egomimic.rldb.lance.dataset import open_table


def _timed(fn, n):
    t0 = time.perf_counter()
    fn()
    return round(1000 * (time.perf_counter() - t0) / n, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hashes", nargs="+", required=True)
    ap.add_argument(
        "--zarr-root", type=Path, default=Path("/coc/flash7/scratch/lance_pilot/zarr")
    )
    ap.add_argument(
        "--lance-root", type=Path, default=Path("/coc/flash7/scratch/lance_pilot/lance")
    )
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    import simplejpeg
    import zarr
    from torchcodec.decoders import VideoDecoder

    rec = (
        open_table(str(args.lance_root / "recordings.lance"))
        .to_table(columns=["recording_id", "video_path"])
        .to_pandas()
    )
    ali = (
        open_table(str(args.lance_root / "aliases.lance"))
        .to_table(columns=["recording_id", "episode_hash"])
        .to_pandas()
    )
    videos = rec.merge(ali, on="recording_id").set_index("episode_hash").video_path

    for h in args.hashes:
        rng = random.Random(0)
        jpegs = zarr.open_group(str(args.zarr_root / h), mode="r")["images.front_1"][:]
        idx = [rng.randrange(len(jpegs)) for _ in range(args.n)]
        out = dict(
            hash=h,
            n=args.n,
            jpeg_ms=_timed(
                lambda: [
                    simplejpeg.decode_jpeg(jpegs[i], colorspace="RGB") for i in idx
                ],
                args.n,
            ),
        )
        out["jpeg_kb"] = round(np.mean([len(jpegs[i]) for i in idx]) / 1024, 1)
        for cam in ("front_1_480", "front_1"):
            mp4 = (args.lance_root / dict(videos[h])[cam]).read_bytes()
            dec = VideoDecoder(
                mp4,
                seek_mode="approximate",
                dimension_order="NHWC",
                num_ffmpeg_threads=1,
            )
            nf = dec.metadata.num_frames
            vidx = [rng.randrange(nf) for _ in range(args.n)]
            res = f"{dec.metadata.height}p"
            out[f"{res}_kb"] = round(len(mp4) / nf / 1024, 1)
            out[f"{res}_random_ms"] = _timed(
                lambda: [dec.get_frame_at(i) for i in vidx], args.n
            )
            out[f"{res}_sorted_ms"] = _timed(
                lambda: dec.get_frames_at(indices=sorted(vidx)), args.n
            )
            s = rng.randrange(nf - args.n)
            out[f"{res}_sequential_ms"] = _timed(
                lambda: dec.get_frames_in_range(start=s, stop=s + args.n), args.n
            )
            out["codec"] = dec.metadata.codec
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
