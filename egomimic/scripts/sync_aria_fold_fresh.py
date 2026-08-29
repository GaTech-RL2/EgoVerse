"""Fresh, verified copy of the aria fold_clothes episodes into a DEDICATED
folder (not the shared mirror — several shared-mirror copies are stale/
corrupt from old partial syncs). Downloads all 709 episodes from S3, then
decode-verifies 4 frames per episode; exits nonzero if anything fails so a
dependent training job won't start on bad data."""

import sys
from pathlib import Path

import numpy as np
import simplejpeg
import zarr

from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
from egomimic.utils.aws.aws_data_utils import load_env

DEST = Path("/storage/project/r-dxu345-0/agao81/ariaFoldZarrDatasets")

FILTERS = [
    "lambda row: '/aria/' in str(row.get('zarr_processed_path', ''))",
    "lambda row: str(row.get('task', '')) == 'fold_clothes'",
]


def frame_ok(arr, i):
    el = arr[i]
    while isinstance(el, np.ndarray) and el.ndim == 0:
        el = el.item()
    b = el if isinstance(el, bytes) else bytes(el)
    if len(b) < 500 or b[:3] != b"\xff\xd8\xff":
        return False
    try:
        simplejpeg.decode_jpeg(b, colorspace="RGB")
        return True
    except Exception:
        return False


def main():
    load_env()
    DEST.mkdir(parents=True, exist_ok=True)
    paths = S3EpisodeResolver.sync_from_filters(
        bucket_name="rldb",
        filters=DatasetFilter(filter_lambdas=FILTERS),
        local_dir=DEST,
        numworkers=20,
    )
    print(f"[sync] synced {len(paths)} episodes -> {DEST}", flush=True)

    bad = []
    for _, h in paths:
        try:
            g = zarr.open(str(DEST / h), mode="r")
            arr = g["images.front_1"]
            idx = np.linspace(0, arr.shape[0] - 1, 4).astype(int)
            if not all(frame_ok(arr, i) for i in idx):
                bad.append(h)
        except Exception:
            bad.append(h)
    print(f"[verify] corrupt after fresh sync: {len(bad)} {bad[:5]}", flush=True)
    if bad:
        sys.exit(1)
    print("[verify] all episodes decode OK", flush=True)


if __name__ == "__main__":
    main()
