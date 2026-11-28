"""Stage 1 (env aria2): extract Aria egocentric RGB, time-align to the 60Hz robot grid.

For each nav recording: read the VRS camera-rgb stream, map each 60Hz grid frame
(nav*_aria60.npz['ts'], absolute device us) to the nearest RGB frame by timestamp,
center-crop + resize, and save a COMPACT npz: the unique decoded frames + a per-grid
index. Stage 2 (env rby1) expands frames[sel] and injects into the HDF5 as obs/aria_image.

Aria Gen2 RGB is already upright (no rotation). Clock == MPS clock (capture_ns//1000 = us).
"""
import argparse
import glob
import os
import re

import numpy as np
from PIL import Image
from projectaria_tools.core import data_provider
from projectaria_tools.core.stream_id import StreamId

DATA = "/media/czhang883/PORTABLE_SSD/mobile_generalist/data/nav_pouring_test"
A = os.environ.get("ARIA_ARTIFACTS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "artifacts", "aria_egoposer")))


def mps_to_tag(mps_dir):
    base = os.path.basename(mps_dir)
    mid = base[len("mps_Nav_pouring_test_"):-len("_vrs")]
    num = mid.split("_")[0]
    return f"nav{num}" if re.fullmatch(r"\d{1,2}", num) else "nav1"


def vrs_for(mps_dir):
    """Sibling VRS file for an mps_*_vrs dir."""
    base = os.path.basename(mps_dir)[len("mps_"):-len("_vrs")]   # Nav_pouring_test_9_...
    p = os.path.join(DATA, base + ".vrs")
    return p if os.path.exists(p) else None


def process(frame, res):
    H, W = frame.shape[:2]
    s = min(H, W)
    y0, x0 = (H - s) // 2, (W - s) // 2
    sq = frame[y0:y0 + s, x0:x0 + s]
    return np.asarray(Image.fromarray(sq).resize((res, res)), dtype=np.uint8)


def extract_one(mps_dir, res=224):
    extract_core(vrs_for(mps_dir), mps_to_tag(mps_dir), res)


def extract_core(vrs, tag, res=224):
    """RGB extract + align for one recording, identified by its VRS + tag (works for both
    MPS recordings and the new on-device VRS -- only needs the VRS + nav{tag}_aria60.npz)."""
    aria60 = os.path.join(A, f"{tag}_aria60.npz")
    if vrs is None or not os.path.exists(aria60):
        print(f"  {tag}: SKIP (vrs={vrs is not None}, aria60={os.path.exists(aria60)})")
        return
    grid_us = np.load(aria60)["ts"].astype(np.int64)               # (T,)
    dp = data_provider.create_vrs_data_provider(vrs)
    sid = dp.get_stream_id_from_label("camera-rgb")
    n = dp.get_num_data(sid)
    rgb_us = np.array([dp.get_image_data_by_index(sid, i)[1].capture_timestamp_ns
                       for i in range(n)], dtype=np.int64) // 1000  # (n,)
    j = np.clip(np.searchsorted(rgb_us, grid_us), 1, n - 1)
    left = j - 1
    sel = np.where(np.abs(rgb_us[j] - grid_us) < np.abs(rgb_us[left] - grid_us), j, left)
    sel = np.clip(sel, 0, n - 1).astype(np.int64)                  # grid frame -> rgb idx

    uniq = np.unique(sel)
    remap = {int(u): k for k, u in enumerate(uniq)}
    frames = np.zeros((len(uniq), res, res, 3), np.uint8)
    for k, u in enumerate(uniq):
        frames[k] = process(dp.get_image_data_by_index(sid, int(u))[0].to_numpy_array(), res)
    sel_compact = np.array([remap[int(u)] for u in sel], np.int32)  # grid frame -> frames idx
    err_ms = np.abs(rgb_us[sel] - grid_us).mean() / 1000.0
    out = os.path.join(A, f"{tag}_rgb{res}.npz")
    np.savez_compressed(out, frames=frames, sel=sel_compact, grid_us=grid_us, res=res)
    print(f"  {tag}: T={len(grid_us)} grid, {len(uniq)} unique RGB (of {n}), "
          f"align err {err_ms:.0f}ms, saved {out} ({frames.nbytes/1e6:.0f}MB raw)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--only", default=None, help="process only this tag (e.g. nav9)")
    ap.add_argument("--vrs", default=None, help="direct: extract from this VRS (with --tag)")
    ap.add_argument("--tag", default=None, help="tag for --vrs direct mode (e.g. nav11)")
    args = ap.parse_args()
    if args.vrs:
        extract_core(args.vrs, args.tag, res=args.res)
        return
    mps_dirs = sorted(glob.glob(os.path.join(DATA, "mps_*_vrs")))
    for md in mps_dirs:
        if args.only and mps_to_tag(md) != args.only:
            continue
        extract_one(md, res=args.res)


if __name__ == "__main__":
    main()
