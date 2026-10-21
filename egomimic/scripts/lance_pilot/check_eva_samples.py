"""Check that the Lance reader serves the same EVA training samples as zarr.

eva_to_zarr keeps every frame, so zarr frame i is Lance frame i. Numeric outputs after the
full transform list must match to float32 precision at anchors both sample; images are
compared against the zarr JPEGs downscaled to the 360p stream.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from egomimic.rldb.embodiment.eva import Eva
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.lance.dataset import LanceEpisodeResolver
from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver

CAMERAS = {
    "images.front_1": "front_1",
    "images.left_wrist": "left_wrist",
    "images.right_wrist": "right_wrist",
}


def _hwc(img: torch.Tensor) -> np.ndarray:
    a = img.float().numpy()
    return a if a.shape[-1] == 3 else a.transpose(1, 2, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--zarr-dir",
        type=Path,
        default=Path("/coc/flash7/scratch/lance_pilot/eva_zarr"),
    )
    ap.add_argument(
        "--lance-root",
        type=Path,
        default=Path("/coc/flash7/scratch/lance_pilot/lance_eva"),
    )
    ap.add_argument("--hashes", type=Path, required=True)
    ap.add_argument("--embodiment", default="eva_bimanual")
    ap.add_argument("--per-recording", type=int, default=20)
    args = ap.parse_args()

    key_map = Eva.get_keymap(keymap_mode="cartesian")
    tl = Eva.get_transform_list(mode="cartesian_wristframe_6d")
    if args.embodiment != "eva_bimanual":
        # The Eva transforms are bimanual only: compare a one-arm robot's raw arrays.
        absent = "left." if args.embodiment == "eva_right_arm" else "right."
        key_map = {
            k: v
            for k, v in key_map.items()
            if not v["zarr_key"].startswith(absent) and v["key_type"] != "camera_keys"
        }
        tl = None
    f = DatasetFilter(episode_hashes=args.hashes.read_text().split())
    zl = S3EpisodeResolver(args.zarr_dir, key_map=key_map, transform_list=tl).resolve(
        filters=f, expected_embodiment=args.embodiment
    )
    lance_res = LanceEpisodeResolver(
        args.lance_root,
        lab="eva",
        key_map=key_map,
        transform_list=tl,
        cameras=CAMERAS,
        require_hands_in_frame="none",
    )
    keys = lance_res.dataset_keys(zl)
    ll = lance_res.resolve(filters=f, expected_embodiment=args.embodiment)

    rng = np.random.default_rng(0)
    worst: dict[str, float] = {}
    img_err: dict[str, list] = {}
    n_checked = 0
    for h, Z in zl.items():
        L = ll[keys[h]]
        if L.total_frames != Z.total_frames:
            print(f"{h}: zarr {Z.total_frames} frames, Lance {L.total_frames}")
            continue
        for i in rng.choice(
            len(Z), size=min(args.per_recording, len(Z)), replace=False
        ):
            a, b = Z[int(i)], L[int(i)]
            for k, v in a.items():
                if not isinstance(v, torch.Tensor) or k not in b:
                    continue
                if k in key_map and key_map[k]["key_type"] == "camera_keys":
                    za, lb = _hwc(v), _hwc(b[k])
                    small = cv2.resize(
                        za, lb.shape[1::-1], interpolation=cv2.INTER_AREA
                    )
                    img_err.setdefault(k, []).append(float(np.abs(small - lb).mean()))
                    continue
                if not v.is_floating_point():
                    continue
                d = float((v - b[k]).abs().nan_to_num(nan=float("inf")).max())
                worst[k] = max(worst.get(k, 0.0), d)
            n_checked += 1

    print(f"recordings {len(zl)}; samples checked {n_checked}")
    for k, d in sorted(worst.items(), key=lambda kv: -kv[1]):
        print(f"  {k:40s} max |diff| {d:.2e}")
    for k, e in img_err.items():
        print(f"  {k:40s} image mean |diff| {np.mean(e):.4f} max {np.max(e):.4f}")


if __name__ == "__main__":
    main()
