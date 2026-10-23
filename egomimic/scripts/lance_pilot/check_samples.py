"""Check that the Lance reader serves the same training samples as zarr where both can.

Comparable anchors are zarr frames whose whole action chunk is contiguous in time (the
zarr converter deleted untracked frames, so elsewhere its "next 30 frames" skip ahead).
Numeric outputs after the full transform list must match to float32 precision; images
are compared on the 480p stream against the JPEGs.
"""

import argparse
from pathlib import Path

import lance
import numpy as np
import torch
import zarr

from egomimic.rldb.embodiment.human import Human
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.lance.dataset import LanceEpisodeResolver
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver
from egomimic.scripts.lance_pilot import PILOT

HORIZON = 30
PERIOD_NS = 1e9 / 30


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zarr-dir", type=Path, default=PILOT / "zarr")
    ap.add_argument("--lance-root", type=Path, default=PILOT / "lance")
    ap.add_argument("--hashes", type=Path, required=True)
    ap.add_argument("--per-recording", type=int, default=20)
    args = ap.parse_args()

    key_map = Human.get_keymap(keymap_mode="keypoints", include_ee_pose=True)
    tl = Human.get_transform_list(
        mode="keypoints_wristframe_6d",
        include_ee_pose=True,
        pad_proprio_gripper=True,
        stride=3,
    )
    hashes = args.hashes.read_text().split()
    f = DatasetFilter(episode_hashes=hashes)
    zl = LocalEpisodeResolver(
        args.zarr_dir, key_map=key_map, transform_list=tl
    ).resolve(filters=f)
    ll = LanceEpisodeResolver(
        args.lance_root,
        key_map=key_map,
        transform_list=tl,
        cameras={"images.front_1": "front_1_480"},
    ).resolve(filters=f)
    rng = np.random.default_rng(0)
    worst: dict[str, float] = {}
    img_err, n_checked, n_comparable, n_zarr = [], 0, 0, 0
    for h in hashes:
        Z, L = zl[h], ll[h]
        zts = np.asarray(
            zarr.open(str(args.zarr_dir / h), mode="r")["obs_rgb_timestamps_ns"][
                : Z.total_frames
            ]
        )
        steps = lance.dataset(L.steps_uri, version=L.steps_version)
        lts = (
            steps.take(
                list(range(L.row_start, L.row_start + L.total_frames)),
                columns=["timestamp_ns"],
            )
            .column(0)
            .to_numpy()
        )
        n = len(zts)
        n_zarr += n
        contiguous = np.zeros(n, bool)
        gap_ok = np.abs(np.diff(zts) - PERIOD_NS) < 0.5 * PERIOD_NS
        c = np.r_[0, np.cumsum(gap_ok)]
        for i in range(n):
            end = min(i + HORIZON, n) - 1
            contiguous[i] = c[end] - c[i] == end - i
        lance_frame = np.searchsorted(lts, zts)
        local = {L.anchor(j): j for j in range(len(L))}
        cands = [i for i in np.flatnonzero(contiguous) if int(lance_frame[i]) in local]
        n_comparable += len(cands)
        for i in (
            rng.choice(cands, size=min(args.per_recording, len(cands)), replace=False)
            if cands
            else []
        ):
            a, b = Z[int(i)], L[local[int(lance_frame[i])]]
            for k, v in a.items():
                if (
                    not isinstance(v, torch.Tensor)
                    or k not in b
                    or v.dtype != torch.float32
                ):
                    continue
                if k.startswith("observations.images"):
                    img_err.append(float((v - b[k]).abs().mean()))
                    continue
                d = float((v - b[k]).abs().max())
                worst[k] = max(worst.get(k, 0.0), d)
            n_checked += 1

    print(
        f"zarr anchors {n_zarr:,}; comparable (contiguous chunk, eligible in Lance) {n_comparable:,}; checked {n_checked}"
    )
    for k, d in sorted(worst.items(), key=lambda kv: -kv[1]):
        print(f"  {k:40s} max |diff| {d:.2e}")
    if img_err:
        print(
            f"  image mean |diff| (AV1 480p vs JPEG): mean {np.mean(img_err):.4f} max {np.max(img_err):.4f}"
        )


if __name__ == "__main__":
    main()
