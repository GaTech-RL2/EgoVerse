"""Trim each demo to its valid ACTIVE window, removing (a) dead/frozen hand-tracking DROPOUT
tails (nav1-3) and (b) trailing idle. TAIL-ONLY trim (keep_start=0) so the base origin and the
SE(2) action deltas / obs/robot_base_pose stay consistent (no re-zeroing needed). Slices every
(N, ...) dataset under the demo (obs/*, actions/*). Cheap post-step; re-runnable.

  python trim_hdf5.py --in_hdf5 ..._clean.hdf5 --out_hdf5 ..._clean.hdf5   (in-place via temp)
"""
import argparse
import os
import sys

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from firm_grasp import frozen_mask  # noqa: E402

A = os.environ.get("ARIA_ARTIFACTS", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "artifacts", "aria_egoposer")))


def keep_end(g, tag, buffer=18):
    """Last frame to KEEP+1 = min(last active motion + buffer, first dropout, N)."""
    obs = g["obs"]["robot0_joint_pos"][:]
    bp = g["obs"]["robot_base_pose"][:] if "robot_base_pose" in g["obs"] else None
    N = len(obs)
    arm = obs[:, 10:24]
    armspd = np.concatenate([[0.0], np.abs(np.diff(arm, axis=0)).sum(1)])
    act = armspd > np.percentile(armspd, 40) * 1.5
    if bp is not None:
        basespd = np.concatenate([[0.0], np.hypot(np.diff(bp[:, 0]), np.diff(bp[:, 1]))
                                  + 0.1 * np.abs(np.diff(bp[:, 2]))])
        act = act | (basespd > 0.003)
    last = N - 1 - int(np.argmax(act[::-1]))
    drop = N
    p = os.path.join(A, f"{tag}_smpl.npz")
    if os.path.exists(p):
        fz = frozen_mask(np.load(p)["right_landmarks"])
        if fz[0] and not fz.all():   # hand not yet tracked at the start (aria_mps_reader holds its first valid frame): not a tail
            fz[:int(np.argmin(fz))] = False
        if fz.any():
            drop = int(np.argmax(fz))
    return int(min(last + buffer, drop, N)), N, (drop < N)


def _copy_trimmed(src, dst, N, end):
    for k, v in src.attrs.items():
        dst.attrs[k] = v
    for name, item in src.items():
        if isinstance(item, h5py.Group):
            _copy_trimmed(item, dst.create_group(name), N, end)
        else:
            arr = item[:]
            if getattr(arr, "shape", None) and len(arr.shape) >= 1 and arr.shape[0] == N:
                arr = arr[:end]
            comp = "gzip" if name == "aria_image" else None
            dst.create_dataset(name, data=arr, compression=comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_hdf5", default=os.path.join(A, "aria_egoposer_train_clean.hdf5"))
    ap.add_argument("--out_hdf5", default=os.path.join(A, "aria_egoposer_train_clean.hdf5"))
    ap.add_argument("--buffer", type=int, default=18, help="frames to keep after last motion")
    args = ap.parse_args()

    tmp = args.out_hdf5 + ".tmp"
    print(f"trim: {os.path.basename(args.in_hdf5)} -> {os.path.basename(args.out_hdf5)}\n")
    print(f"{'demo':9s}{'tag':7s}{'N':>6s}{'-> keep':>9s}{'new_s':>7s}{'cut_s':>7s}{'flag'}")
    print("-" * 52)
    with h5py.File(args.in_hdf5, "r") as fin, h5py.File(tmp, "w") as fout:
        din = fin["data"]
        dout = fout.create_group("data")
        for k, v in din.attrs.items():
            dout.attrs[k] = v
        for dk in sorted(din.keys(), key=lambda s: int(s.split("_")[1])):
            g = din[dk]
            tag = g.attrs.get("source_tag", dk)
            end, N, dropped = keep_end(g, tag, args.buffer)
            ddemo = dout.create_group(dk)
            _copy_trimmed(g, ddemo, N, end)
            ddemo.attrs["num_samples"] = end
            ddemo.attrs["orig_num_samples"] = N
            if "base_path_m" in g.attrs and "robot_base_pose" in ddemo["obs"]:
                bp = ddemo["obs"]["robot_base_pose"][:]
                ddemo.attrs["base_path_m"] = float(np.sum(np.hypot(np.diff(bp[:, 0]), np.diff(bp[:, 1]))))
            flag = " DROPOUT-trim" if dropped else (" trim" if end < N - 5 else "")
            print(f"{dk:9s}{tag:7s}{N:6d}{end:9d}{end/60:7.1f}{(N-end)/60:7.1f}{flag}")
    os.replace(tmp, args.out_hdf5)
    print(f"\nwrote {args.out_hdf5}")


if __name__ == "__main__":
    main()
