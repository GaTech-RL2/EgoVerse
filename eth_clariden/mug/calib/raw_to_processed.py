#!/usr/bin/env python3
"""
Convert raw cvg-dex-teleop recordings (single Franka Panda + ORCA right hand,
30 Hz) into the "processed FAIVE" H5 layout that convert_faive_to_lerobot.py
consumes (see FaiveHD5Extractor.process_episode docstring).

RAW layout (recorder):                       PROCESSED layout (this script):
  arm/current_joint   (T,9)  measured joints   observations/qpos_arm_right   (T,7) FK(q[:7]) + tool offset, xyz + quat xyzw
  arm/ee_target       (T,7)  cmd xyz + xyzw    actions_arm_right             (T,7) copied as-is (already xyzw)
  hand/current_joint  (T,17) measured          observations/qpos_hand_right  (T,17)
  hand/target_joint   (T,17) commanded         actions_hand_right            (T,17)
  camera/front        (T,H,W,3)                observations/images/aria_rgb_cam/color            -> front_img_1
  camera/wrist        (T,H,W,3)                observations/images/oakd_right_wrist_view/color   -> right_wrist_img
  depth/*, hand/landmarks_odom, wrench, ...    dropped

Conventions were verified against the recorded commanded pose on all 148
episodes (fk_final.py): Panda modified-DH FK + constant tool offset
[0.1325, -0.0049, 0.0735] m in the flange frame reproduces arm/ee_target to
7.8 mm / 2.5 deg (= controller tracking lag; measured trails commanded by ~3
frames). Quaternion order in the raw file is xyzw (2.3 deg residual vs 40 deg
for wxyz), matching what the converter expects ([x,y,z,qx,qy,qz,qw]).
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

# ---------------------------------------------------------------- Panda FK
# Craig (modified) DH parameters, flange frame (d8 = 0.107), no gripper.
_A = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
_D = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
_ALPHA = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0]
# Tool offset (flange frame) fitted on the recordings; rotation offset is identity.
TOOL_OFFSET = np.array([0.1325, -0.0049, 0.0735])
# Virtual tool-frame flip. The recorded tool frame has its rest orientation at a
# 180 deg rotation about x, which puts the ZYX yaw used by the converter on the
# +-180 seam for ~15% of frames (and inside action chunks). Post-multiplying by
# Rx(180) moves rest to identity: 0% seam frames for every Euler convention
# (seam_check.py). Applied identically to qpos and actions; at deployment,
# post-multiply the predicted orientation by Rx(180) again to get the real tool.
TOOL_FLIP_XYZW = np.array([1.0, 0.0, 0.0, 0.0])   # Rx(180 deg) as xyzw


def quat_mul_xyzw(q1, q2):
    """Hamilton product q1 * q2 for (N,4) xyzw arrays (rotation q1 then... i.e. R(q1) @ R(q2))."""
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
    return np.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], axis=1)


def apply_tool_flip(pose7):
    """(N,7) xyz+xyzw -> same position, orientation R @ Rx(180)."""
    out = pose7.copy()
    out[:, 3:] = quat_mul_xyzw(pose7[:, 3:], TOOL_FLIP_XYZW)
    return out


def _tf(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array(
        [[ct, -st, 0, a],
         [st * ca, ct * ca, -sa, -sa * d],
         [st * sa, ct * sa, ca, ca * d],
         [0, 0, 0, 1]]
    )


def panda_flange(q):
    T = np.eye(4)
    for i in range(7):
        T = T @ _tf(_A[i], _D[i], _ALPHA[i], q[i])
    return T @ _tf(_A[7], _D[7], _ALPHA[7], 0.0)


def mat2quat_xyzw(R):
    """Rotation matrix (3,3) -> unit quaternion [x,y,z,w] (Shepperd's method)."""
    m = R
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def ee_pose_from_joints(q7, ref_quat=None):
    """(T,7) joints -> (T,7) xyz + xyzw in the same frame as arm/ee_target.

    ref_quat: optional (T,4) xyzw to pick the quaternion sign continuous with
    the commanded pose (q and -q are the same rotation; keep them close so
    obs/action quaternions don't flip sign relative to each other).
    """
    out = np.zeros((len(q7), 7))
    for i, q in enumerate(q7):
        T = panda_flange(q)
        out[i, :3] = T[:3, 3] + T[:3, :3] @ TOOL_OFFSET
        quat = mat2quat_xyzw(T[:3, :3])
        if ref_quat is not None and np.dot(quat, ref_quat[i]) < 0:
            quat = -quat
        out[i, 3:] = quat
    return out


# ---------------------------------------------------------------- conversion
def convert_file(src, dst, min_frames, upsample_hw=None, compression="lzf"):
    with h5py.File(src, "r") as f:
        T = f["timestamps_ns"].shape[0]
        if T < min_frames:
            return None, f"skip ({T} frames < {min_frames})"
        q = f["arm/current_joint"][:, :7].astype(np.float64)
        ee_cmd = f["arm/ee_target"][:].astype(np.float64)          # xyz + xyzw
        hand_obs = f["hand/current_joint"][:].astype(np.float64)
        hand_cmd = f["hand/target_joint"][:].astype(np.float64)
        front = f["camera/front"][:]
        wrist = f["camera/wrist"][:]
        attrs = {k: f.attrs[k] for k in f.attrs}

    assert ee_cmd.shape == (T, 7) and hand_obs.shape == (T, 17) and hand_cmd.shape == (T, 17)
    assert front.shape[0] == T and wrist.shape[0] == T and front.dtype == np.uint8
    qn = np.linalg.norm(ee_cmd[:, 3:], axis=1)
    assert np.allclose(qn, 1.0, atol=1e-3), "ee_target quaternions not unit norm"

    ee_obs = ee_pose_from_joints(q, ref_quat=ee_cmd[:, 3:])
    resid = np.linalg.norm(ee_obs[:, :3] - ee_cmd[:, :3], axis=1)
    # Same virtual tool frame for proprio and actions (see TOOL_FLIP_XYZW).
    ee_obs = apply_tool_flip(ee_obs)
    ee_cmd = apply_tool_flip(ee_cmd)

    if upsample_hw is not None:
        front = _resize_batch(front, upsample_hw)
        wrist = _resize_batch(wrist, upsample_hw)

    tmp = dst + ".tmp"
    with h5py.File(tmp, "w") as o:
        for k, v in attrs.items():
            o.attrs[k] = v
        o.attrs["raw_source"] = os.path.abspath(src)
        o.attrs["arm"] = "right"
        o.attrs["qpos_arm_source"] = "panda_fk(arm/current_joint) + tool_offset"
        o.attrs["tool_offset_flange_frame_m"] = TOOL_OFFSET
        o.attrs["quat_order"] = "xyzw"
        o.attrs["tool_frame_flip_xyzw"] = TOOL_FLIP_XYZW
        o.attrs["tool_frame_note"] = "orientations are R_tool @ Rx(180deg); post-multiply predictions by Rx(180deg) to recover the recorded tool frame"
        o.create_dataset("observations/qpos_arm_right", data=ee_obs)
        o.create_dataset("observations/qpos_hand_right", data=hand_obs)
        o.create_dataset("actions_arm_right", data=ee_cmd)
        o.create_dataset("actions_hand_right", data=hand_cmd)
        for name, arr in (("aria_rgb_cam", front), ("oakd_right_wrist_view", wrist)):
            o.create_dataset(
                f"observations/images/{name}/color", data=arr,
                chunks=(1,) + arr.shape[1:], compression=compression,
            )
    os.replace(tmp, dst)
    return T, f"ok  T={T}  |fk-cmd| mean {resid.mean()*1000:.1f} mm  max {resid.max()*1000:.1f} mm"


def _resize_batch(imgs, hw):
    import cv2  # only needed with --upsample
    H, W = hw
    return np.stack([cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR) for im in imgs])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-frames", type=int, default=60, help="drop episodes shorter than this (2 broken 7-8 frame files)")
    ap.add_argument("--upsample", default=None, help="optional HxW, e.g. 480x640, to match the old Aria front-cam token geometry")
    ap.add_argument("--limit", type=int, default=None, help="only convert the first N files (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    hw = tuple(int(x) for x in args.upsample.split("x")) if args.upsample else None
    files = sorted(glob.glob(os.path.join(args.raw_dir, "*.h5")))
    if args.limit:
        files = files[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(files)} raw files -> {args.out_dir}  (upsample={hw})")
    n_ok = n_skip = 0
    total_frames = 0
    for i, src in enumerate(files):
        dst = os.path.join(args.out_dir, os.path.basename(src))
        if os.path.exists(dst) and not args.overwrite:
            print(f"[{i+1}/{len(files)}] {os.path.basename(src)}: exists, skipping")
            n_ok += 1
            continue
        T, msg = convert_file(src, dst, args.min_frames, hw)
        print(f"[{i+1}/{len(files)}] {os.path.basename(src)}: {msg}", flush=True)
        if T is None:
            n_skip += 1
        else:
            n_ok += 1
            total_frames += T
    print(f"done: {n_ok} written, {n_skip} skipped, {total_frames} frames ({total_frames/30/60:.1f} min @30Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
