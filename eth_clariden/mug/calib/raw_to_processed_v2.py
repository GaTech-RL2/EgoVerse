#!/usr/bin/env python3
"""
Convert v2 cvg-dex-teleop recordings (BIMANUAL: two Franka Panda/FR3 + ORCA hands, 30 Hz)
into the processed FAIVE H5 layout consumed by convert_faive_to_lerobot.py (--arm both|right).

RAW v2 layout:                                  PROCESSED layout (this script):
  arm/current_joint/{left,right}  (T,9)           observations/qpos_arm_{left,right}   (T,7) FK(q[:7]) + TCP, xyz + quat xyzw
  arm/ee_target/{left,right}      (T,7) xyz+xyzw  actions_arm_{left,right}             (T,7)
  hand/current_joint/{left,right} (T,17)          observations/qpos_hand_{left,right}  (T,17)
  hand/target_joint/{left,right}  (T,17)          actions_hand_{left,right}            (T,17)
  camera/image/top     (T,H,W,3) BGR              observations/images/aria_rgb_cam/color           -> front_img_1
  camera/image/left    (T,H,W,3) BGR              observations/images/oakd_left_wrist_view/color   -> left_wrist_img
  camera/image/right   (T,H,W,3) BGR              observations/images/oakd_right_wrist_view/color  -> right_wrist_img
  camera/depth/top, hand/landmarks_odom, wrench, gripper_target, calibration  -> dropped

Conventions (same as the single-arm v1 adapter, verified on the v1 mug data and re-checked on v2):
  * FK: Panda modified-DH, controller TCP = flange + (0.13, 0, 0.07) m, identity rotation
    (Franka GUI "Flange to End-Effector"). Quaternions are xyzw.
  * Each arm's poses stay in ITS OWN base frame (that is how arm/ee_target is recorded and how
    the EVE bimanual data was handled: per-arm extrinsics do the base->camera step at viz time).
    Per Alexey the left base = right base translated +0.7 m along the right base's +Y.
  * Tool-frame flip: orientations post-multiplied by Rx(180 deg) so the converter's ZYX yaw never
    sits on the +-180 seam. Deployment must post-multiply predictions by Rx(180 deg).
  * Images are stored BGR by the recorder -> swapped to RGB here.
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

# ---------------------------------------------------------------- Panda FK (same as v1)
_A = [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0]
_D = [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107]
_ALPHA = [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0]
TCP_OFFSET = np.array([0.13, 0.0, 0.07])          # Franka GUI flange->EE (identity rotation)
TOOL_FLIP_XYZW = np.array([1.0, 0.0, 0.0, 0.0])   # Rx(180 deg) as xyzw


def _tf(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -sa * d], [st * sa, ct * sa, ca, ca * d], [0, 0, 0, 1]])


def panda_flange(q):
    T = np.eye(4)
    for i in range(7):
        T = T @ _tf(_A[i], _D[i], _ALPHA[i], q[i])
    return T @ _tf(_A[7], _D[7], _ALPHA[7], 0.0)


def mat2quat_xyzw(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w]); return q / np.linalg.norm(q)


def quat_mul_xyzw(q1, q2):
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]; x2, y2, z2, w2 = q2
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=1)


def tcp_pose_from_joints(q7, ref_quat):
    out = np.zeros((len(q7), 7))
    for i, q in enumerate(q7):
        T = panda_flange(q); out[i, :3] = T[:3, 3] + T[:3, :3] @ TCP_OFFSET
        quat = mat2quat_xyzw(T[:3, :3])
        if np.dot(quat, ref_quat[i]) < 0:
            quat = -quat
        out[i, 3:] = quat
    return out


def apply_tool_flip(pose7):
    out = pose7.copy(); out[:, 3:] = quat_mul_xyzw(pose7[:, 3:], TOOL_FLIP_XYZW); return out


CAMS = {"top": "aria_rgb_cam", "left": "oakd_left_wrist_view", "right": "oakd_right_wrist_view"}


def convert_file(src, dst, arms, min_frames, compression="lzf"):
    with h5py.File(src, "r") as f:
        T = f["timestamps_ns"].shape[0]
        if T < min_frames:
            return None, f"skip ({T} frames < {min_frames})"
        attrs = {k: f.attrs[k] for k in f.attrs}
        arm_data = {}
        for side in arms:
            q = f[f"arm/current_joint/{side}"][:, :7].astype(np.float64)
            ee_cmd = f[f"arm/ee_target/{side}"][:].astype(np.float64)
            assert ee_cmd.shape == (T, 7) and np.allclose(np.linalg.norm(ee_cmd[:, 3:], axis=1), 1.0, atol=1e-3)
            ee_obs = tcp_pose_from_joints(q, ref_quat=ee_cmd[:, 3:])
            resid = np.linalg.norm(ee_obs[:, :3] - ee_cmd[:, :3], axis=1)
            arm_data[side] = dict(qpos_arm=apply_tool_flip(ee_obs), act_arm=apply_tool_flip(ee_cmd),
                                  qpos_hand=f[f"hand/current_joint/{side}"][:].astype(np.float64),
                                  act_hand=f[f"hand/target_joint/{side}"][:].astype(np.float64), resid=resid)
            assert arm_data[side]["qpos_hand"].shape == (T, 17) and arm_data[side]["act_hand"].shape == (T, 17)
        cams = {}
        wanted = ["top"] + [s for s in ("left", "right") if s in arms]
        for cam in wanted:
            img = f[f"camera/image/{cam}"][:]
            assert img.shape[0] == T and img.dtype == np.uint8
            cams[cam] = np.ascontiguousarray(img[..., ::-1])    # BGR -> RGB
    tmp = dst + ".tmp"
    with h5py.File(tmp, "w") as o:
        for k, v in attrs.items():
            o.attrs[k] = v
        o.attrs["raw_source"] = os.path.abspath(src); o.attrs["layout"] = "v2_bimanual"; o.attrs["arms"] = ",".join(arms)
        o.attrs["qpos_arm_source"] = "panda_fk(arm/current_joint) + TCP (0.13,0,0.07)"; o.attrs["quat_order"] = "xyzw"
        o.attrs["tool_frame_flip_xyzw"] = TOOL_FLIP_XYZW; o.attrs["frames"] = "each arm in its own base frame; left base = right base + (0, 0.7, 0) m"
        o.attrs["image_channel_order"] = "RGB (converted from the recorder's BGR)"
        for side, d in arm_data.items():
            o.create_dataset(f"observations/qpos_arm_{side}", data=d["qpos_arm"]); o.create_dataset(f"observations/qpos_hand_{side}", data=d["qpos_hand"])
            o.create_dataset(f"actions_arm_{side}", data=d["act_arm"]); o.create_dataset(f"actions_hand_{side}", data=d["act_hand"])
        for cam, arr in cams.items():
            o.create_dataset(f"observations/images/{CAMS[cam]}/color", data=arr, chunks=(1,) + arr.shape[1:], compression=compression)
    os.replace(tmp, dst)
    res = "  ".join(f"{s}: |fk-cmd| {d['resid'].mean()*1000:.1f} mm" for s, d in arm_data.items())
    return T, f"ok  T={T}  {res}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", default="left,right", help="comma list: left,right (bimanual) or right")
    ap.add_argument("--min-frames", type=int, default=60); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    arms = [s.strip() for s in a.arms.split(",")]
    files = sorted(glob.glob(os.path.join(a.raw_dir, "*.h5")))[: a.limit]
    os.makedirs(a.out_dir, exist_ok=True); print(f"{len(files)} raw v2 files -> {a.out_dir}  arms={arms}")
    n_ok = n_skip = total = 0
    for i, src in enumerate(files):
        dst = os.path.join(a.out_dir, os.path.basename(src))
        if os.path.exists(dst) and not a.overwrite:
            print(f"[{i+1}/{len(files)}] {os.path.basename(src)}: exists, skipping"); n_ok += 1; continue
        T, msg = convert_file(src, dst, arms, a.min_frames)
        print(f"[{i+1}/{len(files)}] {os.path.basename(src)}: {msg}", flush=True)
        if T is None: n_skip += 1
        else: n_ok += 1; total += T
    print(f"done: {n_ok} written, {n_skip} skipped, {total} frames ({total/30/60:.1f} min @30Hz)")


if __name__ == "__main__":
    sys.exit(main())
