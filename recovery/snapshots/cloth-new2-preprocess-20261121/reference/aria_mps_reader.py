"""Stage A: parse the NEW Aria MPS export -> head + 2-wrist world poses + landmarks.

Runs in the `rby1` conda env (needs pandas + scipy). The new MPS export
(`mps_*_vrs/`) differs from the old custom dump the repo's parse_aria() expected:

  slam/closed_loop_trajectory.csv   1000 Hz  T_world_device (xyzw, scalar-last), gravity_world
  hand_tracking/hand_tracking_results.csv  30 Hz  per-hand: 21 landmarks + wrist 6DoF +
        palm/wrist normals, ALL in DEVICE frame; validity via *_tracking_confidence == -1.
  slam/online_calibration.jsonl     per-cam T_Device_Camera (we read camera-rgb).

World frame is gravity-aligned, Z-up (gravity_world = (0,0,-9.81)).

Output: an intermediate .npz resampled to `fps` (default 60 for the base egoposer.pth),
all poses lifted into the Z-up world frame, hand dropouts held at last-valid + a validity
mask. The "head" frame is X-fwd / Y-left / Z-up (the CAM2HEAD convention the repo uses);
EgoPoser's SMPL-joint convention offset is applied later (in the inference driver).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

# Aria RGB-camera -> head(X-fwd,Y-left,Z-up) extrinsic (R_head_cam), supplied by the user
# (same constant used in demo_aria_mink_retarget.py).
CAM2HEAD = np.array([
    [0.01000232, -0.1259538, 0.99198568],
    [-0.99818374, 0.05767904, 0.01738841],
    [-0.05940691, -0.9903579, -0.12514811],
])

N_LM = 21  # MPS hand landmarks per hand: 0=PALM_CENTER 1=WRIST 2..5 thumb .. fingertips 5/9/13/17/20


def _quat_xyzw(arr):
    """arr (...,4) in xyzw -> Rotation."""
    return Rot.from_quat(np.asarray(arr))


def _rgb_cam_R(calib_path):
    with open(calib_path) as f:
        rec = json.loads(f.readline())
    for c in rec["CameraCalibrations"]:
        if c["Label"] == "camera-rgb":
            w, (x, y, z) = c["T_Device_Camera"]["UnitQuaternion"]
            return Rot.from_quat([x, y, z, w]).as_matrix()
    raise RuntimeError("camera-rgb not found in online_calibration.jsonl")


def parse_mps(mps_dir, fps=60.0):
    mps_dir = str(mps_dir)
    traj = pd.read_csv(os.path.join(mps_dir, "slam", "closed_loop_trajectory.csv"))
    hands = pd.read_csv(os.path.join(mps_dir, "hand_tracking", "hand_tracking_results.csv"))
    R_dev_rgb = _rgb_cam_R(os.path.join(mps_dir, "slam", "online_calibration.jsonl"))
    R_dev_head = R_dev_rgb @ CAM2HEAD.T   # device -> head (X-fwd,Y-left,Z-up)

    # ---- device pose stream (1000 Hz) ----
    t_dev = traj["tracking_timestamp_us"].to_numpy().astype(np.float64)
    p_dev = traj[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy()
    q_dev = traj[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].to_numpy()
    R_dev = _quat_xyzw(q_dev)
    slerp_dev = Slerp(t_dev, R_dev)
    grav = traj[["gravity_x_world", "gravity_y_world", "gravity_z_world"]].to_numpy().mean(0)

    # ---- hand stream (30 Hz) ----
    t_hand = hands["tracking_timestamp_us"].to_numpy().astype(np.float64)

    def lm_block(side):  # (M,21,3) device-frame landmarks
        cols = []
        for k in range(N_LM):
            cols += [f"t{a}_{side}_landmark_{k}_device" for a in "xyz"]
        return hands[cols].to_numpy().reshape(-1, N_LM, 3)

    L = {}
    for side in ("left", "right"):
        conf = hands[f"{side}_tracking_confidence"].to_numpy()
        valid = conf >= 0.0
        wp = hands[[f"t{a}_{side}_device_wrist" for a in "xyz"]].to_numpy()
        wq = hands[[f"q{a}_{side}_device_wrist" for a in "xyzw"]].to_numpy()
        palm_n = hands[[f"n{a}_{side}_palm_device" for a in "xyz"]].to_numpy()
        L[side] = dict(valid=valid, wp=wp, wq=wq, lm=lm_block(side), palm_n=palm_n)

    # ---- common 60 Hz timeline (overlap of both streams) ----
    t0 = max(t_dev[0], t_hand[0])
    t1 = min(t_dev[-1], t_hand[-1])
    n = int((t1 - t0) * 1e-6 * fps)
    grid = t0 + np.arange(n) * (1e6 / fps)
    grid = np.clip(grid, t_dev[0], t_dev[-1])

    # head pose at grid (device interp -> head frame)
    p_dev_g = np.column_stack([np.interp(grid, t_dev, p_dev[:, i]) for i in range(3)])
    R_dev_g = slerp_dev(grid)
    R_world_dev_g = R_dev_g.as_matrix()
    head_R = R_world_dev_g @ R_dev_head            # (n,3,3)
    head_p = p_dev_g                               # head origin ~ device origin (CPF offset ~cm)

    out = dict(fps=float(fps), ts=grid, head_R=head_R, head_p=head_p, gravity=grav,
               n=n, mps_dir=mps_dir)

    # hands: lift device->world at each grid time; hold last-valid on dropout
    for side in ("left", "right"):
        s = L[side]
        # validity at grid: nearest hand frame valid?
        j = np.clip(np.searchsorted(t_hand, grid), 0, len(t_hand) - 1)
        jm = np.clip(j - 1, 0, len(t_hand) - 1)
        near = np.where(np.abs(t_hand[j] - grid) <= np.abs(t_hand[jm] - grid), j, jm)
        valid_g = s["valid"][near]

        # interpolate wrist pose over the FULL hand timeline (ignore validity for the
        # interp; we mask with valid_g and hold last-valid below)
        wp_g = np.column_stack([np.interp(grid, t_hand, s["wp"][:, i]) for i in range(3)])
        # robust quat interp: clamp grid into hand span for Slerp
        gg = np.clip(grid, t_hand[0], t_hand[-1])
        wq_clean = s["wq"].copy()
        bad = ~np.isfinite(wq_clean).all(1) | (np.linalg.norm(wq_clean, axis=1) < 1e-6)
        wq_clean[bad] = np.array([0.0, 0.0, 0.0, 1.0])
        wR_g = Slerp(t_hand, _quat_xyzw(wq_clean))(gg).as_matrix()
        lm_g = np.stack([np.column_stack(
            [np.interp(grid, t_hand, s["lm"][:, k, i]) for i in range(3)]) for k in range(N_LM)], axis=1)
        pn_g = np.column_stack([np.interp(grid, t_hand, s["palm_n"][:, i]) for i in range(3)])

        # lift device->world
        wR_w = R_world_dev_g @ wR_g
        wp_w = np.einsum("nij,nj->ni", R_world_dev_g, wp_g) + p_dev_g
        lm_w = np.einsum("nij,nkj->nki", R_world_dev_g, lm_g) + p_dev_g[:, None, :]
        pn_w = np.einsum("nij,nj->ni", R_world_dev_g, pn_g)

        # hold last-valid through dropouts (forward fill)
        wR_h, wp_h, lm_h, pn_h = wR_w.copy(), wp_w.copy(), lm_w.copy(), pn_w.copy()
        last = None
        for i in range(n):
            if valid_g[i]:
                last = i
            elif last is not None:
                wR_h[i], wp_h[i], lm_h[i], pn_h[i] = wR_w[last], wp_w[last], lm_w[last], pn_w[last]
        # a hand not yet tracked at the start: hold its first valid frame (MPS writes zeros there -> singular EgoPoser input)
        if valid_g.any() and not valid_g[0]:
            f = int(np.argmax(valid_g))
            wR_h[:f], wp_h[:f], lm_h[:f], pn_h[:f] = wR_w[f], wp_w[f], lm_w[f], pn_w[f]
        out[f"{side}_wrist_R"] = wR_h
        out[f"{side}_wrist_p"] = wp_h
        out[f"{side}_landmarks"] = lm_h
        out[f"{side}_palm_n"] = pn_h
        out[f"{side}_valid"] = valid_g

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mps_dir", required=True)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = parse_mps(args.mps_dir, fps=args.fps)
    np.savez_compressed(args.out, **{k: v for k, v in d.items() if k != "mps_dir"})
    print(f"saved {args.out}  n={d['n']} @ {d['fps']}Hz")
    print(f"gravity_world = {d['gravity']}  (z-up if z<<0)")
    hp = d["head_p"]
    print(f"head_p z range: {hp[:,2].min():.2f}..{hp[:,2].max():.2f}  "
          f"(walked {np.linalg.norm(hp[-1,:2]-hp[0,:2]):.2f} m horizontally)")
    for side in ("left", "right"):
        v = d[f"{side}_valid"]
        wz = d[f"{side}_wrist_p"][:, 2]
        print(f"  {side}: valid {v.mean()*100:.0f}%  wrist z {wz.min():.2f}..{wz.max():.2f}  "
              f"(head-wrist dz mean {np.mean(hp[:,2]-wz):.2f} m, expect >0)")


if __name__ == "__main__":
    main()
