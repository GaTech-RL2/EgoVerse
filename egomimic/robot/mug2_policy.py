"""Deployment wrapper for the real_mug v2 bimanual policy (train_zarr_eve_mug2_bc_full).

Encapsulates every convention the model was trained with so a robot node only has to hand
over raw sensor values and gets back commands in the robot's own frames:

  observations (per arm, in THAT arm's base frame):
    ee pose  xyz + quat xyzw  (robot TCP = flange + (0.13, 0, 0.07), identity rotation)
    hand     17 ORCA joints   (recorded/controller convention, orca_core v1 order)
  images: top ("front_img_1"), left wrist, right wrist, RGB uint8 HxWx3 (trained at 240x432)

  model inputs  : ee_pose (12 = [L xyz ypr, R xyz ypr], ypr = ZYX Euler of R_tcp @ Rx(180deg)),
                  cartesian_arm (14 = [L xyz xyzw, R xyz xyzw], same flip), joint_positions (34 = [L, R])
  model outputs : actions_cartesian (100 x 12) = next 1.5 s of TCP poses in the flipped tool
                  frame, actions_joints (100 x 34) = next 1.5 s of hand targets.
  commands      : the flip is undone (R_cmd = R_from_ZYX(ypr) @ Rx(180deg)) -> xyz + quat xyzw per
                  arm in its own base frame; hand targets as-is.

`python -m egomimic.robot.mug2_policy --ckpt ... --h5 <processed v2 episode>` replays a recording
through the policy (same proprio the model was trained on, un-flipped back to robot poses first) and writes an overlay video of predicted vs recorded future trajectories + a numeric
error summary, so the whole path can be checked without the robot.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.embodiment.embodiment import get_embodiment_id

EMBODIMENT = "eve_bimanual"
TCP_OFFSET = np.array([0.13, 0.0, 0.07])          # Franka GUI flange -> EE, identity rotation
TOOL_FLIP = R.from_euler("x", 180, degrees=True)   # orientations are stored as R_tcp @ Rx(180)
CHUNK_LEN = 100                                    # steps per predicted chunk
CHUNK_SECONDS = 1.5                                # the chunk spans 1.5 s (45 frames @ 30 Hz resampled to 100)
HAND_DIM = 17
IMG_HW = (240, 432)


def _flip_quat_xyzw(q):
    return (R.from_quat(q) * TOOL_FLIP).as_quat()


def pose_to_model(xyz, quat_xyzw):
    """Robot TCP pose -> (6,) [xyz, ypr(ZYX)] and (7,) [xyz, quat xyzw] in the model's flipped tool frame."""
    r = R.from_quat(quat_xyzw) * TOOL_FLIP
    return np.concatenate([xyz, r.as_euler("ZYX")]), np.concatenate([xyz, r.as_quat()])


def model_to_pose(xyz_ypr):
    """Model (…,6) [xyz, ypr(ZYX)] in the flipped tool frame -> (…,7) [xyz, quat xyzw] robot TCP pose."""
    xyz_ypr = np.asarray(xyz_ypr); flat = xyz_ypr.reshape(-1, 6)
    quat = (R.from_euler("ZYX", flat[:, 3:]) * TOOL_FLIP).as_quat()     # (R_stored @ Rx) because Rx is its own inverse
    return np.concatenate([flat[:, :3], quat], 1).reshape(xyz_ypr.shape[:-1] + (7,))


def resample_chunk(chunk, n_out):
    """Resample a (T, D) chunk along time to n_out steps. 6-D pose blocks get angle-unwrapped
    interpolation (as faive2lerobot does); everything else is linear."""
    T, D = chunk.shape
    if T == n_out:
        return chunk.copy()
    t_in, t_out = np.linspace(0, 1, T), np.linspace(0, 1, n_out)
    out = np.empty((n_out, D))
    for d in range(D):
        col = chunk[:, d]
        if D % 6 == 0 and d % 6 >= 3:       # yaw/pitch/roll columns of a pose block
            col = np.unwrap(col)
            out[:, d] = (np.interp(t_out, t_in, col) + np.pi) % (2 * np.pi) - np.pi
        else:
            out[:, d] = np.interp(t_out, t_in, col)
    return out


@dataclass
class Mug2Observation:
    front_rgb: np.ndarray        # (H, W, 3) uint8 RGB, top camera
    left_wrist_rgb: np.ndarray
    right_wrist_rgb: np.ndarray
    ee_left: np.ndarray          # (7,) xyz + quat xyzw, left TCP in the LEFT base frame
    ee_right: np.ndarray         # (7,) xyz + quat xyzw, right TCP in the RIGHT base frame
    hand_left: np.ndarray        # (17,)
    hand_right: np.ndarray       # (17,)


@dataclass
class Mug2Actions:
    ee_left: np.ndarray          # (100, 7) xyz + quat xyzw, robot frame (flip undone)
    ee_right: np.ndarray
    hand_left: np.ndarray        # (100, 17)
    hand_right: np.ndarray
    raw_cartesian: np.ndarray    # (100, 12) model output (flipped frame), for viz/debug
    raw_joints: np.ndarray       # (100, 34)
    step_dt: float = CHUNK_SECONDS / CHUNK_LEN

    def at_rate(self, hz):
        """Resample the chunk to the control rate: returns the same fields with n = 1.5 s * hz steps."""
        n = max(2, int(round(CHUNK_SECONDS * hz)))
        cart = resample_chunk(self.raw_cartesian, n); joints = resample_chunk(self.raw_joints, n)
        return Mug2Actions(model_to_pose(cart[:, :6]), model_to_pose(cart[:, 6:]), joints[:, :HAND_DIM], joints[:, HAND_DIM:], cart, joints, 1.0 / hz)


class Mug2Policy:
    def __init__(self, ckpt_path: str, device: str | None = None, num_inference_steps: int | None = None, norm_stats_path: str | None = None):
        """norm_stats_path: optional <run>/norm_stats dir (or .json); only needed if the checkpoint
        was saved without stats (checkpoints from trainHydra.py embed them)."""
        from egomimic.pl_utils.pl_model import ModelWrapper
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.wrapper = ModelWrapper.load_from_checkpoint(ckpt_path, weights_only=False, map_location="cpu").to(self.device)
        self.wrapper.eval()
        self.model = self.wrapper.model
        self.model.device = self.device
        self.emb_id = get_embodiment_id(EMBODIMENT)
        self.emb_name = EMBODIMENT
        self.schematic = self.model.data_schematic
        if norm_stats_path or not self.schematic.norm_stats.get(self.emb_id):
            if not norm_stats_path:
                raise RuntimeError("checkpoint carries no normalization stats for eve_bimanual; pass norm_stats_path")
            self.schematic.infer_norm_from_dataset(None, self.emb_id, precomputed_norm_path=norm_stats_path)
        if not self.schematic.norm_stats.get(self.emb_id):
            raise RuntimeError(f"no normalization stats for eve_bimanual in {norm_stats_path}")
        if num_inference_steps is not None:
            for head in self.model.head_specs.values():
                if head is not None:
                    head.num_inference_steps = num_inference_steps
        self.cam_keys = list(self.model.camera_keys[self.emb_id])
        self.proprio_keys = list(self.model.proprio_keys[self.emb_id])

    # ---- batch construction ---------------------------------------------------------------
    @staticmethod
    def _img(rgb):
        if rgb.shape[:2] != IMG_HW:
            import cv2
            rgb = cv2.resize(rgb, (IMG_HW[1], IMG_HW[0]), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0

    def build_batch(self, obs: Mug2Observation):
        ypr_l, q_l = pose_to_model(obs.ee_left[:3], obs.ee_left[3:])
        ypr_r, q_r = pose_to_model(obs.ee_right[:3], obs.ee_right[3:])
        b = {
            "front_img_1": self._img(obs.front_rgb)[None],
            "left_wrist_img": self._img(obs.left_wrist_rgb)[None],
            "right_wrist_img": self._img(obs.right_wrist_rgb)[None],
            "ee_pose": torch.from_numpy(np.concatenate([ypr_l, ypr_r])).float()[None],
            "cartesian_arm": torch.from_numpy(np.concatenate([q_l, q_r])).float()[None],
            "joint_positions": torch.from_numpy(np.concatenate([obs.hand_left, obs.hand_right])).float()[None],
            # dummy targets: only their shapes are used at eval time (chunk length / dims)
            "actions_cartesian": torch.zeros(1, CHUNK_LEN, 12),
            "actions_joints": torch.zeros(1, CHUNK_LEN, 2 * HAND_DIM),
            "pad_mask": torch.ones(1, CHUNK_LEN, 1),
        }
        b = {k: v.to(self.device) for k, v in b.items()}
        b = self.schematic.normalize_data(b, self.emb_id)
        b["embodiment"] = torch.tensor([self.emb_id], dtype=torch.int64, device=self.device)
        return {self.emb_id: b}

    @torch.inference_mode()
    def predict(self, obs: Mug2Observation) -> Mug2Actions:
        out = self.model.forward_eval(self.build_batch(obs))
        cart = out[f"{self.emb_name}_actions_cartesian"][0].float().cpu().numpy()
        joints = out[f"{self.emb_name}_actions_joints"][0].float().cpu().numpy()
        return Mug2Actions(model_to_pose(cart[:, :6]), model_to_pose(cart[:, 6:]), joints[:, :HAND_DIM], joints[:, HAND_DIM:], cart, joints)


# ---- offline replay test -------------------------------------------------------------------
def _replay(args):
    """Feed a PROCESSED v2 episode (eth_clariden/mug/calib/raw_to_processed_v2.py layout: poses
    already FK'd + tool-flipped, images RGB) through the wrapper. The processed poses are
    un-flipped back to robot TCP poses first, so the wrapper's own flip/un-flip is exercised the
    same way it will be on the robot."""
    import cv2, h5py
    from egomimic.utils.egomimicUtils import EXTRINSICS, INTRINSICS, cam_frame_to_cam_pixels, ee_pose_to_cam_frame

    policy = Mug2Policy(args.ckpt, num_inference_steps=args.steps, norm_stats_path=args.norm_stats)
    f = h5py.File(args.h5, "r")
    extr = EXTRINSICS["mug2_front"]; K = INTRINSICS["mug_front"]
    imgs = {k: f[f"observations/images/{c}/color"] for k, c in (("front", "aria_rgb_cam"), ("left", "oakd_left_wrist_view"), ("right", "oakd_right_wrist_view"))}
    T = imgs["front"].shape[0]
    unflip = lambda p7: np.concatenate([p7[:3], (R.from_quat(p7[3:]) * TOOL_FLIP).as_quat()])   # processed (flipped) -> robot TCP

    def project(xyz, cam):      # base-frame points -> pixels through that arm's camera pose
        return cam_frame_to_cam_pixels(ee_pose_to_cam_frame(xyz, extr[cam]), K)[:, :2]

    frames, errs = [], []
    for i in range(0, T - 46, args.stride):
        obs = Mug2Observation(
            front_rgb=imgs["front"][i], left_wrist_rgb=imgs["left"][i], right_wrist_rgb=imgs["right"][i],
            ee_left=unflip(f["observations/qpos_arm_left"][i]), ee_right=unflip(f["observations/qpos_arm_right"][i]),
            hand_left=f["observations/qpos_hand_left"][i].astype(float), hand_right=f["observations/qpos_hand_right"][i].astype(float))
        if i == 0:   # the wrapper's flip must reproduce the processed proprio exactly
            b = policy.build_batch(obs)[policy.emb_id]
            q_proc = np.concatenate([f["observations/qpos_arm_left"][0], f["observations/qpos_arm_right"][0]])
            raw = policy.schematic.unnormalize_data({"cartesian_arm": b["cartesian_arm"]}, policy.emb_id)["cartesian_arm"][0].cpu().numpy()
            d = np.abs(raw - q_proc); d[[3, 4, 5, 6, 10, 11, 12, 13]] = np.minimum(d[[3, 4, 5, 6, 10, 11, 12, 13]], np.abs(raw + q_proc)[[3, 4, 5, 6, 10, 11, 12, 13]])  # q == -q
            print(f"proprio round-trip vs processed file: max |diff| = {d.max():.2e}")
        act = policy.predict(obs)
        # ground truth: recorded commanded poses over the next 1.5 s (45 frames -> 100 steps)
        gt = {s: resample_chunk(f[f"actions_arm_{s}"][i:i + 45, :3].astype(float), CHUNK_LEN) for s in ("left", "right")}
        e = {s: np.linalg.norm(getattr(act, f"ee_{s}")[:, :3] - gt[s], axis=1) for s in ("left", "right")}
        errs.append([e["left"][0], e["left"][50], e["left"][99], e["right"][0], e["right"][50], e["right"][99]])
        img = np.ascontiguousarray(obs.front_rgb)
        for s, cam in (("left", "left_cam"), ("right", "right_cam")):
            for pts, col in ((gt[s], (40, 220, 40)), (getattr(act, f"ee_{s}")[:, :3], (220, 40, 40))):
                for u, v in project(pts, cam):
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        cv2.circle(img, (int(u), int(v)), 2, col, -1)
        cv2.putText(img, f"t={i}  green=recorded next 1.5s  red=policy", (5, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        frames.append(img)
    errs = np.array(errs) * 1000
    print(f"{len(frames)} queries | xyz error (mm) at chunk step 0/50/99:  left {errs[:, :3].mean(0).round(1)}   right {errs[:, 3:].mean(0).round(1)}")
    if args.out:
        H, W = frames[0].shape[:2]
        vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 30 / args.stride, (W * 2, H * 2))
        for fr in frames:
            vw.write(cv2.cvtColor(cv2.resize(fr, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR))
        vw.release(); print("wrote", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay a processed v2 episode through the mug2 policy and report/visualize predictions.")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--h5", required=True, help="PROCESSED v2 episode (raw_to_processed_v2.py output)")
    ap.add_argument("--norm-stats", default=None, help="<run>/norm_stats dir; only if the ckpt lacks stats")
    ap.add_argument("--out", default=None, help="overlay mp4 (mp4v; re-encode with ffmpeg for browsers)")
    ap.add_argument("--stride", type=int, default=3, help="query the policy every N recorded frames")
    ap.add_argument("--steps", type=int, default=None, help="flow-matching inference steps (default: checkpoint's)")
    _replay(ap.parse_args())
