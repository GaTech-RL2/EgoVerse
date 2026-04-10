#!/usr/bin/env python3
"""
IK round-trip quality test for Scale ALOHA (Trossen WXAI) arms.

Pipeline per timestep, per arm:
  1. original_joints (from MCAP)
  2. FK(original_joints) → pos, rot_mat
  3. IK(pos, rot_mat, seed=original_joints) → recovered_joints
  4. FK(recovered_joints) → recovered_pos, recovered_rot
  5. Compare original vs recovered

Generates:
  - Position error (mm) over time
  - Orientation error (deg) over time
  - Per-joint error (rad) over time
  - Projected EE overlay: original (green) vs round-trip (blue)
  - Combined video with error stats
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import simplejpeg
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from egomimic.robot.scale_aloha.scale_aloha_kinematics import (
    MCAP_TO_FK_INDICES,
    ScaleAlohaMinkKinematicsSolver,
)

# cam_high intrinsics from MCAP CameraInfo at native 1280x720,
# scaled for 640x480 output (sx=0.5, sy=0.667).
CAM_INTRINSICS_640x480 = np.array(
    [[324.40, 0.0, 324.79, 0.0], [0.0, 432.06, 240.32, 0.0], [0.0, 0.0, 1.0, 0.0]]
)

# Cam-to-base extrinsics from MCAP /tf_static.
# Convention: T_base_from_cam; inv() gives T_cam_from_base.
EXTRINSICS = {
    "left": np.array([
        [ 1.00000000, -0.00000000,  0.00000000,  0.44850000],
        [-0.00000000, -0.90632455,  0.42258232, -0.31017500],
        [ 0.00000000, -0.42258232, -0.90632455,  1.00777400],
        [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
    ]),
    "right": np.array([
        [-1.00000000, -0.00000000,  0.00000000,  0.46650000],
        [-0.00000000,  0.90632455, -0.42258232,  0.31017500],
        [ 0.00000000, -0.42258232, -0.90632455,  1.00777400],
        [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
    ]),
}

T_CAM_FROM_LEFT = np.linalg.inv(EXTRINSICS["left"])
T_CAM_FROM_RIGHT = np.linalg.inv(EXTRINSICS["right"])


def load_episode(zarr_path: str) -> dict:
    """Load a Zarr episode into a dict."""
    import zarr
    store = zarr.open_group(zarr_path, mode="r")
    data = {}
    data["joint_positions"] = np.array(store["observations.state.joint_positions"])
    if "observations.images.front_img_1" in store:
        data["images_raw"] = store["observations.images.front_img_1"][:]
    data["attrs"] = dict(store.attrs)
    return data


def _decode_jpeg(raw) -> np.ndarray:
    """Decode a single JPEG from Zarr object array."""
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        raw = raw.item()
    if isinstance(raw, bytes):
        return simplejpeg.decode_jpeg(raw, colorspace="RGB")
    return raw


def pos_to_cam(pos_base: np.ndarray, T_cam_from_base: np.ndarray) -> np.ndarray:
    """Transform (3,) base-frame position to camera frame."""
    return (T_cam_from_base @ np.append(pos_base, 1.0))[:3]


def project_to_pixel(pos_cam: np.ndarray, K: np.ndarray) -> tuple[int, int]:
    """Project camera-frame position to pixel coordinates."""
    proj = K @ np.append(pos_cam, 1.0)
    if proj[2] <= 0:
        return -1, -1
    return int(proj[0] / proj[2]), int(proj[1] / proj[2])


def run_roundtrip(solver: ScaleAlohaMinkKinematicsSolver,
                  joints_7: np.ndarray,
                  seed_joints: np.ndarray | None = None) -> dict:
    """Run FK → IK → FK round-trip for a single arm, single timestep.

    Args:
        joints_7: ground-truth 7-element MCAP joint vector
        seed_joints: 6-element joint seed for IK (simulates inference).
                     If None, uses ground-truth (trivial case).
    """
    fk_joints = joints_7[MCAP_TO_FK_INDICES]

    pos_orig, rot_orig = solver.fk_single_pos_rot(fk_joints)
    rot_mat_orig = rot_orig.as_matrix()

    if seed_joints is None:
        seed_joints = fk_joints.copy()

    t0 = time.perf_counter()
    recovered_joints = solver.ik(pos_orig, rot_mat_orig, seed_joints)
    ik_time = time.perf_counter() - t0

    if recovered_joints is None:
        return {
            "success": False,
            "pos_orig": pos_orig,
            "rot_orig": rot_orig,
            "ik_time": ik_time,
        }

    pos_recov, rot_recov = solver.fk_single_pos_rot(recovered_joints)

    pos_err_mm = np.linalg.norm(pos_orig - pos_recov) * 1000
    rot_err_deg = np.abs(
        (rot_orig.inv() * rot_recov).magnitude()
    ) * 180 / np.pi
    joint_err_rad = np.abs(fk_joints - recovered_joints)

    return {
        "success": True,
        "pos_orig": pos_orig,
        "rot_orig": rot_orig,
        "pos_recov": pos_recov,
        "rot_recov": rot_recov,
        "orig_joints": fk_joints.copy(),
        "recov_joints": recovered_joints.copy(),
        "pos_err_mm": pos_err_mm,
        "rot_err_deg": rot_err_deg,
        "joint_err_rad": joint_err_rad,
        "ik_time": ik_time,
    }


def main():
    parser = argparse.ArgumentParser(description="IK round-trip quality test")
    parser.add_argument("zarr_path", help="Path to a .zarr episode")
    parser.add_argument("--out-dir", default="./ik_quality_out")
    parser.add_argument("--max-frames", type=int, default=0, help="0=all")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--step", type=int, default=1,
                        help="Process every Nth frame (default: 1 = all)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading episode...")
    data = load_episode(args.zarr_path)
    joints = data["joint_positions"]  # (T, 14)
    T_total = joints.shape[0]
    T = T_total if args.max_frames <= 0 else min(T_total, args.max_frames)
    indices = list(range(0, T, args.step))
    N = len(indices)
    print(f"  {T_total} total frames, testing {N} (step={args.step})")

    print("Initializing solver...")
    solver = ScaleAlohaMinkKinematicsSolver()

    # Three test scenarios:
    # 1. "prev_step": seed IK with previous timestep's RECOVERED joints (inference sim)
    # 2. "home":      seed IK with zero joints (cold-start worst case)
    # 3. "gt_seed":   seed IK with ground-truth joints (trivial baseline)
    scenarios = {
        "prev_step": "Seed = previous step recovered joints (inference simulation)",
        "home": "Seed = home/zero position (cold-start)",
        "gt_seed": "Seed = ground-truth joints (trivial baseline)",
    }

    all_scenario_results: dict[str, dict] = {}

    for scenario_name, scenario_desc in scenarios.items():
        print(f"\n--- Scenario: {scenario_name} ---")
        print(f"    {scenario_desc}")

        left_results = []
        right_results = []
        prev_left_joints = np.zeros(6)
        prev_right_joints = np.zeros(6)

        t0_sc = time.perf_counter()
        for i, t in enumerate(indices):
            if scenario_name == "gt_seed":
                l_seed = None  # uses ground truth inside run_roundtrip
                r_seed = None
            elif scenario_name == "home":
                l_seed = np.zeros(6)
                r_seed = np.zeros(6)
            else:  # prev_step
                l_seed = prev_left_joints.copy()
                r_seed = prev_right_joints.copy()

            lr = run_roundtrip(solver, joints[t, :7], seed_joints=l_seed)
            rr = run_roundtrip(solver, joints[t, 7:], seed_joints=r_seed)
            left_results.append(lr)
            right_results.append(rr)

            if lr["success"]:
                prev_left_joints = lr["recov_joints"].copy()
            if rr["success"]:
                prev_right_joints = rr["recov_joints"].copy()

            if (i + 1) % 50 == 0 or i == N - 1:
                elapsed = time.perf_counter() - t0_sc
                print(f"  {i + 1}/{N} done ({elapsed:.1f}s, "
                      f"~{elapsed / (i + 1) * 1000:.0f}ms/frame)")

        all_scenario_results[scenario_name] = {
            "left": left_results,
            "right": right_results,
            "desc": scenario_desc,
        }

    # Print summary for each scenario
    for scenario_name, sr in all_scenario_results.items():
        left_results = sr["left"]
        right_results = sr["right"]

        l_pos_err = [r["pos_err_mm"] for r in left_results if r["success"]]
        r_pos_err = [r["pos_err_mm"] for r in right_results if r["success"]]
        l_rot_err = [r["rot_err_deg"] for r in left_results if r["success"]]
        r_rot_err = [r["rot_err_deg"] for r in right_results if r["success"]]
        l_ik_time = [r["ik_time"] for r in left_results]
        r_ik_time = [r["ik_time"] for r in right_results]
        l_joint_err = np.array([r["joint_err_rad"] for r in left_results if r["success"]])
        r_joint_err = np.array([r["joint_err_rad"] for r in right_results if r["success"]])
        l_fail = sum(1 for r in left_results if not r["success"])
        r_fail = sum(1 for r in right_results if not r["success"])

        print(f"\n{'='*60}")
        print(f"IK Round-Trip: {scenario_name}")
        print(f"  {sr['desc']}")
        print(f"{'='*60}")
        print(f"Frames tested: {N}   IK failures: left={l_fail}, right={r_fail}")
        if l_pos_err:
            print(f"\nPosition error (mm):")
            print(f"  Left:  mean={np.mean(l_pos_err):.3f}  median={np.median(l_pos_err):.3f}  "
                  f"max={np.max(l_pos_err):.3f}  p95={np.percentile(l_pos_err, 95):.3f}")
            print(f"  Right: mean={np.mean(r_pos_err):.3f}  median={np.median(r_pos_err):.3f}  "
                  f"max={np.max(r_pos_err):.3f}  p95={np.percentile(r_pos_err, 95):.3f}")
            print(f"\nOrientation error (deg):")
            print(f"  Left:  mean={np.mean(l_rot_err):.3f}  median={np.median(l_rot_err):.3f}  "
                  f"max={np.max(l_rot_err):.3f}  p95={np.percentile(l_rot_err, 95):.3f}")
            print(f"  Right: mean={np.mean(r_rot_err):.3f}  median={np.median(r_rot_err):.3f}  "
                  f"max={np.max(r_rot_err):.3f}  p95={np.percentile(r_rot_err, 95):.3f}")
            print(f"\nPer-joint error (rad):")
            for j in range(6):
                print(f"  Joint {j}: left mean={np.mean(l_joint_err[:, j]):.5f}  "
                      f"right mean={np.mean(r_joint_err[:, j]):.5f}  "
                      f"left max={np.max(l_joint_err[:, j]):.5f}  "
                      f"right max={np.max(r_joint_err[:, j]):.5f}")
            print(f"\nIK solve time: mean={np.mean(l_ik_time + r_ik_time)*1000:.1f}ms  "
                  f"max={np.max(l_ik_time + r_ik_time)*1000:.1f}ms")
        print(f"{'='*60}")

    # --- Plots: one row per scenario ---
    scenario_names = list(all_scenario_results.keys())
    colors = {"prev_step": "tab:blue", "home": "tab:red", "gt_seed": "tab:green"}
    labels = {"prev_step": "prev-step seed", "home": "home seed", "gt_seed": "GT seed"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for sn in scenario_names:
        sr = all_scenario_results[sn]
        lpe = [r["pos_err_mm"] for r in sr["left"] if r["success"]]
        rpe = [r["pos_err_mm"] for r in sr["right"] if r["success"]]
        lre = [r["rot_err_deg"] for r in sr["left"] if r["success"]]
        rre = [r["rot_err_deg"] for r in sr["right"] if r["success"]]
        lje = np.array([r["joint_err_rad"] for r in sr["left"] if r["success"]])
        rje = np.array([r["joint_err_rad"] for r in sr["right"] if r["success"]])

        c = colors[sn]
        lbl = labels[sn]
        axes[0, 0].plot(indices[:len(lpe)], lpe, label=f"L {lbl}", alpha=0.7, color=c)
        axes[0, 0].plot(indices[:len(rpe)], rpe, label=f"R {lbl}", alpha=0.4, color=c, linestyle="--")
        axes[0, 1].plot(indices[:len(lre)], lre, label=f"L {lbl}", alpha=0.7, color=c)
        axes[0, 1].plot(indices[:len(rre)], rre, label=f"R {lbl}", alpha=0.4, color=c, linestyle="--")

        if sn == "prev_step":
            joint_labels = ["j0 pan", "j1 lift", "j2 elbow",
                            "j3 wrist_a", "j4 wrist_r", "j5 grip_r"]
            for j in range(6):
                axes[1, 0].plot(indices[:len(lje)], lje[:, j],
                                label=joint_labels[j], alpha=0.7)
                axes[1, 1].plot(indices[:len(rje)], rje[:, j],
                                label=joint_labels[j], alpha=0.7)

    axes[0, 0].set_xlabel("Frame"); axes[0, 0].set_ylabel("Position error (mm)")
    axes[0, 0].set_title("Position Error (all scenarios)"); axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel("Frame"); axes[0, 1].set_ylabel("Orientation error (deg)")
    axes[0, 1].set_title("Orientation Error (all scenarios)"); axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("Frame"); axes[1, 0].set_ylabel("Joint error (rad)")
    axes[1, 0].set_title("Left Arm Joint Error (prev-step seed)"); axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel("Frame"); axes[1, 1].set_ylabel("Joint error (rad)")
    axes[1, 1].set_title("Right Arm Joint Error (prev-step seed)"); axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("IK Round-Trip Quality (lower = better)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(out_dir / "ik_error_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved error plots -> {out_dir / 'ik_error_plots.png'}")

    # --- Histogram for prev_step scenario ---
    sr = all_scenario_results["prev_step"]
    lpe = [r["pos_err_mm"] for r in sr["left"] if r["success"]]
    rpe = [r["pos_err_mm"] for r in sr["right"] if r["success"]]
    lre = [r["rot_err_deg"] for r in sr["left"] if r["success"]]
    rre = [r["rot_err_deg"] for r in sr["right"] if r["success"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    all_pe = lpe + rpe
    ax.hist(all_pe, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    if all_pe:
        ax.axvline(np.median(all_pe), color="red", linestyle="--",
                   label=f"median={np.median(all_pe):.3f}mm")
    ax.set_xlabel("Position error (mm)"); ax.set_ylabel("Count")
    ax.set_title("Position Error Distribution (prev-step seed)"); ax.legend()

    ax = axes[1]
    all_re = lre + rre
    ax.hist(all_re, bins=50, color="coral", edgecolor="black", alpha=0.8)
    if all_re:
        ax.axvline(np.median(all_re), color="red", linestyle="--",
                   label=f"median={np.median(all_re):.3f}°")
    ax.set_xlabel("Orientation error (deg)"); ax.set_ylabel("Count")
    ax.set_title("Orientation Error Distribution (prev-step seed)"); ax.legend()

    plt.tight_layout()
    plt.savefig(str(out_dir / "ik_error_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram -> {out_dir / 'ik_error_histogram.png'}")

    # --- Video with overlay (prev_step scenario) ---
    if not args.no_video and "images_raw" in data:
        print("Generating overlay video (prev-step scenario)...")
        K = CAM_INTRINSICS_640x480
        sr = all_scenario_results["prev_step"]
        left_results = sr["left"]
        right_results = sr["right"]

        first_img = _decode_jpeg(data["images_raw"][0])
        h, w = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid_path = str(out_dir / "ik_overlay.mp4")
        writer = cv2.VideoWriter(vid_path, fourcc, 30, (w, h))

        for i, t in enumerate(indices):
            img = _decode_jpeg(data["images_raw"][t]).copy()
            lr, rr = left_results[i], right_results[i]

            for arm_label, res, T_cam in [
                ("L", lr, T_CAM_FROM_LEFT),
                ("R", rr, T_CAM_FROM_RIGHT),
            ]:
                if not res["success"]:
                    continue
                pos_orig_cam = pos_to_cam(res["pos_orig"], T_cam)
                px_orig = project_to_pixel(pos_orig_cam, K)

                pos_recov_cam = pos_to_cam(res["pos_recov"], T_cam)
                px_recov = project_to_pixel(pos_recov_cam, K)

                if 0 <= px_orig[0] < w and 0 <= px_orig[1] < h:
                    cv2.circle(img, px_orig, 6, (0, 255, 0), -1)
                    cv2.circle(img, px_orig, 7, (0, 0, 0), 1)
                if 0 <= px_recov[0] < w and 0 <= px_recov[1] < h:
                    cv2.circle(img, px_recov, 4, (0, 100, 255), -1)
                    cv2.circle(img, px_recov, 5, (0, 0, 0), 1)

            l_pe = lr["pos_err_mm"] if lr["success"] else float("nan")
            r_pe = rr["pos_err_mm"] if rr["success"] else float("nan")
            l_re = lr["rot_err_deg"] if lr["success"] else float("nan")
            r_re = rr["rot_err_deg"] if rr["success"] else float("nan")
            texts = [
                f"t={t}  FK->IK->FK (seed=prev step)",
                f"L: pos={l_pe:.2f}mm  rot={l_re:.2f}deg",
                f"R: pos={r_pe:.2f}mm  rot={r_re:.2f}deg",
                "green=original FK  blue=IK recovered",
            ]
            for ti, txt in enumerate(texts):
                cv2.putText(img, txt, (10, 20 + ti * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(img, txt, (10, 20 + ti * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if (i + 1) % 100 == 0 or i == N - 1:
                print(f"  Video frame {i + 1}/{N}")

        writer.release()
        print(f"Saved overlay video -> {vid_path}")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
