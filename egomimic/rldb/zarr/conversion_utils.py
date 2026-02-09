import logging
import os
import time
from pathlib import Path
from typing import Union

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

import egomimic
from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver
from egomimic.rldb.utils import EMBODIMENT
from egomimic.utils.egomimicUtils import (
    ee_orientation_to_cam_frame,
    ee_pose_to_cam_frame,
    interpolate_arr,
    interpolate_arr_euler,
)

DATASET_KEY_MAPPINGS = {
    "joint_positions": "joint_positions",
    "front_img_1": "front_img_1",
    "right_wrist_img": "right_wrist_img",
    "left_wrist_img": "left_wrist_img",
}

EVA_XML_PATH = os.path.join(
    os.path.dirname(egomimic.__file__), "resources/model_x5.xml"
)

HORIZON_BASE = 45   # horizon in real timesteps
CHUNK_LENGTH_BASE = 100  # number of interpolated points per chunk

Array2D = Union[np.ndarray, torch.Tensor]


def _to_numpy(x):
    """Convert torch / numpy / anything array-like to np.ndarray."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_rotation(x):
    """Return a scipy Rotation from either Rotation or 3x3 ndarray."""
    return x if isinstance(x, R) else R.from_matrix(np.asarray(x, dtype=float))


def _as_matrix(x):
    """Return a 3x3 ndarray from either Rotation or 3x3 ndarray."""
    return x.as_matrix() if isinstance(x, R) else np.asarray(x, dtype=float)


def _row_to_numpy(x: Array2D, i: int) -> np.ndarray:
    """Get i-th row as numpy array (handles torch or numpy input)."""
    if isinstance(x, np.ndarray):
        return x[i]
    return x[i].detach().cpu().numpy()


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def assert_fk_matches_eepose(
    actions_base_cartesian: np.ndarray,
    actions_eepose: np.ndarray,
    *,
    arm: str = "both",
    atol_pose: float = 2e-4,
    rtol_pose: float = 0.0,
    mismatch_frac_threshold: float = 0.01,
    skip_first: int = 1,
    zero_eps: float = 1e-12,
):
    # prestack handling
    base0 = (
        actions_base_cartesian[:, 0]
        if actions_base_cartesian.ndim == 3
        else actions_base_cartesian
    )
    eep = actions_eepose

    if eep.ndim != 2 or eep.shape[1] != 14:
        print(f"[FK WARNING] Expected eepose (T,14), got {eep.shape}")
        return

    if base0.ndim != 2:
        print(f"[FK WARNING] Expected base_cartesian 2D, got {base0.shape}")
        return

    if base0.shape[0] != eep.shape[0]:
        print(f"[FK WARNING] T mismatch: base={base0.shape}, eepose={eep.shape}")
        return

    T = eep.shape[0]
    idx_all = np.arange(T)

    # --------------------------------------------------
    # dimension layout
    # --------------------------------------------------
    # base_cartesian:
    #   arm=left/right : (T,7)  -> [x y z yaw pitch roll g]
    #   arm=both       : (T,14) -> [L pose g | R pose g]
    #
    # eepose (always):
    #   [L xyz ypr g | R xyz ypr g]
    # --------------------------------------------------

    if arm == "left":
        b_pos = [0, 1, 2]
        b_ang = [3, 4, 5]
        e_pos = [0, 1, 2]
        e_ang = [3, 4, 5]
        arm_name = "left"

        if base0.shape[1] != 7:
            print(f"[FK WARNING] base_cartesian expected (T,7) for left, got {base0.shape}")
            return

    elif arm == "right":
        b_pos = [0, 1, 2]
        b_ang = [3, 4, 5]
        e_pos = [7, 8, 9]
        e_ang = [10, 11, 12]
        arm_name = "right"

        if base0.shape[1] != 7:
            print(f"[FK WARNING] base_cartesian expected (T,7) for right, got {base0.shape}")
            return

    elif arm == "both":
        b_pos = [0, 1, 2, 7, 8, 9]
        b_ang = [3, 4, 5, 10, 11, 12]
        e_pos = [0, 1, 2, 7, 8, 9]
        e_ang = [3, 4, 5, 10, 11, 12]
        arm_name = "both"

        if base0.shape[1] != 14:
            print(f"[FK WARNING] base_cartesian expected (T,14) for both, got {base0.shape}")
            return

    else:
        print(f"[FK WARNING] Unknown arm='{arm}'")
        return

    # --------------------------------------------------
    # validity mask (eepose only)
    # --------------------------------------------------
    pose_dims_e = e_pos + e_ang
    valid = (np.abs(eep[:, pose_dims_e]) > zero_eps).any(axis=1)
    if skip_first > 0:
        valid[:skip_first] = False

    if not valid.any():
        return

    b = base0[valid]
    a = eep[valid]
    v_idx = idx_all[valid]

    # --------------------------------------------------
    # position diff
    # --------------------------------------------------
    dpos = b[:, b_pos] - a[:, e_pos]
    tolpos = atol_pose + rtol_pose * np.abs(a[:, e_pos])
    viol = (np.abs(dpos) > tolpos).any(axis=1)

    # --------------------------------------------------
    # angle diff (inline wrap)
    # --------------------------------------------------
    dang = b[:, b_ang] - a[:, e_ang]
    dang = (dang + np.pi) % (2 * np.pi) - np.pi
    tolang = atol_pose + rtol_pose * np.abs(a[:, e_ang])
    viol |= (np.abs(dang) > tolang).any(axis=1)

    mismatch_frac = float(viol.mean())
    if mismatch_frac <= mismatch_frac_threshold:
        return

    bad_ts = v_idx[viol]
    show = bad_ts[:20].tolist()
    extra = "" if bad_ts.size <= 20 else f" ... (+{bad_ts.size - 20} more)"

    print(
        f"[FK WARNING] ({arm_name}) FK/eepose mismatch rate "
        f"{mismatch_frac*100:.3f}% "
        f"({int(viol.sum())}/{int(viol.size)} rows). "
        f"First bad timesteps: {show}{extra}. "
        f"(atol={atol_pose}, rtol={rtol_pose}, skip_first={skip_first})"
    )


def print_fk_eepose_diffs(
    actions_base_cartesian: np.ndarray,
    actions_eepose: np.ndarray,
    *,
    atol: float = 2e-4,
    rtol: float = 0.0,
    skip_first: int = 1,
    zero_eps: float = 1e-12,
    max_rows: int = 50,
    max_cols: int = 20,
    with_angles: bool = True,
):
    """
    Print all diffs (per timestep, per dim) where FK(base0) != eepose beyond tolerance.

    - If base is (T,S,D), compares base[:,0,:] against eepose.
    - Drops gripper dims by default via dim selection based on D.
    - For angles, compares wrapped diff into [-pi,pi].
    - Masks out padded all-zero eepose rows in pose dims and also skips first `skip_first`.

    Prints:
      - summary stats
      - list of violating timesteps
      - for each violating timestep: FK, EE, DIFF for selected dims
    """
    base0 = actions_base_cartesian[:, 0] if actions_base_cartesian.ndim == 3 else actions_base_cartesian
    eep = actions_eepose

    if base0.shape != eep.shape:
        raise ValueError(f"Shape mismatch: base0={base0.shape} vs eepose={eep.shape}")

    T, D = base0.shape

    if D == 14:
        pos_dims = [0, 1, 2, 7, 8, 9]
        ang_dims = [3, 4, 5, 10, 11, 12] if with_angles else []
        dim_names = (
            ["Lx","Ly","Lz","L_yaw","L_pitch","L_roll","L_g",
             "Rx","Ry","Rz","R_yaw","R_pitch","R_roll","R_g"]
        )
    elif D == 7:
        pos_dims = [0, 1, 2]
        ang_dims = [3, 4, 5] if with_angles else []
        dim_names = ["x","y","z","yaw","pitch","roll","g"]
    else:
        pos_dims = list(range(0, min(3, D)))
        ang_dims = list(range(3, min(6, D))) if with_angles else []
        dim_names = [f"d{i}" for i in range(D)]

    pose_dims = pos_dims + ang_dims

    # mask padded all-zero-ish rows (in pose dims) + skip_first
    nonzero_row = (np.abs(eep[:, pose_dims]) > zero_eps).any(axis=1)
    mask = nonzero_row.copy()
    if skip_first > 0:
        mask[:skip_first] = False

    idx_all = np.arange(T)
    valid_idx = idx_all[mask]
    if valid_idx.size == 0:
        print(f"[diff] No valid rows after masking. skip_first={skip_first}, zero_eps={zero_eps}")
        return

    b = base0[mask]
    a = eep[mask]

    # build a "diff" array for pose dims: pos direct, angles wrapped
    diff_pose = np.zeros((b.shape[0], len(pose_dims)), dtype=np.float64)
    # positions
    diff_pose[:, :len(pos_dims)] = (b[:, pos_dims] - a[:, pos_dims])

    # angles
    if ang_dims:
        diff_pose[:, len(pos_dims):] = _wrap_to_pi(b[:, ang_dims] - a[:, ang_dims])

    # tolerance per-element
    # elementwise allowed = atol + rtol * abs(desired)
    desired_pose = a[:, pose_dims]
    tol = atol + rtol * np.abs(desired_pose)
    viol = np.abs(diff_pose) > tol

    viol_rows = np.where(viol.any(axis=1))[0]
    if viol_rows.size == 0:
        print(f"[diff] ✅ No violations. compared_rows={b.shape[0]} masked_out={(~mask).sum()}")
        return

    # summary
    abs_diff = np.abs(diff_pose)
    print(f"[diff] compared_rows={b.shape[0]} masked_out={(~mask).sum()}  atol={atol} rtol={rtol}")
    print(f"[diff] violations: {viol_rows.size} / {b.shape[0]} rows ({100*viol_rows.size/b.shape[0]:.3f}%)")
    print(f"[diff] max_abs_over_pose_dims={abs_diff[viol].max():.6g}")
    # per-dim summary
    mean_abs_per_dim = abs_diff.mean(axis=0)
    max_abs_per_dim = abs_diff.max(axis=0)
    dim_labels = [dim_names[d] if d < len(dim_names) else str(d) for d in pose_dims]
    print("[diff] per-dim mean_abs:", dict(zip(dim_labels, mean_abs_per_dim)))
    print("[diff] per-dim max_abs :", dict(zip(dim_labels, max_abs_per_dim)))

    # print violating timesteps + detailed diffs
    print("\n[diff] Violating timesteps (original t indices):")
    ts = valid_idx[viol_rows]
    print(ts.tolist()[:max_rows], ("..." if ts.size > max_rows else ""))

    to_show = min(max_rows, viol_rows.size)
    for k in range(to_show):
        r = viol_rows[k]
        t = valid_idx[r]  # original timestep
        cols = np.where(viol[r])[0]
        cols = cols[:max_cols]

        fk_vals = b[r, pose_dims]
        ee_vals = a[r, pose_dims]
        dvals  = diff_pose[r]

        print(f"\n--- t={t} (valid_row_idx={r}) ---")
        for c in cols:
            name = dim_labels[c]
            fk_v = fk_vals[c]
            ee_v = ee_vals[c]
            dv   = dvals[c]
            allowed = tol[r, c]
            print(f"{name:>8s}: fk={fk_v:+.6f}  ee={ee_v:+.6f}  diff={dv:+.6f}  |diff|>{allowed:.6g}")

    if viol_rows.size > max_rows:
        print(f"\n[diff] (only first {max_rows} violating rows printed; set max_rows larger to see all)")

    return {
        "valid_idx": valid_idx,
        "viol_valid_rows": viol_rows,
        "viol_timesteps": valid_idx[viol_rows],
        "pose_dims": pose_dims,
        "pos_dims": pos_dims,
        "ang_dims": ang_dims,
        "diff_pose": diff_pose,
        "viol_mask": viol,
    }


def fk_xyz(
    joints_2d: Array2D, eva_fk: EvaMinkKinematicsSolver, *, dtype=torch.float32, device=None
) -> torch.Tensor:
    """
    Eva FK positions for a sequence of joint vectors.
    Returns torch.Tensor of shape (T, 3).
    """
    T = joints_2d.shape[0]
    out = torch.empty((T, 3), dtype=dtype, device=device)
    for i in range(T):
        q = _row_to_numpy(joints_2d, i)  # (DoF,)
        pos, _R = eva_fk.fk(q)  # pos: (3,), _R: (3,3) numpy
        out[i] = torch.as_tensor(pos, dtype=dtype, device=device)
    return out


def fk_SE3(
    joints_2d: Array2D, eva_fk: EvaMinkKinematicsSolver, *, dtype=torch.float32, device=None
) -> torch.Tensor:
    """
    Eva FK full SE(3) for a sequence of joint vectors.
    Returns torch.Tensor of shape (T, 4, 4) with bottom-right set to 1.
    Downstream can keep using fk[:, :3, 3] and fk[:, :3, :3].
    """
    T = joints_2d.shape[0]
    out = torch.zeros((T, 4, 4), dtype=dtype, device=device)
    out[:, 3, 3] = 1.0
    for i in range(T):
        q = _row_to_numpy(joints_2d, i)
        pos, R_obj = eva_fk.fk(q)  # numpy: (3,), (3,3)
        Rm = _as_matrix(R_obj)
        out[i, :3, :3] = torch.as_tensor(Rm, dtype=dtype, device=device)
        out[i, :3, 3] = torch.as_tensor(pos, dtype=dtype, device=device)
    return out


def build_future_windows(arr: np.ndarray, horizon: int) -> np.ndarray:
    arr = np.asarray(arr)
    T, D = arr.shape
    out = np.empty((T, horizon, D), dtype=arr.dtype)
    for t in range(T):
        end = min(t + horizon, T)
        length = end - t
        out[t, :length, :] = arr[t:end]
        out[t, length:, :] = arr[end - 1]
    return out


def prestack_with_mode(
    arr: np.ndarray,
    horizon: int,
    chunk_length: int,
    mode: str = "linear",
) -> np.ndarray:
    windows = build_future_windows(arr, horizon)
    if mode == "euler":
        return interpolate_arr_euler(windows, chunk_length)
    else:
        return interpolate_arr(windows, chunk_length)


def joint_to_pose(pose, arm, left_extrinsics=None, right_extrinsics=None, no_rot=False):
    """
    pose: (T, ACTION_DIM)
    arm: left, right, both arms of the robot
    left_extrinsics: the left camera extrinsics
    right_extrinsics: the right camera extrinsics
    """
    eva_fk = EvaMinkKinematicsSolver(model_path=str(EVA_XML_PATH))

    if arm == "both":
        joint_start = 0
        joint_end = 14
        joint_left_start = 0
        joint_left_end = 7
        joint_right_start = 7
        joint_right_end = 14
    else:
        if arm == "left":
            joint_start = 0
            joint_end = 7
        elif arm == "right":
            joint_start = 7
            joint_end = 14

    if no_rot:
        if arm == "both":
            fk_left_positions = fk_xyz(
                pose[:, joint_left_start : joint_left_end - 1], eva_fk
            )
            fk_right_positions = fk_xyz(
                pose[:, joint_right_start : joint_right_end - 1], eva_fk
            )

            fk_left_positions = _to_numpy(fk_left_positions)
            fk_right_positions = _to_numpy(fk_right_positions)

            base_fk_positions = np.concatenate([fk_left_positions, fk_right_positions], axis=1)

            fk_left_positions = ee_pose_to_cam_frame(fk_left_positions, left_extrinsics)
            fk_right_positions = ee_pose_to_cam_frame(
                fk_right_positions, right_extrinsics
            )
            fk_positions = np.concatenate(
                [fk_left_positions, fk_right_positions], axis=1
            )
        else:
            fk_positions = fk_xyz(pose[:, joint_start : joint_end - 1], eva_fk)
            extrinsics = left_extrinsics if arm == "left" else right_extrinsics

            base_fk_positions = fk_positions

            fk_positions = ee_pose_to_cam_frame(fk_positions, extrinsics)

    else:
        if arm == "both":
            fk_left = fk_SE3(pose[:, joint_left_start : joint_left_end - 1], eva_fk)
            fk_right = fk_SE3(pose[:, joint_right_start : joint_right_end - 1], eva_fk)

            fk_left_positions = fk_left[:, :3, 3]
            fk_left_orientations = fk_left[:, :3, :3]
            fk_right_positions = fk_right[:, :3, 3]
            fk_right_orientations = fk_right[:, :3, :3]

            left_gripper = pose[:, joint_left_end - 1].reshape(-1, 1)
            right_gripper = pose[:, joint_right_end - 1].reshape(-1, 1)


            left_ypr_base = R.from_matrix(fk_left_orientations).as_euler("ZYX", degrees=False)
            right_ypr_base = R.from_matrix(fk_right_orientations).as_euler("ZYX", degrees=False)

            base_fk_positions = np.concatenate(
                [
                    fk_left_positions,
                    left_ypr_base,
                    left_gripper,
                    fk_right_positions,
                    right_ypr_base,
                    right_gripper,
                ],
                axis=1,
            )

            fk_left_positions = ee_pose_to_cam_frame(fk_left_positions, left_extrinsics)
            fk_right_positions = ee_pose_to_cam_frame(
                fk_right_positions, right_extrinsics
            )

            fk_left_orientations, fk_left_ypr = ee_orientation_to_cam_frame(
                fk_left_orientations, left_extrinsics
            )
            fk_right_orientations, fk_right_ypr = ee_orientation_to_cam_frame(
                fk_right_orientations, right_extrinsics
            )

            fk_positions = np.concatenate(
                [
                    fk_left_positions,
                    fk_left_ypr,
                    left_gripper,
                    fk_right_positions,
                    fk_right_ypr,
                    right_gripper,
                ],
                axis=1,
            )

        else:
            fk = fk_SE3(pose[:, joint_start : joint_end - 1], eva_fk)

            fk_positions = fk[:, :3, 3]
            fk_orientations = fk[:, :3, :3]

            gripper = pose[:, joint_end - 1].reshape(-1, 1)

            ypr_base = R.from_matrix(fk_orientations).as_euler("ZYX", degrees=False)
            base_fk_positions = np.concatenate([fk_positions, ypr_base, gripper], axis=1)

            extrinsics = left_extrinsics if arm == "left" else right_extrinsics
            fk_positions = ee_pose_to_cam_frame(fk_positions, extrinsics)
            fk_orientations, fk_ypr = ee_orientation_to_cam_frame(
                fk_orientations, extrinsics
            )

            fk_positions = np.concatenate([fk_positions, fk_ypr, gripper], axis=1)

    return fk_positions, base_fk_positions


class EvaHD5Extractor:
    TAGS = ["eva", "robotics", "hdf5"]

    @staticmethod
    def process_episode(
        episode_path, arm, extrinsics, prestack=False, low_res=False, no_rot=False
    ):
        """
        Extracts all feature keys from a given episode and returns as a dictionary
        Parameters
        ----------
        episode_path : str or Path
            Path to the HDF5 file containing the episode data.
        arm : str
            String for which arm to add data for
        extrinsics : np.array
            camera extrinsic, It is a tuple of (left_extrinsics, right_extrinsics) if arm is both
        prestack : bool
            prestack the future actions or not
        Returns
        -------
        episode_feats : dict
            dictionary mapping keys in the episode to episode features
            {
                {action_key} :
                observations :
                    images.{camera_key} :
                    state.{state_key} :
            }

            #TODO: Add metadata to be a nested dict

        """
        left_extrinsics = None
        right_extrinsics = None

        if arm == "both":
            if not isinstance(extrinsics, dict):
                logging.info(
                    "Error: Both arms selected. Expected extrinsics for both arms."
                )
            left_extrinsics = extrinsics["left"]
            right_extrinsics = extrinsics["right"]
        elif arm == "left":
            extrinsics = extrinsics["left"]
            left_extrinsics = extrinsics
        elif arm == "right":
            extrinsics = extrinsics["right"]
            right_extrinsics = extrinsics

        if arm == "left":
            joint_start = 0
            joint_end = 7
        elif arm == "right":
            joint_start = 7
            joint_end = 14
        elif arm == "both":
            joint_start = 0
            joint_end = 14

        episode_feats = dict()

        # TODO: benchmarking only, remove for release
        t0 = time.time()

        with h5py.File(episode_path, "r") as episode:
            # rgb camera
            logging.info(f"[EvaHD5Extractor] Reading HDF5 file: {episode_path}")

            episode_feats["observations"] = dict()

            for camera in EvaHD5Extractor.get_cameras(episode):
                images = (
                    torch.from_numpy(episode["observations"]["images"][camera][:])
                    .permute(0, 3, 1, 2)
                    .float()
                )

                if low_res:
                    images = F.interpolate(
                        images, size=(240, 320), mode="bilinear", align_corners=False
                    )

                images = images.byte().numpy()

                mapped_key = DATASET_KEY_MAPPINGS.get(camera, camera)
                episode_feats["observations"][f"images.{mapped_key}"] = images

            # state
            for state in EvaHD5Extractor.get_state(episode):
                mapped_key = DATASET_KEY_MAPPINGS.get(state, state)
                episode_feats["observations"][f"state.{mapped_key}"] = episode[
                    "observations"
                ][state][:]

            # ee_pose
            episode_feats["observations"][f"state.ee_pose"] = (
                EvaHD5Extractor.get_ee_pose(
                    episode_feats["observations"][f"state.joint_positions"],
                    arm,
                    left_extrinsics=left_extrinsics,
                    right_extrinsics=right_extrinsics,
                    no_rot=no_rot,
                )
            )

            # actions
            joint_actions, cartesian_actions, base_cartesian_actions = EvaHD5Extractor.get_action(
                episode["action"][:],
                arm=arm,
                prestack=prestack,
                HORIZON=HORIZON_BASE,
                CHUNK_LENGTH=CHUNK_LENGTH_BASE,
                left_extrinsics=left_extrinsics,
                right_extrinsics=right_extrinsics,
                no_rot=no_rot,
            )
            # dbg = print_fk_eepose_diffs(
            #     base_cartesian_actions,
            #     episode["actions"]["eepose"][:],
            #     atol=2e-4,
            #     rtol=0.0,
            #     skip_first=1,
            #     max_rows=500,   # set huge if you truly want "all"
            # )
            if "actions" in episode and "eepose" in episode["actions"]:
                assert_fk_matches_eepose(base_cartesian_actions, episode["actions"]["eepose"][:], arm=arm)

            episode_feats["actions_joints"] = joint_actions
            episode_feats["actions_cartesian"] = cartesian_actions
            episode_feats["actions_base_cartesian"] = base_cartesian_actions

            episode_feats["actions_eef_cartesian"] = EvaHD5Extractor.get_eef_action(
                actions_cartesian_base=base_cartesian_actions,
                arm=arm,
                ref_index=0,
            )

            episode_feats["observations"][f"state.joint_positions"] = episode_feats[
                "observations"
            ][f"state.joint_positions"][:, joint_start:joint_end]

            num_timesteps = episode_feats["observations"][f"state.ee_pose"].shape[0]
            if arm == "right":
                value = EMBODIMENT.EVA_RIGHT_ARM.value
            elif arm == "left":
                value = EMBODIMENT.EVA_LEFT_ARM.value
            else:
                value = EMBODIMENT.EVA_BIMANUAL.value

            episode_feats["metadata.embodiment"] = np.full(
                (num_timesteps, 1), value, dtype=np.int32
            )

        # TODO: benchmarking only, remove for release
        elapsed_time = time.time() - t0
        logging.info(
            f"[EvaHD5Extractor] Finished processing episode at {episode_path} in {elapsed_time:.2f} sec"
        )

        return episode_feats

    @staticmethod
    def get_action(
        actions: np.array,
        arm: str,
        prestack: bool = False,
        HORIZON: int = HORIZON_BASE,
        CHUNK_LENGTH: int = CHUNK_LENGTH_BASE,
        left_extrinsics=None,
        right_extrinsics=None,
        no_rot: bool = False,
    ):
        joint_actions = actions

        if arm == "left":
            joint_start = 0
            joint_end = 7
        elif arm == "right":
            joint_start = 7
            joint_end = 14
        elif arm == "both":
            joint_start = 0
            joint_end = 14

        cartesian_actions, base_cartesian_actions = joint_to_pose(
            pose=joint_actions,
            arm=arm,
            left_extrinsics=left_extrinsics,
            right_extrinsics=right_extrinsics,
            no_rot=no_rot,
        )

        joint_actions = joint_actions[:, joint_start:joint_end]

        if prestack:
            horizon = HORIZON

            joint_actions = prestack_with_mode(
                np.asarray(joint_actions),
                horizon=horizon,
                chunk_length=CHUNK_LENGTH,
                mode="linear",
            )

            cart_np = np.asarray(cartesian_actions)
            base_np = np.asarray(base_cartesian_actions)

            if arm == "both":
                left_base = base_np[:, :7]
                right_base = base_np[:, 7:14]
                left_cam = cart_np[:, :7]
                right_cam = cart_np[:, 7:14]

                left_base = prestack_with_mode(
                    left_base,
                    horizon=horizon,
                    chunk_length=CHUNK_LENGTH,
                    mode="euler",
                )
                right_base = prestack_with_mode(
                    right_base,
                    horizon=horizon,
                    chunk_length=CHUNK_LENGTH,
                    mode="euler",
                )
                left_cam = prestack_with_mode(
                    left_cam,
                    horizon=horizon,
                    chunk_length=CHUNK_LENGTH,
                    mode="euler",
                )
                right_cam = prestack_with_mode(
                    right_cam,
                    horizon=horizon,
                    chunk_length=CHUNK_LENGTH,
                    mode="euler",
                )

                base_cartesian_actions = np.concatenate(
                    [left_base, right_base], axis=-1
                )
                cartesian_actions = np.concatenate(
                    [left_cam, right_cam], axis=-1
                )
            else:
                base_cartesian_actions = prestack_with_mode(
                    base_np,
                    horizon=horizon,
                    chunk_length=CHUNK_LENGTH,
                    mode="euler",
                )
                cartesian_actions = prestack_with_mode(
                    cart_np,
                    horizon=horizon,
                    chunk_length=CHUNK_LENGTH,
                    mode="euler",
                )

        return (joint_actions, cartesian_actions, base_cartesian_actions)

    @staticmethod
    def get_eef_action(
        actions_cartesian_base: np.ndarray,
        arm: str,
        ref_index: int = 0,
    ) -> np.ndarray:
        """Compute relative EEF actions from base-frame Cartesian actions.

        Supports both prestacked (N, S, D) and non-prestacked (N, D) inputs.
        For prestacked: each row has S future actions, relative to ref_index action.
        For non-prestacked: computes relative pose from previous frame (or identity for first).

        Args:
            actions_cartesian_base: Shape (N, S, D) or (N, D) where D=7 per arm (xyz + ypr + gripper)
            arm: "left", "right", or "both"
            ref_index: Reference index for relative transform (only used for prestacked)

        Returns:
            Relative actions with same shape as input
        """
        if arm == "both":
            left_base = actions_cartesian_base[..., :7]
            right_base = actions_cartesian_base[..., 7:14]

            left_rel = EvaHD5Extractor.get_eef_action(
                left_base, arm="left", ref_index=ref_index
            )
            right_rel = EvaHD5Extractor.get_eef_action(
                right_base, arm="right", ref_index=ref_index
            )
            return np.concatenate([left_rel, right_rel], axis=-1)

        # Handle both 2D (N, D) and 3D (N, S, D) inputs
        is_2d = actions_cartesian_base.ndim == 2
        if is_2d:
            # Add sequence dimension: (N, D) -> (N, 1, D)
            actions_cartesian_base = actions_cartesian_base[:, np.newaxis, :]

        N, S, D = actions_cartesian_base.shape

        p = actions_cartesian_base[..., :3]     # (N, S, 3)
        ypr = actions_cartesian_base[..., 3:6]  # (N, S, 3)
        g = actions_cartesian_base[..., 6:7]    # (N, S, 1)

        ypr_flat = ypr.reshape(-1, 3)               # (N*S, 3)
        R_flat = R.from_euler("ZYX", ypr_flat, degrees=False).as_matrix()  # (N*S, 3, 3)
        R_seq = R_flat.reshape(N, S, 3, 3)          # (N, S, 3, 3)

        T_seq = np.zeros((N, S, 4, 4), dtype=np.float32)
        T_seq[..., :3, :3] = R_seq
        T_seq[..., :3, 3] = p
        T_seq[..., 3, 3] = 1.0

        T0 = T_seq[:, ref_index, :, :]          # (N, 4, 4)
        T0_inv = np.linalg.inv(T0)              # (N, 4, 4)

        T_rel = T0_inv[:, None, :, :] @ T_seq   # (N, S, 4, 4)

        p_rel = T_rel[..., :3, 3]          # (N, S, 3)
        R_rel = T_rel[..., :3, :3]         # (N, S, 3, 3)

        R_rel_flat = R_rel.reshape(-1, 3, 3)    # (N*S, 3, 3)
        ypr_rel_flat = R.from_matrix(R_rel_flat).as_euler("ZYX", degrees=False)  # (N*S, 3)
        ypr_rel = ypr_rel_flat.reshape(N, S, 3) # (N, S, 3)

        actions_rel = np.empty_like(actions_cartesian_base)
        actions_rel[..., :3] = p_rel
        actions_rel[..., 3:6] = ypr_rel
        actions_rel[..., 6:7] = g  # keep gripper as-is

        # Remove sequence dimension if input was 2D
        if is_2d:
            actions_rel = actions_rel[:, 0, :]  # (N, 1, D) -> (N, D)

        return actions_rel

    @staticmethod
    def get_ee_pose(
        qpos: np.array,
        arm: str,
        left_extrinsics=None,
        right_extrinsics=None,
        no_rot=False,
    ):
        """
        Uses FK to calculate ee pose from joints
        Parameters
        ----------
        qpos : np.array
            array containing joint positions
        arm : str
            arm to convert data for
        left_extrinsics :
            camera extrinsics
        right_extrinsics :
            camera_extrinsics
        no_rot : bool
            calculate full 6dof pose or not
        Returns
        -------
        ee_pose : np.array
            ee_pose SE{3}
        """

        ee_pose, _ = joint_to_pose(
            qpos, arm, left_extrinsics, right_extrinsics, no_rot=no_rot
        )

        return ee_pose

    @staticmethod
    def get_cameras(hdf5_data: h5py.File):
        """
        Extracts the list of RGB camera keys from the given HDF5 data.
        Parameters
        ----------
        hdf5_data : h5py.File
            The HDF5 file object containing the dataset.
        Returns
        -------
        list of str
            A list of keys corresponding to RGB cameras in the dataset.
        """

        rgb_cameras = [
            key for key in hdf5_data["/observations/images"] if "depth" not in key
        ]
        return rgb_cameras

    @staticmethod
    def get_state(hdf5_data: h5py.File):
        """
        Extracts the list of RGB camera keys from the given HDF5 data.
        Parameters
        ----------
        hdf5_data : h5py.File
            The HDF5 file object containing the dataset.
        Returns
        -------
        states : list of str
            A list of keys corresponding to states in the dataset.
        """

        states = [key for key in hdf5_data["/observations"] if "images" not in key]
        return states

    @staticmethod
    def check_format(
        episode_list: list[str] | list[Path], image_compressed: bool = True
    ):
        """
        Check the format of the given list of HDF5 files.
        Parameters
        ----------
        episode_list : list of str or list of Path
            List of paths to the HDF5 files to be checked.
        image_compressed : bool, optional
            Flag indicating whether the images are compressed (default is True).
        Raises
        ------
        ValueError
            If the episode_list is empty.
            If any HDF5 file is missing required keys '/action' or '/observations/joint_positions'.
            If the '/action' or '/observations/joint_positions' keys do not have 2 dimensions.
            If the number of frames in '/action' and '/observations/joint_positions' keys do not match.
            If the number of frames in '/observations/images/{camera}' does not match the number of frames in '/action' and '/observations/joint_positions'.
            If the dimensions of images do not match the expected dimensions based on the image_compressed flag.
            If uncompressed images do not have the expected (h, w, c) format.
        """

        if not episode_list:
            raise ValueError(
                "No hdf5 files found in the raw directory. Make sure they are named '*.hdf5'"
            )
        for episode_path in episode_list:
            with h5py.File(episode_path, "r") as data:
                # Check for required keys - h5py requires checking without leading slash or using get()
                if "action" not in data or "observations" not in data or "joint_positions" not in data["observations"]:
                    raise ValueError(
                        "Missing required keys in the hdf5 file. Make sure the keys '/action' and '/observations/joint_positions' are present."
                    )

                if (
                    not data["/action"].ndim
                    == data["/observations/joint_positions"].ndim
                    == 2
                ):
                    raise ValueError(
                        "The '/action' and '/observations/joint_positions' keys should have both 2 dimensions."
                    )

                if (num_frames := data["/action"].shape[0]) != data[
                    "/observations/joint_positions"
                ].shape[0]:
                    raise ValueError(
                        "The '/action' and '/observations/joint_positions' keys should have the same number of frames."
                    )

                for camera in EvaHD5Extractor.get_cameras(data):
                    if num_frames != data[f"/observations/images/{camera}"].shape[0]:
                        raise ValueError(
                            f"The number of frames in '/observations/images/{camera}' should be the same as in '/action' and '/observations/joint_positions' keys."
                        )

                    expected_dims = 2 if image_compressed else 4
                    if data[f"/observations/images/{camera}"].ndim != expected_dims:
                        raise ValueError(
                            f"Expect {expected_dims} dimensions for {'compressed' if image_compressed else 'uncompressed'} images but {data[f'/observations/images/{camera}'].ndim} provided."
                        )
                    if not image_compressed:
                        b, h, w, c = data[f"/observations/images/{camera}"].shape
                        if not c < h and c < w:
                            raise ValueError(
                                f"Expect (h,w,c) image format but ({h=},{w=},{c=}) provided."
                            )

    @staticmethod
    def extract_episode_frames(
        episode_path: str | Path,
        features: dict[str, dict],
        image_compressed: bool,
        arm: str,
        extrinsics: dict,
        prestack: bool = False,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Extract frames from an episode by processing it and using the feature dictionary.

        Parameters
        ----------
        episode_path : str or Path
            Path to the HDF5 file containing the episode data.
        features : dict of str to dict
            Dictionary where keys are feature identifiers and values are dictionaries with feature details.
        image_compressed : bool
            Flag indicating whether the images are stored in a compressed format.
        arm : str
            The arm to process (e.g., 'left', 'right', or 'both').
        extrinsics : dict
            Camera extrinsics for the episode.
        prestack : bool, optional
            Whether to precompute action chunks, by default False.

        Returns
        -------
        list[dict[str, torch.Tensor]]
            List of frames, where each frame is a dictionary mapping feature identifiers to tensors.
        """
        frames = []
        episode_feats = EvaHD5Extractor.process_episode(
            episode_path, arm=arm, extrinsics=extrinsics, prestack=prestack
        )
        num_frames = next(iter(episode_feats["observations"].values())).shape[0]
        for frame_idx in range(num_frames):
            frame = {}
            for feature_id, feature_info in features.items():
                if "observations" in feature_id:
                    value = episode_feats["observations"][feature_id.split(".", 1)[-1]]
                else:
                    value = episode_feats.get(feature_id, None)
                if value is None:
                    break
                if value is not None:
                    if isinstance(value, np.ndarray):
                        if "images" in feature_id and image_compressed:
                            decompressed_image = cv2.imdecode(value[frame_idx], 1)
                            frame[feature_id] = torch.from_numpy(
                                decompressed_image.transpose(2, 0, 1)
                            )
                        else:
                            frame[feature_id] = torch.from_numpy(value[frame_idx])
                    elif isinstance(value, torch.Tensor):
                        frame[feature_id] = value[frame_idx]
                    else:
                        logging.warning(
                            f"[EvaHD5Extractor] Could not add dataset key at {feature_id} due to unsupported type. Skipping ..."
                        )
                        continue

            frames.append(frame)
        return frames

    @staticmethod
    def define_features(
        episode_feats: dict, image_compressed: bool = True, encode_as_video: bool = True
    ) -> tuple:
        """
        Define features from episode_feats (output of process_episode), including a metadata section.

        Parameters
        ----------
        episode_feats : dict
            The output of the process_episode method, containing feature data.
        image_compressed : bool, optional
            Whether the images are compressed, by default True.
        encode_as_video : bool, optional
            Whether to encode images as video or as images, by default True.

        Returns
        -------
        tuple of dict[str, dict]
            A dictionary where keys are feature names and values are dictionaries
            containing feature information such as dtype, shape, and dimension names,
            and a separate dictionary for metadata (unused for now)
        """
        features = {}
        metadata = {}

        for key, value in episode_feats.items():
            if isinstance(value, dict):  # Handle nested dictionaries recursively
                nested_features, nested_metadata = EvaHD5Extractor.define_features(
                    value, image_compressed, encode_as_video
                )
                features.update(
                    {
                        f"{key}.{nested_key}": nested_value
                        for nested_key, nested_value in nested_features.items()
                    }
                )
                features.update(
                    {
                        f"{key}.{nested_key}": nested_value
                        for nested_key, nested_value in nested_metadata.items()
                    }
                )
            elif isinstance(value, np.ndarray):
                dtype = str(value.dtype)
                if "images" in key:
                    dtype = "video" if encode_as_video else "image"
                    if image_compressed:
                        decompressed_sample = cv2.imdecode(value[0], 1)
                        shape = (
                            decompressed_sample.shape[1],
                            decompressed_sample.shape[0],
                            decompressed_sample.shape[2],
                        )
                    else:
                        shape = value.shape[1:]  # Skip the frame count dimension
                    dim_names = ["channel", "height", "width"]
                elif "actions" in key and len(value[0].shape) > 1:
                    shape = value[0].shape
                    dim_names = ["chunk_length", "action_dim"]
                    dtype = f"prestacked_{str(value.dtype)}"
                else:
                    shape = value[0].shape
                    dim_names = [f"dim_{i}" for i in range(len(shape))]
                features[key] = {
                    "dtype": dtype,
                    "shape": shape,
                    "names": dim_names,
                }
            elif isinstance(value, torch.Tensor):
                dtype = str(value.dtype)
                shape = tuple(value[0].size())
                if "actions" in key and len(tuple(value[0].size())) > 1:
                    dim_names = ["chunk_length", "action_dim"]
                    dtype = f"prestacked_{str(value.dtype)}"
                else:
                    dim_names = [f"dim_{i}" for i in range(len(shape))]
                dim_names = [f"dim_{i}" for i in range(len(shape))]
                features[key] = {
                    "dtype": dtype,
                    "shape": shape,
                    "names": dim_names,
                }
            else:
                metadata[key] = {
                    "dtype": "metadata",
                    "value": value,
                }

        return features, metadata