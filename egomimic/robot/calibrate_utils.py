#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# Make sure we can import oculus_reader from repo root:
# egomimic/robot/calibrate_utils.py  ->  ../../oculus_reader
sys.path.append(
  os.path.join(os.path.dirname(__file__), "..", "..", "oculus_reader")
)
from oculus_reader import OculusReader  # type: ignore


def safe_rot3_from_T(T, ortho_tol=1e-3, det_tol=1e-3):
  Rm = np.asarray(T, dtype=float)[:3, :3]
  if Rm.shape != (3, 3) or not np.all(np.isfinite(Rm)):
    return np.eye(3)
  det = np.linalg.det(Rm)
  if det <= 0 or abs(det - 1.0) > det_tol:
    return np.eye(3)
  if np.linalg.norm(Rm.T @ Rm - np.eye(3), ord="fro") > ortho_tol:
    return np.eye(3)
  return Rm


def quat_xyzw_to_wxyz(qxyzw: np.ndarray) -> np.ndarray:
  return np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(qwxyz: np.ndarray) -> np.ndarray:
  return np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]], dtype=np.float64)


def pose_from_T(T: np.ndarray):
  pos = T[:3, 3].astype(np.float64)
  rot_mat = safe_rot3_from_T(T)
  q_xyzw = R.from_matrix(rot_mat).as_quat()
  q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
  return pos, q_wxyz


def controller_to_internal(pos_xyz: np.ndarray, q_wxyz: np.ndarray):
  """
  Same mapping as in your teleop script.
  """
  A = np.array(
    [[0.0, 0.0, -1.0],
     [0.0, 1.0,  0.0],
     [-1.0, 0.0, 0.0]], dtype=np.float64
  )
  B = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 1.0, 0.0]], dtype=np.float64
  )
  M = B @ A

  R_c = R.from_quat(quat_wxyz_to_xyzw(q_wxyz)).as_matrix()
  pos_i = M @ pos_xyz
  R_i = M @ R_c @ M.T
  q_i = R.from_matrix(R_i).as_quat()
  return pos_i, q_i


def _read_internal_rotation(oculus: OculusReader, hand: str):
  """
  hand: 'r' for right, 'l' for left
  """
  sample = oculus.get_transformations_and_buttons()
  if not sample:
    return None
  transforms, buttons = sample
  if not transforms:
    return None
  T_h = transforms.get(hand, None)
  if T_h is None:
    return None

  pos_raw, quat_wxyz = pose_from_T(np.asarray(T_h))
  _, quat_int_xyzw = controller_to_internal(pos_raw, quat_wxyz)
  R_int = R.from_quat(quat_int_xyzw).as_matrix()
  return R_int


def _collect_mean_rotation(
  dev: OculusReader,
  hand: str,
  msg: str,
  duration: float = 1.0,
):
  print("\n" + msg)
  input("Press ENTER when stable to start sampling...")
  rots = []
  t0 = time.time()
  while time.time() - t0 < duration:
    R_int = _read_internal_rotation(dev, hand)
    if R_int is not None:
      rots.append(R_int)
    time.sleep(0.01)

  if len(rots) == 0:
    print("ERROR: no samples collected.")
    sys.exit(1)

  R_mean = R.from_matrix(np.stack(rots, axis=0)).mean().as_matrix()
  return R_mean


def calibrate_controller(
  hand: str = "r",
  duration: float = 1.0,
  dev: OculusReader | None = None,
  verbose: bool = True,
) -> np.ndarray:
  """
  Calibrate a single controller (right='r' or left='l').
  Returns a 3x3 offset rotation R_off to be used as:

      R_new = R_off @ R_cur

  in your teleop pipeline.
  """
  assert hand in ("r", "l")

  own_dev = dev is None
  if own_dev:
    print(f"Initializing OculusReader ({'right' if hand == 'r' else 'left'} controller)...")
    dev = OculusReader()
    print("Ready.")

  label = "RIGHT" if hand == "r" else "LEFT"

  print(f"\nCalibration procedure ({label} controller):")
  print("  1) Choose a comfortable NEUTRAL pose.")
  print("  2) From that neutral pose, do:")
  print("     - +90 deg PITCH (tip up).")
  print("     - +90 deg YAW (turn about vertical).")
  print("     - +90 deg ROLL (twist clockwise looking along +X).")
  print("  For each pose, you will press ENTER and the script will sample for ~1s.")

  # Neutral orientation
  R_neutral = _collect_mean_rotation(
    dev,
    hand,
    f"Step 1: Hold the {label} controller in your NEUTRAL orientation.",
    duration=duration,
  )

  # +90° pitch (up)
  R_pitch = _collect_mean_rotation(
    dev,
    hand,
    f"Step 2: From the same neutral, rotate +90° PITCH (tip the {label} controller UP) and hold.",
    duration=duration,
  )

  # +90° yaw
  R_yaw = _collect_mean_rotation(
    dev,
    hand,
    f"Step 3: From neutral again, rotate +90° YAW and hold.",
    duration=duration,
  )

  # +90° roll
  R_roll = _collect_mean_rotation(
    dev,
    hand,
    f"Step 4: From neutral again, rotate +90° ROLL CLOCKWISE (twist around forward axis) and hold.",
    duration=duration,
  )

  # We want R_off * R_neutral = I  =>  R_off = R_neutral^T
  R_off = R_neutral.T
  ypr_off = R.from_matrix(R_off).as_euler("ZYX", degrees=False)

  # Sanity checks (optional)
  R_base_neutral = R_off @ R_neutral
  R_base_pitch = R_off @ R_pitch
  R_base_yaw = R_off @ R_yaw
  R_base_roll = R_off @ R_roll

  ypr_neutral = R.from_matrix(R_base_neutral).as_euler("ZYX", degrees=False)
  ypr_pitch = R.from_matrix(R_base_pitch).as_euler("ZYX", degrees=False)
  ypr_yaw = R.from_matrix(R_base_yaw).as_euler("ZYX", degrees=False)
  ypr_roll = R.from_matrix(R_base_roll).as_euler("ZYX", degrees=False)

  if verbose:
    print(f"\n==== Calibration results ({label}) ====")
    print("Neutral (after offset, should be ~[0, 0, 0] in ZYX):")
    print(f"  yaw_pitch_roll_neutral = [{ypr_neutral[0]}, {ypr_neutral[1]}, {ypr_neutral[2]}]")

    print("\n+90° PITCH (expected ~[0, +pi/2, 0] in ZYX):")
    print(f"  yaw_pitch_roll_pitch = [{ypr_pitch[0]}, {ypr_pitch[1]}, {ypr_pitch[2]}]")

    print("\n+90° YAW (expected ~[±pi/2, 0, 0] in ZYX):")
    print(f"  yaw_pitch_roll_yaw   = [{ypr_yaw[0]}, {ypr_yaw[1]}, {ypr_yaw[2]}]")

    print("\n+90° ROLL (expected ~[0, 0, +pi/2] in ZYX):")
    print(f"  yaw_pitch_roll_roll  = [{ypr_roll[0]}, {ypr_roll[1]}, {ypr_roll[2]}]")

    print("\nComputed offset matrix R_off (= R_neutral^T):")
    print(R_off)

    print("\nEuler (ZYX yaw, pitch, roll in radians):")
    print(f"  YPR_OFFSET = [{ypr_off[0]}, {ypr_off[1]}, {ypr_off[2]}]\n")

  if own_dev:
    dev.stop()

  return R_off.astype(np.float64)


def calibrate_right_controller(duration: float = 1.0, verbose: bool = True) -> np.ndarray:
  return calibrate_controller("r", duration=duration, dev=None, verbose=verbose)


def calibrate_left_controller(duration: float = 1.0, verbose: bool = True) -> np.ndarray:
  return calibrate_controller("l", duration=duration, dev=None, verbose=verbose)
