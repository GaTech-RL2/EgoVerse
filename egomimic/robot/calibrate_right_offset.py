#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# ---- Imports to match your existing setup ----

sys.path.append(os.path.join(os.path.dirname(__file__), "oculus_reader"))
from oculus_reader import OculusReader  # type: ignore


# ---- Helpers copied from your main script ----

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
  Same mapping as in your main script.
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

def read_right_internal_rotation(oculus: OculusReader):
  sample = oculus.get_transformations_and_buttons()
  if not sample:
    return None
  transforms, buttons = sample
  if not transforms:
    return None
  T_r = transforms.get("r", None)
  if T_r is None:
    return None

  pos_raw, quat_wxyz = pose_from_T(np.asarray(T_r))
  _, quat_int_xyzw = controller_to_internal(pos_raw, quat_wxyz)
  R_int = R.from_quat(quat_int_xyzw).as_matrix()
  return R_int


def main():
  print("Initializing OculusReader (right controller)...")
  dev = OculusReader()
  print("Ready.")

  print("\nHold the RIGHT controller in your desired NEUTRAL orientation.")
  print("Interpretation with right-hand rule:")
  print("  - +X, +Y, +Z are robot base axes (RHR).")
  print("  - After calibration, at this pose yaw=pitch=roll=0 in that frame.")
  input("When stable, press ENTER to capture neutral...")

  rots = []
  t0 = time.time()
  while time.time() - t0 < 1.0:
    R_int = read_right_internal_rotation(dev)
    if R_int is not None:
      rots.append(R_int)
    time.sleep(0.01)

  if len(rots) == 0:
    print("ERROR: no samples from right controller.")
    return

  R_c_neutral = R.from_matrix(np.stack(rots, axis=0)).mean().as_matrix()

  # We want R_off * R_c_neutral = I  =>  R_off = R_c_neutral^T
  R_off = R_c_neutral.T

  ypr = R.from_matrix(R_off).as_euler("ZYX", degrees=False)

  print("\nNeutral orientation (internal frame) R_c_neutral:")
  print(R_c_neutral)
  print("\nComputed offset R_off = R_c_neutral^T:")
  print(R_off)

  print("\nPaste this into your main teleop script:")
  print(f"R_YPR_OFFSET = [{ypr[0]}, {ypr[1]}, {ypr[2]}]")
  print("(ZYX yaw, pitch, roll in radians, right-hand rule about +Z, +Y, +X.)\n")


if __name__ == "__main__":
  main()
