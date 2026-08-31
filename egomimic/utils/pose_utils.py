import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def xyzw_to_wxyz(xyzw):
    return np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)


def wxyz_to_xyzw(wxyz):
    return np.concatenate([wxyz[..., 1:4], wxyz[..., 0:1]], axis=-1)


def _interpolate_euler(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Euler-aware interpolation for a single (T, 6) or (T, 7) sequence."""
    T, D = seq.shape
    assert D in (6, 7), f"Expected 6 or 7 dims, got {D}"

    if np.any(seq >= 1e8):
        return np.full((chunk_length, D), 1e9)

    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)

    trans_interp = interp1d(old_time, seq[:, :3], axis=0, kind="linear")(new_time)

    rot_unwrapped = np.unwrap(seq[:, 3:6], axis=0)
    rot_interp = interp1d(old_time, rot_unwrapped, axis=0, kind="linear")(new_time)
    rot_interp = (rot_interp + np.pi) % (2 * np.pi) - np.pi

    if D == 6:
        return np.concatenate([trans_interp, rot_interp], axis=-1)

    grip_interp = interp1d(old_time, seq[:, 6:7], axis=0, kind="linear")(new_time)
    return np.concatenate([trans_interp, rot_interp, grip_interp], axis=-1)


def _interpolate_linear(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Simple linear interpolation for arbitrary (T, D) arrays."""
    T, _ = seq.shape
    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)
    return interp1d(old_time, seq, axis=0, kind="linear")(new_time)


def _interpolate_quat_wxyz(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Quaternion-aware interpolation for a single (T, 7) sequence."""
    T, D = seq.shape
    if D != 7:
        raise ValueError(f"Expected 7 dims for xyz+quat(wxyz), got {D}")

    if np.any(seq >= 1e8):
        return np.full((chunk_length, D), 1e9)

    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)

    trans_interp = interp1d(old_time, seq[:, :3], axis=0, kind="linear")(new_time)
    quat_wxyz = np.asarray(seq[:, 3:7], dtype=np.float64)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    norms = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Found zero-norm quaternion in input sequence.")
    quat_xyzw = quat_xyzw / norms

    # Enforce sign continuity to avoid long-path interpolation.
    quat_contiguous = quat_xyzw.copy()
    for i in range(1, T):
        if np.dot(quat_contiguous[i - 1], quat_contiguous[i]) < 0:
            quat_contiguous[i] = -quat_contiguous[i]

    if T == 1:
        quat_interp_xyzw = np.repeat(quat_contiguous[:1], chunk_length, axis=0)
    else:
        slerp = Slerp(old_time, R.from_quat(quat_contiguous))
        quat_interp_xyzw = slerp(new_time).as_quat()

    quat_interp_wxyz = quat_interp_xyzw[:, [3, 0, 1, 2]]
    dtype = seq.dtype if np.issubdtype(seq.dtype, np.floating) else np.float64
    return np.concatenate([trans_interp, quat_interp_wxyz], axis=-1).astype(
        dtype, copy=False
    )


def _interpolate_xyz(seq: np.ndarray, chunk_length: int) -> np.ndarray:
    """Linear interpolation for arbitrary (T, 3) arrays or (T, K, 3) arrays."""
    T = seq.shape[0]
    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, chunk_length)
    return interp1d(old_time, seq, axis=0, kind="linear")(new_time)


def _matrix_to_xyzypr(mats: np.ndarray) -> np.ndarray:
    """
    args:
        mats: (B, 4, 4) array of SE3 transformation matrices
    returns:
        (B, 6) np.array of [[x, y, z, yaw, pitch, roll]]
    """
    if mats.ndim != 3 or mats.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (B, 4, 4) array, got shape {mats.shape}")

    mats = np.asarray(mats)
    dtype = mats.dtype if np.issubdtype(mats.dtype, np.floating) else np.float64

    xyz = mats[:, :3, 3]
    ypr = R.from_matrix(mats[:, :3, :3]).as_euler("ZYX", degrees=False)

    return np.concatenate([xyz, ypr], axis=-1).astype(dtype, copy=False)


def _xyzypr_to_matrix(xyzypr: np.ndarray) -> np.ndarray:
    """
    args:
        xyzypr: (B, 6) np.array of [[x, y, z, yaw, pitch, roll]]
    returns:
        (B, 4, 4) array of SE3 transformation matrices
    """
    if xyzypr.ndim != 2 or xyzypr.shape[-1] != 6:
        raise ValueError(f"Expected (B, 6) array, got shape {xyzypr.shape}")
    B = xyzypr.shape[0]
    dtype = xyzypr.dtype if np.issubdtype(xyzypr.dtype, np.floating) else np.float64

    mats = np.broadcast_to(np.eye(4, dtype=dtype), (B, 4, 4)).copy()
    mats[:, :3, :3] = R.from_euler("ZYX", xyzypr[:, 3:6], degrees=False).as_matrix()
    mats[:, :3, 3] = xyzypr[:, :3]
    return mats


def _matrix_to_xyzwxyz(mats: np.ndarray) -> np.ndarray:
    """
    args:
        mats: (B, 4, 4) array of SE3 transformation matrices
    returns:
        (B, 7) np.array of [[x, y, z, qw, qx, qy, qz]]
    """
    if mats.ndim != 3 or mats.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (B, 4, 4) array, got shape {mats.shape}")

    mats = np.asarray(mats)
    dtype = mats.dtype if np.issubdtype(mats.dtype, np.floating) else np.float64

    xyz = mats[:, :3, 3]
    quat_xyzw = R.from_matrix(mats[:, :3, :3]).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    return np.concatenate([xyz, quat_wxyz], axis=-1).astype(dtype, copy=False)


def _xyzwxyz_to_matrix(xyzwxyz: np.ndarray) -> np.ndarray:
    """
    args:
        xyzwxyz: (B, 7) np.array of [[x, y, z, qw, qx, qy, qz]]
    returns:
        (B, 4, 4) array of SE3 transformation matrices
    """
    if xyzwxyz.ndim != 2 or xyzwxyz.shape[-1] != 7:
        raise ValueError(f"Expected (B, 7) array, got shape {xyzwxyz.shape}")

    B = xyzwxyz.shape[0]
    dtype = xyzwxyz.dtype if np.issubdtype(xyzwxyz.dtype, np.floating) else np.float64

    mats = np.broadcast_to(np.eye(4, dtype=dtype), (B, 4, 4)).copy()
    quat_xyzw = xyzwxyz[:, [4, 5, 6, 3]]

    mats[:, :3, :3] = R.from_quat(quat_xyzw).as_matrix()
    mats[:, :3, 3] = xyzwxyz[:, :3]

    return mats


def T_rot_orientation(T: np.ndarray, rot_orientation: np.ndarray) -> np.ndarray:
    """
    Permute the rotation matrix of a SE(3) transformation.
    """
    rot = T[:3, :3]
    rot = rot @ rot_orientation
    T[:3, :3] = rot
    return T


def _xyz_to_matrix(xyz: np.ndarray) -> np.ndarray:
    """
    args:
        xyz: (B, 3) np.array of [[x, y, z]]
    returns:
        (B, 4, 4) array of SE3 transformation matrices
    """
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected (B, 3) array, got shape {xyz.shape}")
    B = xyz.shape[0]
    dtype = xyz.dtype if np.issubdtype(xyz.dtype, np.floating) else np.float64
    mats = np.broadcast_to(np.eye(4, dtype=dtype), (B, 4, 4)).copy()
    mats[:, :3, 3] = xyz
    return mats


def _matrix_to_xyz(mats: np.ndarray) -> np.ndarray:
    """
    args:
        mats: (B, 4, 4) array of SE3 transformation matrices
    returns:
        (B, 3) np.array of [[x, y, z]]
    """
    if mats.ndim != 3 or mats.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (B, 4, 4) array, got shape {mats.shape}")
    mats = np.asarray(mats)
    dtype = mats.dtype if np.issubdtype(mats.dtype, np.floating) else np.float64
    return mats[:, :3, 3].astype(dtype, copy=False)


def _split_action_pose(actions):
    # 14D layout: [L xyz ypr g, R xyz ypr g]
    # 12D layout: [L xyz ypr, R xyz ypr]
    if actions.shape[-1] == 14:
        left_xyz = actions[..., :3]
        left_ypr = actions[..., 3:6]
        right_xyz = actions[..., 7:10]
        right_ypr = actions[..., 10:13]
    elif actions.shape[-1] == 12:
        left_xyz = actions[..., :3]
        left_ypr = actions[..., 3:6]
        right_xyz = actions[..., 6:9]
        right_ypr = actions[..., 9:12]
    else:
        raise ValueError(f"Unsupported action dim {actions.shape[-1]}")
    return left_xyz, left_ypr, right_xyz, right_ypr


def _split_keypoints(keypoints, wrist_in_data: bool = False, is_quat: bool = True):
    if wrist_in_data:
        xyz_size = 3
        if is_quat:
            angle_size = 4
        else:
            angle_size = 3
        left_xyz_index = xyz_size
        left_angle_index = left_xyz_index + angle_size
        left_keypoints_index = left_angle_index + 21 * 3
        right_xyz_index = left_keypoints_index + xyz_size
        right_angle_index = right_xyz_index + angle_size
        right_keypoints_index = right_angle_index + 21 * 3
        return (
            keypoints[..., :left_xyz_index],
            keypoints[..., left_xyz_index:left_angle_index],
            keypoints[..., left_angle_index:left_keypoints_index],
            keypoints[..., left_keypoints_index:right_xyz_index],
            keypoints[..., right_xyz_index:right_angle_index],
            keypoints[..., right_angle_index:right_keypoints_index],
        )
    else:
        xyz_size = 3
        left_keypoints = keypoints[..., :63]
        right_keypoints = keypoints[..., 63:]
        return left_keypoints, right_keypoints

from scipy.spatial.transform import Rotation
import scipy


# ---- moved from egomimicUtils.py (code unchanged) ----

def ee_pose_to_cam_frame(ee_pose_base, T_cam_base):
    """
    ee_pose_base: (N, 3)
    T_cam_base: (4, 4)

    returns ee_pose_cam: (N, 3)
    """
    N, _ = ee_pose_base.shape
    ee_pose_base = np.concatenate([ee_pose_base, np.ones((N, 1))], axis=1)

    ee_pose_grip_cam = np.linalg.inv(T_cam_base) @ ee_pose_base.T
    return ee_pose_grip_cam.T[:, :3]

def base_frame_to_cam_frame(base_frame, T_cam_base):
    """
    base_frame: (N, 6) (x, y, z, yaw, pitch, roll)
    T_cam_base: (4, 4)

    returns cam_frame: (N, 6) (x, y, z, yaw, pitch, roll)
    """
    N, _ = base_frame.shape
    se3 = np.zeros((N, 4, 4))
    se3[:, :3, :3] = Rotation.from_euler("ZYX", base_frame[:, 3:6]).as_matrix()
    se3[:, :3, 3] = base_frame[:, :3]
    se3[:, 3, 3] = 1
    cam_frame = np.linalg.inv(T_cam_base) @ se3
    xyz = cam_frame[:, :3, 3]
    ypr = Rotation.from_matrix(cam_frame[:, :3, :3]).as_euler("ZYX", degrees=False)
    return np.concatenate([xyz, ypr], axis=1)

def cam_frame_to_base_frame(cam_frame, T_cam_base):
    """
    cam_frame: (N, 6) (x, y, z, yaw, pitch, roll)
    T_cam_base: (4, 4)

    returns base_frame: (N, 6) (x, y, z, yaw, pitch, roll)
    """
    N, _ = cam_frame.shape
    se3 = np.zeros((N, 4, 4))
    se3[:, :3, :3] = Rotation.from_euler("ZYX", cam_frame[:, 3:6]).as_matrix()
    se3[:, :3, 3] = cam_frame[:, :3]
    se3[:, 3, 3] = 1
    base_frame = T_cam_base @ se3
    xyz = base_frame[:, :3, 3]
    ypr = Rotation.from_matrix(base_frame[:, :3, :3]).as_euler("ZYX", degrees=False)
    return np.concatenate([xyz, ypr], axis=1)

def pose_to_transform(pose):
    """
    Convert a 6D pose [x, y, z, yaw, pitch, roll] into a 4x4 homogeneous transform.
    Assumes Euler angles are in radians and follow ZYX (yaw-pitch-roll) order.
    """
    x, y, z, yaw, pitch, roll = pose

    # Compute individual rotation matrices
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    # Combined rotation: note the multiplication order
    R = Rz @ Ry @ Rx

    # Assemble homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def transform_to_pose(T):
    """
    Convert a 4x4 homogeneous transform back to a 6D pose [x, y, z, yaw, pitch, roll].
    Uses the ZYX (yaw-pitch-roll) convention.
    """
    x, y, z = T[:3, 3]
    R = T[:3, :3]

    # Extract pitch from the (3,1) element of R
    pitch = np.arcsin(-R[2, 0])
    # To avoid numerical issues, check for gimbal lock:
    cos_pitch = np.cos(pitch)
    if np.abs(cos_pitch) > 1e-6:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        # Gimbal lock: arbitrarily set yaw=0
        yaw = 0
        roll = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([x, y, z, yaw, pitch, roll])

def cam_frame_to_cam_pixels(ee_pose_cam, intrinsics):
    """
    camera frame 3d coordinates to pixels in camera frame
    ee_pose_cam: (N, 3)
    intrinsics: 3x4 matrix
    """
    N, _ = ee_pose_cam.shape
    ee_pose_cam = np.concatenate([ee_pose_cam, np.ones((N, 1))], axis=1)
    # print("3d pos in cam frame: ", ee_pose_cam)

    # print("intrinsics: ", intrinsics.shape, ee_pose_cam.shape)
    px_val = intrinsics @ ee_pose_cam.T
    px_val = px_val / px_val[2, :]
    # print("2d pos cam frame: ", px_val)

    return px_val.T

def interpolate_arr_euler(v: np.ndarray, seq_length: int) -> np.ndarray:
    """
    Interpolate 6DoF poses (translation + Euler angles in radians),
    optionally with a 7th gripper dimension, along the time axis.

    v: (B, T, 6) or (B, T, 7)
        [x, y, z, yaw, pitch, roll, (optional) gripper]
    """
    assert v.ndim == 3 and v.shape[2] in (
        6,
        7,
    ), "Input v must be of shape (B, T, 6) or (B, T, 7)"
    B, T, D = v.shape

    new_time = np.linspace(0, 1, seq_length)
    old_time = np.linspace(0, 1, T)

    outputs = []

    for i in range(B):
        seq = v[i]  # (T, D)

        if np.any(seq >= 1e8):
            outputs.append(np.full((seq_length, D), 1e9))
            continue

        trans_seq = seq[:, :3]  # x, y, z
        rot_seq = seq[:, 3:6]  # yaw, pitch, roll

        # Avoid discontinuities in angle interpolation
        rot_seq_unwrapped = np.unwrap(rot_seq, axis=0)

        trans_interp_func = scipy.interpolate.interp1d(
            old_time, trans_seq, axis=0, kind="linear"
        )
        rot_interp_func = scipy.interpolate.interp1d(
            old_time, rot_seq_unwrapped, axis=0, kind="linear"
        )

        trans_interp = trans_interp_func(new_time)  # (seq_length, 3)
        rot_interp = rot_interp_func(new_time)  # (seq_length, 3)

        # Wrap back to [-pi, pi)
        rot_interp = (rot_interp + np.pi) % (2 * np.pi) - np.pi

        if D == 6:
            out_seq = np.concatenate([trans_interp, rot_interp], axis=-1)
        else:
            grip_seq = seq[:, 6:7]  # (T, 1)
            grip_interp_func = scipy.interpolate.interp1d(
                old_time, grip_seq, axis=0, kind="linear"
            )
            grip_interp = grip_interp_func(new_time)  # (seq_length, 1)
            out_seq = np.concatenate([trans_interp, rot_interp, grip_interp], axis=-1)

        outputs.append(out_seq)

    return np.stack(outputs, axis=0)  # (B, seq_length, D)

def interpolate_arr(v, seq_length):
    """
    v: (B, T, D)
    seq_length: int
    """
    assert len(v.shape) == 3
    if v.shape[1] == seq_length:
        return

    interpolated = []
    for i in range(v.shape[0]):
        index = v[i]

        interp = scipy.interpolate.interp1d(
            np.linspace(0, 1, index.shape[0]), index, axis=0
        )
        interpolated.append(interp(np.linspace(0, 1, seq_length)))

    return np.array(interpolated)

def get_vector_from_yaw_pitch(
    yaw_rads: float,
    pitch_rads: float,
    depth: float | None = None,
) -> np.ndarray:
    """
    Convert yaw / pitch angles into a 3D gaze vector in CPF coordinates.

    Args:
        yaw_rads: Yaw angle in radians.
        pitch_rads: Pitch angle in radians.
        depth: Optional gaze distance. If provided, returns a vector with this
            magnitude. If None, returns a unit vector.

    Returns:
        np.ndarray: (3,) gaze vector in CPF coordinates.
    """
    z = 1.0
    x = np.tan(yaw_rads) * z
    y = np.tan(pitch_rads) * z

    direction = np.array([x, y, z], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Zero-length direction vector")

    unit_dir = direction / norm

    if depth is None:
        return unit_dir
    else:
        return unit_dir * depth


def safe_rot3_from_T(T, ortho_tol=1e-3, det_tol=1e-3):
    """Extract a trustworthy 3x3 rotation from a pose matrix, else identity.

    Rejects non-finite entries, a non-positive or off-unit determinant, and
    columns that have drifted from orthonormal. Used by the teleop/calibration
    paths, where a silently-degenerate rotation is worse than falling back.
    """
    Rm = np.asarray(T, dtype=float)[:3, :3]
    if Rm.shape != (3, 3) or not np.all(np.isfinite(Rm)):
        return np.eye(3)
    det = np.linalg.det(Rm)
    if det <= 0 or abs(det - 1.0) > det_tol:
        return np.eye(3)
    if np.linalg.norm(Rm.T @ Rm - np.eye(3), ord="fro") > ortho_tol:
        return np.eye(3)
    return Rm
