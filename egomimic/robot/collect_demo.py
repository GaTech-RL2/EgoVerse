#!/usr/bin/env python3
"""
This script collects demonstrations from the robot using a VR controller.

In each iteration, it reads delta poses from the VR controller, computes target
end-effector poses, and uses the robot interface to control the robot.
It also saves observations (images, robot states) and actions to a demo file.
"""

import os
import sys
import time
import numpy as np
import copy
import cv2
import h5py
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# Add path to oculus_reader if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "oculus_reader"))
from oculus_reader import OculusReader

# Import local modules
from robot_utils import RateLoop

# Add path to robot_interface
sys.path.append(os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))
from robot_interface import ARXInterface, YAMInterface
from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver

# Add i2rt to path for YAM kinematics in save_demo
sys.path.insert(0, os.path.expanduser("~/i2rt"))
from i2rt.robots.kinematics import Kinematics as I2RTKinematics
from i2rt.robots.utils import GripperType


# ------------------------- Configuration -------------------------

# Control parameters
DEFAULT_FREQUENCY = 30.0  # Hz
POSITION_SCALE = 1.0  # Scale factor for position deltas
ROTATION_SCALE = 1.0  # Scale factor for rotation deltas

# Velocity limits (per tick) - 0 = disabled
DEFAULT_MAX_DELTA_POS = 0.0  # meters/tick
DEFAULT_MAX_DELTA_ROT_DEG = 0.0  # degrees/tick

# Headset orientation correction (when headset cameras face user instead of away)
# This flips X and Z axes to correct for 180° rotation around Y (vertical) axis
HEADSET_FLIPPED = True  # Set True if headset is worn backwards (cameras facing user)

# Dead-zone thresholds (to filter out jitter)
POS_DEAD_ZONE = 0.002  # meters
ROT_DEAD_ZONE_RAD = np.deg2rad(0.8)  # radians

# R_YPR_OFFSET = [0, 1, 0]
# L_YPR_OFFSET = [0, 1, 0]

R_YPR_OFFSET = np.array([
  [ 0.66509066, -0.16738938,  0.72776041],
  [ 0.22521625,  0.97413813,  0.0182356 ],
  [-0.71199161,  0.15177514,  0.68558898],
], dtype=np.float64)

L_YPR_OFFSET = np.array([
  [0.6785254459380761, 0.036920397978411894, 0.7336484876476287],
  [-0.05616599291181174, 0.9984199955834385, 0.0017010759532792748],
  [-0.7324265153957544, -0.04236031907675143, 0.6795270435479006],
], dtype=np.float64)


NEUTRAL_ROT_OFFSET_R = np.eye(3)
NEUTRAL_ROT_OFFSET_L = np.eye(3)
YPR_VEL = [1.5, 1.5, 1.5]  # rad/s
YPR_RANGE = [2, 2, 2]

# Trigger thresholds for engagement detection
TRIGGER_ON_THRESHOLD = 0.8
TRIGGER_OFF_THRESHOLD = 0.2

# Gripper thresholds for ARX robot
GRIPPER_OPEN_VALUE = 0.08
GRIPPER_CLOSE_VALUE = -0.018
GRIPPER_WIDTH = GRIPPER_OPEN_VALUE - GRIPPER_CLOSE_VALUE
GRIPPER_VEL = 1  # m/s gripper width is normally around 0.08m

# Gripper thresholds for YAM robot (normalized 0-1)
YAM_GRIPPER_OPEN_VALUE = 1.0   # fully open
YAM_GRIPPER_CLOSE_VALUE = 0.0  # fully closed
YAM_GRIPPER_WIDTH = YAM_GRIPPER_OPEN_VALUE - YAM_GRIPPER_CLOSE_VALUE

# Demo recording
DEMO_DIR = "./demos"
MAX_DEMO_LENGTH = 10000  # Maximum number of steps per demo


# ------------------------- Helper Functions -------------------------


def load_velocity_limits_from_config():
    """Load velocity limits from configs_yam.yaml."""
    import yaml
    config_paths = [
        os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/config/configs_yam.yaml"),
        "/home/robot/robot_ws/egomimic/robot/eva/eva_ws/src/config/configs_yam.yaml",
    ]
    for cfg_path in config_paths:
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    vr_cfg = (yaml.safe_load(f) or {}).get("vr_teleop", {})
                return vr_cfg.get("max_delta_pos", DEFAULT_MAX_DELTA_POS), \
                       np.deg2rad(vr_cfg.get("max_delta_rot_deg", DEFAULT_MAX_DELTA_ROT_DEG))
            except Exception:
                pass
    return DEFAULT_MAX_DELTA_POS, np.deg2rad(DEFAULT_MAX_DELTA_ROT_DEG)


def clamp_delta_pos(dpos: np.ndarray, max_delta: float) -> np.ndarray:
    """Clamp position delta magnitude. Returns original if max_delta <= 0."""
    if max_delta <= 0:
        return dpos
    norm = np.linalg.norm(dpos)
    return dpos if (norm <= max_delta or norm < 1e-9) else dpos * (max_delta / norm)


def clamp_delta_rot(rotvec: np.ndarray, max_rad: float) -> np.ndarray:
    """Clamp rotation vector angle. Returns original if max_rad <= 0."""
    if max_rad <= 0:
        return rotvec
    angle = np.linalg.norm(rotvec)
    return rotvec if (angle <= max_rad or angle < 1e-9) else (rotvec / angle) * max_rad


def se3_to_xyzxyzw(se3):
    """Convert SE(3) transformation matrix (4x4) to position and quaternion."""
    rot = se3[:3, :3]
    xyzw = R.from_matrix(rot).as_quat()
    xyz = se3[:3, 3]
    return xyz, xyzw


def xyzxyzw_to_se3(xyz, xyzw):
    """
    Convert position (xyz) and quaternion (xyzw) to SE(3) 4x4 transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = R.from_quat(xyzw).as_matrix()
    T[:3, 3] = xyz
    return T


def flip_roll_only(R_i, up=np.array([0.0, 0.0, 1.0]), add_pi=True):
    # body axes from R_i (columns)
    x = R_i[:, 0]
    y = R_i[:, 1]
    if abs(x @ up) > 0.99:
        up = np.array([0.0, 1.0, 0.0])

    y0 = up - (up @ x) * x
    y0 /= np.linalg.norm(y0)
    z0 = np.cross(x, y0)

    c = y @ y0
    s = y @ z0
    y_flipped = c * y0 - s * z0
    z_flipped = np.cross(x, y_flipped)

    R_out = np.column_stack([x, y_flipped, z_flipped])

    if add_pi:
        # 180° about body X (roll): leaves x col, flips y/z cols
        R_out = R_out @ np.diag([1.0, -1.0, -1.0])
        # equivalently: R_out[:, 1:] *= -1

    return R_out


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


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion in XYZW format."""
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def clip_ypr(ypr, clipped_bound) -> np.ndarray:
    ypr_range = np.array(clipped_bound)
    clipped_ypr = np.clip(np.array(ypr), -ypr_range, ypr_range)
    return clipped_ypr


def limit_delta_quat_by_rate(
    delta_quat_xyzw: np.ndarray, max_rate_rad_s: float, dt: float
) -> np.ndarray:
    # Limit the angular magnitude of the delta quaternion to max_rate * dt
    R_delta = R.from_quat(delta_quat_xyzw)  # xyzw
    rotvec = R_delta.as_rotvec()  # axis * angle
    angle = np.linalg.norm(rotvec)
    max_angle = max_rate_rad_s * dt
    if angle > max_angle and angle > 1e-12:
        rotvec = rotvec * (max_angle / angle)
    return R.from_rotvec(rotvec).as_quat()  # xyzw


def quat_xyzw_to_wxyz(qxyzw: np.ndarray) -> np.ndarray:
    """Convert quaternion from XYZW to WXYZ format."""
    return np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(qwxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion from WXYZ to XYZW format."""
    return np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]], dtype=np.float64)


def pose_from_T(T: np.ndarray):
    """Extract position and quaternion (WXYZ) from transformation matrix."""
    pos = T[:3, 3].astype(np.float64)
    rot_mat = safe_rot3_from_T(T[:3, :3])
    q_xyzw = R.from_matrix(rot_mat).as_quat()
    q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
    return pos, q_wxyz


def get_analog(buttons: dict, keys, default=0.0) -> float:
    """Extract analog value from button dictionary."""
    for k in keys:
        v = buttons.get(k, None)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            try:
                return float(v[0])
            except Exception:
                continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, bool):
            return 1.0 if v else 0.0
    return float(default)


def controller_to_internal(pos_xyz: np.ndarray, q_wxyz: np.ndarray):
    """
    Convert controller coordinates to internal robot frame.

    Applies fixed coordinate transformations as defined in vr_controller.py.
    pos : xyz, quat: xyzw
    """
    A = np.array(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
    )
    B = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    M = B @ A

    R_c = R.from_quat(quat_wxyz_to_xyzw(q_wxyz)).as_matrix()
    pos_i = M @ pos_xyz
    R_i = M @ R_c @ M.T
    q_i = R.from_matrix(R_i).as_quat()
    return pos_i, q_i


def quat_rel_wxyz(q_cur_wxyz: np.ndarray, q_prev_wxyz: np.ndarray) -> np.ndarray:
    """Compute relative quaternion between current and previous orientations."""
    R_cur = R.from_quat(quat_wxyz_to_xyzw(q_cur_wxyz))
    R_prev = R.from_quat(quat_wxyz_to_xyzw(q_prev_wxyz))
    R_rel = R_cur * R_prev.inv()
    return quat_xyzw_to_wxyz(R_rel.as_quat())


def apply_delta_pose(
    current_pos: np.ndarray,
    current_quat_xyzw: np.ndarray,
    delta_pos: np.ndarray,
    delta_quat_xyzw: np.ndarray,
) -> tuple:
    """
    Apply delta pose to current pose.

    Args:
        current_pos: Current position [x, y, z]
        current_quat_xyzw: Current orientation quaternion [x, y, z, w]
        delta_pos: Delta position [dx, dy, dz]
        delta_quat_xyzw: Delta orientation quaternion [w, dx, dy, dz]

    Returns:
        Tuple of (new_pos, new_quat_xyzw)
    """
    # Apply position delta
    new_pos = current_pos + delta_pos

    # Apply rotation delta
    R_current = R.from_quat(current_quat_xyzw)
    R_delta = R.from_quat(delta_quat_xyzw)
    R_new = R_delta * R_current
    new_quat_xyzw = R_new.as_quat()

    return new_pos, new_quat_xyzw


def compute_delta_pose(
    side: str, target_vr_data: dict, cur_pos: dict, cur_quat: np.ndarray
) -> tuple:
    """
    Compute delta pose for one side.

    Args:
        side: 'left' or 'right'
        vr_data: VR controller data dictionary
        prev_pos: Previous position (or None)
        prev_quat: Previous quaternion (or None)

    Returns:
        Tuple of (delta_pos, delta_quat)
    """
    target_side_data = target_vr_data[side]

    target_pos = target_side_data["pos"]
    target_quat = target_side_data["quat"]

    delta_pos = target_pos - cur_pos
    delta_rot = R.from_quat(target_quat) * R.from_quat(cur_quat).inv()

    return delta_pos * POSITION_SCALE, delta_rot.as_quat()


# ------------------------- VR Interface Class -------------------------


class VRInterface:
    """Tracks VR controller state and provides access to VR data."""

    def __init__(self):
        """Initialize VR interface."""
        print("Initializing Oculus Reader...")
        self.device = OculusReader()

        # State tracking for delta computation
        self.r_prev_pos = None
        self.r_prev_quat = None
        self.l_prev_pos = None
        self.l_prev_quat = None

        # Trigger engagement state (hysteresis)
        self.r_engaged = False
        self.l_engaged = False

        self.r_up_edge = False
        self.r_down_edge = False
        self.l_up_edge = False
        self.l_down_edge = False

        # Gripper state
        self.r_gripper_closed = False
        self.l_gripper_closed = False
        self.r_gripper_value = GRIPPER_OPEN_VALUE
        self.l_gripper_value = GRIPPER_OPEN_VALUE

        print("VR Interface initialized!")

    def update_engagement(self, trigger_value: float, arm: str):
        """
        Update engagement with hysteresis and edge detection.
        Returns (rising_edge, falling_edge, engaged_state).
        """
        if arm == "right":
            engaged = self.r_engaged
        else:
            engaged = self.l_engaged

        rising = False
        falling = False

        # clear edge flags each call (one-shot semantics)
        if arm == "right":
            self.r_up_edge = False
            self.r_down_edge = False
        else:
            self.l_up_edge = False
            self.l_down_edge = False

        if not engaged and trigger_value >= TRIGGER_ON_THRESHOLD:
            engaged = True
            rising = True
        elif engaged and trigger_value <= TRIGGER_OFF_THRESHOLD:
            engaged = False
            falling = True

        # write back state + edges
        if arm == "right":
            self.r_engaged = engaged
            self.r_up_edge = rising
            self.r_down_edge = falling
        else:
            self.l_engaged = engaged
            self.l_up_edge = rising
            self.l_down_edge = falling

    def read_vr_controller(self, se3=False):
        """Read VR controller state and return parsed data."""
        sample = self.device.get_transformations_and_buttons()
        # print(f"sample: {sample}")
        if not sample:
            return None

        transforms, buttons = sample
        if not transforms:
            return None

        # Extract button/trigger values
        trig_l = get_analog(buttons, ["leftTrig", "LT", "trigger_l"], 0.0)
        trig_r = get_analog(buttons, ["rightTrig", "RT", "trigger_r"], 0.0)
        idx_l = get_analog(buttons, ["leftGrip", "LG", "grip_l"], trig_l)
        idx_r = get_analog(buttons, ["rightGrip", "RG", "grip_r"], trig_r)

        # Get buttons
        btn_a = bool(buttons.get("A", False))
        btn_b = bool(buttons.get("B", False))
        btn_x = bool(buttons.get("X", False))
        btn_y = bool(buttons.get("Y", False))

        Tl = transforms.get("l", None)
        Tr = transforms.get("r", None)
        if Tl is None or Tr is None:
            return None

        # Convert to internal coordinates
        l_pos_raw, l_quat_raw = pose_from_T(np.asarray(Tl))
        r_pos_raw, r_quat_raw = pose_from_T(np.asarray(Tr))
        l_pos_cur, l_quat_cur = controller_to_internal(l_pos_raw, l_quat_raw)
        r_pos_cur, r_quat_cur = controller_to_internal(r_pos_raw, r_quat_raw)
        l_quat_cur = normalize_quat_xyzw(l_quat_cur)
        r_quat_cur = normalize_quat_xyzw(r_quat_cur)
        
        R_l_cur = R.from_quat(l_quat_cur).as_matrix()
        R_r_cur = R.from_quat(r_quat_cur).as_matrix()

        R_l_new = L_YPR_OFFSET @ R_l_cur
        R_r_new = R_YPR_OFFSET @ R_r_cur

        l_quat_cur = normalize_quat_xyzw(R.from_matrix(R_l_new).as_quat())
        r_quat_cur = normalize_quat_xyzw(R.from_matrix(R_r_new).as_quat())

        # l_quat_cur = R.from_matrix(flip_roll_only(R.from_quat(l_quat_cur).as_matrix())).as_quat()
        # r_quat_cur = R.from_matrix(flip_roll_only(R.from_quat(r_quat_cur).as_matrix())).as_quat()
        # l_quat_cur = normalize_quat_xyzw(l_quat_cur)
        # r_quat_cur = normalize_quat_xyzw(r_quat_cur)

        # Apply ypr offset
        # zero = np.zeros(3)
        # _, l_quat_cur = apply_delta_pose(
        #     l_pos_cur,
        #     l_quat_cur,
        #     zero,
        #     R.from_euler("ZYX", L_YPR_OFFSET, degrees=False).as_quat(),
        # )
        # _, r_quat_cur = apply_delta_pose(
        #     r_pos_cur,
        #     r_quat_cur,
        #     zero,
        #     R.from_euler("ZYX", R_YPR_OFFSET, degrees=False).as_quat(),
        # )
        # print(R.from_quat(l_quat_cur).as_euler("ZYX", degrees=False))

        # eul = R.from_quat(r_quat_cur).as_euler("ZYX", degrees=False)  # [yaw, pitch, roll]
        # eul[2] = -eul[2]                                     # flip roll sign
        # r_quat_cur = R.from_euler("ZYX", eul, degrees=False).as_quat()  # xyzw

        # Create SE(3) matrices with YPR offset applied
        Tl = xyzxyzw_to_se3(l_pos_cur, l_quat_cur)
        Tr = xyzxyzw_to_se3(r_pos_cur, r_quat_cur)

        if se3:
            # Return SE(3) transformation matrices
            return {
                "left": {
                    "T": Tl,
                    "trigger": trig_l,
                    "index": idx_l,
                },
                "right": {
                    "T": Tr,
                    "trigger": trig_r,
                    "index": idx_r,
                },
                "buttons": {"A": btn_a, "B": btn_b, "X": btn_x, "Y": btn_y},
            }
        else:
            # Return position and quaternion format
            return {
                "left": {
                    "pos": l_pos_cur,
                    "quat": l_quat_cur,
                    "trigger": trig_l,
                    "index": idx_l,
                },
                "right": {
                    "pos": r_pos_cur,
                    "quat": r_quat_cur,
                    "trigger": trig_r,
                    "index": idx_r,
                },
                "buttons": {"A": btn_a, "B": btn_b, "X": btn_x, "Y": btn_y},
            }


# ------------------------- Demo Recording Helpers -------------------------


def reset_data(demo_data: dict):
    demo_data["cmd_joint_actions"] = []
    demo_data["robot_joint_actions"] = []
    demo_data["cmd_eepose_actions"] = []
    demo_data["obs"] = []


def save_demo(demo_data: dict, demo_dir, episode_id: int, cam_names, robot_type: str = "arx", yam_gripper_type: str = "linear_4310", save_resolution: dict = None):
    """Save demo to HDF5 file.
    
    Args:
        save_resolution: Optional dict with 'width' and 'height' keys to resize images before saving.
                        Uses bicubic interpolation. If None, images are saved at original resolution.
    """
    data_dict = dict()
    filename = demo_dir / f"demo_{episode_id}.hdf5"

    for cam_name in cam_names:
        image_list = []
        for i in range(len(demo_data["obs"])):
            img = demo_data["obs"][i][cam_name]
            if img is None:
                continue
            img_rgb = img[..., ::-1]
            # Resize if save_resolution is specified
            if save_resolution is not None:
                target_w = save_resolution["width"]
                target_h = save_resolution["height"]
                img_rgb = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            image_list.append(img_rgb)
        data_dict[f"/observations/images/{cam_name}"] = np.array(image_list)
    print(
        f"Saving demo with {len(demo_data['cmd_eepose_actions'])} steps to {filename}"
    )
    data_dict["/observations/joints"] = np.array(demo_data["robot_joint_actions"])
    data_dict["/observations/joint_positions"] = np.array(
        demo_data["robot_joint_actions"]
    )
    # data_dict["/observations/qjointvel"] = joint_vels
    
    # Process cmd_eepose_actions to create different rotation representations
    # Original format: xyz(3) + ypr(3) + gripper(1) per arm = 14 total
    cmd_eepose_ypr = np.array(demo_data["cmd_eepose_actions"])
    
    # Batch convert YPR to rotation matrices for both arms
    left_rot_mats = R.from_euler("ZYX", cmd_eepose_ypr[:, 3:6]).as_matrix()
    right_rot_mats = R.from_euler("ZYX", cmd_eepose_ypr[:, 10:13]).as_matrix()
    
    # 6D rotation: first two columns of rotation matrix, flattened
    left_rot_6d = left_rot_mats[:, :, :2].reshape(-1, 6)
    right_rot_6d = right_rot_mats[:, :, :2].reshape(-1, 6)
    
    # Build eepose_6drot: xyz(3) + rot6d(6) + gripper(1) per arm = 20 total
    cmd_eepose_6drot = np.column_stack([
        cmd_eepose_ypr[:, 0:3], left_rot_6d, cmd_eepose_ypr[:, 6:7],
        cmd_eepose_ypr[:, 7:10], right_rot_6d, cmd_eepose_ypr[:, 13:14],
    ])
    
    # Full rotation matrices: (N, 2, 3, 3)
    cmd_rot_3x3 = np.stack([left_rot_mats, right_rot_mats], axis=1)
    
    data_dict["/actions/eepose_ypr"] = cmd_eepose_ypr
    data_dict["/actions/eepose_6drot"] = cmd_eepose_6drot
    data_dict["/actions/rot_3x3"] = cmd_rot_3x3
    data_dict["/actions/joints"] = np.array(demo_data["cmd_joint_actions"])
    data_dict["/action"] = np.array(demo_data["cmd_joint_actions"])

    # Create kinematics solver based on robot type
    if robot_type == "arx":
        kinematics_solver = EvaMinkKinematicsSolver(
            model_path="/home/robot/robot_ws/egomimic/resources/model_x5.xml"
        )
    elif robot_type == "yam":
        gripper_type_enum = GripperType.from_string_name(yam_gripper_type)
        model_path = gripper_type_enum.get_xml_path()
        kinematics_solver = I2RTKinematics(xml_path=model_path, site_name="grasp_site")
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")
    
    robot_ee_pose = []
    for i in range(len(demo_data["robot_joint_actions"])):
        robot_joint_action = demo_data["robot_joint_actions"][i]
        left_joints = robot_joint_action[:7]
        right_joints = robot_joint_action[7:]
        # check if left is not 0 array
        if not np.allclose(left_joints, 0):
            if robot_type == "arx":
                left_ee_xyz, left_ee_rot = kinematics_solver.fk(left_joints)
                left_ee_ypr = left_ee_rot.as_euler("ZYX", degrees=False)
            else:  # yam
                T = kinematics_solver.fk(left_joints[:6])
                left_ee_xyz = T[:3, 3]
                left_ee_ypr = R.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=False)
        else:
            left_ee_xyz = np.zeros(3)
            left_ee_ypr = np.zeros(3)
        if not np.allclose(right_joints, 0):
            if robot_type == "arx":
                right_ee_xyz, right_ee_rot = kinematics_solver.fk(right_joints)
                right_ee_ypr = right_ee_rot.as_euler("ZYX", degrees=False)
            else:  # yam
                T = kinematics_solver.fk(right_joints[:6])
                right_ee_xyz = T[:3, 3]
                right_ee_ypr = R.from_matrix(T[:3, :3]).as_euler("ZYX", degrees=False)
        else:
            right_ee_xyz = np.zeros(3)
            right_ee_ypr = np.zeros(3)
        left_ee_pose = np.concatenate(
            [left_ee_xyz, left_ee_ypr, [robot_joint_action[6]]]
        )
        right_ee_pose = np.concatenate(
            [right_ee_xyz, right_ee_ypr, [robot_joint_action[13]]]
        )
        robot_ee_pose.append(np.concatenate([left_ee_pose, right_ee_pose]))

    data_dict["/observations/eepose"] = np.array(robot_ee_pose)
    t0 = time.time()
    max_timesteps = len(demo_data["cmd_eepose_actions"])
    with h5py.File(str(filename), "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in cam_names:
            # Get image dimensions from actual data
            img_data = data_dict.get(f"/observations/images/{cam_name}")
            if img_data is not None and len(img_data) > 0:
                img_height, img_width = img_data.shape[1], img_data.shape[2]
            else:
                # Fallback to default dimensions if no data
                img_height, img_width = 480, 640
            _ = image.create_dataset(
                cam_name,
                (max_timesteps, img_height, img_width, 3),
                dtype="uint8",
                chunks=(1, img_height, img_width, 3),
            )
        _ = obs.create_dataset("joints", (max_timesteps, 14))
        # _ = obs.create_dataset("qjointvel", (max_timesteps, 16))
        _ = obs.create_dataset("eepose", (max_timesteps, 14))
        _ = obs.create_dataset("joint_positions", (max_timesteps, 14))
        _ = root.create_group("actions")
        _ = root["actions"].create_dataset("eepose_ypr", (max_timesteps, 14))
        _ = root["actions"].create_dataset("eepose_6drot", (max_timesteps, 20))
        _ = root["actions"].create_dataset("rot_3x3", (max_timesteps, 2, 3, 3))
        _ = root["actions"].create_dataset("joints", (max_timesteps, 14))
        _ = root.create_dataset("action", (max_timesteps, 14))

        for name, array in data_dict.items():
            root[name][...] = array

    print(f"Saving: {(time.time() - t0):.1f} secs")
    return True


# ------------------------- Main Entry Point -------------------------


def collect_demo(
    arms_to_collect: str = "right",
    frequency: float = DEFAULT_FREQUENCY,
    demo_dir: str = DEMO_DIR,
    recording: bool = True,
    auto_episode_start: int = None,
    robot_type: str = "arx",
    yam_gripper_type: str = "linear_4310",
    yam_interfaces: dict = None,
    dry_run: bool = False,
    position_scale: float = POSITION_SCALE,
    rotation_scale: float = ROTATION_SCALE,
    max_delta_pos: float = None,
    max_delta_rot_deg: float = None,
):
    """
    Collect demonstrations using VR controller.

    Args:
        arms: Which arm(s) to control ("left", "right", or "both")
        frequency: Control loop frequency in Hz
        demo_dir: Directory to save demos
        robot_type: Robot type to use ("arx" for ARX X5, "yam" for I2RT YAM)
        yam_gripper_type: Gripper type for YAM robot (only used if robot_type="yam")
        yam_interfaces: CAN interface mapping for YAM robot {"left": "can0", "right": "can1"}
        dry_run: If True, don't actuate robot - just log VR commands
        position_scale: Scale factor for VR position deltas
        rotation_scale: Scale factor for VR rotation deltas
        max_delta_pos: Max position change per tick in meters (None = load from config, 0 = disabled)
        max_delta_rot_deg: Max rotation change per tick in degrees (None = load from config, 0 = disabled)
    """
    # Load velocity limits from config if not specified
    if max_delta_pos is None or max_delta_rot_deg is None:
        cfg_pos, cfg_rot_rad = load_velocity_limits_from_config()
        max_delta_pos = cfg_pos if max_delta_pos is None else max_delta_pos
        max_delta_rot_rad = cfg_rot_rad if max_delta_rot_deg is None else np.deg2rad(max_delta_rot_deg)
    else:
        max_delta_rot_rad = np.deg2rad(max_delta_rot_deg)
    
    if max_delta_pos > 0 or max_delta_rot_rad > 0:
        print(f"[VelocityLimit] pos={max_delta_pos:.4f}m/tick, rot={np.rad2deg(max_delta_rot_rad):.2f}deg/tick")
    
    # Setup demo directory
    demo_dir = Path(demo_dir)
    demo_dir.mkdir(exist_ok=True, parents=True)

    # Initialize VR interface
    vr = VRInterface()
    prev_vr_data = None
    
    # Track previous commanded SE3 for velocity limiting
    prev_cmd_T = {}

    # Initialize robot interfaces (one per arm)
    if arms_to_collect == "both":
        arms = ["right", "left"]
    elif arms_to_collect == "right":
        arms = ["right"]
    elif arms_to_collect == "left":
        arms = ["left"]
    else:
        raise ValueError("Invalid arm values inputted.")
    
    # Select robot interface based on type
    if robot_type == "arx":
        if dry_run:
            print("ARX robot does not support dry run mode - use YAM robot for dry run testing")
            print("Proceeding anyway (will fail if ARX hardware not connected)")
        print("Using ARX robot interface")
        robot_interface = ARXInterface(arms=arms)
        # ARX uses raw gripper values
        gripper_open = GRIPPER_OPEN_VALUE
        gripper_close = GRIPPER_CLOSE_VALUE
        gripper_width = GRIPPER_WIDTH
    elif robot_type == "yam":
        print("Using YAM robot interface")
        gripper_type_enum = GripperType.from_string_name(yam_gripper_type)
        robot_interface = YAMInterface(
            arms=arms,
            gripper_type=gripper_type_enum,
            interfaces=yam_interfaces,
            zero_gravity_mode=False,  # Hold position on startup for safety
            dry_run=dry_run,
            read_all_arms=True,  # Always read proprioception from both arms
        )
        # Print detailed config for YAM
        robot_interface.print_config()
        # YAM uses normalized gripper values (0-1)
        gripper_open = YAM_GRIPPER_OPEN_VALUE
        gripper_close = YAM_GRIPPER_CLOSE_VALUE
        gripper_width = YAM_GRIPPER_WIDTH
    else:
        raise ValueError(f"Unknown robot type: {robot_type}. Use 'arx' or 'yam'.")

    arms_list = []
    if arms_to_collect == "both" or arms_to_collect == "right":
        arms_list.append("right")
    if arms_to_collect == "both" or arms_to_collect == "left":
        arms_list.append("left")

    # Demo recording state
    demo_data = dict()

    camera_names = robot_interface.recorders.keys()
    cmd_pos = dict()
    cmd_quat = dict()
    cmd_joints = dict()
    gripper_pos = dict()
    collecting_data = False
    vr_frame_zero_se3 = dict()
    robot_frame_zero_se3 = dict()
    vr_neutral_frame_delta = dict()
    for arm in arms_list:
        vr_neutral_frame_delta[arm] = np.eye(4)
    print("Waiting for incoming images ----------------")
    all_cam_images_in = False
    with RateLoop(frequency=frequency, verbose=False) as loop:
        for i in loop:
            obs = robot_interface.get_obs()
            all_cam_images_in = True
            for cam_name in robot_interface.recorders.keys():
                if obs[cam_name] is None:
                    all_cam_images_in = False
            if all_cam_images_in is True:
                break
    print("All cameras are ready --------------")
    
    auto_episode_id = auto_episode_start
    
    while True:
        if auto_episode_id is None:
            episode_id = input("Input the episode id: ")
        else:
            episode_id = auto_episode_id
            print(f"Set episode id to {episode_id} teleop enabled")
        
        with RateLoop(frequency=frequency, verbose=False) as loop:
            for i in loop:
                # Read VR controller (get raw transformation matrices)
                vr_data = vr.read_vr_controller(se3=True)
                if vr_data is None:
                    vr_data = prev_vr_data
                    continue

                # Check for recording control buttons
                if vr_data["buttons"]["B"]:
                    if (
                        prev_vr_data is not None
                        and prev_vr_data["buttons"]["B"] == False
                    ):
                        if collecting_data is True:
                            collecting_data = False
                            save_resolution = getattr(robot_interface, 'save_resolution', None)
                            save_demo(demo_data, demo_dir, episode_id, camera_names, robot_type=robot_type, yam_gripper_type=yam_gripper_type, save_resolution=save_resolution)
                            if auto_episode_id is not None:
                                auto_episode_id += 1
                            break
                        else:
                            robot_interface.set_home()
                            print(
                                "Start Collecting Data ------------------------------"
                            )
                            collecting_data = True
                            reset_data(demo_data)

                # x to create the neutral frame transformations
                if (
                    vr_data["buttons"]["X"]
                    and prev_vr_data is not None
                    and prev_vr_data["buttons"]["X"] == False
                ):
                    print("Deleting Data -----------------------------------")
                    # collecting_data = False
                    reset_data(demo_data)
                    # print("set vr neutral arm pose")
                    # for arm in arms_list:
                    #     vr_neutral_frame_delta[arm] = vr_data[arm]["T"]

                # kill the arm
                if vr_data["buttons"]["A"]:
                    break

                if vr_data["buttons"]["Y"]:
                    collecting_data = False
                    reset_data(demo_data)
                    robot_interface.set_home()
                    prev_vr_data = None

                # Update engagement states
                vr.update_engagement(vr_data["right"]["index"], "right")
                vr.update_engagement(vr_data["left"]["index"], "left")
                
                # Clear velocity limit tracking when grip released
                if vr.r_down_edge and "right" in prev_cmd_T:
                    del prev_cmd_T["right"]
                if vr.l_down_edge and "left" in prev_cmd_T:
                    del prev_cmd_T["left"]

                cmd_joint_action = np.zeros(14)
                robot_joint_action = np.zeros(14)
                cmd_eepose_action = np.zeros(14)
                
                for arm in arms_list:
                    if (arm == "left" and vr.l_engaged) or (
                        arm == "right" and vr.r_engaged
                    ):
                        rb_se3 = robot_interface.get_pose(arm, se3=True)

                        if (arm == "right" and vr.r_up_edge) or (
                            arm == "left" and vr.l_up_edge
                        ):
                            # Store VR and robot frames as 4x4 numpy arrays (ensure float64 for numerical stability)
                            vr_frame_zero_se3[arm] = np.asarray(
                                vr_data[arm]["T"], dtype=np.float64
                            )
                            robot_frame_zero_se3[arm] = np.asarray(
                                rb_se3, dtype=np.float64
                            )

                        if (
                            prev_vr_data is not None
                            and vr_data is not None
                            and arm in vr_frame_zero_se3
                            and "T" in vr_data[arm]
                        ):
                            # Compute relative transformation: delta_T = T_vr_zero^-1 @ T_vr_current
                            # This gives the transformation from vr_zero frame to vr_current frame
                            vr_zero_inv = np.linalg.inv(vr_frame_zero_se3[arm])
                            vr_current_T = np.asarray(
                                vr_data[arm]["T"], dtype=np.float64
                            )
                            delta_T = vr_zero_inv @ vr_current_T
                            
                            # Apply headset flip correction if headset is worn backwards
                            # (cameras facing user instead of away)
                            if HEADSET_FLIPPED:
                                # 180° rotation around Y axis: flip X and Z for both position and rotation
                                # Position: negate X and Z
                                delta_T[0, 3] = -delta_T[0, 3]  # flip X
                                delta_T[2, 3] = -delta_T[2, 3]  # flip Z
                                
                                # Rotation: apply 180° Y rotation to the rotation matrix
                                # R_corrected = R_180_y @ R_delta @ R_180_y.T
                                R_180_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
                                delta_T[:3, :3] = R_180_y @ delta_T[:3, :3] @ R_180_y.T
                            
                            # Apply position and rotation scaling to the delta transformation
                            # Scale translation (position)
                            delta_T[:3, 3] *= position_scale
                            
                            # Scale rotation (convert to axis-angle, scale angle, convert back)
                            if rotation_scale != 1.0:
                                delta_rot = R.from_matrix(delta_T[:3, :3])
                                rotvec = delta_rot.as_rotvec()
                                scaled_rotvec = rotvec * rotation_scale
                                delta_T[:3, :3] = R.from_rotvec(scaled_rotvec).as_matrix()
                            
                            cmd_T_raw = robot_frame_zero_se3[arm] @ delta_T

                        else:
                            cmd_T_raw = rb_se3

                        # Apply velocity limiting
                        if arm in prev_cmd_T and (max_delta_pos > 0 or max_delta_rot_rad > 0):
                            prev_T = prev_cmd_T[arm]
                            # Clamp position and rotation deltas
                            clamped_pos = clamp_delta_pos(cmd_T_raw[:3, 3] - prev_T[:3, 3], max_delta_pos)
                            R_prev = R.from_matrix(prev_T[:3, :3])
                            R_delta = R.from_matrix(cmd_T_raw[:3, :3]) * R_prev.inv()
                            clamped_rotvec = clamp_delta_rot(R_delta.as_rotvec(), max_delta_rot_rad)
                            # Build velocity-limited cmd_T
                            cmd_T = np.eye(4, dtype=np.float64)
                            cmd_T[:3, 3] = prev_T[:3, 3] + clamped_pos
                            cmd_T[:3, :3] = (R.from_rotvec(clamped_rotvec) * R_prev).as_matrix()
                        else:
                            cmd_T = cmd_T_raw
                        prev_cmd_T[arm] = cmd_T.copy()

                        gripper_pos[arm] = gripper_open - vr_data[arm]["trigger"] * gripper_width
                        # limit velocity and torque in the robot interface

                        # print(f"gripper_pos: {gripper_pos[arm]}")

                        cmd_pos[arm], cmd_quat[arm] = se3_to_xyzxyzw(cmd_T)
                        cmd_ypr = R.from_quat(cmd_quat[arm]).as_euler(
                            "ZYX", degrees=False
                        )

                        eepose_cmd = np.concatenate([cmd_pos[arm], cmd_ypr])
                        
                        try:
                            solved_joints = robot_interface.solve_ik(eepose_cmd[:6], arm)
                        except Exception as e:
                            print(f"[WARN] IK failed for arm {arm}: {e}")
                            # Skip commanding this arm for this iteration; wait for next VR input
                            continue
                        if solved_joints is not None:
                            cmd_joints[arm] = solved_joints
                            # normalize gripper values
                            cmd_joints[arm] = np.concatenate(
                                [cmd_joints[arm], [gripper_pos[arm]]]
                            )

                        robot_interface.set_joints(cmd_joints[arm], arm)

                        if collecting_data:
                            arm_offset = 0
                            if arm == "right":
                                arm_offset = 7
                            if arm in cmd_pos and arm in cmd_quat:
                                cmd_eepose_action[arm_offset : arm_offset + 3] = (
                                    cmd_pos[arm]
                                )
                                cmd_eepose_action[arm_offset + 3 : arm_offset + 6] = (
                                    R.from_quat(cmd_quat[arm]).as_euler(
                                        "ZYX", degrees=False
                                    )
                                )  # ypr convention
                                # Normalize gripper value to 0-1 for saved data
                                cmd_eepose_action[arm_offset + 6] = (gripper_pos[arm] - gripper_close) / gripper_width

                            if arm in cmd_joints:
                                cmd_joint_action[arm_offset : arm_offset + 7] = (
                                    cmd_joints[arm]
                                )

                # Record data after processing all arms for this timestep
                if collecting_data:
                    obs = robot_interface.get_obs()
                    
                    # Use real proprioception from both arms (read_all_arms=True)
                    # robot_joint_action now contains real joint positions for ALL arms
                    robot_joint_action = obs["joint_positions"].copy()

                    obs_copy = {}
                    for key, val in obs.items():
                        obs_copy[key] = (
                            None if val is None else val.copy()
                        )  # NumPy copy
                    demo_data["obs"].append(obs_copy)
                    demo_data["cmd_joint_actions"].append(cmd_joint_action.copy())
                    demo_data["robot_joint_actions"].append(robot_joint_action)
                    demo_data["cmd_eepose_actions"].append(cmd_eepose_action.copy())

                if vr_data is not None:
                    prev_vr_data = vr_data


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
    description="Collect robot demonstrations using VR controller"
  )
  parser.add_argument(
    "--arms",
    type=str,
    default="right",
    choices=["left", "right", "both"],
    help="Which arm(s) to control",
  )
  parser.add_argument(
    "--frequency",
    type=float,
    default=DEFAULT_FREQUENCY,
    help="Control loop frequency in Hz",
  )
  parser.add_argument(
    "--demo-dir",
    type=str,
    default=DEMO_DIR,
    help="Directory to save demos",
  )
  parser.add_argument(
    "--calibrate",
    action="store_true",
    help="Run VR controller orientation calibration before teleop",
  )
  parser.add_argument(
    "--auto-episode-start",
    type=int,
    default=None,
    help="If set, start at this episode id and auto-increment on each recording",
  )
  parser.add_argument(
    "--robot-type",
    type=str,
    default="arx",
    choices=["arx", "yam"],
    help="Robot type to use: 'arx' for ARX X5, 'yam' for I2RT YAM",
  )
  parser.add_argument(
    "--yam-gripper-type",
    type=str,
    default="linear_4310",
    choices=["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"],
    help="Gripper type for YAM robot (only used if --robot-type=yam)",
  )
  parser.add_argument(
    "--yam-left-can",
    type=str,
    default="can0",
    help="CAN interface for YAM left arm (only used if --robot-type=yam)",
  )
  parser.add_argument(
    "--yam-right-can",
    type=str,
    default="can1",
    help="CAN interface for YAM right arm (only used if --robot-type=yam)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Dry run mode: simulate VR teleop without actuating the robot",
  )
  parser.add_argument(
    "--position-scale",
    type=float,
    default=POSITION_SCALE,
    help=f"Scale factor for VR position deltas (default: {POSITION_SCALE})",
  )
  parser.add_argument(
    "--rotation-scale",
    type=float,
    default=ROTATION_SCALE,
    help=f"Scale factor for VR rotation deltas (default: {ROTATION_SCALE})",
  )
  parser.add_argument(
    "--headset-flipped",
    action="store_true",
    default=HEADSET_FLIPPED,
    help="Enable if headset is worn backwards (cameras facing user). Flips X/Z axes.",
  )
  parser.add_argument(
    "--no-headset-flipped",
    action="store_true",
    help="Disable headset flip correction (normal headset orientation)",
  )

  args = parser.parse_args()
  raise ValueError("Use Collect Demo 2 instead")

  # Handle headset flip flag
  if args.no_headset_flipped:
    HEADSET_FLIPPED = False
  else:
    HEADSET_FLIPPED = args.headset_flipped
  
  # Print configuration summary
  print("\n" + "="*60)
  print("TELEOP CONFIGURATION")
  print("="*60)
  print(f"Robot type:       {args.robot_type}")
  print(f"Arms:             {args.arms}")
  print(f"Frequency:        {args.frequency} Hz")
  print(f"Demo directory:   {args.demo_dir}")
  print(f"Position scale:   {args.position_scale}")
  print(f"Rotation scale:   {args.rotation_scale}")
  print(f"Headset flipped:  {HEADSET_FLIPPED}")
  print(f"Dry run:          {args.dry_run}")
  if args.robot_type == "yam":
    print(f"YAM gripper type: {args.yam_gripper_type}")
    print(f"YAM left CAN:     {args.yam_left_can}")
    print(f"YAM right CAN:    {args.yam_right_can}")
  print("="*60 + "\n")

  if args.calibrate:
    # Import here to avoid dependency if user never calibrates
    from egomimic.robot.calibrate_utils import (
      calibrate_right_controller,
      calibrate_left_controller,
    )

    print("Running VR controller calibration...")
    # Override globals based on which arms are used
    if args.arms in ("right", "both"):
      print("\nCalibrating RIGHT controller...")
      R_off_right = calibrate_right_controller()
      # overwrite module-level constant
      R_YPR_OFFSET = R_off_right

    if args.arms in ("left", "both"):
      print("\nCalibrating LEFT controller...")
      R_off_left = calibrate_left_controller()
      # overwrite module-level constant
      L_YPR_OFFSET = R_off_left

    print("Calibration finished. Using updated offsets for this run.\n")

  # Build YAM interface mapping if using YAM robot
  yam_interfaces = None
  if args.robot_type == "yam":
    yam_interfaces = {
      "left": args.yam_left_can,
      "right": args.yam_right_can,
    }
    print(f"YAM CAN interfaces: {yam_interfaces}")
    print(f"YAM gripper type: {args.yam_gripper_type}")

  collect_demo(
    arms_to_collect=args.arms,
    frequency=args.frequency,
    demo_dir=args.demo_dir,
    auto_episode_start=args.auto_episode_start,
    robot_type=args.robot_type,
    yam_gripper_type=args.yam_gripper_type,
    yam_interfaces=yam_interfaces,
    dry_run=args.dry_run,
    position_scale=args.position_scale,
    rotation_scale=args.rotation_scale,
  )
