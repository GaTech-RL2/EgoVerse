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
from robot_interface import ARXInterface
from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver


# ------------------------- Configuration -------------------------

# Control parameters
DEFAULT_FREQUENCY = 30.0  # Hz
POSITION_SCALE = 1.0  # Scale factor for position deltas
ROTATION_SCALE = 1.0  # Scale factor for rotation deltas

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

# Gripper thresholds
GRIPPER_OPEN_VALUE = 0.08
GRIPPER_CLOSE_VALUE = -0.018
GRIPPER_WIDTH = GRIPPER_OPEN_VALUE - GRIPPER_CLOSE_VALUE
GRIPPER_VEL = 1  # m/s gripper width is normally around 0.08m

# Demo recording
DEMO_DIR = "./demos"
MAX_DEMO_LENGTH = 10000  # Maximum number of steps per demo


# ------------------------- Helper Functions -------------------------


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


def save_demo(demo_data: dict, demo_dir, episode_id: int, cam_names):
    data_dict = dict()
    """Save demo to HDF5 file."""
    filename = demo_dir / f"demo_{episode_id}.hdf5"

    for cam_name in cam_names:
        image_list = []
        for i in range(len(demo_data["obs"])):
            img = demo_data["obs"][i][cam_name]
            if img is None:
                continue
            img_rgb = img[..., ::-1]
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
    data_dict["/actions/eepose"] = np.array(demo_data["cmd_eepose_actions"])
    data_dict["/actions/joints"] = np.array(demo_data["cmd_joint_actions"])
    data_dict["/action"] = np.array(demo_data["cmd_joint_actions"])

    kinematics_solver = EvaMinkKinematicsSolver(
        model_path="/home/robot/robot_ws/egomimic/resources/model_x5.xml"
    )
    robot_ee_pose = []
    for i in range(len(demo_data["robot_joint_actions"])):
        robot_joint_action = demo_data["robot_joint_actions"][i]
        left_joints = robot_joint_action[:7]
        right_joints = robot_joint_action[7:]
        # check if left is not 0 array
        if not np.allclose(left_joints, 0):
            left_ee_xyz, left_ee_rot = kinematics_solver.fk(left_joints)
            left_ee_ypr = left_ee_rot.as_euler("ZYX", degrees=False)
        else:
            left_ee_xyz = np.zeros(3)
            left_ee_ypr = np.zeros(3)
        if not np.allclose(right_joints, 0):
            right_ee_xyz, right_ee_rot = kinematics_solver.fk(right_joints)
            right_ee_ypr = right_ee_rot.as_euler("ZYX", degrees=False)
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
            _ = image.create_dataset(
                cam_name,
                (max_timesteps, 480, 640, 3),
                dtype="uint8",
                chunks=(1, 480, 640, 3),
            )
        _ = obs.create_dataset("joints", (max_timesteps, 14))
        # _ = obs.create_dataset("qjointvel", (max_timesteps, 16))
        _ = obs.create_dataset("eepose", (max_timesteps, 14))
        _ = obs.create_dataset("joint_positions", (max_timesteps, 14))
        _ = root.create_group("actions")
        _ = root["actions"].create_dataset("eepose", (max_timesteps, 14))
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
):
    """
    Collect demonstrations using VR controller.

    Args:
        arms: Which arm(s) to control ("left", "right", or "both")
        frequency: Control loop frequency in Hz
        demo_dir: Directory to save demos
    """
    # Setup demo directory
    demo_dir = Path(demo_dir)
    demo_dir.mkdir(exist_ok=True, parents=True)

    # Initialize VR interface
    vr = VRInterface()
    prev_vr_data = None

    # Initialize robot interfaces (one per arm)
    # robot_interface = SingleARXInterface(arm=arms_to_collect)
    if arms_to_collect == "both":
        arms = ["right", "left"]
    elif arms_to_collect == "right":
        arms = ["right"]
    elif arms_to_collect == "left":
        arms = ["left"]
    else:
        raise ValueError("Invalid arm values inputted.")
    robot_interface = ARXInterface(arms=arms)

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
                # print(f"vr_data: {vr_data}")
                if vr_data is None:
                    # print("Not reading vr data using prev")
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
                            save_demo(demo_data, demo_dir, episode_id, camera_names)
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

                cmd_joint_action = np.zeros(14)
                robot_joint_action = np.zeros(14)
                cmd_eepose_action = np.zeros(14)
                for arm in arms_list:
                    if (arm == "left" and vr.l_engaged) or (
                        arm == "right" and vr.r_engaged
                    ):
                        rb_se3 = robot_interface.get_pose(
                            arm, se3=True
                        )  # TODO need to fix the logic for double arm
                        # print(f"rb_pos {R.from_matrix(rb_rot).as_euler('ZYX', degrees=False)}")

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
                            cmd_T = robot_frame_zero_se3[arm] @ delta_T

                        else:
                            cmd_T = rb_se3

                        gripper_pos[arm] = GRIPPER_OPEN_VALUE -vr_data[arm]["trigger"] * (GRIPPER_WIDTH)
                        # limit velocity and torque in the robot interface

                        # print(f"gripper_pos: {gripper_pos[arm]}")

                        cmd_pos[arm], cmd_quat[arm] = se3_to_xyzxyzw(cmd_T)
                        cmd_ypr = R.from_quat(cmd_quat[arm]).as_euler(
                            "ZYX", degrees=False
                        )

                        eepose_cmd = np.concatenate([cmd_pos[arm], cmd_ypr])
                        # print(f"eepose: {eepose_cmd}")
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

                        # VELOCITY_LIMIT can be done in the interface
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
                                cmd_eepose_action[arm_offset + 6] = (gripper_pos[arm] - GRIPPER_CLOSE_VALUE) / GRIPPER_WIDTH

                            if arm in cmd_joints:
                                cmd_joint_action[arm_offset : arm_offset + 7] = (
                                    cmd_joints[arm]
                                )

                            robot_joint_action[arm_offset : arm_offset + 7] = (
                                robot_interface.get_joints(arm)
                            )

                        if collecting_data:
                            obs = robot_interface.get_obs()

                            obs_copy = {}
                            for key, val in obs.items():
                                obs_copy[key] = (
                                    None if val is None else val.copy()
                                )  # NumPy copy
                            demo_data["obs"].append(obs_copy)
                            demo_data["cmd_joint_actions"].append(cmd_joint_action)
                            demo_data["robot_joint_actions"].append(robot_joint_action)
                            demo_data["cmd_eepose_actions"].append(cmd_eepose_action)

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

  args = parser.parse_args()

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

  collect_demo(
    arms_to_collect=args.arms,
    frequency=args.frequency,
    demo_dir=args.demo_dir,
    auto_episode_start=args.auto_episode_start,
  )
