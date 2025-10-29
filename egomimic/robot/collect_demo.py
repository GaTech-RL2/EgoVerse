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
import h5py
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# Add path to oculus_reader if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "../../external"))
from oculus_reader import OculusReader

# Import local modules
from robot_utils import RateLoop

# Add path to robot_interface
sys.path.append(os.path.join(os.path.dirname(__file__), "eva/eva_ws/src/eva"))
from robot_interface import SingleARXInterface


# ------------------------- Configuration -------------------------

# Control parameters
DEFAULT_FREQUENCY = 30.0  # Hz
POSITION_SCALE = 1.0  # Scale factor for position deltas
ROTATION_SCALE = 1.0  # Scale factor for rotation deltas

# Dead-zone thresholds (to filter out jitter)
POS_DEAD_ZONE = 0.002  # meters
ROT_DEAD_ZONE_RAD = np.deg2rad(0.8)  # radians

YPR_OFFSET = [0, 0, 0]

# Trigger thresholds for engagement detection
TRIGGER_ON_THRESHOLD = 0.8
TRIGGER_OFF_THRESHOLD = 0.2

# Gripper thresholds
GRIPPER_OPEN_VALUE = 0.08
GRIPPER_CLOSE_VALUE = 0.0
GRIPPER_VEL = 0.08  # m/s gripper width is normally around 0.08m

# Demo recording
DEMO_DIR = "./demos"
MAX_DEMO_LENGTH = 10000  # Maximum number of steps per demo


# ------------------------- Helper Functions -------------------------


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

        # Gripper state
        self.r_gripper_closed = False
        self.l_gripper_closed = False
        self.r_gripper_value = GRIPPER_OPEN_VALUE
        self.l_gripper_value = GRIPPER_OPEN_VALUE

        print("VR Interface initialized!")

    def update_engagement(self, trigger_value: float, was_engaged: bool) -> bool:
        """Update engagement state with hysteresis."""
        if not was_engaged and trigger_value > TRIGGER_ON_THRESHOLD:
            return True
        if was_engaged and trigger_value < TRIGGER_OFF_THRESHOLD:
            return False
        return was_engaged

    def read_vr_controller(self):
        """Read VR controller state and return parsed data."""
        sample = self.device.get_transformations_and_buttons()
        if not sample:
            return None

        transforms, buttons = sample
        if not transforms:
            return None

        # Extract transforms
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

        # Apply ypr offset
        # zero = np.zeros(3)
        # _, l_quat_cur = apply_delta_pose(
        #     l_pos_cur,
        #     l_quat_cur,
        #     zero,
        #     R.from_euler("ZYX", YPR_OFFSET, degrees=False).as_quat(),
        # )
        # _, r_quat_cur = apply_delta_pose(
        #     r_pos_cur,
        #     r_quat_cur,
        #     zero,
        #     R.from_euler("ZYX", YPR_OFFSET, degrees=False).as_quat(),
        # )

        # Extract button/trigger values
        trig_l = get_analog(buttons, ["leftTrig", "LT", "trigger_l"], 0.0)
        trig_r = get_analog(buttons, ["rightTrig", "RT", "trigger_r"], 0.0)
        idx_l = get_analog(
            buttons, ["leftIndex", "IndexL", "indexL", "index_l", "leftPinch"], trig_l
        )
        idx_r = get_analog(
            buttons, ["rightIndex", "IndexR", "indexR", "index_r", "rightPinch"], trig_r
        )

        # Get buttons
        btn_a = bool(buttons.get("A", False))
        btn_b = bool(buttons.get("B", False))
        btn_x = bool(buttons.get("X", False))

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
            "buttons": {
                "A": btn_a,
                "B": btn_b,
                "X": btn_x,
            },
        }


# ------------------------- Demo Recording Helpers -------------------------


def save_demo(demo_data: dict, demo_dir: Path):
    """Save demo to HDF5 file."""
    if not demo_data["observations"]:
        print("No data to save!")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = demo_dir / f"demo_{timestamp}.hdf5"

    print(f"Saving demo with {len(demo_data['observations'])} steps to {filename}")

    with h5py.File(filename, "w") as f:
        # Create groups
        obs_grp = f.create_group("observations")
        action_grp = f.create_group("actions")

        # Save observations
        for key in demo_data["observations"][0].keys():
            data = np.array([obs[key] for obs in demo_data["observations"]])
            obs_grp.create_dataset(key, data=data)

        # Save actions
        for key in demo_data["actions"][0].keys():
            data = np.array([act[key] for act in demo_data["actions"]])
            action_grp.create_dataset(key, data=data)

    print("Demo saved successfully!")


# ------------------------- Main Entry Point -------------------------


def collect_demo(
    arms_to_collect: str = "right",
    frequency: float = DEFAULT_FREQUENCY,
    demo_dir: str = DEMO_DIR,
    recording: bool = True,
):
    """
    Collect demonstrations using VR controller.

    Args:
        arms: Which arm(s) to control ("left", "right", or "both")
        frequency: Control loop frequency in Hz
        demo_dir: Directory to save demos
    """
    # Setup demo directory
    demo_path = Path(demo_dir)
    demo_path.mkdir(exist_ok=True, parents=True)

    # Initialize VR interface
    vr = VRInterface()
    prev_vr_data = None

    # Initialize robot interfaces (one per arm)
    robot_interface = SingleARXInterface(arm=arms_to_collect)

    arms_list = []
    if arms_to_collect == "both" or arms_to_collect == "right":
        arms_list.append("right")
    if arms_to_collect == "both" or arms_to_collect == "left":
        arms_list.append("left")

    print("Homing robots...")
    robot_interface.set_home()
    # Home all robots
    time.sleep(2.0)
    print("Robots ready!")

    # Demo recording state
    demo_data = {"observations": [], "actions": []}

    print("\n" + "=" * 60)
    print("Starting demo collection")
    print("=" * 60)
    print("Controls:")
    print("  - Hold RIGHT TRIGGER to engage and control right arm")
    print("  - Hold LEFT TRIGGER to engage and control left arm")
    print("  - Squeeze INDEX to close gripper")
    print("  - Press B to SAVE current demo")
    print("  - Press X to DISCARD current demo")
    print("  - Press Ctrl+C to exit")
    print("=" * 60 + "\n")
    cmd_pos = None
    try:
        with RateLoop(frequency=frequency, verbose=False) as loop:
            for i in loop:
                # Read VR controller
                vr_data = vr.read_vr_controller()
                if vr_data is None:
                    print("Not reading vr data using prev")
                    vr_data = prev_vr_data

                if vr_data is None:
                    print("VR data is None")
                    continue
                # Check for recording control buttons
                if vr_data["buttons"]["B"]:
                    save_demo(demo_data, demo_path)
                    del demo_data
                    demo_data = {"observations": [], "actions": []}

                # x to delete data
                if vr_data["buttons"]["X"]:
                    del demo_data
                    demo_data = {"observations": [], "actions": []}

                # Update engagement states
                vr.r_engaged = vr.update_engagement(
                    vr_data["right"]["trigger"], vr.r_engaged
                )
                vr.l_engaged = vr.update_engagement(
                    vr_data["left"]["trigger"], vr.l_engaged
                )

                # TODO: implement control logic

                for arm in arms_list:
                    if (arm == "left" and vr.l_engaged) or (
                        arm == "right" and vr.r_engaged
                    ):
                        rb_pos, rb_R = robot_interface.get_pose()
                        rb_quat_xyzw = rb_R.as_quat()
                        rb_joint = robot_interface.get_joints()
                        gripper_pos = rb_joint[6]
                        if prev_vr_data is not None and vr_data is not None:
                            delta_pos, delta_quat_xyzw = compute_delta_pose(
                                arm, vr_data, prev_vr_data[arm]["pos"], rb_R.as_quat()
                            )
                            delta_pos *= 3

                            # can limit angle and pos change speed here also (Ask Danfei is it appropriate to have it here and robot_interface)
                            # if np.linalg.norm(delta_pos) < POS_DEAD_ZONE:
                            #     delta_pos[:] = 0.0
                            # delta_pos[:] = 0.0

                            cmd_pos, cmd_quat = apply_delta_pose(
                                rb_pos, rb_quat_xyzw, delta_pos, delta_quat_xyzw
                            )
                        else:
                            cmd_pos, cmd_R = robot_interface.get_pose()
                            cmd_quat = cmd_R.as_quat()

                        # gripper
                        vr_index = vr_data[arm]["index"]
                        gripper_delta = GRIPPER_VEL / frequency
                        if vr_index:
                            # close gripper
                            gripper_delta *= -1

                        # joint limit can also be done here
                        gripper_pos = np.clip(
                            gripper_pos + gripper_delta,
                            GRIPPER_CLOSE_VALUE,
                            GRIPPER_OPEN_VALUE,
                        )
                        cmd_ypr = R.from_quat(cmd_quat).as_euler("ZYX", degrees=False)

                        eepose_cmd = np.concatenate([cmd_pos, cmd_ypr, [gripper_pos]])

                        # VELOCITY_LIMIT can be done in the interface
                        robot_interface.set_pose(eepose_cmd)

                if vr_data is not None:
                    prev_vr_data = vr_data

                # Print status every second
                if i % int(frequency / 8) == 0:
                    # print(
                    #     f"Step {i}, Demo length: {len(demo_data['observations'])}, "
                    #     f"Engaged: R={vr.r_engaged}, L={vr.l_engaged}"
                    # )
                    pass
                    # print(f"Cmd pos {cmd_pos}")
                    # r_pos = vr_data["right"]["pos"]
                    # l_pos = vr_data["left"]["pos"]
                    # print(f"r pos {r_pos}, l pos {}")  # need this for debug

                # Auto-save if demo gets too long
                # if len(demo_data["observations"]) >= MAX_DEMO_LENGTH:
                #     print(
                #         f"Demo reached max length ({MAX_DEMO_LENGTH}), auto-saving..."
                #     )
                #     save_demo(demo_data, demo_path)
                #     demo_data = {"observations": [], "actions": []}

    except KeyboardInterrupt:
        print("\n\nStopping demo collection...")
        if demo_data["observations"]:
            print("Saving current demo before exit...")
            save_demo(demo_data, demo_path)


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
        "--demo-dir", type=str, default=DEMO_DIR, help="Directory to save demos"
    )

    args = parser.parse_args()

    collect_demo(
        arms_to_collect=args.arms,
        frequency=args.frequency,
        demo_dir=args.demo_dir,
    )
