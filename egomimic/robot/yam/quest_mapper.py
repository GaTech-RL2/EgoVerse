"""Clutch-relative Quest → robot-base pose mapping, adapted from gello Quest_Agent.

One OculusReader, two independent clutches: left grip moves the left arm,
right grip moves the right arm. Controller poses are Quest tracking / world
space (X-right, Y-up, -Z-forward). Deltas stay in that frame, then separate
bases map them into each arm's base. Position uses the Panda RAIL layout.
Orientation uses the same basis (``orientation_rx_degrees=0``) now that the
APK logs tracking/world poses; ``90`` was for the old headset-relative APK.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# OculusReader (tracking/world): +X right, +Y up, -Z forward.
# Map onto a robot base with +X forward, +Y left, +Z up (same as the Panda
# layout in gello). Set headset_yaw_degrees=180 if the operator faces the
# robot across a table. This yaw is a room-to-robot offset, not live headset
# heading — so turning your head does not move a still controller.
RAIL_HEADSET_TO_ROBOT = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)

SIDE_BUTTONS = {
    "l": {
        "pose": "l",
        "grip": ("LG", "leftGrip"),
        "trigger": ("LTr", "leftTrig"),
        "joy": "LJ",
    },
    "r": {
        "pose": "r",
        "grip": ("RG", "rightGrip"),
        "trigger": ("RTr", "rightTrig"),
        "joy": "RJ",
    },
}


def headset_to_robot_basis(yaw_degrees=0.0):
    yaw = Rotation.from_euler("y", float(yaw_degrees), degrees=True).as_matrix()
    return RAIL_HEADSET_TO_ROBOT @ yaw


def orientation_axes_to_robot(yaw_degrees=0.0, rx_degrees=0.0):
    """Headset-to-robot basis for rotations.

    With the world-frame APK this matches the Panda RAIL mapping (``rx=0``).
    ``90`` was the YAM correction for the old headset-relative APK; ``-90``
    maps the same axes but inverts yaw and roll; ``180`` flips sense only.
    """
    return headset_to_robot_basis(yaw_degrees) @ Rotation.from_euler(
        "x", float(rx_degrees), degrees=True
    ).as_matrix()


def is_valid_pose_matrix(pose):
    try:
        pose = np.asarray(pose, dtype=float)
    except (TypeError, ValueError):
        return False
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return False
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-3):
        return False
    rotation = pose[:3, :3]
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
    )


def retarget_pose(
    controller_init_pos,
    controller_init_rot,
    controller_curr_pos,
    controller_curr_rot,
    ee_init_pos,
    ee_init_rot,
    position_axes,
    orientation_axes=None,
    translation_gain=1.0,
    orientation_gain=1.0,
):
    """Map clutch-relative controller motion onto the robot EE (base frame)."""
    if orientation_axes is None:
        orientation_axes = position_axes
    controller_delta_pos = controller_curr_pos - controller_init_pos
    controller_delta_rot = controller_curr_rot @ controller_init_rot.T
    robot_delta_rot = orientation_axes @ controller_delta_rot @ orientation_axes.T
    if orientation_gain != 1.0:
        rotvec = Rotation.from_matrix(robot_delta_rot).as_rotvec()
        robot_delta_rot = Rotation.from_rotvec(rotvec * orientation_gain).as_matrix()
    target_pos = ee_init_pos + translation_gain * (position_axes @ controller_delta_pos)
    target_rot = robot_delta_rot @ ee_init_rot
    T = np.eye(4)
    T[:3, :3] = target_rot
    T[:3, 3] = target_pos
    return T


def _input_active(value):
    if isinstance(value, (tuple, list, np.ndarray)):
        return len(value) > 0 and float(value[0]) > 0.5
    return bool(value)


def _input_value(value):
    if isinstance(value, (tuple, list, np.ndarray)):
        if len(value) == 0:
            return 0.0
        value = value[0]
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _button_active(buttons, digital_key, analog_key):
    analog = _input_value(buttons.get(analog_key, buttons.get(digital_key, 0.0)))
    return _input_active(buttons.get(digital_key, False)) or analog > 0.5, analog


def _nlerp_rot(r0, r1, a):
    q0 = Rotation.from_matrix(r0).as_quat()
    q1 = Rotation.from_matrix(r1).as_quat()
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = (1.0 - a) * q0 + a * q1
    n = np.linalg.norm(q)
    if n < 1e-9:
        return r0
    return Rotation.from_quat(q / n).as_matrix()


class _SideClutch:
    def __init__(
        self,
        side,
        position_axes,
        orientation_axes,
        translation_gain,
        orientation_gain,
        gripper_open,
        gripper_closed,
        target_lpf=0.4,
        max_controller_step_m=0.08,
        max_controller_step_deg=40.0,
    ):
        self.side = side
        self.keys = SIDE_BUTTONS[side]
        self.position_axes = np.asarray(position_axes, dtype=float)
        self.orientation_axes = np.asarray(orientation_axes, dtype=float)
        self.translation_gain = float(translation_gain)
        self.orientation_gain = float(orientation_gain)
        self.gripper_open = float(gripper_open)
        self.gripper_closed = float(gripper_closed)
        self.target_lpf = float(np.clip(target_lpf, 0.0, 1.0))
        self.max_controller_step_m = float(max_controller_step_m)
        self.max_controller_step_rad = np.deg2rad(float(max_controller_step_deg))
        self.engaged = False
        self._grip_was_down = False
        self._joy_was_down = False
        self._need_anchor = True
        self._ctrl_init_pos = np.zeros(3)
        self._ctrl_init_rot = np.eye(3)
        self._ee_init_pos = np.zeros(3)
        self._ee_init_rot = np.eye(3)
        self._last_pose = np.eye(4)
        self._last_gripper = self.gripper_open
        self._last_ctrl_pos = None
        self._last_ctrl_rot = None
        self._filt_pos = None
        self._filt_rot = None

    def reset(self):
        """Drop the clutch so the next grip (or still-held grip) re-anchors."""
        self.engaged = False
        self._grip_was_down = False
        self._need_anchor = True
        self._filt_pos = None
        self._filt_rot = None
        self._last_ctrl_pos = None
        self._last_ctrl_rot = None

    def _gripper_from_trigger(self, trigger_value):
        return self.gripper_open + trigger_value * (self.gripper_closed - self.gripper_open)

    def _idle(self, tracked, gripper, teleport=False):
        self.engaged = False
        return {
            "side": self.side,
            "tracked": tracked,
            "engaged": False,
            "teleport": teleport,
            "target_pose": self._last_pose.copy(),
            "gripper": gripper,
        }

    def _remember_controller(self, pos, rot):
        self._last_ctrl_pos = pos.copy()
        self._last_ctrl_rot = rot.copy()

    def _controller_jumped(self, pos, rot):
        """True if this tick's Quest pose is a tracking teleport, not a real hand motion."""
        if self._last_ctrl_pos is None or not self.engaged:
            return False
        dp = float(np.linalg.norm(pos - self._last_ctrl_pos))
        if self.max_controller_step_m > 0.0 and dp > self.max_controller_step_m:
            return True
        if self.max_controller_step_rad <= 0.0:
            return False
        dR = rot @ self._last_ctrl_rot.T
        dang = float(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec()))
        return dang > self.max_controller_step_rad

    def _smooth(self, T):
        pos, rot = T[:3, 3], T[:3, :3]
        a = self.target_lpf
        if a >= 1.0 or self._filt_pos is None:
            self._filt_pos = pos.copy()
            self._filt_rot = rot.copy()
            return T
        self._filt_pos = a * pos + (1.0 - a) * self._filt_pos
        self._filt_rot = _nlerp_rot(self._filt_rot, rot, a)
        out = np.eye(4)
        out[:3, :3] = self._filt_rot
        out[:3, 3] = self._filt_pos
        return out

    def update(self, poses, buttons, ee_pose):
        pose_key = self.keys["pose"]
        raw = poses.get(pose_key) if isinstance(poses, dict) else None
        tracked = is_valid_pose_matrix(raw)
        grip_down, _ = _button_active(buttons, *self.keys["grip"])
        _, trig_value = _button_active(buttons, *self.keys["trigger"])
        joy_down = _input_active(buttons.get(self.keys["joy"], False))
        gripper = self._gripper_from_trigger(trig_value)

        if not tracked:
            self._grip_was_down = False
            self._need_anchor = True
            self._filt_pos = None
            self._last_ctrl_pos = None
            self._last_ctrl_rot = None
            return self._idle(False, gripper)

        pose = np.asarray(raw, dtype=float)
        ctrl_pos, ctrl_rot = pose[:3, 3], pose[:3, :3]
        ee_pose = np.asarray(ee_pose, dtype=float)

        if self._controller_jumped(ctrl_pos, ctrl_rot):
            self._grip_was_down = False
            self._need_anchor = True
            self._filt_pos = None
            self._remember_controller(ctrl_pos, ctrl_rot)
            return self._idle(True, gripper, teleport=True)

        self._remember_controller(ctrl_pos, ctrl_rot)

        if joy_down and not self._joy_was_down:
            self._need_anchor = True
        self._joy_was_down = joy_down

        if grip_down and not self._grip_was_down:
            self._need_anchor = True
        self._grip_was_down = grip_down

        if not grip_down:
            self._need_anchor = True
            self._filt_pos = None
            self._last_gripper = gripper
            return self._idle(True, gripper)

        if self._need_anchor:
            self._ctrl_init_pos = ctrl_pos.copy()
            self._ctrl_init_rot = ctrl_rot.copy()
            self._ee_init_pos = ee_pose[:3, 3].copy()
            self._ee_init_rot = ee_pose[:3, :3].copy()
            self._last_pose = ee_pose.copy()
            self._filt_pos = None
            self._need_anchor = False

        target = self._smooth(
            retarget_pose(
                self._ctrl_init_pos,
                self._ctrl_init_rot,
                ctrl_pos,
                ctrl_rot,
                self._ee_init_pos,
                self._ee_init_rot,
                self.position_axes,
                orientation_axes=self.orientation_axes,
                translation_gain=self.translation_gain,
                orientation_gain=self.orientation_gain,
            )
        )
        self.engaged = True
        self._last_pose = target
        self._last_gripper = gripper
        return {
            "side": self.side,
            "tracked": True,
            "engaged": True,
            "teleport": False,
            "target_pose": target,
            "gripper": gripper,
        }


class QuestBimanualMapper:
    def __init__(
        self,
        translation_gain=0.45,
        orientation_gain=0.4,
        headset_yaw_degrees=0.0,
        gripper_open=1.0,
        gripper_closed=0.0,
        side_yaw=None,
        sides=("l", "r"),
        target_lpf=0.4,
        orientation_rx_degrees=0.0,
        max_controller_step_m=0.08,
        max_controller_step_deg=40.0,
    ):
        side_yaw = side_yaw or {}
        self._sides = {}
        for side in sides:
            if side not in SIDE_BUTTONS:
                raise ValueError(f"controller side must be 'l' or 'r', got {side!r}")
            yaw = side_yaw.get(side, headset_yaw_degrees)
            self._sides[side] = _SideClutch(
                side,
                headset_to_robot_basis(yaw),
                orientation_axes_to_robot(yaw, orientation_rx_degrees),
                translation_gain,
                orientation_gain,
                gripper_open,
                gripper_closed,
                target_lpf=target_lpf,
                max_controller_step_m=max_controller_step_m,
                max_controller_step_deg=max_controller_step_deg,
            )

    def reset_clutches(self):
        for clutch in self._sides.values():
            clutch.reset()

    def will_clutch(self, side, buttons):
        """True if this side is idle and the grip is down (clutch rising edge this tick)."""
        clutch = self._sides.get(side)
        if clutch is None or clutch.engaged:
            return False
        down, _ = _button_active(buttons or {}, *clutch.keys["grip"])
        return down

    def update(self, poses, buttons, ee_poses):
        """Return ``{'l': cmd, 'r': cmd}``. ``ee_poses`` is side → 4x4 FK pose."""
        poses = poses or {}
        buttons = buttons or {}
        out = {}
        for side, clutch in self._sides.items():
            out[side] = clutch.update(poses, buttons, ee_poses[side])
        return out
