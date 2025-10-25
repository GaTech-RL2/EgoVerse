#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Bool, Int8, Float32
from oculus_reader import OculusReader

# ------------------------- config -------------------------

# [x, y, z, yaw(Z), pitch(Y), roll(X)]
RINIT_POSE = [
    0.12769999980926514,
    -0.0004995886120013893,
    -0.22650001168251038,
    0,
    0,
    0,
]
LINIT_POSE = RINIT_POSE

# Orientation offsets to apply (yaw, pitch, roll) in radians
ROFFSET_POSE = [0.0, 0.0, 0.0, -0.45, -0.8, 0.4]
LOFFSET_POSE = [0.0, 0.0, 0.0, -0.45, -0.8, 0.4]

VR_PREFIX = "/vr"
FRAME_ID = "map"

# Homing speed limits
MAX_LIN_VEL = 0.2  # m/s
MAX_ANG_VEL = 1.0  # rad/s
POS_EPS = 1e-4
ANG_EPS = 1e-3

# Delta thresholds
POS_DEAD = 0.002  # m dead-zone for jitter (2 mm)
ROT_DEAD_RAD = np.deg2rad(0.8)

# Gripper interpolation config (0 -> open=5.0, 1 -> close=0.0)
GRIP_OPEN_POS = 5.0
GRIP_CLOSE_POS = 0.0
GRIP_MAX_VEL = 5.0  # units/s interpolation speed


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quat_xyzw_to_wxyz(qxyzw: np.ndarray) -> np.ndarray:
    return np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]], dtype=np.float64)


def quat_wxyz_to_xyzw(qwxyz: np.ndarray) -> np.ndarray:
    return np.array([qwxyz[1], qwxyz[2], qwxyz[3], qwxyz[0]], dtype=np.float64)


def euler_zyx_to_quat_wxyz(yaw: float, pitch: float, roll: float) -> np.ndarray:
    q_xyzw = R.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    return quat_xyzw_to_wxyz(q_xyzw)


def pose_from_T(T: np.ndarray):
    pos = T[:3, 3].astype(np.float64)
    q_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
    return pos, q_wxyz


def _get_analog(buttons: dict, keys, default=0.0) -> float:
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


def _apply_offset(q_wxyz: np.ndarray, off_pose: list) -> np.ndarray:
    # q_wxyz: [w,x,y,z]; off_pose[3:6] = [yaw_off, pitch_off, roll_off] (radians)
    q_xyzw_in = quat_wxyz_to_xyzw(q_wxyz)
    r_in = R.from_quat(q_xyzw_in)  # current
    r_off = R.from_euler("ZYX", [off_pose[3], off_pose[4], off_pose[5]], degrees=False)
    r_out = r_in * r_off.inv()  # subtract offsets
    return quat_xyzw_to_wxyz(r_out.as_quat())


def _controller_to_internal(pos_xyz: np.ndarray, q_wxyz: np.ndarray):
    # Fixed transforms (as in your original)
    A = np.array(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
    )
    B = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    M = B @ A

    R_c = R.from_quat(quat_wxyz_to_xyzw(q_wxyz)).as_matrix()
    pos_i = M @ pos_xyz
    R_i = M @ R_c @ M.T
    q_i = quat_xyzw_to_wxyz(R.from_matrix(R_i).as_quat())
    return pos_i, q_i


def _home_step(cur_t, cur_q_wxyz, tgt_t, tgt_q_wxyz, dt, max_lin, max_ang):
    d = tgt_t - cur_t
    dist = float(np.linalg.norm(d))
    if dist <= POS_EPS or dt <= 0.0:
        new_t = tgt_t.copy()
        lin_done = True
    else:
        step = min(max_lin * dt, dist)
        new_t = cur_t + (d / dist) * step
        lin_done = (dist - step) <= POS_EPS

    R_cur = R.from_quat(quat_wxyz_to_xyzw(cur_q_wxyz))
    R_tgt = R.from_quat(quat_wxyz_to_xyzw(tgt_q_wxyz))
    R_err = R_tgt * R_cur.inv()
    rotvec = R_err.as_rotvec()
    ang = float(np.linalg.norm(rotvec))
    if ang <= ANG_EPS or dt <= 0.0:
        new_q = tgt_q_wxyz.copy()
        ang_done = True
    else:
        frac = min(1.0, (max_ang * dt) / ang)
        R_step = R.from_rotvec(rotvec * frac)
        R_new = R_step * R_cur
        new_q = quat_xyzw_to_wxyz(R_new.as_quat())
        ang_done = (ang * (1.0 - frac)) <= ANG_EPS

    return new_t, new_q, (lin_done and ang_done)


def _quat_rel_wxyz(q_cur_wxyz: np.ndarray, q_prev_wxyz: np.ndarray) -> np.ndarray:
    R_cur = R.from_quat(quat_wxyz_to_xyzw(q_cur_wxyz))
    R_prev = R.from_quat(quat_wxyz_to_xyzw(q_prev_wxyz))
    R_rel = R_cur * R_prev.inv()
    return quat_xyzw_to_wxyz(R_rel.as_quat())


# -------------------------------- node --------------------------------


class VrPublisher(Node):
    def __init__(self):
        super().__init__("vr_publisher")
        self.device = OculusReader()
        # Topic publishers
        self.pose_pub = {
            "l": self.create_publisher(PoseStamped, f"{VR_PREFIX}/l/pose", 10),
            "r": self.create_publisher(PoseStamped, f"{VR_PREFIX}/r/pose", 10),
        }
        self.delta_pub = {
            "l": self.create_publisher(PoseStamped, f"{VR_PREFIX}/l/delta_pose", 10),
            "r": self.create_publisher(PoseStamped, f"{VR_PREFIX}/r/delta_pose", 10),
        }
        self.rpy_pub = {
            "l": self.create_publisher(Vector3, f"{VR_PREFIX}/l/rpy", 10),
            "r": self.create_publisher(Vector3, f"{VR_PREFIX}/r/rpy", 10),
        }
        self.engaged_pub = {
            "l": self.create_publisher(Bool, f"{VR_PREFIX}/l/engaged", 10),
            "r": self.create_publisher(Bool, f"{VR_PREFIX}/r/engaged", 10),
        }
        self.grip_pub = {
            "l": self.create_publisher(Int8, f"{VR_PREFIX}/l/gripper_act", 10),
            "r": self.create_publisher(Int8, f"{VR_PREFIX}/r/gripper_act", 10),
        }
        self.grip_pos_pub = {
            "l": self.create_publisher(Float32, f"{VR_PREFIX}/l/gripper_pos", 10),
            "r": self.create_publisher(Float32, f"{VR_PREFIX}/r/gripper_pos", 10),
        }
        self.side_trigger_pub = {
            "l": self.create_publisher(Int8, f"{VR_PREFIX}/l/side_trigger", 10),
            "r": self.create_publisher(Int8, f"{VR_PREFIX}/r/side_trigger", 10),
        }
        self.save_demo_pub = self.create_publisher(Bool, f"{VR_PREFIX}/save_demo", 10)
        self.delete_demo_pub = self.create_publisher(
            Bool, f"{VR_PREFIX}/delete_demo", 10
        )
        self.button_a_pub = self.create_publisher(Bool, f"{VR_PREFIX}/button_a", 10)
        self.button_x_pub = self.create_publisher(Bool, f"{VR_PREFIX}/button_x", 10)

        # ABS pose state used only for homing path output
        self.r_abs_t = np.asarray(RINIT_POSE[:3], dtype=np.float64)
        self.l_abs_t = np.asarray(LINIT_POSE[:3], dtype=np.float64)
        self.r_abs_q = euler_zyx_to_quat_wxyz(*RINIT_POSE[3:6])
        self.l_abs_q = euler_zyx_to_quat_wxyz(*LINIT_POSE[3:6])

        # Previous raw sample (for delta)
        self.r_prev_t = None
        self.l_prev_t = None
        self.r_prev_q = None
        self.l_prev_q = None

        # Trigger hysteresis (engaged)
        self.on_th = 0.8
        self.off_th = 0.2
        self.r_pressed = False
        self.l_pressed = False

        # Index/gripper hysteresis
        self.grip_on_th = 0.8
        self.grip_off_th = 0.2
        self.r_grip_closed = False
        self.l_grip_closed = False

        # Interpolated gripper positions (start opened)
        self.l_grip_pos = float(GRIP_OPEN_POS)
        self.r_grip_pos = float(GRIP_OPEN_POS)

        # Button A rising-edge latch (homing)
        self._prev_btn_a = False

        # Homing state
        self.l_homing = False
        self.r_homing = False
        self.l_home_target_t = np.asarray(LINIT_POSE[:3], dtype=np.float64)
        self.r_home_target_t = np.asarray(RINIT_POSE[:3], dtype=np.float64)
        self.l_home_target_q = euler_zyx_to_quat_wxyz(*LINIT_POSE[3:6])
        self.r_home_target_q = euler_zyx_to_quat_wxyz(*RINIT_POSE[3:6])
        self.max_lin_vel = float(MAX_LIN_VEL)
        self.max_ang_vel = float(MAX_ANG_VEL)

        # Timing
        self._last_time = time.monotonic()
        self.timer = self.create_timer(1.0 / 60.0, self._tick)

    def set_home_speed(self, lin_mps: float, ang_rps: float):
        self.max_lin_vel = float(lin_mps)
        self.max_ang_vel = float(ang_rps)

    def _edge_update(
        self, t: float, was_pressed: bool, on_th: float = None, off_th: float = None
    ) -> bool:
        on_th = self.on_th if on_th is None else on_th
        off_th = self.off_th if off_th is None else off_th
        if not was_pressed and t > on_th:
            return True
        if was_pressed and t < off_th:
            return False
        return was_pressed

    def _publish_pose(self, side: str, pos_t: np.ndarray, q_wxyz: np.ndarray):
        q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.pose.position.x = float(pos_t[0])
        msg.pose.position.y = float(pos_t[1])
        msg.pose.position.z = float(pos_t[2])
        msg.pose.orientation.x = float(q_xyzw[0])
        msg.pose.orientation.y = float(q_xyzw[1])
        msg.pose.orientation.z = float(q_xyzw[2])
        msg.pose.orientation.w = float(q_xyzw[3])
        self.pose_pub[side].publish(msg)

    def _publish_delta_pose(self, side: str, dpos_t: np.ndarray, dq_wxyz: np.ndarray):
        q_xyzw = quat_wxyz_to_xyzw(dq_wxyz)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.pose.position.x = float(dpos_t[0])
        msg.pose.position.y = float(dpos_t[1])
        msg.pose.position.z = float(dpos_t[2])
        msg.pose.orientation.x = float(q_xyzw[0])
        msg.pose.orientation.y = float(q_xyzw[1])
        msg.pose.orientation.z = float(q_xyzw[2])
        msg.pose.orientation.w = float(q_xyzw[3])
        self.delta_pub[side].publish(msg)

    def _publish_rpy(self, side: str, q_wxyz: np.ndarray):
        q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
        ypr = R.from_quat(q_xyzw).as_euler("ZYX", degrees=False)
        v = Vector3()
        v.x = float(ypr[2])  # roll
        v.y = float(ypr[1])  # pitch
        v.z = float(ypr[0])  # yaw
        self.rpy_pub[side].publish(v)

    def _start_homing(self):
        self.l_home_target_t = np.asarray(LINIT_POSE[:3], dtype=np.float64)
        self.r_home_target_t = np.asarray(RINIT_POSE[:3], dtype=np.float64)
        self.l_home_target_q = euler_zyx_to_quat_wxyz(*LINIT_POSE[3:6])
        self.r_home_target_q = euler_zyx_to_quat_wxyz(*RINIT_POSE[3:6])
        self.l_homing = True
        self.r_homing = True
        # Reset prevs so Δ after homing doesn't spike
        self.l_prev_t = None
        self.r_prev_t = None
        self.l_prev_q = None
        self.r_prev_q = None

    def _tick(self):
        try:
            now = time.monotonic()
            dt = now - self._last_time
            self._last_time = now

            sample = self.device.get_transformations_and_buttons()
            if not sample:
                return
            transforms, buttons = sample
            if not transforms:
                return
            Tl = transforms.get("l", None)
            Tr = transforms.get("r", None)
            if Tl is None or Tr is None:
                return

            # Raw from device
            l_pos_raw, l_quat_raw = pose_from_T(np.asarray(Tl))
            r_pos_raw, r_quat_raw = pose_from_T(np.asarray(Tr))
            # Internal coord and orientation offsets
            l_pos_cur, l_quat_cur = _controller_to_internal(l_pos_raw, l_quat_raw)
            r_pos_cur, r_quat_cur = _controller_to_internal(r_pos_raw, r_quat_raw)
            l_quat_cur = _apply_offset(l_quat_cur, LOFFSET_POSE)
            r_quat_cur = _apply_offset(r_quat_cur, ROFFSET_POSE)
            l_quat_cur = _normalize_quat_wxyz(l_quat_cur)
            r_quat_cur = _normalize_quat_wxyz(r_quat_cur)

            # Triggers / indices
            trig_l = _get_analog(buttons, ["leftTrig", "LT", "trigger_l"], 0.0)
            trig_r = _get_analog(buttons, ["rightTrig", "RT", "trigger_r"], 0.0)
            idx_l = _get_analog(
                buttons,
                ["leftIndex", "IndexL", "indexL", "index_l", "leftPinch"],
                trig_l,
            )
            idx_r = _get_analog(
                buttons,
                ["rightIndex", "IndexR", "indexR", "index_r", "rightPinch"],
                trig_r,
            )
            grip_l = _get_analog(buttons, ["leftGrip", "LG", "grip_l"], 0.0)
            grip_r = _get_analog(buttons, ["rightGrip", "RG", "grip_r"], 0.0)

            # Buttons
            btn_a = bool(buttons.get("A", False))
            btn_b = bool(buttons.get("B", False))
            btn_x = bool(buttons.get("X", False))
            self.button_a_pub.publish(Bool(data=btn_a))
            self.button_x_pub.publish(Bool(data=btn_x))

            # Rising edge -> start homing
            if btn_a and not self._prev_btn_a:
                self._start_homing()
            self._prev_btn_a = btn_a

            # Engaged states (hysteresis)
            new_l_pressed = self._edge_update(trig_l, self.l_pressed)
            new_r_pressed = self._edge_update(trig_r, self.r_pressed)

            # ---------- LEFT absolute pose ----------
            if self.l_homing:
                self.l_abs_t, self.l_abs_q, l_done = _home_step(
                    self.l_abs_t,
                    self.l_abs_q,
                    self.l_home_target_t,
                    self.l_home_target_q,
                    dt,
                    self.max_lin_vel,
                    self.max_ang_vel,
                )
                if l_done:
                    self.l_homing = False
                    self.l_prev_t = None
                    self.l_prev_q = None
                # While homing, publish zero delta
                self._publish_delta_pose(
                    "l", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
                )
            else:
                # Direct pose from current controller (no accumulation)
                self.l_abs_t = l_pos_cur
                self.l_abs_q = l_quat_cur
                if new_l_pressed:
                    if self.l_prev_t is not None and self.l_prev_q is not None:
                        dpos = l_pos_cur - self.l_prev_t
                        if np.linalg.norm(dpos) < POS_DEAD:
                            dpos[:] = 0.0
                        dq = _quat_rel_wxyz(l_quat_cur, self.l_prev_q)
                        # Small rotation dead-zone
                        cosh = abs(
                            float(
                                np.dot(
                                    _normalize_quat_wxyz(l_quat_cur),
                                    _normalize_quat_wxyz(self.l_prev_q),
                                )
                            )
                        )
                        ang = 2.0 * np.arccos(np.clip(cosh, -1.0, 1.0))
                        if ang < ROT_DEAD_RAD:
                            dq = np.array([1.0, 0.0, 0.0, 0.0])
                        self._publish_delta_pose("l", dpos, dq)
                    else:
                        self._publish_delta_pose(
                            "l", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
                        )
                    self.l_prev_t = l_pos_cur.copy()
                    self.l_prev_q = l_quat_cur.copy()
                else:
                    self.l_prev_t = None
                    self.l_prev_q = None
                    self._publish_delta_pose(
                        "l", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
                    )

            # ---------- RIGHT absolute pose ----------
            if self.r_homing:
                self.r_abs_t, self.r_abs_q, r_done = _home_step(
                    self.r_abs_t,
                    self.r_abs_q,
                    self.r_home_target_t,
                    self.r_home_target_q,
                    dt,
                    self.max_lin_vel,
                    self.max_ang_vel,
                )
                if r_done:
                    self.r_homing = False
                    self.r_prev_t = None
                    self.r_prev_q = None
                self._publish_delta_pose(
                    "r", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
                )
            else:

                if new_r_pressed:
                    self.r_abs_t = r_pos_cur
                    self.r_abs_q = r_quat_cur
                    if self.r_prev_t is not None and self.r_prev_q is not None:
                        dpos = r_pos_cur - self.r_prev_t
                        if np.linalg.norm(dpos) < POS_DEAD:
                            dpos[:] = 0.0
                        dq = _quat_rel_wxyz(r_quat_cur, self.r_prev_q)
                        cosh = abs(
                            float(
                                np.dot(
                                    _normalize_quat_wxyz(r_quat_cur),
                                    _normalize_quat_wxyz(self.r_prev_q),
                                )
                            )
                        )
                        ang = 2.0 * np.arccos(np.clip(cosh, -1.0, 1.0))
                        if ang < ROT_DEAD_RAD:
                            dq = np.array([1.0, 0.0, 0.0, 0.0])
                        self._publish_delta_pose("r", dpos, dq)
                    else:
                        self._publish_delta_pose(
                            "r", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
                        )
                    self.r_prev_t = r_pos_cur.copy()
                    self.r_prev_q = r_quat_cur.copy()
                else:
                    self.r_prev_t = None
                    self.r_prev_q = None
                    self._publish_delta_pose(
                        "r", np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
                    )

            # Update pressed states after logic
            self.l_pressed = new_l_pressed
            self.r_pressed = new_r_pressed

            # --------- Other topics ---------
            self.save_demo_pub.publish(Bool(data=btn_b))  # B
            self.delete_demo_pub.publish(Bool(data=btn_x))  # X

            self.engaged_pub["l"].publish(Bool(data=self.l_pressed))
            self.engaged_pub["r"].publish(Bool(data=self.r_pressed))

            # Gripper act (binary via index hysteresis)
            self.l_grip_closed = self._edge_update(
                idx_l, self.l_grip_closed, self.grip_on_th, self.grip_off_th
            )
            self.r_grip_closed = self._edge_update(
                idx_r, self.r_grip_closed, self.grip_on_th, self.grip_off_th
            )
            self.grip_pub["l"].publish(Int8(data=1 if self.l_grip_closed else 0))
            self.grip_pub["r"].publish(Int8(data=1 if self.r_grip_closed else 0))

            # Interpolated absolute gripper positions from analog grips
            self.l_grip_closed = self._edge_update(
                grip_l, self.l_grip_closed, self.grip_on_th, self.grip_off_th
            )
            self.r_grip_closed = self._edge_update(
                grip_r, self.r_grip_closed, self.grip_on_th, self.grip_off_th
            )

            self.side_trigger_pub["l"].publish(Int8(data=int(grip_l)))
            self.side_trigger_pub["r"].publish(Int8(data=int(grip_r)))
            l_target = GRIP_OPEN_POS + (GRIP_CLOSE_POS - GRIP_OPEN_POS) * float(grip_l)
            r_target = GRIP_OPEN_POS + (GRIP_CLOSE_POS - GRIP_OPEN_POS) * float(grip_r)
            max_step = GRIP_MAX_VEL * max(dt, 1e-3)

            def _step(cur, tgt, step):
                d = tgt - cur
                if abs(d) <= step:
                    return tgt
                return cur + (step if d > 0.0 else -step)

            self.l_grip_pos = _step(self.l_grip_pos, l_target, max_step)
            self.r_grip_pos = _step(self.r_grip_pos, r_target, max_step)
            self.grip_pos_pub["l"].publish(Float32(data=float(self.l_grip_pos)))
            self.grip_pos_pub["r"].publish(Float32(data=float(self.r_grip_pos)))

            # Publish absolute poses (homing overrides, otherwise direct)
            self._publish_pose("l", self.l_abs_t, self.l_abs_q)
            self._publish_pose("r", self.r_abs_t, self.r_abs_q)

            # RPY derived from absolute quats
            self._publish_rpy("l", self.l_abs_q)
            self._publish_rpy("r", self.r_abs_q)
        except Exception as e:
            self.get_logger().error(f"vr_publisher tick error: {e}")


def main():
    rclpy.init()
    node = VrPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
