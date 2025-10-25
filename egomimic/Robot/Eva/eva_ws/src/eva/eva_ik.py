# ================= eva_replay_ros.py (home-anchored world sweep + delta-driven teleop with A-button homing) ================= #WORKING
from typing import Optional, List, Tuple
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int8, Float32
from sensor_msgs.msg import JointState

import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
import os
import yaml
import tempfile
from ament_index_python.packages import get_package_share_directory


# Joint-space start / home
START_ARM_POSE = [0, 0, 0, 0, 0, 0]
HOME_JOINTS = [0, 0, 0, 0, 0, 0]

# Per-joint velocity limits [rad/s] during homing (6-DoF)
DEFAULT_JOINT_VEL_LIMS = [0.5, 0.5, 0.5, 0.6, 0.6, 0.8]
JOINT_EPS = 1e-3  # rad tolerance to consider a joint at target

# Keep this at zero unless you intentionally want a fixed yaw/pitch/roll offset in addition to controller motion.
OFFSET_YPR = [0.0, 0.0, 0.0]
YPR_VEL_LIMIT = np.array([2.4, 2.4, 2.4])
YPR_LIMIT = np.array([np.pi, np.pi, np.pi]) * 3 / 4


def q_conjugate_xyzw(q_xyzw):
    # For unit quats, conj == inverse
    x, y, z, w = q_xyzw
    return np.array([-x, -y, -z, w], dtype=np.float64)


class EvaIK:
    def __init__(self, urdf_path: str, gui: bool = False) -> None:
        self.urdf = urdf_path
        self.robot_id = self._init_pybullet(urdf_path=self.urdf, GUI=gui)
        self.endEffectorIndex = 5
        # Cache FK(HOME) once, without disturbing state
        self.home_pos_world, self.home_quat_world = self.fk_at_joints(START_ARM_POSE)

    def _init_pybullet(self, urdf_path: str, GUI: bool = False) -> int:
        if GUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Base mount orientation — PyBullet quats are XYZW
        self.base_quat_xyzw = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self.R_flip = np.eye(3)

        # Resolve package:// paths if present by rewriting URDF to a temp file
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.abspath(urdf_path)
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF path does not exist: {urdf_path}")

        # Add search path to the model root (one level above the urdf dir)
        model_root_dir = os.path.abspath(
            os.path.join(os.path.dirname(urdf_path), os.pardir)
        )
        p.setAdditionalSearchPath(model_root_dir)

        load_path = urdf_path
        try:
            with open(urdf_path, "r", encoding="utf-8") as f:
                urdf_text = f.read()
            if "package://" in urdf_text:
                # Replace known package URIs with absolute paths
                # Example: package://X5A/... -> /.../X5A/...
                # Model root contains the X5A directory
                urdf_text = urdf_text.replace("package://X5A", model_root_dir)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf")
                tmp.write(urdf_text.encode("utf-8"))
                tmp.flush()
                tmp.close()
                load_path = tmp.name
        except Exception:
            load_path = urdf_path

        # print(f"Loading URDF: {load_path}")

        robot_id = p.loadURDF(
            load_path,
            basePosition=[0, 0, 0],
            baseOrientation=self.base_quat_xyzw,
            useFixedBase=True,
        )
        p.setGravity(0, 0, -9.81)
        return robot_id

    def _set_joints(self, joints6: List[float]) -> None:
        for j_idx in range(min(6, p.getNumJoints(self.robot_id))):
            p.resetJointState(self.robot_id, j_idx, float(joints6[j_idx]))

    def _get_fk_current(self) -> Tuple[np.ndarray, np.ndarray]:
        ls = p.getLinkState(
            self.robot_id, self.endEffectorIndex, computeForwardKinematics=True
        )
        pos = np.array(ls[4], dtype=np.float64)
        quat_xyzw = np.array(ls[5], dtype=np.float64)
        return pos, quat_xyzw

    def fk_at_joints(self, joints6: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """FK for given joints (restore the original state after)."""
        # snapshot current
        saved = []
        for j_idx in range(min(6, p.getNumJoints(self.robot_id))):
            saved.append(p.getJointState(self.robot_id, j_idx)[0])

        # compute FK
        self._set_joints(joints6)
        pos, quat = self._get_fk_current()

        # restore
        for j_idx in range(min(6, p.getNumJoints(self.robot_id))):
            p.resetJointState(self.robot_id, j_idx, float(saved[j_idx]))

        return pos, quat

    def solve_single(
        self,
        target_pos_world: np.ndarray,
        target_orientation_world_xyzw: np.ndarray,
        current_joints: List[float],
    ) -> np.ndarray:
        joint_dampening = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.01, 0.01]

        q = np.asarray(target_orientation_world_xyzw, dtype=np.float64)
        qn = float(np.linalg.norm(q))
        if not (0.999 <= qn <= 1.001):
            q = q / max(qn, 1e-9)

        joint_angle = p.calculateInverseKinematics(
            self.robot_id,
            endEffectorLinkIndex=self.endEffectorIndex,
            targetPosition=target_pos_world.tolist(),
            targetOrientation=q.tolist(),
            currentPositions=current_joints[:6] + [0.02239, -0.02239],
            maxNumIterations=100,
            residualThreshold=0.002,
            jointDamping=joint_dampening,
        )
        return np.array(list(joint_angle)[:6], dtype=np.float64)


class EvaIKNode(Node):
    def __init__(
        self,
        urdf_path: str,
        topic_prefix_in: str,
        topic_prefix_out: str,
        arm: Optional[str],
        rate_hz: float,
        only_when_engaged: bool,
        gui: bool,
        diag_seconds: float,
        diag_amplitude: float,
        diag_segment_s: float,
        joint_vel_lims: List[float],
    ) -> None:
        super().__init__("eva_ik_node")
        self.topic_prefix_in = topic_prefix_in
        self.topic_prefix_out = topic_prefix_out
        # Determine arm from node name if not provided
        if arm is None:
            name = self.get_name().lower()
            if name.endswith("_l") or "left" in name:
                self.arm = "l"
            elif name.endswith("_r") or "right" in name:
                self.arm = "r"
            else:
                self.arm = "r"
        else:
            self.arm = arm
        self.only_when_engaged = only_when_engaged

        qos = QoSProfile(depth=50)
        self.ik = EvaIK(urdf_path=urdf_path, gui=gui)

        # State
        self.pose_r: Optional[PoseStamped] = None
        self.delta_r: Optional[PoseStamped] = None  # per-frame delta pose
        self.engaged: bool = False
        self.prev_engaged: bool = False
        self.button_a: bool = False
        self._prev_button_a: bool = False
        self.gripper_r: int = 0
        self.gripper_r_send: int = 0
        self.current_joints: List[float] = START_ARM_POSE[:6].copy()
        self.cur_ypr = np.array([0, 0, 0])
        self.prev_ypr = np.array([0, 0, 0])

        # Gripper
        self.gripper_act = 0
        self.gripper_vel = 0.10  # m/s
        self.gripper_length = 0.08  # m
        self.cur_gripper_pos = 0.0

        # VR anchoring (used with accumulated delta from topic)
        self.vr_anchor_pos_world: Optional[np.ndarray] = None
        self.vr_anchor_quat_world_xyzw: Optional[np.ndarray] = None

        self.tgt_pos_world, cur_quat_world = self.ik.fk_at_joints(self.current_joints)

        # Accumulated delta (sum of per-frame Δ from topic)
        self.delta_accum_pos: np.ndarray = np.zeros(3, dtype=np.float64)
        self.delta_accum_quat_xyzw: np.ndarray = np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float64
        )

        # Homing
        self.homing_active: bool = False
        self.joint_vel_lims = np.asarray(joint_vel_lims, dtype=np.float64)
        self._last_tick_time = time.time()

        # Diagnostics (home-anchored world sweep)
        self.start_time = time.time()
        self.diag_seconds = max(0.0, float(diag_seconds))
        self.diag_amplitude = float(diag_amplitude)
        self.diag_segment_s = max(0.2, float(diag_segment_s))
        self._printed_first_ik = False
        self._last_phase = ""
        self.side_trigger = 0

        # Subscriptions
        self.create_subscription(
            PoseStamped,
            f"/{self.topic_prefix_in}/{self.arm}/pose",
            self._on_pose_r,
            qos,
        )
        self.create_subscription(
            PoseStamped,
            f"/{self.topic_prefix_in}/{self.arm}/delta_pose",
            self._on_delta_r,
            qos,
        )
        self.create_subscription(
            Bool, f"/{self.topic_prefix_in}/{self.arm}/engaged", self._on_engaged, qos
        )
        self.create_subscription(
            Bool, f"/{self.topic_prefix_in}/button_a", self._on_button_a, qos
        )
        self.create_subscription(
            Int8,
            f"/{self.topic_prefix_in}/{self.arm}/side_trigger",
            self._on_side_trigger,
            qos,
        )
        # Publishers
        self.pub_joint_state = self.create_publisher(
            JointState, f"/{self.topic_prefix_out}/{self.arm}/joint_state", qos
        )
        self.pub_engaged = self.create_publisher(
            Bool, f"/{self.topic_prefix_out}/{self.arm}/engaged", qos
        )

        self.pub_joint_state = self.create_publisher(
            JointState, f"/{self.topic_prefix_out}/{self.arm}/joint_state", qos
        )
        self.pub_engaged = self.create_publisher(
            Bool, f"/{self.topic_prefix_out}/{self.arm}/engaged", qos
        )
        self.pub_fk_ee_pose = self.create_publisher(
            PoseStamped, f"/{self.topic_prefix_out}/{self.arm}/fk_ee_pose", qos
        )
        self.pub_target_ee_pose = self.create_publisher(
            PoseStamped, f"/{self.topic_prefix_out}/{self.arm}/target_ee_pose", qos
        )

        # Timer
        self.dt = 1.0 / max(rate_hz, 1e-6)
        self.timer = self.create_timer(self.dt, self._on_timer)

        self.get_logger().info(
            f"IK node rate={rate_hz:.2f}Hz, only_when_engaged={self.only_when_engaged}"
        )
        self.get_logger().info(
            f"R_flip=\n{self.ik.R_flip}\nOFFSET_YPR={OFFSET_YPR}, base_quat_xyzw={self.ik.base_quat_xyzw}"
        )
        self.get_logger().info(
            f"FK(HOME): pos={self.ik.home_pos_world.tolist()}, quat={self.ik.home_quat_world.tolist()}"
        )
        if self.diag_seconds > 0:
            self.get_logger().warn(
                f"[DIAG: home-anchored straight lines] total={self.diag_seconds:.1f}s, "
                f"amp={self.diag_amplitude:.3f} m, seg={self.diag_segment_s:.2f}s per direction "
                f"(+X,-X,+Y,-Y,+Z,-Z), ori locked to FK(HOME)"
            )

    def _on_pose_r(self, msg: PoseStamped) -> None:
        self.pose_r = msg

    def _on_delta_r(self, msg: PoseStamped) -> None:
        self.delta_r = msg

    def _on_engaged(self, msg: Bool) -> None:
        self.engaged = bool(msg.data)

    def _on_button_a(self, msg: Bool) -> None:
        self.button_a = bool(msg.data)

    def _on_side_trigger(self, msg: Int8) -> None:
        self.side_trigger = int(msg.data)

    # ---------- DIAG: home-anchored sweep in WORLD ----------
    def _diag_target_world(
        self, t_elapsed: float
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        dirs = np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ],
            dtype=np.float64,
        )
        total_cycle = 6 * self.diag_segment_s
        tt = t_elapsed % total_cycle
        seg_idx = int(tt // self.diag_segment_s)
        alpha = (tt - seg_idx * self.diag_segment_s) / self.diag_segment_s  # 0->1
        dir_vec = dirs[seg_idx]
        phase = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"][seg_idx]

        pos = self.ik.home_pos_world + dir_vec * (self.diag_amplitude * alpha)
        quat = self.ik.home_quat_world.copy()
        return pos, quat, phase

    # ----------------- Math helpers -----------------
    @staticmethod
    def _quat_normalize(q_xyzw: np.ndarray) -> np.ndarray:
        q = np.asarray(q_xyzw, dtype=np.float64)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / n

    def _compose_xyzw(self, qa_xyzw: np.ndarray, qb_xyzw: np.ndarray) -> np.ndarray:
        """Quaternion composition (xyzw): qc = qa ⊗ qb."""
        _, qc_xyzw = p.multiplyTransforms(
            [0, 0, 0], qa_xyzw.tolist(), [0, 0, 0], qb_xyzw.tolist()
        )
        return np.array(qc_xyzw, dtype=np.float64)

    @staticmethod
    def _ypr_to_quat_xyzw(ypr: np.ndarray) -> np.ndarray:
        # ypr = [yaw(Z), pitch(Y), roll(X)]
        return R.from_euler("ZYX", ypr, degrees=False).as_quat()  # [x,y,z,w]

    @staticmethod
    def _quat_to_ypr_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
        return R.from_quat(q_xyzw).as_euler("ZYX", degrees=False)  # [yaw, pitch, roll]

    # --------- Homing helpers ---------
    def _start_homing(self):
        self.homing_active = True

    def _homing_step(self, dt: float) -> List[float]:
        """One bounded-velocity joint-space step toward HOME_JOINTS."""

        cur = np.asarray(self.current_joints, dtype=np.float64)
        tgt = np.asarray(HOME_JOINTS, dtype=np.float64)
        diff = tgt - cur

        # Per-joint clip by velocity*dt
        max_step = self.joint_vel_lims * max(dt, 1e-3)
        step = np.clip(diff, -max_step, max_step)
        new = cur + step

        # Check completion
        if np.all(np.abs(tgt - new) <= JOINT_EPS):
            new = tgt.copy()
            self.homing_active = False
        return new.tolist()

    def _reset_anchor_to_current_fk(self):

        # Re-anchor VR target to current EE pose and clear delta accumulators
        cur_pos_world, cur_quat_world = self.ik.fk_at_joints(self.current_joints)
        self.vr_anchor_pos_world = cur_pos_world.copy()
        self.vr_anchor_quat_world_xyzw = cur_quat_world.copy()
        self.delta_accum_pos[:] = 0.0
        self.delta_accum_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    # ----------------- Main loop -----------------
    def _on_timer(self) -> None:
        # Republish engaged/gripper
        self.pub_engaged.publish(Bool(data=self.engaged))

        # dt for homing velocity limits
        now = time.time()
        dt_real = now - self._last_tick_time
        self._last_tick_time = now
        elapsed = now - self.start_time
        using_diag = self.diag_seconds > 0 and elapsed < self.diag_seconds

        # Rising edges
        rising_engaged = (not self.prev_engaged) and self.engaged
        self.prev_engaged = self.engaged
        rising_a = (not self._prev_button_a) and self.button_a
        self._prev_button_a = self.button_a

        # A button -> begin homing
        if rising_a:
            self.get_logger().info("[A] Homing to HOME_JOINTS with velocity limits.")
            self._start_homing()

        # If we're homing, drive joint-space and publish; skip IK/diag/VR deltas
        if self.homing_active:
            gripper_delta_limit = self.gripper_vel * dt_real
            target = self.gripper_length
            gripper_delta = target - self.cur_gripper_pos
            gripper_delta = np.clip(
                gripper_delta, -gripper_delta_limit, gripper_delta_limit
            )
            self.cur_gripper_pos = self.cur_gripper_pos + gripper_delta
            self.cur_gripper_pos = float(
                np.clip(self.cur_gripper_pos, 0, self.gripper_length)
            )

            self.cur_ypr = np.array([0, 0, 0])
            self.prev_ypr = np.array([0, 0, 0])

            self.gripper_r_send
            self.current_joints = self._homing_step(dt_real)

            self._reset_anchor_to_current_fk()
            self.tgt_pos_world = self.vr_anchor_pos_world
            target_ee = PoseStamped()
            target_ee.header.stamp = self.get_clock().now().to_msg()
            target_ee.header.frame_id = "world"
            (
                target_ee.pose.position.x,
                target_ee.pose.position.y,
                target_ee.pose.position.z,
            ) = map(float, self.vr_anchor_pos_world)
            (
                target_ee.pose.orientation.x,
                target_ee.pose.orientation.y,
                target_ee.pose.orientation.z,
                target_ee.pose.orientation.w,
            ) = map(
                float,
                R.from_euler("ZYX", np.zeros((3,)), degrees=False).as_quat().tolist(),
            )
            self.pub_target_ee_pose.publish(target_ee)

            if not self.homing_active:
                # finished
                self.get_logger().info(
                    "[A] Homing complete. Resetting VR anchor to FK(home)."
                )

            joint_state = JointState()
            joint_state.header.stamp = self.get_clock().now().to_msg()
            joint_state.name = [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "gripper",
            ]
            joint_state.position = [
                float(self.current_joints[0]),
                float(self.current_joints[1]),
                float(self.current_joints[2]),
                float(self.current_joints[3]),
                float(self.current_joints[4]),
                float(self.current_joints[5]),
                float(self.cur_gripper_pos),
            ]
            self.pub_joint_state.publish(joint_state)

            ee_pos, ee_quat = self.ik.fk_at_joints(self.current_joints)
            ee = PoseStamped()
            ee.header.stamp = self.get_clock().now().to_msg()
            ee.header.frame_id = "world"
            ee.pose.position.x, ee.pose.position.y, ee.pose.position.z = map(
                float, ee_pos.tolist()
            )
            (
                ee.pose.orientation.x,
                ee.pose.orientation.y,
                ee.pose.orientation.z,
                ee.pose.orientation.w,
            ) = map(float, ee_quat.tolist())
            self.pub_fk_ee_pose.publish(ee)

            return

        if using_diag:
            self.tgt_pos_world, tgt_quat_world_xyzw, phase = self._diag_target_world(
                elapsed
            )
            if phase != self._last_phase:
                self._last_phase = phase
                self.get_logger().info(
                    f"[DIAG] {phase} from FK(HOME): self.tgt_pos_world={self.tgt_pos_world.tolist()}"
                )
        else:

            # Need a pose stamp for timing/sync; otherwise do nothing
            if self.pose_r is None:
                return

            # On engaged rising edge: re-anchor to current EE pose and clear deltas
            if (self.vr_anchor_pos_world is None) or rising_engaged:
                cur_pos_world, cur_quat_world = self.ik.fk_at_joints(
                    self.current_joints
                )
                self.vr_anchor_pos_world = cur_pos_world.copy()
                self.vr_anchor_quat_world_xyzw = cur_quat_world.copy()
                self.delta_accum_pos[:] = 0.0
                self.delta_accum_quat_xyzw = np.array(
                    [0.0, 0.0, 0.0, 1.0], dtype=np.float64
                )

            # Apply per-frame delta from topic (already t vs t-1)
            if self.delta_r is not None and self.engaged:

                # Gripper control
                target = None
                gripper_delta_limit = self.gripper_vel * dt_real
                if not self.engaged:
                    target = self.cur_gripper_pos
                else:
                    target = 0.0 if self.side_trigger == 1 else self.gripper_length
                gripper_delta = target - self.cur_gripper_pos
                gripper_delta = np.clip(
                    gripper_delta, -gripper_delta_limit, gripper_delta_limit
                )
                self.cur_gripper_pos = self.cur_gripper_pos + gripper_delta
                self.cur_gripper_pos = float(
                    np.clip(self.cur_gripper_pos, 0, self.gripper_length)
                )

                # YPR + translation control

                dp = self.delta_r.pose.position
                dq = self.delta_r.pose.orientation  # ROS xyzw
                q = self.pose_r.pose.orientation

                dpos = np.array([dp.x, dp.y, dp.z], dtype=np.float64)

                self.delta_accum_pos += dpos * 1.35

                self.tgt_pos_world = (
                    self.vr_anchor_pos_world + self.delta_accum_pos
                ).astype(np.float64)
                vr_xyzw = self._quat_normalize(
                    np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
                )

                err = (
                    R.from_quat(vr_xyzw)
                    * R.from_euler("ZYX", self.prev_ypr, degrees=False).inv()
                )

                # Rate-limit per axis
                delta_limit = YPR_VEL_LIMIT * max(dt_real, 1e-3)
                delta = np.clip(
                    err.as_euler("ZYX", degrees=False), -delta_limit, delta_limit
                )
                LIMIT_JOINT = np.array([2, 2, 2])
                # Update global orientation (unbounded) and store
                self.cur_ypr = self.prev_ypr + delta
                self.cur_ypr = np.clip(self.cur_ypr, -LIMIT_JOINT, LIMIT_JOINT)
                self.prev_ypr = self.cur_ypr.copy()

        # IK toward target
        temp_ypr = self.cur_ypr.copy()
        temp_ypr[0] *= 1
        temp_ypr[1] *= 1
        temp_ypr[2] *= 1
        joints6 = self.ik.solve_single(
            self.tgt_pos_world,
            R.from_euler("ZYX", temp_ypr, degrees=False).as_quat(),
            self.current_joints,
        )
        # if not self._printed_first_ik:
        #     self._printed_first_ik = True
        #     self.get_logger().info(
        #         f"[FIRST IK] self.tgt_pos_world={self.tgt_pos_world.tolist()} "
        #         f"tgt_quat_xyzw={tgt_quat_world_xyzw.tolist()} -> joints={joints6.tolist()}"
        #     )
        self.current_joints = joints6.tolist()
        ypr_list = self.cur_ypr.tolist()

        ee_pos, ee_quat = self.ik.fk_at_joints(self.current_joints)
        fk_ee = PoseStamped()
        fk_ee.header.stamp = self.get_clock().now().to_msg()
        fk_ee.header.frame_id = "world"
        fk_ee.pose.position.x, fk_ee.pose.position.y, fk_ee.pose.position.z = map(
            float, ee_pos.tolist()
        )
        (
            fk_ee.pose.orientation.x,
            fk_ee.pose.orientation.y,
            fk_ee.pose.orientation.z,
            fk_ee.pose.orientation.w,
        ) = map(float, ee_quat.tolist())
        self.pub_fk_ee_pose.publish(fk_ee)

        target_ee = PoseStamped()
        target_ee.header.stamp = self.get_clock().now().to_msg()
        target_ee.header.frame_id = "world"
        (
            target_ee.pose.position.x,
            target_ee.pose.position.y,
            target_ee.pose.position.z,
        ) = map(float, self.tgt_pos_world)
        (
            target_ee.pose.orientation.x,
            target_ee.pose.orientation.y,
            target_ee.pose.orientation.z,
            target_ee.pose.orientation.w,
        ) = map(
            float, R.from_euler("ZYX", self.cur_ypr, degrees=False).as_quat().tolist()
        )
        self.pub_target_ee_pose.publish(target_ee)

        # Publish joints
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "gripper",
        ]
        joint_state.position = [
            float(joints6[0]),
            float(joints6[1]),
            float(joints6[2]),
            float(joints6[3]),  # Pitch joint
            float(joints6[4]),  # Yaw Joint
            float(joints6[5]),  # Roll joint,
            float(self.cur_gripper_pos),
        ]
        self.pub_joint_state.publish(joint_state)
        # print(
        #     f"YPR: [{ypr_list[0]:.3f}, {ypr_list[1]:.3f}, {ypr_list[2]:.3f}], world pos: [{self.vr_anchor_pos_world[0]:.3f}, {self.vr_anchor_pos_world[1]:.3f}, {self.vr_anchor_pos_world[2]:.3f}]"
        # )


def _parse_joint_vel_list(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 6:
        raise ValueError("joint velocity limits must have 6 comma-separated values")
    return vals


def main(argv=None) -> None:
    rclpy.init()

    # Load config from package share
    share = get_package_share_directory("eva")
    cfg_path = os.path.join(share, "config", "configs.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    urdf_path = cfg.get("urdf", "")
    topic_prefix_in = cfg.get("topic_prefix_in", "vr")
    topic_prefix_out = cfg.get("topic_prefix_out", "eva_ik")
    rate_hz = float(cfg.get("ik_rate", 50.0))
    joint_vel_lims = DEFAULT_JOINT_VEL_LIMS

    node = EvaIKNode(
        urdf_path=urdf_path,
        topic_prefix_in=topic_prefix_in,
        topic_prefix_out=topic_prefix_out,
        arm=None,
        rate_hz=rate_hz,
        only_when_engaged=False,
        gui=False,
        diag_seconds=0.0,
        diag_amplitude=0.20,
        diag_segment_s=2.0,
        joint_vel_lims=joint_vel_lims,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
