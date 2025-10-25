from typing import Tuple, overload
import numpy as np
import numpy.typing as npt
from enum import Enum

class MotorType:
    EC_A4310: "MotorType"
    DM_J4310: "MotorType"
    DM_J4340: "MotorType"
    NONE: "MotorType"

class RobotConfig:
    """Does not have a constructor, use RobotConfigFactory.get_instance().get_config(...) instead."""

    robot_model: str
    joint_pos_min: np.ndarray
    joint_pos_max: np.ndarray
    joint_vel_max: np.ndarray
    joint_torque_max: np.ndarray
    ee_vel_max: np.ndarray
    gripper_vel_max: float
    gripper_torque_max: float
    gripper_width: float
    gripper_open_readout: float
    joint_dof: int
    motor_id: list[int]
    motor_type: list[MotorType]
    gripper_motor_id: int
    gripper_motor_type: MotorType
    gravity_vector: np.ndarray
    base_link_name: str
    eef_link_name: str
    urdf_path: str

class ControllerConfig:
    """Does not have a constructor, use ControllerConfigFactory.get_instance().get_config(...) instead."""

    controller_type: str
    default_kp: np.ndarray
    default_kd: np.ndarray
    default_gripper_kp: float
    default_gripper_kd: float
    over_current_cnt_max: int
    controller_dt: float
    gravity_compensation: bool
    background_send_recv: bool
    shutdown_to_passive: bool
    interpolation_method: str
    default_preview_time: float

class RobotConfigFactory:
    @classmethod
    def get_instance(cls) -> RobotConfigFactory: ...
    def get_config(self, robot_model: str) -> RobotConfig: ...

class ControllerConfigFactory:
    @classmethod
    def get_instance(cls) -> ControllerConfigFactory: ...
    def get_config(self, controller_type: str, joint_dof: int) -> ControllerConfig: ...

class LogLevel:
    TRACE: "LogLevel"
    DEBUG: "LogLevel"
    INFO: "LogLevel"
    WARNING: "LogLevel"
    ERROR: "LogLevel"
    CRITICAL: "LogLevel"
    OFF: "LogLevel"

class Gain:
    def kp(self) -> npt.NDArray[np.float64]: ...
    def kd(self) -> npt.NDArray[np.float64]: ...
    gripper_kp: float
    gripper_kd: float
    @overload
    def __init__(self, dof: int) -> None: ...
    @overload
    def __init__(
        self,
        kp: npt.NDArray[np.float64],
        kd: npt.NDArray[np.float64],
        gripper_kp: float,
        gripper_kd: float,
    ) -> None: ...
    def __add__(self, other: Gain) -> Gain: ...
    def __mul__(self, scalar: float) -> Gain: ...

class JointState:
    timestamp: float
    gripper_pos: float
    gripper_vel: float
    gripper_torque: float
    @overload
    def __init__(self, dof: int) -> None: ...
    @overload
    def __init__(
        self,
        pos: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        torque: npt.NDArray[np.float64],
        gripper_pos: float,
    ) -> None: ...
    @overload
    def __init__(
        self,
        pos: npt.NDArray[np.float64],
        vel: npt.NDArray[np.float64],
        torque: npt.NDArray[np.float64],
        gripper_pos: float,
        gripper_vel: float,
        gripper_torque: float,
    ) -> None: ...
    def __add__(self, other: JointState) -> JointState: ...
    def __mul__(self, scalar: float) -> JointState: ...
    def pos(self) -> npt.NDArray[np.float64]: ...
    def vel(self) -> npt.NDArray[np.float64]: ...
    def torque(self) -> npt.NDArray[np.float64]: ...

class Arx5JointController:
    @overload
    def __init__(
        self,
        model: str,
        interface_name: str,
    ) -> None: ...
    @overload
    def __init__(
        self,
        robot_config: RobotConfig,
        controller_config: ControllerConfig,
        interface_name: str,
    ) -> None: ...
    def send_recv_once(self) -> None: ...
    def recv_once(self) -> None: ...
    def set_joint_cmd(self, cmd: JointState) -> None: ...
    def set_joint_traj(self, traj: list[JointState]) -> None: ...
    def get_joint_cmd(self) -> JointState: ...
    def get_timestamp(self) -> float: ...
    def get_joint_state(self) -> JointState: ...
    def get_eef_state(self) -> EEFState: ...
    def get_home_pose(self) -> np.ndarray: ...
    def set_gain(self, gain: Gain) -> None: ...
    def get_gain(self) -> Gain: ...
    def get_robot_config(self) -> RobotConfig: ...
    def get_controller_config(self) -> ControllerConfig: ...
    def reset_to_home(self) -> None: ...
    def set_to_damping(self) -> None: ...
    def calibrate_gripper(self) -> None: ...
    def calibrate_joint(self, joint_id: int) -> None: ...
    def set_log_level(self, level: LogLevel) -> None: ...

class EEFState:
    timestamp: float
    gripper_pos: float
    gripper_vel: float
    gripper_torque: float
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(
        self, pose_6d: npt.NDArray[np.float64], gripper_pos: float
    ) -> None: ...
    @overload
    def __init__(
        self,
        pose_6d: npt.NDArray[np.float64],
        gripper_pos: float,
        gripper_vel: float,
        gripper_torque: float,
    ) -> None: ...
    def __add__(self, other: EEFState) -> EEFState: ...
    def __mul__(self, scalar: float) -> EEFState: ...
    def pose_6d(self) -> npt.NDArray[np.float64]: ...

class Arx5CartesianController:
    @overload
    def __init__(self, model: str, interface_name: str) -> None: ...
    @overload
    def __init__(
        self,
        robot_config: RobotConfig,
        controller_config: ControllerConfig,
        interface_name: str,
    ) -> None: ...
    def set_eef_cmd(self, cmd: EEFState) -> None: ...
    def set_eef_traj(self, traj: list[EEFState]) -> None: ...
    def get_joint_cmd(self) -> JointState: ...
    def get_eef_cmd(self) -> EEFState: ...
    def get_eef_state(self) -> EEFState: ...
    def get_joint_state(self) -> JointState: ...
    def get_timestamp(self) -> float: ...
    def set_gain(self, gain: Gain) -> None: ...
    def get_gain(self) -> Gain: ...
    def get_home_pose(self) -> np.ndarray: ...
    def set_log_level(self, level: LogLevel) -> None: ...
    def get_robot_config(self) -> RobotConfig: ...
    def get_controller_config(self) -> ControllerConfig: ...
    def reset_to_home(self) -> None: ...
    def set_to_damping(self) -> None: ...

class Arx5Solver:
    @overload
    def __init__(
        self,
        urdf_path: str,
        joint_dof: int,
        joint_pos_min: npt.NDArray[np.float64],
        joint_pos_max: npt.NDArray[np.float64],
    ) -> None: ...
    @overload
    def __init__(
        self,
        urdf_path: str,
        joint_dof: int,
        joint_pos_min: npt.NDArray[np.float64],
        joint_pos_max: npt.NDArray[np.float64],
        base_link: str,
        eef_link: str,
        gravity_vector: npt.NDArray[np.float64],
    ) -> None: ...
    def inverse_dynamics(
        self,
        joint_pos: npt.NDArray[np.float64],
        joint_vel: npt.NDArray[np.float64],
        joint_acc: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: ...
    def inverse_kinematics(
        self,
        target_pose_6d: npt.NDArray[np.float64],
        current_joint_pos: npt.NDArray[np.float64],
    ) -> Tuple[int, npt.NDArray[np.float64]]: ...
    def multi_trial_ik(
        self,
        target_pose_6d: npt.NDArray[np.float64],
        current_joint_pos: npt.NDArray[np.float64],
        additional_trial_num: int,
    ) -> Tuple[int, npt.NDArray[np.float64]]: ...
    def get_ik_status_name(self, status: int) -> str: ...
    def forward_kinematics(
        self, joint_pos: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
