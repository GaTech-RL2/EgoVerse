"""USB/Dynamixel GELLO input and safe joint-space YAM follower control.

This is a local implementation built on the official Dynamixel SDK. It does
not import or depend on a separate GELLO repository. Hardware imports are lazy,
so configuration tests open no serial ports, cameras, CAN buses, or robots.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable, Mapping, Sequence

import numpy as np

from egomimic.robot.yam_station.interface import ARM_OFFSET, joint_vector


class GelloInputError(RuntimeError):
    """A leader sample cannot safely drive a follower command."""


@dataclass(frozen=True)
class GelloSample:
    """One direct read from every configured USB leader."""

    joints: Mapping[str, np.ndarray]
    timestamps: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.joints or set(self.joints) != set(self.timestamps):
            raise ValueError("GELLO joints and timestamps must cover the same arms")
        joints = {arm: joint_vector(value) for arm, value in self.joints.items()}
        timestamps = {arm: float(value) for arm, value in self.timestamps.items()}
        if any(not np.isfinite(value) or value < 0 for value in timestamps.values()):
            raise ValueError("GELLO timestamps must be finite and nonnegative")
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "timestamps", timestamps)


class DynamixelPositionBus:
    """Minimal protocol-2.0 sync reader that never sends position commands."""

    def __init__(
        self,
        port: str,
        ids: Sequence[int],
        baudrate: int = 57600,
        protocol_version: float = 2.0,
        torque_enable_address: int = 64,
        present_position_address: int = 132,
        present_position_length: int = 4,
        encoder_resolution: int = 4096,
        sdk=None,
    ) -> None:
        self.port = str(port)
        self.ids = tuple(int(value) for value in ids)
        self.encoder_resolution = int(encoder_resolution)
        self.present_position_address = int(present_position_address)
        self.present_position_length = int(present_position_length)
        if not self.port or not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("A Dynamixel bus needs a port and distinct servo IDs")
        if any(not 0 <= value <= 252 for value in self.ids):
            raise ValueError("Dynamixel IDs must be between 0 and 252")
        if (
            min(
                int(baudrate),
                int(torque_enable_address),
                self.present_position_address,
                self.present_position_length,
                self.encoder_resolution,
            )
            <= 0
        ):
            raise ValueError(
                "Dynamixel baudrate, addresses, and sizes must be positive"
            )
        if self.present_position_length not in (1, 2, 4):
            raise ValueError("Dynamixel position length must be 1, 2, or 4 bytes")
        if sdk is None:
            import dynamixel_sdk as sdk

        self._sdk = sdk
        self._port_handler = sdk.PortHandler(self.port)
        self._packet_handler = sdk.PacketHandler(float(protocol_version))
        self._group_read = sdk.GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            self.present_position_address,
            self.present_position_length,
        )
        self._opened = False
        try:
            if not self._port_handler.openPort():
                raise RuntimeError(f"Failed to open GELLO serial port {self.port}")
            self._opened = True
            if not self._port_handler.setBaudRate(int(baudrate)):
                raise RuntimeError(
                    f"Failed to set {self.port} to Dynamixel baudrate {baudrate}"
                )
            for servo_id in self.ids:
                if not self._group_read.addParam(servo_id):
                    raise RuntimeError(
                        f"Failed to add Dynamixel ID {servo_id} on {self.port}"
                    )
            # A GELLO is a passive input device. Explicitly disable servo torque
            # before any reads and never expose a write-position method.
            for servo_id in self.ids:
                comm_result, device_error = self._packet_handler.write1ByteTxRx(
                    self._port_handler,
                    servo_id,
                    int(torque_enable_address),
                    0,
                )
                if comm_result != sdk.COMM_SUCCESS or device_error:
                    raise RuntimeError(
                        f"Failed to disable torque on Dynamixel ID {servo_id} "
                        f"at {self.port}: comm={comm_result}, device={device_error}"
                    )
        except BaseException:
            self.close()
            raise

    def read_radians(self) -> np.ndarray:
        if not self._opened:
            raise GelloInputError(f"Dynamixel bus {self.port} is closed")
        result = self._group_read.txRxPacket()
        if result != self._sdk.COMM_SUCCESS:
            detail = self._packet_handler.getTxRxResult(result)
            raise GelloInputError(
                f"Dynamixel sync read failed on {self.port}: {detail}"
            )
        values = []
        bits = 8 * self.present_position_length
        sign_bit, modulus = 1 << (bits - 1), 1 << bits
        for servo_id in self.ids:
            if not self._group_read.isAvailable(
                servo_id,
                self.present_position_address,
                self.present_position_length,
            ):
                raise GelloInputError(
                    f"No Dynamixel position for ID {servo_id} on {self.port}"
                )
            raw = int(
                self._group_read.getData(
                    servo_id,
                    self.present_position_address,
                    self.present_position_length,
                )
            )
            if raw & sign_bit:
                raw -= modulus
            values.append(raw * 2 * np.pi / self.encoder_resolution)
        positions = np.asarray(values, dtype=float)
        if positions.shape != (len(self.ids),) or not np.isfinite(positions).all():
            raise GelloInputError(f"Invalid Dynamixel positions from {self.port}")
        return positions

    def close(self) -> None:
        if self._opened:
            self._port_handler.closePort()
            self._opened = False


class DynamixelGelloLeader:
    """Map one six-joint-plus-gripper USB chain into EgoVerse's 7D convention."""

    def __init__(
        self,
        port: str,
        joint_ids: Sequence[int],
        gripper_id: int,
        joint_signs: Sequence[float],
        joint_offsets_rad: Sequence[float],
        gripper_open_rad: float,
        gripper_closed_rad: float,
        filter_alpha: float = 0.99,
        baudrate: int = 57600,
        protocol_version: float = 2.0,
        torque_enable_address: int = 64,
        present_position_address: int = 132,
        present_position_length: int = 4,
        encoder_resolution: int = 4096,
        bus_factory: Callable = DynamixelPositionBus,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.port = str(port)
        self.joint_ids = tuple(int(value) for value in joint_ids)
        self.gripper_id = int(gripper_id)
        self.joint_signs = np.asarray(joint_signs, dtype=float)
        self.joint_offsets = np.asarray(joint_offsets_rad, dtype=float)
        self.gripper_open = float(gripper_open_rad)
        self.gripper_closed = float(gripper_closed_rad)
        self.filter_alpha = float(filter_alpha)
        if len(self.joint_ids) != 6 or len(set(self.joint_ids)) != 6:
            raise ValueError("A YAM GELLO leader needs six distinct arm joint IDs")
        if self.gripper_id in self.joint_ids:
            raise ValueError("The GELLO gripper ID must differ from arm joint IDs")
        if self.joint_signs.shape != (6,) or not np.all(
            np.isin(self.joint_signs, (-1.0, 1.0))
        ):
            raise ValueError("GELLO joint_signs must contain six values of -1 or 1")
        if (
            self.joint_offsets.shape != (6,)
            or not np.isfinite(self.joint_offsets).all()
        ):
            raise ValueError("GELLO joint_offsets_rad must contain six finite values")
        if not np.isfinite(
            [self.gripper_open, self.gripper_closed]
        ).all() or np.isclose(self.gripper_open, self.gripper_closed):
            raise ValueError(
                "GELLO gripper open/closed radians must be finite and distinct"
            )
        if not np.isfinite(self.filter_alpha) or not 0 < self.filter_alpha <= 1:
            raise ValueError("GELLO filter_alpha must be in (0, 1]")
        self._time_fn = time_fn
        self._last_joints = None
        self._bus = bus_factory(
            port=self.port,
            ids=(*self.joint_ids, self.gripper_id),
            baudrate=baudrate,
            protocol_version=protocol_version,
            torque_enable_address=torque_enable_address,
            present_position_address=present_position_address,
            present_position_length=present_position_length,
            encoder_resolution=encoder_resolution,
        )

    def read(self) -> tuple[np.ndarray, float]:
        if self._bus is None:
            raise GelloInputError(f"GELLO leader {self.port} is closed")
        raw = np.asarray(self._bus.read_radians(), dtype=float)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise GelloInputError(f"GELLO leader {self.port} needs seven positions")
        joints = (raw[:6] - self.joint_offsets) * self.joint_signs
        # EgoVerse stores gripper opening: 0 is closed and 1 is open.
        opening = (raw[6] - self.gripper_closed) / (
            self.gripper_open - self.gripper_closed
        )
        joints = joint_vector(np.r_[joints, np.clip(opening, 0.0, 1.0)])
        if self._last_joints is not None:
            joints = (
                1.0 - self.filter_alpha
            ) * self._last_joints + self.filter_alpha * joints
        self._last_joints = joints
        return joints.copy(), float(self._time_fn())

    def close(self) -> None:
        bus, self._bus = self._bus, None
        if bus is not None:
            bus.close()


class BimanualGelloReader:
    """Continuously cache both serial leaders so control reads never block."""

    def __init__(
        self,
        arms: Sequence[str],
        leaders: Mapping[str, Mapping],
        leader_factory: Callable = DynamixelGelloLeader,
    ) -> None:
        self.arms = tuple(arms)
        if (
            not self.arms
            or len(set(self.arms)) != len(self.arms)
            or not set(self.arms) <= ARM_OFFSET.keys()
        ):
            raise ValueError("GELLO arms must select left/right without duplicates")
        if set(leaders) != set(self.arms):
            raise ValueError("GELLO leader config must cover exactly the selected arms")
        self.leaders = {}
        self._stop = Event()
        self._lock = Lock()
        self._threads = {}
        self._joints = {}
        self._timestamps = {}
        self._errors = {}
        try:
            for arm in self.arms:
                self.leaders[arm] = leader_factory(**dict(leaders[arm]))
            for arm in self.arms:
                thread = Thread(
                    target=self._read_loop,
                    args=(arm,),
                    name=f"gello-read-{arm}",
                    daemon=True,
                )
                self._threads[arm] = thread
                thread.start()
        except BaseException:
            self._stop.set()
            self._close_leaders()
            raise

    def _read_loop(self, arm: str) -> None:
        leader = self.leaders[arm]
        while not self._stop.is_set():
            try:
                joints, timestamp = leader.read()
                with self._lock:
                    self._joints[arm] = joints
                    self._timestamps[arm] = timestamp
                    self._errors.pop(arm, None)
            except Exception as error:
                with self._lock:
                    self._errors[arm] = error
                self._stop.wait(0.01)
            else:
                # Real sync reads pace themselves; this keeps fake/test leaders
                # from spinning and competing with the 60 Hz control loop.
                self._stop.wait(0.001)

    def read(self) -> GelloSample:
        if self._stop.is_set():
            raise GelloInputError("GELLO reader is closed")
        with self._lock:
            errors = [self._errors[arm] for arm in self.arms if arm in self._errors]
            missing = [arm for arm in self.arms if arm not in self._joints]
            joints = {
                arm: self._joints[arm].copy()
                for arm in self.arms
                if arm in self._joints
            }
            timestamps = {
                arm: self._timestamps[arm]
                for arm in self.arms
                if arm in self._timestamps
            }
        if errors:
            if len(errors) == 1:
                raise errors[0]
            raise ExceptionGroup("Both GELLO serial reads failed", errors)
        if missing:
            raise GelloInputError(
                "Waiting for first GELLO sample: " + ", ".join(missing)
            )
        return GelloSample(joints, timestamps)

    def _close_leaders(self) -> list[Exception]:
        leaders, self.leaders = self.leaders, {}
        errors = []
        for leader in leaders.values():
            try:
                leader.close()
            except Exception as error:
                errors.append(error)
        return errors

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        threads, self._threads = self._threads, {}
        for thread in threads.values():
            thread.join(timeout=2.0)
        errors = self._close_leaders()
        alive = [arm for arm, thread in threads.items() if thread.is_alive()]
        if alive:
            errors.append(
                RuntimeError("GELLO reader threads did not stop: " + ", ".join(alive))
            )
        if errors:
            raise ExceptionGroup("GELLO leader cleanup failed", errors)

    stop = close


def list_serial_ports(root: str | Path = "/dev/serial/by-id") -> tuple[str, ...]:
    """List stable serial symlinks without opening any device."""
    root = Path(root)
    if not root.is_dir():
        return ()
    return tuple(str(path) for path in sorted(root.iterdir()) if path.is_symlink())


def validate_serial_ports(ports: Sequence[str]) -> tuple[str, ...]:
    """Require distinct existing stable device paths; open none."""
    names = tuple(str(port) for port in ports)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("GELLO serial paths must be nonempty and distinct")
    missing = [name for name in names if not Path(name).exists()]
    if missing:
        raise RuntimeError("GELLO serial paths are unavailable: " + ", ".join(missing))
    unstable = [name for name in names if not Path(name).is_symlink()]
    if unstable:
        raise RuntimeError(
            "GELLO serial paths must be stable symlinks: " + ", ".join(unstable)
        )
    resolved = [str(Path(name).resolve()) for name in names]
    if len(set(resolved)) != len(resolved):
        raise ValueError("GELLO serial paths resolve to the same device")
    return names


def validate_can_interfaces(
    channels: Sequence[str], runner: Callable = subprocess.run
) -> tuple[str, ...]:
    """Read-only follower SocketCAN preflight; send no CAN frames."""
    names = tuple(str(channel) for channel in channels)
    if not names or any(not name for name in names):
        raise ValueError("CAN interface names must be nonempty")
    if len(set(names)) != len(names):
        raise ValueError("Follower CAN interfaces must be distinct")
    failures = []
    for name in names:
        try:
            result = runner(
                ["ip", "-details", "link", "show", "dev", name],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "The `ip` command is required for CAN preflight"
            ) from error
        output = f"{result.stdout}\n{result.stderr}"
        lines = result.stdout.splitlines()
        first_line = lines[0] if lines else ""
        flags = (
            first_line.split("<", 1)[1].split(">", 1)[0].split(",")
            if "<" in first_line
            else []
        )
        reason = None
        if result.returncode:
            reason = "missing"
        elif "link/can" not in output:
            reason = "not SocketCAN"
        elif "UP" not in flags:
            reason = "down"
        if reason:
            failures.append(f"{name} ({reason})")
    if failures:
        raise RuntimeError("CAN preflight failed: " + ", ".join(failures))
    return names


@dataclass(frozen=True)
class GelloCommand:
    joints: np.ndarray
    ee_poses: np.ndarray
    active_arms: frozenset[str]
    phase: str


class GelloTeleopControl:
    """Rate-limited relative or absolute GELLO-to-YAM joint control."""

    def __init__(
        self,
        robot,
        frequency: float,
        alignment: str = "relative",
        max_input_age: float = 0.25,
        max_joint_velocity: float = 2.5,
        max_alignment_velocity: float = 0.5,
        activation_gripper_threshold: float = 0.9,
    ) -> None:
        self.robot = robot
        self.arms = tuple(robot.arms)
        if (
            not self.arms
            or len(set(self.arms)) != len(self.arms)
            or not set(self.arms) <= ARM_OFFSET.keys()
        ):
            raise ValueError("Robot arms must select left/right without duplicates")
        self.frequency = float(frequency)
        self.max_input_age = float(max_input_age)
        self.max_joint_velocity = float(max_joint_velocity)
        self.max_alignment_velocity = float(max_alignment_velocity)
        self.activation_gripper_threshold = float(activation_gripper_threshold)
        limits = np.array(
            [
                self.frequency,
                self.max_input_age,
                self.max_joint_velocity,
                self.max_alignment_velocity,
            ]
        )
        if not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("GELLO rates, age, and velocity limits must be positive")
        if not 0 < self.activation_gripper_threshold < 1:
            raise ValueError("GELLO activation gripper threshold must be in (0, 1)")
        self.alignment = str(alignment)
        if self.alignment not in ("relative", "absolute"):
            raise ValueError("GELLO alignment must be `relative` or `absolute`")
        self.home_offsets = (
            {arm: self._home_joints(arm)[:6].copy() for arm in self.arms}
            if self.alignment == "absolute"
            else {arm: np.zeros(6) for arm in self.arms}
        )
        self.last = None
        self.active = {arm: False for arm in self.arms}
        self.relative_offsets = {arm: None for arm in self.arms}
        self.aligning = {arm: False for arm in self.arms}
        self.armed = False
        self._triggers_released = False

    @property
    def active_arms(self) -> frozenset[str]:
        return frozenset(arm for arm in self.arms if self.active[arm])

    def reset(self) -> None:
        self.last = None
        self.disarm()

    def disarm(self) -> None:
        self.armed = False
        self._triggers_released = False
        for arm in self.arms:
            self.active[arm] = False
            self.relative_offsets[arm] = None
            self.aligning[arm] = False

    @property
    def phase(self) -> str:
        """Human-readable activation state for the station UI."""
        if self.active_arms:
            return "active"
        if self.armed and not self._triggers_released:
            return "armed: release both GELLO triggers"
        if self.armed:
            return "armed: press both GELLO triggers"
        return "disarmed"

    def _home_joints(self, arm: str) -> np.ndarray:
        """Configured follower home; GELLO calibrated zeros map onto this pose."""
        home = getattr(self.robot, "home", None)
        if not isinstance(home, Mapping) or arm not in home:
            raise ValueError(f"GELLO alignment requires robot.home[{arm!r}]")
        return joint_vector(home[arm])

    def _toggle_bimanual_arming(self) -> None:
        """Arm both followers, or cancel a pending/active bimanual session."""
        if self.armed or self.active_arms:
            self.disarm()
            return
        self.armed = True
        self._triggers_released = False

    def _activate_bimanual_when_triggered(
        self, measured: np.ndarray, sample: GelloSample
    ) -> None:
        """Require a fresh, simultaneous light squeeze before motion can begin."""
        if not self.armed:
            return
        openings = np.asarray([sample.joints[arm][6] for arm in self.arms])
        if np.all(openings >= self.activation_gripper_threshold):
            self._triggers_released = True
        if not self._triggers_released or not np.all(
            openings <= self.activation_gripper_threshold
        ):
            return
        self.armed = False
        for arm in self.arms:
            self._toggle(arm, measured, sample.joints[arm])

    def _measured_joints(self, obs) -> np.ndarray:
        measured = np.asarray(obs.get("joint_positions"), dtype=float)
        if measured.shape != (14,) or not np.isfinite(measured).all():
            raise GelloInputError("Robot observation needs 14 finite joint positions")
        for arm in self.arms:
            offset = ARM_OFFSET[arm]
            joint_vector(measured[offset : offset + 7])
        return measured.copy()

    def _validate_sample(self, sample: GelloSample, now: float) -> None:
        if set(sample.joints) != set(self.arms):
            raise GelloInputError("GELLO sample must cover exactly the robot arms")
        for arm in self.arms:
            age = now - sample.timestamps[arm]
            if not np.isfinite(age) or age < -1e-3 or age > self.max_input_age:
                raise GelloInputError(
                    f"GELLO {arm} sample age {age:.3f}s exceeds "
                    f"{self.max_input_age:.3f}s"
                )

    def _toggle(self, arm: str, measured: np.ndarray, leader: np.ndarray) -> None:
        self.active[arm] = not self.active[arm]
        if not self.active[arm]:
            self.relative_offsets[arm] = None
            self.aligning[arm] = False
            return
        offset = ARM_OFFSET[arm]
        self.last[offset : offset + 7] = measured[offset : offset + 7]
        if self.alignment == "relative":
            self.relative_offsets[arm] = measured[offset : offset + 6] - leader[:6]
            self.aligning[arm] = False
        else:
            self.relative_offsets[arm] = self.home_offsets[arm].copy()
            self.aligning[arm] = True

    def step(
        self,
        sample: GelloSample,
        obs,
        toggle_arms: Sequence[str] = (),
        toggle_bimanual_arming: bool = False,
        now: float | None = None,
    ) -> GelloCommand:
        now = time.monotonic() if now is None else float(now)
        try:
            measured = self._measured_joints(obs)
            self._validate_sample(sample, now)
        except (TypeError, ValueError, GelloInputError):
            self.disarm()
            raise
        if self.last is None:
            self.last = measured.copy()
        toggles = tuple(toggle_arms)
        if len(set(toggles)) != len(toggles) or not set(toggles) <= set(self.arms):
            self.disarm()
            raise ValueError(
                "toggle_arms must select configured arms without duplicates"
            )
        for arm in toggles:
            self._toggle(arm, measured, sample.joints[arm])
        if toggle_bimanual_arming:
            self._toggle_bimanual_arming()
        self._activate_bimanual_when_triggered(measured, sample)

        commands = self.last.copy()
        for arm in self.arms:
            offset = ARM_OFFSET[arm]
            if not self.active[arm]:
                commands[offset : offset + 7] = measured[offset : offset + 7]
                continue
            target = sample.joints[arm].copy()
            # Relative: clutch offset captured at activation.
            # Absolute: configured home, so calibrated GELLO zeros command home
            # and the current leader pose commands home + (gello - zero).
            target[:6] += self.relative_offsets[arm]
            # GELLO gripper calibration already maps the physical endpoints to
            # EgoVerse's absolute [closed=0, open=1] convention. Unlike the arm
            # joints, applying a relative clutch offset here prevents either
            # endpoint from being reachable after activation.
            target[6] = np.clip(target[6], 0.0, 1.0)
            joint_velocity = (
                self.max_alignment_velocity
                if self.aligning[arm]
                else self.max_joint_velocity
            )
            joint_limit = joint_velocity / self.frequency
            previous = commands[offset : offset + 7].copy()
            delta = target - previous
            command = previous.copy()
            command[:6] += np.clip(delta[:6], -joint_limit, joint_limit)
            # Match the working rl2-yam GELLO path: only arm joints use a
            # per-tick jump guard; the normalized gripper target is direct.
            command[6] = target[6]
            commands[offset : offset + 7] = joint_vector(command)
            if self.aligning[arm] and np.all(np.abs(delta[:6]) <= joint_limit):
                self.aligning[arm] = False

        ee_poses = np.zeros(14)
        for arm in self.arms:
            offset = ARM_OFFSET[arm]
            command = joint_vector(commands[offset : offset + 7])
            try:
                pose = np.asarray(
                    self.robot.forward_kinematics(command[:6], arm), dtype=float
                )
            except Exception:
                self.disarm()
                raise
            if pose.shape != (6,) or not np.isfinite(pose).all():
                self.disarm()
                raise GelloInputError(
                    f"Forward kinematics for {arm} did not return six finite values"
                )
            ee_poses[offset : offset + 7] = np.r_[pose, command[6]]

        active_arms = self.active_arms
        # Both commands and both FK results are validated before either write.
        try:
            for arm in self.arms:
                if arm in active_arms:
                    offset = ARM_OFFSET[arm]
                    self.robot.set_joints(commands[offset : offset + 7].copy(), arm)
        except Exception:
            self.disarm()
            raise
        self.last = commands
        return GelloCommand(commands.copy(), ee_poses, active_arms, self.phase)
