"""Collect YAM demonstrations with two passive USB/Dynamixel GELLO leaders."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from egomimic.robot.yam.gello import (
    BimanualGelloReader,
    GelloInputError,
    GelloTeleopControl,
    list_serial_ports,
    validate_can_interfaces,
    validate_serial_ports,
)
from egomimic.robot.yam_station.cameras import validate_camera_devices
from egomimic.robot.yam_station.episode_writer import EpisodeWriter, next_episode_path
from egomimic.robot.yam_station.interface import ARM_OFFSET, create_robot, joint_vector
from egomimic.robot.yam_station.teleop_dashboard import (
    TeleopDashboard,
    validate_dashboard_config,
    validate_episode_length,
)


def _positive(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be positive") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _leader_specs(config: dict) -> dict[str, dict]:
    common = dict(config["gello"]["bus"])
    common["filter_alpha"] = config["gello"]["input_filter_alpha"]
    return {
        arm: {**common, **dict(spec)}
        for arm, spec in config["gello"]["leaders"].items()
    }


def validate_gello_config(config, require_calibrated: bool = False) -> dict:
    """Validate static configuration without importing a hardware driver."""
    config = copy.deepcopy(dict(config))
    robot = dict(config.get("robot", {}))
    gello = dict(config.get("gello", {}))
    if robot.get("kind") != "yam":
        raise ValueError("GELLO collection currently requires robot.kind=yam")
    arms = tuple(robot.get("arms", ()))
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= ARM_OFFSET.keys():
        raise ValueError("robot.arms must select left/right without duplicates")
    follower_channels = dict(robot.get("channels", {}))
    if set(arms) - follower_channels.keys():
        raise ValueError("Every selected arm needs a follower CAN channel")
    channels = [str(follower_channels[arm]) for arm in arms]
    if any(not channel for channel in channels) or len(set(channels)) != len(channels):
        raise ValueError("Follower CAN channels must be nonempty and distinct")

    calibrated = gello.get("calibrated") is True
    if require_calibrated and not calibrated:
        raise RuntimeError(
            "GELLO calibration is incomplete. Fill stable USB paths, joint signs, "
            "joint offsets, and gripper endpoints, then set gello.calibrated=true."
        )
    if str(gello.get("sdk_version")) != "4.0.5":
        raise ValueError("gello.sdk_version must match dynamixel-sdk==4.0.5")
    bus = dict(gello.get("bus", {}))
    for name in (
        "baudrate",
        "protocol_version",
        "torque_enable_address",
        "present_position_address",
        "present_position_length",
        "encoder_resolution",
    ):
        _positive(bus.get(name), f"gello.bus.{name}")
    if int(bus["present_position_length"]) not in (1, 2, 4):
        raise ValueError("gello.bus.present_position_length must be 1, 2, or 4")

    leaders = dict(gello.get("leaders", {}))
    if set(leaders) != set(arms):
        raise ValueError("gello.leaders must cover exactly the selected arms")
    ports = []
    for arm, raw_spec in leaders.items():
        spec = dict(raw_spec)
        port = str(spec.get("port", ""))
        if not port.startswith("/dev/serial/by-id/"):
            raise ValueError(
                f"gello.leaders.{arm}.port must be a stable /dev/serial/by-id path"
            )
        ports.append(port)
        joint_ids = tuple(int(value) for value in spec.get("joint_ids", ()))
        if (
            len(joint_ids) != 6
            or len(set(joint_ids)) != 6
            or any(not 0 <= value <= 252 for value in joint_ids)
        ):
            raise ValueError(
                f"gello.leaders.{arm}.joint_ids must be six distinct IDs in [0, 252]"
            )
        gripper_id = int(spec.get("gripper_id", -1))
        if not 0 <= gripper_id <= 252 or gripper_id in joint_ids:
            raise ValueError(
                f"gello.leaders.{arm}.gripper_id must be distinct and in [0, 252]"
            )
        signs = np.asarray(spec.get("joint_signs"), dtype=float)
        if signs.shape != (6,) or not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError(
                f"gello.leaders.{arm}.joint_signs must contain six values of -1 or 1"
            )
        offsets = np.asarray(spec.get("joint_offsets_rad"), dtype=float)
        if offsets.shape != (6,) or not np.isfinite(offsets).all():
            raise ValueError(
                f"gello.leaders.{arm}.joint_offsets_rad must contain six finite values"
            )
        endpoints = (spec.get("gripper_open_rad"), spec.get("gripper_closed_rad"))
        if calibrated or any(value is not None for value in endpoints):
            try:
                endpoints = np.asarray(endpoints, dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"gello.leaders.{arm} needs numeric gripper endpoints"
                ) from error
            if not np.isfinite(endpoints).all() or np.isclose(*endpoints):
                raise ValueError(
                    f"gello.leaders.{arm} gripper endpoints must be finite and distinct"
                )
    if len(set(ports)) != len(ports):
        raise ValueError("GELLO leaders must use distinct serial paths")

    frequency = _positive(config.get("frequency"), "frequency")
    _positive(config.get("startup_timeout"), "startup_timeout")
    robot = dict(config.get("robot", {}))
    for name in ("gripper_kp", "gripper_kd", "gripper_force_limit"):
        _positive(robot.get(name), f"robot.{name}")
    try:
        input_filter_alpha = float(gello["input_filter_alpha"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("gello.input_filter_alpha must be in (0, 1]") from error
    if not np.isfinite(input_filter_alpha) or not 0 < input_filter_alpha <= 1:
        raise ValueError("gello.input_filter_alpha must be in (0, 1]")
    for name in (
        "max_input_age",
        "max_joint_velocity",
        "max_alignment_velocity",
    ):
        _positive(gello.get(name), f"gello.{name}")
    if gello.get("alignment") not in ("relative", "absolute"):
        raise ValueError("gello.alignment must be `relative` or `absolute`")
    if gello.get("alignment") == "absolute":
        homes = dict(robot.get("home", {}))
        for arm in arms:
            try:
                joint_vector(homes[arm])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"gello.alignment=absolute requires robot.home.{arm} "
                    "as a 7D joint pose"
                ) from error
    try:
        trigger_threshold = float(gello["activation_gripper_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "gello.activation_gripper_threshold must be in (0, 1)"
        ) from error
    if not 0 < trigger_threshold < 1:
        raise ValueError("gello.activation_gripper_threshold must be in (0, 1)")
    recording = dict(config.get("recording", {}))
    validate_episode_length(recording.get("episode_length"))
    record_rate = _positive(recording.get("rate_hz", frequency), "recording.rate_hz")
    if record_rate > frequency:
        raise ValueError("Recording cannot be faster than GELLO control")
    keys = dict(config.get("keys", {}))
    expected_keys = {
        "activate_both",
        "record",
        "stop",
        "home",
        "quit",
    }
    if set(keys) != expected_keys:
        raise ValueError("GELLO keys must define activation and collection controls")
    if any(not isinstance(value, str) or len(value) != 1 for value in keys.values()):
        raise ValueError("Every GELLO keyboard binding must be one character")
    if len({value.lower() for value in keys.values()}) != len(keys):
        raise ValueError("GELLO keyboard bindings must be distinct")
    validate_dashboard_config(config.get("preview"))
    return config


def _control(robot, config) -> GelloTeleopControl:
    gello = config["gello"]
    return GelloTeleopControl(
        robot,
        frequency=config["frequency"],
        alignment=gello["alignment"],
        max_input_age=gello["max_input_age"],
        max_joint_velocity=gello["max_joint_velocity"],
        max_alignment_velocity=gello["max_alignment_velocity"],
        activation_gripper_threshold=gello["activation_gripper_threshold"],
    )


class KeyEdge:
    """Suppress held-key repeats until a frame reports no key."""

    def __init__(self) -> None:
        self.last = None

    def update(self, key):
        key = key.lower() if isinstance(key, str) else key
        pressed = key if key is not None and key != self.last else None
        self.last = key
        return pressed


class TerminalKeyView:
    """Nonblocking single-key input for teleop without camera windows."""

    def __init__(
        self,
        stream=None,
        termios_module=None,
        tty_module=None,
        select_fn=None,
    ) -> None:
        if stream is None:
            import sys

            stream = sys.stdin
        if termios_module is None:
            import termios as termios_module
        if tty_module is None:
            import tty as tty_module
        if select_fn is None:
            from select import select as select_fn

        if not stream.isatty():
            raise RuntimeError(
                "GELLO teleop without camera preview requires an interactive TTY"
            )
        self.stream = stream
        self._termios = termios_module
        self._select = select_fn
        self._fd = stream.fileno()
        self._settings = termios_module.tcgetattr(self._fd)
        tty_module.setcbreak(self._fd)

    def update(self, obs, recording=False):
        readable, _, _ = self._select([self.stream], [], [], 0.0)
        if not readable:
            return None
        key = self.stream.read(1)
        return key or None

    def close(self) -> None:
        settings, self._settings = self._settings, None
        if settings is not None:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, settings)


def _take_camera_reconnect_request(view) -> bool:
    """Consume one dashboard-only RGB recovery request, if the view has one."""
    take = getattr(view, "take_camera_reconnect_request", None)
    return bool(take()) if callable(take) else False


def run_collection(robot, reader, config, view=None, max_steps=None):
    """Run direct leader/follower control and incremental HDF5 capture."""
    config = validate_gello_config(config)
    frequency = float(config["frequency"])
    recording = config["recording"]
    episode_length = validate_episode_length(recording.get("episode_length"))
    record_rate = float(recording.get("rate_hz", frequency))
    control = _control(robot, config)
    if view is None:
        view = (
            TeleopDashboard(
                robot.camera_res,
                config["keys"],
                recording["directory"],
                gripper_force_limits={
                    arm: config["robot"]["gripper_force_limit"] for arm in robot.arms
                },
                **config["preview"],
            )
            if config["preview"]["enabled"]
            else TerminalKeyView()
        )
    keys = {action: value.lower() for action, value in config["keys"].items()}
    key_edge = KeyEdge()
    episode_id = int(
        next_episode_path(
            recording["directory"], recording["episode_start"]
        ).stem.removeprefix("demo_")
    )
    writer, steps, ready, record_phase = None, 0, False, 0.0
    startup_complete = False
    record_pending = False
    arm_record_on_next_sample = False
    started = time.monotonic()
    last_warning = -np.inf

    def update_dashboard_status():
        set_status = getattr(view, "set_status", None)
        if set_status is not None:
            # Recording has its own red indicator. Keep the follower state
            # visible so arming/disarming never disappears from the dashboard.
            set_status(control.phase)

    def update_dashboard_episode():
        set_episode = getattr(view, "set_episode", None)
        if set_episode is not None:
            state = (
                "recording"
                if writer is not None
                else "queued"
                if record_pending
                else "next"
            )
            set_episode(episode_id, state)

    def advance_episode():
        nonlocal episode_id
        episode_id = int(
            next_episode_path(recording["directory"], episode_id + 1).stem.removeprefix(
                "demo_"
            )
        )
        update_dashboard_episode()

    def begin_recording():
        path = Path(recording["directory"]) / f"demo_{episode_id}.hdf5"
        return EpisodeWriter(path, robot.camera_res)

    def update_dashboard_limit():
        set_limit = getattr(view, "set_episode_length", None)
        if callable(set_limit):
            set_limit(
                episode_length, writer.frames if writer is not None else 0, record_rate
            )

    update_dashboard_episode()
    update_dashboard_limit()
    try:
        while max_steps is None or steps < max_steps:
            tick = time.monotonic()
            obs = robot.get_obs()
            cameras_ready = all(obs.get(name) is not None for name in robot.camera_res)
            sample = None
            try:
                sample = reader.read()
            except Exception as error:
                control.disarm()
                if record_pending:
                    arm_record_on_next_sample = True
                if tick - last_warning >= 1.0:
                    print(f"GELLO input unavailable; followers hold position: {error}")
                    last_warning = tick
            update_dashboard_status()
            update_dashboard_limit()
            view_event = view.update(obs, recording=writer is not None)
            if _take_camera_reconnect_request(view):
                # A camera recovery never commands or closes the follower. Hold
                # it first, preserve an in-progress take, then require the
                # normal g + release/squeeze gate before teleop can resume.
                control.disarm()
                record_pending = False
                arm_record_on_next_sample = False
                if writer is not None:
                    writer_path = writer.path
                    writer.close(complete=False)
                    writer = None
                    record_phase = 0.0
                    print(f"Preserved interrupted episode: {writer_path}")
                    advance_episode()
                update_dashboard_episode()
                set_status = getattr(view, "set_status", None)
                if callable(set_status):
                    set_status("Reconnecting RGB cameras — followers disarmed")
                reconnect = getattr(robot, "reconnect_cameras", None)
                try:
                    if not callable(reconnect):
                        raise RuntimeError(
                            "This robot runtime does not support camera recovery"
                        )
                    reconnect()
                except Exception as error:
                    print(f"Camera reconnect failed; replug and try again: {error}")
                    if callable(set_status):
                        set_status("Camera reconnect failed — replug and try again")
                else:
                    print("RGB cameras reconnected; followers remain disarmed.")
                    if callable(set_status):
                        set_status(
                            "RGB cameras reconnected — press g, then squeeze both GELLO triggers"
                        )
                # The prior frame and leader sample were taken before recovery.
                # Reacquire both on a future tick before accepting any command.
                ready = False
                steps += 1
                time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
                continue
            if isinstance(view_event, dict) and "episode" in view_event:
                requested_episode = int(view_event["episode"])
                requested_path = (
                    Path(recording["directory"]) / f"demo_{requested_episode}.hdf5"
                )
                if writer is not None:
                    print("Episode ID cannot change while recording is active.")
                elif requested_path.exists():
                    print(
                        f"Cannot use existing {requested_path.name}; episode ID unchanged."
                    )
                else:
                    episode_id = requested_episode
                    print(f"Next recording will use demo_{episode_id}.hdf5")
                update_dashboard_episode()
                view_event = None
            if isinstance(view_event, dict) and "episode_length" in view_event:
                try:
                    episode_length = validate_episode_length(
                        view_event["episode_length"]
                    )
                except ValueError as error:
                    print(f"Episode limit unchanged: {error}")
                else:
                    print(
                        "Episode limit disabled; save recording manually."
                        if episode_length is None
                        else f"Episode auto-save limit: {episode_length} frames."
                    )
                update_dashboard_limit()
                view_event = None
            if isinstance(view_event, dict) and "gripper_force" in view_event:
                request = view_event["gripper_force"]
                try:
                    limits = robot.set_gripper_force_limit(
                        request["value"], arm=request["arm"]
                    )
                    set_limits = getattr(view, "set_gripper_force_limits", None)
                    if callable(set_limits):
                        set_limits(limits)
                    print(
                        f"Updated {request['arm']} gripper force limit to {request['value']:g} N"
                    )
                except (KeyError, TypeError, ValueError, RuntimeError) as error:
                    print(f"Could not update gripper force limit: {error}")
                view_event = None
            event = key_edge.update(view_event)
            if event in (keys["quit"], "\x1b"):
                break

            if event in (keys["stop"], keys["home"]):
                record_pending = False
                arm_record_on_next_sample = False
                if writer is not None:
                    writer_path = writer.path
                    writer.close(complete=False)
                    writer = None
                    record_phase = 0.0
                    if event == keys["stop"]:
                        # X explicitly discards the current take. Remove the
                        # incomplete file so the same episode ID remains usable.
                        writer_path.unlink()
                        print(f"Discarded {writer_path}; episode ID unchanged.")
                    else:
                        print(f"Preserved interrupted episode: {writer_path}")
                        advance_episode()
                if event == keys["home"]:
                    control.reset()
                    robot.set_home()
                    obs = robot.get_obs()
                else:
                    control.disarm()
                    print("GELLO followers disarmed.")
                update_dashboard_episode()

            input_ready = sample is not None
            if not ready:
                ready = cameras_ready and input_ready
                if ready:
                    startup_complete = True
                elif not startup_complete and tick - started > float(
                    config["startup_timeout"]
                ):
                    raise TimeoutError(
                        "Waiting for configured cameras and both USB GELLO leaders; "
                        "check serial paths, Dynamixel IDs/baudrate, and camera serials."
                    )

            if event == keys["record"] and ready and recording["enabled"]:
                if writer is not None:
                    writer.close()
                    print(f"Saved {writer.path} ({writer.frames} frames)")
                    writer = None
                    record_phase = 0.0
                    advance_episode()
                elif record_pending:
                    record_pending = False
                    arm_record_on_next_sample = False
                    control.disarm()
                    update_dashboard_episode()
                    print("Cancelled pending recording; GELLO followers disarmed.")
                else:
                    control.reset()
                    robot.set_home()
                    obs = robot.get_obs()
                    # The sample was captured before homing and is now stale.
                    # Arm from a fresh sample on the next control tick.
                    sample = None
                    record_pending = True
                    arm_record_on_next_sample = True
                    update_dashboard_episode()
                    print(
                        "Reset home; recording queued. Release, then press both "
                        "GELLO triggers to start recording and move followers "
                        "to the calibrated GELLO pose."
                    )

            command = None
            if ready and sample is not None:
                try:
                    arm_for_record = arm_record_on_next_sample
                    command = control.step(
                        sample,
                        obs,
                        toggle_bimanual_arming=(
                            event == keys["activate_both"] or arm_for_record
                        ),
                        now=time.monotonic(),
                    )
                    if arm_for_record:
                        arm_record_on_next_sample = False
                    if event == keys["activate_both"]:
                        print(f"GELLO followers: {command.phase}")
                except (GelloInputError, TypeError, ValueError) as error:
                    control.disarm()
                    if record_pending:
                        arm_record_on_next_sample = True
                    if tick - last_warning >= 1.0:
                        print(f"GELLO input rejected; followers hold position: {error}")
                        last_warning = tick
            if (
                record_pending
                and command is not None
                and command.active_arms == frozenset(robot.arms)
            ):
                writer = begin_recording()
                record_pending = False
                record_phase = 0.0
                update_dashboard_episode()
                print(
                    f"GELLO active; recording {writer.path} while aligning to "
                    "the calibrated leader pose"
                )
            update_dashboard_status()
            if writer is not None and command is not None and cameras_ready:
                record_phase += record_rate / frequency
                if writer.frames == 0 or record_phase >= 1.0:
                    writer.append(obs, command.joints, command.ee_poses)
                    record_phase = max(0.0, record_phase - 1.0)
            # Also apply a newly lowered limit when no new camera row arrived.
            if (
                writer is not None
                and episode_length is not None
                and writer.frames >= episode_length
            ):
                writer.close()
                print(f"Saved {writer.path} ({writer.frames} frames)")
                writer = None
                record_phase = 0.0
                advance_episode()
            update_dashboard_limit()
            steps += 1
            time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
    finally:
        if writer is not None:
            writer.close(complete=False)
            print(f"Preserved interrupted episode: {writer.path}")
        view.close()
    return steps


def _without_cameras(config: dict) -> dict:
    config = copy.deepcopy(config)
    config["robot"]["cameras"] = {}
    config["preview"]["enabled"] = False
    config["recording"]["enabled"] = False
    return config


def _follower_can_channels(config: dict) -> tuple[str, ...]:
    return tuple(config["robot"]["channels"][arm] for arm in config["robot"]["arms"])


def _gello_serial_ports(config: dict) -> tuple[str, ...]:
    return tuple(
        config["gello"]["leaders"][arm]["port"] for arm in config["robot"]["arms"]
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAM GELLO collection YAML")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate YAML only; import no drivers and open no devices",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="list stable serial paths without opening devices",
    )
    parser.add_argument(
        "--check-devices",
        action="store_true",
        help="check paths, follower CAN, and cameras without opening motors",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="teleop only: disable cameras, preview, and HDF5 recording",
    )
    args = parser.parse_args(argv)
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.no_cameras:
        config = _without_cameras(config)
    config = validate_gello_config(config)
    if args.list_ports:
        ports = list_serial_ports()
        print("\n".join(ports) if ports else "No stable serial devices found.")
        return 0
    if args.check_config:
        calibration = "complete" if config["gello"]["calibrated"] else "required"
        print(
            "GELLO configuration structure is valid; no devices were opened. "
            f"Hardware calibration: {calibration}."
        )
        return 0

    if args.check_devices:
        validate_serial_ports(_gello_serial_ports(config))
        validate_can_interfaces(_follower_can_channels(config))
        validate_camera_devices(config["robot"]["cameras"])
        print("GELLO paths, follower CAN, and cameras passed read-only preflight.")
        return 0
    config = validate_gello_config(config, require_calibrated=True)
    dashboard = None
    if config["preview"]["enabled"]:
        camera_names = tuple(
            name
            for name, spec in config["robot"]["cameras"].items()
            if spec.get("enabled", True)
        )
        dashboard = TeleopDashboard(
            camera_names,
            config["keys"],
            config["recording"]["directory"],
            gripper_force_limits={
                arm: config["robot"]["gripper_force_limit"]
                for arm in config["robot"]["arms"]
            },
            **config["preview"],
        )
        dashboard.set_status("Starting: checking USB GELLO paths")

    robot = reader = None
    try:
        validate_serial_ports(_gello_serial_ports(config))
        if dashboard is not None:
            dashboard.set_status("Starting: checking follower CAN")
        validate_can_interfaces(_follower_can_channels(config))
        if dashboard is not None:
            dashboard.set_status("Starting: checking camera serials")
        validate_camera_devices(config["robot"]["cameras"])
        if dashboard is not None:
            dashboard.set_status("Starting: opening passive GELLO leaders")
        reader = BimanualGelloReader(
            arms=config["robot"]["arms"], leaders=_leader_specs(config)
        )
        if dashboard is not None:
            dashboard.set_status("Starting: opening YAM followers and cameras")
        robot = create_robot(config["robot"])
        if dashboard is not None:
            dashboard.set_status("Waiting for fresh cameras and GELLO input")
        run_collection(robot, reader, config, view=dashboard)
        dashboard = None  # run_collection owns and closes the supplied view.
    except KeyboardInterrupt:
        pass
    finally:
        errors = []
        if robot is not None:
            try:
                robot.close()
            except Exception as error:
                errors.append(error)
        if reader is not None:
            try:
                reader.close()
            except Exception as error:
                errors.append(error)
        if dashboard is not None:
            dashboard.close()
        if errors:
            raise ExceptionGroup("GELLO collector cleanup failed", errors)
    return 0


if __name__ == "__main__":
    main()
