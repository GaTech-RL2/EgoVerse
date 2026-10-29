"""Shared Eva/Yam Quest collection with the existing demonstration HDF5 format."""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from egomimic.robot.yam_station.cameras import CameraView
from egomimic.robot.yam_station.interface import create_robot
from egomimic.robot.yam_station.teleop import (
    ButtonEdge,
    TeleopControl,
    WorldFrameTeleop,
)

VECTOR_KEYS = (
    "observations/joints",
    "observations/joint_positions",
    "observations/eepose",
    "actions/eepose",
    "actions/joints",
    "action",
)


class EpisodeWriter:
    """Append complete synchronized rows; exclusive creation never overwrites a demo."""

    def __init__(self, path, camera_res):
        self.path, self.camera_res = Path(path), dict(camera_res)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "x")
        self.file.attrs["sim"] = False
        self.file.attrs["complete"] = False
        self.frames = 0
        self.file.require_group("observations/images")
        for key in VECTOR_KEYS:
            self.file.create_dataset(
                key, (0, 14), maxshape=(None, 14), dtype="float32", chunks=(1, 14)
            )
        for camera, (height, width) in self.camera_res.items():
            shape = (height, width, 3)
            self.file.create_dataset(
                f"observations/images/{camera}",
                (0, *shape),
                maxshape=(None, *shape),
                dtype="uint8",
                chunks=(1, *shape),
            )
        self.file.flush()

    def append(self, obs, joints, ee_pose):
        values = {
            "observations/joints": obs["joint_positions"],
            "observations/joint_positions": obs["joint_positions"],
            "observations/eepose": obs["ee_poses"],
            "actions/eepose": ee_pose,
            "actions/joints": joints,
            "action": joints,
        }
        for key, value in values.items():
            value = np.asarray(value)
            if value.shape != (14,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid 14D recording row: {key}")
        for name, (height, width) in self.camera_res.items():
            image = obs.get(name)
            if (
                image is None
                or image.shape != (height, width, 3)
                or image.dtype != np.uint8
            ):
                raise ValueError(f"Missing or malformed camera frame: {name}")
            # The existing drivers publish BGR; the existing HDF5 files store RGB.
            values[f"observations/images/{name}"] = image[..., ::-1]
        for key, value in values.items():
            dataset = self.file[key]
            dataset.resize(self.frames + 1, axis=0)
            dataset[self.frames] = value
        self.frames += 1
        self.file.flush()

    def close(self, complete=True):
        if self.file is not None:
            self.file.attrs["complete"] = bool(complete)
            self.file.close()
            self.file = None


def next_episode_path(directory, start=0):
    directory = Path(directory)
    episode = int(start)
    while (directory / f"demo_{episode}.hdf5").exists():
        episode += 1
    return directory / f"demo_{episode}.hdf5"


def save_demo(demo_data, demo_dir, episode_id, camera_res):
    """Compatibility entry point; measured EEF poses come from the robot observations."""
    count = len(demo_data["obs"])
    if not count or any(
        len(demo_data[key]) != count
        for key in ("cmd_joint_actions", "cmd_eepose_actions")
    ):
        raise ValueError(
            "A demo must contain equally sized nonempty observation/action lists"
        )
    writer = EpisodeWriter(Path(demo_dir) / f"demo_{episode_id}.hdf5", camera_res)
    complete = False
    try:
        for obs, joints, pose in zip(
            demo_data["obs"],
            demo_data["cmd_joint_actions"],
            demo_data["cmd_eepose_actions"],
        ):
            writer.append(obs, joints, pose)
        complete = True
    finally:
        writer.close(complete)
    return True


def start_reader(config):
    # The bundled reader is an existing standalone package used by Eva.
    package = str(Path(__file__).parent / "oculus_reader")
    if package not in sys.path:
        sys.path.insert(0, package)
    from oculus_reader import OculusReader

    return OculusReader(
        pose_frame="world",
        max_age=float(config["max_age"]),
        ip_address=config.get("ip_address"),
    )


def run_collection(robot, reader, config, view=None, max_steps=None):
    frequency = float(config["frequency"])
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("Collection frequency must be positive")
    record = config["recording"]
    if int(record["episode_length"]) <= 0:
        raise ValueError("episode_length must be positive")
    record_rate = float(record.get("rate_hz", frequency))
    if not np.isfinite(record_rate) or not 0 < record_rate <= frequency:
        raise ValueError("Recording rate must be positive and no faster than control")
    control = TeleopControl(
        robot,
        WorldFrameTeleop(robot.arms, **config["teleop"]),
        frequency,
        config["max_joint_velocity"],
    )
    view = view or CameraView(robot.camera_res, **config["preview"])
    edges = {action: ButtonEdge(key) for action, key in config["buttons"].items()}
    writer, steps, ready, record_phase = None, 0, False, 0.0
    started = time.monotonic()
    try:
        while max_steps is None or steps < max_steps:
            tick = time.monotonic()
            obs = robot.get_obs()
            poses, buttons = reader.get_transformations_and_buttons()
            key = view.update(obs, recording=writer is not None)
            events = {action: edge.pressed(buttons) for action, edge in edges.items()}
            if events.get("quit") or key in ("q", "\x1b"):
                break
            cameras_ready = all(obs.get(name) is not None for name in robot.camera_res)
            if not ready:
                ready = cameras_ready and all(arm[0] in poses for arm in robot.arms)
                if not ready and tick - started > float(config["startup_timeout"]):
                    raise TimeoutError(
                        "Waiting for cameras and Quest WORLD poses. Install the "
                        "bundled RAIL world-frame APK (wE9ryARX tag); check "
                        "camera configuration."
                    )
            if events.get("stop") or key == "x" or events.get("home") or key == "y":
                if writer is not None:
                    writer.close(complete=False)
                    print(f"Preserved interrupted episode: {writer.path}")
                    writer = None
                    record_phase = 0.0
                if events.get("home") or key == "y":
                    control.reset()
                    robot.set_home()
                    # Re-read observations after home before sending another command.
                    obs = robot.get_obs()
            if (events.get("record") or key == "b") and ready and record["enabled"]:
                if writer is not None:
                    writer.close()
                    print(f"Saved {writer.path} ({writer.frames} frames)")
                    writer = None
                    record_phase = 0.0
                else:
                    writer = EpisodeWriter(
                        next_episode_path(record["directory"], record["episode_start"]),
                        robot.camera_res,
                    )
                    record_phase = 0.0
                    print(f"Recording {writer.path}")
            if ready:
                joints, ee_pose = control.step(poses, buttons, obs)
                if writer is not None and cameras_ready:
                    record_phase += record_rate / frequency
                    if writer.frames == 0 or record_phase >= 1.0:
                        writer.append(obs, joints, ee_pose)
                        record_phase = max(0.0, record_phase - 1.0)
                        if writer.frames >= int(record["episode_length"]):
                            writer.close()
                            print(f"Saved {writer.path} ({writer.frames} frames)")
                            writer = None
                            record_phase = 0.0
            steps += 1
            time.sleep(max(0.0, 1 / frequency - (time.monotonic() - tick)))
    finally:
        if writer is not None:
            writer.close(complete=False)
            print(f"Preserved interrupted episode: {writer.path}")
        view.close()
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Eva or Yam collection YAML")
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    robot = reader = None
    try:
        reader = start_reader(config["quest"])
        robot = create_robot(config["robot"])
        run_collection(robot, reader, config)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if robot is not None:
                robot.close()
        finally:
            if reader is not None:
                reader.stop()


if __name__ == "__main__":
    main()
