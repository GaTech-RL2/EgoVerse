"""Evaluator-only binding to the official, unmodified LIBERO release."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import digest, file_hash

LIBERO_COMMIT = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"


def configure(root, config_directory):
    root, config_directory = Path(root).resolve(), Path(config_directory).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    if head != LIBERO_COMMIT or dirty.strip():
        raise ValueError("LIBERO_source_not_pinned_clean")
    package = root / "libero/libero"
    config_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "assets": str(package / "assets"),
        "datasets": str(config_directory / "unavailable-demonstrations"),
    }
    config_file = config_directory / "config.yaml"
    if config_file.exists():
        if json.loads(config_file.read_text()) != paths:
            raise ValueError("existing_LIBERO_configuration_differs")
    else:
        with config_file.open("x") as stream:
            stream.write(json.dumps(paths))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_directory)
    sys.path.insert(0, str(root))
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()


def array_hash(array):
    array = np.ascontiguousarray(array, dtype=np.float64)
    import hashlib

    return hashlib.sha256(array.tobytes()).hexdigest()


def catalog(root, registry, indices):
    suite = registry["libero_10"](task_order_index=0)
    if suite.n_tasks != 10:
        raise ValueError("unexpected_long_suite")
    rows = []
    for task_id in range(10):
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        state_file = (
            Path(root)
            / "libero/libero/init_files"
            / task.problem_folder
            / task.init_states_file
        )
        bddl = (
            Path(root)
            / "libero/libero/bddl_files"
            / task.problem_folder
            / task.bddl_file
        )
        rows.append(
            {
                "task_id": task_id,
                "name": task.name,
                "instruction": task.language,
                "bddl_sha256": file_hash(bddl),
                "init_file_sha256": file_hash(state_file),
                "init_state_hashes": {str(i): array_hash(states[i]) for i in indices},
            }
        )
    return rows


class Environment:
    def __init__(
        self,
        registry,
        root,
        suite_name,
        task_id,
        init_index,
        seed,
        *,
        settling_steps=10,
        horizon=1000,
        image_size=128,
    ):
        import robosuite
        from libero.libero.envs import OffScreenRenderEnv

        if robosuite.__version__ != "1.4.1":
            raise ValueError("robosuite_version")
        np.random.seed(seed)
        random.seed(seed)
        suite = registry[suite_name](task_order_index=0)
        self.task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        initial = np.asarray(states[init_index], dtype=np.float64)
        bddl = (
            Path(root)
            / "libero/libero/bddl_files"
            / self.task.problem_folder
            / self.task.bddl_file
        )
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=image_size,
            camera_widths=image_size,
            control_freq=20,
            horizon=horizon + settling_steps,
            ignore_done=True,
        )
        try:
            self.env.seed(seed)
            self.env.reset()
            observation = self.env.set_init_state(initial)
            actual = np.asarray(self.env.get_sim_state(), dtype=np.float64)
            if not np.array_equal(actual, initial):
                raise ValueError("official_state_setter_mismatch")
            for _ in range(settling_steps):
                observation, _, _, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
            self.observation = observation
            self.reset_receipt = {
                "official_state_sha256": array_hash(initial),
                "settled_state_sha256": array_hash(self.env.get_sim_state()),
                "model_body_pos_sha256": array_hash(self.env.sim.model.body_pos),
                "model_body_quat_sha256": array_hash(self.env.sim.model.body_quat),
                "settling_steps": settling_steps,
                "settling_action": [0, 0, 0, 0, 0, 0, -1],
                "bddl_sha256": file_hash(bddl),
                "initial_success": bool(self.env.check_success()),
            }
            self.controller = self._controller()
        except BaseException:
            self.close()
            raise

    def _controller(self):
        robot = self.env.robots[0]
        controller = robot.controller
        low, high = robot.action_limits
        if (
            len(low) != 7
            or not controller.use_delta
            or controller.impedance_mode != "fixed"
        ):
            raise ValueError("unsupported_controller")
        cfg = {
            "version": "robosuite-1.4.1:OSC_POSE",
            "frequency_hz": 20,
            "input_min": np.asarray(low).tolist(),
            "input_max": np.asarray(high).tolist(),
            "output_min_pose": controller.output_min.tolist(),
            "output_max_pose": controller.output_max.tolist(),
            "frame": "world",
            "orientation": "axis-angle delta, left-multiplied onto current world rotation",
            "action_order": ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"],
            "gripper": "-1 open, +1 close; sign integrates finger command with internal clipping",
            "normalization": "pose affine scaling after controller input clipping; actor out-of-range inputs rejected",
            "position_limits": None
            if controller.position_limits is None
            else controller.position_limits.tolist(),
            "orientation_limits": None
            if controller.orientation_limits is None
            else controller.orientation_limits.tolist(),
            "collision_checks": "unchanged MuJoCo collision/contact dynamics",
            "hazardous_constraint": "out-of-controller-input-bounds request; attempted violation, never applied",
        }
        if cfg["input_min"] != [-1.0] * 7 or cfg["input_max"] != [1.0] * 7:
            raise ValueError("unexpected_controller_bounds")
        return cfg

    def read_sensors(self):
        self.env._update_observables(force=True)
        return self.env.env._get_observations()

    def step(self, action):
        return self.env.step(np.asarray(action, dtype=np.float64))

    def check_success(self):
        return self.env.check_success()

    def terminal_snapshot(self, destination):
        np.savez_compressed(
            destination,
            state=self.env.get_sim_state(),
            body_pos=self.env.sim.model.body_pos,
            body_quat=self.env.sim.model.body_quat,
        )

    def close(self):
        self.env.close()


def asset_manifest(root):
    root = Path(root)
    files = {}
    for name in ("bddl_files", "init_files", "assets"):
        for path in sorted((root / "libero/libero" / name).rglob("*")):
            if path.is_file():
                if path.is_symlink():
                    raise ValueError("symlink_benchmark_asset")
                files[str(path.relative_to(root))] = file_hash(path)
    return {
        "files": files,
        "sha256": digest(files),
        "source_commit": LIBERO_COMMIT,
        "license": "LIBERO code MIT; benchmark/dataset attribution retained; no expert demonstrations downloaded",
    }
