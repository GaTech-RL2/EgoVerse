"""Pinned LIBERO protocols, frozen reset manifests, and simulator execution."""

import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .checkpoint import inspect_checkpoint
from .config import BenchmarkConfig
from .records import digest


def revision(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def configure_libero(root, benchmark: BenchmarkConfig, runtime_directory):
    """Use a process-local path config, never ~/.libero or another installation."""
    root = Path(root).resolve()
    package = root / "libero" / "libero"
    if not (package / "benchmark" / "__init__.py").is_file():
        raise ValueError(
            f"LIBERO package not found at {package}; initialize a separate pinned checkout"
        )
    if revision(root) != benchmark.environment_revision:
        raise ValueError(f"LIBERO revision must be {benchmark.environment_revision}")
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", "."],
        capture_output=True,
        text=True,
        check=True,
    )
    if status.stdout.strip():
        raise ValueError(
            "LIBERO source/assets have local changes; pin them as a separate protocol"
        )
    runtime = Path(runtime_directory).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    config_path = runtime / "config.yaml"
    paths = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "datasets": str(root / "datasets"),
        "assets": str(package / "assets"),
    }
    # JSON is valid YAML and avoids importing LIBERO before its config exists.
    if config_path.exists() and json.loads(config_path.read_text()) != paths:
        raise ValueError(
            "Runtime path configuration already belongs to another checkout"
        )
    config_path.write_text(json.dumps(paths, indent=2) + "\n")
    for name, module in sys.modules.items():
        if (
            name == "libero.libero"
            and Path(module.__file__).resolve().parent != package
        ):
            raise ValueError(
                "A different LIBERO package is already imported; use a fresh process"
            )
    os.environ["LIBERO_CONFIG_PATH"] = str(runtime)
    sys.path.insert(0, str(root))
    from libero.libero import benchmark as registry
    from libero.libero.envs import OffScreenRenderEnv

    suite_map = registry.get_benchmark_dict()
    if benchmark.suite not in suite_map:
        raise ValueError(f"Pinned environment does not register {benchmark.suite}")
    suite = suite_map[benchmark.suite]()
    if suite.n_tasks != 10:
        raise ValueError("Published suite must have exactly ten registered tasks")
    if benchmark.suite != "libero_10":
        for name in ("libero_goal_ood", "libero_spatial_ood"):
            if name not in suite_map or suite_map[name]().n_tasks != 10:
                raise ValueError(
                    "Both published OOD suites must register all twenty tasks"
                )

    def create(task_id, seed):
        task = suite.get_task(task_id)
        bddl = package / "bddl_files" / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
            control_freq=benchmark.control_frequency_hz,
        )
        env.seed(seed)
        return env, task, bddl

    return suite, create


def create_task_manifest(root, config, output, *, task_ids=None, trials=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace frozen manifest {output}")
    protocol = config.benchmark
    ids = list(range(10)) if task_ids is None else list(task_ids)
    count = protocol.trials_per_task if trials is None else trials
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(type(i) is not int or i < 0 or i >= 10 for i in ids)
    ):
        raise ValueError("Task IDs must be distinct integers in [0, 9]")
    if type(count) is not int or count < 1 or count > protocol.trials_per_task:
        raise ValueError("Trial count must be within the preset")
    if config.evaluation.split == "test" and (
        ids != list(range(10)) or count != protocol.trials_per_task
    ):
        raise ValueError(
            "Test manifests must include the complete published suite and trial count"
        )
    suite, create = configure_libero(
        root, protocol, output.parent / (output.stem + "_libero_config")
    )
    np.random.seed(config.seed)
    random.seed(config.seed)
    episodes = []
    for task_id in ids:
        env, task, bddl = create(task_id, config.seed)
        try:
            states = None
            if protocol.reset_source == "prescribed_initial_states":
                import torch

                # The verified, clean LIBERO revision stores NumPy arrays in
                # these files. PyTorch 2.6 changed its default to weights_only,
                # which cannot read them. Keep this exception local to the
                # pinned reset assets, without changing torch.load globally.
                state_file = (
                    Path(root).resolve()
                    / "libero"
                    / "libero"
                    / "init_files"
                    / task.problem_folder
                    / task.init_states_file
                )
                states = torch.load(state_file, map_location="cpu", weights_only=False)
            if states is not None and len(states) < count:
                raise ValueError("Not enough prescribed initial states")
            for trial in range(count):
                env.reset()
                if states is not None:
                    env.set_init_state(states[trial])
                state = np.asarray(env.get_sim_state(), dtype=np.float64)
                # Capture the released seeded reset stream once, before any model
                # draws. Every method then replays the exact captured reset state.
                entry = {
                    "episode_id": f"{protocol.suite}:task{task_id}:state{trial}",
                    "suite": protocol.suite,
                    "task_id": task_id,
                    "initial_state_id": trial,
                    "instruction": task.language,
                    "seed": config.seed,
                    "reset_state": state.tolist(),
                    "reset_state_sha256": digest(state),
                    "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(),
                    "initially_successful": bool(env.check_success()),
                }
                if protocol.reset_source == "seeded_reset_stream":
                    # LIBERO randomizes fixture poses in MjModel, outside the
                    # flattened MjData state. Preserve the complete body arrays
                    # so a fresh environment can replay every reset trial.
                    model = {
                        name: np.array(
                            getattr(env.sim.model, name), dtype=np.float64, copy=True
                        )
                        for name in ("body_pos", "body_quat")
                    }
                    entry["reset_model"] = {
                        name: value.tolist() for name, value in model.items()
                    }
                    entry["reset_model_sha256"] = digest(model)
                episodes.append(entry)
        finally:
            env.close()
    manifest = {
        "schema_version": (
            "1.1" if protocol.reset_source == "seeded_reset_stream" else "1.0"
        ),
        "benchmark": asdict(protocol),
        "split": config.evaluation.split,
        "libero_root": str(Path(root).resolve()),
        "seed": config.seed,
        "reset_procedure": protocol.reset_source + "_captured_before_stabilization",
        "episodes": episodes,
    }
    manifest["sha256"] = digest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def _reset_model_arrays(entry):
    """Validate OOD model poses independently of the outer manifest checksum."""
    if "reset_model" not in entry or "reset_model_sha256" not in entry:
        raise ValueError(
            "OOD reset manifest lacks model body poses; regenerate the manifest"
        )
    snapshot = entry["reset_model"]
    if not isinstance(snapshot, dict) or set(snapshot) != {"body_pos", "body_quat"}:
        raise ValueError("Captured model body poses have invalid fields")
    try:
        model = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in snapshot.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("Captured model body poses must be numeric arrays") from exc
    positions, quaternions = model["body_pos"], model["body_quat"]
    if (
        positions.ndim != 2
        or positions.shape[0] == 0
        or positions.shape[1] != 3
        or quaternions.shape != (positions.shape[0], 4)
        or not all(np.isfinite(value).all() for value in model.values())
    ):
        raise ValueError("Captured model body poses have invalid shapes or values")
    if digest(model) != entry["reset_model_sha256"]:
        raise ValueError("Captured model body poses hash mismatch")
    return model


def load_task_manifest(path, config):
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop("sha256")
    if digest(manifest) != expected:
        raise ValueError("Frozen task manifest has been changed")
    manifest["sha256"] = expected
    if (
        manifest["benchmark"] != asdict(config.benchmark)
        or manifest["split"] != config.evaluation.split
    ):
        raise ValueError(
            "Manifest protocol/split differs from the resolved configuration"
        )
    if not manifest["episodes"]:
        raise ValueError("Task manifest contains no episodes")
    for episode in manifest["episodes"]:
        if (
            digest(np.asarray(episode["reset_state"], dtype=np.float64))
            != episode["reset_state_sha256"]
        ):
            raise ValueError("Captured reset state hash mismatch")
        if config.benchmark.reset_source == "seeded_reset_stream":
            _reset_model_arrays(episode)
    return manifest


def shard_entries(manifest, index=0, count=1):
    """Partition one frozen manifest without changing its identity or episodes."""
    if (
        type(index) is not int
        or type(count) is not int
        or count < 1
        or not 0 <= index < count
        or count > len(manifest["episodes"])
    ):
        raise ValueError("Invalid evaluation shard index/count")
    return manifest["episodes"][index::count]


def quat_to_axisangle(quaternion):
    quaternion = np.array(quaternion, dtype=np.float64, copy=True)
    quaternion[3] = np.clip(quaternion[3], -1, 1)
    den = math.sqrt(1 - quaternion[3] ** 2)
    return (
        np.zeros(3)
        if math.isclose(den, 0.0)
        else quaternion[:3] * (2 * math.acos(quaternion[3]) / den)
    )


def policy_observation(observation, image_size=224):
    result = {}
    for source, target in (
        ("agentview_image", "observation/image"),
        ("robot0_eye_in_hand_image", "observation/wrist_image"),
    ):
        frame = np.ascontiguousarray(observation[source][::-1, ::-1])
        if image_size is not None:
            from openpi_client import image_tools

            frame = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(frame, image_size, image_size)
            )
        result[target] = frame
    result["observation/state"] = np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat_to_axisangle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    # This allowlist is also the agent boundary: object poses and simulator
    # success predicates are never included in its observations or history.
    return result


class LiberoEpisode:
    def __init__(
        self, env, entry, benchmark, *, video_path=None, policy_image_size=224
    ):
        self.env, self.entry = env, entry
        self.policy_image_size = policy_image_size
        self.frames = []
        self.video_path = video_path
        env.reset()
        if benchmark.reset_source == "seeded_reset_stream":
            model = _reset_model_arrays(entry)
            if any(
                value.shape != getattr(env.sim.model, name).shape
                for name, value in model.items()
            ):
                raise ValueError(
                    "Captured model body poses differ from the environment"
                )
            for name, value in model.items():
                getattr(env.sim.model, name)[...] = value
        # set_init_state forwards the simulation and refreshes observations;
        # fixture poses must already be restored when that happens.
        self.raw = env.set_init_state(np.asarray(entry["reset_state"]))
        self.success, self.terminated = bool(env.check_success()), False
        for _ in range(benchmark.stabilization_steps):
            if self.success:
                break
            self.raw, _, done, _ = env.step([0.0] * 6 + [-1.0])
            self.success = bool(env.check_success())
            self.terminated = bool(done and not self.success)
            if self.terminated:
                break
        self._observation = policy_observation(self.raw, self.policy_image_size)

    def observe(self):
        return self._observation

    def step(self, action):
        if self.video_path:
            self.frames.append(self._observation["observation/image"].copy())
        self.raw, _, done, _ = self.env.step(action.tolist())
        self.success = bool(self.env.check_success())
        self.terminated = bool(done and not self.success)
        self._observation = policy_observation(self.raw, self.policy_image_size)

    def close(self):
        try:
            if self.video_path and self.frames:
                import imageio.v2 as imageio

                imageio.mimwrite(self.video_path, self.frames, fps=20)
        finally:
            self.env.close()


def preflight(config, libero_root=None):
    config.validate()
    packages = {}
    for name in (
        "numpy",
        "PIL",
        "torch",
        "jax",
        "flax",
        "openpi",
        "openpi_client",
        "lerobot",
        "transformers",
        "robosuite",
        "imageio",
    ):
        try:
            packages[name] = importlib.util.find_spec(name) is not None
        except (ModuleNotFoundError, ValueError):
            packages[name] = False
    checkpoint = (
        Path(config.policy.checkpoint).expanduser()
        if config.policy.checkpoint
        else None
    )
    artifact = bool(
        checkpoint
        and (
            (checkpoint / "model.safetensors").exists()
            or (checkpoint / "params").is_dir()
        )
    )
    report = {
        "dependencies": packages,
        "checkpoint_available_locally": artifact,
        "checkpoint": config.policy.checkpoint,
        "benchmark": asdict(config.benchmark),
        "astra_model_version": config.agent.model_version,
        "experiments_run": False,
    }
    if checkpoint is not None:
        report["checkpoint_details"] = inspect_checkpoint(checkpoint)
    if libero_root:
        root = Path(libero_root)
        report["libero_source_exists"] = (
            root / "libero" / "libero" / "benchmark" / "__init__.py"
        ).is_file()
        report["libero_revision"] = (
            revision(root) if report["libero_source_exists"] else None
        )
    native = report.get("checkpoint_details", {}).get("format") == "lerobot_pi05"
    required = ["numpy", "PIL", "torch", "robosuite", "imageio"]
    required += (
        ["lerobot", "transformers"]
        if native
        else ["jax", "flax", "openpi", "openpi_client"]
    )
    report["required_dependencies"] = required
    report["ready"] = (
        all(packages[name] for name in required)
        and artifact
        and report.get("checkpoint_details", {}).get("loader_compatible", False)
        and bool(report.get("libero_source_exists"))
        and report.get("libero_revision") == config.benchmark.environment_revision
    )
    return report
