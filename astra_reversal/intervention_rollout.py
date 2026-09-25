"""Frozen resets and closed-loop attempts for intervention adaptation.

An attempt consumes a full simulator rollout after restoring the same initial
scene. It is online adaptation with reset access, not zero-shot evaluation.
Policy loading, intervention selection, model solves and token accounting belong
to the caller. The callback receives only the ordinary policy observation.
"""

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import libero_runner
from .records import digest

CAMERAS = ("observation/image", "observation/wrist_image")
POLICY_KEYS = (*CAMERAS, "observation/state")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def _model_snapshot(env):
    return {
        name: np.array(getattr(env.sim.model, name), dtype=np.float64, copy=True)
        for name in ("body_pos", "body_quat")
    }


def _state(env):
    value = np.asarray(env.get_sim_state(), dtype=np.float64)
    _require(
        value.ndim == 1 and value.size and np.isfinite(value).all(),
        "Invalid simulator state",
    )
    return value


def _validate_entry(entry, benchmark):
    _require(
        entry["suite"] == benchmark.suite, "Reset suite differs from the benchmark"
    )
    _require(_integer(entry["seed"]) and entry["seed"] < 2**32, "Invalid reset seed")
    _require(_integer(entry["task_id"]) and entry["task_id"] < 10, "Invalid task ID")
    _require(_integer(entry["initial_state_id"]), "Invalid initial-state index")
    expected_id = f"{benchmark.suite}:seed{entry['seed']}:task{entry['task_id']}:state{entry['initial_state_id']}"
    _require(
        entry["episode_id"] == expected_id, "Reset episode ID must include its seed"
    )
    state = np.asarray(entry["reset_state"], dtype=np.float64)
    _require(
        state.ndim == 1 and state.size and np.isfinite(state).all(),
        "Invalid captured reset state",
    )
    _require(
        digest(state) == entry["reset_state_sha256"],
        "Captured reset state hash mismatch",
    )
    return libero_runner._reset_model_arrays(entry)


def _prescribed_states(root, task):
    import torch

    path = (
        Path(root).resolve()
        / "libero/libero/init_files"
        / task.problem_folder
        / task.init_states_file
    )
    return torch.load(path, map_location="cpu", weights_only=False), path


def capture_reset_manifest(
    libero_root, benchmark, *, seed, cases, output, split="followup_adaptation"
):
    """Capture explicit (task ID, reset index) cases before loading the policy.

    OOD indexes address a seeded reset stream; standard indexes address the
    pinned prescribed-state asset. These manifests have a separate schema and
    do not change the published ten-trial OOD or fifty-trial standard presets.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace frozen reset manifest {output}")
    _require(_integer(seed) and seed < 2**32, "Seed must be an integer in [0, 2**32)")
    _require(
        split in ("development", "followup_adaptation"), "Unknown adaptation split"
    )
    cases = [tuple(case) for case in cases]
    _require(
        bool(cases) and len(set(cases)) == len(cases),
        "Cases must be nonempty and distinct",
    )
    _require(
        all(
            len(case) == 2 and _integer(case[0]) and case[0] < 10 and _integer(case[1])
            for case in cases
        ),
        "Invalid task/reset case",
    )
    _require(
        benchmark.reset_source in ("prescribed_initial_states", "seeded_reset_stream"),
        "Unsupported reset source",
    )
    _, create = libero_runner.configure_libero(
        libero_root, benchmark, output.parent / (output.stem + "_libero_config")
    )
    np.random.seed(seed)
    random.seed(seed)
    captured = {}
    for task_id in dict.fromkeys(task for task, _ in cases):
        requested = {index for task, index in cases if task == task_id}
        env, task, bddl = create(task_id, seed)
        try:
            states, asset = None, None
            if benchmark.reset_source == "prescribed_initial_states":
                states, asset = _prescribed_states(libero_root, task)
                _require(
                    max(requested) < len(states),
                    "Prescribed initial-state index is unavailable",
                )
            for index in range(max(requested) + 1):
                env.reset()
                if states is not None:
                    env.set_init_state(states[index])
                if index not in requested:
                    continue
                state, model = _state(env), _model_snapshot(env)
                entry = {
                    "episode_id": f"{benchmark.suite}:seed{seed}:task{task_id}:state{index}",
                    "suite": benchmark.suite,
                    "task_id": task_id,
                    "initial_state_id": index,
                    "seed": seed,
                    "instruction": task.language,
                    "reset_state": state.tolist(),
                    "reset_state_sha256": digest(state),
                    "reset_model": {
                        name: value.tolist() for name, value in model.items()
                    },
                    "reset_model_sha256": digest(model),
                    "bddl_sha256": hashlib.sha256(Path(bddl).read_bytes()).hexdigest(),
                    "initially_successful": bool(env.check_success()),
                    "prescribed_state_asset_sha256": (
                        hashlib.sha256(asset.read_bytes()).hexdigest()
                        if asset
                        else None
                    ),
                }
                _validate_entry(entry, benchmark)
                captured[task_id, index] = entry
        finally:
            env.close()
    manifest = {
        "schema_version": "intervention_reset_v1",
        "evaluation_type": "online_adaptation_with_simulator_reset",
        "split": split,
        "seed": seed,
        "benchmark": asdict(benchmark),
        "libero_root": str(Path(libero_root).resolve()),
        "reset_procedure": benchmark.reset_source + "_captured_before_stabilization",
        "cases": [list(case) for case in cases],
        "episodes": [captured[case] for case in cases],
    }
    manifest["sha256"] = digest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def load_reset_manifest(path, benchmark):
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop("sha256")
    _require(digest(manifest) == expected, "Intervention reset manifest hash mismatch")
    _require(
        manifest["schema_version"] == "intervention_reset_v1",
        "Wrong reset manifest schema",
    )
    _require(
        manifest["benchmark"] == asdict(benchmark),
        "Manifest benchmark differs from the run",
    )
    entries = manifest["episodes"]
    _require(bool(entries), "Empty reset manifest")
    _require(
        len({entry["episode_id"] for entry in entries}) == len(entries),
        "Duplicate reset episode",
    )
    _require(
        manifest["cases"]
        == [[entry["task_id"], entry["initial_state_id"]] for entry in entries],
        "Reset case list differs from its episodes",
    )
    for entry in entries:
        _validate_entry(entry, benchmark)
        _require(
            entry["seed"] == manifest["seed"],
            "Reset entry seed differs from its manifest",
        )
    manifest["sha256"] = expected
    return manifest


def _policy_view(observation):
    result = {key: np.array(observation[key], copy=True) for key in POLICY_KEYS}
    for key in CAMERAS:
        value = result[key]
        _require(
            value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == 3,
            "Policy cameras must be uint8 HWC RGB",
        )
    state = result["observation/state"]
    _require(
        state.shape == (8,) and np.isfinite(state).all(),
        "Policy state must contain eight finite values",
    )
    return result


class _RestoringEnvironment:
    """Restore MjModel for both reset sources before LiberoEpisode sets MjData."""

    def __init__(self, env, entry, model):
        self.env, self.entry, self.model = env, entry, model
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        result = self.env.reset()
        for name, value in self.model.items():
            target = getattr(self.env.sim.model, name)
            _require(
                target.shape == value.shape,
                "Reset model topology differs from the environment",
            )
            target[...] = value
        return result

    def set_init_state(self, state):
        result = self.env.set_init_state(state)
        _require(
            digest(_state(self.env)) == self.entry["reset_state_sha256"],
            "Simulator did not restore the captured dynamic state exactly",
        )
        _require(
            digest(_model_snapshot(self.env)) == self.entry["reset_model_sha256"],
            "Simulator did not restore the captured model poses exactly",
        )
        return result

    def step(self, action):
        result = self.env.step(action)
        self.steps += 1
        return result


def _sparse_snapshots(frames):
    indexes = np.rint(np.linspace(0, len(frames) - 1, min(4, len(frames)))).astype(int)
    result = []
    for ordinal, index in enumerate(indexes):
        label = (
            "first_and_last"
            if len(indexes) == 1
            else "first"
            if ordinal == 0
            else "last"
            if ordinal == len(indexes) - 1
            else f"progress_{ordinal}"
        )
        result.append(
            {"label": label, "step": int(index), "observation": frames[index]}
        )
    return result


def run_rollout(
    env,
    entry,
    benchmark,
    action_policy,
    *,
    execute_steps=5,
    action_budget=None,
    expected_reset=None,
    policy_image_size=None,
    video_path=None,
):
    """Run one attempt and always close its environment, including on errors.

    ``action_policy(observation, step)`` returns finite controller actions [H,7].
    The helper neither loads a model nor chooses a solver. The policy observation
    contains only both cameras and robot proprioception, with copies protecting
    the simulator's original frames from intervention edits. The returned
    metrics/reset audit are JSONable; only snapshots contain image arrays.
    Wall time includes reset, simulator closure and optional video encoding.
    """
    began = time.perf_counter()
    episode = None
    try:
        _require(_integer(execute_steps, minimum=1), "Execution chunk must be positive")
        budget = (
            benchmark.task_action_budget if action_budget is None else action_budget
        )
        _require(
            _integer(budget, minimum=1) and budget <= benchmark.task_action_budget,
            "Action budget must be positive and within the benchmark cap",
        )
        model = _validate_entry(entry, benchmark)
        native = getattr(env, "env", env)
        bddl = getattr(native, "bddl_file_name", None)
        _require(
            bddl is not None
            and hashlib.sha256(Path(bddl).read_bytes()).hexdigest()
            == entry["bddl_sha256"],
            "Environment BDDL differs from the frozen reset task",
        )
        lower, upper = (np.asarray(value) for value in native.action_spec)
        _require(
            lower.shape == upper.shape == (7,)
            and np.isfinite([lower, upper]).all()
            and np.all(lower < upper),
            "Invalid environment controller bounds",
        )
        env.seed(entry["seed"])
        restoring = _RestoringEnvironment(env, entry, model)
        episode = libero_runner.LiberoEpisode(
            restoring,
            entry,
            benchmark,
            video_path=video_path,
            policy_image_size=policy_image_size,
        )
        observation = _policy_view(episode.observe())
        audit = {
            "episode_id": entry["episode_id"],
            "seed": entry["seed"],
            "reset_state_sha256": entry["reset_state_sha256"],
            "reset_model_sha256": entry["reset_model_sha256"],
            "bddl_sha256": entry["bddl_sha256"],
            "post_stabilization_state_sha256": digest(_state(env)),
            "post_stabilization_model_sha256": digest(_model_snapshot(env)),
            "post_stabilization_observation_sha256": digest(observation),
            "stabilization_steps": restoring.steps,
            "initial_success": bool(episode.success),
            "initial_terminated": bool(episode.terminated),
        }
        audit["sha256"] = digest(audit)
        if expected_reset is not None:
            _require(
                audit == expected_reset,
                "Attempt reset differs from the paired initial scene",
            )
        reset_seconds = time.perf_counter() - began
        initial_success = bool(episode.success)
        frames = [{key: observation[key].copy() for key in POLICY_KEYS}]
        actions = replans = 0
        policy_seconds = environment_seconds = 0.0
        while actions < budget and not episode.success and not episode.terminated:
            start = time.perf_counter()
            chunk = np.asarray(action_policy(_policy_view(episode.observe()), actions))
            policy_seconds += time.perf_counter() - start
            needed = min(execute_steps, budget - actions)
            _require(
                chunk.ndim == 2
                and chunk.shape[1] == 7
                and chunk.shape[0] >= needed
                and chunk.dtype.kind in "fiu"
                and np.isfinite(chunk).all(),
                "Policy callback must return enough finite numeric [H,7] actions",
            )
            _require(
                np.all(chunk >= lower) and np.all(chunk <= upper),
                "Policy actions exceed environment controller bounds",
            )
            replans += 1
            for action in chunk[:needed]:
                start = time.perf_counter()
                episode.step(np.asarray(action, dtype=np.float32))
                environment_seconds += time.perf_counter() - start
                actions += 1
                current = _policy_view(episode.observe())
                frames.append({key: current[key].copy() for key in POLICY_KEYS})
                if episode.success or episode.terminated:
                    break
        result = {
            "episode_id": entry["episode_id"],
            "success": bool(episode.success),
            "actions_executed": actions,
            "policy_replans": replans,
            "wall_seconds": time.perf_counter() - began,
            "reset_seconds": reset_seconds,
            "policy_seconds": policy_seconds,
            "environment_seconds": environment_seconds,
            "initial_success": initial_success,
            "captured_initial_success": bool(entry["initially_successful"]),
            "zero_action_success": bool(episode.success and actions == 0),
            "terminated": bool(episode.terminated),
            "action_budget": budget,
            "execute_steps": execute_steps,
            "reset_audit": audit,
            "snapshots": _sparse_snapshots(frames),
            "video_path": str(video_path) if video_path is not None else None,
        }
    finally:
        if episode is None:
            env.close()
        else:
            episode.close()
    result["wall_seconds"] = time.perf_counter() - began
    return result


def rollout_feedback(result):
    """Only frames, robot proprioception, indexes and binary success reach Astra."""
    return {
        "success": bool(result["success"]),
        "snapshots": [
            {
                "label": frame["label"],
                "step": frame["step"],
                "observation": {
                    key: np.array(frame["observation"][key], copy=True)
                    for key in POLICY_KEYS
                },
            }
            for frame in result["snapshots"]
        ],
    }
