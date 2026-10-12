"""Source-locked LIBERO-MAX substrate and exact native-prefix checks.

This adapter uses the released event implementation. It never substitutes the
ordinary LIBERO task registry for the Plus or PRO source of a frozen case.
"""

import copy
import math
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .common import digest, file_hash, strict_json
from .libero import Environment as LiberoEnvironment
from .libero import array_hash

MAX_COMMIT = "01625e290b1aa07511bdc820781e635e851353cc"
SOURCE_COMMITS = {
    "plus": "4976dc30028e805ff8094b55501d532c48fec182",
    "pro": "2b910b5b5f53016bef9907632f6f840f1ce2229c",
}
PRO_DATA_COMMIT = "c86fc3b8293185a6f373677018ff3e37f8391602"
RELEASES = {
    "max": (
        "benchmark/max8000/libero_max_8000.json",
        8000,
        "d26c38517b08709645a44b7fcf572f47eca056f7d01f1a7f5555dacf159dd1ed",
    ),
    "lite": (
        "benchmark/lite/libero_max_lite.json",
        800,
        "0f2b7fc879e91f1ab85156f005adbb708c68966c631603f179f8153f4f2e22f7",
    ),
}
HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
EVENTS = frozenset(
    {
        "target_relocation",
        "receptacle_relocation",
        "camera_shift",
        "sensor_noise_onset",
        "illumination_switch",
        "visual_theme_switch",
        "distractor_burst",
        "obstacle_insertion",
    }
)


def check_checkout(root, commit):
    root = Path(root).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if actual != commit or dirty.strip():
        raise ValueError("liberomax_source_pin_or_dirty_checkout")


def source_of(case):
    variant = case.get("substrate_variant")
    if variant is None:
        return "plus"
    if variant.get("benchmark") != "LIBERO-PRO":
        raise ValueError("unknown_case_substrate")
    return "pro"


def load_release(root, track):
    root = Path(root).resolve()
    check_checkout(root, MAX_COMMIT)
    relative, count, sha = RELEASES[track]
    path = root / relative
    if file_hash(path) != sha:
        raise ValueError("frozen_manifest_changed")
    manifest = strict_json(path.read_bytes())
    cases = manifest["cases"]
    if len(cases) != count or len({c["case_id"] for c in cases}) != count:
        raise ValueError("released_case_count_or_identity")
    counts = Counter((source_of(c), c["scenario"]["change_type"]) for c in cases)
    expected = {
        (source, event): count * n // 8000
        for source, n in (("plus", 700), ("pro", 300))
        for event in EVENTS
    }
    if dict(counts) != expected:
        raise ValueError("released_source_event_composition")
    # Keep upstream's schema/semantic validation in addition to byte identity.
    subprocess.run(
        [sys.executable, "-m", "libero_max", "validate-manifest", str(path)],
        cwd=root,
        env=upstream_environment(root),
        check=True,
        capture_output=True,
    )
    return manifest


def upstream_environment(root):
    import os

    return {**os.environ, "PYTHONPATH": str(Path(root).resolve() / "src")}


def commissioning_cases(manifest, source):
    """One metadata-selected case per event; no policy outcomes are consulted."""
    selected = {}
    for case in manifest["cases"]:
        if source_of(case) == source:
            selected.setdefault(case["scenario"]["change_type"], case)
    if set(selected) != EVENTS:
        raise ValueError("commissioning_event_coverage_incomplete")
    return [copy.deepcopy(selected[event]) for event in sorted(EVENTS)]


def event_probe(case):
    """A clearly unscored event-mechanism probe, never a released benchmark case."""
    probe = copy.deepcopy(case)
    probe["case_id"] = "commission-" + case["case_id"]
    probe["scenario"]["scenario_id"] = "commission-" + case["scenario"]["scenario_id"]
    probe["scenario"]["trigger"] = {"type": "fixed_step", "value": 2}
    return probe


def verify_pair(control, intervention):
    """Require exact executed actions through the step which introduces the event."""
    fields = (
        "scenario_id",
        "scenario_seed",
        "episode_index",
        "task_suite_name",
        "original_task_index",
        "init_state_index",
        "task_description",
        "policy_seed",
        "init_state_sha256",
        "query_interval",
        "max_policy_steps",
    )
    if (control["arm"], intervention["arm"]) != ("control", "intervention"):
        raise ValueError("pair_arm_labels")
    if any(control[field] != intervention[field] for field in fields):
        raise ValueError("pair_initial_identity_mismatch")
    if control["intervention_event_count"] != 0 or control["intervention_events"]:
        raise ValueError("control_received_event")
    count = intervention["intervention_event_count"]
    if (
        type(count) is not int
        or count not in (0, 1)
        or len(intervention["intervention_events"]) != count
    ):
        raise ValueError("intervention_event_count")
    event_step = (
        intervention["intervention_events"][0]["cosmos_query_boundary_step"]
        if count
        else None
    )
    if count and (
        type(event_step) is not int
        or not 1 <= event_step <= control["max_policy_steps"]
    ):
        raise ValueError("invalid_event_step")
    actions = []
    for row in (control, intervention):
        trace = row["executed_actions"]
        if [x["policy_step"] for x in trace] != list(range(1, len(trace) + 1)):
            raise ValueError("noncontiguous_native_action_trace")
        if any(
            not isinstance(x["action"], list)
            or len(x["action"]) != 7
            or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in x["action"]
            )
            for x in trace
        ):
            raise ValueError("invalid_native_action_trace")
        prefix = trace if event_step is None else trace[:event_step]
        if event_step is not None and len(prefix) != event_step:
            raise ValueError("missing_native_prefix_actions")
        actions.append([x["action"] for x in prefix])
    if actions[0] != actions[1]:
        raise ValueError("executed_native_prefix_mismatch")
    if count == 0 and control["success"] != intervention["success"]:
        raise ValueError("unreached_event_terminal_disagreement")
    return {
        "matched": True,
        "event_step": event_step,
        "trigger_reached": bool(count),
        "prefix_steps": len(actions[0]),
        "native_prefix_sha256": digest(actions[0]),
        "control_success": bool(control["success"]),
        "intervention_success": bool(intervention["success"]),
    }


class Environment:
    """Private evaluator wrapper; actor reads retain all observation disturbances."""

    def __init__(self, case, arm, trace_path, *, image_size=128, horizon=None):
        import mujoco
        import robosuite
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from libero_max.cosmos_integration import CosmosInterventionEnv
        from libero_max.env_factory import create_libero_env_with_retry
        from libero_max.pro_runtime import wrap_case_env
        from libero_max.substrate import load_case_task

        if robosuite.__version__ != "1.4.0" or mujoco.__version__ != "3.2.6":
            raise ValueError("liberomax_reference_simulator_versions_required")
        self.case = copy.deepcopy(case)
        self.horizon = HORIZONS[case["task_suite_name"]] if horizon is None else horizon
        seed = int(case["policy_seed"])

        def reseed(value):
            random.seed(value)
            np.random.seed(value)
            torch.manual_seed(value)

        reseed(seed)
        task, states = load_case_task(case, benchmark)
        initial = np.asarray(states[case["init_state_index"]])
        bddl = (
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )

        def factory():
            native = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=image_size,
                camera_widths=image_size,
                control_freq=20,
                horizon=self.horizon + 10,
                ignore_done=True,
            )
            native.seed(seed)
            return native, task.language

        native, self.instruction = create_libero_env_with_retry(
            factory, policy_seed=seed, reseed=reseed
        )
        try:
            self.env = wrap_case_env(native, case)
            self.env.seed(seed)
            self.wrapped = CosmosInterventionEnv(
                self.env,
                self.instruction,
                case["scenario"],
                arm,
                Path(trace_path),
                case["task_index"],
                init_state_index=case["init_state_index"],
            )
            self.wrapped.configure_episode(
                case["task_suite_name"], seed, 1, self.horizon
            )
            self.wrapped.reset()
            self.observation = self.wrapped.set_init_state(initial)
            for _ in range(self.wrapped.warmup_steps):
                self.observation, _, _, _ = self.wrapped.step([0, 0, 0, 0, 0, 0, -1])
            self.controller = LiberoEnvironment._controller(self)
            self.controller["version"] = "robosuite-1.4.0:OSC_POSE"
            self.reset_receipt = {
                "official_state_sha256": array_hash(initial),
                "settled_state_sha256": array_hash(self.env.get_sim_state()),
                "bddl_sha256": file_hash(bddl),
                "source": source_of(case),
                "initial_success": bool(self.env.check_success()),
            }
        except BaseException:
            native.close()
            raise

    def read_sensors(self):
        # The ordinary LIBERO raw getter would silently remove sensor corruption,
        # brightness changes, and PRO substrate transforms from actor observations.
        self.observation = self.wrapped.backend.refresh_observation()
        return self.observation

    def step(self, action):
        result = self.wrapped.step(np.asarray(action, dtype=np.float64).copy())
        self.observation = result[0]
        return result

    def check_success(self):
        return self.env.check_success()

    def terminal_snapshot(self, destination):
        return LiberoEnvironment.terminal_snapshot(self, destination)

    def close(self):
        self.env.close()
