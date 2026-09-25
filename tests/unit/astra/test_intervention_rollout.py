"""Synthetic simulator tests, not benchmark or policy performance evidence."""

import copy
import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal import intervention_rollout as rollout
from astra_reversal.config import BenchmarkConfig
from astra_reversal.records import digest


class SyntheticSimulator:
    def __init__(
        self,
        bddl,
        *,
        success_after=None,
        terminate_after=None,
        state_drift=0.0,
        model_drift=0.0,
        image_drift=0,
    ):
        self.env = SimpleNamespace(
            action_spec=(-np.ones(7), np.ones(7)), bddl_file_name=str(bddl)
        )
        self.sim = SimpleNamespace(
            model=SimpleNamespace(
                body_pos=np.zeros((3, 3)),
                body_quat=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
            )
        )
        self.success_after, self.terminate_after = success_after, terminate_after
        self.state_drift, self.model_drift, self.image_drift = (
            state_drift,
            model_drift,
            image_drift,
        )
        self.closed = False
        self.seed(0)

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.state = self.rng.uniform(size=8)
        self.sim.model.body_pos[:] = self.rng.uniform(size=(3, 3))
        self.sim.model.body_quat[:] = [1.0, 0.0, 0.0, 0.0]
        self.steps = 0
        self.actions = []
        return self.observe()

    def observe(self):
        pixel = (self.steps + self.image_drift) % 256
        return {
            "agentview_image": np.full((12, 12, 3), pixel, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.full(
                (12, 12, 3), (pixel + 1) % 256, dtype=np.uint8
            ),
            "robot0_eef_pos": self.state[:3].copy(),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": self.state[3:5].copy(),
            "object_xyz": np.array([3.0, 4.0, 5.0]),
            "dense_oracle_reward": 0.95,
        }

    def get_sim_state(self):
        return self.state.copy()

    def set_init_state(self, state):
        self.state = state.copy()
        self.steps = 0
        return self.observe()

    def check_success(self):
        return self.success_after is not None and self.steps >= self.success_after

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.steps += 1
        self.state += 0.001 + self.state_drift
        self.sim.model.body_pos += self.model_drift
        done = self.terminate_after is not None and self.steps >= self.terminate_after
        return self.observe(), 123.0, done, {"privileged_object_xyz": [9, 8, 7]}

    def close(self):
        self.closed = True


@pytest.fixture
def factory(tmp_path, monkeypatch):
    files = []
    for task in range(10):
        path = tmp_path / f"task_{task}.bddl"
        path.write_text(f"synthetic task {task}")
        files.append(path)
    created = []

    def create(task_id, seed):
        env = SyntheticSimulator(files[task_id])
        env.seed(seed)
        created.append(env)
        task = SimpleNamespace(
            language=f"synthetic task {task_id}",
            problem_folder="libero_10",
            init_states_file="states",
        )
        return env, task, files[task_id]

    monkeypatch.setattr(
        rollout.libero_runner, "configure_libero", lambda *args: (None, create)
    )
    return SimpleNamespace(files=files, created=created, root=tmp_path)


@pytest.fixture
def frozen(factory):
    benchmark = BenchmarkConfig.preset("libero_goal_ood")
    path = factory.root / "reset.json"
    manifest = rollout.capture_reset_manifest(
        factory.root, benchmark, seed=19, cases=[(0, 0)], output=path
    )
    return benchmark, manifest["episodes"][0], path


def test_seeded_stream_indexes_full_scenes_and_seed_namespace(factory):
    benchmark = BenchmarkConfig.preset("libero_spatial_ood")
    path = factory.root / "ood.json"
    manifest = rollout.capture_reset_manifest(
        factory.root, benchmark, seed=19, cases=[(0, 0), (0, 11), (1, 0)], output=path
    )
    assert manifest["benchmark"] == asdict(benchmark)
    assert manifest["schema_version"] == "intervention_reset_v1"
    assert manifest["split"] == "followup_adaptation"
    assert [entry["episode_id"] for entry in manifest["episodes"]] == [
        "libero_spatial_ood:seed19:task0:state0",
        "libero_spatial_ood:seed19:task0:state11",
        "libero_spatial_ood:seed19:task1:state0",
    ]
    reference = SyntheticSimulator(factory.files[0])
    reference.seed(19)
    for _ in range(12):
        reference.reset()
    np.testing.assert_array_equal(
        manifest["episodes"][1]["reset_state"], reference.get_sim_state()
    )
    np.testing.assert_array_equal(
        manifest["episodes"][1]["reset_model"]["body_pos"], reference.sim.model.body_pos
    )
    assert rollout.load_reset_manifest(path, benchmark) == manifest
    assert all(env.closed for env in factory.created)
    with pytest.raises(FileExistsError):
        rollout.capture_reset_manifest(
            factory.root, benchmark, seed=19, cases=[(0, 0)], output=path
        )


def test_standard_prescribed_index_and_fixture_restore(factory, monkeypatch):
    benchmark = BenchmarkConfig()
    asset = factory.root / "trusted_states"
    asset.write_bytes(b"synthetic pinned state asset")
    states = [np.full(8, value, dtype=np.float64) for value in range(50)]
    monkeypatch.setattr(rollout, "_prescribed_states", lambda *args: (states, asset))
    manifest = rollout.capture_reset_manifest(
        factory.root,
        benchmark,
        seed=19,
        cases=[(0, 1), (1, 1)],
        output=factory.root / "standard.json",
        split="development",
    )
    entry = manifest["episodes"][0]
    np.testing.assert_array_equal(entry["reset_state"], states[1])
    assert (
        entry["prescribed_state_asset_sha256"]
        == hashlib.sha256(asset.read_bytes()).hexdigest()
    )
    first = rollout.run_rollout(
        SyntheticSimulator(factory.files[0]),
        entry,
        benchmark,
        lambda obs, step: np.zeros((10, 7)),
        action_budget=5,
    )
    second_env = SyntheticSimulator(factory.files[0])
    second_env.reset()
    second_env.sim.model.body_pos += 100
    second = rollout.run_rollout(
        second_env,
        entry,
        benchmark,
        lambda obs, step: np.zeros((10, 7)),
        action_budget=5,
        expected_reset=first["reset_audit"],
    )
    assert first["reset_audit"] == second["reset_audit"]
    np.testing.assert_array_equal(
        second_env.sim.model.body_pos, entry["reset_model"]["body_pos"]
    )


def test_closed_loop_chunking_early_success_snapshots_and_feedback(factory, frozen):
    benchmark, entry, _ = frozen
    calls = []

    def policy(observation, step):
        assert set(observation) == set(rollout.POLICY_KEYS)
        calls.append((step, int(observation["observation/image"][0, 0, 0])))
        observation["observation/image"][:] = 255  # Must not modify recorded frames.
        return np.zeros((10, 7), dtype=np.float32)

    env = SyntheticSimulator(factory.files[0], success_after=17)
    result = rollout.run_rollout(env, entry, benchmark, policy)
    assert env.closed and result["success"] and not result["terminated"]
    assert result["actions_executed"] == 7 and result["policy_replans"] == 2
    assert result["reset_audit"]["stabilization_steps"] == 10
    assert len(env.actions) == 17 and calls == [(0, 10), (5, 15)]
    assert [frame["step"] for frame in result["snapshots"]] == [0, 2, 5, 7]
    assert [
        int(frame["observation"]["observation/image"][0, 0, 0])
        for frame in result["snapshots"]
    ] == [10, 12, 15, 17]
    assert [frame["label"] for frame in result["snapshots"]] == [
        "first",
        "progress_1",
        "progress_2",
        "last",
    ]
    feedback = rollout.rollout_feedback(result)
    assert set(feedback) == {"success", "snapshots"}
    assert all(
        set(frame["observation"]) == set(rollout.POLICY_KEYS)
        for frame in feedback["snapshots"]
    )
    assert all(
        frame["observation"]["observation/state"].shape == (8,)
        for frame in feedback["snapshots"]
    )
    feedback["snapshots"][0]["observation"]["observation/image"][:] = 99
    assert result["snapshots"][0]["observation"]["observation/image"][0, 0, 0] == 10
    json.dumps(
        {key: value for key, value in result.items() if key != "snapshots"},
        allow_nan=False,
    )


@pytest.mark.parametrize(
    "drift", [{"state_drift": 0.001}, {"model_drift": 0.001}, {"image_drift": 1}]
)
def test_exact_paired_reset_checks_dynamic_state_fixtures_and_pixels(
    factory, frozen, drift
):
    benchmark, entry, _ = frozen
    first = rollout.run_rollout(
        SyntheticSimulator(factory.files[0]),
        entry,
        benchmark,
        lambda obs, step: np.zeros((10, 7)),
        action_budget=1,
    )
    env = SyntheticSimulator(factory.files[0], **drift)
    called = []
    with pytest.raises(ValueError, match="paired initial scene"):
        rollout.run_rollout(
            env,
            entry,
            benchmark,
            lambda *args: called.append(args),
            action_budget=1,
            expected_reset=first["reset_audit"],
        )
    assert env.closed and not called


def test_action_budget_is_separate_from_stabilization(factory, frozen):
    benchmark, entry, _ = frozen
    calls = []
    env = SyntheticSimulator(factory.files[0])
    result = rollout.run_rollout(
        env,
        entry,
        benchmark,
        lambda obs, step: calls.append(step) or np.zeros((10, 7)),
        action_budget=12,
    )
    assert calls == [0, 5, 10]
    assert result["actions_executed"] == 12 and len(env.actions) == 22
    assert not result["success"] and not result["terminated"]
    assert [frame["step"] for frame in result["snapshots"]] == [0, 4, 8, 12]


def test_success_during_reset_is_flagged_without_policy_credit(factory, frozen):
    benchmark, entry, _ = frozen
    result = rollout.run_rollout(
        SyntheticSimulator(factory.files[0], success_after=5),
        entry,
        benchmark,
        lambda *args: pytest.fail("Policy must not run for an initial success"),
    )
    assert result["initial_success"] and result["zero_action_success"]
    assert result["actions_executed"] == result["policy_replans"] == 0
    assert result["reset_audit"]["stabilization_steps"] == 5
    assert (
        len(result["snapshots"]) == 1
        and result["snapshots"][0]["label"] == "first_and_last"
    )


def test_non_successful_termination_stops_mid_chunk(factory, frozen):
    benchmark, entry, _ = frozen
    result = rollout.run_rollout(
        SyntheticSimulator(factory.files[0], terminate_after=14),
        entry,
        benchmark,
        lambda obs, step: np.zeros((10, 7)),
    )
    assert result["terminated"] and not result["success"]
    assert result["actions_executed"] == 4 and result["policy_replans"] == 1


@pytest.mark.parametrize(
    "chunk",
    [
        np.full((10, 7), np.nan),
        np.full((10, 7), 1.01),
        np.zeros((4, 7)),
        np.zeros((10, 7), dtype=bool),
    ],
)
def test_invalid_actions_fail_before_execution_and_close(factory, frozen, chunk):
    benchmark, entry, _ = frozen
    env = SyntheticSimulator(factory.files[0])
    with pytest.raises(ValueError):
        rollout.run_rollout(env, entry, benchmark, lambda *args: chunk)
    assert env.closed and len(env.actions) == 10


def test_callback_error_is_not_reclassified_as_task_failure(factory, frozen):
    benchmark, entry, _ = frozen
    env = SyntheticSimulator(factory.files[0])

    def fail(*args):
        raise RuntimeError("synthetic execution failure")

    with pytest.raises(RuntimeError, match="synthetic execution failure"):
        rollout.run_rollout(env, entry, benchmark, fail)
    assert env.closed


def test_rehashed_outer_manifest_does_not_hide_scene_tampering(factory, frozen):
    benchmark, _, path = frozen
    manifest = json.loads(path.read_text())
    manifest["episodes"][0]["reset_model"]["body_pos"][0][0] += 1
    manifest.pop("sha256")
    manifest["sha256"] = digest(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model body poses hash mismatch"):
        rollout.load_reset_manifest(path, benchmark)


def test_wrong_task_is_rejected_and_closed(factory, frozen):
    benchmark, entry, _ = frozen
    env = SyntheticSimulator(factory.files[1])
    with pytest.raises(ValueError, match="BDDL"):
        rollout.run_rollout(
            env, copy.deepcopy(entry), benchmark, lambda *args: np.zeros((10, 7))
        )
    assert env.closed
