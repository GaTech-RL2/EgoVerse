import copy
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from astra_reversal import libero_runner
from astra_reversal.config import BenchmarkConfig, RunConfig
from astra_reversal.records import digest


class RandomFixtureEnvironment:
    """Mimic LIBERO's split between dynamic state and randomized MjModel poses."""

    def __init__(self, seed=7, bodies=3):
        self.rng = np.random.RandomState(seed)
        self.sim = SimpleNamespace(
            model=SimpleNamespace(
                body_pos=np.zeros((bodies, 3)),
                body_quat=np.zeros((bodies, 4)),
            )
        )
        self.closed = False
        self.actions = []

    def reset(self):
        self.state = self.rng.uniform(size=8)
        self.sim.model.body_pos[:] = self.rng.uniform(
            size=self.sim.model.body_pos.shape
        )
        quaternions = self.rng.normal(size=self.sim.model.body_quat.shape)
        self.sim.model.body_quat[:] = quaternions / np.linalg.norm(
            quaternions, axis=1, keepdims=True
        )
        return self.observe()

    def observe(self):
        return {
            "state": self.state.copy(),
            "fixtures": self.sim.model.body_pos.copy(),
            "orientations": self.sim.model.body_quat.copy(),
        }

    def get_sim_state(self):
        return self.state.copy()

    def set_init_state(self, state):
        # Like the release wrapper, restoring MjData does not restore MjModel.
        self.state = state.copy()
        self.observation_at_restore = self.observe()
        return self.observe()

    def check_success(self):
        return False

    def step(self, action):
        self.actions.append(action)
        return self.observe(), 0, False, {}

    def close(self):
        self.closed = True


@pytest.fixture
def ood_manifest(tmp_path, monkeypatch):
    config = RunConfig.from_dict({"seed": 7, "benchmark": {"suite": "libero_goal_ood"}})
    bddl = tmp_path / "task.bddl"
    bddl.write_text("released task fixture")
    created = []

    def create(task_id, seed):
        env = RandomFixtureEnvironment(seed)
        created.append(env)
        return env, SimpleNamespace(language="put the bowl on the stove"), bddl

    monkeypatch.setattr(libero_runner, "configure_libero", lambda *args: (None, create))
    monkeypatch.setattr(
        libero_runner, "policy_observation", lambda observation, size: observation
    )
    path = tmp_path / "ood.json"
    manifest = libero_runner.create_task_manifest(
        tmp_path, config, path, task_ids=[0], trials=3
    )
    assert all(env.closed for env in created)
    return config, manifest, path


def test_ood_replays_each_full_scene_after_unrelated_resets(ood_manifest):
    config, manifest, path = ood_manifest
    loaded = libero_runner.load_task_manifest(path, config)
    assert loaded == manifest
    assert loaded["schema_version"] == "1.1"
    entries = loaded["episodes"]
    assert len({entry["reset_model_sha256"] for entry in entries}) == 3

    # Obtain the reference reset stream independently of manifest serialization.
    reference = RandomFixtureEnvironment(seed=7)
    for trial, entry in enumerate(entries):
        expected = reference.reset()
        # Each rollout starts with a fresh environment, whose reset would
        # otherwise overwrite the fixture poses captured for this trial.
        for seed in (7, 123):
            env = RandomFixtureEnvironment(seed)
            for _ in range(trial + 1):
                env.reset()
            episode = libero_runner.LiberoEpisode(env, entry, config.benchmark)
            for name, value in expected.items():
                np.testing.assert_array_equal(episode.observe()[name], value)
                np.testing.assert_array_equal(env.observation_at_restore[name], value)
            assert env.actions == [[0.0] * 6 + [-1.0]] * 10
            episode.close()
    # Captured lists must also remain independent of later reset mutations.
    assert libero_runner.load_task_manifest(path, config) == loaded


def _write_rehashed(path, manifest):
    manifest.pop("sha256", None)
    manifest["sha256"] = digest(manifest)
    path.write_text(json.dumps(manifest))


def test_ood_rejects_legacy_snapshot_without_fixture_poses(ood_manifest):
    config, manifest, path = ood_manifest
    for entry in manifest["episodes"]:
        del entry["reset_model"]
        del entry["reset_model_sha256"]
    _write_rehashed(path, manifest)
    with pytest.raises(ValueError, match="regenerate"):
        libero_runner.load_task_manifest(path, config)


def test_ood_detects_fixture_edit_even_with_recomputed_manifest_hash(ood_manifest):
    config, manifest, path = ood_manifest
    manifest["episodes"][1]["reset_model"]["body_pos"][1][0] += 0.02
    _write_rehashed(path, manifest)
    with pytest.raises(ValueError, match="model body poses hash mismatch"):
        libero_runner.load_task_manifest(path, config)


def test_ood_rejects_inconsistent_body_pose_dimensions(ood_manifest):
    config, manifest, path = ood_manifest
    entry = manifest["episodes"][0]
    entry["reset_model"]["body_quat"].pop()
    entry["reset_model_sha256"] = digest(
        {name: np.asarray(value) for name, value in entry["reset_model"].items()}
    )
    _write_rehashed(path, manifest)
    with pytest.raises(ValueError, match="invalid shapes"):
        libero_runner.load_task_manifest(path, config)


def test_ood_rejects_snapshot_from_different_body_topology(ood_manifest):
    config, manifest, _ = ood_manifest
    env = RandomFixtureEnvironment(bodies=4)
    with pytest.raises(ValueError, match="differ from the environment"):
        libero_runner.LiberoEpisode(env, manifest["episodes"][0], config.benchmark)


def test_standard_libero_legacy_manifest_and_reset_remain_compatible(
    tmp_path, monkeypatch
):
    config = RunConfig()
    reference = RandomFixtureEnvironment()
    reference.reset()
    state = reference.get_sim_state()
    entry = {"reset_state": state.tolist(), "reset_state_sha256": digest(state)}
    manifest = {
        "schema_version": "1.0",
        "benchmark": asdict(BenchmarkConfig()),
        "split": "development",
        "episodes": [entry],
    }
    path = tmp_path / "standard.json"
    _write_rehashed(path, manifest)
    assert libero_runner.load_task_manifest(path, config) == manifest
    monkeypatch.setattr(
        libero_runner, "policy_observation", lambda observation, size: observation
    )
    original_entry = copy.deepcopy(entry)
    episode = libero_runner.LiberoEpisode(
        RandomFixtureEnvironment(seed=123), entry, config.benchmark
    )
    np.testing.assert_array_equal(episode.observe()["state"], state)
    assert entry == original_entry
    episode.close()
