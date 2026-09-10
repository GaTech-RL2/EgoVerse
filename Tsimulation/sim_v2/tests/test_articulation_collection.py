"""Collection contracts: physical success, reproducible actions, durable quotas."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest
import simplejpeg
import zarr

from Tsimulation.sim_v2.collect.articulation_collect import (
    collect,
    replay,
    write_episode,
)
from Tsimulation.sim_v2.collect.articulation_probe import rollout


@pytest.mark.parametrize("emb", ["u_socket", "gripper"])
def test_articulated_success_replays_and_roundtrips(tmp_path, emb):
    result, actions, states, held, _ = rollout(emb, "ideal", 0, 900)
    assert result["success"] and result["engaged_fraction"] > 0.1
    assert result["carry_distance"] > 30 and result["angle_travel"] > 0.01
    assert result["jerk_speed"] < 0.5 and result["max_position_step"] < 6.0
    data = replay(emb, "ideal", 0, actions, states)
    assert data["valid"] and data["state_error"] == 0.0
    name = write_episode(tmp_path, 0, emb, "ideal", 0, actions, data, result)
    store = zarr.open_group(tmp_path / name, mode="r")
    np.testing.assert_array_equal(store["actions"][:], actions.astype(np.float32))
    np.testing.assert_array_equal(store["engaged"][:].ravel(), held)
    assert store.attrs["total_frames"] == len(actions)
    assert store.attrs["control_gap"] == "ideal"
    assert len(store.attrs["action_spec"]) == actions.shape[1]
    if emb == "gripper":
        assert np.ptp(store["observations.mechanics"][:, 0]) > 10.0
        assert np.max(store["actions"][:, 3]) == 1.0
    encoded = store["observations.images.front_img_1"][:1][0]
    assert simplejpeg.decode_jpeg(bytes(encoded)).shape == (96, 96, 3)
    assert not list(tmp_path.glob(".pending-*"))
    with pytest.raises(FileExistsError):
        write_episode(tmp_path, 0, emb, "ideal", 0, actions, data, result)


def test_replay_drift_is_rejected():
    _, actions, states, _, _ = rollout("gripper", "tight", 0, 900)
    result = replay("gripper", "tight", 0, actions, states + 1.0)
    assert not result["valid"]


def test_resume_counts_committed_episodes(tmp_path):
    first = collect(tmp_path, "gripper", "ideal", 1, 0, 2, 900)
    assert first["complete"] and first["kept"] == 1
    (tmp_path / "progress.json").unlink()  # crash after episode rename, before progress
    again = collect(tmp_path, "gripper", "ideal", 1, 0, 2, 900)
    assert again["complete"] and again["kept"] == 1
    assert len(list(tmp_path.glob("episode_*.zarr"))) == 1
    with pytest.raises(ValueError, match="Refusing to mix gap"):
        collect(tmp_path, "gripper", "loose", 1, 0, 2, 900)


def test_fast_scoop_search_matches_untouched_simulator():
    args = ("scoop", "ideal", 6090000007, 5000)
    quality, actions, states, _, _ = rollout(*args, fast_queries=False)
    fast_quality, fast_actions, fast_states, _, _ = rollout(*args, fast_queries=True)
    assert quality["success"] and fast_quality["success"]
    np.testing.assert_array_equal(fast_actions, actions)
    np.testing.assert_array_equal(fast_states, states)
    data = replay("scoop", "ideal", 6090000007, fast_actions, fast_states)
    assert data["valid"] and data["state_error"] == 0.0
