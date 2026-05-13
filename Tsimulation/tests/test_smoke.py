"""Smoke tests for PushShapes env and zarr writer round-trip.

Run with::

    pytest Tsimulation/tests/test_smoke.py -q
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import zarr

# Headless pygame: required when CI / a remote shell has no display.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from Tsimulation.collect.zarr_writer import (
    ACTION_KEY,
    GOAL_KEY,
    IMAGE_KEY,
    REWARD_KEY,
    STATE_KEY,
    ZarrDemoWriter,
)
from Tsimulation.pushshapes.env import PushShapesEnv
from Tsimulation.pushshapes.shapes import SHAPES

SHAPES_TO_TEST = list(SHAPES.keys())
PUSHERS = ["circle", "stick"]
OBSTACLES = [0, 1, 2, 3]


@pytest.mark.parametrize("object_shape", SHAPES_TO_TEST)
@pytest.mark.parametrize("pusher_shape", PUSHERS)
@pytest.mark.parametrize("obstacle_level", OBSTACLES)
def test_env_step_smoke(object_shape, pusher_shape, obstacle_level):
    env = PushShapesEnv(
        object_shape=object_shape,
        pusher_shape=pusher_shape,
        obstacle_level=obstacle_level,
        image_size=96,
        seed=42,
    )
    try:
        obs, info = env.reset(seed=42)

        assert obs["agent_pos"].shape == (2,)
        assert obs["agent_pos"].dtype == np.float32
        assert obs["object_pose"].shape == (3,)
        assert obs["object_pose"].dtype == np.float32
        assert obs["goal_pose"].shape == (3,)
        assert obs["goal_pose"].dtype == np.float32
        assert obs["image"].shape == (96, 96, 3)
        assert obs["image"].dtype == np.uint8
        assert "coverage" in info

        for _ in range(5):
            action = np.array([256.0, 256.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs["image"].shape == (96, 96, 3)
            assert obs["image"].dtype == np.uint8
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert 0.0 <= reward <= 1.0
    finally:
        env.close()


def test_writer_round_trip():
    """Synthesize 2 fake episodes (3 and 5 steps), commit, reopen the store,
    verify per-episode counts and array shapes."""
    with tempfile.TemporaryDirectory() as tmp:
        env_args = {"object_shape": "T", "pusher_shape": "circle", "obstacle_level": 0}
        writer = ZarrDemoWriter(path=tmp, env_args=env_args, image_size=8)
        assert writer.next_episode_index == 0

        rng = np.random.default_rng(0)
        episode_lengths = [3, 5]
        for ep_len in episode_lengths:
            writer.start_episode()
            for _ in range(ep_len):
                writer.add_step(
                    image=rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8),
                    state=rng.standard_normal(5).astype(np.float32),
                    action=rng.uniform(0, 512, size=2).astype(np.float32),
                    reward=float(rng.uniform()),
                    goal_pose=rng.standard_normal(3).astype(np.float32),
                )
            idx = writer.commit_episode()
            assert idx >= 0

        writer.close()

        # Reopen each episode store and verify.
        for ep_idx, ep_len in enumerate(episode_lengths):
            ep_path = os.path.join(tmp, f"episode_{ep_idx:06d}.zarr")
            assert os.path.isdir(ep_path), f"missing {ep_path}"
            store = zarr.open_group(ep_path, mode="r")
            attrs = dict(store.attrs)
            assert attrs["embodiment"] == "pushshapes_sim"
            assert attrs["total_frames"] == ep_len
            assert attrs["task_name"] == "pushshapes"

            # task_description should be JSON of env_args.
            desc = json.loads(attrs["task_description"])
            assert desc["env_args"]["object_shape"] == "T"

            features = attrs["features"]
            assert STATE_KEY in features
            assert ACTION_KEY in features
            assert REWARD_KEY in features
            assert GOAL_KEY in features
            assert IMAGE_KEY in features
            assert features[IMAGE_KEY]["dtype"] == "jpeg"

            # Numeric arrays at least as long as episode (writer may pad to
            # chunk_timesteps for sharding alignment).
            state_arr = store[STATE_KEY][:ep_len]
            assert state_arr.shape == (ep_len, 5)
            action_arr = store[ACTION_KEY][:ep_len]
            assert action_arr.shape == (ep_len, 2)


def test_writer_resumes_index_after_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        env_args = {"object_shape": "T", "pusher_shape": "circle", "obstacle_level": 0}

        w1 = ZarrDemoWriter(path=tmp, env_args=env_args, image_size=8)
        w1.start_episode()
        rng = np.random.default_rng(1)
        for _ in range(2):
            w1.add_step(
                image=rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8),
                state=rng.standard_normal(5).astype(np.float32),
                action=rng.uniform(0, 512, size=2).astype(np.float32),
                reward=0.5,
                goal_pose=rng.standard_normal(3).astype(np.float32),
            )
        idx = w1.commit_episode()
        assert idx == 0
        w1.close()

        w2 = ZarrDemoWriter(path=tmp, env_args=env_args, image_size=8)
        assert (
            w2.next_episode_index == 1
        ), "writer should resume at idx 1 when episode_000000 already exists"
        w2.close()


def test_writer_abort_does_not_create_store():
    with tempfile.TemporaryDirectory() as tmp:
        env_args = {"object_shape": "T", "pusher_shape": "circle", "obstacle_level": 0}
        writer = ZarrDemoWriter(path=tmp, env_args=env_args, image_size=8)
        writer.start_episode()
        writer.add_step(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            state=np.zeros(5, dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
            reward=0.0,
            goal_pose=np.zeros(3, dtype=np.float32),
        )
        writer.abort_episode()
        writer.close()
        # Nothing should have been written.
        assert not any(p.name.endswith(".zarr") for p in os.scandir(tmp))
