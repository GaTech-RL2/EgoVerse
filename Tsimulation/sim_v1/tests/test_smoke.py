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

from Tsimulation.sim_v1.collect.zarr_writer import (
    ACTION_KEY,
    CMD_PUSHER_KEY,
    GOAL_KEY,
    IMAGE_KEY,
    REWARD_KEY,
    STATE_KEY,
    ZarrDemoWriter,
)
from Tsimulation.sim_v1.pushshapes.env import PushShapesEnv
from Tsimulation.sim_v1.pushshapes.shapes import SHAPES

SHAPES_TO_TEST = list(SHAPES.keys())
PUSHERS = ["circle", "stick"]
OBSTACLES = [0, 1, 2, 3]


def _add_fake_step(
    writer: ZarrDemoWriter,
    rng: np.random.Generator,
    image_size: int = 8,
    reward: float | None = None,
) -> None:
    """Helper: feed the writer one synthetic step with the new split-pose API."""
    writer.add_step(
        image=rng.integers(0, 255, size=(image_size, image_size, 3), dtype=np.uint8),
        pusher_obs_pose=rng.standard_normal(2).astype(np.float32),
        object_obs_pose=rng.standard_normal(3).astype(np.float32),
        pusher_cmd_pose=rng.uniform(0, 512, size=2).astype(np.float32),
        action=rng.uniform(0, 512, size=2).astype(np.float32),
        reward=float(rng.uniform()) if reward is None else reward,
        goal_pose=rng.standard_normal(3).astype(np.float32),
    )


def _episode_filename(env_args: dict, idx: int) -> str:
    return (
        f"episode_{env_args['object_shape']}_{env_args['pusher_shape']}"
        f"_obs{env_args['obstacle_level']}_{idx:06d}.zarr"
    )


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
                _add_fake_step(writer, rng)
            idx = writer.commit_episode()
            assert idx >= 0

        writer.close()

        # Reopen each episode store and verify.
        for ep_idx, ep_len in enumerate(episode_lengths):
            ep_path = os.path.join(tmp, _episode_filename(env_args, ep_idx))
            assert os.path.isdir(ep_path), f"missing {ep_path}"
            store = zarr.open_group(ep_path, mode="r")
            attrs = dict(store.attrs)
            assert attrs["embodiment"] == "pushshapes_sim"
            assert attrs["total_frames"] == ep_len
            assert attrs["task_name"] == "pushshapes"

            desc = json.loads(attrs["task_description"])
            assert desc["env_args"]["object_shape"] == "T"

            features = attrs["features"]
            for key in (
                STATE_KEY,
                CMD_PUSHER_KEY,
                ACTION_KEY,
                REWARD_KEY,
                GOAL_KEY,
                IMAGE_KEY,
            ):
                assert key in features, f"missing feature {key!r}"
            assert features[IMAGE_KEY]["dtype"] == "jpeg"

            # Numeric arrays at least as long as episode (writer may pad to
            # chunk_timesteps for sharding alignment).
            state_arr = store[STATE_KEY][:ep_len]
            assert state_arr.shape == (ep_len, 5)
            action_arr = store[ACTION_KEY][:ep_len]
            assert action_arr.shape == (ep_len, 2)
            cmd_arr = store[CMD_PUSHER_KEY][:ep_len]
            assert cmd_arr.shape == (ep_len, 2)


def test_writer_resumes_index_after_reopen():
    with tempfile.TemporaryDirectory() as tmp:
        env_args = {"object_shape": "T", "pusher_shape": "circle", "obstacle_level": 0}

        w1 = ZarrDemoWriter(path=tmp, env_args=env_args, image_size=8)
        w1.start_episode()
        rng = np.random.default_rng(1)
        for _ in range(2):
            _add_fake_step(w1, rng, reward=0.5)
        idx = w1.commit_episode()
        assert idx == 0
        w1.close()

        w2 = ZarrDemoWriter(path=tmp, env_args=env_args, image_size=8)
        assert (
            w2.next_episode_index == 1
        ), "writer should resume at idx 1 when an episode_*_000000.zarr already exists"
        w2.close()


def test_writer_abort_does_not_create_store():
    with tempfile.TemporaryDirectory() as tmp:
        env_args = {"object_shape": "T", "pusher_shape": "circle", "obstacle_level": 0}
        writer = ZarrDemoWriter(path=tmp, env_args=env_args, image_size=8)
        writer.start_episode()
        writer.add_step(
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            pusher_obs_pose=np.zeros(2, dtype=np.float32),
            object_obs_pose=np.zeros(3, dtype=np.float32),
            pusher_cmd_pose=np.zeros(2, dtype=np.float32),
            action=np.zeros(2, dtype=np.float32),
            reward=0.0,
            goal_pose=np.zeros(3, dtype=np.float32),
        )
        writer.abort_episode()
        writer.close()
        # Nothing should have been written.
        assert not any(p.name.endswith(".zarr") for p in os.scandir(tmp))


def test_zarrdataset_end_to_end_load():
    """Write an episode, then load it back via ZarrDataset using the same
    key_map the training pipeline uses. Proves the writer/loader pair is
    compatible end-to-end — not just that the raw zarr file is shaped right."""
    # These imports drag in heavy egomimic deps; skip the test cleanly if any
    # of them aren't available (e.g. on a stripped-down sim-only install).
    ZarrDataset = pytest.importorskip(
        "egomimic.rldb.zarr.zarr_dataset_multi"
    ).ZarrDataset
    get_keymap = pytest.importorskip("egomimic.rldb.embodiment.pushshapes").get_keymap

    with tempfile.TemporaryDirectory() as tmp:
        env_args = {"object_shape": "T", "pusher_shape": "circle", "obstacle_level": 0}
        writer = ZarrDemoWriter(path=tmp, env_args=env_args, image_size=8)
        writer.start_episode()
        rng = np.random.default_rng(7)
        ep_len = 4
        for _ in range(ep_len):
            _add_fake_step(writer, rng)
        idx = writer.commit_episode()
        writer.close()
        assert idx == 0

        ep_path = os.path.join(tmp, _episode_filename(env_args, 0))
        dataset = ZarrDataset(ep_path, key_map=get_keymap())
        sample = dataset[0]

        # The loader should hand back each configured key with the right
        # shape; ZarrDataset returns images in torch (C, H, W) layout after
        # decoding the JPEG bytes.
        assert "front_img_1" in sample
        assert "state_agent_obj" in sample
        assert "actions" in sample
        img = sample["front_img_1"]
        assert img.shape[0] == 3, f"expected C=3, got shape {tuple(img.shape)}"
        assert len(img.shape) == 3
        assert sample["state_agent_obj"].shape[-1] == 5
        # action_horizon=32 set in get_keymap → loader returns (32, 2) per sample.
        assert sample["actions"].shape == (32, 2)
