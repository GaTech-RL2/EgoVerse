"""Smoke tests for the non-physics features:

- scripted_action heuristic returns valid targets
- scripted_collect runs an end-to-end attempt without raising
- dataset_stats aggregates a synthesized dataset
- visualize_episode renders frames headlessly to an MP4

Run with::

    SDL_VIDEODRIVER=dummy pytest Tsimulation/tests/test_features.py -q
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from Tsimulation.collect.replay_init import (
    REPLAY_SOURCE_KEY,
    collected_resume_keys,
    collected_seed_keys,
    init_pose_key,
    load_replay_inits,
    load_replay_manifest,
    reset_to_init,
)
from Tsimulation.collect.scripted_collect import (
    APPROACH_OFFSET,
    scripted_action,
)
from Tsimulation.collect.zarr_writer import ZarrDemoWriter


def test_scripted_action_clipped_and_finite():
    action = scripted_action(
        agent_xy=np.array([100.0, 100.0], dtype=np.float32),
        object_xy=np.array([200.0, 200.0], dtype=np.float32),
        goal_xy=np.array([400.0, 200.0], dtype=np.float32),
        world_size=512.0,
    )
    assert action.shape == (2,)
    assert action.dtype == np.float64
    assert np.all(action >= 0.0)
    assert np.all(action <= 512.0)
    assert np.all(np.isfinite(action))


def test_scripted_action_approaches_when_far():
    """When the pusher is far from the contact point, target is roughly the
    approach point behind the object (opposite the goal direction)."""
    agent_xy = np.array([10.0, 10.0], dtype=np.float32)
    object_xy = np.array([256.0, 256.0], dtype=np.float32)
    goal_xy = np.array([456.0, 256.0], dtype=np.float32)
    action = scripted_action(
        agent_xy=agent_xy, object_xy=object_xy, goal_xy=goal_xy, world_size=512.0
    )
    expected = object_xy - np.array([1.0, 0.0], dtype=np.float32) * APPROACH_OFFSET
    assert float(np.linalg.norm(action - expected)) < 1e-3


def test_scripted_action_pushes_when_close():
    """When the pusher is already at the contact point, action target is past
    the goal in the push direction."""
    object_xy = np.array([200.0, 200.0], dtype=np.float32)
    goal_xy = np.array([400.0, 200.0], dtype=np.float32)
    push_dir = np.array([1.0, 0.0], dtype=np.float32)
    agent_xy = object_xy - push_dir * APPROACH_OFFSET  # exactly on approach point
    action = scripted_action(
        agent_xy=agent_xy, object_xy=object_xy, goal_xy=goal_xy, world_size=512.0
    )
    # Action target should be in the +x direction from the object.
    assert action[0] > object_xy[0]


def test_scripted_action_handles_zero_distance():
    xy = np.array([256.0, 256.0], dtype=np.float32)
    action = scripted_action(agent_xy=xy, object_xy=xy, goal_xy=xy, world_size=512.0)
    assert np.all(np.isfinite(action))


def test_scripted_collector_runs_end_to_end():
    """We don't assert success — physics can be sticky — but the collector
    must run a few attempts without raising and report a saved count >= 0."""
    from Tsimulation.collect import scripted_collect

    with tempfile.TemporaryDirectory() as tmp:
        parser = scripted_collect.build_parser()
        args = parser.parse_args(
            [
                "--output",
                tmp,
                "--object",
                "T",
                "--pusher",
                "circle",
                "--obstacles",
                "0",
                "--num-episodes",
                "1",
                "--image-size",
                "32",
                "--max-steps",
                "60",
                "--max-tries",
                "2",
                "--success-threshold",
                "0.0",  # commit any non-empty attempt — we only care that it runs
                "--seed",
                "7",
                "--jitter",
                "0.0",
                "--headless",
            ]
        )
        # rc may be 0 or 1; we only assert it didn't raise.
        scripted_collect.run(args)


def test_mouse_collector_accepts_visual_variant_options():
    from Tsimulation.collect import mouse_collect

    args = mouse_collect.build_parser().parse_args(
        [
            "--output",
            "unused",
            "--success-threshold",
            "0.97",
            "--pusher-color",
            "blue",
            "--variant-name",
            "blue_circle",
        ]
    )

    assert args.success_threshold == 0.97
    assert args.pusher_color == "blue"
    assert args.variant_name == "blue_circle"


def _add_step(
    writer: ZarrDemoWriter,
    rng: np.random.Generator,
    image_size: int = 8,
    reward: float | None = None,
) -> None:
    """Helper: feed one synthetic step through the new split-pose API."""
    writer.add_step(
        image=rng.integers(0, 255, size=(image_size, image_size, 3), dtype=np.uint8),
        pusher_obs_pose=rng.standard_normal(2).astype(np.float32),
        object_obs_pose=rng.standard_normal(3).astype(np.float32),
        pusher_cmd_pose=rng.uniform(0, 512, size=2).astype(np.float32),
        action=rng.uniform(0, 512, size=2).astype(np.float32),
        reward=float(rng.uniform()) if reward is None else reward,
        goal_pose=rng.standard_normal(3).astype(np.float32),
    )


def test_dataset_stats_on_synthesized_dataset():
    """Synthesize two short episodes via the writer, then run stats."""
    from Tsimulation.examples import dataset_stats

    with tempfile.TemporaryDirectory() as tmp:
        env_args = {
            "object_shape": "T",
            "pusher_shape": "circle",
            "obstacle_level": 0,
        }
        writer = ZarrDemoWriter(path=tmp, env_args=env_args, image_size=8)
        rng = np.random.default_rng(0)
        for ep_len in (3, 5):
            writer.start_episode()
            for k in range(ep_len):
                _add_step(writer, rng, reward=float(k) / ep_len)
            writer.commit_episode()
        writer.close()

        stats = dataset_stats._summarize(Path(tmp))
        assert stats.n_episodes == 2
        assert stats.n_with_data == 2
        assert stats.total_frames == 3 + 5
        assert stats.length_min == 3
        assert stats.length_max == 5
        assert ("T", "circle", 0) in stats.per_config
        assert stats.per_config[("T", "circle", 0)].count == 2
        # printing should not raise
        dataset_stats._print(stats, success_threshold=0.5)


def test_visualize_episode_save_mp4(tmp_path):
    """Headless render: save a short MP4 from a synthesized 4-frame episode."""
    # The visualize module lives at the Tsimulation package root on this
    # branch (not under Tsimulation/examples/).
    from Tsimulation import visualize_episode

    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    env_args = {
        "object_shape": "T",
        "pusher_shape": "circle",
        "obstacle_level": 0,
    }
    writer = ZarrDemoWriter(path=dataset_dir, env_args=env_args, image_size=32)
    writer.start_episode()
    for k in range(4):
        writer.add_step(
            image=np.zeros((32, 32, 3), dtype=np.uint8),
            pusher_obs_pose=np.array([100.0 + k, 100.0 + k], dtype=np.float32),
            object_obs_pose=np.array([200.0, 200.0, 0.1 * k], dtype=np.float32),
            pusher_cmd_pose=np.array([150.0 + k * 10, 150.0], dtype=np.float32),
            action=np.array([150.0 + k * 10, 150.0], dtype=np.float32),
            reward=0.1 * k,
            goal_pose=np.array([400.0, 300.0, 0.5], dtype=np.float32),
        )
    writer.commit_episode()
    writer.close()

    out_mp4 = tmp_path / "out.mp4"
    rc = visualize_episode.main(
        ["--dataset", str(dataset_dir), "--episode", "0", "--save", str(out_mp4)]
    )
    assert rc == 0
    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 0


def _example_episode_init(*, obstacle_level: int = 3) -> dict:
    return {
        "agent_pos": [100.0, 110.0],
        "object_pose": [200.0, 210.0, 0.25],
        "goal_pose": [300.0, 310.0, -0.5],
        "obstacle_level": obstacle_level,
    }


def test_replay_manifest_normalizes_source_marker(tmp_path):
    manifest_path = tmp_path / "replay.json"
    manifest_path.write_text(
        json.dumps(
            {"episodes": [{**_example_episode_init(), "source": "original.zarr"}]}
        )
    )

    episodes = load_replay_manifest(manifest_path)

    assert len(episodes) == 1
    assert episodes[0][REPLAY_SOURCE_KEY] == "original.zarr"


def test_reset_to_init_restores_recorded_obstacle_geometry():
    """Replay geometry must not depend on the current obstacle-level catalog."""
    from Tsimulation.pushshapes.env import PushShapesEnv

    recorded_obstacles = [[[41.0, 73.0], [419.0, 73.0]]]
    episode_init = {
        **_example_episode_init(obstacle_level=2),
        "agent_angle": 0.0,
        "obstacles": recorded_obstacles,
        "reset_seed": 17,
    }
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="circle",
        obstacle_level=2,
        image_size=16,
        solid_pusher=False,
        socket_inside_friction_only=False,
        solid_contact_guard=False,
    )
    env._skip_obs_render = True
    try:
        reset_to_init(env, episode_init)
        assert env.get_episode_init()["obstacles"] == recorded_obstacles
        assert env.solid_pusher is True
        assert env.socket_inside_friction_only is True
        assert env.solid_contact_guard is True

        # Neither legacy constructor values nor replay metadata may switch
        # the fixed Sim V2 collision model.
        reset_to_init(
            env,
            {
                **episode_init,
                "solid_pusher": False,
                "socket_inside_friction_only": False,
                "solid_contact_guard": False,
            },
        )
        assert env.solid_pusher is True
        assert env.socket_inside_friction_only is True
        assert env.solid_contact_guard is True
        with pytest.raises(AttributeError):
            env.solid_contact_guard = False
    finally:
        env.close()


def test_pose_resume_key_rounds_jitter_and_separates_obstacle_levels():
    original = _example_episode_init(obstacle_level=2)
    jittered = {
        **original,
        "agent_pos": [100.0000001, 109.9999999],
    }
    other_level = {**original, "obstacle_level": 4}

    assert init_pose_key(jittered) == init_pose_key(original)
    assert init_pose_key(other_level) != init_pose_key(original)


def test_replay_zarr_loading_and_resume_keys(tmp_path):
    episode_init = {
        **_example_episode_init(obstacle_level=0),
        REPLAY_SOURCE_KEY: "source_episode.zarr",
    }
    writer = ZarrDemoWriter(
        path=tmp_path,
        env_args={
            "object_shape": "T",
            "pusher_shape": "circle",
            "obstacle_level": 0,
        },
        image_size=8,
    )
    writer.start_episode(init_state=episode_init)
    _add_step(writer, np.random.default_rng(12))
    writer.commit_episode()
    writer.close()

    loaded = load_replay_inits(tmp_path)
    source_names, pose_keys = collected_resume_keys(tmp_path)

    assert len(loaded) == 1
    assert loaded[0][REPLAY_SOURCE_KEY].endswith(".zarr")
    assert source_names == {"source_episode.zarr"}
    assert pose_keys == set()


def test_resume_pose_fallback_is_only_used_without_source_marker(tmp_path):
    legacy_init = _example_episode_init(obstacle_level=0)
    writer = ZarrDemoWriter(
        path=tmp_path,
        env_args={
            "object_shape": "T",
            "pusher_shape": "circle",
            "obstacle_level": 0,
        },
        image_size=8,
    )
    writer.start_episode(init_state=legacy_init)
    _add_step(writer, np.random.default_rng(13))
    writer.commit_episode()
    writer.close()

    source_names, pose_keys = collected_resume_keys(tmp_path)

    assert source_names == set()
    assert pose_keys == {init_pose_key(legacy_init)}


def test_collected_seed_keys_include_obstacle_level(tmp_path):
    episode_init = {
        **_example_episode_init(obstacle_level=7),
        "reset_seed": 123,
    }
    writer = ZarrDemoWriter(
        path=tmp_path,
        env_args={
            "object_shape": "T",
            "pusher_shape": "circle",
            "obstacle_level": 7,
        },
        image_size=8,
    )
    writer.start_episode(init_state=episode_init)
    _add_step(writer, np.random.default_rng(14))
    writer.commit_episode()
    writer.close()

    assert collected_seed_keys(tmp_path) == {(7, 123)}
