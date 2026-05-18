"""Smoke tests for the non-physics features:

- scripted_action heuristic returns valid targets
- scripted_collect runs an end-to-end attempt without raising
- dataset_stats aggregates a synthesized dataset
- visualize_episode renders frames headlessly to an MP4

Run with::

    SDL_VIDEODRIVER=dummy pytest Tsimulation/tests/test_features.py -q
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

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
    assert action.dtype == np.float32
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
                writer.add_step(
                    image=rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8),
                    state=rng.standard_normal(5).astype(np.float32),
                    action=rng.uniform(0, 512, size=2).astype(np.float32),
                    reward=float(k) / ep_len,  # rising reward
                    goal_pose=rng.standard_normal(3).astype(np.float32),
                )
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
    from Tsimulation.examples import visualize_episode

    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    env_args = {
        "object_shape": "T",
        "pusher_shape": "circle",
        "obstacle_level": 0,
    }
    writer = ZarrDemoWriter(path=dataset_dir, env_args=env_args, image_size=32)
    rng = np.random.default_rng(123)
    writer.start_episode()
    for k in range(4):
        writer.add_step(
            image=rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
            state=np.array(
                [100.0 + k, 100.0 + k, 200.0, 200.0, 0.1 * k], dtype=np.float32
            ),
            action=np.array([150.0 + k * 10, 150.0], dtype=np.float32),
            reward=0.1 * k,
            goal_pose=np.array([400.0, 300.0, 0.5], dtype=np.float32),
        )
    writer.commit_episode()
    writer.close()

    out_mp4 = tmp_path / "out.mp4"
    rc = visualize_episode.main(
        [
            "--dataset",
            str(dataset_dir),
            "--episode",
            "0",
            "--save",
            str(out_mp4),
        ]
    )
    assert rc == 0
    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 0
