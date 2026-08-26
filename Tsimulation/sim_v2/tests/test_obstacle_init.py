from __future__ import annotations

import json
from collections import Counter

import pytest
import zarr

from Tsimulation.sim_v2.collect.obstacle_init import (
    curate_manifest,
    evaluate_candidate,
    level_entries,
    load_manifest,
    reset_to_manifest_entry,
    write_manifest,
)
from Tsimulation.sim_v2.collect.replay_init import (
    ObstacleInitKey,
    collected_obstacle_init_keys,
)
from Tsimulation.sim_v2.examples.curate_obstacle_inits import (
    parse_levels,
    plot_silhouette_manifest,
)
from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv


def test_level_specific_manifest_roundtrip_and_replay(tmp_path):
    manifest = curate_manifest(levels=[23], count=4)
    path = write_manifest(manifest, tmp_path / "obstacle_inits.json")
    loaded = load_manifest(path)
    entries = level_entries(loaded, 23)

    assert len(entries) == 4
    assert len({entry["seed"] for entry in entries}) == 4
    assert manifest["criteria"]["min_arena_clearance"] == 8.0
    assert all(entry["start_arena_clearance"] >= 8.0 for entry in entries)
    assert all(entry["goal_arena_clearance"] >= 8.0 for entry in entries)
    assert all(entry["start_obstacle_clearance"] >= 10.0 for entry in entries)
    assert all(entry["goal_obstacle_clearance"] >= 10.0 for entry in entries)
    assert Counter(
        (entry["primary_hit_segment"], entry["crossing_direction"]) for entry in entries
    ) == Counter({(0, -1): 1, (0, 1): 1, (1, -1): 1, (1, 1): 1})

    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=23,
        image_size=8,
    )
    try:
        for entry in entries:
            reset_to_manifest_entry(env, entry, verify=True)
    finally:
        env.close()


def test_manifest_rejects_wrong_level_and_stale_geometry(tmp_path):
    manifest = curate_manifest(levels=[23], count=1)
    entry = manifest["levels"]["23"][0]

    wrong_level = dict(entry, level=24)
    manifest["levels"]["23"] = [wrong_level]
    path = write_manifest(manifest, tmp_path / "wrong_level.json")
    with pytest.raises(ValueError, match="mislabeled"):
        level_entries(load_manifest(path), 23)

    stale = dict(entry, geometry_hash="0" * 64)
    manifest["levels"]["23"] = [stale]
    path = write_manifest(manifest, tmp_path / "stale_geometry.json")
    with pytest.raises(ValueError, match="stale"):
        level_entries(load_manifest(path), 23)


def test_candidate_evaluation_preserves_render_setting():
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=23,
        image_size=8,
    )
    try:
        env._skip_obs_render = False
        evaluate_candidate(env, seed=0)
        assert env._skip_obs_render is False

        env._skip_obs_render = True
        evaluate_candidate(env, seed=1)
        assert env._skip_obs_render is True
    finally:
        env.close()


def test_silhouette_plot_is_a_separate_png(tmp_path):
    manifest = curate_manifest(levels=[23], count=4)
    destination = tmp_path / "silhouettes.png"
    assert plot_silhouette_manifest(manifest, destination) == destination.resolve()
    assert destination.stat().st_size > 10_000


def test_curated_resume_requires_full_manifest_identity(tmp_path):
    output = tmp_path / "episodes"
    output.mkdir()
    legacy = zarr.open_group(str(output / "legacy.zarr"), mode="w")
    legacy.attrs["episode_init"] = json.dumps({"obstacle_level": 23, "reset_seed": 7})

    current = zarr.open_group(str(output / "current.zarr"), mode="w")
    current.attrs["episode_init"] = json.dumps(
        {
            "obstacle_level": 23,
            "reset_seed": 7,
            "control_gap_mode": "ideal",
            "obstacle_init": {
                "manifest_sha256": "abc",
                "sampler_revision": "v1",
                "geometry_hash": "geometry",
                "entry_index": 2,
            },
        }
    )

    assert collected_obstacle_init_keys(output) == {
        ObstacleInitKey("abc", "v1", "geometry", 23, 7, 2, "ideal")
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1-3", [1, 2, 3]), ("1,4-5,30", [1, 4, 5, 30])],
)
def test_parse_levels(raw, expected):
    assert parse_levels(raw) == expected
