from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import zarr
from shapely.geometry import Point, box

from Tsimulation.sim_v2.collect.obstacle_init import (
    curate_manifest,
    evaluate_candidate,
    level_bank_sha256,
    level_entries,
    level_init_policy,
    load_manifest,
    reset_to_manifest_entry,
    resolve_seed_search_limit,
    serialize_level_init_policy,
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
from Tsimulation.sim_v2.pushshapes.shapes import object_polygon

CANONICAL_MANIFEST_PATH = (
    Path(__file__).parents[1]
    / "collect"
    / "manifests"
    / "chain_obstacle_seed_bank_v1.json"
)


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
    manifest["level_bank_sha256"]["23"] = level_bank_sha256(manifest, 23)
    path = write_manifest(manifest, tmp_path / "wrong_level.json")
    with pytest.raises(ValueError, match="mislabeled"):
        level_entries(load_manifest(path), 23)

    stale = dict(entry, geometry_hash="0" * 64)
    manifest["levels"]["23"] = [stale]
    manifest["level_bank_sha256"]["23"] = level_bank_sha256(manifest, 23)
    path = write_manifest(manifest, tmp_path / "stale_geometry.json")
    with pytest.raises(ValueError, match="stale"):
        level_entries(load_manifest(path), 23)


@pytest.mark.parametrize(
    ("level", "corner", "rejected_seed"),
    [(5, (0.0, 0.0), 404), (6, (512.0, 0.0), 967)],
)
def test_corner_policy_uses_full_start_and_goal_silhouettes(
    level, corner, rejected_seed
):
    policy = level_init_policy(level)
    assert len(policy.spawn_exclusions) == 1
    assert policy.spawn_exclusions[0].center == corner
    assert policy.spawn_exclusions[0].radius == 200.0

    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=level,
        image_size=8,
    )
    try:
        assert evaluate_candidate(env, rejected_seed) is None
        entries = curate_manifest(levels=[level], count=4)["levels"][str(level)]
        anchor = Point(corner)
        for entry in entries:
            for pose, clearance_key in (
                (entry["object_pose"], "start_spawn_exclusion_clearance"),
                (entry["goal_pose"], "goal_spawn_exclusion_clearance"),
            ):
                clearance = object_polygon("T", pose[:2], pose[2]).distance(anchor)
                assert clearance >= 200.0
                assert entry[clearance_key] == pytest.approx(clearance - 200.0)
    finally:
        env.close()


def test_manifest_rejects_stale_level_init_policy(tmp_path):
    manifest = curate_manifest(levels=[5], count=1)
    manifest["level_policies"]["5"]["spawn_exclusions"][0]["radius"] = 199.0
    path = write_manifest(manifest, tmp_path / "stale_policy.json")
    with pytest.raises(ValueError, match="level-init policies"):
        load_manifest(path)


def test_empty_policy_serialization_preserves_existing_bank_identity():
    assert serialize_level_init_policy(1) == {"spawn_exclusions": []}
    assert set(serialize_level_init_policy(5)) == {"spawn_exclusions"}

    manifest = json.loads(CANONICAL_MANIFEST_PATH.read_text())
    assert {
        level: manifest["level_bank_sha256"][level]
        for level in ("1", "5", "6", "21")
    } == {
        "1": "fb088d7a7a184ec285c5b436f39d2f07160b6d8fd20473b9b3b6384dcc062269",
        "5": "e13df1fff6d19ee4e99b32ebf100778e2cafb99e0b449e6e6b93d86b951a881f",
        "6": "0bf092a21c1cf763a5b221cd92f3d863ccf5289801f35cbc676b4ce4548d2edf",
        "21": "84af16690f43b78523411c22fb26e29a615c08c5f4ff61e70954b8e239f44695",
    }


def test_canonical_manifest_is_current_and_complete():
    manifest = load_manifest(CANONICAL_MANIFEST_PATH)
    assert sorted(int(level) for level in manifest["levels"]) == list(range(1, 31))
    assert set(manifest["level_seed_search_limits"]) == {
        str(level) for level in range(1, 31)
    }
    assert all(len(level_entries(manifest, level)) == 32 for level in range(1, 31))


def test_legacy_manifest_without_search_provenance_still_loads(tmp_path):
    manifest = curate_manifest(levels=[1], count=1)
    del manifest["level_seed_search_limits"]
    path = write_manifest(manifest, tmp_path / "legacy_manifest.json")
    assert len(level_entries(load_manifest(path), 1)) == 1


def test_manifest_rejects_search_limit_below_selected_seed(tmp_path):
    manifest = curate_manifest(levels=[1], count=1)
    selected_seed = manifest["levels"]["1"][0]["seed"]
    manifest["level_seed_search_limits"]["1"] = selected_seed
    path = write_manifest(manifest, tmp_path / "bad_search_limit.json")
    with pytest.raises(ValueError, match="seed-search limits"):
        load_manifest(path)


@pytest.mark.parametrize(
    ("level", "bounds"),
    [
        (25, ((0.0, 0.0, 192.0, 192.0), (320.0, 320.0, 512.0, 512.0))),
        (26, ((320.0, 0.0, 512.0, 192.0), (0.0, 320.0, 192.0, 512.0))),
    ],
)
def test_corner_gate_policy_excludes_sealed_pockets(level, bounds):
    policy = level_init_policy(level)
    assert (
        tuple(exclusion.bounds for exclusion in policy.box_spawn_exclusions) == bounds
    )
    manifest = curate_manifest(levels=[level], count=4)
    assert manifest["criteria"]["seed_limit"] == 10_000
    assert manifest["level_seed_search_limits"][str(level)] == 200_000
    entries = manifest["levels"][str(level)]
    forbidden = [box(*region) for region in bounds]
    for entry in entries:
        for pose, clearance_key in (
            (entry["object_pose"], "start_box_spawn_exclusion_clearance"),
            (entry["goal_pose"], "goal_box_spawn_exclusion_clearance"),
        ):
            polygon = object_polygon("T", pose[:2], pose[2])
            assert not any(polygon.intersects(region) for region in forbidden)
            assert entry[clearance_key] == pytest.approx(
                min(polygon.distance(region) for region in forbidden)
            )


def test_seed_search_limit_defaults_and_explicit_overrides():
    assert resolve_seed_search_limit(1) == 10_000
    assert resolve_seed_search_limit(25) == 200_000
    assert resolve_seed_search_limit(25, seed_limit=5_000) == 5_000
    assert resolve_seed_search_limit(25, seed_limit=200_000) == 200_000
    assert resolve_seed_search_limit(25, seed_limit=300_000) == 300_000
    with pytest.raises(ValueError, match="positive"):
        resolve_seed_search_limit(25, seed_limit=0)


def test_approved_levels_have_no_added_spawn_policy():
    manifest = curate_manifest(levels=[1, 2, 3, 4], count=1)
    assert manifest["level_policies"] == {}
    assert all(
        "start_spawn_exclusion_clearance" not in entries[0]
        for entries in manifest["levels"].values()
    )


def test_level_bank_hash_is_independent_of_other_levels():
    level_one = curate_manifest(levels=[1], count=2)
    combined = curate_manifest(levels=[1, 2], count=2)
    assert level_one["levels"]["1"] == combined["levels"]["1"]
    assert level_one["level_bank_sha256"]["1"] == combined["level_bank_sha256"]["1"]


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


def test_curated_resume_requires_level_bank_identity(tmp_path):
    output = tmp_path / "episodes"
    output.mkdir()
    legacy = zarr.open_group(str(output / "legacy.zarr"), mode="w")
    legacy.attrs["episode_init"] = json.dumps({"obstacle_level": 23, "reset_seed": 7})

    manifest_only = zarr.open_group(str(output / "manifest_only.zarr"), mode="w")
    manifest_only.attrs["episode_init"] = json.dumps(
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

    current = zarr.open_group(str(output / "current.zarr"), mode="w")
    current.attrs["episode_init"] = json.dumps(
        {
            "obstacle_level": 23,
            "reset_seed": 7,
            "control_gap_mode": "ideal",
            "obstacle_init": {
                "manifest_sha256": "abc",
                "level_bank_sha256": "level-abc",
                "sampler_revision": "v1",
                "geometry_hash": "geometry",
                "entry_index": 2,
            },
        }
    )

    assert collected_obstacle_init_keys(output) == {
        ObstacleInitKey("level-abc", "v1", "geometry", 23, 7, 2, "ideal")
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1-3", [1, 2, 3]), ("1,4-5,30", [1, 4, 5, 30])],
)
def test_parse_levels(raw, expected):
    assert parse_levels(raw) == expected
