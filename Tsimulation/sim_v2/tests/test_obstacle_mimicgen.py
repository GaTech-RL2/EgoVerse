from __future__ import annotations

import numpy as np
import pytest

from Tsimulation.sim_v2.generate.from_collected_obstacle_local import (
    jittered_state,
)
from Tsimulation.sim_v2.generate.from_collected_obstacle_local import (
    transformed_agent_start as local_transformed_agent_start,
)
from Tsimulation.sim_v2.generate.from_collected_obstacles import (
    ManualSource,
    parse_levels,
    rank_sources,
    transformed_agent_start,
)
from Tsimulation.sim_v2.generate.mimicgen import SourceDemo


def _entry(seed: int, *, crossing_direction: int, x: float) -> dict:
    return {
        "seed": seed,
        "agent_pos": [x, 100.0],
        "agent_angle": 0.0,
        "chain_joint_angle": 0.12,
        "object_pose": [x, 200.0, 0.0],
        "goal_pose": [x, 400.0, 0.0],
        "route_type": "wall_crossing",
        "primary_hit_segment": 0,
        "crossing_direction": crossing_direction,
        "blocked_fraction": 0.5,
        "start_goal_distance": 200.0,
    }


def _source(entry: dict) -> ManualSource:
    seed = int(entry["seed"])
    init = {
        "reset_seed": seed,
        "obstacle_init": {"entry_index": 0, "level_bank_sha256": "bank"},
    }
    return ManualSource(
        path=__import__("pathlib").Path(f"episode_{seed}.zarr"),
        demo=SourceDemo(
            agent="chain_gripper",
            actions=np.zeros((4, 4)),
            object_pose=tuple(entry["object_pose"]),
            goal_pose=tuple(entry["goal_pose"]),
            agent_pos=tuple(entry["agent_pos"]),
            obstacle_level=1,
        ),
        episode_init=init,
        entry=entry,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1", [1]), ("1-3", [1, 2, 3]), ("1-2,5,7-8", [1, 2, 5, 7, 8])],
)
def test_parse_levels(raw, expected):
    assert parse_levels(raw) == expected


@pytest.mark.parametrize("raw", ["", "3-1", "0", "31", "1,1"])
def test_parse_levels_rejects_invalid_input(raw):
    with pytest.raises(ValueError):
        parse_levels(raw)


def test_rank_sources_prefers_route_group_then_feature_distance():
    target = _entry(100, crossing_direction=1, x=256.0)
    wrong_group = _source(_entry(1, crossing_direction=-1, x=256.0))
    same_far = _source(_entry(2, crossing_direction=1, x=100.0))
    same_near = _source(_entry(3, crossing_direction=1, x=250.0))
    ranked = rank_sources(target, [wrong_group, same_far, same_near])
    assert [source.reset_seed for source in ranked] == [3, 2, 1]


def test_transformed_agent_start_tracks_target_object_frame():
    source = _source(_entry(3, crossing_direction=1, x=200.0))
    target = _entry(4, crossing_direction=1, x=260.0)
    position, angle = transformed_agent_start(source, target)
    assert position == pytest.approx((260.0, 100.0))
    assert angle == pytest.approx(0.0)


def test_local_jitter_is_deterministic_and_transforms_agent_with_object():
    source = _source(_entry(3, crossing_direction=1, x=200.0))
    kwargs = dict(
        level=1,
        variant_index=2,
        retry_index=0,
        generation_seed=260826,
        jitter_xy=1.5,
        jitter_angle_radians=0.01,
    )
    first = jittered_state(source, **kwargs)
    second = jittered_state(source, **kwargs)
    assert first == second
    expected_pos, expected_angle = local_transformed_agent_start(
        source, first["object_pose"]
    )
    assert first["agent_pos"] == pytest.approx(expected_pos)
    assert first["agent_angle"] == pytest.approx(expected_angle)
    assert np.linalg.norm(first["object_delta"]) > 0


def test_local_jitter_anneals_after_each_retry_block():
    source = _source(_entry(3, crossing_direction=1, x=200.0))
    common = dict(
        source=source,
        level=1,
        variant_index=2,
        generation_seed=260826,
        jitter_xy=1.5,
        jitter_angle_radians=0.01,
    )
    assert jittered_state(retry_index=31, **common)["retry_scale"] == 1.0
    assert jittered_state(retry_index=32, **common)["retry_scale"] == 0.5
    assert jittered_state(retry_index=64, **common)["retry_scale"] == 0.25
