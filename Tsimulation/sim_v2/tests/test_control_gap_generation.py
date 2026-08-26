import json

import numpy as np
import zarr

from Tsimulation.sim_v2.generate.from_collected import load_sources
from Tsimulation.sim_v2.generate.mimicgen import (
    SourceDemo,
    apply_source_control_gap,
    retarget,
    sample_equivariant_layout,
    wrap,
)
from Tsimulation.sim_v2.pushshapes.agents import CONTROL_GAPS
from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv


def test_load_sources_keeps_control_gap_and_mode(tmp_path):
    episode = tmp_path / "episode_T_circle_obs0_000000.zarr"
    group = zarr.open_group(str(episode), mode="w")
    group.attrs["episode_init"] = json.dumps({
        "agent_pos": [100.0, 100.0],
        "agent_angle": 0.0,
        "object_pose": [200.0, 200.0, 0.0],
        "goal_pose": [300.0, 300.0, 0.0],
        "object_shape": "T",
        "pusher_shape": "circle",
        "obstacle_level": 0,
        "control_gap": CONTROL_GAPS["tight"].as_dict(),
        "control_gap_mode": "tight",
    })
    group.attrs["total_frames"] = 2
    group.create_array("actions", data=np.zeros((2, 2), dtype=np.float64))

    demo = load_sources(tmp_path)[0]

    assert demo.control_gap == CONTROL_GAPS["tight"].as_dict()
    assert demo.control_gap_mode == "tight"


def test_apply_source_control_gap_replaces_default_ideal_controller():
    env = PushShapesEnv(object_shape="T", pusher_shape="circle")
    env.reset(seed=7)
    demo = SourceDemo(
        agent="circle",
        actions=np.zeros((2, 2), dtype=np.float64),
        object_pose=(200.0, 200.0, 0.0),
        goal_pose=(300.0, 300.0, 0.0),
        agent_pos=(100.0, 100.0),
        control_gap=CONTROL_GAPS["tight"].as_dict(),
        control_gap_mode="tight",
    )

    apply_source_control_gap(env, demo)

    assert env.agent.control_gap.as_dict() == CONTROL_GAPS["tight"].as_dict()
    assert env.agent.randomize_gap is False
    assert env.agent._cmd_queue == []


def test_equivariant_layout_preserves_complete_scene_geometry():
    actions = np.array([
        [90.0, 120.0, -0.2],
        [140.0, 160.0, 0.1],
        [220.0, 240.0, 0.4],
        [300.0, 280.0, 0.7],
    ])
    demo = SourceDemo(
        agent="triangle",
        actions=actions,
        object_pose=(180.0, 190.0, 0.3),
        goal_pose=(300.0, 310.0, -0.4),
        agent_pos=(90.0, 120.0),
    )

    obj, goal, agent = sample_equivariant_layout(
        demo, np.random.default_rng(19)
    )
    transformed = retarget(demo, obj, goal)
    rotation = wrap(obj[2] - demo.object_pose[2])

    assert np.isclose(
        np.linalg.norm(np.subtract(goal[:2], obj[:2])),
        np.linalg.norm(np.subtract(demo.goal_pose[:2], demo.object_pose[:2])),
    )
    assert np.isclose(wrap(goal[2] - demo.goal_pose[2]), rotation)
    assert np.isclose(wrap(transformed[0, 2] - actions[0, 2]), rotation)
    assert np.all((transformed[:, :2] >= 20.0)
                  & (transformed[:, :2] <= 492.0))
    assert np.all(np.asarray(obj[:2]) >= 70.0)
    assert np.all(np.asarray(obj[:2]) <= 442.0)
    assert np.all(np.asarray(goal[:2]) >= 70.0)
    assert np.all(np.asarray(goal[:2]) <= 442.0)
    assert np.allclose(agent, transformed[0, :2])


def test_fixed_angle_stick_uses_translation_only_augmentation():
    actions = np.array([
        [80.0, 100.0],
        [150.0, 170.0],
        [240.0, 260.0],
    ])
    demo = SourceDemo(
        agent="stick",
        actions=actions,
        object_pose=(180.0, 180.0, 0.2),
        goal_pose=(300.0, 300.0, -0.5),
        agent_pos=(80.0, 100.0),
    )

    obj, goal, agent = sample_equivariant_layout(
        demo, np.random.default_rng(23)
    )
    transformed = retarget(demo, obj, goal)

    assert np.isclose(obj[2], demo.object_pose[2])
    assert np.isclose(goal[2], demo.goal_pose[2])
    assert np.allclose(
        transformed[:, :2] - actions[:, :2],
        transformed[0, :2] - actions[0, :2],
    )
    assert np.allclose(agent, transformed[0, :2])
