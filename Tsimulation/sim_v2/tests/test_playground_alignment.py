from __future__ import annotations

import numpy as np

from Tsimulation.sim_v2.examples.playground import _write_aligned_step
from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv


class _CaptureWriter:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def add_step(self, **row: object) -> None:
        self.rows.append(row)


def test_two_steps_keep_each_action_with_its_pre_step_observation() -> None:
    writer = _CaptureWriter()
    observations = [
        {
            "image": np.full((2, 2, 3), i, dtype=np.uint8),
            "agent_pos": np.array([10.0 + i, 20.0 + i]),
            "agent_angle": np.array([0.1 + i]),
            "object_pose": np.array([30.0 + i, 40.0 + i, 0.2 + i]),
            "goal_pose": np.array([50.0 + i, 60.0 + i, 0.3 + i]),
        }
        for i in range(2)
    ]
    actions = [
        np.array([100.0, 200.0, 0.4, 1.0]),
        np.array([101.0, 201.0, 0.5, 0.0]),
    ]

    for i in range(2):
        _write_aligned_step(
            writer, pre_obs=observations[i], action=actions[i], reward=float(i)
        )

    assert len(writer.rows) == 2
    for i, row in enumerate(writer.rows):
        np.testing.assert_array_equal(row["image"], observations[i]["image"])
        np.testing.assert_array_equal(
            row["pusher_obs_pose"],
            np.concatenate(
                [observations[i]["agent_pos"], observations[i]["agent_angle"]]
            ),
        )
        np.testing.assert_array_equal(
            row["object_obs_pose"], observations[i]["object_pose"]
        )
        np.testing.assert_array_equal(row["action"], actions[i])
        assert row["reward"] == float(i)


def _free_space_motion(pusher_shape: str) -> tuple[float, float]:
    env = PushShapesEnv(object_shape="T", pusher_shape=pusher_shape)
    env._skip_obs_render = True
    try:
        env.reset(seed=0)
        env.set_state(
            agent_pos=(128.0, 128.0),
            agent_angle=0.0,
            object_pose=(400.0, 400.0, 0.0),
        )
        before_pos = np.asarray(env.agent_pos, dtype=np.float64)
        action = (
            np.array([400.0, 128.0, 1.0, 0.0], dtype=np.float64)
            if pusher_shape == "chain_gripper"
            else np.array([400.0, 128.0, 1.0], dtype=np.float64)
        )
        env.step(action)
        translation = float(
            np.linalg.norm(np.asarray(env.agent_pos, dtype=np.float64) - before_pos)
        )
        rotation = abs(float(env.pusher_angle))
        return translation, rotation
    finally:
        env.close()


def test_chain_gripper_and_u_socket_have_equal_pose_speed_limits() -> None:
    socket_motion = _free_space_motion("u_socket")
    chain_motion = _free_space_motion("chain_gripper")
    np.testing.assert_allclose(chain_motion, socket_motion, rtol=0.0, atol=1e-9)


def test_chain_gripper_regression_seeds_start_clear_and_respect_speed_cap() -> None:
    # These three reset seeds produced every >200-unit/s transition in the
    # first 84 manually collected episodes before the spawn-clearance fix.
    cases = (
        (3133846279, np.array([60.8, 348.8, 0.0, 0.0])),
        (1681990799, np.array([63.2, 372.8, 0.0, 0.0])),
        (975634435, np.array([432.8, 251.2, 0.0, 0.0])),
    )
    max_step = PushShapesEnv.PUSHER_SPEED * PushShapesEnv.DT
    for seed, action in cases:
        env = PushShapesEnv(object_shape="T", pusher_shape="chain_gripper")
        env._skip_obs_render = True
        try:
            env.reset(seed=seed)
            before = np.asarray(env.agent_pos, dtype=np.float64)
            assert 86.0 <= before[0] <= env.WORLD_SIZE - 86.0
            assert 86.0 <= before[1] <= env.WORLD_SIZE - 86.0
            env.step(action)
            after = np.asarray(env.agent_pos, dtype=np.float64)
            assert np.linalg.norm(after - before) <= max_step + 1e-9
        finally:
            env.close()
