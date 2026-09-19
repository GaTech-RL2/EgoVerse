import numpy as np
import pytest

from Tsimulation.sim_v2.collect.contact_controller import ContactController
from Tsimulation.sim_v2.generate.cross_embodiment import make_env, path_waypoints, state


@pytest.fixture
def source():
    return {"init": {"object_shape": "T", "object_pose": [256, 256, 0],
                     "goal_pose": [320, 256, 0], "agent_pos": [100, 100],
                     "agent_angle": 0.0},
            "object_path": np.array([[256, 256, 0], [280, 256, 0], [310, 256, 0]]),
            "provenance": {"family_id": "source/test", "split": "train"}}


def test_source_scene_survives_embodiment_swap(source):
    for embodiment in ("gripper", "umi", "circle_small", "stick", "triangle", "pentagon"):
        env = make_env(source, embodiment)
        try:
            np.testing.assert_allclose(state(env), [100, 100, 0, 256, 256, 0])
            np.testing.assert_allclose(env.goal_pose, source["init"]["goal_pose"])
        finally:
            env.close()


def test_translation_only_contact_tool_does_not_wait_for_rotation(source):
    env = make_env(source, "circle_small")
    try:
        controller = ContactController(env)
        for _ in range(5):
            action = controller()
            assert action.shape == (2,)
            assert np.isfinite(action).all()
            env.step(action)
        assert controller.state != "AIM"
    finally:
        env.close()


def test_path_preserves_goal_and_rotation_wrap(source):
    source["init"]["object_pose"][2] = np.pi - .01
    source["object_path"] = np.array([[256, 256, -np.pi + .01]])
    points = path_waypoints(source)
    assert len(points) == 2  # Crossing the angle branch cut is not a full turn.
    np.testing.assert_allclose(points[-1], source["init"]["goal_pose"])


def test_pentagon_has_five_physical_faces_and_controlled_yaw(source):
    env = make_env(source, "pentagon")
    try:
        assert env.agent.action_spec == ("x", "y", "angle")
        vertices = env._pusher_shapes[0].get_vertices()
        assert len(vertices) == 5
        np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 20.0)
        for _ in range(30):
            env.step(np.array([100.0, 100.0, .3]))
        assert abs(env.pusher_angle - .3) < .01
        assert not env.agent.active_constraints()
    finally:
        env.close()
