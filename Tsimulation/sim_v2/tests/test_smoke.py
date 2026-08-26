"""Smoke tests for PushShapes env and zarr writer round-trip.

Run with::

    pytest Tsimulation/tests/test_smoke.py -q
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pymunk
import pytest
import zarr

# Headless pygame: required when CI / a remote shell has no display.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from Tsimulation.collect.zarr_writer import (
    ACTION_KEY,
    CMD_PUSHER_KEY,
    GOAL_KEY,
    IMAGE_KEY,
    REWARD_KEY,
    STATE_KEY,
    ZarrDemoWriter,
)
from Tsimulation.pushshapes.env import PushShapesEnv
from Tsimulation.pushshapes.render import PUSHER_COLOR
from Tsimulation.pushshapes.shapes import (
    CHAIN_GRIPPER_LINK_HALF_W,
    CHAIN_GRIPPER_LINK_LEN,
    FLIPPER_SWING,
    SHAPES,
    U_SOCKET_CROSSBAR_INNER_X,
    U_SOCKET_INNER_GAP,
    U_SOCKET_PRONG_LENGTH,
)

SHAPES_TO_TEST = list(SHAPES.keys())
PUSHERS = ["circle", "stick", "u_socket"]
OBSTACLES = [0, 1, 2, 3]


def test_stick_keeps_one_fixed_world_angle_during_translation():
    """The 2-D stick action may translate but can never rotate or auto-yaw."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="stick",
        obstacle_level=0,
        image_size=16,
        seed=9,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=9)
        env.set_obstacles([])
        env.set_state(
            agent_pos=(100.0, 100.0),
            agent_angle=1.2,
            object_pose=(400.0, 400.0, 0.0),
            goal_pose=(450.0, 450.0, 0.0),
        )

        for target in ((300.0, 100.0), (300.0, 300.0), (100.0, 300.0)):
            for _ in range(10):
                env.step(np.asarray(target, dtype=np.float64))
            assert env.pusher_angle == pytest.approx(0.0, abs=1e-12)

        assert env.agent.action_spec == ("x", "y")
        assert env.agent.auto_orients is False
        assert env.agent.fixed_angle == pytest.approx(0.0)
    finally:
        env.close()


def test_flipper_bar_is_solid_and_swing_is_rate_limited():
    env = PushShapesEnv(
        object_shape="T", pusher_shape="flipper", image_size=16, seed=0,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=0)
        env.set_obstacles([])
        env.set_state(
            object_pose=(256.0, 256.0, 0.0),
            goal_pose=(350.0, 350.0, 0.0),
            agent_pos=(188.0, 256.0),
            agent_angle=0.0,
        )

        assert env.agent._bar_shape in env.agent.physics_shapes(env)
        action = np.array([188.0, 256.0, 0.0, 1.0])
        env.step(action)

        expected = math.radians(90.0) * env.DT / FLIPPER_SWING
        assert env.agent._swing_state == pytest.approx(expected)
        assert env.agent._swing_state < 0.03
        assert env._pusher_object_penetration_depth() <= 0.5 + 1e-8
    finally:
        env.close()


def test_flipper_full_sweep_pushes_without_crossing_through_t():
    env = PushShapesEnv(
        object_shape="T", pusher_shape="flipper", image_size=16, seed=0,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=0)
        env.set_obstacles([])
        env.set_state(
            object_pose=(256.0, 256.0, 0.0),
            goal_pose=(350.0, 350.0, 0.0),
            agent_pos=(188.0, 256.0),
            agent_angle=0.0,
        )
        start = np.asarray(env.object_pose[:2])
        action = np.array([188.0, 256.0, 0.0, 1.0])
        max_depth = 0.0
        for _ in range(55):
            env.step(action)
            max_depth = max(max_depth, env._pusher_object_penetration_depth())

        travelled = np.linalg.norm(np.asarray(env.object_pose[:2]) - start)
        assert env.agent._swing_state > 0.99
        assert travelled > 1.0
        assert max_depth <= 0.5 + 1e-8
    finally:
        env.close()


def test_chain_gripper_is_exactly_four_rigid_serial_links():
    """Only three hinges move; none of the four bars bends or changes size."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=0,
        image_size=32,
        seed=2,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=2)
        env.set_obstacles([])
        env.set_state(
            agent_pos=(256.0, 256.0),
            agent_angle=0.0,
            object_pose=(430.0, 430.0, 0.0),
            goal_pose=(80.0, 80.0, 0.0),
        )

        assert env._pusher_shapes == []  # no hidden palm, hub, or crossbar
        shapes = env.agent.physics_shapes(env)
        assert len(env.agent._link_shapes) == 4
        assert len(env.agent._joint_shapes) == 3
        assert len(shapes) == 7
        assert env.action_space.shape == (4,)
        assert env.solid_pusher is True
        assert env.solid_contact_guard is True

        open_gap = env.agent.mouth_gap
        for _ in range(60):
            env.step(np.array([256.0, 256.0, 0.0, 1.0], dtype=np.float64))

        bodies = env.agent._link_bodies
        assert len(bodies) == 4
        assert env.agent.joint_angles == pytest.approx(
            (env.agent.joint_angle,) * 3,
            abs=1e-12,
        )
        for left, right in zip(bodies, bodies[1:]):
            left_end = left.local_to_world((CHAIN_GRIPPER_LINK_LEN / 2.0, 0.0))
            right_start = right.local_to_world((-CHAIN_GRIPPER_LINK_LEN / 2.0, 0.0))
            assert (left_end - right_start).length <= 1e-9
            relative_angle = (right.angle - left.angle + np.pi) % (2 * np.pi) - np.pi
            assert relative_angle == pytest.approx(env.agent.joint_angle, abs=1e-12)

        for shape in env.agent._link_shapes:
            assert isinstance(shape, pymunk.Poly)
            vertices = np.asarray([(v.x, v.y) for v in shape.get_vertices()])
            assert np.ptp(vertices[:, 0]) == pytest.approx(CHAIN_GRIPPER_LINK_LEN)
            assert np.ptp(vertices[:, 1]) == pytest.approx(
                2.0 * CHAIN_GRIPPER_LINK_HALF_W
            )
        assert all(isinstance(shape, pymunk.Circle) for shape in env.agent._joint_shapes)
        assert env.agent.mouth_gap < open_gap - 80.0
    finally:
        env.close()


def test_chain_gripper_solid_guard_includes_every_link():
    """A T teleported through a link is restored to the previous safe pose."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=0,
        image_size=16,
        seed=3,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=3)
        env.set_obstacles([])
        safe_object_pose = (430.0, 430.0, 0.0)
        env.set_state(
            agent_pos=(256.0, 256.0),
            agent_angle=0.0,
            object_pose=safe_object_pose,
            goal_pose=(80.0, 80.0, 0.0),
        )
        captured = env.agent.pre_substep(env)

        link = env.agent._link_bodies[0]
        env.set_state(object_pose=(float(link.position.x), float(link.position.y), 0.0))
        assert env._pusher_object_penetration_depth() > 0.5

        env.agent.post_substep(env, captured)

        assert env.object_pose == pytest.approx(safe_object_pose)
        assert env._pusher_object_penetration_depth() <= 0.5 + 1e-6

    finally:
        env.close()


def test_chain_gripper_closes_around_and_rotates_t_without_tunnelling():
    """The one curl command forms a guarded grasp and keeps all links solid."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="chain_gripper",
        obstacle_level=0,
        image_size=16,
        seed=4,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=4)
        env.set_obstacles([])
        env.set_state(
            agent_pos=(256.0, 350.0),
            agent_angle=np.pi,
            object_pose=(256.0, 256.0, 0.0),
            goal_pose=(80.0, 80.0, 0.0),
        )

        max_depth = 0.0
        for _ in range(60):
            env.step(np.array([256.0, 350.0, np.pi, 1.0], dtype=np.float64))
            max_depth = max(max_depth, env._pusher_object_penetration_depth())
            if env.agent.grasped:
                break
        assert env.agent.grasped
        assert max_depth <= 0.5 + 1e-6

        # Releasing the close key sends the current fraction, holding the
        # aperture while A/D continues to rotate the caught T.
        held_grip = env.agent.grip_fraction
        held_joint_angle = env.agent.joint_angle
        master_before = env.pusher_angle
        object_before = env.object_pose[2]
        target_angle = master_before + np.pi / 2.0
        for _ in range(30):
            env.step(
                np.array(
                    [256.0, 350.0, target_angle, held_grip],
                    dtype=np.float64,
                )
            )
            max_depth = max(max_depth, env._pusher_object_penetration_depth())

        master_delta = env.pusher_angle - master_before
        object_delta = env.object_pose[2] - object_before
        assert master_delta > 1.4
        assert object_delta == pytest.approx(master_delta, abs=1e-8)
        assert env.agent.joint_angle == pytest.approx(held_joint_angle, abs=1e-12)
        assert max_depth <= 0.5 + 1e-6

        closed_gap = env.agent.mouth_gap
        for _ in range(60):
            env.step(
                np.array([256.0, 350.0, target_angle, 0.0], dtype=np.float64)
            )
        assert not env.agent.grasped
        assert env.agent.mouth_gap > closed_gap + 60.0
    finally:
        env.close()


def test_gripper_observation_renders_live_parallel_jaws():
    """Recorded camera frames must show the articulated jaw configuration."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="gripper",
        obstacle_level=0,
        image_size=512,
        seed=42,
    )
    try:
        env.reset(seed=42)
        env.set_state(
            agent_pos=(256.0, 256.0),
            agent_angle=0.0,
            object_pose=(100.0, 100.0, 0.0),
            goal_pose=(400.0, 400.0, 0.0),
        )

        open_obs, *_ = env.step(
            np.array([256.0, 256.0, 0.0, 0.0], dtype=np.float64)
        )
        for _ in range(12):
            closed_obs, *_ = env.step(
                np.array([256.0, 256.0, 0.0, 1.0], dtype=np.float64)
            )

        colour = np.asarray(PUSHER_COLOR, dtype=np.uint8)
        open_mask = np.all(open_obs["image"] == colour, axis=-1)
        closed_mask = np.all(closed_obs["image"] == colour, axis=-1)
        # Look in front of the palm, where only the two fingers are present.
        open_x = np.flatnonzero(open_mask[268:290].any(axis=0))
        closed_x = np.flatnonzero(closed_mask[268:290].any(axis=0))

        assert open_x.size > 0 and closed_x.size > 0
        assert np.ptp(open_x) > np.ptp(closed_x) + 20
        assert not np.array_equal(open_obs["image"], closed_obs["image"])
    finally:
        env.close()


def test_gripper_solid_contact_guard_includes_both_jaws():
    """A jaw-only overlap must be detected and rolled back by Sim V2."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="gripper",
        obstacle_level=0,
        image_size=16,
        seed=1,
    )
    try:
        env.reset(seed=1)
        env.set_state(
            agent_pos=(256.0, 256.0),
            agent_angle=0.0,
            object_pose=(400.0, 400.0, 0.0),
            goal_pose=(420.0, 420.0, 0.0),
        )
        env.agent._sync(env)
        captured = env.agent.pre_substep(env)

        # At this pose the T overlaps the left jaw by 18 units but does not
        # touch the palm.  This was invisible when the guard considered only
        # env._pusher_shapes (the palm returned by Agent.build()).
        env.set_state(object_pose=(180.0, 310.0, 0.0))
        env.agent._sync(env)
        assert len(env.agent.physics_shapes(env)) == 3
        assert env._pusher_object_penetration_depth() > 0.5

        env.agent.post_substep(env, captured)

        assert env.object_pose == pytest.approx((400.0, 400.0, 0.0))
        assert env._pusher_object_penetration_depth() <= 0.5 + 1e-6
    finally:
        env.close()


def test_gripper_solid_static_guard_includes_both_jaws():
    """The jaw tips, not merely the palm, must stay inside arena walls."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="gripper",
        obstacle_level=0,
        image_size=16,
        seed=1,
    )
    try:
        env.reset(seed=1)
        env.set_state(
            agent_pos=(256.0, 480.0),
            agent_angle=0.0,
            object_pose=(100.0, 100.0, 0.0),
            goal_pose=(420.0, 420.0, 0.0),
        )
        env.agent._sync(env)

        before = env._shapes_static_penetration_depth(
            env._pusher_body,
            list(env.agent.physics_shapes(env)),
        )
        assert before > 1.0

        env._clamp_pusher_to_static()

        after = env._shapes_static_penetration_depth(
            env._pusher_body,
            list(env.agent.physics_shapes(env)),
        )
        assert after <= 1e-6
    finally:
        env.close()


def test_gripper_cannot_ratchet_jaws_through_object_into_wall():
    """Jaw sync must not hide progressively accumulated penetration."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="gripper",
        obstacle_level=0,
        image_size=16,
        seed=1,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=1)
        env.set_obstacles([((40.0, 256.0), (472.0, 256.0))])
        env.set_state(
            agent_pos=(256.0, 70.0),
            agent_angle=0.0,
            object_pose=(256.0, 170.0, 0.0),
            goal_pose=(400.0, 400.0, 0.0),
        )

        max_pusher_depth = 0.0
        max_static_depth = 0.0
        for _ in range(120):
            env.step(np.array([256.0, 430.0, 0.0, 0.0], dtype=np.float64))
            max_pusher_depth = max(
                max_pusher_depth,
                env._pusher_object_penetration_depth(),
            )
            max_static_depth = max(
                max_static_depth,
                env._object_static_penetration_depth(),
            )

        assert max_pusher_depth <= 0.5 + 1e-6
        assert max_static_depth <= 0.2 + 1e-6
        assert env.object_pose[1] < 180.0
    finally:
        env.close()


@pytest.mark.parametrize("agent_x", [255.0, 256.0, 257.0])
def test_gripper_grasp_transfers_commanded_wrist_rotation_to_t(agent_x):
    """A caught T must rotate rigidly with the parallel-gripper wrist."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="gripper",
        obstacle_level=0,
        image_size=16,
        seed=4,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=4)
        env.set_state(
            # Centre the open jaws around the distal end of the 30-wide stem.
            agent_pos=(agent_x, 341.0),
            agent_angle=np.pi,
            object_pose=(256.0, 256.0, 0.0),
            goal_pose=(100.0, 100.0, 0.0),
        )
        env.agent._jaw_cmd = 1.0
        env.agent._sync(env)
        for _ in range(20):
            env.step(np.array([agent_x, 341.0, np.pi, 1.0], dtype=np.float64))
            if env.agent.grasped:
                break
        assert env.agent.grasped
        assert env._pusher_object_penetration_depth() <= 0.5 + 1e-6

        wrist_before = env.pusher_angle
        object_before = env.object_pose[2]
        captured_local_pos = env.agent._grasp_local_object_pos
        captured_angle_offset = env.agent._grasp_angle_offset
        target_angle = wrist_before + np.pi / 2
        max_local_position_error = 0.0
        max_angle_offset_error = 0.0
        max_gripper_object_depth = 0.0
        for _ in range(30):
            x, y = env.agent_pos
            env.step(np.array([x, y, target_angle, 1.0], dtype=np.float64))
            local_pos = env._pusher_body.world_to_local(env._object_body.position)
            max_local_position_error = max(
                max_local_position_error,
                np.linalg.norm(np.asarray(local_pos) - np.asarray(captured_local_pos)),
            )
            max_angle_offset_error = max(
                max_angle_offset_error,
                abs(
                    (env.object_pose[2] - env.pusher_angle)
                    - captured_angle_offset
                ),
            )
            max_gripper_object_depth = max(
                max_gripper_object_depth,
                env._pusher_object_penetration_depth(),
            )

        wrist_delta = env.pusher_angle - wrist_before
        object_delta = env.object_pose[2] - object_before
        assert env.agent.grasped
        assert wrist_delta > 1.4
        assert object_delta == pytest.approx(wrist_delta, abs=1e-8)
        assert max_local_position_error <= 1e-8
        assert max_angle_offset_error <= 1e-8
        assert max_gripper_object_depth <= 0.5 + 1e-6
    finally:
        env.close()


def test_umi_solid_guard_sees_and_stops_revolute_fingers():
    """Closing either orange UMI finger into the T must stop at contact."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="umi",
        obstacle_level=0,
        image_size=16,
        seed=4,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=4)
        env.set_obstacles([])
        env.set_state(
            agent_pos=(256.0, 175.0),
            agent_angle=0.0,
            object_pose=(256.0, 256.0, 0.0),
            goal_pose=(80.0, 80.0, 0.0),
        )

        # Wrist circle + both articulated finger polygons. Before this fix only
        # the wrist was returned, so a finger could be 25 px inside the T while
        # the environment incorrectly reported zero penetration.
        assert len(env.agent.physics_shapes(env)) == 3
        assert env._pusher_object_penetration_depth() == pytest.approx(0.0)

        max_depth = 0.0
        for _ in range(160):
            env.step(np.array([256.0, 175.0, 0.0, 1.0], dtype=np.float64))
            max_depth = max(max_depth, env._pusher_object_penetration_depth())

        assert max_depth <= 0.5 + 1e-6
        # The close command remains active, but the physical aperture stalls at
        # the last collision-safe state instead of teleporting through the T.
        assert env.agent._grip == pytest.approx(0.0)
        assert env.agent._grip_state > env.agent._grip
    finally:
        env.close()


def test_umi_solid_guard_stops_wrist_motion_before_finger_tunnels_through_t():
    """Translation is guarded by the fingers, not only by the wrist circle."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="umi",
        obstacle_level=0,
        image_size=16,
        seed=5,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=5)
        env.set_obstacles([])
        env.set_state(
            agent_pos=(256.0, 140.0),
            agent_angle=0.0,
            object_pose=(256.0, 256.0, 0.0),
            goal_pose=(80.0, 80.0, 0.0),
        )

        max_depth = 0.0
        for _ in range(120):
            env.step(np.array([256.0, 330.0, 0.0, 1.0], dtype=np.float64))
            max_depth = max(max_depth, env._pusher_object_penetration_depth())

        assert max_depth <= 0.5 + 1e-6
        assert env.agent_pos[1] < 200.0
    finally:
        env.close()


def test_spring_is_one_continuous_solid_guarded_assembly():
    """Housing, telescoping shaft and auxiliary tip must all be solid."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="spring",
        obstacle_level=0,
        image_size=16,
        seed=6,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=6)
        env.set_obstacles([])
        safe_object_pose = (430.0, 430.0, 0.0)
        env.set_state(
            agent_pos=(256.0, 256.0),
            agent_angle=0.0,
            object_pose=safe_object_pose,
            goal_pose=(80.0, 80.0, 0.0),
        )
        captured = env.agent.pre_substep(env)

        shapes = env.agent.physics_shapes(env)
        assert len(shapes) == 3
        assert all(not shape.sensor for shape in shapes)
        assert all(shape.friction == pytest.approx(0.0) for shape in shapes)
        assert env.agent._shaft_shape in shapes
        assert env.agent._tip_shape in shapes
        assert isinstance(env.agent._tip_shape, pymunk.Segment)

        # At full extension the tapered shaft exactly bridges the old empty
        # pocket between the housing face and the rear face of the tip.
        shaft_vertices = env.agent._shaft_shape.get_vertices()
        shaft_ys = [float(vertex.y) for vertex in shaft_vertices]
        assert min(shaft_ys) == pytest.approx(-41.0)
        assert max(shaft_ys) == pytest.approx(-15.0)

        # The visible orange tip is a separate body, but its collision is now
        # part of the same guarded geometry as the housing and shaft.
        tip = env.agent._tip
        env.set_state(
            object_pose=(float(tip.position.x), float(tip.position.y), 0.0)
        )
        assert env._pusher_object_penetration_depth() > 0.5

        env.agent.post_substep(env, captured)

        assert env.object_pose == pytest.approx(safe_object_pose)
        assert env._pusher_object_penetration_depth() <= 0.5 + 1e-6

        # This pose puts only the far edge of the T's top bar in the former
        # empty housing-to-tip gap. It touched neither old disconnected part,
        # which is how the T slipped between them and became stuck. The new
        # shaft occupies that exact pocket and the same guard rejects it.
        captured = env.agent.pre_substep(env)
        env.set_state(object_pose=(180.0, 246.0, math.pi / 12.0))

        def object_penetration(shape):
            env._space.reindex_shapes_for_body(env._object_body)
            env._space.reindex_shapes_for_body(shape.body)
            depths = [
                abs(float(point.distance))
                for query in env._space.shape_query(shape)
                if query.shape.body is env._object_body
                for point in query.contact_point_set.points
            ]
            return max(depths, default=0.0)

        assert object_penetration(env._pusher_shapes[0]) <= 1e-6  # housing
        assert object_penetration(env.agent._shaft_shape) > 0.5
        assert object_penetration(env.agent._tip_shape) < 0.1
        assert env._pusher_object_penetration_depth() > 0.5

        env.agent.post_substep(env, captured)

        assert env.object_pose == pytest.approx(safe_object_pose)
        assert env._pusher_object_penetration_depth() <= 0.5 + 1e-6
    finally:
        env.close()


def test_soft_spring_compresses_instead_of_deadlocking_at_blocked_contact():
    """A wall-blocked T must not pin the base before its spring retracts."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="spring",
        obstacle_level=0,
        image_size=16,
        seed=7,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=7)
        env.set_obstacles([])
        # T's top edge is flush with the arena wall. It cannot yield farther
        # upward when the spring approaches its stem from below.
        env.set_state(
            agent_pos=(256.0, 200.0),
            agent_angle=0.0,
            object_pose=(256.0, 46.0, 0.0),
            goal_pose=(400.0, 400.0, 0.0),
        )

        start_base_y = env.agent_pos[1]
        max_tip_depth = 0.0
        max_wall_depth = 0.0
        max_compression = 0.0
        for _ in range(30):
            env.step(np.array([256.0, 60.0, 0.0, 0.0], dtype=np.float64))
            max_compression = max(max_compression, env.agent.compression)
            max_tip_depth = max(
                max_tip_depth,
                env._pusher_object_penetration_depth(),
            )
            max_wall_depth = max(
                max_wall_depth,
                env._object_static_penetration_depth(),
            )

        assert max_compression > 15.0
        assert start_base_y - env.agent_pos[1] > 45.0
        assert max_tip_depth <= 0.5 + 1e-6
        assert max_wall_depth <= 0.2 + 1e-6
    finally:
        env.close()


def test_bottomed_out_spring_pushes_through_corner_solver_residue():
    """Normal transient solver residue must not pin a bottomed-out spring."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="spring",
        obstacle_level=0,
        image_size=16,
        seed=8,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=8)
        env.set_obstacles([])
        # Approach the lower-left convex corner. Before contact projection the
        # most substeps were rolled back at shallow transient penetration.
        corner = np.array([241.0, 331.0])
        forward = -(corner - np.array([256.0, 256.0]))
        forward /= np.linalg.norm(forward)
        base = corner - forward * 55.0
        angle = math.atan2(forward[0], -forward[1])
        env.set_state(
            agent_pos=(float(base[0]), float(base[1])),
            agent_angle=angle,
            object_pose=(256.0, 256.0, 0.0),
            goal_pose=(400.0, 400.0, 0.0),
        )
        start = np.array(env.agent_pos)
        object_start = np.array(env.object_pose[:2])
        target = start + forward * 80.0
        max_depth = 0.0
        for _ in range(40):
            env.step(np.array([*target, angle, 0.0], dtype=np.float64))
            max_depth = max(max_depth, env._pusher_object_penetration_depth())

        assert np.linalg.norm(np.array(env.agent_pos) - start) > 70.0
        assert np.linalg.norm(np.array(env.object_pose[:2]) - object_start) > 10.0
        assert max_depth <= 0.5 + 1e-6
    finally:
        env.close()


def test_umi_centered_stem_grasp_turns_green_without_penetration():
    """A valid two-finger stem contact must create the displayed grasp state."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="umi",
        obstacle_level=0,
        image_size=16,
        seed=6,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=6)
        env.set_obstacles([])
        env.set_state(
            # Fingers point upward around the distal 30-pixel T stem.
            agent_pos=(256.0, 365.0),
            agent_angle=np.pi,
            object_pose=(256.0, 256.0, 0.0),
            goal_pose=(80.0, 80.0, 0.0),
        )

        max_depth = 0.0
        for _ in range(40):
            env.step(np.array([256.0, 365.0, np.pi, 1.0], dtype=np.float64))
            max_depth = max(max_depth, env._pusher_object_penetration_depth())
            if env.agent.grasped:
                break

        assert env.agent.grasped
        assert env.agent.mode == "clamped"
        assert env.agent._both_fingers_contact_object(env)
        assert max_depth <= 0.5 + 1e-6
        assert env.agent._held_gap > 30.0
    finally:
        env.close()


def test_umi_grasped_pair_cannot_drag_t_through_arena_edge():
    """A constrained UMI must stop at the wall and release inside the arena."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="umi",
        obstacle_level=0,
        image_size=16,
        seed=6,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=6)
        env.set_obstacles([])
        env.set_state(
            # Rotate the T so the downward-facing UMI can clamp its stem, then
            # command the exact failure from the collector: drag it through
            # the lower arena edge and subsequently open/move away.
            agent_pos=(256.0, 145.0),
            agent_angle=0.0,
            object_pose=(256.0, 256.0, np.pi),
            goal_pose=(80.0, 80.0, 0.0),
        )

        for _ in range(40):
            env.step(np.array([256.0, 145.0, 0.0, 1.0], dtype=np.float64))
            if env.agent.grasped:
                break
        assert env.agent.grasped

        max_overflow = 0.0
        for _ in range(160):
            env.step(np.array([256.0, 512.0, 0.0, 1.0], dtype=np.float64))
            overflow, _clearance = env._object_arena_metrics()
            max_overflow = max(max_overflow, overflow)
            assert env.agent.grasped

        # The whole pair is blocked while the T remains entirely in-bounds.
        assert max_overflow <= 1e-6
        assert env.agent_pos[1] < 420.0

        for _ in range(80):
            env.step(np.array([256.0, 100.0, 0.0, 0.0], dtype=np.float64))
            overflow, _clearance = env._object_arena_metrics()
            max_overflow = max(max_overflow, overflow)

        assert not env.agent.grasped
        assert env.agent_pos[1] < 150.0
        assert max_overflow <= 1e-6
    finally:
        env.close()


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
        assert obs["agent_pos"].dtype == np.float64
        assert obs["agent_angle"].shape == (1,)
        assert obs["agent_angle"].dtype == np.float64
        assert obs["object_pose"].shape == (3,)
        assert obs["object_pose"].dtype == np.float64
        assert obs["goal_pose"].shape == (3,)
        assert obs["goal_pose"].dtype == np.float64
        assert obs["image"].shape == (96, 96, 3)
        assert obs["image"].dtype == np.uint8
        assert "coverage" in info

        for _ in range(5):
            action = (
                np.array([256.0, 256.0, 0.0], dtype=np.float32)
                if pusher_shape == "u_socket"
                else np.array([256.0, 256.0], dtype=np.float32)
            )
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs["image"].shape == (96, 96, 3)
            assert obs["image"].dtype == np.uint8
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert 0.0 <= reward <= 1.0
    finally:
        env.close()


def test_u_socket_latches_aligned_t_stem_and_moves_as_one_body():
    assert U_SOCKET_INNER_GAP == 32.0
    assert U_SOCKET_PRONG_LENGTH == 30.0
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="u_socket",
        obstacle_level=0,
        image_size=96,
        seed=7,
    )
    try:
        env.reset(seed=7)
        # Socket local +X points right. A T at +pi/2 has its stem direction
        # pointing left, so the 30-wide stem seats inside the 36-wide opening.
        env._pusher_body.angle = 0.0
        pusher_x = 200.0
        # At +pi/2, the T stem bottom is 75 units left of object_pos.
        object_x = pusher_x + U_SOCKET_CROSSBAR_INNER_X + 75.0
        env.set_state(
            agent_pos=(pusher_x, 256.0),
            object_pose=(object_x, 256.0, np.pi / 2),
        )

        _, _, _, _, info = env.step(np.array([pusher_x, 256.0, 0.0]))
        assert info["socket_latched"] is True
        assert env.socket_latched is True

        for _ in range(5):
            env.step(np.array([300.0, 256.0, 0.0]))

        stem_bottom = env._object_body.local_to_world((0.0, 75.0))
        socket_contact = env._pusher_body.local_to_world(
            (U_SOCKET_CROSSBAR_INNER_X, 0.0)
        )
        assert stem_bottom.get_distance(socket_contact) < 0.5

        env.reset(seed=8)
        assert env.socket_latched is False
    finally:
        env.close()


@pytest.mark.parametrize(
    ("object_x", "should_latch"),
    [
        (250.0, True),  # top-bar end touches the inner crossbar face
        (120.0, False),  # top-bar end touches the outside/back face
    ],
)
def test_u_socket_latches_any_inner_face_but_not_crossbar_back(object_x, should_latch):
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="u_socket",
        obstacle_level=0,
        image_size=32,
        seed=9,
    )
    try:
        env.reset(seed=9)
        env.set_state(
            agent_pos=(200.0, 256.0),
            agent_angle=0.0,
            # At y=286 the 30-thick T top bar fits inside the 32-wide socket.
            object_pose=(object_x, 286.0, 0.0),
        )
        _, _, _, _, info = env.step(np.array([200.0, 256.0, 0.0]))
        assert info["socket_latched"] is should_latch
    finally:
        env.close()


def test_v3_u_socket_mouth_corner_is_frictionless():
    """A diagonal T touching a prong tip is outside, not in the pocket.

    This pose is reconstructed from the reported collector screenshot.  The
    old rectangular classifier saw the pusher-side point at ``(20, -16)`` and
    retained friction even though the T approached from outside the mouth.
    """
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="u_socket",
        obstacle_level=0,
        image_size=32,
        solid_pusher=True,
        socket_inside_friction_only=True,
    )
    env._skip_obs_render = True
    observed_friction = []
    original_callback = env.agent._socket_friction_pre_solve

    def record_friction(agent_env, arbiter, space, data):
        original_callback(agent_env, arbiter, space, data)
        observed_friction.append(float(arbiter.friction))

    env.agent._socket_friction_pre_solve = record_friction
    try:
        env.reset(seed=1)
        pusher_position = (131.42, 155.13)
        pusher_angle = np.deg2rad(-29.8)
        env.set_state(
            agent_pos=pusher_position,
            agent_angle=pusher_angle,
            object_pose=(156.60, 83.43, np.deg2rad(20.6)),
            goal_pose=(400.0, 400.0, 0.0),
        )
        _, _, _, _, info = env.step(
            np.array([*pusher_position, pusher_angle], dtype=np.float64)
        )

        assert observed_friction
        assert max(observed_friction) == 0.0
        assert info["socket_latched"] is False
    finally:
        env.close()


def test_v3_u_socket_inner_crossbar_keeps_friction():
    """The mouth-corner fix must preserve genuine pocket friction."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="u_socket",
        obstacle_level=0,
        image_size=32,
        solid_pusher=True,
        socket_inside_friction_only=True,
    )
    env._skip_obs_render = True
    observed_friction = []
    original_callback = env.agent._socket_friction_pre_solve

    def record_friction(agent_env, arbiter, space, data):
        original_callback(agent_env, arbiter, space, data)
        observed_friction.append(float(arbiter.friction))

    env.agent._socket_friction_pre_solve = record_friction
    try:
        env.reset(seed=9)
        env.set_state(
            agent_pos=(200.0, 256.0),
            agent_angle=0.0,
            object_pose=(250.0, 286.0, 0.0),
        )
        _, _, _, _, info = env.step(np.array([200.0, 256.0, 0.0]))

        assert observed_friction
        assert max(observed_friction) > 0.0
        assert info["socket_latched"] is True
    finally:
        env.close()


def test_v3_u_socket_friction_is_limited_to_pocket_bottom():
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="u_socket",
        obstacle_level=0,
        image_size=16,
        solid_pusher=True,
        socket_inside_friction_only=True,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=3)
        negative_prong, positive_prong, crossbar = env._pusher_shapes
        point = pymunk.Vec2d

        # Only the closed bottom of the U pocket retains friction.
        assert env.agent._socket_contact_is_on_inner_face(
            env, crossbar, point(-10.0, 0.0), point(-10.0, 0.0)
        )

        # Both inner side walls, tips, outer sides, the back, and ambiguous
        # corners are frictionless.
        outside_contacts = [
            (negative_prong, point(0.0, -16.0), point(0.0, -16.0)),
            (positive_prong, point(0.0, 16.0), point(0.0, 16.0)),
            (negative_prong, point(20.0, -16.0), point(20.0, -16.0)),
            (positive_prong, point(20.0, 16.0), point(20.0, 16.0)),
            (negative_prong, point(0.0, -26.0), point(0.0, -26.0)),
            (positive_prong, point(0.0, 26.0), point(0.0, 26.0)),
            (crossbar, point(-20.0, 0.0), point(-20.0, 0.0)),
            (crossbar, point(-10.0, 16.0), point(-10.0, 16.0)),
            (negative_prong, point(-10.0, -16.0), point(-10.0, -16.0)),
        ]
        assert all(
            not env.agent._socket_contact_is_on_inner_face(
                env, shape, pusher_pt, object_pt
            )
            for shape, pusher_pt, object_pt in outside_contacts
        )

        # V3 shapes themselves have no fallback friction; only the callback
        # can opt a genuine inner-face arbiter back in.
        assert all(float(shape.friction) == 0.0 for shape in env._pusher_shapes)
    finally:
        env.close()


def test_u_socket_angle_is_explicit_not_velocity_aligned():
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="u_socket",
        obstacle_level=0,
        image_size=32,
        seed=11,
    )
    try:
        env.reset(seed=11)
        env.set_state(agent_pos=(100.0, 100.0), agent_angle=0.0)

        # Moving vertically with a zero target angle must not auto-orient the
        # socket toward its velocity.
        env.step(np.array([100.0, 200.0, 0.0]))
        assert abs(env.pusher_angle) < 1e-6

        # It must also rotate in place when theta changes but XY does not.
        x, y = env.agent_pos
        for _ in range(5):
            env.step(np.array([x, y, np.pi / 2]))
        assert env.pusher_angle > 0.5
    finally:
        env.close()


def test_solid_u_socket_stays_latched_when_driven_into_obstacle():
    """A strong command into a wall must stop the pair, not break the weld."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="u_socket",
        obstacle_level=1,
        image_size=32,
        seed=0,
        solid_pusher=True,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=0)
        # Level 1 has a horizontal wall at y=256. Engage the socket above it,
        # then keep commanding the pair straight through the wall.
        env.set_state(
            object_pose=(160.0, 120.0, np.pi),
            agent_pos=(160.0, 20.0),
            agent_angle=np.pi / 2,
            goal_pose=(400.0, 450.0, 0.0),
        )
        for _ in range(400):
            env.step(np.array([160.0, 55.0, np.pi / 2], dtype=np.float64))
            if env.socket_latched:
                break
        assert env.socket_latched

        for _ in range(100):
            env.step(np.array([160.0, 500.0, np.pi / 2], dtype=np.float64))
            assert env.socket_latched

        # Before the fix the latch released after ~13 steps and the T center
        # crossed y=300. It must now remain on the near side of the wall.
        assert env.object_pose[1] < 256.0
        stem_bottom = env._object_body.local_to_world((0.0, 75.0))
        socket_contact = env._pusher_body.local_to_world(
            (U_SOCKET_CROSSBAR_INNER_X, 0.0)
        )
        assert stem_bottom.get_distance(socket_contact) < 0.5

        # Contact solvers can leave a tiny amount of overlap. Even if the pair
        # starts a step deeper than the normal allowance, an outward command
        # must reduce that overlap instead of being rolled back forever.
        for body in (env._pusher_body, env._object_body):
            body.position = body.position + (0.0, 1.0)
            env._space.reindex_shapes_for_body(body)
        wall_y = env.agent_pos[1]
        for _ in range(30):
            env.step(np.array([160.0, 0.0, np.pi / 2], dtype=np.float64))
            assert env.socket_latched
        assert wall_y - env.agent_pos[1] > 20.0
    finally:
        env.close()


@pytest.mark.parametrize(
    "pusher_shape",
    ["circle", "circle_small", "stick", "L", "u_socket"],
)
def test_solid_pusher_cannot_bulldoze_object_through_obstacle(pusher_shape):
    """Every fixed-v2 embodiment must keep the T outside a static wall."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape=pusher_shape,
        obstacle_level=0,
        image_size=32,
        seed=1,
        solid_pusher=True,
        solid_contact_guard=True,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=1)
        env.set_obstacles([((40.0, 256.0), (472.0, 256.0))])
        env.set_state(
            agent_pos=(256.0, 70.0),
            agent_angle=0.0,
            object_pose=(256.0, 170.0, 0.0),
            goal_pose=(400.0, 400.0, 0.0),
        )

        max_depth = env._object_static_penetration_depth()
        max_unlatched_depth = 0.0
        for _ in range(120):
            action = (
                np.array([256.0, 430.0, 0.0], dtype=np.float64)
                if pusher_shape == "u_socket"
                else np.array([256.0, 430.0], dtype=np.float64)
            )
            env.step(action)
            max_depth = max(max_depth, env._object_static_penetration_depth())
            if not env.socket_latched:
                max_unlatched_depth = max(
                    max_unlatched_depth,
                    env._pusher_object_penetration_depth(),
                )

        assert max_depth <= 0.2 + 1e-6
        assert max_unlatched_depth <= 0.5 + 1e-6
        assert env.object_pose[1] < 180.0
    finally:
        env.close()


@pytest.mark.parametrize("pusher_shape", ["circle", "circle_small", "stick", "L"])
def test_every_simple_pusher_uses_solid_contact_guard_by_default(pusher_shape):
    """Every non-socket embodiment must roll back a tunnelling substep."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape=pusher_shape,
        obstacle_level=0,
        image_size=32,
        seed=1,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=1)
        env.set_obstacles([])
        env.set_state(
            agent_pos=(120.0, 120.0),
            agent_angle=0.0,
            object_pose=(300.0, 300.0, 0.0),
            goal_pose=(400.0, 400.0, 0.0),
        )
        assert env.agent.solid_pusher
        assert env.agent.solid_contact_guard

        safe_agent_pose = (*env.agent_pos, env.pusher_angle)
        safe_object_pose = env.object_pose
        captured = env.agent.pre_substep(env)

        # Simulate one high-speed substep placing the pusher deeply inside the
        # object. The normal step loop invokes this same post-substep hook.
        env._pusher_body.position = env._object_body.position
        env._space.reindex_shapes_for_body(env._pusher_body)
        assert env._pusher_object_penetration_depth() > 0.5
        env.agent.post_substep(env, captured)

        assert (*env.agent_pos, env.pusher_angle) == pytest.approx(safe_agent_pose)
        assert env.object_pose == pytest.approx(safe_object_pose)
        assert env._pusher_object_penetration_depth() <= 0.5 + 1e-6
    finally:
        env.close()


@pytest.mark.parametrize(
    "pusher_shape",
    ["circle", "circle_small", "stick", "L", "u_socket"],
)
def test_solid_contact_guard_preserves_free_space_pushing(pusher_shape):
    """The anti-tunnelling guard must still allow ordinary solid pushing."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape=pusher_shape,
        obstacle_level=0,
        image_size=32,
        seed=1,
        solid_pusher=True,
        solid_contact_guard=True,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=1)
        env.set_obstacles([])
        env.set_state(
            agent_pos=(256.0, 90.0),
            agent_angle=0.0,
            object_pose=(256.0, 230.0, 0.0),
            goal_pose=(400.0, 400.0, 0.0),
        )
        initial_y = env.object_pose[1]
        max_unlatched_depth = 0.0
        for _ in range(90):
            action = (
                np.array([256.0, 430.0, 0.0], dtype=np.float64)
                if pusher_shape == "u_socket"
                else np.array([256.0, 430.0], dtype=np.float64)
            )
            env.step(action)
            if not env.socket_latched:
                max_unlatched_depth = max(
                    max_unlatched_depth,
                    env._pusher_object_penetration_depth(),
                )

        assert env.object_pose[1] - initial_y > 100.0
        assert max_unlatched_depth <= 0.5 + 1e-6
    finally:
        env.close()


def test_solid_u_socket_can_pull_away_while_rotation_is_blocked():
    """Unsafe rotation must not cancel safe translation away from a wall."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="u_socket",
        obstacle_level=2,
        image_size=32,
        seed=0,
        solid_pusher=True,
    )
    env._skip_obs_render = True
    try:
        socket_angle = np.pi / 2
        env.reset(seed=0)
        env.set_state(
            agent_pos=(280.0, 90.0),
            agent_angle=socket_angle,
            object_pose=(280.0, 155.0, np.pi),
            goal_pose=(100.0, 400.0, 0.0),
        )
        for _ in range(20):
            env.step(np.array([280.0, 90.0, socket_angle]))
            if env.socket_latched:
                break
        assert env.socket_latched

        for _ in range(100):
            env.step(np.array([400.0, 450.0, socket_angle]))
        wall_position = np.asarray(env.agent_pos)

        # Pull upward while requesting a rotation that initially presses the
        # T into the wall. Previously the whole movement was rolled back.
        for _ in range(60):
            env.step(np.array([280.0, 40.0, socket_angle + 0.5]))
            assert env.socket_latched
        pull_distance = float(np.linalg.norm(np.asarray(env.agent_pos) - wall_position))
        assert pull_distance > 20.0
    finally:
        env.close()


@pytest.mark.parametrize("object_angle", [0.0, -np.pi / 2, np.pi])
def test_solid_u_socket_cannot_push_unlatched_t_outside_arena(object_angle):
    """A solid socket must not bulldoze an unlatched T through an edge wall."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="u_socket",
        obstacle_level=0,
        image_size=32,
        seed=0,
        solid_pusher=True,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=0)
        env.set_state(
            agent_pos=(320.0, 256.0),
            agent_angle=0.0,
            object_pose=(430.0, 256.0, object_angle),
            goal_pose=(100.0, 100.0, 0.0),
        )

        for _ in range(120):
            env.step(np.array([512.0, 256.0, 0.0], dtype=np.float64))
            xmin, ymin, xmax, ymax = env._build_object_polygon(
                tuple(env.object_pose[:2]), float(env.object_pose[2])
            ).bounds
            assert xmin >= -1e-6
            assert ymin >= -1e-6
            assert xmax <= env.WORLD_SIZE + 1e-6
            assert ymax <= env.WORLD_SIZE + 1e-6
            assert env._pusher_object_penetration_depth() <= 0.5 + 1e-6
    finally:
        env.close()


def test_solid_object_arena_containment_is_noop_in_free_space():
    """The edge guard must not perturb ordinary in-arena motion."""
    env = PushShapesEnv(
        object_shape="T",
        pusher_shape="u_socket",
        obstacle_level=0,
        image_size=32,
        seed=0,
        solid_pusher=True,
    )
    env._skip_obs_render = True
    try:
        env.reset(seed=0)
        env.set_state(object_pose=(256.0, 256.0, 0.4))
        env._object_body.velocity = (12.0, -7.0)
        before_pose = np.asarray(env.object_pose, dtype=np.float64)
        before_velocity = np.asarray(env._object_body.velocity, dtype=np.float64)

        previous_pose = env.agent._capture_solid_unlatched_edge_pose(env)
        env.agent._guard_solid_unlatched_object_at_arena_edge(env, previous_pose)

        np.testing.assert_array_equal(
            np.asarray(env.object_pose, dtype=np.float64), before_pose
        )
        np.testing.assert_array_equal(
            np.asarray(env._object_body.velocity, dtype=np.float64), before_velocity
        )
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


def test_writer_preserves_direct_variant_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        env_args = {"object_shape": "T", "pusher_shape": "circle", "obstacle_level": 0}
        writer = ZarrDemoWriter(
            path=tmp,
            env_args=env_args,
            image_size=8,
            metadata_override={
                "speed_factor": 0.5,
                "pusher_color": "blue",
                "embodiment_variant": "blue_circle",
            },
        )
        writer.start_episode(init_state={"agent_pos": [10.0, 20.0]})
        _add_fake_step(writer, np.random.default_rng(10))
        idx = writer.commit_episode()
        writer.close()

        store = zarr.open_group(
            os.path.join(tmp, _episode_filename(env_args, idx)), mode="r"
        )
        attrs = dict(store.attrs)
        assert attrs["speed_factor"] == 0.5
        assert attrs["pusher_color"] == "blue"
        assert attrs["embodiment_variant"] == "blue_circle"
        assert json.loads(attrs["episode_init"])["agent_pos"] == [10.0, 20.0]


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
        assert w2.next_episode_index == 1, (
            "writer should resume at idx 1 when an episode_*_000000.zarr already exists"
        )
        w2.close()


def test_writer_existing_episode_count_filters_exact_family():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(2)

        def _write_one(env_args: dict) -> None:
            writer = ZarrDemoWriter(path=tmp, env_args=env_args, image_size=8)
            writer.start_episode()
            _add_fake_step(writer, rng, reward=0.25)
            writer.commit_episode()
            writer.close()

        _write_one({"object_shape": "T", "pusher_shape": "circle", "obstacle_level": 0})
        _write_one({"object_shape": "T", "pusher_shape": "circle", "obstacle_level": 1})
        _write_one(
            {"object_shape": "T", "pusher_shape": "circle_small", "obstacle_level": 0}
        )

        circle_obs0 = ZarrDemoWriter(
            path=tmp,
            env_args={
                "object_shape": "T",
                "pusher_shape": "circle",
                "obstacle_level": 0,
            },
            image_size=8,
        )
        circle_obs1 = ZarrDemoWriter(
            path=tmp,
            env_args={
                "object_shape": "T",
                "pusher_shape": "circle",
                "obstacle_level": 1,
            },
            image_size=8,
        )
        small_obs0 = ZarrDemoWriter(
            path=tmp,
            env_args={
                "object_shape": "T",
                "pusher_shape": "circle_small",
                "obstacle_level": 0,
            },
            image_size=8,
        )
        circle_obs2 = ZarrDemoWriter(
            path=tmp,
            env_args={
                "object_shape": "T",
                "pusher_shape": "circle",
                "obstacle_level": 2,
            },
            image_size=8,
        )

        assert circle_obs0.existing_episode_count() == 1
        assert circle_obs1.existing_episode_count() == 1
        assert small_obs0.existing_episode_count() == 1
        assert circle_obs2.existing_episode_count() == 0


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
    get_keymap_hpt = pytest.importorskip(
        "egomimic.rldb.embodiment.pushshapes"
    ).get_keymap_hpt

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
        dataset = ZarrDataset(ep_path, key_map=get_keymap_hpt(action_horizon=32))
        sample = dataset[0]

        # The current Pipeline keymap uses one observation and a 32-action
        # future chunk. Images are decoded to channel-first layout (C, H, W).
        assert "front_img_1" in sample
        assert "state_agent_obj" in sample
        assert "actions" in sample
        img = sample["front_img_1"]
        assert img.shape == (3, 8, 8)
        assert sample["state_agent_obj"].shape == (5,)
        # action_horizon=32 set in get_keymap_hpt -> loader returns (32, 2).
        assert sample["actions"].shape == (32, 2)
