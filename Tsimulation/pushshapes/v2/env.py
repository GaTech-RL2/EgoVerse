"""PushShapesEnv — gym-pusht adapted with multiple shapes, pushers, obstacles.

A 512x512 top-down arena where a kinematic pusher shoves
a single rigid body (T/U/Z) toward a goal pose. Reward = IoU between the
object polygon and the goal polygon. Episodes terminate when IoU clears
SUCCESS_THRESHOLD; truncation is disabled (the loop runs indefinitely
until terminated, the caller stops it, or set_state() resets things).
"""

from __future__ import annotations

# FROZEN v2 snapshot -- sim_v2_backup_pre_v3_20260806_011326.
# Do NOT edit: reproduces solid-pusher-era physics exactly.
# New work goes in v3/.

import hashlib
import json
import math
import os
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
import pymunk
from gymnasium import spaces
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .obstacles import (
    OBSTACLE_LEVELS,
    WALL_RADIUS,
    Segment,
    build_obstacle_segments,
    build_obstacles,
)
from .render import (
    draw_arena,
    surface_to_rgb_array,
    to_image_obs,
)
from .shapes import (
    SHAPES,
    U_SOCKET_CROSSBAR_INNER_X,
    make_object,
    make_pusher,
    pusher_radius,
)

_VALID_PUSHERS = ("circle", "circle_small", "stick", "L", "u_socket")
# non-symmetric pushers that re-orient toward velocity. The L stays fixed at
# its spawn angle instead — pushing with a rigid axis-aligned tool.
_ORIENTED_PUSHERS = ("stick",)

# Tunables not exposed via __init__ — surfaced here for visibility.
_MIN_TARGET_DIST = 1e-3  # below this, treat pusher as on-target
_MIN_STICK_TURN_DIST = 1.0  # oriented pushers only re-orient when moving meaningfully
_WALL_INSET = 5.0  # rejection clearance from arena edges
_PUSHER_OBJECT_MIN_DIST = 80.0  # pusher cannot spawn on top of object
_GOAL_OBJECT_MIN_DIST = 120.0  # goal pose must be visibly different from object
_SPAWN_MAX_TRIES = 50  # rejection-sampling budget per spawn
_SOCKET_LATCH_FACE_TOL = 2.0
# Maximum static penetration tolerated before rolling a latched pair back to
# its previous substep pose. This is twice pymunk's default collision slop:
# ordinary resting contact remains untouched, while a kinematic socket cannot
# drag its welded object through a wall.
_LATCH_STATIC_MAX_DEPTH = 0.2
# Preserve the original breakaway behavior when replaying datasets collected
# before solid-pusher physics existed.
_LEGACY_SOCKET_UNLATCH_DEPTH = 0.5
_SOCKET_RELATCH_BLOCK = 20
_LATCH_DEPTH_EPSILON = 1e-9
# NOTE: capping the socket weld's max_force does stop the latched object being
# dragged through walls (measured: <= 3e6 no longer tunnels), but it also lets
# the weld slip during ordinary dragging, which broke 256 of 263 sampled
# u_socket episodes (median coverage -0.583). Not worth it -- see the module
# docstring of scripts/verify_new_physics.py for how that was measured.


class PushShapesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    WORLD_SIZE: float = 512.0
    PUSHER_SPEED: float = 200.0
    DT: float = 1.0 / 30.0
    SUBSTEPS: int = 20
    DAMPING: float = 0
    STICK_TURN_RATE: float = 4.0  # rad/s — max kinematic rotation of stick pusher
    SUCCESS_THRESHOLD: float = 0.95
    SPAWN_MARGIN: float = 60.0

    def __init__(
        self,
        object_shape: str = "T",
        pusher_shape: str = "circle",
        obstacle_level: int = 0,
        render_mode: str | None = None,
        image_size: int = 96,
        seed: int | None = None,
        solid_pusher: bool = False,
    ):
        super().__init__()

        if object_shape not in SHAPES:
            raise ValueError(f"object_shape {object_shape!r} not in {list(SHAPES)}")
        if pusher_shape not in _VALID_PUSHERS:
            raise ValueError(f"pusher_shape {pusher_shape!r} not in {_VALID_PUSHERS}")
        if obstacle_level not in OBSTACLE_LEVELS:
            raise ValueError(
                f"obstacle_level {obstacle_level} not in {sorted(OBSTACLE_LEVELS)}"
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode {render_mode!r} not in {self.metadata['render_modes']}"
            )

        self.object_shape = object_shape
        self.pusher_shape = pusher_shape
        self.obstacle_level = obstacle_level
        # Off by default so every dataset recorded before this existed replays
        # bit-for-bit; new collection should turn it on.
        self.solid_pusher = bool(solid_pusher)
        self.render_mode = render_mode
        self.image_size = int(image_size)
        # Replay/coverage checks can disable image rendering while retaining
        # the same physics and numeric observations.
        self._skip_obs_render = False

        if self.pusher_shape == "u_socket":
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -math.pi], dtype=np.float64),
                high=np.array(
                    [self.WORLD_SIZE, self.WORLD_SIZE, math.pi], dtype=np.float64
                ),
                dtype=np.float64,
            )
        else:
            self.action_space = spaces.Box(
                low=0.0, high=float(self.WORLD_SIZE), shape=(2,), dtype=np.float64
            )
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    0.0, float(self.WORLD_SIZE), (2,), dtype=np.float64
                ),
                "agent_angle": spaces.Box(-np.inf, np.inf, (1,), dtype=np.float64),
                "object_pose": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
                "goal_pose": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64),
                "image": spaces.Box(
                    0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8
                ),
            }
        )

        self._np_random = np.random.default_rng(seed)
        self._world_surface: pygame.Surface | None = None
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._step_count = 0
        self._space: pymunk.Space | None = None
        self._object_body: pymunk.Body | None = None
        self._pusher_body: pymunk.Body | None = None
        self._object_shapes: list[pymunk.Shape] = []
        self._pusher_shapes: list[pymunk.Shape] = []
        self._socket_constraints: tuple[pymunk.Constraint, ...] | None = None
        self._socket_latch_local_object_pos: tuple[float, float] | None = None
        self._socket_latch_angle_offset: float | None = None
        self._socket_relatch_block = 0
        self._obstacle_segments: list[pymunk.Segment] = []
        self._goal_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._goal_polygon: Polygon | None = None

    # ------------------------------------------------------------------ #
    # Public accessors — prefer these over the underscored attrs.
    # ------------------------------------------------------------------ #

    @property
    def step_count(self) -> int:
        """Number of step() calls since the last reset()."""
        return self._step_count

    @property
    def agent_pos(self) -> tuple[float, float]:
        p = self._pusher_body.position
        return (float(p.x), float(p.y))

    @property
    def pusher_angle(self) -> float:
        return float(self._pusher_body.angle)

    @property
    def object_pose(self) -> tuple[float, float, float]:
        b = self._object_body
        return (float(b.position.x), float(b.position.y), float(b.angle))

    @property
    def goal_pose(self) -> tuple[float, float, float]:
        return self._goal_pose

    @property
    def socket_latched(self) -> bool:
        """Whether the T stem is rigidly latched to a ``u_socket`` pusher."""
        return self._socket_constraints is not None

    def get_episode_init(self) -> dict:
        """Capture full episode init state for deterministic replay."""
        obstacles = [
            [list(segment.a), list(segment.b)]
            for segment in self._obstacle_segments
        ]
        init = {
            "agent_pos": list(self.agent_pos),
            "agent_angle": self.pusher_angle,
            "object_pose": list(self.object_pose),
            "goal_pose": list(self.goal_pose),
            "object_shape": self.object_shape,
            "pusher_shape": self.pusher_shape,
            "obstacle_level": self.obstacle_level,
            "solid_pusher": self.solid_pusher,
            "obstacles": obstacles,
            "reset_seed": getattr(self, "_last_reset_seed", None),
        }
        init["config_hash"] = hashlib.sha256(
            json.dumps(init, sort_keys=True).encode()
        ).hexdigest()[:16]
        return init

    # ------------------------------------------------------------------ #
    # gym API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._last_reset_seed = seed
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Fresh space each episode — simpler than tearing down individual bodies.
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)
        self._space.damping = self.DAMPING
        self._socket_constraints = None
        self._socket_latch_local_object_pos = None
        self._socket_latch_angle_offset = None
        self._socket_relatch_block = 0

        self._build_boundary_walls()
        self._obstacle_segments = build_obstacles(self._space, self.obstacle_level)

        # Rejection-sample non-overlapping object / goal / pusher placements.
        obstacle_polys = self._obstacle_polygons()
        object_pos, object_angle = self._sample_object_pose(obstacle_polys)
        goal_pos, goal_angle = self._sample_object_pose(
            obstacle_polys, away_from=object_pos
        )
        pusher_pos = self._sample_pusher_pos(obstacle_polys, object_pos)

        self._object_body, self._object_shapes = make_object(
            self.object_shape, self._space, object_pos, object_angle
        )
        self._pusher_body, self._pusher_shapes = make_pusher(
            self.pusher_shape, self._space, pusher_pos
        )

        self._goal_pose = (float(goal_pos[0]), float(goal_pos[1]), float(goal_angle))
        self._goal_polygon = self._build_object_polygon(goal_pos, goal_angle)
        self._step_count = 0

        return self._get_obs(), {
            "coverage": float(self._coverage()),
            "socket_latched": self.socket_latched,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        expected_shape = (3,) if self.pusher_shape == "u_socket" else (2,)
        if action.shape != expected_shape:
            raise ValueError(
                f"action must be shape {expected_shape} for "
                f"pusher={self.pusher_shape!r}, got {action.shape}"
            )

        # Action = desired pusher XY in world coords. Walk toward it at
        # PUSHER_SPEED via kinematic velocity commands; pymunk's solver still
        # resolves contact forces against the object.
        tx = float(np.clip(action[0], 0.0, self.WORLD_SIZE))
        ty = float(np.clip(action[1], 0.0, self.WORLD_SIZE))
        target_angle = (
            (float(action[2]) + math.pi) % (2 * math.pi) - math.pi
            if self.pusher_shape == "u_socket"
            else None
        )

        dt_sub = self.DT / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            latched_pose = (
                self._capture_latched_pair_pose()
                if self.solid_pusher and self.socket_latched
                else None
            )
            self._drive_pusher_toward(tx, ty, dt_sub, target_angle)
            self._space.step(dt_sub)
            self._clamp_pusher_to_static()
            self._maybe_latch_socket()
            self._enforce_solid_socket_latch()
            self._guard_socket_penetration(latched_pose)

        # Zero pusher velocity (and angular velocity) between outer steps so
        # stale motion doesn't drift contacts when no new action comes in.
        self._pusher_body.velocity = (0.0, 0.0)
        self._pusher_body.angular_velocity = 0.0
        self._step_count += 1

        coverage = float(self._coverage())
        reward = float(np.clip(coverage, 0.0, 1.0))
        terminated = coverage >= self.SUCCESS_THRESHOLD
        truncated = False  # episode cutoff disabled — caller decides when to stop
        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {
                "coverage": coverage,
                "socket_latched": self.socket_latched,
            },
        )

    def set_state(
        self,
        agent_pos: tuple[float, float] | None = None,
        agent_angle: float | None = None,
        object_pose: tuple[float, float, float] | None = None,
        goal_pose: tuple[float, float, float] | None = None,
    ) -> None:
        """Override live env state after reset(). Any subset of args may be
        passed. Velocities are zeroed so the next step() starts at rest."""
        if self._space is None:
            raise RuntimeError("call reset() before set_state()")

        if (
            agent_pos is not None or agent_angle is not None or object_pose is not None
        ) and self.socket_latched:
            self._release_socket_latch()

        if agent_pos is not None:
            self._pusher_body.position = (float(agent_pos[0]), float(agent_pos[1]))
            self._pusher_body.velocity = (0.0, 0.0)

        if agent_angle is not None:
            self._pusher_body.angle = float(agent_angle)
            self._pusher_body.angular_velocity = 0.0

        if object_pose is not None:
            # Set angle BEFORE position: pymunk's body.position is the CoG
            # in world space.  When center_of_gravity is non-zero (T-shape),
            # setting body.angle after body.position rotates the CoG offset
            # and silently shifts body.position.  Angle-first avoids this.
            self._object_body.angle = float(object_pose[2])
            self._object_body.position = (float(object_pose[0]), float(object_pose[1]))
            self._object_body.velocity = (0.0, 0.0)
            self._object_body.angular_velocity = 0.0

        if goal_pose is not None:
            gx, gy, gt = float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2])
            self._goal_pose = (gx, gy, gt)
            self._goal_polygon = self._build_object_polygon((gx, gy), gt)

    def set_obstacles(self, obstacle_segments: list[Segment]) -> None:
        """Replace live obstacles with explicit recorded segment geometry.

        This is primarily a replay operation. It must be called after
        :meth:`reset` and before stepping the environment.
        """
        if self._space is None:
            raise RuntimeError("call reset() before set_obstacles()")
        if self.socket_latched:
            raise RuntimeError("cannot replace obstacles while the socket is latched")
        if self._obstacle_segments:
            self._space.remove(*self._obstacle_segments)
        normalized = [
            (
                (float(segment[0][0]), float(segment[0][1])),
                (float(segment[1][0]), float(segment[1][1])),
            )
            for segment in obstacle_segments
        ]
        self._obstacle_segments = build_obstacle_segments(self._space, normalized)

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        surface = self._render_world()
        if self.render_mode == "rgb_array":
            return surface_to_rgb_array(surface)

        # human mode: lazily open a pygame window on first render call.
        if self._screen is None:
            pygame.init()
            pygame.display.init()
            self._screen = pygame.display.set_mode(
                (int(self.WORLD_SIZE), int(self.WORLD_SIZE))
            )
            pygame.display.set_caption(
                f"PushShapes [{self.object_shape}/{self.pusher_shape}/obs={self.obstacle_level}]"
            )
            self._clock = pygame.time.Clock()
        self._screen.blit(surface, (0, 0))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
        return None

    def world_surface(self) -> pygame.Surface:
        """512x512 world surface, for callers compositing their own window
        (e.g. mouse_collect overlays a stats panel)."""
        return self._render_world()

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            self._screen = None
            self._clock = None
        self._world_surface = None
        self._space = None
        self._object_body = None
        self._pusher_body = None
        self._object_shapes = []
        self._pusher_shapes = []
        self._socket_constraints = None
        self._socket_latch_local_object_pos = None
        self._socket_latch_angle_offset = None
        self._obstacle_segments = []
        self._goal_polygon = None

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _maybe_latch_socket(self) -> None:
        """Rigidly attach any T face touching the socket's inner crossbar.

        Contact on the crossbar's outer/back face is deliberately ignored.
        A pivot plus a 1:1 gear joint acts as a planar weld while still letting
        the dynamic T collide with walls and obstacles.
        """
        if (
            self.socket_latched
            or self.pusher_shape != "u_socket"
            or self.object_shape != "T"
        ):
            return
        if self._socket_relatch_block > 0:
            self._socket_relatch_block -= 1
            return

        pusher = self._pusher_body
        obj = self._object_body
        crossbar = self._pusher_shapes[-1]
        latch_points: list[pymunk.Vec2d] = []

        c, s = math.cos(float(pusher.angle)), math.sin(float(pusher.angle))
        for query in self._space.shape_query(crossbar):
            if query.shape.body is not obj:
                continue
            normal = query.contact_point_set.normal
            normal_local_x = c * float(normal.x) + s * float(normal.y)
            if normal_local_x <= 0.5:
                continue
            for point in query.contact_point_set.points:
                contact_local = pusher.world_to_local(point.point_a)
                if (
                    abs(float(contact_local.x) - U_SOCKET_CROSSBAR_INNER_X)
                    <= _SOCKET_LATCH_FACE_TOL
                ):
                    latch_points.append(point.point_a)

        if not latch_points:
            return

        contact_world = sum(latch_points, pymunk.Vec2d(0.0, 0.0)) / len(latch_points)
        pivot = pymunk.PivotJoint(pusher, obj, tuple(contact_world))
        gear = pymunk.GearJoint(
            pusher,
            obj,
            phase=float(obj.angle) - float(pusher.angle),
            ratio=1.0,
        )
        # Keep the socket and the T colliding while welded. With collisions
        # off, any slip in the weld let the T sink bodily into the prongs
        # (measured up to 306 units^2 of overlap on real episodes) because
        # nothing was left to push it back out.
        pivot.collide_bodies = True
        gear.collide_bodies = True
        # Correct positional drift in the weld immediately instead of bleeding
        # it off over ~a second, which is what allowed the slip to accumulate.
        pivot.error_bias = 0.0
        gear.error_bias = 0.0
        self._space.add(pivot, gear)
        self._socket_constraints = (pivot, gear)
        local_object_pos = pusher.world_to_local(obj.position)
        self._socket_latch_local_object_pos = (
            float(local_object_pos.x),
            float(local_object_pos.y),
        )
        self._socket_latch_angle_offset = float(obj.angle) - float(pusher.angle)

    def _enforce_solid_socket_latch(self) -> None:
        """Keep a solid-physics latch at its exact captured relative pose.

        Pymunk constraints alone can stretch when a kinematic body drives a
        welded dynamic body against an immovable wall. For new solid-pusher
        collection, make the intended rigid attachment explicit after every
        substep. The penetration guard can then stop the complete rigid pair.
        """
        if (
            not self.solid_pusher
            or not self.socket_latched
            or self._socket_latch_local_object_pos is None
            or self._socket_latch_angle_offset is None
        ):
            return

        pusher = self._pusher_body
        obj = self._object_body
        object_position = pusher.local_to_world(self._socket_latch_local_object_pos)
        obj.angle = float(pusher.angle) + self._socket_latch_angle_offset
        obj.position = object_position

        # Match the instantaneous rigid-body velocity at the object's center.
        offset = object_position - pusher.position
        omega = float(pusher.angular_velocity)
        obj.velocity = (
            float(pusher.velocity.x) - omega * float(offset.y),
            float(pusher.velocity.y) + omega * float(offset.x),
        )
        obj.angular_velocity = omega
        self._space.reindex_shapes_for_body(obj)

    def _clamp_pusher_to_static(self) -> None:
        """Keep the kinematic pusher out of walls and obstacles.

        The pusher has infinite mass and no contact response, so nothing stops
        it driving through static geometry -- and once it is inside a wall it
        bulldozes the object through too. Rather than make it dynamic (which
        would change how it moves in open space, invalidating every recorded
        episode), leave it kinematic and simply push it back out whenever it
        ends a substep overlapping something static, cancelling the velocity
        component heading into the surface.

        The important property is that this is a no-op unless the pusher is
        actually touching static geometry, so free-space motion -- and hence
        replay of episodes that never touch a wall -- is unchanged.

        Note ``shape_query`` only returns genuine overlaps, but its sign
        convention is not consistent across shape types in this pymunk build:
        Poly-vs-Segment reports penetration positive, Circle-vs-Segment
        reports it negative. Use the magnitude, not the sign.
        """
        if not self.solid_pusher:
            return
        body = self._pusher_body
        for shape in self._pusher_shapes:
            for query in self._space.shape_query(shape):
                if query.shape.body.body_type != pymunk.Body.STATIC:
                    continue
                points = query.contact_point_set.points
                if not points:
                    continue
                depth = max(abs(float(p.distance)) for p in points)
                if depth <= 0.0:
                    continue
                n = query.contact_point_set.normal
                correction = -n * depth
                body.position = body.position + correction
                if self.socket_latched:
                    # Preserve the welded relative pose when the socket itself
                    # is projected out of static geometry.
                    self._object_body.position = self._object_body.position + correction
                v = body.velocity
                into = v.dot(n)
                if into > 0.0:
                    body.velocity = v - n * into

    def _capture_latched_pair_pose(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        float,
    ]:
        """Capture both poses and their static penetration before a substep."""
        pusher = self._pusher_body
        obj = self._object_body
        return (
            (float(pusher.position.x), float(pusher.position.y), float(pusher.angle)),
            (float(obj.position.x), float(obj.position.y), float(obj.angle)),
            self._object_static_penetration_depth(),
        )

    def _shapes_static_penetration_depth(
        self,
        body: pymunk.Body,
        shapes: list[pymunk.Shape],
    ) -> float:
        """Maximum current overlap depth between body shapes and statics."""
        self._space.reindex_shapes_for_body(body)
        depths: list[float] = []
        for shape in shapes:
            for query in self._space.shape_query(shape):
                if query.shape.body.body_type != pymunk.Body.STATIC:
                    continue
                points = query.contact_point_set.points
                if points:
                    depths.append(max(abs(float(point.distance)) for point in points))
        return max(depths, default=0.0)

    def _object_static_penetration_depth(self) -> float:
        return self._shapes_static_penetration_depth(
            self._object_body,
            self._object_shapes,
        )

    def _latched_pair_static_penetration_depth(self) -> float:
        return max(
            self._object_static_penetration_depth(),
            self._shapes_static_penetration_depth(
                self._pusher_body,
                self._pusher_shapes,
            ),
        )

    def _set_solid_latched_pair_pose(
        self,
        position: tuple[float, float],
        angle: float,
    ) -> None:
        """Set the socket pose and restore the captured rigid attachment."""
        self._pusher_body.position = position
        self._pusher_body.angle = angle
        self._space.reindex_shapes_for_body(self._pusher_body)
        self._enforce_solid_socket_latch()

    def _guard_socket_penetration(
        self,
        previous_pose: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            float,
        ]
        | None,
    ) -> None:
        """Keep a welded socket/object pair out of static geometry.

        A kinematic pusher has effectively infinite mass, so pymunk contact
        impulses cannot stop its weld from dragging the dynamic T through a
        wall. If a substep creates meaningful static penetration, restore both
        bodies to their last valid poses and stop their motion. The weld stays
        intact: pushing harder cannot tear the T out of the socket.
        """
        if self._socket_constraints is None:
            return

        penetration_depths: list[float] = []
        if previous_pose is None:
            # Legacy replay uses live arbiters to preserve the old breakaway
            # path exactly.
            def _measure(arbiter: pymunk.Arbiter, depths: list[float]) -> None:
                if not any(
                    shape.body.body_type == pymunk.Body.STATIC
                    for shape in arbiter.shapes
                ):
                    return
                for point in arbiter.contact_point_set.points:
                    depths.append(max(0.0, -float(point.distance)))

            self._object_body.each_arbiter(_measure, penetration_depths)
        else:
            # Solid-physics object poses are explicitly locked after the space
            # step, so query their current geometry rather than stale arbiters.
            # object-only (2026-08-05): freeze ONLY when the *inserted T* is
            # driven into static geometry. The socket's own body touching an
            # obstacle from OUTSIDE is handled softly by _clamp_pusher_to_static
            # (slide out), so a socketed pair no longer sticks against a wall
            # unless the T itself would tunnel through it.
            penetration_depths.append(self._object_static_penetration_depth())
        if not penetration_depths:
            return
        max_depth = max(penetration_depths)

        if previous_pose is None:
            if not self.solid_pusher and max_depth > _LEGACY_SOCKET_UNLATCH_DEPTH:
                self._release_socket_latch()
                self._socket_relatch_block = _SOCKET_RELATCH_BLOCK
            return
        pusher_pose, object_pose, previous_depth = previous_pose
        allowed_depth = max(_LATCH_STATIC_MAX_DEPTH, previous_depth)
        if max_depth <= allowed_depth + _LATCH_DEPTH_EPSILON:
            # Crucially, a pair that begins slightly embedded may move outward.
            # Only increasing penetration is blocked; otherwise it can become
            # permanently trapped at the wall by its own safety guard.
            return

        candidate_position = (
            float(self._pusher_body.position.x),
            float(self._pusher_body.position.y),
        )
        candidate_angle = float(self._pusher_body.angle)
        candidate_velocity = (
            float(self._pusher_body.velocity.x),
            float(self._pusher_body.velocity.y),
        )
        candidate_angular_velocity = float(self._pusher_body.angular_velocity)

        # If only rotation presses into the wall, preserve the safe outward
        # translation and temporarily block angular motion.
        self._pusher_body.velocity = candidate_velocity
        self._pusher_body.angular_velocity = 0.0
        self._set_solid_latched_pair_pose(candidate_position, pusher_pose[2])
        if (
            self._object_static_penetration_depth()
            <= allowed_depth + _LATCH_DEPTH_EPSILON
        ):
            return

        # Conversely, allow a safe rotation in place when translation is the
        # component trying to move deeper into static geometry.
        self._pusher_body.velocity = (0.0, 0.0)
        self._pusher_body.angular_velocity = candidate_angular_velocity
        self._set_solid_latched_pair_pose(pusher_pose[:2], candidate_angle)
        if (
            self._object_static_penetration_depth()
            <= allowed_depth + _LATCH_DEPTH_EPSILON
        ):
            return

        self._pusher_body.position = pusher_pose[:2]
        self._pusher_body.angle = pusher_pose[2]
        self._pusher_body.velocity = (0.0, 0.0)
        self._pusher_body.angular_velocity = 0.0

        # Angle first for the T: its non-zero center of gravity means changing
        # angle after position would shift the body's world-space position.
        self._object_body.angle = object_pose[2]
        self._object_body.position = object_pose[:2]
        self._object_body.velocity = (0.0, 0.0)
        self._object_body.angular_velocity = 0.0

        self._space.reindex_shapes_for_body(self._pusher_body)
        self._space.reindex_shapes_for_body(self._object_body)

    def _release_socket_latch(self) -> None:
        """Remove the current socket weld, if any."""
        if self._socket_constraints is None:
            return
        if self._space is not None:
            self._space.remove(*self._socket_constraints)
        self._socket_constraints = None
        self._socket_latch_local_object_pos = None
        self._socket_latch_angle_offset = None

    def _drive_pusher_toward(
        self,
        tx: float,
        ty: float,
        dt_sub: float,
        target_angle: float | None = None,
    ) -> None:
        """Single-substep velocity command toward (tx, ty) in world coords.

        For the stick pusher we set `angular_velocity` (rather than snapping
        `angle`) so contact impulses on the object stay smooth — the
        kinematic body integrates the spin over the substep.
        """
        body = self._pusher_body
        pos = body.position
        dx, dy = tx - pos.x, ty - pos.y
        dist = math.hypot(dx, dy)
        if dist < _MIN_TARGET_DIST:
            body.velocity = (0.0, 0.0)
            ux = uy = 0.0
        else:
            # Cap by PUSHER_SPEED but also by distance-remaining-per-substep so
            # we don't overshoot the target inside a single substep.
            speed = min(self.PUSHER_SPEED, dist / dt_sub)
            ux, uy = dx / dist, dy / dist
            body.velocity = (ux * speed, uy * speed)

        desired: float | None = None
        if self.pusher_shape == "u_socket":
            desired = target_angle
        elif self.pusher_shape in _ORIENTED_PUSHERS and dist > _MIN_STICK_TURN_DIST:
            # Shortest signed angle diff to the velocity direction, capped at
            # STICK_TURN_RATE so oriented pushers ease into a new heading
            # rather than rotating instantly through contacts.
            desired = math.atan2(uy, ux)

        if desired is not None:
            diff = (desired - float(body.angle) + math.pi) % (2 * math.pi) - math.pi
            body.angular_velocity = max(
                -self.STICK_TURN_RATE, min(self.STICK_TURN_RATE, diff / dt_sub)
            )
        else:
            body.angular_velocity = 0.0

    def _build_boundary_walls(self) -> None:
        """Four static segments enclosing the 512x512 arena."""
        s = self.WORLD_SIZE
        corners = [(0, 0), (s, 0), (s, s), (0, s), (0, 0)]
        walls = []
        for a, b in zip(corners, corners[1:]):
            seg = pymunk.Segment(self._space.static_body, a, b, 1.0)
            seg.friction = 0.7
            walls.append(seg)
        self._space.add(*walls)

    def _get_obs(self) -> dict[str, np.ndarray]:
        pos = self._pusher_body.position
        obj = self._object_body
        image = (
            np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            if self._skip_obs_render
            else to_image_obs(self._render_world(), self.image_size)
        )
        return {
            "agent_pos": np.array([pos.x, pos.y], dtype=np.float64),
            "agent_angle": np.array([self._pusher_body.angle], dtype=np.float64),
            "object_pose": np.array(
                [obj.position.x, obj.position.y, obj.angle], dtype=np.float64
            ),
            "goal_pose": np.array(self._goal_pose, dtype=np.float64),
            "image": image,
        }

    def _ensure_pygame_initialized(self) -> None:
        # Headless callers (tests, training) get the dummy SDL driver so
        # pygame.init() doesn't try to open a display.
        if not pygame.get_init():
            if self.render_mode != "human" and "SDL_VIDEODRIVER" not in os.environ:
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()

    def _render_world(self) -> pygame.Surface:
        self._ensure_pygame_initialized()
        if self._world_surface is None:
            self._world_surface = pygame.Surface(
                (int(self.WORLD_SIZE), int(self.WORLD_SIZE))
            )
        obj = self._object_body
        pusher = self._pusher_body
        draw_arena(
            self._world_surface,
            object_shape=self.object_shape,
            object_pose=(obj.position.x, obj.position.y, obj.angle),
            goal_pose=self._goal_pose,
            pusher_shape=self.pusher_shape,
            pusher_pos=(pusher.position.x, pusher.position.y),
            pusher_angle=pusher.angle,
            obstacle_segments=self._obstacle_segments,
        )
        return self._world_surface

    def _obstacle_polygons(self) -> list[Polygon]:
        # Buffer each segment by (radius + clearance) so rejection sampling
        # leaves breathing room around walls.
        return [
            LineString([(seg.a.x, seg.a.y), (seg.b.x, seg.b.y)]).buffer(
                WALL_RADIUS + 10.0
            )
            for seg in self._obstacle_segments
        ]

    def _build_object_polygon(
        self, position: tuple[float, float], angle: float
    ) -> Polygon:
        """Union of the object's component rects rotated and translated into
        world coords. Used for IoU coverage and spawn collision tests."""
        bx, by = position
        c, s = math.cos(angle), math.sin(angle)
        polys = []
        for cx, cy, w, h in SHAPES[self.object_shape]:
            hw, hh = w / 2.0, h / 2.0
            local = [
                (cx - hw, cy - hh),
                (cx + hw, cy - hh),
                (cx + hw, cy + hh),
                (cx - hw, cy + hh),
            ]
            world = [(bx + c * lx - s * ly, by + s * lx + c * ly) for lx, ly in local]
            polys.append(Polygon(world))
        return unary_union(polys)

    def _coverage(self) -> float:
        """IoU of current object polygon against the goal polygon. 1.0 = perfect."""
        if self._goal_polygon is None or self._goal_polygon.area <= 0:
            return 0.0
        current = self._build_object_polygon(
            (self._object_body.position.x, self._object_body.position.y),
            self._object_body.angle,
        )
        return float(
            current.intersection(self._goal_polygon).area / self._goal_polygon.area
        )

    def _sample_object_pose(
        self,
        obstacle_polys: list[Polygon],
        away_from: tuple[float, float] | None = None,
    ) -> tuple[tuple[float, float], float]:
        """Rejection-sample a pose that fits the arena, clears obstacles,
        and (if `away_from` is set) keeps a minimum distance from it."""
        m = self.SPAWN_MARGIN
        for _ in range(_SPAWN_MAX_TRIES):
            x = float(self._np_random.uniform(m, self.WORLD_SIZE - m))
            y = float(self._np_random.uniform(m, self.WORLD_SIZE - m))
            th = float(self._np_random.uniform(-math.pi, math.pi))
            poly = self._build_object_polygon((x, y), th)
            xmin, ymin, xmax, ymax = poly.bounds
            if xmin < _WALL_INSET or ymin < _WALL_INSET:
                continue
            if (
                xmax > self.WORLD_SIZE - _WALL_INSET
                or ymax > self.WORLD_SIZE - _WALL_INSET
            ):
                continue
            if any(poly.intersects(op) for op in obstacle_polys):
                continue
            if (
                away_from is not None
                and math.hypot(x - away_from[0], y - away_from[1])
                < _GOAL_OBJECT_MIN_DIST
            ):
                continue
            return (x, y), th
        # Densely packed level: fall back to centered rest pose rather than crashing.
        return (self.WORLD_SIZE / 2.0, self.WORLD_SIZE / 2.0), 0.0

    def _sample_pusher_pos(
        self,
        obstacle_polys: list[Polygon],
        object_pos: tuple[float, float],
    ) -> tuple[float, float]:
        m = self.SPAWN_MARGIN
        radius = pusher_radius(self.pusher_shape) + 5.0
        for _ in range(_SPAWN_MAX_TRIES):
            x = float(self._np_random.uniform(m, self.WORLD_SIZE - m))
            y = float(self._np_random.uniform(m, self.WORLD_SIZE - m))
            disk = Point(x, y).buffer(radius)
            if any(disk.intersects(op) for op in obstacle_polys):
                continue
            if (
                math.hypot(x - object_pos[0], y - object_pos[1])
                < _PUSHER_OBJECT_MIN_DIST
            ):
                continue
            return (x, y)
        return (self.WORLD_SIZE / 4.0, self.WORLD_SIZE / 4.0)
