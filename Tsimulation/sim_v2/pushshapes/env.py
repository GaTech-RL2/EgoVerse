"""PushShapesEnv — gym-pusht adapted with multiple shapes, pushers, obstacles.

A 512x512 top-down arena where a kinematic pusher shoves
a single rigid body (T/U/Z) toward a goal pose. Reward = IoU between the
object polygon and the goal polygon. Episodes terminate when IoU clears
SUCCESS_THRESHOLD; truncation is disabled (the loop runs indefinitely
until terminated, the caller stops it, or set_state() resets things).
"""

from __future__ import annotations

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

from .agents import NEW_AGENTS as _NEW_AGENTS
from .agents import make_agent
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
    make_object,
    pusher_radius,
)

# Single source of truth: the agent registry. Adding an agent to
# agents.make_agent is enough to make it constructible here.
from .agents import VALID_PUSHERS as _VALID_PUSHERS  # noqa: E402
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
SIM_VERSION = 2  # v2 = v2 geometry + pocket-bottom-only socket friction (the FIX;
                 # the all-faces-grip intermediate was a bug, never a release)
# Slack on the pocket test, so solver jitter at the mouth does not flicker a
# genuine inside contact to frictionless for a substep.
# A contact exactly on a prong tip is at the open mouth, not inside the
# pocket.  Keep a small inward margin so diagonal outside contacts cannot use
# the shared tip/inner-face corner to retain friction.
# Corners belong to an inner face and an outer/tip face at the same time. Keep
# them frictionless so only an unambiguous inner-face contact can grip.
_CT_PUSHER = 1
_CT_OBJECT = 2
# Maximum static penetration tolerated before rolling a latched pair back to
# its previous substep pose. This is twice pymunk's default collision slop:
# ordinary resting contact remains untouched, while a kinematic socket cannot
# drag its welded object through a wall.
# Maximum penetration allowed by the compatibility-gated solid-contact guard.
# The static limit is twice pymunk's collision slop. The pusher/object limit
# permits normal pushing contact while rejecting a kinematic pusher tunnelling
# through an unlatched object.
# Preserve the original breakaway behavior when replaying datasets collected
# before solid-pusher physics existed.
# NOTE: capping the socket weld's max_force does stop the latched object being
# dragged through walls (measured: <= 3e6 no longer tunnels), but it also lets
# the weld slip during ordinary dragging, which broke 256 of 263 sampled
# u_socket episodes (median coverage -0.583). Not worth it -- see the module
# docstring of scripts/verify_new_physics.py for how that was measured.


class PushShapesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    WORLD_SIZE: float = 512.0
    PUSHER_SPEED: float = 200.0
    #: Control-fidelity preset (agents.CONTROL_GAPS). "ideal" = perfect
    #: tracking, which is what every pre-existing dataset was collected under.
    CONTROL_GAP: str = "ideal"
    DT: float = 1.0 / 30.0
    SUBSTEPS: int = 20
    DAMPING: float = 0
    STICK_TURN_RATE: float = 4.0  # rad/s — max kinematic rotation of stick pusher
    SUCCESS_THRESHOLD: float = 0.95
    SPAWN_MARGIN: float = 60.0
    # Sim V2 has exactly one collision model. These are properties below so
    # callers can inspect them but cannot mutate them per episode.
    SOLID_PUSHER: bool = True
    SOCKET_INSIDE_FRICTION_ONLY: bool = True
    SOLID_CONTACT_GUARD: bool = True

    def __init__(
        self,
        object_shape: str = "T",
        pusher_shape: str = "circle",
        obstacle_level: int = 0,
        render_mode: str | None = None,
        image_size: int = 96,
        seed: int | None = None,
        **legacy_physics_options: Any,
    ):
        super().__init__()

        # Accept old call sites without letting their flags alter Sim V2.
        # Keeping these out of the public signature prevents new code from
        # treating fixed collision physics as an episode-level option.
        known_legacy = {
            "solid_pusher",
            "socket_inside_friction_only",
            "solid_contact_guard",
        }
        unknown = set(legacy_physics_options) - known_legacy
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unexpected PushShapesEnv option(s): {names}")

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
        # The agent owns its action space, body and contact model, so the env
        # never branches on pusher_shape again.
        agent_options = {
            "solid_pusher": self.SOLID_PUSHER,
            "solid_contact_guard": self.SOLID_CONTACT_GUARD,
            # How faithfully this embodiment executes its commands. A name
            # from agents.CONTROL_GAPS, or a ControlGap. Default "ideal"
            # keeps every existing dataset bit-identical.
            "control_gap": self.CONTROL_GAP,
        }
        if pusher_shape == "u_socket":
            agent_options["socket_inside_friction_only"] = (
                self.SOCKET_INSIDE_FRICTION_ONLY
            )
        self.agent = make_agent(pusher_shape, **agent_options)
        self.obstacle_level = obstacle_level
        self.render_mode = render_mode
        self.image_size = int(image_size)
        # Replay/coverage checks can disable image rendering while retaining
        # the same physics and numeric observations.
        self._skip_obs_render = False
        if self.agent.action_dim == 3:
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
        self._obstacle_segments: list[pymunk.Segment] = []
        self._goal_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._goal_polygon: Polygon | None = None

    @property
    def solid_pusher(self) -> bool:
        """Fixed Sim V2 collision behavior (read-only)."""
        return bool(getattr(self.agent, "solid_pusher", self.SOLID_PUSHER))

    @property
    def socket_inside_friction_only(self) -> bool:
        """Fixed Sim V2 U-socket friction behavior (read-only)."""
        return bool(getattr(self.agent, "socket_inside_friction_only",
                            self.SOCKET_INSIDE_FRICTION_ONLY))

    @property
    def solid_contact_guard(self) -> bool:
        """Fixed Sim V2 penetration protection (read-only)."""
        return self.SOLID_CONTACT_GUARD

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
        return bool(getattr(self.agent, "socket_latched", False))

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
            "sim_version": SIM_VERSION,
            "solid_pusher": self.solid_pusher,
            "socket_inside_friction_only": self.socket_inside_friction_only,
            "solid_contact_guard": self.solid_contact_guard,
            "obstacles": obstacles,
            "reset_seed": getattr(self, "_last_reset_seed", None),
            # The compliance actually realised this episode. With
            # CONTROL_GAP="random" this differs per episode, so it has to be
            # stored or the data cannot be replayed or stratified by it.
            "control_gap": self.agent.control_gap.as_dict(),
            "control_gap_randomized": bool(getattr(self.agent, "randomize_gap", False)),
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
        # Before on_reset: subclasses override on_reset and do not call
        # super(), so doing this inside Agent.on_reset would be skipped by
        # every agent that has its own -- leaking the latency queue and
        # leaving the noise RNG unreseeded across episodes.
        self.agent.reset_control_gap(self)
        self.agent.on_reset(self)
        self._pusher_body, self._pusher_shapes = self.agent.build(
            self._space, pusher_pos
        )
        # Space is rebuilt per episode, so the handler is re-registered here.
        if getattr(self.agent, "socket_inside_friction_only", False):
            for shape in self._pusher_shapes:
                # Outside contacts are frictionless by construction. The
                # pre-solve callback restores the original combined friction
                # only for an unambiguous inner face.
                shape.friction = 0.0
                shape.collision_type = _CT_PUSHER
            for shape in self._object_shapes:
                shape.collision_type = _CT_OBJECT
            self._space.on_collision(
                _CT_PUSHER,
                _CT_OBJECT,
                pre_solve=lambda a, sp, d: self.agent._socket_friction_pre_solve(self, a, sp, d),
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
        expected_shape = (self.agent.action_dim,)
        if action.shape != expected_shape:
            raise ValueError(
                f"action must be shape {expected_shape} for "
                f"pusher={self.pusher_shape!r}, got {action.shape}"
            )

        # Action = desired pusher XY in world coords. Walk toward it at
        # PUSHER_SPEED via kinematic velocity commands; pymunk's solver still
        # resolves contact forces against the object.
        tx, ty, target_angle = self.agent.target_pose(action)
        tx = float(np.clip(tx, 0.0, self.WORLD_SIZE))
        ty = float(np.clip(ty, 0.0, self.WORLD_SIZE))

        dt_sub = self.DT / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            captured = self.agent.pre_substep(self)
            self._drive_pusher_toward(tx, ty, dt_sub, target_angle)
            # Articulated bodies must receive the velocity selected above,
            # not the previous substep's stale master-body velocity.
            self.agent.sync_auxiliary_bodies(self)
            self._space.step(dt_sub)
            self._clamp_pusher_to_static()
            self.agent.post_substep(self, captured)

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
                # Gap between the command and where the body actually is.
                # 0.0 for an ideal agent; non-zero is the embodiment's
                # execution error, which teleop should surface to the operator.
                "tracking_error": self.agent.tracking_error(self),
                "command_gap": self.agent.command_gap(),
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
            self.agent._release_socket_latch(self)

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
        self._obstacle_segments = []
        self._goal_polygon = None

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

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
        for shape in self.agent.physics_shapes(self):
            self._space.reindex_shapes_for_body(shape.body)
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
                if self.agent.active_constraints():
                    # Preserve the constrained relative pose when an attached
                    # embodiment is projected out of static geometry.
                    self._object_body.position = self._object_body.position + correction
                self.agent.sync_auxiliary_bodies(self)
                for agent_shape in self.agent.physics_shapes(self):
                    self._space.reindex_shapes_for_body(agent_shape.body)
                v = body.velocity
                into = v.dot(n)
                if into > 0.0:
                    body.velocity = v - n * into

    def _object_arena_metrics(self) -> tuple[float, float]:
        """Return ``(overflow, edge_clearance)`` for the full object polygon."""
        body = self._object_body
        xmin, ymin, xmax, ymax = self._build_object_polygon(
            (float(body.position.x), float(body.position.y)),
            float(body.angle),
        ).bounds
        overflow = max(
            0.0,
            -xmin,
            -ymin,
            xmax - self.WORLD_SIZE,
            ymax - self.WORLD_SIZE,
        )
        clearance = min(xmin, ymin, self.WORLD_SIZE - xmax, self.WORLD_SIZE - ymax)
        return float(overflow), float(clearance)

    def _pusher_object_penetration_depth(self) -> float:
        """Maximum current pusher/object overlap depth from shape queries."""
        self._space.reindex_shapes_for_body(self._object_body)
        depths: list[float] = []
        for pusher_shape in self.agent.physics_shapes(self):
            self._space.reindex_shapes_for_body(pusher_shape.body)
            for query in self._space.shape_query(pusher_shape):
                if query.shape.body is not self._object_body:
                    continue
                for point in query.contact_point_set.points:
                    depths.append(abs(float(point.distance)))
        return max(depths, default=0.0)

    def _shapes_static_penetration_depth(
        self,
        body: pymunk.Body,
        shapes: list[pymunk.Shape],
    ) -> float:
        """Maximum current overlap depth between body shapes and statics."""
        self._space.reindex_shapes_for_body(body)
        depths: list[float] = []
        for shape in shapes:
            self._space.reindex_shapes_for_body(shape.body)
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
                list(self.agent.physics_shapes(self)),
            ),
        )

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
        if self.agent.controls_angle:
            # The agent commands orientation explicitly (3-DOF and up).
            desired = target_angle
        elif self.agent.auto_orients and dist > _MIN_STICK_TURN_DIST:
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
            pusher_physics_shapes=(
                shape
                for shape in self._space.shapes
                if shape.body.body_type == pymunk.Body.KINEMATIC
            )
            if self.pusher_shape in _NEW_AGENTS
            else None,
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
