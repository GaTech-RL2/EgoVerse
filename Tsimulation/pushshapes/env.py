"""PushShapesEnv — gym-pusht adapted with multiple shapes, pushers, obstacles.

A 512x512 top-down arena where a kinematic pusher (circle or stick) shoves
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

from Tsimulation.pushshapes.obstacles import (
    OBSTACLE_LEVELS,
    WALL_RADIUS,
    build_obstacles,
)
from Tsimulation.pushshapes.render import (
    draw_arena,
    surface_to_rgb_array,
    to_image_obs,
)
from Tsimulation.pushshapes.shapes import (
    PUSHER_RADIUS,
    SHAPES,
    make_object,
    make_pusher,
)

# Tunables not exposed via __init__ — surfaced here for visibility.
_MIN_TARGET_DIST = 1e-3  # below this, treat pusher as on-target
_MIN_STICK_TURN_DIST = 1.0  # stick only re-orients when moving meaningfully
_WALL_INSET = 5.0  # rejection clearance from arena edges
_PUSHER_OBJECT_MIN_DIST = 80.0  # pusher cannot spawn on top of object
_GOAL_OBJECT_MIN_DIST = 120.0  # goal pose must be visibly different from object
_SPAWN_MAX_TRIES = 50  # rejection-sampling budget per spawn


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
    ):
        super().__init__()

        if object_shape not in SHAPES:
            raise ValueError(f"object_shape {object_shape!r} not in {list(SHAPES)}")
        if pusher_shape not in ("circle", "stick"):
            raise ValueError(
                f"pusher_shape {pusher_shape!r} not in ('circle', 'stick')"
            )
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
        self.render_mode = render_mode
        self.image_size = int(image_size)

        self.action_space = spaces.Box(
            low=0.0, high=float(self.WORLD_SIZE), shape=(2,), dtype=np.float64
        )
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    0.0, float(self.WORLD_SIZE), (2,), dtype=np.float64
                ),
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
    def object_pose(self) -> tuple[float, float, float]:
        b = self._object_body
        return (float(b.position.x), float(b.position.y), float(b.angle))

    @property
    def goal_pose(self) -> tuple[float, float, float]:
        return self._goal_pose

    def get_episode_init(self) -> dict:
        """Capture full episode init state for deterministic replay."""
        obstacles = [[list(a), list(b)] for a, b in OBSTACLE_LEVELS.get(self.obstacle_level, [])]
        init = {
            "agent_pos": list(self.agent_pos),
            "object_pose": list(self.object_pose),
            "goal_pose": list(self.goal_pose),
            "object_shape": self.object_shape,
            "pusher_shape": self.pusher_shape,
            "obstacle_level": self.obstacle_level,
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

        self._build_boundary_walls()
        self._obstacle_segments = build_obstacles(self._space, self.obstacle_level)

        # Rejection-sample non-overlapping object / goal / pusher placements.
        obstacle_polys = self._obstacle_polygons()
        object_pos, object_angle = self._sample_object_pose(obstacle_polys)
        goal_pos, goal_angle = self._sample_object_pose(
            obstacle_polys, away_from=object_pos
        )
        pusher_pos = self._sample_pusher_pos(obstacle_polys, object_pos)

        self._object_body, _ = make_object(
            self.object_shape, self._space, object_pos, object_angle
        )
        self._pusher_body, _ = make_pusher(self.pusher_shape, self._space, pusher_pos)

        self._goal_pose = (float(goal_pos[0]), float(goal_pos[1]), float(goal_angle))
        self._goal_polygon = self._build_object_polygon(goal_pos, goal_angle)
        self._step_count = 0

        return self._get_obs(), {"coverage": float(self._coverage())}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (2,):
            raise ValueError(f"action must be shape (2,), got {action.shape}")

        # Action = desired pusher XY in world coords. Walk toward it at
        # PUSHER_SPEED via kinematic velocity commands; pymunk's solver still
        # resolves contact forces against the object.
        tx = float(np.clip(action[0], 0.0, self.WORLD_SIZE))
        ty = float(np.clip(action[1], 0.0, self.WORLD_SIZE))

        dt_sub = self.DT / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            self._drive_pusher_toward(tx, ty, dt_sub)
            self._space.step(dt_sub)

        # Zero pusher velocity (and angular velocity) between outer steps so
        # stale motion doesn't drift contacts when no new action comes in.
        self._pusher_body.velocity = (0.0, 0.0)
        self._pusher_body.angular_velocity = 0.0
        self._step_count += 1

        coverage = float(self._coverage())
        reward = float(np.clip(coverage, 0.0, 1.0))
        terminated = coverage >= self.SUCCESS_THRESHOLD
        truncated = False  # episode cutoff disabled — caller decides when to stop
        return self._get_obs(), reward, terminated, truncated, {"coverage": coverage}

    def set_state(
        self,
        agent_pos: tuple[float, float] | None = None,
        object_pose: tuple[float, float, float] | None = None,
        goal_pose: tuple[float, float, float] | None = None,
    ) -> None:
        """Override live env state after reset(). Any subset of args may be
        passed. Velocities are zeroed so the next step() starts at rest."""
        if self._space is None:
            raise RuntimeError("call reset() before set_state()")

        if agent_pos is not None:
            self._pusher_body.position = (float(agent_pos[0]), float(agent_pos[1]))
            self._pusher_body.velocity = (0.0, 0.0)

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
        self._obstacle_segments = []
        self._goal_polygon = None

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _drive_pusher_toward(self, tx: float, ty: float, dt_sub: float) -> None:
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
            body.angular_velocity = 0.0
            return

        # Cap by PUSHER_SPEED but also by distance-remaining-per-substep so
        # we don't overshoot the target inside a single substep.
        speed = min(self.PUSHER_SPEED, dist / dt_sub)
        ux, uy = dx / dist, dy / dist
        body.velocity = (ux * speed, uy * speed)

        if self.pusher_shape == "stick" and dist > _MIN_STICK_TURN_DIST:
            # Shortest signed angle diff to the velocity direction, capped at
            # STICK_TURN_RATE so the stick eases into its new heading rather
            # than rotating instantly through contacts.
            desired = math.atan2(uy, ux)
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
        surface = self._render_world()
        return {
            "agent_pos": np.array([pos.x, pos.y], dtype=np.float64),
            "object_pose": np.array(
                [obj.position.x, obj.position.y, obj.angle], dtype=np.float64
            ),
            "goal_pose": np.array(self._goal_pose, dtype=np.float64),
            "image": to_image_obs(surface, self.image_size),
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
        radius = PUSHER_RADIUS + 5.0
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
