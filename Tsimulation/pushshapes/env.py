"""PushShapesEnv — gym-pusht adapted with multiple shapes, pushers, obstacles."""

from __future__ import annotations

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


class PushShapesEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    WORLD_SIZE: float = 512.0
    PUSHER_SPEED: float = 200.0
    DT: float = 1.0 / 30.0
    SUBSTEPS: int = 10
    DAMPING: float = 0.85
    SUCCESS_THRESHOLD: float = 0.95
    SPAWN_MARGIN: float = 60.0

    def __init__(
        self,
        object_shape: str = "T",
        pusher_shape: str = "circle",
        obstacle_level: int = 0,
        render_mode: str | None = None,
        image_size: int = 96,
        max_episode_steps: int = 300,
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
        self.max_episode_steps = int(max_episode_steps)

        self.action_space = spaces.Box(
            low=0.0, high=float(self.WORLD_SIZE), shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    0.0, float(self.WORLD_SIZE), (2,), dtype=np.float32
                ),
                "object_pose": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
                "goal_pose": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
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
    # gym API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)
        self._space.damping = self.DAMPING

        # Arena boundary walls (so bodies can't escape the 512x512 view).
        S = self.WORLD_SIZE
        boundary = []
        for a, b in [
            ((0, 0), (S, 0)),
            ((S, 0), (S, S)),
            ((S, S), (0, S)),
            ((0, S), (0, 0)),
        ]:
            seg = pymunk.Segment(self._space.static_body, a, b, 1.0)
            seg.friction = 0.7
            boundary.append(seg)
        self._space.add(*boundary)

        self._obstacle_segments = build_obstacles(self._space, self.obstacle_level)
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

        obs = self._get_obs()
        info = {"coverage": float(self._coverage())}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (2,):
            raise ValueError(f"action must be shape (2,), got {action.shape}")
        target = np.clip(action, 0.0, self.WORLD_SIZE)

        dt_sub = self.DT / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            cur = np.array(
                [self._pusher_body.position.x, self._pusher_body.position.y],
                dtype=np.float32,
            )
            delta = target - cur
            dist = float(np.linalg.norm(delta))
            if dist < 1e-3:
                self._pusher_body.velocity = (0.0, 0.0)
            else:
                speed = min(self.PUSHER_SPEED, dist / dt_sub)
                ux, uy = delta[0] / dist, delta[1] / dist
                self._pusher_body.velocity = (float(ux * speed), float(uy * speed))
                if self.pusher_shape == "stick" and dist > 1.0:
                    self._pusher_body.angle = math.atan2(uy, ux)
            self._space.step(dt_sub)

        self._pusher_body.velocity = (0.0, 0.0)
        self._step_count += 1

        coverage = float(self._coverage())
        reward = float(np.clip(coverage, 0.0, 1.0))
        terminated = coverage >= self.SUCCESS_THRESHOLD
        truncated = False
        obs = self._get_obs()
        info = {"coverage": coverage}
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        surface = self._render_world()
        if self.render_mode == "rgb_array":
            return surface_to_rgb_array(surface)

        # human mode
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
        """Public accessor for the 512x512 world surface (for external windows
        that want to overlay text or share the rendered scene)."""
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

    def _get_obs(self) -> dict[str, np.ndarray]:
        pos = self._pusher_body.position
        agent_pos = np.array([pos.x, pos.y], dtype=np.float32)
        obj = self._object_body
        object_pose = np.array(
            [obj.position.x, obj.position.y, obj.angle], dtype=np.float32
        )
        goal_pose = np.array(self._goal_pose, dtype=np.float32)
        surface = self._render_world()
        image = to_image_obs(surface, self.image_size)
        return {
            "agent_pos": agent_pos,
            "object_pose": object_pose,
            "goal_pose": goal_pose,
            "image": image,
        }

    def _ensure_pygame_initialized(self) -> None:
        # Use the dummy video driver if we don't have a display and the user
        # hasn't asked for the human window.
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
        polys: list[Polygon] = []
        for seg in self._obstacle_segments:
            line = LineString([(seg.a.x, seg.a.y), (seg.b.x, seg.b.y)])
            polys.append(line.buffer(WALL_RADIUS + 10.0))
        return polys

    def _build_object_polygon(
        self, position: tuple[float, float], angle: float
    ) -> Polygon:
        rects = SHAPES[self.object_shape]
        polys = []
        bx, by = position
        c, s = math.cos(angle), math.sin(angle)
        for cx, cy, w, h in rects:
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
        margin = self.SPAWN_MARGIN
        for _ in range(50):
            x = float(self._np_random.uniform(margin, self.WORLD_SIZE - margin))
            y = float(self._np_random.uniform(margin, self.WORLD_SIZE - margin))
            th = float(self._np_random.uniform(-math.pi, math.pi))
            poly = self._build_object_polygon((x, y), th)
            xmin, ymin, xmax, ymax = poly.bounds
            if (
                xmin < 5
                or ymin < 5
                or xmax > self.WORLD_SIZE - 5
                or ymax > self.WORLD_SIZE - 5
            ):
                continue
            if any(poly.intersects(op) for op in obstacle_polys):
                continue
            if (
                away_from is not None
                and math.hypot(x - away_from[0], y - away_from[1]) < 120.0
            ):
                continue
            return (x, y), th
        # Fallback: center, no rotation. Better than crashing if the level is overly packed.
        return (self.WORLD_SIZE / 2.0, self.WORLD_SIZE / 2.0), 0.0

    def _sample_pusher_pos(
        self,
        obstacle_polys: list[Polygon],
        object_pos: tuple[float, float],
    ) -> tuple[float, float]:
        margin = self.SPAWN_MARGIN
        radius = PUSHER_RADIUS + 5.0
        for _ in range(50):
            x = float(self._np_random.uniform(margin, self.WORLD_SIZE - margin))
            y = float(self._np_random.uniform(margin, self.WORLD_SIZE - margin))
            disk = Point(x, y).buffer(radius)
            if any(disk.intersects(op) for op in obstacle_polys):
                continue
            if math.hypot(x - object_pos[0], y - object_pos[1]) < 80.0:
                continue
            return (x, y)
        return (self.WORLD_SIZE / 4.0, self.WORLD_SIZE / 4.0)
