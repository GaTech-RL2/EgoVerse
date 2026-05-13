"""pygame drawing helpers for PushShapesEnv.

Renders a 512x512 top-down view of the arena: obstacles (gray), goal pose
outline (translucent green), object (blue), pusher (red). The arena
surface can be downsampled with `to_image_obs()` for the agent observation.
"""

from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np
import pygame
import pymunk

from Tsimulation.pushshapes.shapes import (
    PUSHER_RADIUS,
    SHAPES,
    STICK_HALF_LEN,
    STICK_HALF_THICK,
)

BG_COLOR = (240, 240, 240)
WALL_COLOR = (90, 90, 90)
GOAL_COLOR = (60, 180, 90)
OBJECT_COLOR = (60, 100, 200)
PUSHER_COLOR = (210, 60, 60)


def _world_rect_to_polygon(
    cx: float,
    cy: float,
    w: float,
    h: float,
    body_pos: tuple[float, float],
    body_angle: float,
) -> list[tuple[float, float]]:
    hw, hh = w / 2.0, h / 2.0
    local = [
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh),
    ]
    c, s = math.cos(body_angle), math.sin(body_angle)
    bx, by = body_pos
    return [(bx + c * lx - s * ly, by + s * lx + c * ly) for lx, ly in local]


def draw_arena(
    surface: pygame.Surface,
    *,
    object_shape: str,
    object_pose: tuple[float, float, float],
    goal_pose: tuple[float, float, float],
    pusher_shape: str,
    pusher_pos: tuple[float, float],
    pusher_angle: float,
    obstacle_segments: Iterable[pymunk.Segment],
) -> None:
    surface.fill(BG_COLOR)

    # Obstacles
    for seg in obstacle_segments:
        a = seg.a
        b = seg.b
        pygame.draw.line(
            surface,
            WALL_COLOR,
            (int(a.x), int(a.y)),
            (int(b.x), int(b.y)),
            int(max(2, seg.radius * 2)),
        )

    # Goal pose outline (translucent green).
    goal_x, goal_y, goal_theta = goal_pose
    goal_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for cx, cy, w, h in SHAPES[object_shape]:
        verts = _world_rect_to_polygon(cx, cy, w, h, (goal_x, goal_y), goal_theta)
        pygame.draw.polygon(
            goal_surface, (*GOAL_COLOR, 70), [(int(x), int(y)) for x, y in verts]
        )
        pygame.draw.polygon(
            goal_surface, (*GOAL_COLOR, 220), [(int(x), int(y)) for x, y in verts], 2
        )
    surface.blit(goal_surface, (0, 0))

    # Object current pose.
    obj_x, obj_y, obj_theta = object_pose
    for cx, cy, w, h in SHAPES[object_shape]:
        verts = _world_rect_to_polygon(cx, cy, w, h, (obj_x, obj_y), obj_theta)
        pygame.draw.polygon(surface, OBJECT_COLOR, [(int(x), int(y)) for x, y in verts])

    # Pusher.
    px, py = pusher_pos
    if pusher_shape == "circle":
        pygame.draw.circle(
            surface, PUSHER_COLOR, (int(px), int(py)), int(PUSHER_RADIUS)
        )
    else:
        # stick: rectangle + two end caps, drawn at current angle.
        c, s = math.cos(pusher_angle), math.sin(pusher_angle)
        hw, hh = STICK_HALF_LEN, STICK_HALF_THICK
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        world = [(px + c * lx - s * ly, py + s * lx + c * ly) for lx, ly in corners]
        pygame.draw.polygon(surface, PUSHER_COLOR, [(int(x), int(y)) for x, y in world])
        for offset in (-STICK_HALF_LEN, STICK_HALF_LEN):
            ex = px + c * offset
            ey = py + s * offset
            pygame.draw.circle(
                surface, PUSHER_COLOR, (int(ex), int(ey)), int(STICK_HALF_THICK)
            )


def surface_to_rgb_array(surface: pygame.Surface) -> np.ndarray:
    """Returns a (H, W, 3) uint8 RGB array."""
    arr = pygame.surfarray.array3d(surface)
    # pygame uses (W, H, 3) with axes swapped; transpose to (H, W, 3).
    return np.transpose(arr, (1, 0, 2)).astype(np.uint8, copy=False)


def to_image_obs(surface: pygame.Surface, size: int) -> np.ndarray:
    rgb = surface_to_rgb_array(surface)
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
