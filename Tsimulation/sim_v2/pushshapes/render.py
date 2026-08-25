"""pygame drawing helpers for PushShapesEnv.

Renders a 512x512 top-down view of the arena: obstacles (gray), goal pose
outline (translucent green), object (blue), pusher (red). The arena
surface can be downsampled with `to_image_obs()` for the agent observation.
"""

from __future__ import annotations

import math
from typing import Iterable

import cv2  # provided by `opencv-python` (pinned via pyproject.toml)
import numpy as np
import pygame
import pymunk

from .shapes import (
    L_RECTS,
    SHAPES,
    STICK_HALF_LEN,
    STICK_HALF_THICK,
    U_SOCKET_RECTS,
    pusher_radius,
)

BG_COLOR = (240, 240, 240)
WALL_COLOR = (90, 90, 90)
GOAL_COLOR = (60, 180, 90)
OBJECT_COLOR = (60, 100, 200)
PUSHER_COLOR = (210, 60, 60)

# Goal pose fill is translucent; outline is nearly opaque. Tuples are
# appended to GOAL_COLOR to form RGBA values.
_GOAL_FILL_ALPHA = 70
_GOAL_OUTLINE_ALPHA = 220
_GOAL_OUTLINE_WIDTH = 2


def _rotate_translate(
    local_pts: list[tuple[float, float]],
    body_pos: tuple[float, float],
    body_angle: float,
) -> list[tuple[float, float]]:
    """Rotate `local_pts` by `body_angle` (rad) then translate by `body_pos`."""
    c, s = math.cos(body_angle), math.sin(body_angle)
    bx, by = body_pos
    return [(bx + c * lx - s * ly, by + s * lx + c * ly) for lx, ly in local_pts]


def _rect_world_verts(
    cx: float,
    cy: float,
    w: float,
    h: float,
    body_pos: tuple[float, float],
    body_angle: float,
) -> list[tuple[float, float]]:
    """Local axis-aligned rect → world-space corners under (pos, angle)."""
    hw, hh = w / 2.0, h / 2.0
    local = [
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh),
    ]
    return _rotate_translate(local, body_pos, body_angle)


def _draw_polygon(
    surface: pygame.Surface,
    color: tuple,
    verts: list[tuple[float, float]],
    width: int = 0,
) -> None:
    pygame.draw.polygon(surface, color, [(int(x), int(y)) for x, y in verts], width)


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
    pusher_physics_shapes: Iterable[pymunk.Shape] | None = None,
) -> None:
    """Composite the full scene onto `surface` (in-place)."""
    surface.fill(BG_COLOR)

    # Obstacles — pymunk segments rendered as thick lines.
    for seg in obstacle_segments:
        pygame.draw.line(
            surface,
            WALL_COLOR,
            (int(seg.a.x), int(seg.a.y)),
            (int(seg.b.x), int(seg.b.y)),
            int(max(2, seg.radius * 2)),
        )

    # Goal pose outline — drawn onto an alpha surface so the fill is
    # translucent over whatever's underneath.
    gx, gy, gtheta = goal_pose
    goal_layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    for cx, cy, w, h in SHAPES[object_shape]:
        verts = _rect_world_verts(cx, cy, w, h, (gx, gy), gtheta)
        _draw_polygon(goal_layer, (*GOAL_COLOR, _GOAL_FILL_ALPHA), verts)
        _draw_polygon(
            goal_layer, (*GOAL_COLOR, _GOAL_OUTLINE_ALPHA), verts, _GOAL_OUTLINE_WIDTH
        )
    surface.blit(goal_layer, (0, 0))

    # Object current pose — solid fill, no outline needed.
    ox, oy, otheta = object_pose
    for cx, cy, w, h in SHAPES[object_shape]:
        _draw_polygon(
            surface,
            OBJECT_COLOR,
            _rect_world_verts(cx, cy, w, h, (ox, oy), otheta),
        )

    # New embodiments can own several bodies (for example the parallel
    # gripper's palm and two moving jaws).  Render their live physics shapes
    # so the image stored in a demonstration matches what the operator sees.
    # Legacy pushers keep the original drawing path for pixel compatibility.
    if pusher_physics_shapes is None:
        _draw_pusher(surface, pusher_shape, pusher_pos, pusher_angle)
    else:
        _draw_physics_shapes(surface, pusher_physics_shapes)


def _draw_physics_shapes(
    surface: pygame.Surface,
    shapes: Iterable[pymunk.Shape],
) -> None:
    """Draw live Pymunk geometry in world coordinates.

    Agent-owned auxiliary bodies are not represented by ``pusher_pos`` alone,
    so this path is required for articulated embodiments such as ``gripper``.
    """
    for shape in shapes:
        body = shape.body
        if isinstance(shape, pymunk.Circle):
            center = body.local_to_world(shape.offset)
            pygame.draw.circle(
                surface,
                PUSHER_COLOR,
                (int(round(center.x)), int(round(center.y))),
                max(1, int(round(shape.radius))),
            )
        elif isinstance(shape, pymunk.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            if len(verts) >= 3:
                _draw_polygon(
                    surface,
                    PUSHER_COLOR,
                    [(v.x, v.y) for v in verts],
                )
        elif isinstance(shape, pymunk.Segment):
            start = body.local_to_world(shape.a)
            end = body.local_to_world(shape.b)
            width = max(1, int(round(2.0 * shape.radius)))
            pygame.draw.line(
                surface,
                PUSHER_COLOR,
                (int(round(start.x)), int(round(start.y))),
                (int(round(end.x)), int(round(end.y))),
                width,
            )
            radius = max(1, int(round(shape.radius)))
            pygame.draw.circle(
                surface,
                PUSHER_COLOR,
                (int(round(start.x)), int(round(start.y))),
                radius,
            )
            pygame.draw.circle(
                surface,
                PUSHER_COLOR,
                (int(round(end.x)), int(round(end.y))),
                radius,
            )


def _draw_pusher(
    surface: pygame.Surface,
    pusher_shape: str,
    pos: tuple[float, float],
    angle: float,
) -> None:
    px, py = pos
    if pusher_shape in ("circle", "circle_small"):
        pygame.draw.circle(
            surface,
            PUSHER_COLOR,
            (int(px), int(py)),
            max(1, int(round(pusher_radius(pusher_shape)))),
        )
        return

    if pusher_shape == "L":
        for cx, cy, w, h in L_RECTS:
            _draw_polygon(
                surface,
                PUSHER_COLOR,
                _rect_world_verts(cx, cy, w, h, (px, py), angle),
            )
        return

    if pusher_shape == "u_socket":
        for cx, cy, w, h in U_SOCKET_RECTS:
            _draw_polygon(
                surface,
                PUSHER_COLOR,
                _rect_world_verts(cx, cy, w, h, (px, py), angle),
            )
        return

    # stick: rectangle body + two end-cap circles, drawn at current angle.
    rect_verts = _rect_world_verts(
        0.0, 0.0, 2 * STICK_HALF_LEN, 2 * STICK_HALF_THICK, (px, py), angle
    )
    _draw_polygon(surface, PUSHER_COLOR, rect_verts)
    c, s = math.cos(angle), math.sin(angle)
    for offset in (-STICK_HALF_LEN, STICK_HALF_LEN):
        ex = px + c * offset
        ey = py + s * offset
        pygame.draw.circle(
            surface, PUSHER_COLOR, (int(ex), int(ey)), int(STICK_HALF_THICK)
        )


def surface_to_rgb_array(surface: pygame.Surface) -> np.ndarray:
    """Return a (H, W, 3) uint8 RGB array from a pygame Surface."""
    # pygame.surfarray returns (W, H, 3); transpose to image convention.
    arr = pygame.surfarray.array3d(surface)
    return np.transpose(arr, (1, 0, 2)).astype(np.uint8, copy=False)


def to_image_obs(surface: pygame.Surface, size: int) -> np.ndarray:
    """RGB image at the agent's observation resolution. Skips the resize
    when the surface is already the right size to avoid a copy."""
    rgb = surface_to_rgb_array(surface)
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
