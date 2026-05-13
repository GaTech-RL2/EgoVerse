"""Geometry factories for pushable objects and pusher tools.

Each pushable shape is expressed as a list of axis-aligned rectangles
`(cx, cy, w, h)` in body-local coordinates. To add a new shape, append an
entry to ``SHAPES``:

    SHAPES["L"] = [
        (0, -30, 30, 120),   # vertical stem
        (30, 30, 90, 30),    # horizontal foot
    ]
"""

from __future__ import annotations

from typing import Literal

import pymunk

SHAPES: dict[str, list[tuple[float, float, float, float]]] = {
    # gym-pusht's canonical T: 120x30 top bar + 30x90 stem below it.
    "T": [
        (0.0, -30.0, 120.0, 30.0),
        (0.0, 30.0, 30.0, 90.0),
    ],
    # U-shape opening upward: two vertical legs + a bottom bar.
    "U": [
        (-45.0, 0.0, 30.0, 120.0),
        (45.0, 0.0, 30.0, 120.0),
        (0.0, 60.0, 120.0, 30.0),
    ],
    # Z built from three axis-aligned rectangles (keeps physics stable —
    # slanted polys can produce unstable collision contacts at low mass).
    "Z": [
        (-15.0, -30.0, 90.0, 30.0),
        (0.0, 0.0, 30.0, 30.0),
        (15.0, 30.0, 90.0, 30.0),
    ],
}

OBJECT_DENSITY = 0.01
OBJECT_FRICTION = 15.0
PUSHER_RADIUS = 15.0
STICK_HALF_LEN = 30.0
STICK_HALF_THICK = 5.0


def _rect_verts(cx: float, cy: float, w: float, h: float) -> list[tuple[float, float]]:
    hw, hh = w / 2.0, h / 2.0
    return [
        (cx - hw, cy - hh),
        (cx + hw, cy - hh),
        (cx + hw, cy + hh),
        (cx - hw, cy + hh),
    ]


def make_object(
    shape: Literal["T", "U", "Z"],
    space: pymunk.Space,
    position: tuple[float, float],
    angle: float = 0.0,
) -> tuple[pymunk.Body, list[pymunk.Poly]]:
    if shape not in SHAPES:
        raise ValueError(f"unknown object shape '{shape}', valid: {list(SHAPES)}")

    body = pymunk.Body()
    body.position = position
    body.angle = angle

    polys: list[pymunk.Poly] = []
    for cx, cy, w, h in SHAPES[shape]:
        poly = pymunk.Poly(body, _rect_verts(cx, cy, w, h))
        poly.density = OBJECT_DENSITY
        poly.friction = OBJECT_FRICTION
        polys.append(poly)

    space.add(body, *polys)
    return body, polys


def make_pusher(
    shape: Literal["circle", "stick"],
    space: pymunk.Space,
    position: tuple[float, float],
) -> tuple[pymunk.Body, list[pymunk.Shape]]:
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = position

    shapes: list[pymunk.Shape] = []
    if shape == "circle":
        s = pymunk.Circle(body, PUSHER_RADIUS)
        s.friction = OBJECT_FRICTION
        shapes.append(s)
    elif shape == "stick":
        rect = pymunk.Poly(
            body,
            _rect_verts(0.0, 0.0, 2 * STICK_HALF_LEN, 2 * STICK_HALF_THICK),
        )
        rect.friction = OBJECT_FRICTION
        end_a = pymunk.Circle(body, STICK_HALF_THICK, offset=(-STICK_HALF_LEN, 0.0))
        end_b = pymunk.Circle(body, STICK_HALF_THICK, offset=(STICK_HALF_LEN, 0.0))
        end_a.friction = OBJECT_FRICTION
        end_b.friction = OBJECT_FRICTION
        shapes.extend([rect, end_a, end_b])
    else:
        raise ValueError(f"unknown pusher shape '{shape}', valid: ['circle', 'stick']")

    space.add(body, *shapes)
    return body, shapes


def aabb(shape: str) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box (xmin, ymin, xmax, ymax) of the object in
    its rest pose. Used for rejection-sampling spawn positions."""
    rects = SHAPES[shape]
    xs_min = [cx - w / 2 for cx, _cy, w, _h in rects]
    xs_max = [cx + w / 2 for cx, _cy, w, _h in rects]
    ys_min = [cy - h / 2 for _cx, cy, _w, h in rects]
    ys_max = [cy + h / 2 for _cx, cy, _w, h in rects]
    return (min(xs_min), min(ys_min), max(xs_max), max(ys_max))
