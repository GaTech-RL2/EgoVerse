"""Static obstacle layouts for PushShapesEnv.

Each level is a list of segments `((x1, y1), (x2, y2))` inside the 512x512
arena. Segments become static `pymunk.Segment` shapes attached to the
space's static body.
"""

from __future__ import annotations

import pymunk

OBSTACLE_LEVELS: dict[int, list[tuple[tuple[float, float], tuple[float, float]]]] = {
    0: [],
    1: [
        ((180.0, 180.0), (180.0, 332.0)),
    ],
    2: [
        ((180.0, 100.0), (180.0, 280.0)),
        ((332.0, 232.0), (332.0, 412.0)),
    ],
    3: [
        # Narrow vertical corridor down the middle plus a top deflector,
        # forcing pushes around the obstacles to reach goals on either side.
        ((220.0, 0.0), (220.0, 280.0)),
        ((292.0, 232.0), (292.0, 512.0)),
        ((100.0, 140.0), (220.0, 140.0)),
    ],
}

WALL_RADIUS = 4.0
WALL_FRICTION = 0.7


def build_obstacles(space: pymunk.Space, level: int) -> list[pymunk.Segment]:
    if level not in OBSTACLE_LEVELS:
        raise ValueError(
            f"unknown obstacle_level {level}, valid: {sorted(OBSTACLE_LEVELS)}"
        )
    static_body = space.static_body
    segments: list[pymunk.Segment] = []
    for (x1, y1), (x2, y2) in OBSTACLE_LEVELS[level]:
        seg = pymunk.Segment(static_body, (x1, y1), (x2, y2), WALL_RADIUS)
        seg.friction = WALL_FRICTION
        segments.append(seg)
    if segments:
        space.add(*segments)
    return segments
