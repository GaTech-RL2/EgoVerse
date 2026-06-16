"""Static obstacle layouts for PushShapesEnv.

Each level is a list of segments `((x1, y1), (x2, y2))` inside the 512x512
arena. Segments are added to the space's static body as `pymunk.Segment`
shapes. Level 0 is empty; higher levels progressively constrain the routes
the pusher can take.
"""

from __future__ import annotations

import pymunk

WALL_RADIUS = 4.0
WALL_FRICTION = 0.7

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


def build_obstacles(space: pymunk.Space, level: int) -> list[pymunk.Segment]:
    """Add the configured obstacles for `level` to `space` and return them."""
    if level not in OBSTACLE_LEVELS:
        raise ValueError(
            f"unknown obstacle_level {level}, valid: {sorted(OBSTACLE_LEVELS)}"
        )
    segments = [
        pymunk.Segment(space.static_body, a, b, WALL_RADIUS)
        for a, b in OBSTACLE_LEVELS[level]
    ]
    for seg in segments:
        seg.friction = WALL_FRICTION
    if segments:
        space.add(*segments)
    return segments
