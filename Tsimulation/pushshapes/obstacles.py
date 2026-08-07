"""Static obstacle layouts for PushShapesEnv.

Each level is a list of segments `((x1, y1), (x2, y2))` inside the 512x512
arena. Segments are added to the space's static body as `pymunk.Segment`
shapes. Level 0 is empty; higher levels progressively constrain the routes
the pusher can take.

Levels 4..19 were designed to be solvable: every corridor that an object
must traverse is at least ``_MIN_CORRIDOR`` units wide so the T-shape
(120x120 AABB) plus pusher (30 diameter) can fit through. The accompanying
:func:`verify_level_solvable` helper probes each level by sampling many
(object, goal) pairs and reporting spawn-fallback rate and per-quadrant
reachability.
"""

from __future__ import annotations

from typing import Iterable

import pymunk

WALL_RADIUS = 4.0
WALL_FRICTION = 0.7

# Minimum gap that a T-shape (120x120 AABB) + circle pusher (30 diameter)
# can navigate through with a safety margin. Used to size every corridor
# in the level designs below.
_MIN_CORRIDOR = 150.0

# 512x512 arena. Levels assume this and place obstacles inside.
_W = 512.0

Segment = tuple[tuple[float, float], tuple[float, float]]


# ---------------------------------------------------------------------- #
# Composable obstacle primitives.
# Each helper returns a list of Segments. Higher levels compose these.
# ---------------------------------------------------------------------- #


def _wall(p1: tuple[float, float], p2: tuple[float, float]) -> list[Segment]:
    return [(p1, p2)]


def _box(cx: float, cy: float, w: float, h: float) -> list[Segment]:
    """Closed axis-aligned box (4 segments)."""
    hw, hh = w / 2.0, h / 2.0
    p1 = (cx - hw, cy - hh)
    p2 = (cx + hw, cy - hh)
    p3 = (cx + hw, cy + hh)
    p4 = (cx - hw, cy + hh)
    return [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]


def _l_shape(
    corner: tuple[float, float],
    arm_x: float,
    arm_y: float,
) -> list[Segment]:
    """L-shape with the elbow at ``corner``. Positive arm goes right/down;
    negative arm goes left/up."""
    cx, cy = corner
    return [
        (corner, (cx + arm_x, cy)),
        (corner, (cx, cy + arm_y)),
    ]


def _cross(cx: float, cy: float, arm: float) -> list[Segment]:
    """Plus / cross: two crossing segments of total length ``2*arm``."""
    return [
        ((cx - arm, cy), (cx + arm, cy)),
        ((cx, cy - arm), (cx, cy + arm)),
    ]


def _diamond(cx: float, cy: float, r: float) -> list[Segment]:
    """Rotated square (diamond) outline."""
    return [
        ((cx, cy - r), (cx + r, cy)),
        ((cx + r, cy), (cx, cy + r)),
        ((cx, cy + r), (cx - r, cy)),
        ((cx - r, cy), (cx, cy - r)),
    ]


def _hexagon(cx: float, cy: float, r: float) -> list[Segment]:
    """Regular hexagon (6 short segments) — pointy-top orientation."""
    import math

    verts = [
        (cx + r * math.cos(math.pi / 2 + i * math.pi / 3),
         cy + r * math.sin(math.pi / 2 + i * math.pi / 3))
        for i in range(6)
    ]
    return [(verts[i], verts[(i + 1) % 6]) for i in range(6)]


def _triangle(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> list[Segment]:
    return [(p1, p2), (p2, p3), (p3, p1)]


def _arc(
    cx: float,
    cy: float,
    r: float,
    t_start: float,
    t_end: float,
    n: int = 16,
) -> list[Segment]:
    """Circular arc approximated by ``n`` straight segments.

    Angles use math convention (radians) but the env is +y-down (pygame), so
    ``t = 0`` is right, ``t = pi/2`` is down, ``t = pi`` is left, ``t = 3*pi/2``
    is up. Increase ``n`` for smoother curvature on larger radii."""
    import math

    pts = [
        (
            cx + r * math.cos(t_start + (t_end - t_start) * i / n),
            cy + r * math.sin(t_start + (t_end - t_start) * i / n),
        )
        for i in range(n + 1)
    ]
    return [(pts[i], pts[i + 1]) for i in range(n)]


# ---------------------------------------------------------------------- #
# Level layouts.
# ---------------------------------------------------------------------- #


def _level_3_chicane() -> list[Segment]:
    """Two staggered horizontal walls leaving a wide alternating gap. Fixes
    the original level-3 which had a 72-wide corridor (unsolvable for T)."""
    # Top wall: x in [0, 280], y = 180
    # Bottom wall: x in [232, 512], y = 332
    # Vertical corridor between them is ~150 wide (332-180=152), the lateral
    # detour at top requires going right of x=280, at bottom left of x=232.
    return [
        ((0.0, 180.0), (280.0, 180.0)),
        ((232.0, 332.0), (_W, 332.0)),
    ]


def _level_4_single_box() -> list[Segment]:
    return _box(256.0, 256.0, 80.0, 80.0)


def _level_5_corner_l() -> list[Segment]:
    """Free-standing L-shaped obstacle in upper-left arena region. Arms point
    inward (right + down) and never touch the arena edges so nothing is
    enclosed."""
    return _l_shape(corner=(180.0, 180.0), arm_x=100.0, arm_y=100.0)


def _level_6_t_obstacle() -> list[Segment]:
    """T-shaped wall — horizontal top + vertical stem (entirely above
    center so the bottom half is open and any object/goal can be placed)."""
    return [
        ((180.0, 130.0), (332.0, 130.0)),
        ((256.0, 130.0), (256.0, 256.0)),
    ]


def _level_7_central_cross() -> list[Segment]:
    """Small plus in center. Corridors around it are >150."""
    return _cross(256.0, 256.0, 70.0)


def _level_8_diamond() -> list[Segment]:
    return _diamond(256.0, 256.0, 90.0)


def _level_9_two_boxes() -> list[Segment]:
    """Two boxes on the diagonal leaving a wide zigzag path."""
    return _box(170.0, 170.0, 80.0, 80.0) + _box(342.0, 342.0, 80.0, 80.0)


def _level_10_zigzag_walls() -> list[Segment]:
    """Two partial horizontal walls with alternating side openings; the
    walls are 200 apart vertically so the T (120 tall) has a clear
    horizontal band between them."""
    return [
        ((0.0, 156.0), (240.0, 156.0)),     # opening on right: x[240,512] = 272 wide
        ((272.0, 356.0), (_W, 356.0)),       # opening on left:  x[0,272]   = 272 wide
    ]


def _level_11_corner_ls() -> list[Segment]:
    """Two free-standing L-shapes in opposite arena regions. Arms point
    inward so neither L touches the arena edges."""
    return (
        _l_shape(corner=(170.0, 170.0), arm_x=100.0, arm_y=100.0)
        + _l_shape(corner=(342.0, 342.0), arm_x=-100.0, arm_y=-100.0)
    )


def _level_12_central_cross_plus_boxes() -> list[Segment]:
    """Central cross + one diagonal satellite box. Smaller features so the
    T can navigate around either obstacle in any direction."""
    return (
        _cross(256.0, 256.0, 40.0)
        + _box(140.0, 380.0, 40.0, 40.0)
    )


def _level_13_dual_diamonds() -> list[Segment]:
    """Two free-standing diamonds on the diagonal, sized so the T can
    navigate fully around either one."""
    return _diamond(170.0, 170.0, 50.0) + _diamond(342.0, 342.0, 50.0)


def _level_14_hexagon_center() -> list[Segment]:
    return _hexagon(256.0, 256.0, 75.0)


def _level_15_hexagon_with_walls() -> list[Segment]:
    """Hexagon plus two short interior walls — none of which touch the arena
    edges, so no region is enclosed."""
    return (
        _hexagon(256.0, 256.0, 65.0)
        + _wall((100.0, 130.0), (260.0, 130.0))
        + _wall((252.0, 382.0), (412.0, 382.0))
    )


def _level_16_three_boxes_row() -> list[Segment]:
    """Three small boxes in a horizontal row with wide vertical lanes around."""
    return (
        _box(140.0, 256.0, 50.0, 50.0)
        + _box(256.0, 256.0, 50.0, 50.0)
        + _box(372.0, 256.0, 50.0, 50.0)
    )


def _level_17_triangle_and_box() -> list[Segment]:
    """Triangle wedge plus a box, both pulled away from arena edges."""
    return (
        _triangle((190.0, 190.0), (260.0, 190.0), (225.0, 250.0))
        + _box(360.0, 360.0, 60.0, 60.0)
    )


def _level_18_double_zigzag() -> list[Segment]:
    """S-shaped slalom (two staggered horizontal walls with alternating
    side openings, 200 apart vertically)."""
    return [
        ((0.0, 156.0), (240.0, 156.0)),     # opening on right
        ((272.0, 356.0), (_W, 356.0)),       # opening on left
    ]


def _level_19_maze() -> list[Segment]:
    """Densest solvable layout: small central diamond + 4 free-standing L's
    pointing inward from each corner region. Tuned so all corridors stay
    in a single navigable component."""
    return (
        _diamond(256.0, 256.0, 45.0)
        + _l_shape(corner=(170.0, 170.0), arm_x=60.0, arm_y=60.0)
        + _l_shape(corner=(342.0, 170.0), arm_x=-60.0, arm_y=60.0)
        + _l_shape(corner=(170.0, 342.0), arm_x=60.0, arm_y=-60.0)
        + _l_shape(corner=(342.0, 342.0), arm_x=-60.0, arm_y=-60.0)
    )


# ---------------------------------------------------------------------- #
# Curve-based levels (20..29). All composed from `_arc` primitives; radii
# and gaps tuned so the T (120 AABB) + circle pusher can still navigate.
# ---------------------------------------------------------------------- #


def _level_20_c_curve() -> list[Segment]:
    """Single C-shaped arc opening to the right (180-deg sweep on left half
    of center)."""
    import math

    return _arc(256.0, 256.0, 80.0, math.pi / 2, 3 * math.pi / 2, n=24)


def _level_21_s_curve() -> list[Segment]:
    """S-curve: top half-circle bulging down + bottom half-circle bulging up,
    diagonally offset so neither arc closes off any region."""
    import math

    return (
        _arc(180.0, 180.0, 55.0, 0.0, math.pi, n=18)
        + _arc(342.0, 342.0, 55.0, math.pi, 2 * math.pi, n=18)
    )


def _level_22_open_ring() -> list[Segment]:
    """Ring with a single 60-degree gap on the right side."""
    import math

    return _arc(256.0, 256.0, 80.0, math.pi / 6, 2 * math.pi - math.pi / 6, n=28)


def _level_23_four_quarter_corners() -> list[Segment]:
    """Quarter-arc in each corner curving inward toward the arena center."""
    import math

    r = 70.0
    inset = 130.0
    return (
        # top-left: arc from right (0) sweeping down (pi/2)
        _arc(inset, inset, r, 0.0, math.pi / 2, n=10)
        # top-right: arc from down (pi/2) sweeping left (pi)
        + _arc(_W - inset, inset, r, math.pi / 2, math.pi, n=10)
        # bottom-right: arc from left (pi) sweeping up (3pi/2)
        + _arc(_W - inset, _W - inset, r, math.pi, 3 * math.pi / 2, n=10)
        # bottom-left: arc from up (3pi/2) sweeping right (2pi)
        + _arc(inset, _W - inset, r, 3 * math.pi / 2, 2 * math.pi, n=10)
    )


def _level_24_arch_top() -> list[Segment]:
    """Top arch: half-circle hanging from the top edge of the playable area
    (sits well below the wall so a T can squeeze through underneath)."""
    import math

    return _arc(256.0, 130.0, 90.0, 0.0, math.pi, n=20)


def _level_25_bowl_bottom() -> list[Segment]:
    """Bottom bowl: half-circle resting near the bottom of the arena. Mirror
    of level 24 — visually similar primitive, different placement."""
    import math

    return _arc(256.0, 382.0, 90.0, math.pi, 2 * math.pi, n=20)


def _level_26_wave_floor() -> list[Segment]:
    """Three small half-circle bumps spaced across the middle (wavy ridge),
    each bumping downward. Wide vertical lanes above and below."""
    import math

    r = 35.0
    y = 256.0
    return (
        _arc(130.0, y, r, math.pi, 2 * math.pi, n=12)
        + _arc(256.0, y, r, math.pi, 2 * math.pi, n=12)
        + _arc(382.0, y, r, math.pi, 2 * math.pi, n=12)
    )


def _level_27_two_loops() -> list[Segment]:
    """Two open loops side-by-side (figure-8-ish silhouette): each is an arc
    with a small gap facing outward so neither encloses anything."""
    import math

    return (
        _arc(180.0, 256.0, 60.0, math.pi / 6, 2 * math.pi - math.pi / 6, n=22)
        + _arc(332.0, 256.0, 60.0,
               math.pi - math.pi / 6, 3 * math.pi - math.pi / 6, n=22)
    )


def _level_28_corner_quarter_pipe() -> list[Segment]:
    """Quarter-pipe in the upper-left corner (arc) + a short diagonal wall
    in the opposite area for asymmetry."""
    import math

    return (
        _arc(120.0, 120.0, 110.0, 0.0, math.pi / 2, n=14)
        + _wall((320.0, 360.0), (420.0, 420.0))
    )


def _level_29_concentric_rings() -> list[Segment]:
    """Two concentric arcs (inner + outer), each with a gap, gaps on opposite
    sides so a path winds between them like a circular maze."""
    import math

    # Outer ring with gap on the right.
    outer = _arc(256.0, 256.0, 130.0, math.pi / 5, 2 * math.pi - math.pi / 5, n=32)
    # Inner ring with gap on the left.
    inner = _arc(256.0, 256.0, 55.0, math.pi + math.pi / 4, 3 * math.pi - math.pi / 4, n=20)
    return outer + inner


OBSTACLE_LEVELS: dict[int, list[Segment]] = {
    0: [],
    1: [
        ((180.0, 180.0), (180.0, 332.0)),
    ],
    2: [
        ((180.0, 100.0), (180.0, 280.0)),
        ((332.0, 232.0), (332.0, 412.0)),
    ],
    3: _level_3_chicane(),
    4: _level_4_single_box(),
    5: _level_5_corner_l(),
    6: _level_6_t_obstacle(),
    7: _level_7_central_cross(),
    8: _level_8_diamond(),
    9: _level_9_two_boxes(),
    10: _level_10_zigzag_walls(),
    11: _level_11_corner_ls(),
    12: _level_12_central_cross_plus_boxes(),
    13: _level_13_dual_diamonds(),
    14: _level_14_hexagon_center(),
    15: _level_15_hexagon_with_walls(),
    16: _level_16_three_boxes_row(),
    17: _level_17_triangle_and_box(),
    18: _level_18_double_zigzag(),
    19: _level_19_maze(),
    20: _level_20_c_curve(),
    21: _level_21_s_curve(),
    22: _level_22_open_ring(),
    23: _level_23_four_quarter_corners(),
    24: _level_24_arch_top(),
    25: _level_25_bowl_bottom(),
    26: _level_26_wave_floor(),
    27: _level_27_two_loops(),
    28: _level_28_corner_quarter_pipe(),
    29: _level_29_concentric_rings(),
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


def all_levels() -> Iterable[int]:
    return sorted(OBSTACLE_LEVELS)
