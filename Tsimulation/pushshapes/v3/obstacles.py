"""Static obstacle layouts for PushShapesEnv.

Each level is a list of segments `((x1, y1), (x2, y2))` inside the 512x512
arena. Segments are added to the space's static body as `pymunk.Segment`
shapes. Level 0 is empty; higher levels progressively constrain the routes
the pusher can take.

Level 0 is the empty arena. Levels 1..30 are the sketch-derived families:
five base designs, each expanded over every flip / rotation of where the
obstacle sits (see :data:`SKETCH_FAMILY_NAMES`). Every corridor an object
must traverse is at least ``_MIN_CORRIDOR`` units wide so the T-shape
(120x120 AABB) plus pusher (30 diameter) can fit through.

``scripts/plot_obstacle_levels.py`` renders any range of levels into one
contact sheet and reports, per level, how many connected components the
navigable space breaks into once eroded by the T's clearance. Anything
other than 1 component means part of the arena is sealed off.
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


def _rotate_about(
    segments: list[Segment],
    cx: float,
    cy: float,
    theta: float,
) -> list[Segment]:
    """Rotate every segment endpoint by ``theta`` radians about ``(cx, cy)``."""
    import math

    c, s = math.cos(theta), math.sin(theta)

    def rot(p: tuple[float, float]) -> tuple[float, float]:
        dx, dy = p[0] - cx, p[1] - cy
        return (cx + c * dx - s * dy, cy + s * dx + c * dy)

    return [(rot(a), rot(b)) for a, b in segments]


def _u_pocket(
    cx: float,
    cy: float,
    width: float,
    depth: float,
    theta: float = 0.0,
    extend_flank: bool = False,
) -> list[Segment]:
    """Three-sided rectangular pocket ("U") centred on ``(cx, cy)``.

    At ``theta = 0`` the mouth faces +x (right): the back wall sits at
    ``x = cx - depth/2`` and the two flanks run along ``y = cy +/- width/2``.
    ``theta`` rotates the whole pocket (and therefore the mouth heading)
    about its centre.

    With ``extend_flank`` one flank keeps going past the mouth until it hits
    the arena boundary, so the pocket is anchored to the wall instead of
    free-standing. The extension has to be applied after the rotation -- how
    far the flank runs depends on which wall it happens to be pointing at.
    """
    hw, hd = width / 2.0, depth / 2.0
    back_t = (cx - hd, cy - hw)
    back_b = (cx - hd, cy + hw)
    mouth_t = (cx + hd, cy - hw)
    mouth_b = (cx + hd, cy + hw)
    segs = [
        (back_t, back_b),      # closed back
        (back_t, mouth_t),     # flank
        (back_b, mouth_b),     # long flank when extend_flank
    ]
    if theta:
        segs = _rotate_about(segs, cx, cy, theta)
    if extend_flank:
        import math

        a, b = segs[2]
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dx, dy)
        direction = (dx / norm, dy / norm)
        chord = _clip_ray_to_arena(b, direction)
        if chord is not None:
            t = max(chord[1], 0.0)
            segs[2] = (a, (b[0] + t * direction[0], b[1] + t * direction[1]))
    return segs


def _clip_ray_to_arena(
    base: tuple[float, float],
    direction: tuple[float, float],
) -> tuple[float, float] | None:
    """Parameter range ``(t_min, t_max)`` of ``base + t * direction`` inside
    the arena box. ``None`` if the line misses the arena entirely."""
    t_min, t_max = -1e9, 1e9
    for p, q in (
        (-direction[0], base[0]),
        (direction[0], _W - base[0]),
        (-direction[1], base[1]),
        (direction[1], _W - base[1]),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)
    return (t_min, t_max) if t_min < t_max else None


def _boundary_radius(cx: float, cy: float, theta: float) -> float:
    """Distance from ``(cx, cy)`` to the arena boundary along ``theta``."""
    import math

    c, s = math.cos(theta), math.sin(theta)
    best = float("inf")
    for p, q in ((c, _W - cx), (-c, cx), (s, _W - cy), (-s, cy)):
        if p > 1e-12:
            best = min(best, q / p)
    return best


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
# Sketch-derived family levels (1..30).
#
# Five base designs, each expanded into every orientation you get by
# flipping / rotating where the obstacle sits. 6 + 4 + 4 + 8 + 8 = 30.
#
#    1..6   edge_wall  — one stick in from an edge (4 axis + 2 diagonal)
#    7..10  chicane    — two staggered walls, openings on opposite sides
#   11..14  diagonal   — two staggered diagonal walls (diagonal zig-zag)
#   15..22  pocket     — U pocket, one flank run to the wall, 8 headings
#   23..30  spiral     — cyclone: 2 spiral arms + open eye, 8 headings
#
# Every layout keeps at least ~150 units of clear corridor (the T's 120
# AABB plus the 30-diameter pusher) around and through it, and no region
# is fully sealed off — the pocket / ring mouths are all >= 150 wide.
# ---------------------------------------------------------------------- #

# Single wall: length as a fraction of the arena, leaving a wide bypass.
_EDGE_WALL_LEN = 330.0
_EDGE_WALL_MID = _W / 2.0
# Diagonal stick: how far along each axis it reaches out of its corner.
_EDGE_WALL_DIAG_LEN = 300.0

_INV_SQRT2 = 0.7071067811865476
# Diagonal chicane: half-separation between the two walls (perpendicular),
# and how much of each wall's chord is left open at its free end.
_DIAG_OFFSET = 115.0
_DIAG_GAP = 230.0
# Cyclone: _SPIRAL_ARMS spiral arms pinwheeling out of an open eye. The eye
# radius is what lets the T inside (2*r - 2*WALL_RADIUS must beat the T's 120
# AABB); the arm count and sweep set how wide the channels between arms are.
_SPIRAL_EYE_R = 80.0
_SPIRAL_ARMS = 2
_SPIRAL_ARM_SWEEP_DEG = 220.0   # how far each arm curls before the wall
_SPIRAL_TAIL_EASE = 1.0         # 1.0 => Archimedean (radius linear in angle)
_SPIRAL_TAIL_REACH = 420.0      # target radius; > max corner distance (362)


def _edge_wall_levels() -> list[list[Segment]]:
    """One straight stick anchored to the arena boundary, free at the far end.

    Four axis-aligned placements (in from each edge), then the same stick
    rotated 45 degrees -- run out of a corner along each diagonal.
    """
    far = _W - _EDGE_WALL_LEN  # 182 -> bypass gap at the free end
    d = _EDGE_WALL_DIAG_LEN
    return [
        _wall((0.0, _EDGE_WALL_MID), (_EDGE_WALL_LEN, _EDGE_WALL_MID)),  # from left
        _wall((_W, _EDGE_WALL_MID), (far, _EDGE_WALL_MID)),              # from right
        _wall((_EDGE_WALL_MID, 0.0), (_EDGE_WALL_MID, _EDGE_WALL_LEN)),  # from top
        _wall((_EDGE_WALL_MID, _W), (_EDGE_WALL_MID, far)),              # from bottom
        _wall((0.0, 0.0), (d, d)),                                       # from top-left
        _wall((_W, 0.0), (_W - d, d)),                                   # from top-right
    ]


def _chicane_levels() -> list[list[Segment]]:
    """Two staggered walls 200 apart with openings on opposite sides.

    Horizontal pair, its left-right mirror, then the same thing transposed
    into a vertical slalom and mirrored again.
    """
    near, far = 156.0, 356.0     # the two wall lines
    a, b = 300.0, 212.0          # where each wall stops -> 212-wide openings
    return [
        # horizontal: top wall from the left, bottom wall from the right
        [((0.0, near), (a, near)), ((b, far), (_W, far))],
        # horizontal mirrored in x
        [((_W, near), (_W - a, near)), ((0.0, far), (_W - b, far))],
        # vertical: left wall from the top, right wall from the bottom
        [((near, 0.0), (near, a)), ((far, b), (far, _W))],
        # vertical mirrored in y
        [((near, _W), (near, _W - a)), ((far, 0.0), (far, _W - b))],
    ]


def _diagonal_levels() -> list[list[Segment]]:
    """Diagonal chicane: two parallel diagonal walls, staggered.

    The diagonal counterpart of :func:`_chicane_levels`. Each wall is
    anchored to the arena boundary at one end and stops short at the other,
    and the two stop short at *opposite* ends — so the T has to slip round
    one free tip, run up the channel between the walls, then round the
    other tip. A diagonal zig-zag rather than a single slash.
    """
    out: list[list[Segment]] = []
    for dy in (1.0, -1.0):          # main diagonal, then anti-diagonal
        for flip in (False, True):  # which end of each wall is anchored
            d = (_INV_SQRT2, dy * _INV_SQRT2)
            perp = (-d[1], d[0])
            walls: list[Segment] = []
            for i, off in enumerate((_DIAG_OFFSET, -_DIAG_OFFSET)):
                base = (256.0 + off * perp[0], 256.0 + off * perp[1])
                chord = _clip_ray_to_arena(base, d)
                if chord is None:  # pragma: no cover - offsets keep us inside
                    continue
                t0, t1 = chord
                # Anchor wall 0 at the low-t end and wall 1 at the high-t
                # end (swapped when ``flip``), leaving _DIAG_GAP free at the
                # opposite end.
                at_low = (i == 0) != flip
                if at_low:
                    a, b = t0, t1 - _DIAG_GAP
                else:
                    a, b = t0 + _DIAG_GAP, t1
                walls.append(
                    (
                        (base[0] + a * d[0], base[1] + a * d[1]),
                        (base[0] + b * d[0], base[1] + b * d[1]),
                    )
                )
            out.append(walls)
    return out


def _pocket_levels() -> list[list[Segment]]:
    """U pocket in the arena centre with one flank run out to the wall.

    170-wide mouth, 130 deep, sized off the worst case (the 45-degree
    rotations, whose corners swing toward the arena corners). One flank keeps
    going past the mouth until it meets the arena boundary, so the pocket is
    anchored to the wall rather than free-standing -- the long flank splits
    the surrounding space and you have to work round the closed end of the U.

    Mouth on 8 headings.
    """
    import math

    return [
        _u_pocket(
            256.0,
            256.0,
            width=170.0,
            depth=130.0,
            theta=k * math.pi / 4.0,
            extend_flank=True,
        )
        for k in range(8)
    ]


def _spiral_tail(
    cx: float,
    cy: float,
    r0: float,
    theta0: float,
    sweep: float,
    n: int = 20,
) -> list[Segment]:
    """One spiral arm: starts at radius ``r0`` and curls outward to the wall.

    The radius grows as ``f ** _SPIRAL_TAIL_EASE`` over the sweep, so ease
    1.0 gives a plain Archimedean arm and larger values make it hug ``r0``
    before swinging out. It stops the moment it touches the arena boundary.
    """
    import math

    pts: list[tuple[float, float]] = []
    for i in range(n + 1):
        f = i / n
        th = theta0 + sweep * f
        # Target radius is a constant reach, NOT the local wall distance --
        # tracking the wall would kink the curve every time it sweeps past
        # an arena corner. Clip to the wall instead and stop there.
        r = r0 + (_SPIRAL_TAIL_REACH - r0) * (f ** _SPIRAL_TAIL_EASE)
        r_wall = _boundary_radius(cx, cy, th)
        hit = r >= r_wall
        r = min(r, r_wall)
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))
        if hit:
            break
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _spiral_levels() -> list[list[Segment]]:
    """Cyclone: spiral arms pinwheeling out of an open eye.

    Two Archimedean arms, 180 degrees apart, each starting at the rim of the
    eye and curling 240 degrees outward until it meets the arena wall.

    The eye is 162 of clear space across, so the T's 120 AABB fits inside it,
    and the channels between the arms are wide enough to push the T in and
    out. Neither arm closes anything off: each is a curve with one free end,
    so you can always round it -- but because both free ends are at the eye,
    crossing from one side of an arm to the other means going through the
    middle. The eye is the hub of the level.

    Eight variants: the pinwheel rotated in 22.5-degree steps. A two-arm
    pinwheel maps onto itself under a 180-degree turn, so its distinct
    orientations live in a half-turn -- 45-degree steps would only give four
    layouts and repeat. (Mirroring instead of rotating also gives eight
    distinct point sets, but a mirrored pinwheel reads as the same picture
    flipped, so rotation is what actually buys eight different levels.)

    Note the arms cannot be made to *wrap* into a tight multi-turn coil at
    this scale. A wrapping spiral has to fit eye + lane + wall + outer lane
    inside a 256 radius, which caps the lane at ~99 against the T's 120 --
    the T would be locked out of its own eye. That needs a bigger arena.
    """
    import math

    out: list[list[Segment]] = []
    # Distinct orientations span 2*pi / _SPIRAL_ARMS (the pinwheel's own
    # rotational symmetry), split into 8 steps.
    step = 2.0 * math.pi / _SPIRAL_ARMS / 8.0
    for k in range(8):
        phi = k * step
        segs: list[Segment] = []
        for j in range(_SPIRAL_ARMS):
            segs += _spiral_tail(
                256.0,
                256.0,
                _SPIRAL_EYE_R,
                phi + j * 2.0 * math.pi / _SPIRAL_ARMS,
                sweep=math.radians(_SPIRAL_ARM_SWEEP_DEG),
                n=48,
            )
        out.append(segs)
    return out


def _sketch_family_levels() -> dict[int, list[Segment]]:
    """Levels 1..30, in family order."""
    families = (
        _edge_wall_levels()
        + _chicane_levels()
        + _diagonal_levels()
        + _pocket_levels()
        + _spiral_levels()
    )
    assert len(families) == 30, len(families)
    return {1 + i: segs for i, segs in enumerate(families)}


# Human-readable family name per level in 1..30, for plots and logs.
SKETCH_FAMILY_NAMES: dict[int, str] = {
    **{1 + i: "edge_wall" for i in range(6)},
    **{7 + i: "chicane" for i in range(4)},
    **{11 + i: "diagonal" for i in range(4)},
    **{15 + i: "pocket" for i in range(8)},
    **{23 + i: "spiral" for i in range(8)},
}


# Level 0 is the empty arena; 1..30 are the sketch families above.
OBSTACLE_LEVELS: dict[int, list[Segment]] = {
    0: [],
    **_sketch_family_levels(),
}


def build_obstacles(space: pymunk.Space, level: int) -> list[pymunk.Segment]:
    """Add the configured obstacles for `level` to `space` and return them."""
    if level not in OBSTACLE_LEVELS:
        raise ValueError(
            f"unknown obstacle_level {level}, valid: {sorted(OBSTACLE_LEVELS)}"
        )
    return build_obstacle_segments(space, OBSTACLE_LEVELS[level])


def build_obstacle_segments(
    space: pymunk.Space,
    obstacle_segments: Iterable[Segment],
) -> list[pymunk.Segment]:
    """Add explicit obstacle geometry to ``space``.

    Replays use this path because an episode's recorded segments remain the
    source of truth even if the named obstacle-level catalog later changes.
    """
    segments = [
        pymunk.Segment(space.static_body, a, b, WALL_RADIUS)
        for a, b in obstacle_segments
    ]
    for seg in segments:
        seg.friction = WALL_FRICTION
    if segments:
        space.add(*segments)
    return segments


def all_levels() -> Iterable[int]:
    return sorted(OBSTACLE_LEVELS)
