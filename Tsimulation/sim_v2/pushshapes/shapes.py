"""Geometry factories for pushable objects and pusher tools.

Each pushable shape is a list of axis-aligned rectangles `(cx, cy, w, h)` in
body-local coords. To add a new shape, append an entry to ``SHAPES``::

    SHAPES["L"] = [
        (0, -30, 30, 120),   # vertical stem
        (30, 30, 90, 30),    # horizontal foot
    ]

------------------------------------------------------------------------
SHAPE FIDELITY NOTE — important when adding/rotating shapes
------------------------------------------------------------------------
Shapes are decomposed into axis-aligned rectangles for stable pymunk
contacts (slanted polys with low mass produce unstable contact normals
and jittery integration). The trade-off:

  * Symmetric shapes (T, U) rotate cleanly — the AABB-rect union is the
    true geometry.
  * The "Z" shape here approximates a Z with three axis-aligned blocks
    (top bar, middle joint, bottom bar). Under non-zero rotation the
    visible silhouette and IoU still match what is rendered (render.py
    draws the same rects), but the result does NOT look like a real Z
    rotated by `theta` — it looks like three rectangles rotated by
    `theta`. If you need a true rotated-Z silhouette, model it as a
    single rotated polygon and accept the contact-stability cost.
"""

from __future__ import annotations

from typing import Literal

import math

import pymunk

SHAPES: dict[str, list[tuple[float, float, float, float]]] = {
    # gym-pusht's canonical T: 120x30 top bar + 30x90 stem below it.
    "T": [
        (0.0, -30.0, 120.0, 30.0),
        (0.0, 30.0, 30.0, 90.0),
    ],
    # U opening upward: two vertical legs + a bottom bar.
    "U": [
        (-45.0, 0.0, 30.0, 120.0),
        (45.0, 0.0, 30.0, 120.0),
        (0.0, 60.0, 120.0, 30.0),
    ],
    # APPROXIMATE Z — see "SHAPE FIDELITY NOTE" in the module docstring.
    # Three axis-aligned blocks stacked diagonally; not a true Z under rotation.
    "Z": [
        (-15.0, -30.0, 90.0, 30.0),
        (0.0, 0.0, 30.0, 30.0),
        (15.0, 30.0, 90.0, 30.0),
    ],
}

OBJECT_DENSITY = 0.30
OBJECT_FRICTION = 0.6
PUSHER_RADIUS = 15.0
PUSHER_RADIUS_SMALL = 5.0  # circle_small: 3x smaller than the standard circle
STICK_HALF_LEN = 30.0
STICK_HALF_THICK = 5.0

# T-stem socket pusher. Its local +X axis points through the open end, so an
# oriented controller can aim the socket simply by rotating +X toward travel.
# The 32-unit opening leaves 1 unit of clearance on either side of the
# standard T's 30-unit stem.
U_SOCKET_INNER_GAP = 32.0
U_SOCKET_PRONG_THICK = 10.0
U_SOCKET_PRONG_LENGTH = 30.0
U_SOCKET_CROSSBAR_THICK = 10.0
U_SOCKET_OUTER_WIDTH = U_SOCKET_INNER_GAP + 2 * U_SOCKET_PRONG_THICK
U_SOCKET_CROSSBAR_INNER_X = (
    -U_SOCKET_PRONG_LENGTH / 2 + U_SOCKET_CROSSBAR_THICK / 2
)
U_SOCKET_RECTS: list[tuple[float, float, float, float]] = [
    (
        5.0,
        -(U_SOCKET_INNER_GAP + U_SOCKET_PRONG_THICK) / 2,
        U_SOCKET_PRONG_LENGTH,
        U_SOCKET_PRONG_THICK,
    ),
    (
        5.0,
        (U_SOCKET_INNER_GAP + U_SOCKET_PRONG_THICK) / 2,
        U_SOCKET_PRONG_LENGTH,
        U_SOCKET_PRONG_THICK,
    ),
    (
        -U_SOCKET_PRONG_LENGTH / 2,
        0.0,
        U_SOCKET_CROSSBAR_THICK,
        U_SOCKET_OUTER_WIDTH,
    ),
]

# Pocket interior in socket-local coords -- the open region bounded by the
# crossbar's inner face (x_min) and the two prong tips (x_max), spanning the
# inner gap in y. pymunk friction is per-shape rather than per-face, so this
# rectangle is what lets a contact be classified as inside vs outside.
U_SOCKET_POCKET_X_MIN = U_SOCKET_CROSSBAR_INNER_X
U_SOCKET_POCKET_X_MAX = max(cx + w / 2 for cx, _cy, w, _h in U_SOCKET_RECTS[:2])
U_SOCKET_POCKET_Y_HALF = U_SOCKET_INNER_GAP / 2

# L pusher: two axis-aligned rects sharing a corner. Body origin sits at the
# geometric centroid so pymunk's rotation-around-CoG matches the visual pivot.
# Rect centers are the closed-form centroid-shifted positions:
#   vertical stem @ ((t-L)/4, (t-L)/4), dims (t, L+t)
#   horizontal foot @ ((L-t)/4, (L+t)/4), dims (L, t)
L_ARM = 45.0
L_THICK = 15.0
L_RECTS: list[tuple[float, float, float, float]] = [
    ((L_THICK - L_ARM) / 4, (L_THICK - L_ARM) / 4, L_THICK, L_ARM + L_THICK),
    ((L_ARM - L_THICK) / 4, (L_ARM + L_THICK) / 4, L_ARM, L_THICK),
]

# --- geometry for the behaviourally-distinct agents (see agents.py) -------
# Kept here with the other shape constants so make_pusher stays the one place
# that knows how a body is built.

GRIPPER_PALM_HALF_W = 22.0   # palm spans the jaw travel
GRIPPER_PALM_HALF_H = 6.0
GRIPPER_JAW_HALF_W = 5.0
GRIPPER_JAW_HALF_H = 20.0
GRIPPER_JAW_MAX_GAP = 46.0   # fully open, outer face to outer face
GRIPPER_JAW_MIN_GAP = 8.0    # fully closed

# Each end effector gets its OWN silhouette. The first version made suction,
# two_point, tether, magnet and compliant plain circles differing only in
# radius -- compliant was r=15.0, identical to `circle`. The reasoning was
# that behaviour lives in the agent, not the geometry, which is wrong twice
# over: a teleoperator cannot tell them apart, and a vision policy cannot
# identify which embodiment it is driving from pixels that match.

# Suction: pad on a stem, so the contact face reads as a flat disc.
SUCTION_PAD_HALF_W = 13.0
SUCTION_PAD_HALF_H = 3.5
SUCTION_STEM_HALF_W = 3.0
SUCTION_STEM_HALF_H = 9.0
SUCTION_RADIUS = 13.0

TWO_POINT_RADIUS = 6.0

# Tether: an open hook, not a disc -- it should look like something a rope
# attaches to.
TETHER_HOOK_R = 9.0
TETHER_HOOK_THICK = 3.0
TETHER_RADIUS = 9.0

# Magnet: horseshoe. Two poles and a yoke, opening forward.
MAGNET_HALF_W = 13.0
MAGNET_POLE_HALF_H = 11.0
MAGNET_POLE_THICK = 4.5
MAGNET_RADIUS = 14.0

# Compliant: a hollow ring, reading as springy rather than solid.
COMPLIANT_R = 15.0
COMPLIANT_THICK = 4.0

TAPPER_HALF_LEN = 14.0
TAPPER_HALF_THICK = 4.0

# --- new end effectors ---------------------------------------------------
# Rake: a spine with four prongs. Wide sweep, but the gaps mean it cannot
# apply a concentrated force -- thin features slip between the teeth.
RAKE_SPINE_HALF_W = 20.0
RAKE_SPINE_HALF_H = 3.0
RAKE_TOOTH_HALF_W = 2.5
RAKE_TOOTH_HALF_H = 9.0
RAKE_TOOTH_XS = (-15.0, -5.0, 5.0, 15.0)

# Roller: a wide barrel that spins about its own axis, so it drives the object
# by rolling friction rather than by pushing a face into it.
ROLLER_HALF_W = 4.0
ROLLER_HALF_H = 18.0

# Scoop: a concave arc that cradles the object -- carries without grasping,
# but only while the opening stays roughly upright relative to travel.
SCOOP_R = 20.0
SCOOP_THICK = 4.0
SCOOP_SEGMENTS = 5

# Tow bar: a hitch ball. The bar itself is a PinJoint, drawn as a line.
TOWBAR_R = 9.0
TOWBAR_LENGTH = 70.0

# Wrench: an open spanner head. Rotational intent should be legible at a
# glance -- it is the only agent whose whole job is angle.
WRENCH_R = 15.0
WRENCH_THICK = 5.0
WRENCH_OPENING = 1.15   # radians of missing arc (the jaw gap)

# Soft body: a deformable pad. Unlike every other end effector this one is not
# a rigid outline -- SOFT_NODES dynamic discs are sprung to a kinematic root,
# so the contact face CHANGES SHAPE against the object instead of transmitting
# force through a fixed geometry.
SOFT_NODES = 7
# Span must leave the nodes non-overlapping: 34/6 = 5.7 spacing against a
# 12-wide node meant every disc was embedded in its neighbours, and the
# contact solver blew the pad apart -- 51.8 units of "deformation" with
# nothing touching it. Nodes also get a shared collision group so they
# cannot push each other at all; only the object may deform the pad.
SOFT_SPAN = 60.0
SOFT_NODE_R = 5.0
# Tuned by sweep, not guessed. The pad has to do two things at once and the
# window is narrow, and NOT monotonic -- tuning by intuition would miss it:
#   2e4  deflect 10.2  push 181   <- chosen
#   5e4  deflect 20.1  push  24   (too soft; also pushes WORSE than 2e4)
#   1e5  deflect  1.0  push 206   (rigid bar with extra steps)
#   2e5  deflect  0.9  push 101
# Measured through the real constructor path. An earlier sweep that
# monkeypatched these module globals reported 7.9 deflection at 2e5, which
# the actual code path then produced 0.49 for -- patch the object, not the
# module, when tuning.
SOFT_STIFFNESS = 2.0e4
# 2.5x CRITICAL, deliberately over-damped. The trial that produced the chosen
# (deflect 10.2 / push 181) ran at this value; recomputing it to 0.8x critical
# for the new stiffness -- the "principled" move -- turned the same pad into
# deflect 18.0 / push 30. Over-damping is what lets the pad deform without
# bouncing, so it stiffens under load and still transmits force.
SOFT_DAMPING = 1.43e3
SOFT_NODE_MASS = 4.0

# Per-shape effective pusher radius — used by env spawn-clearance and renderer.
# Stick uses its end-cap radius (the largest contact circle on its body).
_PUSHER_RADII: dict[str, float] = {
    "circle": PUSHER_RADIUS,
    "circle_small": PUSHER_RADIUS_SMALL,
    "stick": STICK_HALF_THICK,
    "L": L_THICK / 2.0,
    "u_socket": (
        (U_SOCKET_PRONG_LENGTH / 2 + U_SOCKET_CROSSBAR_THICK) ** 2
        + (U_SOCKET_OUTER_WIDTH / 2) ** 2
    )
    ** 0.5,
    # Spawn clearance uses the widest extent the body can reach, so the
    # gripper reports its OPEN half-span rather than its palm.
    "gripper": (GRIPPER_JAW_MAX_GAP / 2 + 2 * GRIPPER_JAW_HALF_W),
    "suction": SUCTION_RADIUS,
    "two_point": TWO_POINT_RADIUS,
    "tether": TETHER_RADIUS,
    "magnet": MAGNET_RADIUS,
    "compliant": COMPLIANT_R,
    "tapper": TAPPER_HALF_LEN,
    "rake": RAKE_SPINE_HALF_W,
    "roller": ROLLER_HALF_H,
    "scoop": SCOOP_R,
    "soft": SOFT_SPAN / 2 + SOFT_NODE_R,
    "wrench": WRENCH_R,
    "towbar": TOWBAR_R,
    "compliant": COMPLIANT_R,
}


def pusher_radius(shape: str) -> float:
    """Effective contact radius for ``shape``. Raises on unknown shapes."""
    if shape not in _PUSHER_RADII:
        raise ValueError(
            f"unknown pusher shape '{shape}', valid: {list(_PUSHER_RADII)}"
        )
    return _PUSHER_RADII[shape]


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
    """Create a dynamic body composed of the shape's rectangles."""
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
    shape: Literal["circle", "circle_small", "stick", "L", "u_socket"],
    space: pymunk.Space,
    position: tuple[float, float],
) -> tuple[pymunk.Body, list[pymunk.Shape]]:
    """Create a KINEMATIC pusher whose position/velocity is driven by env.step().

    Kinematic means infinite mass and no contact response, so the pusher is
    never deflected by the object -- that is deliberate. Keeping it out of
    walls is handled separately by ``PushShapesEnv._clamp_pusher_to_static``
    so that free-space motion stays byte-identical to the original sim.
    """
    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    body.position = position

    if shape in ("circle", "circle_small"):
        s = pymunk.Circle(body, pusher_radius(shape))
        s.friction = OBJECT_FRICTION
        space.add(body, s)
        return body, [s]

    if shape == "stick":
        # Capsule = rectangle + two end-cap circles, so the stick has a
        # smooth contact profile at its tips instead of sharp corners.
        rect = pymunk.Poly(
            body,
            _rect_verts(0.0, 0.0, 2 * STICK_HALF_LEN, 2 * STICK_HALF_THICK),
        )
        end_a = pymunk.Circle(body, STICK_HALF_THICK, offset=(-STICK_HALF_LEN, 0.0))
        end_b = pymunk.Circle(body, STICK_HALF_THICK, offset=(STICK_HALF_LEN, 0.0))
        for s in (rect, end_a, end_b):
            s.friction = OBJECT_FRICTION
        space.add(body, rect, end_a, end_b)
        return body, [rect, end_a, end_b]

    if shape == "L":
        polys = [pymunk.Poly(body, _rect_verts(*r)) for r in L_RECTS]
        for p in polys:
            p.friction = OBJECT_FRICTION
        space.add(body, *polys)
        return body, list(polys)

    if shape == "u_socket":
        polys = [pymunk.Poly(body, _rect_verts(*r)) for r in U_SOCKET_RECTS]
        for p in polys:
            p.friction = OBJECT_FRICTION
        space.add(body, *polys)
        return body, list(polys)

    if shape == "magnet":
        # NON-CONTACT (sensor=True): a magnet that still collides is just a
        # circle pusher with extra force, which defeats the embodiment -- its
        # whole character is that there is no surface to push against.
        # Horseshoe silhouette so it is not yet another disc on screen.
        yoke = pymunk.Poly(body, _rect_verts(
            -MAGNET_POLE_HALF_H, 0.0,
            2 * MAGNET_POLE_THICK, 2 * MAGNET_HALF_W))
        poles = [
            pymunk.Poly(body, _rect_verts(
                0.0, sign * (MAGNET_HALF_W - MAGNET_POLE_THICK),
                2 * MAGNET_POLE_HALF_H, 2 * MAGNET_POLE_THICK))
            for sign in (-1.0, 1.0)
        ]
        parts = [yoke, *poles]
        for x in parts:
            x.sensor = True
        space.add(body, *parts)
        return body, parts

    if shape == "two_point":
        # Deliberately the only round one left: it reads as a PAIR of small
        # points (the agent owns the second body), which is its signature.
        s = pymunk.Circle(body, TWO_POINT_RADIUS)
        s.friction = OBJECT_FRICTION
        space.add(body, s)
        return body, [s]

    if shape == "suction":
        pad = pymunk.Poly(body, _rect_verts(
            0.0, 0.0, 2 * SUCTION_PAD_HALF_W, 2 * SUCTION_PAD_HALF_H))
        stem = pymunk.Poly(body, _rect_verts(
            0.0, -SUCTION_PAD_HALF_H - SUCTION_STEM_HALF_H,
            2 * SUCTION_STEM_HALF_W, 2 * SUCTION_STEM_HALF_H))
        for x in (pad, stem):
            x.friction = OBJECT_FRICTION
        space.add(body, pad, stem)
        return body, [pad, stem]

    if shape == "tether":
        # Open hook: an arc of segments with a gap, so it is visibly not a disc.
        parts = []
        for i in range(6):
            a0 = math.pi * 0.25 + i * (math.pi * 1.5 / 6)
            a1 = a0 + (math.pi * 1.5 / 6) * 0.85
            seg = pymunk.Segment(
                body,
                (TETHER_HOOK_R * math.cos(a0), TETHER_HOOK_R * math.sin(a0)),
                (TETHER_HOOK_R * math.cos(a1), TETHER_HOOK_R * math.sin(a1)),
                TETHER_HOOK_THICK / 2,
            )
            seg.friction = OBJECT_FRICTION
            parts.append(seg)
        space.add(body, *parts)
        return body, parts

    if shape == "compliant":
        # Hollow ring -- same footprint as `circle` on purpose (it is the
        # control for contact AUTHORITY) but unmistakable on screen.
        parts = []
        for i in range(8):
            a0 = i * (2 * math.pi / 8)
            a1 = a0 + (2 * math.pi / 8) * 0.9
            seg = pymunk.Segment(
                body,
                (COMPLIANT_R * math.cos(a0), COMPLIANT_R * math.sin(a0)),
                (COMPLIANT_R * math.cos(a1), COMPLIANT_R * math.sin(a1)),
                COMPLIANT_THICK / 2,
            )
            seg.friction = OBJECT_FRICTION
            parts.append(seg)
        space.add(body, *parts)
        return body, parts

    if shape == "rake":
        parts = [pymunk.Poly(body, _rect_verts(
            0.0, 0.0, 2 * RAKE_SPINE_HALF_W, 2 * RAKE_SPINE_HALF_H))]
        for tx in RAKE_TOOTH_XS:
            parts.append(pymunk.Poly(body, _rect_verts(
                tx, RAKE_SPINE_HALF_H + RAKE_TOOTH_HALF_H,
                2 * RAKE_TOOTH_HALF_W, 2 * RAKE_TOOTH_HALF_H)))
        for x in parts:
            x.friction = OBJECT_FRICTION
        space.add(body, *parts)
        return body, parts

    if shape == "towbar":
        sh = pymunk.Circle(body, TOWBAR_R)
        sh.friction = OBJECT_FRICTION
        space.add(body, sh)
        return body, [sh]

    if shape == "wrench":
        # Sensor: the wrench acts through a GearJoint, never through contact.
        # A colliding wrench would just be another pusher.
        parts = []
        n = 7
        span = 2 * math.pi - WRENCH_OPENING
        for i in range(n):
            a0 = WRENCH_OPENING / 2 + i * (span / n)
            a1 = a0 + (span / n) * 0.92
            seg = pymunk.Segment(
                body,
                (WRENCH_R * math.cos(a0), WRENCH_R * math.sin(a0)),
                (WRENCH_R * math.cos(a1), WRENCH_R * math.sin(a1)),
                WRENCH_THICK / 2)
            parts.append(seg)
        space.add(body, *parts)
        return body, parts

    if shape == "soft":
        # Root only -- the deformable nodes are dynamic bodies owned by
        # SoftBodyAgent, because the env drives exactly one kinematic body and
        # these must be free to be pushed OUT OF SHAPE by contact.
        root = pymunk.Circle(body, 3.0)
        root.sensor = True
        space.add(body, root)
        return body, [root]

    if shape == "roller":
        barrel = pymunk.Poly(body, _rect_verts(
            0.0, 0.0, 2 * ROLLER_HALF_W, 2 * ROLLER_HALF_H))
        # High friction: the roller works by gripping and rolling, so a
        # slippery barrel would just slide past the object.
        barrel.friction = 2.0
        space.add(body, barrel)
        return body, [barrel]

    if shape == "scoop":
        parts = []
        span = math.pi * 1.1
        for i in range(SCOOP_SEGMENTS):
            a0 = -span / 2 + i * (span / SCOOP_SEGMENTS)
            a1 = a0 + span / SCOOP_SEGMENTS
            seg = pymunk.Segment(
                body,
                (SCOOP_R * math.cos(a0), SCOOP_R * math.sin(a0)),
                (SCOOP_R * math.cos(a1), SCOOP_R * math.sin(a1)),
                SCOOP_THICK / 2,
            )
            seg.friction = OBJECT_FRICTION
            parts.append(seg)
        space.add(body, *parts)
        return body, parts

    if shape == "tapper":
        # A short bar that acts ONLY through the impulses its agent applies.
        # sensor=True for the same reason as the magnet: if the bar also
        # pushed continuously, the ballistic character -- the entire point --
        # would be replaced by ordinary pushing between strikes.
        rect = pymunk.Poly(
            body, _rect_verts(0.0, 0.0, 2 * TAPPER_HALF_LEN, 2 * TAPPER_HALF_THICK)
        )
        rect.sensor = True
        space.add(body, rect)
        return body, [rect]

    if shape == "gripper":
        # Palm only. The two jaws are separate bodies owned by GripperAgent,
        # because their gap is commanded per-step and the env drives exactly
        # one body.
        palm = pymunk.Poly(
            body,
            _rect_verts(0.0, 0.0, 2 * GRIPPER_PALM_HALF_W, 2 * GRIPPER_PALM_HALF_H),
        )
        palm.friction = OBJECT_FRICTION
        space.add(body, palm)
        return body, [palm]

    raise ValueError(
        f"unknown pusher shape '{shape}', valid: {list(_PUSHER_RADII)}"
    )


def aabb(shape: str) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box `(xmin, ymin, xmax, ymax)` of the shape in
    its rest pose. Used for rejection-sampling spawn positions."""
    rects = SHAPES[shape]
    xs_min = [cx - w / 2 for cx, _cy, w, _h in rects]
    xs_max = [cx + w / 2 for cx, _cy, w, _h in rects]
    ys_min = [cy - h / 2 for _cx, cy, _w, h in rects]
    ys_max = [cy + h / 2 for _cx, cy, _w, h in rects]
    return (min(xs_min), min(ys_min), max(xs_max), max(ys_max))
