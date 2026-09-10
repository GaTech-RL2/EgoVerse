"""Engage-then-transport controller for PushShapes SE(2) manipulation.

Replaces pose_controller.PosePushController, which had three defects that made
its entire corpus unusable for anything above two DOF:

  * it never wrote the grip channel, so every 4-DOF agent collected with the
    jaws permanently OPEN -- a "grasp" dataset in which nothing is ever grasped;
  * the collector set ``a[2] = current agent_angle``, so the commanded angle was
    bit-for-bit constant and no oriented agent ever aimed at anything;
  * it emitted absolute waypoints with steps up to 200 world units, giving a
    mean jerk 1.8x the mean speed -- the commanded direction reversed more than
    once per step on average.

The task here is ENGAGE THEN TRANSPORT: capture the T by its stem and carry it
to the goal while held. Pushing is not a success.

GEOMETRY (from shapes.py, not guessed)
  T object   : bar (0,-30,120x30), stem (0,30,30x90) in object-local coords, so
               the stem runs along local +y and its tip is at local (0, 75).
  u_socket   : local +X points through the open end. Pocket interior spans
               local x in [-10, 20], |y| <= 16. The 32-unit gap takes the
               30-unit stem with 1 unit of clearance a side.
  grippers   : jaw gap 8 (closed) to 58 (open); the 30-unit stem fits.

APPROACH FRAME
  With the socket's opening (+X) pointing back down the stem, a socket placed
  at ``tip + d * stem_dir`` puts the stem tip exactly ``d`` units inside the
  pocket. Engaging is therefore just driving ``d`` from a standoff down to
  ``SEAT_DEPTH`` along ``-stem_dir``; no search is required.
"""

from __future__ import annotations

import math

import numpy as np

# Where the stem tip should sit inside the pocket (socket-local +X units).
SEAT_DEPTH = 5.0
# Standoff before insertion, along the stem axis beyond the tip.
STANDOFF = 55.0
# Alignment gates before we commit to inserting.
ALIGN_ANG_TOL = 0.12          # rad
ALIGN_LAT_TOL = 7.0           # world units off the stem axis
ARENA_MARGIN = 22.0           # keep approach points clear of the walls
# Routing radius around the object. A straight line to a pre-engage point on the
# far side of the T drives the agent INTO the T and it stalls there forever --
# the defect that made the first version of this controller engage 0/10. Route
# around the object on this circle until the approach ray is clear.
ORBIT_R = 118.0
ORBIT_STEP = 0.28             # rad per step around the object
# Rate limits -- the whole point of the rewrite. A commanded pose that moves
# more than this per step is what produced the old jerk.
MAX_DPOS = 6.0                # world units per step
MAX_DANG = 0.10               # rad per step
# Grip ramp. Jaws travel 50 units at 0.25/substep = 200 substeps = ~10 frames,
# so a single-step toggle would command a closure the physics cannot follow.
GRIP_RAMP = 0.10
TRANSPORT_LEAD = 40.0         # how far ahead on the goal path to aim
JAM_STEPS = 90
JAM_MOVE = 3.0


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


class EngageController:
    """Emits a full (x, y, angle, grip) command, rate-limited every step."""

    #: object-local stem tip and stem direction for the canonical T
    STEM_TIP_LOCAL = np.array([0.0, 75.0])
    STEM_DIR_LOCAL = np.array([0.0, 1.0])

    def __init__(self, world_size, *, needs_grip: bool, rng=None):
        self.world = float(world_size)
        self.needs_grip = bool(needs_grip)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.state = "ALIGN"
        self.grip = 0.0
        self._cmd_pos = None
        self._cmd_ang = None
        self._since_progress = 0
        self._last_obj = None

    # ---------------------------------------------------------------- frames
    def _stem(self, object_pose):
        """World-frame stem tip and unit stem direction (base -> tip)."""
        c = np.asarray(object_pose[:2], dtype=np.float64)
        th = float(object_pose[2])
        R = _rot(th)
        tip = c + R @ self.STEM_TIP_LOCAL
        d = R @ self.STEM_DIR_LOCAL
        return tip, d / max(np.linalg.norm(d), 1e-9)

    def _in_arena(self, p) -> bool:
        return bool(
            ARENA_MARGIN <= p[0] <= self.world - ARENA_MARGIN
            and ARENA_MARGIN <= p[1] <= self.world - ARENA_MARGIN
        )

    def _target_for_depth(self, object_pose, depth):
        """Pose that places the stem tip ``depth`` units inside the pocket.

        The approach runs along the stem axis, so a stem pointing at a wall puts
        the standoff outside the arena. Commanding that point clips against the
        boundary and the agent grinds into the wall forever, never reaching the
        axis. Shorten the standoff until the point fits instead.
        """
        tip, sdir = self._stem(object_pose)
        ang = math.atan2(-sdir[1], -sdir[0])   # opening (+X) faces back down the stem
        d = float(depth)
        pos = tip + d * sdir
        while d > SEAT_DEPTH and not self._in_arena(pos):
            d -= 5.0
            pos = tip + d * sdir
        return np.clip(pos, ARENA_MARGIN, self.world - ARENA_MARGIN), ang

    # ----------------------------------------------------------- rate limiter
    def _smooth(self, pos, ang):
        if self._cmd_pos is None:
            self._cmd_pos = np.asarray(pos, dtype=np.float64).copy()
            self._cmd_ang = float(ang)
            return self._cmd_pos, self._cmd_ang
        dp = np.asarray(pos, dtype=np.float64) - self._cmd_pos
        n = float(np.linalg.norm(dp))
        if n > MAX_DPOS:
            dp *= MAX_DPOS / n
        self._cmd_pos = np.clip(self._cmd_pos + dp, 0.0, self.world)
        da = _wrap(float(ang) - self._cmd_ang)
        self._cmd_ang = _wrap(self._cmd_ang + float(np.clip(da, -MAX_DANG, MAX_DANG)))
        return self._cmd_pos, self._cmd_ang

    def _emit(self, pos, ang, grip):
        """Rate-limit and flatten to the (x, y, angle, grip) the env expects."""
        p, a = self._smooth(pos, ang)
        return float(p[0]), float(p[1]), float(a), float(grip)

    # ---------------------------------------------------------------- routing
    def _blocked(self, a, target, obj_c):
        """True if the straight segment a->target passes inside ORBIT_R."""
        d = target - a
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return False
        t = float(np.clip(np.dot(obj_c - a, d) / (n * n), 0.0, 1.0))
        closest = a + t * d
        return bool(np.linalg.norm(closest - obj_c) < ORBIT_R * 0.82)

    def _route(self, a, target, obj_c):
        """Waypoint that reaches ``target`` without crossing the object."""
        if not self._blocked(a, target, obj_c):
            return target
        va, vt = a - obj_c, target - obj_c
        aa = math.atan2(va[1], va[0])
        at = math.atan2(vt[1], vt[0])
        step = ORBIT_STEP if _wrap(at - aa) > 0 else -ORBIT_STEP
        ang = aa + step
        p = obj_c + ORBIT_R * np.array([math.cos(ang), math.sin(ang)])
        return np.clip(p, ARENA_MARGIN, self.world - ARENA_MARGIN)

    # ------------------------------------------------------------------ step
    def __call__(self, agent_pos, agent_angle, object_pose, goal_pose, engaged):
        a = np.asarray(agent_pos, dtype=np.float64)
        ang = float(np.asarray(agent_angle).reshape(-1)[0])
        obj = np.asarray(object_pose, dtype=np.float64)
        goal = np.asarray(goal_pose, dtype=np.float64)

        if self._last_obj is None or np.linalg.norm(obj[:2] - self._last_obj[:2]) > JAM_MOVE:
            self._since_progress = 0
            self._last_obj = obj.copy()
        else:
            self._since_progress += 1

        if engaged:
            self.state = "TRANSPORT"
        elif self.state == "TRANSPORT":
            self.state = "ALIGN"          # lost it; re-approach

        if self.state == "TRANSPORT":
            # Latched/held: drive the OBJECT toward the goal. The commanded
            # agent pose is the engage pose of the object's DESIRED next pose,
            # so position and orientation are corrected together.
            to_goal = goal[:2] - obj[:2]
            dist = float(np.linalg.norm(to_goal))
            step = min(TRANSPORT_LEAD, dist)
            nxt = obj.copy()
            nxt[:2] = obj[:2] + (to_goal / max(dist, 1e-9)) * step
            dth = _wrap(float(goal[2]) - float(obj[2]))
            nxt[2] = float(obj[2]) + np.clip(dth, -MAX_DANG * 3, MAX_DANG * 3)
            pos, tang = self._target_for_depth(nxt, SEAT_DEPTH)
            self.grip = min(1.0, self.grip + GRIP_RAMP) if self.needs_grip else 0.0
            return self._emit(pos, tang, self.grip)

        # Not engaged: align at standoff, then insert along the stem axis.
        pre_pos, tang = self._target_for_depth(obj, SEAT_DEPTH + STANDOFF)
        seat_pos, _ = self._target_for_depth(obj, SEAT_DEPTH)
        tip, sdir = self._stem(obj)
        _v = a - tip
        # 2-D cross product; numpy 2 removed the 2-vector overload.
        lateral = abs(float(sdir[0] * _v[1] - sdir[1] * _v[0]))
        aligned = abs(_wrap(tang - ang)) < ALIGN_ANG_TOL and lateral < ALIGN_LAT_TOL

        if self._since_progress > JAM_STEPS:
            # Back off along the approach axis and retry rather than grinding.
            self._since_progress = 0
            self.state = "ALIGN"
            self.grip = 0.0
            back, _ = self._target_for_depth(obj, SEAT_DEPTH + STANDOFF * 1.8)
            return self._emit(self._route(a, back, obj[:2]), tang, self.grip)

        if aligned:
            self.state = "INSERT"
            # Grippers open on approach and close only once seated.
            if self.needs_grip:
                near = float(np.linalg.norm(a - seat_pos)) < 12.0
                self.grip = min(1.0, self.grip + GRIP_RAMP) if near else 0.0
            return self._emit(seat_pos, tang, self.grip)

        self.state = "ALIGN"
        self.grip = 0.0
        return self._emit(self._route(a, pre_pos, obj[:2]), tang, self.grip)
