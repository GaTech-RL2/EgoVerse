"""Scripted SE(2) push controller: servo the object's POSE, not just position.

collect/scripted_collect.py's scripted_action returns a (2,) target computed
purely from object->goal displacement. It never reasons about orientation, so on
a success criterion of 0.95 IoU -- which needs position AND angle -- it tops out
partway: measured 0/40 successes on L, circle, stick and circle_small, with mean
coverage 0.06-0.14 and a best of 0.85. It gets the object next to the goal and
then cannot finish.

This adds the missing half. Two primitives:

  translate  push through the centroid along (goal - object)
  rotate     push tangentially at a lever point offset from the centroid, so
             the contact generates torque about the object's center

and an orbit-then-approach repositioning rule, because the pusher is solid: a
straight line to the next contact point would bulldoze the object on the way.
The pusher first backs out to a safe radius, travels around the object on that
circle, then moves in.
"""
from __future__ import annotations

import numpy as np

R_SAFE = 135.0         # orbit radius. Must clear the object: at 85 the
                       # pusher jammed against the T at |p-c|=77.7 and the
                       # whole episode froze -- the tangential orbit move was
                       # blocked by the very object it was circling.
APPROACH_OFFSET = 42.0  # how far behind the contact point to stage
LOOKAHEAD = 55.0        # how far past the contact point to aim when pushing
LEVER_R = 38.0          # lever arm for the rotate primitive
ANGLE_TOL = 0.10        # rad; below this, stop rotating
CONTACT_TOL = 14.0      # px; close enough to the staging point to push
POS_TOL = 45.0          # px; below this, position is good enough to work on angle
ORBIT_TOL = 0.25        # rad of arc to step around per move


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _perp(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=np.float64)


def pose_action(agent_xy, object_pose, goal_pose, world_size,
                angle_weight=140.0):
    """Return a (2,) world-frame target for the pusher.

    angle_weight converts radians of orientation error into the pixels of
    position error they are worth, so the controller works on whichever is
    currently costlier rather than always finishing translation first.
    """
    p = np.asarray(agent_xy, dtype=np.float64)
    c = np.asarray(object_pose[:2], dtype=np.float64)
    th = float(object_pose[2])
    g = np.asarray(goal_pose[:2], dtype=np.float64)
    gth = float(goal_pose[2])

    d_pos = g - c
    pos_err = float(np.linalg.norm(d_pos))
    # The T has 2-fold-ish symmetry in practice only through a full turn, so
    # compare on the wrapped angle rather than a modulus that could pick a pose
    # the IoU metric does not consider equivalent.
    ang_err = _wrap(gth - th)

    # Translate first while the object is far. Weighting angle against position
    # (|ang|*140 vs pos_err) made "rotate" win on every one of 900 steps with the
    # object still 280px from the goal, so it never translated at all.
    if pos_err <= POS_TOL and abs(ang_err) > ANGLE_TOL:
        # ---- rotate: contact tangentially at a lever point
        # Choose the lever direction that is currently easiest to reach, so the
        # pusher does not orbit half way around for an equivalent contact.
        best = None
        for u_ang in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            u = np.array([np.cos(u_ang), np.sin(u_ang)])
            lever = c + u * LEVER_R
            push_dir = _perp(u) * (1.0 if ang_err > 0 else -1.0)
            stage = lever - push_dir * APPROACH_OFFSET
            cost = float(np.linalg.norm(p - stage))
            if best is None or cost < best[0]:
                best = (cost, lever, push_dir, stage)
        _, lever, push_dir, stage = best
        # Tried scaling this with the remaining error; it HURT (L best coverage
        # 0.9226 -> 0.6026) because the small end of the range is below the
        # object's stiction and the push simply does not move it.
        target = lever + push_dir * LOOKAHEAD
    else:
        # ---- translate: push through the centroid toward the goal
        if pos_err < 1e-6:
            return np.clip(c, 0.0, world_size)
        push_dir = d_pos / pos_err
        stage = c - push_dir * APPROACH_OFFSET
        target = c + push_dir * min(LOOKAHEAD, pos_err + 20.0)

    # ---- reposition without bulldozing: back out, orbit, then move in
    if float(np.linalg.norm(p - stage)) > CONTACT_TOL:
        r_p = p - c
        r_s = stage - c
        dist_p = float(np.linalg.norm(r_p))
        a_p = float(np.arctan2(r_p[1], r_p[0]))
        a_s = float(np.arctan2(r_s[1], r_s[0]))
        da = _wrap(a_s - a_p)
        # Check ANGULAR alignment before radial clearance. The reverse order
        # deadlocks: once aligned and moving in to push, the pusher crosses
        # R_SAFE-10 and is immediately shoved back out, so it oscillates
        # between "orbit" and "out" at |p-c| ~ 122-129 forever and the object
        # never moves. Only back out when we still have to travel AROUND.
        if abs(da) > ORBIT_TOL:
            if dist_p < R_SAFE - 10.0:
                out = r_p / max(dist_p, 1e-6)
                return np.clip(c + out * R_SAFE, 0.0, world_size)
            a_next = a_p + np.sign(da) * ORBIT_TOL
            nxt = c + np.array([np.cos(a_next), np.sin(a_next)]) * R_SAFE
            return np.clip(nxt, 0.0, world_size)
        return np.clip(stage, 0.0, world_size)

    return np.clip(target, 0.0, world_size)
