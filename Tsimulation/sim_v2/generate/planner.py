"""Scripted TAMP-style planners: solve the task without a human.

Each planner is a small state machine over the same phases a person uses --
reach a graspable feature, engage, transport, align, release -- with the
per-embodiment differences confined to which phases exist and what `grip`
means. They exist to seed MimicGen: generating a handful of source demos by
hand is the slow step, and these produce them in seconds.

They are NOT meant to be good policies. They are open-loop-ish scripts with a
success check, so their output must be FILTERED on coverage rather than
trusted -- see `generate()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv
from Tsimulation.sim_v2.pushshapes.shapes import SHAPES


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def limb_tip(env, obj_shape: str = "T") -> tuple[float, float]:
    """World point of the object's most distal limb -- the graspable feature.

    The T spans 120 but its limbs are 30 wide, so anything with jaws has to
    aim at a limb; aiming at the centroid drives them into the junction.
    """
    ox, oy, oth = env.object_pose
    ct, st = math.cos(oth), math.sin(oth)
    best = None
    for cx, cy, _w, _h in SHAPES[obj_shape]:
        wx, wy = ox + cx * ct - cy * st, oy + cx * st + cy * ct
        d = math.hypot(wx - ox, wy - oy)
        if best is None or d > best[0]:
            best = (d, wx, wy)
    return best[1], best[2]


def _step_toward(cur, tgt, speed):
    dx, dy = tgt[0] - cur[0], tgt[1] - cur[1]
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return tgt
    k = min(speed, d) / d
    return (cur[0] + dx * k, cur[1] + dy * k)


@dataclass
class Plan:
    """One generated attempt."""

    actions: list = field(default_factory=list)
    init: dict = field(default_factory=dict)
    success: bool = False
    coverage: float = 0.0
    steps: int = 0


def _grasp_planner(env, max_steps: int, close_at: float = 0.0) -> Plan:
    """Reach a limb, close, carry to the goal pose, for anything that grasps."""
    acts = []
    held = lambda: bool(getattr(env.agent, "grasped", False)
                        or getattr(env.agent, "attached", False)
                        or getattr(env.agent, "mode", "") == "clamped")
    n = len(env.agent.action_spec)
    best = 0.0
    for t in range(max_steps):
        ox, oy, oth = env.object_pose
        gx, gy, gth = env.goal_pose
        px, py = env.agent_pos
        if not held():
            lx, ly = limb_tip(env)
            tx, ty = _step_toward((px, py), (lx, ly), 3.0)
            ang = math.atan2(ly - oy, lx - ox) + math.pi / 2
            grip = 1.0 if t > 30 else 0.0          # open while approaching
        else:
            tx = px + float(np.clip(gx - ox, -3.0, 3.0))
            ty = py + float(np.clip(gy - oy, -3.0, 3.0))
            ang = env.pusher_angle + float(np.clip(wrap(gth - oth), -0.03, 0.03))
            grip = 1.0
        a = np.array([tx, ty, ang, grip][:n], dtype=np.float64)
        acts.append(a)
        _o, _r, term, _tr, info = env.step(a)
        best = max(best, info["coverage"])
        if term:
            return Plan(acts, {}, True, best, t + 1)
    return Plan(acts, {}, False, best, max_steps)


def _push_planner(env, max_steps: int) -> Plan:
    """Stage behind the object, then drive it goalward. For pure pushers."""
    acts = []
    n = len(env.agent.action_spec)
    pushing = False
    best = 0.0
    for t in range(max_steps):
        ox, oy, oth = env.object_pose
        gx, gy, gth = env.goal_pose
        px, py = env.agent_pos
        vx, vy = gx - ox, gy - oy
        d = math.hypot(vx, vy) or 1e-9
        ux, uy = vx / d, vy / d
        # Steer orientation by where we contact: an offset perpendicular to
        # travel makes the push off-centre and torques the object. Pushing
        # straight through the centroid (the first version) controlled
        # position only and never satisfied a POSE goal -- 0/12 on every pure
        # pusher, because IoU 0.95 needs the angle too.
        nx, ny = -uy, ux
        dth = wrap(gth - oth)
        lever = float(np.clip(dth * 34.0, -30.0, 30.0))
        sx = ox - ux * 56.0 + nx * lever
        sy = oy - uy * 56.0 + ny * lever
        if not pushing:
            if math.hypot(px - sx, py - sy) < 10.0:
                pushing = True                     # latch, never re-stage
            tx, ty = _step_toward((px, py), (sx, sy), 5.0)
        else:
            if math.hypot(px - ox, py - oy) > 130.0:
                pushing = False
            v = float(np.clip(d * 0.06, 0.0, 3.5))  # P-control: no overshoot
            # Track the moving lever point rather than driving blindly along
            # u, so the contact stays off-centre by the amount the angle error
            # calls for.
            aim = (ox - ux * 26.0 + nx * lever, oy - uy * 26.0 + ny * lever)
            tx, ty = _step_toward((px, py), aim, max(v, 1.2))
        ang = math.atan2(uy, ux)
        a = np.array([tx, ty, ang, 1.0][:n], dtype=np.float64)
        acts.append(a)
        _o, _r, term, _tr, info = env.step(a)
        best = max(best, info["coverage"])
        if term:
            return Plan(acts, {}, True, best, t + 1)
    return Plan(acts, {}, False, best, max_steps)


#: Which planner suits which embodiment.
GRASPERS = ("gripper", "umi", "suction")
PLANNERS: dict[str, Callable] = {}


def plan_for(agent: str) -> Callable:
    return _grasp_planner if agent in GRASPERS else _push_planner


def generate(agent: str, n: int, *, object_shape: str = "T",
             obstacle_level: int = 0, max_steps: int = 1200,
             seed0: int = 0) -> list[Plan]:
    """Run the planner over `n` random layouts, returning ONLY successes.

    Filtering on the env's own success check is the point: the planners are
    scripts, not policies, and their raw attempts include plenty of failures.
    """
    out = []
    for i in range(n):
        env = PushShapesEnv(object_shape=object_shape, pusher_shape=agent,
                            obstacle_level=obstacle_level)
        env.reset(seed=seed0 + i)
        env._skip_obs_render = True      # 3.3x faster; only coverage is read
        init = env.get_episode_init()
        p = plan_for(agent)(env, max_steps)
        p.init = init
        if p.success:
            out.append(p)
    return out
