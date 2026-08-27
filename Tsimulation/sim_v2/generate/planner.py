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


def limb_points(env, obj_shape: str = "T") -> list[tuple[float, float]]:
    """World centres of every limb -- all the candidate graspable features."""
    ox, oy, oth = env.object_pose
    ct, st = math.cos(oth), math.sin(oth)
    return [(ox + cx * ct - cy * st, oy + cx * st + cy * ct)
            for cx, cy, _w, _h in SHAPES[obj_shape]]


def limb_tip(env, obj_shape: str = "T", rng=None) -> tuple[float, float]:
    """A graspable limb: the most distal one, or a RANDOM one if `rng` given.

    Always choosing the most distal limb makes every seed the same manoeuvre,
    which is the root of the redundancy this module now works against: 854
    generated demos had an intrinsic dimensionality of 5 because they all
    descended from one trajectory shape.
    """
    pts = limb_points(env, obj_shape)
    ox, oy, _ = env.object_pose
    if rng is not None and len(pts) > 1:
        wx, wy = pts[int(rng.integers(len(pts)))]
    else:
        wx, wy = max(pts, key=lambda q: math.hypot(q[0] - ox, q[1] - oy))
    return wx, wy


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


def _grasp_planner(env, max_steps: int, close_at: float = 0.0, rng=None) -> Plan:
    """Reach a limb, close, carry to the goal pose, for anything that grasps.

    Every free parameter is SAMPLED when `rng` is supplied -- which limb, how
    fast to approach, how long to stay open, how hard to servo the carry.
    A deterministic planner emits one trajectory shape however many times it
    is run, and every retarget inherits that shape.
    """
    acts = []
    r = rng
    approach_speed = float(r.uniform(2.0, 4.5)) if r is not None else 3.0
    open_for = int(r.integers(18, 46)) if r is not None else 30
    carry_gain = float(r.uniform(2.0, 4.0)) if r is not None else 3.0
    ang_gain = float(r.uniform(0.02, 0.05)) if r is not None else 0.03
    target = None
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
            if target is None:
                target = limb_tip(env, rng=r)      # sampled limb, held fixed
            lx, ly = target
            tx, ty = _step_toward((px, py), (lx, ly), approach_speed)
            ang = math.atan2(ly - oy, lx - ox) + math.pi / 2
            grip = 1.0 if t > open_for else 0.0
        else:
            target = None
            tx = px + float(np.clip(gx - ox, -carry_gain, carry_gain))
            ty = py + float(np.clip(gy - oy, -carry_gain, carry_gain))
            ang = env.pusher_angle + float(np.clip(wrap(gth - oth), -ang_gain, ang_gain))
            grip = 1.0
        a = np.array([tx, ty, ang, grip][:n], dtype=np.float64)
        acts.append(a)
        _o, _r, term, _tr, info = env.step(a)
        best = max(best, info["coverage"])
        if term:
            return Plan(acts, {}, True, best, t + 1)
    return Plan(acts, {}, False, best, max_steps)


def _push_planner(env, max_steps: int, rng=None) -> Plan:
    """Stage behind the object, then drive it goalward. For pure pushers."""
    acts = []
    n = len(env.agent.action_spec)
    pushing = False
    best = 0.0
    r = rng
    standoff = float(r.uniform(44.0, 74.0)) if r is not None else 56.0
    stage_speed = float(r.uniform(3.5, 6.5)) if r is not None else 5.0
    kp = float(r.uniform(0.04, 0.09)) if r is not None else 0.06
    vmax = float(r.uniform(2.5, 4.5)) if r is not None else 3.5
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
        sx = ox - ux * standoff + nx * lever
        sy = oy - uy * standoff + ny * lever
        if not pushing:
            if math.hypot(px - sx, py - sy) < 10.0:
                pushing = True                     # latch, never re-stage
            tx, ty = _step_toward((px, py), (sx, sy), stage_speed)
        else:
            if math.hypot(px - ox, py - oy) > 130.0:
                pushing = False
            v = float(np.clip(d * kp, 0.0, vmax))  # P-control: no overshoot
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


def generate(agent: str, n: int, *, object_shape: str | None = "T",
             obstacle_level: int = 0, max_steps: int = 1200,
             seed0: int = 0, randomize: bool = True) -> list[Plan]:
    """Run the planner over `n` random layouts, returning ONLY successes.

    `object_shape=None` samples T/U/Z, and `randomize` samples the planner's
    free parameters and the initial agent pose per attempt. Both exist because
    seed DIVERSITY multiplies through every downstream retarget -- fifteen
    copies of one manoeuvre cannot be rescued by generating more of them.
    """
    out = []
    shapes = [object_shape] if object_shape else list(SHAPES)
    for i in range(n):
        rng = np.random.default_rng(seed0 + i) if randomize else None
        shp = shapes[int(rng.integers(len(shapes)))] if rng is not None else shapes[0]
        env = PushShapesEnv(object_shape=shp, pusher_shape=agent,
                            obstacle_level=obstacle_level)
        env.reset(seed=seed0 + i)
        env._skip_obs_render = True      # 3.3x faster; only coverage is read
        if rng is not None:
            # Vary the initial effector pose. Leaving it at the reset default
            # gave 854 episodes with agent_angle == 0.000 in every one.
            env.set_state(agent_angle=float(rng.uniform(-math.pi, math.pi)))
        init = env.get_episode_init()
        p = plan_for(agent)(env, max_steps, rng=rng)
        p.init = init
        if p.success:
            out.append(p)
    return out
