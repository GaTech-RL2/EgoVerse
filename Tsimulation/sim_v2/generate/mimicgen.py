"""MimicGen-style demo generation for PushShapes.

The idea, following https://mimicgen.github.io: take a handful of source
demonstrations, and for each NEW scene layout transform the recorded
end-effector trajectory into the new object's frame, replay it open-loop, and
KEEP ONLY THE ATTEMPTS THAT SUCCEED. A few human demos become many.

What makes it applicable here: the task is single-object and single-subtask,
so MimicGen's hard part -- segmenting a demo into object-centric subtasks --
collapses. What remains is the frame transform and the success filter, and the
env already provides deterministic reset (`set_state`), full init capture
(`get_episode_init`) and a programmatic success check (coverage >= 0.95).

TWO REFERENCE FRAMES, not one. A pushing demo has an approach phase, which is
positioned relative to the OBJECT, and a transport phase, which is aimed at
the GOAL. Transforming the whole trajectory in the object frame reproduces the
approach but sends the transport wherever the object frame happens to point,
so the two phases are transformed separately and blended across the segment
boundary.

The `grip` channel is NOT transformed. Position and angle map into a new frame
cleanly; *when* to close the jaws does not, because it is tied to reaching a
contact rather than to a timestep. It is replayed as recorded, which is why
generation succeeds far more often for pure pushers than for graspers -- see
the measured rates in the module tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from Tsimulation.sim_v2.pushshapes.env import PushShapesEnv


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class SourceDemo:
    """One successful demonstration, with the scene it was recorded in."""

    agent: str
    actions: np.ndarray            # (T, dim)
    object_pose: tuple             # (x, y, theta) at t=0
    goal_pose: tuple
    agent_pos: tuple
    object_shape: str = "T"
    obstacle_level: int = 0

    @property
    def horizon(self) -> int:
        return len(self.actions)


@dataclass
class GenResult:
    demos: list = field(default_factory=list)
    attempts: int = 0

    @property
    def rate(self) -> float:
        return len(self.demos) / self.attempts if self.attempts else 0.0


def se2(dx: float, dy: float, dth: float):
    """Return a function applying the SE(2) transform about the origin."""
    c, s = math.cos(dth), math.sin(dth)

    def f(x, y):
        return (c * x - s * y + dx, s * x + c * y + dy)

    return f


def _frame_delta(src: tuple, dst: tuple):
    """Transform mapping points expressed about `src` onto `dst`."""
    sx, sy, sth = src
    dx, dy, dth = dst
    rot = wrap(dth - sth)
    c, s = math.cos(rot), math.sin(rot)

    def f(px, py):
        # into src frame, rotate, out into dst frame
        ox, oy = px - sx, py - sy
        return (dx + c * ox - s * oy, dy + s * ox + c * oy)

    return f, rot


def contact_split(demo: SourceDemo, env_shape: str = "T") -> int:
    """Index separating approach from transport.

    Taken as the first step where the commanded position stops closing on the
    object -- a cheap stand-in for 'contact was made' that needs no replay.
    """
    obj = np.array(demo.object_pose[:2])
    d = np.linalg.norm(demo.actions[:, :2] - obj, axis=1)
    if len(d) < 4:
        return len(d)
    # first index after which distance stops decreasing meaningfully
    for i in range(2, len(d) - 1):
        if d[i] <= d.min() * 1.08:
            return i
    return len(d) // 2


def retarget(demo: SourceDemo, new_object: tuple, new_goal: tuple) -> np.ndarray:
    """Map a source trajectory onto a new (object, goal) layout."""
    f_obj, rot_obj = _frame_delta(demo.object_pose, new_object)
    f_goal, rot_goal = _frame_delta(demo.goal_pose, new_goal)
    split = contact_split(demo)
    out = demo.actions.copy()
    n = len(out)
    for i in range(n):
        # Blend from the object frame to the goal frame across the transport
        # phase, so the trajectory is continuous rather than jumping at the
        # segment boundary.
        if i <= split:
            w = 0.0
        else:
            w = min(1.0, (i - split) / max(1.0, 0.35 * (n - split)))
        ax, ay = out[i, 0], out[i, 1]
        ox, oy = f_obj(ax, ay)
        gx, gy = f_goal(ax, ay)
        out[i, 0] = (1 - w) * ox + w * gx
        out[i, 1] = (1 - w) * oy + w * gy
        if out.shape[1] >= 3:
            out[i, 2] = wrap(out[i, 2] + (1 - w) * rot_obj + w * rot_goal)
    return out


def replay(demo: SourceDemo, new_object: tuple, new_goal: tuple,
           new_agent: tuple | None = None, extra_steps: int = 120):
    """Retarget and roll out. Returns (success, coverage, actions_played)."""
    env = PushShapesEnv(object_shape=demo.object_shape,
                        pusher_shape=demo.agent,
                        obstacle_level=demo.obstacle_level)
    env.reset(seed=0)
    f_obj, _ = _frame_delta(demo.object_pose, new_object)
    start = new_agent or f_obj(*demo.agent_pos)
    env.set_state(object_pose=new_object, goal_pose=new_goal,
                  agent_pos=(float(start[0]), float(start[1])))
    acts = retarget(demo, new_object, new_goal)
    best = 0.0
    played = []
    for a in acts:
        played.append(a)
        _o, _r, term, _tr, info = env.step(np.asarray(a, dtype=np.float64))
        best = max(best, info["coverage"])
        if term:
            return True, best, np.array(played)
    # Let the last command settle -- the retargeted stroke can land slightly
    # short of the threshold even when the pose is essentially right.
    hold = acts[-1]
    for _ in range(extra_steps):
        played.append(hold)
        _o, _r, term, _tr, info = env.step(np.asarray(hold, dtype=np.float64))
        best = max(best, info["coverage"])
        if term:
            return True, best, np.array(played)
    return False, best, np.array(played)


def sample_layout(rng, world: float = 512.0, margin: float = 110.0):
    """A fresh (object, goal) pair, well inside the arena."""
    def pose():
        return (float(rng.uniform(margin, world - margin)),
                float(rng.uniform(margin, world - margin)),
                float(rng.uniform(-math.pi, math.pi)))
    o = pose()
    g = pose()
    while math.hypot(g[0] - o[0], g[1] - o[1]) < 90.0:
        g = pose()
    return o, g


def generate(sources: list, n_attempts: int, seed: int = 0) -> GenResult:
    """Expand `sources` into new successful demos over random layouts."""
    rng = np.random.default_rng(seed)
    res = GenResult(attempts=0)
    for i in range(n_attempts):
        src = sources[i % len(sources)]
        obj, goal = sample_layout(rng)
        res.attempts += 1
        ok, cov, played = replay(src, obj, goal)
        if ok:
            res.demos.append(SourceDemo(
                agent=src.agent, actions=played, object_pose=obj,
                goal_pose=goal, agent_pos=src.agent_pos,
                object_shape=src.object_shape,
                obstacle_level=src.obstacle_level))
    return res
