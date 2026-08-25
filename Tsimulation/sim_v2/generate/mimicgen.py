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

from Tsimulation.sim_v2.pushshapes.agents import ControlGap
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
    agent_angle: float = 0.0
    object_shape: str = "T"
    obstacle_level: int = 0
    # Exact command-compliance physics used by the collected source.  Keeping
    # this on every generated demo is essential for a control-gap dataset: the
    # old generator reconstructed PushShapesEnv with its default Ideal gap,
    # silently turning Tight/Loose/etc. sources into Ideal episodes.
    control_gap: dict | None = None
    control_gap_mode: str | None = None

    @property
    def horizon(self) -> int:
        return len(self.actions)


def apply_source_control_gap(env: PushShapesEnv, demo: SourceDemo) -> None:
    """Install a source demo's exact command-compliance physics on ``env``."""
    if demo.control_gap is None:
        return
    env.agent.randomize_gap = False
    env.agent.control_gap = ControlGap(**demo.control_gap)
    # The controller owns latency queues, noise state, and its RNG.  Reset all
    # of them after replacing the gap so replay starts from a clean episode.
    env.agent.reset_control_gap(env)


@dataclass
class GenResult:
    demos: list = field(default_factory=list)
    attempts: int = 0
    layout_failures: int = 0

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
    """Retarget and roll out.

    Returns (success, coverage, actions_played, start_pos). The START POSITION
    is returned because it is part of the demo: it is derived from the object
    transform, so a caller that re-runs the trajectory from the SOURCE demo's
    start will not reproduce the result -- measured 2 of 9 generated demos
    surviving a re-replay before this was threaded through.
    """
    env = PushShapesEnv(object_shape=demo.object_shape,
                        pusher_shape=demo.agent,
                        obstacle_level=demo.obstacle_level)
    env.reset(seed=0)
    apply_source_control_gap(env, demo)
    # Generation never looks at the pixels, only at coverage, and rendering is
    # 3.3x of the step cost (266 -> 890 steps/sec with it off).
    env._skip_obs_render = True
    f_obj, rot_obj = _frame_delta(demo.object_pose, new_object)
    start = new_agent or f_obj(*demo.agent_pos)
    start_angle = wrap(float(demo.agent_angle) + rot_obj)
    env.set_state(object_pose=new_object, goal_pose=new_goal,
                  agent_pos=(float(start[0]), float(start[1])),
                  agent_angle=start_angle)
    acts = retarget(demo, new_object, new_goal)
    best = 0.0
    played = []
    for a in acts:
        played.append(a)
        _o, _r, term, _tr, info = env.step(np.asarray(a, dtype=np.float64))
        best = max(best, info["coverage"])
        if term:
            return True, best, np.array(played), start
    # Let the last command settle -- the retargeted stroke can land slightly
    # short of the threshold even when the pose is essentially right.
    hold = acts[-1]
    for _ in range(extra_steps):
        played.append(hold)
        _o, _r, term, _tr, info = env.step(np.asarray(hold, dtype=np.float64))
        best = max(best, info["coverage"])
        if term:
            return True, best, np.array(played), start
    return False, best, np.array(played), start


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


def sample_equivariant_layout(
    demo: SourceDemo,
    rng,
    world: float = 512.0,
    command_margin: float = 20.0,
    object_margin: float = 70.0,
    max_tries: int = 128,
):
    """Rigidly translate/rotate one complete demonstrated scene.

    Independently sampling a new object and goal changes the demonstrated
    contact path.  That produced zero accepted examples for even the simple
    circle pusher.  A single SE(2) transform preserves every relative pose,
    contact normal, path length, and engage time while still varying absolute
    location and orientation.  Rejection only enforces arena margins.
    """
    anchor = np.asarray(demo.object_pose[:2], dtype=np.float64)
    motion_points = np.vstack((
        np.asarray(demo.actions[:, :2], dtype=np.float64),
        np.asarray(demo.agent_pos, dtype=np.float64)[None],
    ))
    pose_points = np.vstack((
        anchor[None],
        np.asarray(demo.goal_pose[:2], dtype=np.float64)[None],
    ))
    motion_rel = motion_points - anchor
    pose_rel = pose_points - anchor

    for _ in range(max_tries):
        # Sim V2's Stick is deliberately locked to one world angle. Rotating
        # its scene would rotate the required contact normal while the actual
        # pusher remains horizontal, destroying equivariance. It still gets
        # full 2-D translation augmentation.
        rotation = (
            0.0 if demo.agent == "stick"
            else float(rng.uniform(-math.pi, math.pi))
        )
        c, s = math.cos(rotation), math.sin(rotation)
        matrix = np.array(((c, -s), (s, c)), dtype=np.float64)
        rotated_motion = motion_rel @ matrix.T
        rotated_pose = pose_rel @ matrix.T

        lower = np.maximum(
            command_margin - rotated_motion.min(axis=0),
            object_margin - rotated_pose.min(axis=0),
        )
        upper = np.minimum(
            world - command_margin - rotated_motion.max(axis=0),
            world - object_margin - rotated_pose.max(axis=0),
        )
        if np.all(lower <= upper):
            new_anchor = rng.uniform(lower, upper)
            goal_xy = new_anchor + rotated_pose[1]
            agent_xy = new_anchor + rotated_motion[-1]
            obj = (
                float(new_anchor[0]),
                float(new_anchor[1]),
                wrap(float(demo.object_pose[2]) + rotation),
            )
            goal = (
                float(goal_xy[0]),
                float(goal_xy[1]),
                wrap(float(demo.goal_pose[2]) + rotation),
            )
            return obj, goal, (float(agent_xy[0]), float(agent_xy[1]))

    raise RuntimeError(
        "could not fit a rigidly augmented demonstration inside the arena; "
        f"agent={demo.agent} horizon={demo.horizon}"
    )


def generate(sources: list, n_attempts: int, seed: int = 0) -> GenResult:
    """Expand sources with contact-preserving random SE(2) augmentation."""
    rng = np.random.default_rng(seed)
    res = GenResult(attempts=0)
    for i in range(n_attempts):
        src = sources[i % len(sources)]
        res.attempts += 1
        try:
            obj, goal, agent = sample_equivariant_layout(src, rng)
        except RuntimeError as exc:
            # A collected trajectory can be valid in its original scene yet
            # have a command extent too wide to fit the augmentation margins
            # (observed for one fixed-angle Jittery/Stick source). That source
            # is unusable for this attempted augmentation, but it must not
            # terminate the entire resumable cell job. Do not swallow unrelated
            # RuntimeErrors from elsewhere in generation.
            if not str(exc).startswith(
                "could not fit a rigidly augmented demonstration inside the arena"
            ):
                raise
            res.layout_failures += 1
            continue
        ok, cov, played, start = replay(src, obj, goal, new_agent=agent)
        if ok:
            _, rot_obj = _frame_delta(src.object_pose, obj)
            res.demos.append(SourceDemo(
                agent=src.agent, actions=played, object_pose=obj,
                goal_pose=goal, agent_pos=(float(start[0]), float(start[1])),
                agent_angle=wrap(float(src.agent_angle) + rot_obj),
                object_shape=src.object_shape,
                obstacle_level=src.obstacle_level,
                control_gap=(dict(src.control_gap)
                             if src.control_gap is not None else None),
                control_gap_mode=src.control_gap_mode))
    return res
