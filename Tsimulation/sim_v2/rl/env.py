"""Low-dimensional RL view of PushShapesEnv, one env per (effector, control gap).

The scripted planners in generate/planner.py only solve some initial layouts and
generate() discards the rest, so the dataset is capped by planner competence and
carries only the behaviour a script produces. An RL policy trained per control
gap can solve layouts the script cannot AND adapts its strategy TO the gap,
which is the point: a laggy embodiment needs a different policy, not the same
policy executed worse.

Training runs on privileged low-dim state with rendering off (~3.3x faster);
episodes are re-rolled with rendering on when recording, so the emitted dataset
keeps the same image observations the BC policies consume.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from Tsimulation.pushshapes import PushShapesEnv
from Tsimulation.sim_v2.pushshapes.agents import CONTROL_GAPS


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class PushShapesRLEnv(gym.Env):
    """Flat-vector, shaped-reward view. Action space is the effector's native width."""

    metadata = {"render_modes": []}

    def __init__(self, pusher_shape="gripper", object_shape="T", control_gap="ideal",
                 obstacle_level=0, max_steps=600, image_size=96,
                 shaping=1.0, success_bonus=50.0, step_cost=0.01,
                 render_obs=False, seed=None,
                 action_mode="residual", max_delta=512.0):
        super().__init__()
        self.kw = dict(object_shape=object_shape, pusher_shape=pusher_shape,
                       obstacle_level=obstacle_level, image_size=image_size)
        self.control_gap = control_gap
        self.max_steps = int(max_steps)
        self.shaping = float(shaping)
        self.success_bonus = float(success_bonus)
        self.step_cost = float(step_cost)
        self.render_obs = bool(render_obs)
        self._env = PushShapesEnv(**self.kw)
        self._env._skip_obs_render = not self.render_obs
        self._apply_gap()
        self.W = float(self._env.WORLD_SIZE)
        # The native action is an ABSOLUTE target pose in a 512x512 box, but the
        # expert moves 0.8-1.1 px per step, so the useful manifold is ~1/500th
        # of that space. Exploring it directly teleports the target across the
        # arena every step: a first SAC run peaked at 300k steps and then
        # collapsed to exactly zero coverage. Commanding a bounded DELTA from
        # the current pose keeps exploration local and matches how the action is
        # actually structured (position + small correction).
        # action_mode:
        #   "absolute" -- [-1,1]^n mapped affinely onto the native bounds. The
        #     expert commands targets up to 469px from its current pose (p90 is
        #     already 12-57px), so a bounded delta CANNOT express what the
        #     demonstrations do and BC in delta space would fit a truncated
        #     target. Absolute is the space the demos actually live in.
        #   "delta"    -- bounded offset from the current pose; keeps
        #     exploration local, but only reaches +/- max_delta per step.
        #   "native"   -- raw env units, for replaying recorded actions.
        self.action_mode = str(action_mode)
        self.max_delta = float(max_delta)
        self._native = self._env.action_space
        n = self._native.shape[0]
        if self.action_mode == "native":
            self.action_space = self._native
        else:
            self.action_space = spaces.Box(-1.0, 1.0, (n,), dtype=np.float32)
        # [agent_xy, cos/sin(agent_ang), obj_xy, cos/sin(obj_ang), goal_xy,
        #  cos/sin(goal_ang), obj->goal delta, agent->obj delta, coverage]
        self.observation_space = spaces.Box(-np.inf, np.inf, (19,), dtype=np.float32)
        self._t = 0
        self._prev_phi = 0.0
        self._seed = seed

    def _apply_gap(self):
        if self.control_gap is not None:
            gap = (CONTROL_GAPS[self.control_gap]
                   if isinstance(self.control_gap, str) else self.control_gap)
            self._env.agent.control_gap = gap

    def _obs(self):
        o = self._env._get_obs()
        ax, ay = o["agent_pos"]; aa = float(o["agent_angle"][0])
        ox, oy, oa = o["object_pose"]; gx, gy, ga = o["goal_pose"]
        cov = float(self._env._coverage())
        return np.array([
            ax / self.W, ay / self.W, np.cos(aa), np.sin(aa),
            ox / self.W, oy / self.W, np.cos(oa), np.sin(oa),
            gx / self.W, gy / self.W, np.cos(ga), np.sin(ga),
            (gx - ox) / self.W, (gy - oy) / self.W, _wrap(ga - oa) / np.pi,
            (ox - ax) / self.W, (oy - ay) / self.W,
            np.hypot(gx - ox, gy - oy) / self.W, cov,
        ], dtype=np.float32)

    def _phi(self):
        """Potential: coverage high, object near the goal in position AND angle,
        agent near the object.

        Coverage belongs INSIDE the potential. Paying raw coverage per step made
        it a level rather than a delta, so return integrated over time: two
        episodes both succeeding at 0.95 scored 41.8 and 246.6 purely because
        one ran 2.5x longer. The optimal policy under that reward is to hover
        just below the 0.95 threshold harvesting ~0.9/step forever rather than
        terminate for a one-off bonus -- which is what both collapsed runs did.
        Inside the potential the telescoping sum is bounded and
        length-independent, so finishing sooner strictly wins."""
        o = self._env._get_obs()
        ax, ay = o["agent_pos"]
        ox, oy, oa = o["object_pose"]; gx, gy, ga = o["goal_pose"]
        cov = float(self._env._coverage())
        d_og = np.hypot(gx - ox, gy - oy) / self.W
        d_ang = abs(_wrap(ga - oa)) / np.pi
        d_ao = np.hypot(ox - ax, oy - ay) / self.W
        return 5.0 * cov - (d_og + 0.5 * d_ang + 0.25 * d_ao)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset(seed=seed if seed is not None else self._seed)
        self._env._skip_obs_render = not self.render_obs
        self._apply_gap()
        self._t = 0
        self._prev_phi = self._phi()
        return self._obs(), {}

    def to_norm(self, cmd):
        """Native command -> the current action space's [-1,1]^n encoding.

        For "residual" this is (cmd - current pose)/max_delta. Predicting the
        residual matters because an absolute target is dominated by "copy the
        current pose": BC reached MSE 0.0044 on absolute targets, which sounds
        small but is ~17px of command error over a 512px range, and scored
        exactly 0.0000 coverage. The residual is centred on zero (the expert
        holds position >50% of steps) so the same MSE buys far more accuracy.
        max_delta spans the arena because the expert commands targets up to
        469px away; the earlier +/-10px clip could not express that at all."""
        cmd = np.asarray(cmd, float)
        if self.action_mode == "residual":
            o = self._env._get_obs()
            px, py = o["agent_pos"]; pa = float(o["agent_angle"][0])
            n = self._native.shape[0]
            out = np.zeros(n)
            out[0] = (cmd[0] - px) / self.max_delta
            out[1] = (cmd[1] - py) / self.max_delta
            if n >= 3: out[2] = _wrap(cmd[2] - pa) / np.pi
            if n >= 4:
                lo, hi = self._native.low[3], self._native.high[3]
                out[3] = 2.0 * (cmd[3] - lo) / (hi - lo) - 1.0
            return np.clip(out, -1.0, 1.0)
        lo, hi = self._native.low, self._native.high
        return np.clip(2.0 * (cmd - lo) / (hi - lo) - 1.0, -1.0, 1.0)

    def _residual_to_native(self, a):
        o = self._env._get_obs()
        px, py = o["agent_pos"]; pa = float(o["agent_angle"][0])
        n = self._native.shape[0]
        cmd = np.empty(n)
        cmd[0] = px + a[0] * self.max_delta
        cmd[1] = py + a[1] * self.max_delta
        if n >= 3: cmd[2] = pa + a[2] * np.pi
        if n >= 4:
            lo, hi = self._native.low[3], self._native.high[3]
            cmd[3] = lo + (a[3] + 1.0) * 0.5 * (hi - lo)
        return cmd

    def _abs_to_native(self, a):
        lo, hi = self._native.low, self._native.high
        return lo + (np.asarray(a, float) + 1.0) * 0.5 * (hi - lo)

    def _to_native(self, a):
        """Map a in [-1,1]^n onto an absolute command near the current pose."""
        o = self._env._get_obs()
        px, py = o["agent_pos"]; pa = float(o["agent_angle"][0])
        n = self._native.shape[0]
        cmd = np.empty(n, dtype=np.float64)
        cmd[0] = px + a[0] * self.max_delta
        cmd[1] = py + a[1] * self.max_delta
        if n >= 3:                      # angle: small increment, radians
            cmd[2] = pa + a[2] * (self.max_delta / 30.0)
        if n >= 4:                      # grip is absolute in [0,1], not a delta
            lo, hi = self._native.low[3], self._native.high[3]
            cmd[3] = lo + (a[3] + 1.0) * 0.5 * (hi - lo)
        return cmd

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        a = np.clip(a, self.action_space.low, self.action_space.high)
        if self.action_mode == "delta":
            a = self._to_native(a)
        elif self.action_mode == "absolute":
            a = self._abs_to_native(a)
        elif self.action_mode == "residual":
            a = self._residual_to_native(a)
        a = np.clip(a, self._native.low, self._native.high)
        _o, _r, terminated, _tr, info = self._env.step(a)
        self._t += 1
        cov = float(info.get("coverage", 0.0))
        phi = self._phi()
        reward = self.shaping * (phi - self._prev_phi) - self.step_cost
        self._prev_phi = phi
        if terminated:
            reward += self.success_bonus
        truncated = self._t >= self.max_steps
        info["coverage"] = cov
        info["is_success"] = bool(terminated)
        return self._obs(), float(reward), bool(terminated), bool(truncated), info
