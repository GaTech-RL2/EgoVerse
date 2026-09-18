"""Regenerate source scenes and object paths with another physical embodiment.

No source actions or pixels are relabeled. The target executes new commands in
unchanged Sim V2 physics. Every accepted trajectory is replayed from its initial
state before writing pre-action observations to the standard Zarr schema.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import zarr

from ..collect.articulation_controller import ArticulationController, rot, wrap
from ..collect.articulation_quality import MechanismAudit, interaction
from ..collect.contact_controller import ContactController
from ..pushshapes.agents import ControlGap
from ..pushshapes.env import PushShapesEnv


def source_record(path, provenance):
    group = zarr.open_group(str(path), mode="r")
    init = group.attrs["episode_init"]
    init = json.loads(init) if isinstance(init, str) else dict(init)
    n = int(group.attrs["total_frames"])
    states = np.asarray(group["observations.state"][:n], dtype=np.float64)
    if int(init.get("obstacle_level", 0)) != 0 or init.get("obstacles"):
        raise ValueError("This collector supports level-zero sources only")
    if states.ndim != 2 or states.shape[1] < 6 or n < 2:
        raise ValueError("Invalid source object trajectory")
    if not np.isfinite(states).all():
        raise ValueError("Nonfinite source states")
    return {"init": init, "object_path": states[:, 3:6],
            "provenance": provenance, "source_frames": n}


def make_env(source, embodiment, image_size=96, render=False):
    init = source["init"]
    env = PushShapesEnv(object_shape=init["object_shape"],
                        pusher_shape=embodiment, obstacle_level=0,
                        image_size=image_size, render_mode="rgb_array")
    env._skip_obs_render = not render
    env.reset(seed=int(init.get("reset_seed") or 0))
    if "control_gap" in init:
        env.agent.randomize_gap = False
        env.agent.control_gap = ControlGap(**init["control_gap"])
        env.agent.reset_control_gap(env)
    env.set_state(object_pose=tuple(init["object_pose"]),
                  goal_pose=tuple(init["goal_pose"]),
                  agent_pos=tuple(init["agent_pos"]),
                  agent_angle=float(init.get("agent_angle", 0)))
    return env


def path_waypoints(source, spacing=20.0, angle_spacing=0.2):
    points = [np.asarray(source["init"]["object_pose"], dtype=float)]
    for pose in source["object_path"]:
        if (np.linalg.norm(pose[:2] - points[-1][:2]) >= spacing
                or abs(wrap(pose[2] - points[-1][2])) >= angle_spacing):
            points.append(pose.copy())
    points.append(np.asarray(source["init"]["goal_pose"], dtype=float))
    return np.asarray(points)


class SourcePathGraspController(ArticulationController):
    """Use the verified approach, then follow the recorded object-space path."""
    def __init__(self, env, source, config):
        super().__init__(env)
        self.waypoints = path_waypoints(source, config["waypoint_spacing"],
                                       config["waypoint_angle_spacing"])
        self.path_index = 0
        self.position_tolerance = config["waypoint_position_tolerance"]
        self.angle_tolerance = config["waypoint_angle_tolerance"]

    def __call__(self):
        p, angle, center, theta = self._pose()
        constraints = self.env.agent.active_constraints()
        ready = bool(constraints) and (self.emb != "umi" or len(constraints) == 2)
        if self._grasp_offset is None and not ready:
            return super().__call__()
        if self._grasp_offset is None:
            self._grasp_offset = rot(-theta) @ (p - center)
            self._grasp_angle = wrap(angle - theta)
        if not constraints:
            self.reason = "grasp_lost"
        self.state = "SOURCE_PATH"
        while self.path_index < len(self.waypoints) - 1:
            goal = self.waypoints[self.path_index]
            if (np.linalg.norm(center - goal[:2]) > self.position_tolerance
                    or abs(wrap(theta - goal[2])) > self.angle_tolerance):
                break
            self.path_index += 1
        goal = self.waypoints[self.path_index]
        desired_theta = theta + np.clip(wrap(goal[2] - theta), -0.022, 0.022)
        if self.emb == "umi":
            position = goal[:2] + rot(desired_theta - theta) @ (p - center)
            desired_angle = angle + wrap(desired_theta - theta)
        else:
            position = goal[:2] + rot(desired_theta) @ self._grasp_offset
            desired_angle = desired_theta + self._grasp_angle
        return self.emit(position, desired_angle, 1.0)


def state(env):
    return np.r_[env.agent_pos, env.pusher_angle, env.object_pose]


def rollout(source, embodiment, config):
    env = make_env(source, embodiment)
    grasping = "grip" in env.agent.action_spec and embodiment in ("gripper", "umi")
    controller = (SourcePathGraspController(env, source, config)
                  if grasping else ContactController(env))
    audit = MechanismAudit(env)
    actions, states, rewards, held = [], [], [], []
    initial_position = np.asarray(env.agent_pos)
    initial_object = np.asarray(env.object_pose[:2])
    initial_coverage = float(env._coverage())
    max_penetration = max_overflow = 0.0
    started = time.monotonic()
    try:
        for _ in range(config["max_steps"]):
            action = controller().astype(np.float32)
            if controller.reason:
                break
            if not np.isfinite(action).all() or (action[:2] < 0).any() or (action[:2] > env.WORLD_SIZE).any():
                controller.reason = "unreachable_command"
                break
            _, reward, _, _, _ = env.step(action.astype(np.float64))
            engaged = bool(interaction(env))
            audit.observe(env, engaged)
            actions.append(action)
            states.append(state(env))
            rewards.append(float(reward))
            held.append(engaged)
            overflow, _ = env._object_arena_metrics()
            max_overflow = max(max_overflow, float(overflow))
            max_penetration = max(max_penetration, float(env._pusher_object_penetration_depth()))
            if max_overflow > config["max_object_overflow"]:
                controller.reason = "object_outside_arena"
                break
            moved = np.linalg.norm(np.asarray(env.object_pose[:2]) - initial_object)
            if reward >= config["success_coverage"] and sum(held) >= config["min_contact_steps"] and moved >= config["min_object_displacement"]:
                break
        aa = np.asarray(actions, dtype=np.float32).reshape(-1, len(env.agent.action_spec))
        velocity = np.diff(np.vstack([initial_position, aa[:, :2]]), axis=0)
        speeds = np.linalg.norm(velocity, axis=1)
        accel = np.linalg.norm(np.diff(velocity, axis=0), axis=1)
        jerk_speed = float(accel.mean() / max(speeds.mean(), 1e-9)) if len(accel) else 0.0
        angle_travel = (float(np.abs((np.diff(aa[:, 2]) + np.pi) % (2*np.pi) - np.pi).sum())
                        if "angle" in env.agent.action_spec else None)
        final = rewards[-1] if rewards else initial_coverage
        moved = float(np.linalg.norm(np.asarray(env.object_pose[:2]) - initial_object))
        quality = dict(embodiment=embodiment, steps=len(aa), initial_coverage=initial_coverage,
                       final_coverage=final, peak_coverage=max(rewards, default=initial_coverage),
                       engaged_steps=sum(held), object_displacement=moved,
                       jerk_speed=jerk_speed, angle_travel=angle_travel,
                       max_grip=float(aa[:, 3].max(initial=0)) if grasping else None,
                       max_object_overflow=max_overflow, max_penetration=max_penetration,
                       failure=controller.reason, seconds=time.monotonic()-started,
                       method="source_object_path" if grasping else "source_scene_contact_replanning",
                       source=source["provenance"], **audit.metrics())
        quality["success"] = bool(
            not controller.reason and final >= config["success_coverage"]
            and sum(held) >= config["min_contact_steps"]
            and moved >= config["min_object_displacement"]
            and jerk_speed < config["max_jerk_speed"]
            and max_overflow <= config["max_object_overflow"]
            and max_penetration <= config["max_penetration"]
            and audit.steps >= config["min_mechanism_work_steps"]
            and (not grasping or (held and held[-1] and quality["max_grip"] > 0.5))
            and (angle_travel is None or angle_travel > config["min_angle_travel"]))
        return quality, aa, np.asarray(states)
    finally:
        env.close()


def replay_and_write(source, embodiment, config, actions, expected, quality, output):
    """Store only trajectories whose exact f32 actions reproduce their states."""
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    env = make_env(source, embodiment, config["image_size"], render=True)
    images, states, rewards, held, post = [], [], [], [], []
    init = env.get_episode_init()
    try:
        for action in actions:
            obs = env._get_obs()
            images.append(obs["image"])
            states.append(state(env))
            _, reward, _, _, _ = env.step(action.astype(np.float64))
            post.append(state(env))
            rewards.append(reward)
            held.append(interaction(env))
        error = float(np.max(np.abs(np.asarray(post) - expected)))
        if error > config["replay_tolerance"] or rewards[-1] < config["success_coverage"]:
            raise ValueError(f"Acceptance replay differs: error={error}, coverage={rewards[-1]}")
        quality = dict(quality, replay_max_state_error=error)
        n = len(actions)
        writer = ZarrWriter(episode_path=output, embodiment=f"pushshapes_sim_{embodiment}",
                            fps=30, task_name="pushshapes", chunk_timesteps=n,
                            task_description=json.dumps({"env_args": dict(object_shape="T", pusher_shape=embodiment,
                                obstacle_level=0, collector="cross-embodiment-source-v1"), "version": "0.3"}),
                            annotations=[("Move the T onto the goal pose.", 0, n-1)])
        cmd = np.zeros((n, 3), np.float32)
        cmd[:, :min(3, actions.shape[1])] = actions[:, :3]
        if actions.shape[1] < 3:
            cmd[:, 2] = float(init.get("agent_angle", 0))
        writer.write(numeric_data={"actions": actions,
            "observations.state": np.asarray(states, dtype=np.float32),
            "observations.pusher_cmd_pose": cmd,
            "reward": np.asarray(rewards, np.float32)[:, None],
            "goal_pose": np.tile(np.asarray(init["goal_pose"], np.float32), (n, 1)),
            "engaged": np.asarray(held, np.uint8)[:, None]},
            image_data={"observations.images.front_img_1": np.asarray(images)},
            metadata_override={"episode_init": json.dumps(init), "state_timing": "before_action",
                "action_target_offset_obs2": 1, "action_spec": list(env.agent.action_spec),
                "collector": "cross-embodiment-source-v1", "generation": source["provenance"],
                "split": source["provenance"]["split"], "quality": quality,
                "source_capsule_sha256": os.environ.get("SOURCE_CAPSULE_SHA256"),
                "actions_sha256": hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest()})
        return quality, np.asarray(images)
    finally:
        env.close()
