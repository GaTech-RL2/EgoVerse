"""Audit downloaded numeric/image payloads and replay a sample from every cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import simplejpeg
import zarr

from ..pushshapes.agents import CONTROL_GAPS
from ..pushshapes.env import PushShapesEnv
from .articulation_quality import MechanismAudit, interaction


def replay_saved(attrs, actions, states, rewards, engaged):
    emb = attrs["quality"]["embodiment"]
    env = PushShapesEnv(pusher_shape=emb)
    env.agent.control_gap = CONTROL_GAPS[attrs["control_gap"]]
    env._skip_obs_render = True
    obs, _ = env.reset(seed=attrs["reset_seed"])
    audit = MechanismAudit(env)
    state_error = reward_error = 0.0
    held = []
    try:
        for index, action in enumerate(actions):
            state = np.r_[obs["agent_pos"], obs["agent_angle"], obs["object_pose"]]
            delta = state - states[index]
            delta[[2, 5]] = (delta[[2, 5]] + np.pi) % (2 * np.pi) - np.pi
            state_error = max(state_error, float(np.max(np.abs(delta))))
            obs, reward, *_ = env.step(action.astype(np.float64))
            reward_error = max(reward_error, abs(reward - float(rewards[index])))
            held.append(interaction(env))
            audit.observe(env, held[-1])
        assert state_error < 1e-4, ("replay state", state_error)
        assert reward_error < 1e-4, ("replay reward", reward_error)
        np.testing.assert_array_equal(held, engaged)
        assert audit.passes(emb), audit.metrics()
        return dict(
            state_error=state_error, reward_error=reward_error, **audit.metrics()
        )
    finally:
        env.close()


def audit_episode(path, replay=False):
    store = zarr.open_group(path, mode="r")
    attrs = dict(store.attrs)
    quality = attrs["quality"]
    actions = store["actions"][:]
    states = store["observations.state"][:]
    rewards = store["reward"][:].ravel()
    engaged = store["engaged"][:].ravel()
    assert quality["success"] and quality["replay_state_error"] < 1e-5
    assert (
        len(actions)
        == attrs["total_frames"]
        == len(states)
        == len(rewards)
        == len(engaged)
    )
    assert list(actions.shape) == [len(actions), len(attrs["action_spec"])]
    assert (
        hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest()
        == attrs["actions_sha256"]
    )
    assert np.isfinite(actions).all() and np.isfinite(states).all()
    assert engaged[-1] and engaged.sum() >= 20 and rewards[-1] >= 0.95 - 1e-6
    assert quality["carry_distance"] >= 30 and quality["jerk_speed"] < 0.5
    assert (
        quality["max_object_overflow"] < 1e-4
        and quality["max_pusher_object_penetration"] < 0.6
    )
    velocity = np.diff(
        np.vstack([states[0, :2], actions[:, :2]]).astype(np.float64), axis=0
    )
    speeds = np.linalg.norm(velocity, axis=1)
    jerk = np.linalg.norm(np.diff(velocity, axis=0), axis=1).mean() / max(
        speeds.mean(), 1e-9
    )
    angles = np.abs(
        (np.diff(actions[:, 2].astype(np.float64)) + np.pi) % (2 * np.pi) - np.pi
    )
    assert jerk < 0.5 and speeds.max() < 6.0 and angles.max() < 0.1
    assert (
        angles.sum() > 0.01
        and (actions[:, :2] >= 0).all()
        and (actions[:, :2] <= 512).all()
    )
    if "grip" in attrs["action_spec"]:
        grip = actions[:, attrs["action_spec"].index("grip")]
        assert grip.max() > 0 and np.ptp(grip) > 0
        assert np.abs(np.diff(grip)).max() < 0.051
    if "mechanism_work" in store:
        assert store["mechanism_work"][:].sum() >= 5
    encoded = store["observations.images.front_img_1"][:1][0]
    assert simplejpeg.decode_jpeg(bytes(encoded)).shape == (96, 96, 3)
    result = dict(
        path=str(path),
        seed=attrs["reset_seed"],
        actions_sha256=attrs["actions_sha256"],
        embodiment=quality["embodiment"],
        gap=attrs["control_gap"],
        frames=len(actions),
        final_coverage=float(rewards[-1]),
        jerk_speed=jerk,
        engaged_fraction=float(engaged.mean()),
        carry_distance=quality["carry_distance"],
    )
    if replay:
        result["replay"] = replay_saved(attrs, actions, states, rewards, engaged)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--replay-per-cell", type=int, default=1)
    a = ap.parse_args()
    paths = sorted(a.root.glob("*/*/shard*/episode_*.zarr"))
    assert paths, "No downloaded episodes found"
    cells = defaultdict(list)
    for path in paths:
        cell = (path.parent.parent.name, path.parent.parent.parent.name)
        row = audit_episode(path, replay=len(cells[cell]) < a.replay_per_cell)
        cells[cell].append(row)
    rows = [row for group in cells.values() for row in group]
    assert len({r["seed"] for r in rows}) == len(rows), "Duplicate reset seeds"
    assert len({r["actions_sha256"] for r in rows}) == len(
        rows
    ), "Duplicate action sequences"
    report = dict(
        episodes_checked=len(rows),
        replayed=sum("replay" in r for r in rows),
        cells=[
            dict(
                embodiment=emb,
                gap=gap,
                episodes=len(group),
                min_final_coverage=min(r["final_coverage"] for r in group),
                max_jerk_speed=max(r["jerk_speed"] for r in group),
                min_carry_distance=min(r["carry_distance"] for r in group),
                frames=sum(r["frames"] for r in group),
            )
            for (emb, gap), group in cells.items()
        ],
        episodes=rows,
        passed=True,
    )
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
