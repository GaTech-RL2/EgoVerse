"""Collect verified articulated demos, with atomic episodes and resumable quotas.

Each output directory belongs to one (embodiment, gap, seed range) shard.
Failed search/replay/quality checks never count towards its target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np

from ..pushshapes.agents import CONTROL_GAPS, make_agent
from ..pushshapes.env import PushShapesEnv
from .articulation_controller import GRASPING
from .articulation_probe import rollout
from .articulation_quality import ARTICULATED, MechanismAudit, interaction

COLLECTOR_VERSION = "articulated-v1"


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def mechanics(env):
    agent = env.agent
    return np.array(
        [
            float(getattr(agent, "_jaw_gap_state", 0.0)),
            float(getattr(agent, "_grip_state", 0.0)),
            float(getattr(agent, "joint_angle", 0.0)),
            float(getattr(agent, "swing_deg", 0.0)),
            float(getattr(agent, "compression", 0.0)),
        ],
        dtype=np.float32,
    )


def replay(emb, gap, seed, actions, expected, image_size=96):
    env = PushShapesEnv(pusher_shape=emb, image_size=image_size)
    env.agent.control_gap = CONTROL_GAPS[gap]
    obs, _ = env.reset(seed=seed)
    init = env.get_episode_init()
    audit = MechanismAudit(env)
    images, states, engaged, actuator, rewards = [], [], [], [], []
    post_states = []
    max_overflow = max_penetration = 0.0
    try:
        for action in actions:
            images.append(obs["image"])
            states.append(
                np.r_[obs["agent_pos"], obs["agent_angle"], obs["object_pose"]]
            )
            actuator.append(mechanics(env))
            obs, reward, _, _, _ = env.step(action)
            post_states.append(
                np.r_[obs["agent_pos"], obs["agent_angle"], obs["object_pose"]]
            )
            engaged.append(interaction(env))
            audit.observe(env, engaged[-1])
            rewards.append(reward)
            overflow, _ = env._object_arena_metrics()
            max_overflow = max(max_overflow, float(overflow))
            max_penetration = max(
                max_penetration, env._pusher_object_penetration_depth()
            )
        state_error = float(np.max(np.abs(np.asarray(post_states) - expected)))
        valid = (
            state_error < 1e-5
            and rewards[-1] >= 0.95
            and engaged[-1]
            and sum(engaged) >= 20
            and max_overflow < 1e-4
            and max_penetration < 0.6
            and audit.passes(emb)
        )
        return dict(
            valid=valid,
            state_error=state_error,
            max_object_overflow=max_overflow,
            max_pusher_object_penetration=max_penetration,
            images=np.asarray(images),
            states=np.asarray(states),
            engagement=np.asarray(engaged),
            mechanics=np.asarray(actuator),
            mechanism_work=np.asarray(audit.work),
            rewards=np.asarray(rewards),
            init=init,
            goal=np.asarray(env._goal_pose),
        )
    finally:
        env.close()


def write_episode(out, index, emb, gap, seed, actions, data, quality):
    from egomimic.rldb.zarr.zarr_writer import ZarrWriter

    name = f"episode_T_{emb}_obs0_{index:06d}.zarr"
    final = out / name
    if final.exists():
        raise FileExistsError(final)
    staging = out / (".pending-" + uuid.uuid4().hex + ".zarr")
    frames = len(actions)
    env_args = dict(
        object_shape="T",
        pusher_shape=emb,
        obstacle_level=0,
        control_gap=gap,
        collector=COLLECTOR_VERSION,
    )
    description = json.dumps(
        dict(env_args=env_args, version="0.3"), separators=(",", ":")
    )
    quality = dict(
        quality,
        replay_state_error=data["state_error"],
        max_object_overflow=data["max_object_overflow"],
        max_pusher_object_penetration=data["max_pusher_object_penetration"],
    )
    metadata = dict(
        episode_init=json.dumps(data["init"]),
        collector=COLLECTOR_VERSION,
        control_gap=gap,
        control_gap_parameters=CONTROL_GAPS[gap].as_dict(),
        action_spec=list(make_agent(emb).action_spec),
        reset_seed=int(seed),
        quality=quality,
        actions_sha256=hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest(),
        mechanics_spec=[
            "parallel_jaw_gap",
            "umi_open_fraction",
            "chain_joint_angle",
            "flipper_swing_degrees",
            "spring_compression",
        ],
        engagement_timing="after_action",
        state_timing="before_action",
    )
    if os.environ.get("ARTICULATED_SOURCE_SHA256"):
        metadata["source_capsule_sha256"] = os.environ["ARTICULATED_SOURCE_SHA256"]
    metadata["interaction_kind"] = (
        "attachment_constraint"
        if emb in GRASPING
        else "material_in_scoop_pocket"
        if emb == "scoop"
        else "physical_contact"
    )
    numeric = {
        "observations.state": data["states"].astype(np.float32),
        "observations.pusher_cmd_pose": actions[:, :3].astype(np.float32),
        "observations.mechanics": data["mechanics"],
        "actions": actions.astype(np.float32),
        "reward": data["rewards"].astype(np.float32)[:, None],
        "goal_pose": np.tile(data["goal"], (frames, 1)).astype(np.float32),
        "engaged": data["engagement"].astype(np.uint8)[:, None],
        "mechanism_work": data["mechanism_work"].astype(np.uint8)[:, None],
    }
    try:
        writer = ZarrWriter(
            episode_path=staging,
            embodiment=f"pushshapes_sim_{emb}",
            fps=30,
            task_name="pushshapes",
            task_description=description,
            annotations=[
                (
                    "Use the articulated tool to move the T onto the goal pose.",
                    0,
                    frames - 1,
                )
            ],
            chunk_timesteps=frames,
        )
        writer.write(
            numeric_data=numeric,
            image_data={"observations.images.front_img_1": data["images"]},
            metadata_override=metadata,
        )
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return name


def collect(out, emb, gap, target, seed0, max_attempts, max_steps, image_size=96):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(
        collector=COLLECTOR_VERSION,
        embodiment=emb,
        gap=gap,
        target=target,
        seed0=seed0,
        max_attempts=max_attempts,
        max_steps=max_steps,
        image_size=image_size,
    )
    config_path = out / "collection.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        for key in ("collector", "embodiment", "gap", "seed0", "image_size"):
            if previous[key] != config[key]:
                raise ValueError(
                    f"Refusing to mix {key}: {previous[key]} vs {config[key]}"
                )
    atomic_json(config_path, config)
    # Each atomically committed episode is the source of truth after a crash.
    existing = sorted(out.glob(f"episode_T_{emb}_obs0_*.zarr"))
    seeds = set()
    for ep in existing:
        attrs = json.loads((ep / "zarr.json").read_text())["attributes"]
        if (
            attrs.get("collector") != COLLECTOR_VERSION
            or attrs.get("control_gap") != gap
        ):
            raise ValueError(f"foreign or unverified episode in output: {ep}")
        seeds.add(int(attrs["reset_seed"]))
    if len(seeds) != len(existing):
        raise ValueError("Duplicate episode seeds in this shard")
    kept = len(existing)
    next_index = (
        max((int(ep.stem.rsplit("_", 1)[1]) for ep in existing), default=-1) + 1
    )
    progress_path = out / "progress.json"
    attempt = 0
    if progress_path.exists():
        attempt = int(json.loads(progress_path.read_text())["attempts"])
    rejected_replay = 0
    started = time.monotonic()
    while kept < target and attempt < max_attempts:
        seed = seed0 + attempt
        attempt += 1
        if seed in seeds:
            continue
        quality, actions, expected, _, _ = rollout(emb, gap, seed, max_steps)
        if quality["success"]:
            data = replay(emb, gap, seed, actions, expected, image_size)
            if data["valid"]:
                name = write_episode(
                    out, next_index, emb, gap, seed, actions, data, quality
                )
                next_index += 1
                kept += 1
                seeds.add(seed)
                with (out / "episodes.jsonl").open("a") as manifest:
                    manifest.write(json.dumps(dict(episode=name, **quality)) + "\n")
            else:
                rejected_replay += 1
        progress = dict(
            **config,
            kept=kept,
            attempts=attempt,
            replay_or_physics_rejected=rejected_replay,
            elapsed_seconds=time.monotonic() - started,
            complete=kept >= target,
        )
        if quality["success"] or attempt % 10 == 0:
            atomic_json(progress_path, progress)
            print(json.dumps(progress), flush=True)
    progress = dict(
        **config,
        kept=kept,
        attempts=attempt,
        replay_or_physics_rejected=rejected_replay,
        elapsed_seconds=time.monotonic() - started,
        complete=kept >= target,
    )
    atomic_json(progress_path, progress)
    print(json.dumps(progress), flush=True)
    return progress


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--embodiment", choices=ARTICULATED, required=True)
    ap.add_argument("--gap", choices=CONTROL_GAPS, default="ideal")
    ap.add_argument("--target", type=int, default=125)
    ap.add_argument("--seed0", type=int, default=1000000)
    ap.add_argument("--max-attempts", type=int, default=20000)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--image-size", type=int, default=96)
    a = ap.parse_args()
    if min(a.target, a.max_attempts, a.max_steps, a.image_size) <= 0 or a.seed0 < 0:
        ap.error(
            "target, attempts, steps and image size must be positive; seed0 nonnegative"
        )
    result = collect(
        a.out,
        a.embodiment,
        a.gap,
        a.target,
        a.seed0,
        a.max_attempts,
        a.max_steps,
        a.image_size,
    )
    raise SystemExit(0 if result["complete"] else 2)


if __name__ == "__main__":
    main()
