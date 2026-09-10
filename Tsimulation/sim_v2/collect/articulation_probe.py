"""Small-seed audits before any bulk articulated collection."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np

from ..pushshapes.agents import CONTROL_GAPS
from ..pushshapes.env import PushShapesEnv
from .articulation_controller import GRASPING, ArticulationController
from .articulation_quality import ARTICULATED, MechanismAudit, interaction
from .contact_controller import ContactController


def rollout(emb, gap, seed, max_steps=1000, trace=False, fast_queries=None):
    env = PushShapesEnv(object_shape="T", pusher_shape=emb)
    env.agent.control_gap = CONTROL_GAPS[gap]
    env._skip_obs_render = True
    env.reset(seed=seed)
    if fast_queries is None:
        fast_queries = os.environ.get("ARTICULATED_FAST_SEARCH") == "1"
    fast_queries = bool(fast_queries and emb not in GRASPING)
    if fast_queries:
        from .articulation_fast_search import enable_fast_queries

        enable_fast_queries(env)
    controller = (
        ArticulationController(env) if emb in GRASPING else ContactController(env)
    )
    audit = MechanismAudit(env)
    initial_position = np.array(env._pusher_body.position)
    actions, coverage, held, positions, states = [], [], [], [], Counter()
    initial_coverage = env._coverage()
    started = time.monotonic()
    engagement_start = None
    carry_distance = 0.0
    for step in range(max_steps):
        # Execute the exact f32 action that will be stored and replayed.
        action = controller().astype(np.float32).astype(np.float64)
        if controller.reason:
            break
        if np.any(action[:2] < 0.0) or np.any(action[:2] > env.WORLD_SIZE):
            controller.reason = "unreachable_command"
            break
        obs, reward, term, trunc, info = env.step(action)
        engaged = interaction(env)
        audit.observe(env, engaged)
        if engaged and engagement_start is None:
            engagement_start = np.array(obs["object_pose"][:2])
        if engaged:
            carry_distance = max(
                carry_distance,
                float(np.linalg.norm(obs["object_pose"][:2] - engagement_start)),
            )
        actions.append(action)
        coverage.append(reward)
        held.append(engaged)
        positions.append(
            np.r_[obs["agent_pos"], obs["agent_angle"], obs["object_pose"]]
        )
        states[controller.state] += 1
        overflow, _ = env._object_arena_metrics()
        if overflow > 1e-4:
            controller.reason = "object_outside_arena"
            break
        if trace and (step % 30 == 0 or (engaged and sum(held) == 1)):
            print(
                json.dumps(
                    dict(
                        step=step,
                        state=controller.state,
                        p=np.round(obs["agent_pos"], 2).tolist(),
                        obj=np.round(obs["object_pose"], 2).tolist(),
                        cmd=np.round(action, 2).tolist(),
                        held=engaged,
                        coverage=round(reward, 3),
                    )
                ),
                flush=True,
            )
        if term and engaged and sum(held) >= 20 and carry_distance >= 30:
            break
    aa = np.asarray(actions).reshape(-1, len(env.agent.action_spec))
    velocity = np.diff(np.vstack([initial_position, aa[:, :2]]), axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    jerk = np.linalg.norm(np.diff(velocity, axis=0), axis=1)
    angle_delta = np.abs((np.diff(aa[:, 2]) + np.pi) % (2 * np.pi) - np.pi)
    result = dict(
        embodiment=emb,
        gap=gap,
        seed=seed,
        steps=len(actions),
        engaged=any(held),
        engaged_fraction=float(np.mean(held)) if held else 0.0,
        engaged_steps=sum(held),
        carry_distance=carry_distance,
        initial_coverage=initial_coverage,
        final_coverage=coverage[-1] if coverage else initial_coverage,
        peak_coverage=max(coverage, default=initial_coverage),
        peak_engaged_coverage=max(
            (c for c, h in zip(coverage, held) if h), default=0.0
        ),
        jerk_speed=float(jerk.mean() / max(speed.mean(), 1e-9)) if len(jerk) else 0.0,
        max_position_step=float(speed.max(initial=0)),
        angle_travel=float(angle_delta.sum()),
        max_angle_step=float(angle_delta.max(initial=0)),
        max_grip=float(aa[:, 3].max(initial=0))
        if "grip" in env.agent.action_spec
        else None,
        states=dict(states),
        failure=controller.reason,
        seconds=time.monotonic() - started,
        search_query_shortcuts=fast_queries,
    )
    result.update(audit.metrics())
    result["success"] = bool(
        not controller.reason
        and held
        and held[-1]
        and result["final_coverage"] >= 0.95
        and sum(held) >= 20
        and carry_distance >= 30
        and result["jerk_speed"] < 0.5
        and result["angle_travel"] > 0.01
        and audit.passes(emb)
    )
    env.close()
    return result, aa, np.asarray(positions), np.asarray(held), np.asarray(coverage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embodiments", nargs="+", default=list(ARTICULATED), choices=ARTICULATED
    )
    ap.add_argument("--gaps", nargs="+", default=["ideal"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for emb in args.embodiments:
        for gap in args.gaps:
            for seed in range(args.seed0, args.seed0 + args.seeds):
                result, actions, positions, held, coverage = rollout(
                    emb, gap, seed, args.max_steps, args.trace
                )
                name = f"{emb}_{gap}_{seed}"
                (args.output / f"{name}.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                np.savez_compressed(
                    args.output / f"{name}.npz",
                    actions=actions,
                    positions=positions,
                    engaged=held,
                    coverage=coverage,
                )
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
