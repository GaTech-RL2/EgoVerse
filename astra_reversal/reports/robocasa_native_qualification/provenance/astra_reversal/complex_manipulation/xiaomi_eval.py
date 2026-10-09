"""Instrument the unmodified Xiaomi native evaluator for a registered screen.

The upstream evaluator retains its observation history, input processor, socket
client, action decoding and rollout loop. Wrappers only record observations,
timings, reset accounting and finite action checks; they do not steer the policy.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np

from astra_reversal.complex_manipulation.worker import write_json
from astra_reversal.records import digest


def summarize(rows, intended):
    identities = [(r["cohort"], r["task"], r["seed"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate physical episode identity")
    completed = [r for r in rows if r["completed"]]
    successes = sum(r["success"] for r in completed)
    return {"intended_episodes": intended, "recorded_episodes": len(rows),
            "completed_episodes": len(completed), "successes": successes,
            "completed_episode_sr": successes / len(completed) if completed else None,
            "intended_batch_sr": successes / intended if len(completed) == intended else None,
            "policy_control_steps": sum(r["steps"] for r in rows),
            "reset_free_action_chunks": sum(r["policy_queries"] for r in rows),
            "astra_calls": 0, "astra_tokens": 0, "episodes": rows}


class Recorder:
    def __init__(self, root, cohort, task, horizon):
        self.root, self.cohort, self.task, self.horizon = root, cohort, task, horizon
        self.directory = None
        self.rows = []

    def reset(self, env, obs, seed):
        import imageio.v2 as imageio
        self.directory = self.root / f"seed{seed}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.seed, self.started, self.queries, self.steps = seed, time.perf_counter(), 0, 0
        self.policy_seconds, self.env_seconds = 0.0, 0.0
        raw = env.unwrapped.env
        xml = raw.sim.model.get_xml().encode()
        state = raw.sim.get_state().flatten()
        write_json(self.directory / "reset.json", {
            "task": self.task, "seed": seed, "instruction": obs["annotation.human.task_description"],
            "raw_observation_sha256": digest(obs), "state_sha256": digest(state),
            "model_sha256": hashlib.sha256(xml).hexdigest(), "initial_success": bool(raw._check_success()),
            "horizon": self.horizon, "execution_prefix": 16, "extra_stabilization_steps": 0,
        })
        np.savez_compressed(self.directory / "initial_observation.npz", **obs, simulator_state=state)
        cameras = [obs[f"video.robot0_{c}"] for c in ("agentview_left", "agentview_right", "eye_in_hand")]
        imageio.imwrite(self.directory / "starting_image.png", np.concatenate(cameras, axis=1))
        if raw._check_success():
            raise ValueError("An initially successful reset is not a valid policy success")

    def finish(self, result):
        row = {**result, "cohort": self.cohort, "task": self.task, "horizon": self.horizon,
               "completed": bool(result["success"] or result["steps"] == self.horizon),
               "policy_queries": self.queries, "explicit_episode_resets": 1,
               "physical_retries": 0, "policy_seconds": self.policy_seconds,
               "environment_seconds": self.env_seconds, "wall_seconds": time.perf_counter() - self.started}
        # Upstream may stop on environment termination before its explicit horizon.
        # The step wrapper records that condition, so it is a completed failure.
        row["completed"] = row["completed"] or self.terminated
        write_json(self.directory / "result.json", row)
        self.rows.append(row)
        return row


class RecordedEnvironment:
    def __init__(self, env, recorder):
        self.env, self.recorder = env, recorder

    def reset(self, *, seed):
        obs, info = self.env.reset(seed=seed)
        self.recorder.terminated = False
        self.recorder.reset(self.env, obs, seed)
        return obs, info

    def step(self, action):
        started = time.perf_counter()
        value = self.env.step(action)
        self.recorder.env_seconds += time.perf_counter() - started
        self.recorder.steps += 1
        self.recorder.terminated = bool(value[2] or value[3])
        return value

    def close(self):
        self.env.close()


class RecordedClient:
    def __init__(self, client, recorder):
        self.client, self.recorder = client, recorder

    def infer(self, states, images, instruction):
        start = time.perf_counter()
        actions = self.client.infer(states, images, instruction)
        seconds = time.perf_counter() - start
        if actions.ndim != 2 or actions.shape[1] != 12 or len(actions) < 16 or not np.isfinite(actions).all():
            raise ValueError("Native policy action contract violated")
        r = self.recorder
        r.queries += 1
        r.policy_seconds += seconds
        with (r.directory / "policy_queries.jsonl").open("a") as stream:
            stream.write(json.dumps({"query": r.queries, "control_step": r.steps,
                "seconds": seconds, "shape": list(actions.shape), "actions_sha256": digest(actions),
                "min": float(actions.min()), "max": float(actions.max())}) + "\n")
        # This record is sufficient for a monitoring client without reading logs.
        write_json(r.directory / "progress.json", {"queries": r.queries, "control_steps": r.steps,
                                                    "elapsed_seconds": time.perf_counter() - r.started})
        return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    repo = Path("/opt/astra-xiaomi/sources/xiaomi")
    sys.path.insert(0, str(repo / "eval_robocasa365"))
    import entry
    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action

    native = entry.EvalClient("/opt/astra-xiaomi/weights", "127.0.0.1", 10086, "robocasa365", .95)
    write_json(args.output / "environment.json", {"versions": {
        name: importlib.metadata.version(name) for name in
        ("torch", "torchvision", "transformers", "flash-attn", "numpy", "mujoco", "robosuite", "robocasa")},
        "evaluator_sha256": hashlib.sha256((repo / "eval_robocasa365/entry.py").read_bytes()).hexdigest(),
        "protocol": protocol, "native_inference_modified": False})
    rows, constructors = [], []
    intended = sum(len(g["episodes"]) for g in protocol["groups"])
    try:
        for group in protocol["groups"]:
            task = group["task"]
            if get_task_horizon(task) != group["horizon"]:
                raise ValueError("Simulator task horizon differs from the registered protocol")
            output = args.output / group["id"]
            record = Recorder(output, group["cohort"], task, group["horizon"])
            # Proxy the gym module, leaving the upstream loop and constructor kwargs intact.
            class GymProxy:
                def make(self, *positional, **kwargs):
                    ctor = {"group": group["id"], "task": task, "status": "started",
                            "constructor_seed": kwargs["seed"], "documented_setup_resets": None}
                    constructors.append(ctor)
                    write_json(args.output / "constructors.json", constructors)
                    if group["cohort"] == "previous_pi05_seeds":
                        np.random.seed(group["base_seed"])
                        kwargs["disable_env_checker"] = True
                    env = gym.make(*positional, **kwargs)
                    ctor.update(status="ready", documented_setup_resets=1)
                    write_json(args.output / "constructors.json", constructors)
                    return RecordedEnvironment(env, record)

            native_args = entry.parse_args([])
            for key, value in {"split": "pretrain", "seed": group["base_seed"],
                               "num_trials": group["native_num_trials"], "save_videos": True,
                               "video_stride": 2, "video_fps": 10}.items():
                setattr(native_args, key, value)
            # Hook the upstream stats writes to finalize the corresponding episode
            # immediately, retaining results even if a later episode is interrupted.
            original_dump = entry.json.dump
            def record_dump(value, stream, **kwargs):
                result = original_dump(value, stream, **kwargs)
                if isinstance(value, dict) and value.get("env_name") == task and "episodes" in value:
                    rows.append(record.finish(value["episodes"][-1]))
                    write_json(args.output / "summary.json", summarize(rows, intended))
                return result
            entry.json.dump = record_dump
            try:
                entry.evaluate_task(task, group["task_index"], native_args,
                    RecordedClient(native, record), GymProxy(), get_task_horizon,
                    convert_action, output, episode_indices=group["episodes"], show_progress=False)
            finally:
                entry.json.dump = original_dump
        write_json(args.output / "summary.json", summarize(rows, intended))
    finally:
        native.close()


if __name__ == "__main__":
    main()
