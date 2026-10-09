"""Audit the released Bench2Dex policy session without changing predictions."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--active-dof", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import jax
    import yaml
    from robots.active_dof_utils import get_active_dof_info
    from script.policy_model_server import PolicyModelServer
    from script.policy_sessions import load_policy_session

    if not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("Native policy inference requires the allocated GPU")
    dof = get_active_dof_info(args.robot)
    if dof.active_dof != args.active_dof:
        raise ValueError("Released robot map differs from the declared action width")
    robot_registry = yaml.safe_load(
        (args.source / "robots/active_dof_maps.yml").read_text()
    )
    expected_mimics = robot_registry["robots"][args.robot]["mimic_dof"]
    if len(dof.mimic_rules) != expected_mimics:
        raise ValueError("A declared native mimic rule is missing")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "policy_name": "pi05",
        "train_config_name": "pi05_base_dex2bench_full",
        "checkpoint_path": str(args.checkpoint),
        "robot_key": args.robot,
        "use_active_dof": True,
        "train_action_horizon": 20,
        "eval_action_horizon": 20,
    }
    (args.output / "native_policy_config.json").write_text(json.dumps(config, indent=2))
    session = load_policy_session("pi05", config)

    class AuditedSession:
        def __init__(self):
            self.query = 0
            self.episode = 0

        def reset(self, instruction=None, *, seed=None):
            result = session.reset(instruction, seed=seed)
            self.episode += 1
            with (args.output / "resets.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "episode": self.episode,
                            "seed": seed,
                            "instruction": instruction,
                            "unix": time.time(),
                        }
                    )
                    + "\n"
                )
            return result

        def get_action_chunk(self, observation, instruction=None):
            started = time.perf_counter()
            result = session.get_action_chunk(observation, instruction=instruction)
            elapsed = time.perf_counter() - started
            actions = np.asarray(result)
            if actions.shape != (20, args.active_dof) or not np.isfinite(actions).all():
                raise ValueError("Nonfinite or malformed native action chunk")
            np.savez_compressed(
                args.output / f"prediction_{self.query:05d}.npz", actions=actions
            )
            with (args.output / "inference.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "query": self.query,
                            "episode": self.episode,
                            "seconds": elapsed,
                            "shape": list(actions.shape),
                            "teacher_calls": 0,
                            "teacher_tokens": 0,
                        }
                    )
                    + "\n"
                )
            self.query += 1
            return result

        def update_after_action(self, observation, instruction=None):
            return session.update_after_action(observation, instruction=instruction)

        def close(self):
            return session.close()

    server = PolicyModelServer(AuditedSession(), "127.0.0.1", args.port)
    try:
        server.serve_forever()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
