"""Validate generated task definitions in LIBERO; this is not policy evaluation."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from .ood_extensions import parse, sha, validate


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(task_directory, libero_root, output, task_limit=None):
    import numpy as np
    from PIL import Image

    from .config import BenchmarkConfig
    from .libero_runner import configure_libero

    if output.exists():
        raise FileExistsError("Refusing to replace a validation run")
    output.mkdir(parents=True)
    configure_libero(
        libero_root,
        BenchmarkConfig.preset("libero_goal_ood"),
        output / "runtime_config",
    )
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from robosuite.utils.errors import RandomizationError

    manifest = json.loads((task_directory / "manifest.json").read_text())
    tasks = manifest["tasks"][:task_limit] if task_limit else manifest["tasks"]
    (output / "previews").mkdir()
    (output / "states").mkdir()
    rows = []
    started = time.perf_counter()
    for index, task in enumerate(tasks):
        bddl = task_directory / task["bddl"]
        assert sha(bddl) == task["bddl_sha256"]
        tree = parse(bddl.read_text())
        validate(tree)
        actual = robosuite_parse_problem(str(bddl))
        assert " ".join(actual["language_instruction"]).lower() == task["instruction"]
        assert actual["goal_state"] == [
            [x.lower() for x in atom] for atom in task["goal_atoms"]
        ]
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
            control_freq=20,
        )
        try:
            for seed in (71, 73):
                env.seed(seed)
                retries = 0
                while True:
                    try:
                        observation = env.env.reset()
                        break
                    except RandomizationError:
                        retries += 1
                        if retries >= 5:
                            raise
                assert not env.check_success(), "Task is already solved at reset"
                for _ in range(10):
                    observation, _, _, _ = env.step(np.zeros(7))
                assert not env.check_success(), "Task solved during stabilization"
                initial = np.asarray(env.get_sim_state()).copy()
                assert np.isfinite(initial).all()
                key = f"task_{index:02d}_seed{seed}"
                state_path = output / "states" / (key + ".npy")
                np.save(state_path, initial, allow_pickle=False)
                camera_hashes = {}
                for camera in ("agentview_image", "robot0_eye_in_hand_image"):
                    frame = observation[camera]
                    assert frame.shape == (256, 256, 3) and frame.dtype == np.uint8
                    assert float(frame.std()) > 1, "Blank render"
                    camera_hashes[camera] = hashlib.sha256(frame.tobytes()).hexdigest()
                preview = None
                if seed == 71:
                    image_path = output / "previews" / (key + ".png")
                    Image.fromarray(
                        np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
                    ).save(image_path)
                    preview = {
                        "path": str(image_path.relative_to(output)),
                        "sha256": sha(image_path),
                        "transform": "both image axes reversed, matching policy display convention",
                    }
                target = env.env.object_states_dict[task["destination"]]
                source = env.env.get_object(task["source_object"])
                target_position = np.asarray(target.get_geom_state()["pos"]).copy()
                positive = None
                # A predicate witness checks that the declared goal can be true.
                # It is not a controller trajectory or a reachability certificate.
                for dz in np.linspace(0.18, -0.04, 221):
                    qpos = np.r_[target_position + [0, 0, dz], [1, 0, 0, 0]]
                    env.sim.data.set_joint_qpos(source.joints[0], qpos)
                    env.sim.data.qvel[:] = 0
                    env.sim.forward()
                    if env.check_success():
                        positive = {
                            "source_qpos": qpos.tolist(),
                            "source_center_minus_target_z": float(dz),
                        }
                        break
                assert (
                    positive is not None
                ), "No contact/predicate witness found in prescribed vertical scan"
                env.set_state(initial)
                env.sim.forward()
                assert (
                    not env.check_success()
                ), "Restored unsolved state became successful"
                # Round trip verifies that these saved states are usable for replay.
                assert np.array_equal(np.asarray(env.get_sim_state()), initial)
                rows.append(
                    {
                        "task_id": task["id"],
                        "instruction": task["instruction"],
                        "seed": seed,
                        "bddl_sha256": sha(bddl),
                        "reset_retries": retries,
                        "initial_success": False,
                        "after_10_idle_steps_success": False,
                        "state": {
                            "path": str(state_path.relative_to(output)),
                            "sha256": sha(state_path),
                        },
                        "camera_array_sha256": camera_hashes,
                        "preview": preview,
                        "constructed_goal_witness": positive,
                        "restored_initial_success": False,
                        "state_roundtrip_exact": True,
                    }
                )
                write_json(
                    output / "progress.json",
                    {"completed_reset_checks": len(rows), "rows": rows},
                )
        finally:
            env.close()
        print(
            json.dumps(
                {
                    "task": task["id"],
                    "validated_resets": 2,
                    "remaining_tasks": len(tasks) - index - 1,
                }
            ),
            flush=True,
        )
    receipt = {
        "schema_version": "astra-ood-simulator-validation-1.0",
        "status": "passed",
        "complete_task_coverage": len(tasks) == len(manifest["tasks"]),
        "task_count": len(tasks),
        "reset_checks": len(rows),
        "seeds": [71, 73],
        "rows": rows,
        "generator_manifest_sha256": sha(task_directory / "manifest.json"),
        "validator_sha256": sha(Path(__file__)),
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "mujoco", "robosuite")
            },
        },
        "wall_seconds": time.perf_counter() - started,
        "policy_rollouts": 0,
        "provider_calls": 0,
        "scope": "Actual parser, simulator construction, bounded reset attempts, ten zero-action stabilization steps, camera rendering, initial-failure check, constructed goal-predicate witness and exact state restore. Witnesses teleport the source object and are not successful policy trajectories or proof that a robot can reach the goal.",
    }
    write_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-limit", type=int)
    args = parser.parse_args()
    value = run(
        args.tasks.resolve(),
        args.libero_root.resolve(),
        args.output.resolve(),
        args.task_limit,
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "tasks": value["task_count"],
                "resets": value["reset_checks"],
                "policy_rollouts": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
