"""Run the unchanged native evaluator with extra command and joint-map journals.

The command journal is written before the native physics step. On interruption
its final entry is an attempted command, not proof the simulator applied it.
Native per_episode.jsonl remains the source of completed-step and success data.
"""

import dataclasses
import importlib.metadata
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np


def main():
    source = Path(os.environ["ASTRA_BENCH_SOURCE"])
    output = Path(os.environ["ASTRA_BENCH_AUDIT"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "simulator_environment.json").write_text(
        json.dumps(
            {
                "python": sys.executable,
                "numpy": np.__version__,
                "versions": {
                    n: importlib.metadata.version(n)
                    for n in ("torch", "isaaclab", "h5py", "Flask")
                },
            },
            indent=2,
        )
    )
    sys.path.insert(0, str(source))
    from robots import active_dof_utils
    from utils.inference_recorder import InferenceRecorder

    native_map = active_dof_utils.get_active_dof_info_for_runtime

    def audited_map(robot_key, runtime_joint_names):
        value = native_map(robot_key, runtime_joint_names)
        record = dataclasses.asdict(value)
        record["active_indices"] = record["active_indices"].tolist()
        (output / "runtime_joint_map.json").write_text(json.dumps(record, indent=2))
        return value

    active_dof_utils.get_active_dof_info_for_runtime = audited_map
    native_start = InferenceRecorder.start_episode
    native_record = InferenceRecorder.record_step
    counts = {"reset_episodes_started": 0, "recorded_command_steps": 0}

    def start_episode(self, *args, **kwargs):
        result = native_start(self, *args, **kwargs)
        counts["reset_episodes_started"] += 1
        (output / "recording_started.json").write_text(json.dumps(counts))
        return result

    def record_step(self, *, qpos, action, object_states, camera_frames):
        result = native_record(
            self,
            qpos=qpos,
            action=action,
            object_states=object_states,
            camera_frames=camera_frames,
        )
        with (output / "commands.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "command_index": counts["recorded_command_steps"],
                        "qpos": np.asarray(qpos).tolist(),
                        "action": np.asarray(action).tolist(),
                        "stage": "recorded_before_physics",
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        counts["recorded_command_steps"] += 1
        return result

    InferenceRecorder.start_episode = start_episode
    InferenceRecorder.record_step = record_step
    sys.argv[0] = str(source / "run_policy.py")
    try:
        runpy.run_path(str(source / "run_policy.py"), run_name="__main__")
    finally:
        (output / "recording_counts.json").write_text(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
