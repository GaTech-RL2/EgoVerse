"""Pin upstream task definitions and replay two basic tasks without an LLM."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import runpy
import subprocess
import sys

UPSTREAM_REVISION = 'de9130b983236fc4b97ea8121801d09417e410dc'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference, output = args.reference.resolve(), args.output.resolve()
    revision = subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError('Upstream revision changed; review before updating the pin')
    subprocess.run(['git', '-C', str(reference), 'diff', '--exit-code'], check=True, capture_output=True)
    sys.path.insert(0, str(reference))
    import numpy as np
    from builderbench.creative_cube_env import CreativeCubeEnv
    from builderbench.creative_cube_language_env import CreativeCubeLanguageWrapper, ENV_SYSTEM_PROMPT

    definitions = runpy.run_path(str(reference / 'builderbench/create_task_data.py'))
    task_dir = reference / 'builderbench/tasks'
    task_dir.mkdir(exist_ok=True)
    tasks = []
    for cubes in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50]:
        source = definitions[f'cube_{cubes}_tasks']
        path = task_dir / f'cube-{cubes}.npz'
        np.savez(path, task_names=source['task_names'], episode_lengths=source['episode_lengths'],
                 starts=source['starts'], goals=source['goals'], goal_masks=source['masks'],
                 return_defaults=source['return_defaults'])
        for index, name in enumerate(source['task_names']):
            tasks.append(dict(id=f'cube-{cubes}-task-{index + 1}', name=name, cubes=cubes,
                              active_targets=int(np.asarray(source['masks'][index]).sum()),
                              timestep_budget=int(source['episode_lengths'][index]),
                              return_effector=bool(source['return_defaults'][index]),
                              task_file_sha256=digest(path)))
    if len(tasks) != 51 or len({t['id'] for t in tasks}) != 51:
        raise ValueError('Unexpected task inventory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'task-inventory.json').write_text(json.dumps(tasks, indent=2) + '\n')
    rollouts = []
    for task_id, action_type in [('cube-1-task-1', 'pick_and_place'), ('cube-1-task-2', 'pick_and_hold')]:
        # Upstream macro-action planners also use NumPy's global RNG.
        random.seed(0)
        np.random.seed(0)
        env = CreativeCubeLanguageWrapper(CreativeCubeEnv(task_id))
        try:
            observation, info = env.reset(seed=0)
            target = env.unwrapped.cur_task_info['goal_xyzs'][0].tolist()
            action = dict(type=action_type, cube_id=0, grasp_yaw=0, pos=target, yaw=0.0)
            steps = []
            for index in range(3):
                checked, valid = env.check_action_validity(json.dumps(action))
                if not valid:
                    raise ValueError('Deterministic diagnostic action failed upstream validation')
                before = observation
                observation, reward, terminated, truncated, info = env.step(checked, render=False)
                steps.append(dict(observation=before, action=checked, next_observation=observation,
                                  simulator_steps=int(info['action_steps']), reward=float(reward),
                                  terminated=bool(terminated), truncated=bool(truncated)))
                if terminated or truncated:
                    break
                # Correct the measured cube error, including controller/grasp offset.
                # This is a diagnostic feedback policy, not an expert demonstration.
                correction = np.asarray(target) - info['privileged/block_0_pos']
                action = (dict(type='eef_target', pos=[0.3, 0.0, 0.25], yaw=0.0, gripper=0.0)
                          if action_type == 'pick_and_place' else
                          dict(type='eef_target', pos=(np.asarray(action['pos']) + correction).tolist(),
                               yaw=0.0, gripper=1.0))
            successes = [bool(info[f'privileged/block_{i}_success'][0])
                         for i in range(env.unwrapped._num_cubes) if env.unwrapped._goal_masks[i]]
            effector = bool(info['privileged/effector_success'][0])
            rollouts.append(dict(task=task_id, seed=0, policy='seeded diagnostic with cube-position feedback; no model',
                                 steps=steps, cube_progress=sum(successes) / len(successes),
                                 effector_success=effector, success=all(successes) and effector,
                                 terminated=bool(terminated), truncated=bool(truncated)))
        finally:
            env.close()
    (output / 'smoke-rollouts.json').write_text(json.dumps(rollouts, indent=2) + '\n')
    passed = all(r['success'] and r['terminated'] for r in rollouts)
    report = dict(updated_utc=datetime.now(timezone.utc).isoformat(), passed=passed,
                  upstream='https://github.com/rajghugare19/builderbench', upstream_revision=revision,
                  task_count=len(tasks), simulator_replays=len(rollouts),
                  simulator_successes=sum(r['success'] for r in rollouts), expert_calls=0,
                  diagnostic_revision=2, diagnostic_source_sha256=digest(Path(__file__)),
                  versions={name: version(name) for name in ['mujoco', 'dm-control', 'gymnasium', 'numpy', 'scipy']},
                  task_source_sha256=digest(reference / 'builderbench/create_task_data.py'),
                  system_prompt_sha256=hashlib.sha256(ENV_SYSTEM_PROMPT.encode()).hexdigest(),
                  dependency_lock_sha256=digest(reference / 'uv.lock'),
                  interpretation='Infrastructure check only; this does not establish any model or expert capability.')
    (output / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit('Simulator replay needs inspection')


if __name__ == '__main__':
    main()
