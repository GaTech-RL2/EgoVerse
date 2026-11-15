"""Run a bounded GPT-6 capability probe and retain every public decision/rollout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from ..data import file_sha256, fingerprint, write_json
from . import expert
from .metrics import aggregate
from .preflight import UPSTREAM_REVISION

DEFAULT_TASKS = ['cube-1-task-1', 'cube-1-task-2', 'cube-2-task-1',
                 'cube-2-task-3', 'cube-3-task-2', 'cube-3-task-6']


def utc():
    return datetime.now(timezone.utc).isoformat()


def outcome(env, info):
    flags = [bool(info[f'privileged/block_{i}_success'][0])
             for i in range(env.unwrapped._num_cubes) if env.unwrapped._goal_masks[i]]
    if not flags:
        raise ValueError('Task has no active cube targets')
    effector = bool(info['privileged/effector_success'][0])
    return dict(cube_progress=sum(flags) / len(flags), per_target_success=flags,
                effector_success=effector, success=all(flags) and effector)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', default=DEFAULT_TASKS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--max-actions', type=int, default=40)
    parser.add_argument('--study-kind', choices=['probe', 'teacher_collection'], default='probe')
    args = parser.parse_args()
    if not 1 <= args.max_actions <= 200 or len(set(args.tasks)) != len(args.tasks) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Invalid or duplicate rollout settings')
    reference, output = args.reference.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'collector.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    revision = subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError('Upstream simulator revision changed')
    subprocess.run(['git', '-C', str(reference), 'diff', '--exit-code'], check=True, capture_output=True)
    sys.path.insert(0, str(reference))
    import numpy as np
    from builderbench.creative_cube_env import CreativeCubeEnv
    from builderbench.creative_cube_language_env import CreativeCubeLanguageWrapper, ENV_SYSTEM_PROMPT

    inventory = json.loads((Path(__file__).resolve().parents[1] / 'artifacts/builderbench/preflight/task-inventory.json').read_text())
    tasks = {task['id']: task for task in inventory}
    if set(args.tasks) - tasks.keys():
        raise ValueError('Unknown task')
    for task_id in args.tasks:
        path = reference / 'builderbench/tasks' / f'cube-{tasks[task_id]["cubes"]}.npz'
        if file_sha256(path) != tasks[task_id]['task_file_sha256']:
            raise ValueError('Task data differs from simulator preflight')
    code_files = [Path(__file__), Path(expert.__file__), Path(__file__).with_name('metrics.py'),
                  Path(__file__).resolve().parents[1] / 'astra_expert/client.py']
    spec = dict(model=expert.MODEL, endpoint=expert.ENDPOINT, prompt_version=expert.PROMPT_VERSION,
                settings=expert.SETTINGS, upstream_revision=revision,
                task_ids=args.tasks, seeds=args.seeds, max_actions=args.max_actions,
                history_observations=expert.HISTORY_OBSERVATIONS,
                task_metadata=[tasks[t] for t in args.tasks], system_prompt_sha256=fingerprint(ENV_SYSTEM_PROMPT),
                source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])): file_sha256(p) for p in code_files},
                versions={name: version(name) for name in ['mujoco', 'dm-control', 'gymnasium', 'numpy', 'scipy']},
                purpose=args.study_kind + '; task selection and settings frozen before model calls',
                role='expert', candidates_per_decision=1, concurrent_decisions=1,
                timing_protocol=dict(version=1, clock='time.perf_counter (monotonic)',
                    request='Complete response including reasoning/action, network, provider scheduling and validation',
                    rollout='After environment reset through final decision and simulator step; excludes model loading and reset',
                    simulation='Wall time spent in env.step; simulator-time units never used as wall time',
                    recovery='Cached request replays are flagged; no replayed episode is an end-to-end speed trial'),
                adaptation='Upstream simulator and public observations; structured public-summary/action response, no private reasoning fields',
                failure_policy='Invalid model output ends that episode without action repair; transport/auth/budget failure stops collection',
                scope=f'{len(args.tasks)} tasks × {len(args.seeds)} seeds; {args.study_kind}',
                student_context_audit='Pending separate Qwen3 tokenization; no tasks silently excluded from expert collection')
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError('Existing run has different provenance')
    write_json(manifest, spec)
    run_sha = fingerprint(spec)
    records = []
    def progress(state, **extra):
        record = dict(updated_utc=utc(), model=expert.MODEL, status=state, pid=os.getpid(),
                      run_sha256=run_sha, output=str(output), planned_rollouts=len(args.tasks)*len(args.seeds),
                      planned_tasks=len(args.tasks), inference_seeds=args.seeds, study_kind=args.study_kind,
                      completed_rollouts=len(records), successful_rollouts=sum(r['success'] for r in records),
                      completed=records, **extra)
        write_json(output / 'status.json', record)
        write_json(args.status, record)
    progress('running')
    try:
        for task_id in args.tasks:
            for seed in args.seeds:
                directory = output / 'rollouts' / task_id / f'seed-{seed}'
                final = directory / 'result.json'
                if final.exists():
                    record = json.loads(final.read_text())
                    if record.get('run_sha256') != run_sha:
                        raise ValueError('Rollout belongs to another configuration')
                    records.append(record)
                    continue
                random.seed(seed)
                np.random.seed(seed)
                env = CreativeCubeLanguageWrapper(CreativeCubeEnv(task_id))
                try:
                    observation, info = env.reset(seed=seed)
                    rollout_started = time.perf_counter()
                    simulator_wall, decision_wall, request_wall, replayed_calls = 0.0, 0.0, 0.0, 0
                    history, terminated, truncated = [], False, False
                    stop = None
                    invalid = 0
                    usage = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                    for step in range(args.max_actions):
                        progress('running', task=task_id, seed=seed, action_index=step,
                                 current_outcome=outcome(env, info))
                        call_path = directory / 'calls' / f'{step:03d}.json'
                        replayed = call_path.exists()
                        replayed_calls += int(replayed)
                        decision_started = time.perf_counter()
                        try:
                            response = expert.decide(ENV_SYSTEM_PROMPT, observation, history, call_path,
                                                     cubes=env.unwrapped._num_cubes,
                                                     identity=dict(run_sha256=run_sha, task=task_id, seed=seed, action_index=step))
                        except ValueError:
                            if not call_path.exists():
                                raise
                            failed = json.loads(call_path.read_text())
                            if failed.get('http_status') != 200 or failed.get('accepted'):
                                raise
                            if failed.get('response', {}).get('model') != expert.MODEL:
                                raise
                            for key in usage:
                                usage[key] += failed.get('response', {}).get('usage', {}).get(key, 0) or 0
                            decision_wall += time.perf_counter() - decision_started
                            request_wall += failed['latency_seconds']
                            invalid += 1
                            stop = 'invalid_model_output'
                            break
                        elapsed_decision = time.perf_counter() - decision_started
                        decision_wall += elapsed_decision
                        request_wall += response['latency_seconds']
                        decision = response['result']
                        action, valid = env.check_action_validity(json.dumps(decision['action']))
                        if not valid or action != decision['action']:
                            raise ValueError('Action failed upstream validation after passing our contract')
                        before = observation
                        simulation_started = time.perf_counter()
                        observation, reward, terminated, truncated, info = env.step(action, render=False)
                        elapsed_simulation = time.perf_counter() - simulation_started
                        simulator_wall += elapsed_simulation
                        step_record = dict(observation=before, action=action, decision_summary=decision['decision_summary'],
                                           next_observation=observation, simulator_steps=int(info['action_steps']),
                                           reward=float(reward), terminated=bool(terminated), truncated=bool(truncated),
                                           request_sha256=response['request_sha256'], response_id=response['response'].get('id'),
                                           usage=response['response'].get('usage', {}), outcome=outcome(env, info),
                                           timing=dict(request_latency_seconds=response['latency_seconds'],
                                               decision_wall_seconds=elapsed_decision,
                                               simulator_wall_seconds=elapsed_simulation, replayed_call=replayed))
                        write_json(directory / 'steps' / f'{step:03d}.json', step_record)
                        history.append(step_record)
                        for key in usage:
                            usage[key] += step_record['usage'].get(key, 0) or 0
                        print(json.dumps(dict(task=task_id, seed=seed, action_index=step,
                                              action_type=action['type'], **outcome(env, info))), flush=True)
                        if terminated or truncated:
                            stop = 'environment_terminated' if terminated else 'environment_time_limit'
                            break
                    if stop is None:
                        stop = 'api_action_budget'
                    record = dict(task=task_id, seed=seed, run_sha256=run_sha, model=expert.MODEL,
                                  **outcome(env, info), terminated=bool(terminated), truncated=bool(truncated),
                                  stop_reason=stop, actions=len(history), invalid_outputs=invalid, usage=usage,
                                  timing=dict(rollout_wall_seconds=time.perf_counter() - rollout_started,
                                              simulator_wall_seconds=simulator_wall,
                                              decision_wall_seconds=decision_wall,
                                              model_request_seconds=request_wall, replayed_calls=replayed_calls),
                                  final_observation=observation, updated_utc=utc())
                    write_json(final, record)
                    records.append(record)
                    progress('running')
                finally:
                    env.close()
        metrics = aggregate(records, tasks=args.tasks, seeds=args.seeds,
                            partition='predeclared ' + args.study_kind + ' tasks', scope=spec['scope'])
        metrics.update(model=expert.MODEL, run_sha256=run_sha, invalid_outputs=sum(r['invalid_outputs'] for r in records))
        write_json(output / 'metrics.json', metrics)
        progress('completed', metrics=metrics)
    except Exception as error:
        progress('stopped', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
