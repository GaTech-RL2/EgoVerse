"""Serial online student rollouts per GPU, retaining actions and measured latency."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys
import time

from ..data import file_sha256, fingerprint, write_json
from .expert import HISTORY_OBSERVATIONS
from .metrics import aggregate
from .student import Backend, SETTINGS, decision_seed, split_output
from .student_data import PROMPT_VERSION, context, forward_prompt, load_corpus
from .simulator import Simulator


def run(args):
    _, corpus = load_corpus(args.corpus)
    import torch
    simulator = Simulator(args.simulator_python, args.reference)
    if simulator.metadata['system_prompt_sha256'] != fingerprint(corpus['system_prompt']):
        raise ValueError('Simulator public prompt differs')
    metadata = corpus['task_metadata']
    for task in metadata:
        if file_sha256(args.reference / 'builderbench/tasks' / f'cube-{task["cubes"]}.npz') != task['task_file_sha256']:
            raise ValueError('Evaluation task data differs from frozen corpus')
    stage = json.loads(args.stage.read_text())
    thinking = stage['method'] != 'bc'
    spec = dict(schema_version=1, model='Qwen/Qwen3-8B', checkpoint=str(args.checkpoint),
        checkpoint_sha256=stage['checkpoint_sha256'],
        role={'base':'base_student','bc':'bc_student','joint':'warmstart_student',
              'leact':'leact_student','exit':'exit_student'}[stage['method']],
        label=stage['label'], stage_id=stage['id'], method=stage['method'], partition=corpus['partition'],
        task_ids=corpus['task_ids'], task_metadata=metadata, seeds=args.seeds,
        upstream_revision=corpus['upstream_revision'], max_actions=200, history_observations=HISTORY_OBSERVATIONS,
        settings=SETTINGS, prompt_version=PROMPT_VERSION + ('-no-thinking' if not thinking else ''),
        candidates_per_decision=1, concurrent_decisions=1,
        deployment=dict(accelerator=torch.cuda.get_device_name(0), accelerator_count=1, precision='bfloat16',
            engine='vllm', engine_version=version('vllm'), tensor_parallel=1, cache_policy='prefix caching disabled',
            workers_per_node=8, note='Independent serial rollout worker on each GPU; eight workers share the node'),
        versions=simulator.metadata['versions'],
        failure_policy='Invalid output or context overflow terminates the episode; all tasks remain in the denominator',
        source_sha256={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')})
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != spec:
        raise ValueError('Evaluation manifest differs')
    # Shards atomically write the same immutable manifest with unique temp paths.
    if not manifest_path.exists():
        temporary = args.output / f'manifest-{args.shard}.tmp'
        temporary.write_text(json.dumps(spec, sort_keys=True, indent=2) + '\n')
        temporary.replace(manifest_path)
    run_sha = fingerprint(spec)
    work_seeds = args.work_seeds or args.seeds
    if not set(work_seeds) <= set(args.seeds):
        raise ValueError('Work seeds must belong to the immutable evaluation plan')
    jobs = [(t, s) for s in work_seeds for t in corpus['task_ids']][args.shard::args.shards]
    backend = None
    for task, seed in jobs:
        directory = args.output / 'rollouts' / task / f'seed-{seed}'
        final = directory / 'result.json'
        if final.exists():
            old = json.loads(final.read_text())
            if old['run_sha256'] != run_sha:
                raise ValueError('Cached evaluation belongs to another model or protocol')
            continue
        if backend is None:
            backend = Backend(args.checkpoint)
            _, warm_seconds = backend.generate('Reply with one short word.', seed=0)
            write_json(args.output / f'startup-{args.shard}.json', dict(load_seconds=backend.load_seconds,
                warmup_seconds=warm_seconds, excluded_from_online_latency=True))
        random.seed(seed)
        try:
            simulation = simulator.request('reset', task=task, seed=seed)
            observation, current_outcome, cubes = simulation['observation'], simulation['outcome'], simulation['cubes']
            started = time.perf_counter()
            history, invalid, replayed = [], 0, 0
            simulator_wall = request_wall = 0.0
            terminated = truncated = False
            stop = 'api_action_budget'
            for index in range(200):
                state = dict(context=context(corpus['system_prompt'], observation, history))
                prompt = forward_prompt(state, thinking=thinking)
                request = dict(prompt=prompt, thinking=thinking, settings=SETTINGS,
                               seed=decision_seed(seed, task, index), model=str(args.checkpoint))
                request_sha = fingerprint(request)
                call_path = directory / 'calls' / f'{index:03d}.json'
                replayed_call = call_path.exists()
                if replayed_call:
                    call = json.loads(call_path.read_text())
                    if call['run_sha256'] != run_sha or call['request_sha256'] != request_sha:
                        raise ValueError('Resume action differs from the original rollout request')
                    replayed += 1
                else:
                    try:
                        generated, latency = backend.generate(prompt, seed=request['seed'], thinking=thinking)
                    except OverflowError:
                        stop = 'context_overflow'
                        break
                    raw = generated[0]
                    try:
                        if raw['finish_reason'] != 'stop':
                            raise ValueError('Generation hit the token budget')
                        reasoning, action = split_output(raw['text'], thinking=thinking, cubes=cubes)
                        if not simulator.request('validate', action=action)['valid']:
                            raise ValueError('Action rejected by simulator')
                        accepted, error = True, None
                    except (ValueError, TypeError, KeyError) as exc:
                        accepted, error, reasoning, action = False, str(exc), '', None
                    call = dict(run_sha256=run_sha, task=task, seed=seed, action_index=index,
                        request=request, request_sha256=request_sha, accepted=accepted, latency_seconds=latency,
                        reasoning=reasoning, action=action, error=error, generated=raw,
                        response=dict(usage=dict(prompt_tokens=raw['prompt_tokens'], completion_tokens=raw['generated_tokens'],
                            total_tokens=raw['prompt_tokens'] + raw['generated_tokens'])))
                    write_json(call_path, call)
                request_wall += call['latency_seconds']
                if not call['accepted']:
                    invalid += 1; stop = 'invalid_model_output'; break
                simulation_started = time.perf_counter()
                before = observation
                simulation = simulator.request('step', action=call['action'])
                observation, current_outcome = simulation['observation'], simulation['outcome']
                reward, terminated, truncated = simulation['reward'], simulation['terminated'], simulation['truncated']
                elapsed = time.perf_counter() - simulation_started
                simulator_wall += elapsed
                record = dict(observation=before, action=call['action'], decision_summary=call['reasoning'],
                    next_observation=observation, reward=float(reward), outcome=current_outcome,
                    simulator_steps=simulation['simulator_steps'], request_sha256=request_sha,
                    timing=dict(request_latency_seconds=call['latency_seconds'], simulator_wall_seconds=elapsed,
                                replayed_call=replayed_call))
                write_json(directory / 'steps' / f'{index:03d}.json', record)
                history.append(record)
                if terminated or truncated:
                    stop = 'environment_terminated' if terminated else 'environment_time_limit'; break
            result = dict(run_sha256=run_sha, task=task, seed=seed, model='Qwen/Qwen3-8B',
                **current_outcome, actions=len(history), invalid_outputs=invalid, stop_reason=stop,
                terminated=bool(terminated), truncated=bool(truncated),
                timing=dict(rollout_wall_seconds=time.perf_counter()-started, simulator_wall_seconds=simulator_wall,
                    model_request_seconds=request_wall, replayed_calls=replayed),
                updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(final, result)
            print(json.dumps(dict(stage=stage['id'], task=task, seed=seed, success=result['success'],
                                  progress=result['cube_progress'], stop=stop)), flush=True)
        finally:
            pass
    simulator.close()
    write_json(args.output / f'shard-{args.shard}-complete.json', dict(run_sha256=run_sha, episodes=len(jobs)))


def summarize(output, corpus_path, seeds):
    _, corpus = load_corpus(corpus_path)
    spec = json.loads((output / 'manifest.json').read_text())
    run_sha = fingerprint(spec)
    rows = [r for p in sorted(output.glob('rollouts/*/seed-*/result.json'))
            if (r := json.loads(p.read_text()))['seed'] in seeds]
    if any(r['run_sha256'] != run_sha for r in rows):
        raise ValueError('Evaluation includes another run')
    report = dict(label=spec['label'], method=spec['method'], stage_id=spec['stage_id'], run_sha256=run_sha,
                  checkpoint=spec['checkpoint'], context_overflows=sum(r['stop_reason']=='context_overflow' for r in rows),
                  invalid_outputs=sum(r['invalid_outputs'] for r in rows), partitions={},
                  evaluated_seeds=seeds, planned_seeds=spec['seeds'], complete=set(seeds)==set(spec['seeds']))
    for name, tasks in dict(all=corpus['task_ids'], **corpus['partition']).items():
        if name == 'rule' or not tasks:
            continue
        report['partitions'][name] = aggregate([r for r in rows if r['task'] in tasks], tasks=tasks, seeds=seeds,
            partition=name, scope='GPT-6-specific fixed task split; context overflows remain in the denominator')
    write_json(output / 'metrics.json', report)
    preview = []
    for path in sorted(output.glob('rollouts/*/seed-*/calls/*.json'), key=lambda p:p.stat().st_mtime)[-6:]:
        call = json.loads(path.read_text())
        preview.append({k:call.get(k) for k in ['task','seed','action_index','accepted','reasoning','action','error','latency_seconds','generated']})
    write_json(output/'preview.json', preview)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['reference', 'corpus', 'checkpoint', 'output', 'stage', 'simulator-python']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--work-seeds', nargs='+', type=int)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=8)
    run(p.parse_args())


if __name__ == '__main__':
    main()
