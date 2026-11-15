"""BuilderBench progress and pass@N, following LeAct Appendix D.1.

The task/seed denominator is supplied explicitly. Missing runs never disappear
from an aggregate, and full-task success remains distinct from cube progress.
"""
from __future__ import annotations

import math
from statistics import fmean, pstdev


def aggregate(rollouts, *, tasks, seeds, partition, scope):
    tasks, seeds = list(tasks), list(seeds)
    if not tasks or not seeds or len(set(tasks)) != len(tasks) or len(set(seeds)) != len(seeds):
        raise ValueError('Expected nonempty, unique task and inference-seed lists')
    if not partition or not scope:
        raise ValueError('Name the partition and evaluation scope')
    expected = {(task, seed) for task in tasks for seed in seeds}
    values = {}
    for row in rollouts:
        key = row['task'], row['seed']
        if key not in expected:
            raise ValueError(f'Unexpected task/seed: {key}')
        if key in values:
            raise ValueError(f'Duplicate task/seed: {key}')
        progress = row['cube_progress']
        if isinstance(progress, bool) or not isinstance(progress, (int, float)) or not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError(f'Invalid cube progress: {key}')
        if not isinstance(row['success'], bool):
            raise ValueError(f'Full-task success must be a boolean: {key}')
        if row['success'] and progress != 1:
            raise ValueError(f'Full success requires all target cubes placed: {key}')
        values[key] = row
    missing = expected - values.keys()
    if missing:
        raise ValueError(f'Incomplete evaluation: {len(missing)} task/seed rows missing')

    # Average tasks equally, including tasks with different numbers of cubes.
    per_seed = [dict(seed=seed, progress=fmean(values[task, seed]['cube_progress'] for task in tasks))
                for seed in seeds]
    per_task = [dict(task=task,
                     single_progress=fmean(values[task, seed]['cube_progress'] for seed in seeds),
                     best_of_n_progress=max(values[task, seed]['cube_progress'] for seed in seeds),
                     full_successes=sum(values[task, seed]['success'] for seed in seeds),
                     full_pass_at_n=any(values[task, seed]['success'] for seed in seeds))
                for task in tasks]
    passed = sum(row['full_pass_at_n'] for row in per_task)
    return dict(partition=partition, scope=scope, task_count=len(tasks),
                inference_rollouts_per_task=len(seeds), rollout_count=len(values),
                single_progress_mean=fmean(row['progress'] for row in per_seed),
                single_progress_std=pstdev(row['progress'] for row in per_seed) if len(seeds) > 1 else None,
                std_definition='Population std across inference-seed cross-task means; not training seeds',
                best_of_n_progress=fmean(row['best_of_n_progress'] for row in per_task),
                full_task_pass_at_n_count=passed, full_task_pass_at_n_fraction=passed / len(tasks),
                per_seed=per_seed, per_task=per_task)
