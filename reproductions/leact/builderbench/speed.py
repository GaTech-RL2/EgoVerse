"""Auditable online rollout timings and paired expert/student comparisons.

Request latency includes the full response (reasoning and action), not just the
first token. Legacy runs have request timings but no measured simulator or
episode wall time; those fields stay missing rather than being reconstructed
from simulation time, file timestamps, or batch throughput.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import fmean

from ..data import fingerprint, write_json


def stats(values):
    values = list(values)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Timing/token measurements must be finite and nonnegative')
    if not values:
        return dict(count=0, mean=None, median=None, p95=None, total=None)
    ordered = sorted(values)
    def percentile(q):
        index = (len(ordered) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return dict(count=len(values), mean=fmean(values), median=percentile(.5),
                p95=percentile(.95), total=math.fsum(values))


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator is not None and denominator > 0 else None


def summarize(root: Path):
    """Read immutable call evidence; never modify source rollouts or manifests."""
    root = root.resolve()
    spec = json.loads((root / 'manifest.json').read_text())
    run_sha = fingerprint(spec)
    expected = {(task, seed) for task in spec['task_ids'] for seed in spec['seeds']}
    rows, seen, all_latencies, all_usage = [], set(), [], []
    for path in sorted(root.glob('rollouts/*/seed-*/result.json')):
        result = json.loads(path.read_text())
        key = result['task'], result['seed']
        if key not in expected or key in seen or result['run_sha256'] != run_sha:
            raise ValueError('Unexpected, duplicate or foreign rollout')
        if path.parent.name != f'seed-{key[1]}' or path.parent.parent.name != key[0]:
            raise ValueError('Rollout path disagrees with its identity')
        seen.add(key)
        calls = [json.loads(p.read_text()) for p in sorted((path.parent / 'calls').glob('*.json'))]
        if len(calls) != result['actions'] + result['invalid_outputs']:
            raise ValueError('Completed rollout has missing or extra call evidence')
        attempts = [json.loads(p.read_text()) for p in sorted(
            (path.parent / 'transport-attempts').glob('*/*.json'))]
        for attempt in attempts:
            index = attempt.get('action_index')
            if type(index) is not int or not 0 <= index < len(calls):
                raise ValueError('Transport attempt has no corresponding decision')
            call = calls[index]
            if any(attempt.get(k) != call.get(k) for k in
                   ['run_sha256', 'task', 'seed', 'action_index', 'request_sha256']):
                raise ValueError('Transport attempt belongs to a different request')
            if (attempt.get('accepted') is not False or attempt.get('http_status') == 200
                    or attempt.get('error') != 'NVIDIA inference transport failed'
                    or fingerprint(attempt['request']) != call['request_sha256']):
                raise ValueError('Invalid archived transport attempt')
        latencies, usage = [], []
        for index, call in enumerate(calls):
            if (call.get('run_sha256') != run_sha or call.get('task') != key[0]
                    or call.get('seed') != key[1] or call.get('action_index') != index):
                raise ValueError('Call belongs to a different rollout or decision')
            if call.get('request_sha256') != fingerprint(call['request']):
                raise ValueError('Call request hash mismatch')
            latency = call.get('latency_seconds')
            if latency is not None:
                stats([latency])  # reject NaN, infinity, booleans and negatives
                latencies.append(latency)
            usage.append(call.get('response', {}).get('usage', {}))
        if sum(bool(c.get('accepted')) for c in calls) != result['actions']:
            raise ValueError('Accepted calls disagree with executed action count')
        for attempt in attempts:
            if attempt.get('latency_seconds') is not None:
                stats([attempt['latency_seconds']])
                latencies.append(attempt['latency_seconds'])
            usage.append(attempt.get('response', {}).get('usage', {}))
        requests = len(calls) + len(attempts)
        complete_timing = len(latencies) == requests and bool(requests)
        timing = result.get('timing', {})
        # Replaying cached decisions is useful for recovery, not a fresh speed trial.
        wall = timing.get('rollout_wall_seconds') if not attempts and timing.get('replayed_calls', 0) == 0 else None
        sim = timing.get('simulator_wall_seconds') if not attempts and timing.get('replayed_calls', 0) == 0 else None
        for value in [wall, sim]:
            if value is not None:
                stats([value])
        row = dict(task=key[0], seed=key[1], success=result['success'],
                   cube_progress=result['cube_progress'], actions=result['actions'],
                   requests=requests, transport_failures=len(attempts), invalid_outputs=result['invalid_outputs'],
                   stop_reason=result['stop_reason'], timed_requests=len(latencies),
                   model_seconds=math.fsum(latencies) if complete_timing else None,
                   rollout_wall_seconds=wall, simulator_wall_seconds=sim,
                   mean_decision_seconds=fmean(latencies) if complete_timing else None,
                   prompt_tokens=sum(u.get('prompt_tokens', 0) or 0 for u in usage),
                   completion_tokens=sum(u.get('completion_tokens', 0) or 0 for u in usage))
        rows.append(row)
        all_latencies.extend(latencies)
        all_usage.extend(usage)
    decision = stats(all_latencies)
    model_times = [r['model_seconds'] for r in rows if r['model_seconds'] is not None]
    successes = sum(r['success'] for r in rows)
    prompt_tokens = [u['prompt_tokens'] for u in all_usage if u.get('prompt_tokens') is not None]
    output_tokens = [u['completion_tokens'] for u in all_usage if u.get('completion_tokens') is not None]
    reasoning_tokens = [u['completion_tokens_details']['reasoning_tokens'] for u in all_usage
                        if u.get('completion_tokens_details', {}).get('reasoning_tokens') is not None]
    cached_tokens = [u['prompt_tokens_details']['cached_tokens'] for u in all_usage
                     if u.get('prompt_tokens_details', {}).get('cached_tokens') is not None]
    model_total = math.fsum(model_times) if len(model_times) == len(rows) and rows else None
    return dict(schema_version=1, source=str(root), run_sha256=run_sha, model=spec['model'],
                role=spec.get('role', 'expert' if spec.get('endpoint') else 'unspecified'),
                checkpoint=spec.get('checkpoint'),
                baseline_run_sha256=spec.get('baseline_run_sha256'), seed_ids=spec['seeds'], task_ids=spec['task_ids'],
                label=spec.get('label', root.name),
                complete=seen == expected, planned_rollouts=len(expected), completed_rollouts=len(rows),
                upstream_revision=spec['upstream_revision'], task_metadata=spec['task_metadata'],
                protocol=dict(max_actions=spec['max_actions'],
                              history_observations=spec['history_observations'],
                              candidates_per_decision=spec.get('candidates_per_decision', 1),
                              concurrent_decisions=spec.get('concurrent_decisions', 1),
                              settings=spec['settings'], prompt_version=spec['prompt_version']),
                deployment=spec.get('deployment', dict(endpoint=spec.get('endpoint'),
                    hardware='Provider-managed; undisclosed', cache_policy='Provider-reported usage; cache behavior not assumed')),
                coverage=dict(requests=sum(r['requests'] for r in rows), timed_requests=len(all_latencies),
                              transport_failures=sum(r['transport_failures'] for r in rows),
                              rollout_wall_times=sum(r['rollout_wall_seconds'] is not None for r in rows)),
                full_successes=successes, success_rate=successes / len(rows) if rows else None,
                mean_cube_progress=fmean(r['cube_progress'] for r in rows) if rows else None,
                invalid_outputs=sum(r['invalid_outputs'] for r in rows),
                decision_seconds=decision, rollout_model_seconds=stats(model_times),
                rollout_wall_seconds=stats(r['rollout_wall_seconds'] for r in rows if r['rollout_wall_seconds'] is not None),
                prompt_tokens=stats(prompt_tokens), completion_tokens=stats(output_tokens),
                reasoning_tokens=stats(reasoning_tokens), cached_prompt_tokens=stats(cached_tokens),
                effective_completion_tokens_per_second=(ratio(sum(output_tokens), decision['total'])
                    if len(output_tokens) == len(all_latencies) == len(all_usage) else None),
                successful_rollouts_per_model_hour=ratio(successes * 3600, model_total),
                limitations=['API latency includes network and provider scheduling; provider hardware is unknown.',
                             'Effective tokens/s includes prefill, full reasoning and response delivery; it is not decode-only throughput.',
                             'Missing episode wall time stays unknown; simulation seconds are not wall-clock seconds.',
                             'Archived transport failures count toward request latency and model rollout cost; their token usage may be unknown.',
                             'Collection timings are descriptive; headline speedups require matched student evaluations.'],
                rows=rows)


def compare(reference, student):
    """Ratio of times on identical task/seed cohorts; failures stay in the denominator."""
    for report in [reference, student]:
        if report['protocol']['candidates_per_decision'] != 1 or report['protocol']['concurrent_decisions'] != 1:
            raise ValueError('Online speed comparison requires batch 1 and one candidate per decision')
    if reference['upstream_revision'] != student['upstream_revision']:
        raise ValueError('Simulator revision mismatch')
    for name in ['max_actions', 'history_observations']:
        if reference['protocol'][name] != student['protocol'][name]:
            raise ValueError(f'Rollout protocol mismatch: {name}')
    if reference['role'] == 'base_student':
        for name in ['accelerator', 'accelerator_count', 'precision', 'engine', 'engine_version',
                     'tensor_parallel', 'cache_policy']:
            if reference['deployment'].get(name) is None or reference['deployment'].get(name) != student['deployment'].get(name):
                raise ValueError(f'Uncontrolled base-student comparison: {name}')
        for name in ['settings', 'prompt_version']:
            if reference['protocol'][name] != student['protocol'][name]:
                raise ValueError(f'Uncontrolled base-student comparison: {name}')
    left = {(r['task'], r['seed']): r for r in reference['rows']}
    right = {(r['task'], r['seed']): r for r in student['rows']}
    keys = sorted(left.keys() & right.keys())
    metadata = [{r['id']: r['task_file_sha256'] for r in v['task_metadata']} for v in [reference, student]]
    if any(metadata[0].get(task) != metadata[1].get(task) for task, _ in keys):
        raise ValueError('Task data mismatch')
    if not keys:
        return dict(reference=reference['label'], student=student['label'], matched_rollouts=0,
                    status='No completed matched task/seed pairs')
    def cohort(selected):
        a, b = [left[k] for k in selected], [right[k] for k in selected]
        def total(rows, field):
            values = [r[field] for r in rows]
            return math.fsum(values) if values and all(v is not None for v in values) else None
        model_a, model_b = total(a, 'model_seconds'), total(b, 'model_seconds')
        decisions_a, decisions_b = sum(r['requests'] for r in a), sum(r['requests'] for r in b)
        return dict(rollouts=len(selected), reference_successes=sum(r['success'] for r in a),
                    student_successes=sum(r['success'] for r in b),
                    reference_cube_progress=fmean(r['cube_progress'] for r in a) if a else None,
                    student_cube_progress=fmean(r['cube_progress'] for r in b) if b else None,
                    model_rollout_speedup=ratio(model_a, model_b),
                    decision_speedup=ratio(ratio(model_a, decisions_a), ratio(model_b, decisions_b)),
                    end_to_end_speedup=ratio(total(a, 'rollout_wall_seconds'), total(b, 'rollout_wall_seconds')),
                    reference_requests=decisions_a, student_requests=decisions_b)
    return dict(reference=reference['label'], student=student['label'],
                reference_run_sha256=reference['run_sha256'], student_run_sha256=student['run_sha256'],
                matched_rollouts=len(keys), planned_reference_rollouts=reference['planned_rollouts'],
                planned_student_rollouts=student['planned_rollouts'],
                complete=reference['complete'] and student['complete'] and left.keys() == right.keys(),
                all_matched=cohort(keys), both_successful=cohort([k for k in keys if left[k]['success'] and right[k]['success']]),
                interpretation='Reference time / student time. Ratios above 1 mean faster; assess success and progress alongside them.',
                training_attribution='Astra comparison measures deployment speedup. Isolating LeAct requires the base student on identical hardware and inference settings.')


def publish(base: Path):
    artifacts = base / 'artifacts/builderbench'
    roots = [artifacts / 'expert-probe/gpt6-v2', artifacts / 'teacher-gpt6-v1']
    roots += [p.parent for p in sorted((artifacts / 'evaluations').glob('*/manifest.json'))]
    reports = [summarize(root) for root in roots if (root / 'manifest.json').exists()]
    imported = artifacts / 'imported-speed-reports.json'
    if imported.exists():
        reports += json.loads(imported.read_text())['reports']
    comparisons = []
    references = [r for r in reports if r['label'] == 'teacher-gpt6-v1' or r['role'] == 'base_student']
    for student in [r for r in reports if r['role'] not in {'expert', 'unspecified'}]:
        for reference in references:
            if reference['run_sha256'] == student['run_sha256']:
                continue
            try:
                comparisons.append(compare(reference, student))
            except ValueError as error:
                comparisons.append(dict(reference=reference['label'], student=student['label'],
                                        status='Not comparable: ' + str(error), matched_rollouts=0))
    result = dict(schema_version=1, updated_utc=datetime.now(timezone.utc).isoformat(),
                  reports=reports, comparisons=comparisons,
                  status='Student comparisons available' if any(c['matched_rollouts'] for c in comparisons)
                  else 'Astra timing baseline available; student speedup not measured yet')
    write_json(artifacts / 'speed-report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = publish(args.base)
    print(json.dumps(dict(status=result['status'], reports=len(result['reports']), comparisons=len(result['comparisons']))))


if __name__ == '__main__':
    main()
