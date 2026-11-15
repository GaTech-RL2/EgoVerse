import math

import pytest

from reproductions.leact.builderbench.speed import compare, stats, summarize
from reproductions.leact.data import fingerprint, write_json


def report():
    return dict(label='expert', role='expert', run_sha256='expert-sha', complete=True,
                planned_rollouts=2, upstream_revision='simulator-revision',
                task_metadata=[dict(id='task', task_file_sha256='task-sha')],
                protocol=dict(max_actions=200, history_observations=16,
                              candidates_per_decision=1, concurrent_decisions=1,
                              settings={'max_tokens': 4096}, prompt_version='v1'),
                deployment=dict(accelerator='H100', accelerator_count=1, precision='bf16',
                                engine='vllm', engine_version='pinned', tensor_parallel=1, cache_policy='fixed'),
                rows=[dict(task='task', seed=i, success=True, cube_progress=1,
                           requests=2, model_seconds=t, rollout_wall_seconds=t+5)
                      for i, t in enumerate([20, 40])])


def test_failed_fast_rollouts_are_retained_and_success_only_cohort_is_explicit():
    teacher, student = report(), report()
    student.update(label='LeAct', role='leact_student', run_sha256='student-sha')
    student['rows'][0].update(model_seconds=10, rollout_wall_seconds=15)
    student['rows'][1].update(model_seconds=2, rollout_wall_seconds=7, success=False, cube_progress=0)
    value = compare(teacher, student)
    # Failure cannot silently disappear from the reported quality or cost.
    assert value['all_matched']['model_rollout_speedup'] == 5
    assert value['all_matched']['reference_successes'] == 2
    assert value['all_matched']['student_successes'] == 1
    assert value['both_successful']['rollouts'] == 1
    assert value['both_successful']['model_rollout_speedup'] == 2


def test_missing_time_or_rollout_never_becomes_zero_or_a_complete_benchmark():
    teacher, student = report(), report()
    student['rows'][1]['model_seconds'] = None
    student['rows'][0]['rollout_wall_seconds'] = None
    result = compare(teacher, student)
    assert result['all_matched']['model_rollout_speedup'] is None
    assert result['all_matched']['end_to_end_speedup'] is None
    student['rows'].pop()
    result = compare(teacher, student)
    assert result['matched_rollouts'] == 1
    assert result['complete'] is False


@pytest.mark.parametrize('field,value', [('candidates_per_decision', 8), ('concurrent_decisions', 32),
                                        ('max_actions', 40), ('history_observations', 8)])
def test_incompatible_rollout_budgets_or_batch_throughput_cannot_claim_online_speedup(field, value):
    teacher, student = report(), report()
    student['protocol'][field] = value
    with pytest.raises(ValueError):
        compare(teacher, student)


def test_different_tasks_or_simulators_cannot_be_paired():
    teacher, student = report(), report()
    student['task_metadata'][0]['task_file_sha256'] = 'different'
    with pytest.raises(ValueError, match='Task data mismatch'):
        compare(teacher, student)
    student = report()
    student['upstream_revision'] = 'different'
    with pytest.raises(ValueError, match='Simulator revision mismatch'):
        compare(teacher, student)


def test_training_attribution_requires_same_student_deployment_and_decoding():
    base, student = report(), report()
    base['role'] = 'base_student'
    assert compare(base, student)['all_matched']['model_rollout_speedup'] == 1
    student['deployment']['accelerator_count'] = 8
    with pytest.raises(ValueError, match='accelerator_count'):
        compare(base, student)
    student = report()
    student['protocol']['settings']['max_tokens'] = 8192
    with pytest.raises(ValueError, match='settings'):
        compare(base, student)


def evidence(tmp_path):
    spec = dict(model='teacher', endpoint='test-endpoint', task_ids=['task'], seeds=[0, 1],
                upstream_revision='sim', task_metadata=[dict(id='task', task_file_sha256='hash')],
                max_actions=200, history_observations=16, settings={}, prompt_version='v1')
    write_json(tmp_path / 'manifest.json', spec)
    sha = fingerprint(spec)
    root = tmp_path / 'rollouts/task/seed-0'
    for index, duration in enumerate([10, 20]):
        request = {'model': 'teacher', 'messages': [{'role': 'user', 'content': str(index)}]}
        write_json(root / 'calls' / f'{index:03d}.json', dict(
            task='task', seed=0, run_sha256=sha, action_index=index, accepted=index == 0,
            request=request, request_sha256=fingerprint(request), latency_seconds=duration,
            response={'usage': dict(prompt_tokens=20, completion_tokens=10,
                                   completion_tokens_details={'reasoning_tokens': 3},
                                   prompt_tokens_details={'cached_tokens': 8})}))
    result = dict(task='task', seed=0, run_sha256=sha, actions=1, invalid_outputs=1,
                  success=False, cube_progress=.5, stop_reason='invalid_model_output')
    write_json(root / 'result.json', result)
    return root, result


def test_legacy_call_timings_include_invalid_output_cost_without_inventing_wall_time(tmp_path):
    evidence(tmp_path)
    r = summarize(tmp_path)
    assert r['complete'] is False
    assert r['completed_rollouts'] == 1
    assert r['planned_rollouts'] == 2
    assert r['decision_seconds']['median'] == 15
    assert r['rollout_model_seconds']['mean'] == 30
    assert r['rollout_wall_seconds']['count'] == 0
    assert r['effective_completion_tokens_per_second'] == 20/30
    assert r['reasoning_tokens']['total'] == 6
    assert r['cached_prompt_tokens']['total'] == 16
    assert r['invalid_outputs'] == 1


def test_cached_replay_cannot_be_reported_as_a_fresh_fast_episode(tmp_path):
    root, result = evidence(tmp_path)
    result['timing'] = dict(rollout_wall_seconds=.01, simulator_wall_seconds=.005, replayed_calls=1)
    write_json(root / 'result.json', result)
    r = summarize(tmp_path)
    assert r['rows'][0]['rollout_wall_seconds'] is None
    assert r['rollout_model_seconds']['mean'] == 30


def test_altered_request_is_rejected_by_timing_import(tmp_path):
    import json
    root, _ = evidence(tmp_path)
    path = root / 'calls/000.json'
    call = json.loads(path.read_text())
    call['request']['model'] = 'different'
    write_json(path, call)
    with pytest.raises(ValueError, match='hash mismatch'):
        summarize(tmp_path)


def test_recovery_preserves_timeout_cost_and_unknown_token_usage(tmp_path):
    import json
    root, result = evidence(tmp_path)
    call = json.loads((root / 'calls/000.json').read_text())
    call.update(accepted=False, error='NVIDIA inference transport failed', latency_seconds=180)
    call.pop('response')
    write_json(root / 'transport-attempts/000/attempt-000.json', call)
    result['timing'] = dict(rollout_wall_seconds=30, simulator_wall_seconds=1)
    write_json(root / 'result.json', result)
    r = summarize(tmp_path)
    assert r['coverage']['requests'] == 3
    assert r['coverage']['transport_failures'] == 1
    assert r['rollout_model_seconds']['mean'] == 210
    assert r['rollout_wall_seconds']['count'] == 0
    assert r['effective_completion_tokens_per_second'] is None
    call['request_sha256'] = 'foreign-request'
    write_json(root / 'transport-attempts/000/attempt-000.json', call)
    with pytest.raises(ValueError, match='different request'):
        summarize(tmp_path)


@pytest.mark.parametrize('value', [math.nan, math.inf, -1, True])
def test_invalid_measurements_are_not_aggregated(value):
    with pytest.raises(ValueError):
        stats([value])
