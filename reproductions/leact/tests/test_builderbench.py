import math

import pytest

from reproductions.leact.builderbench.metrics import aggregate


def rows():
    return [dict(task='one-cube', seed=0, cube_progress=1.0, success=False),
            dict(task='one-cube', seed=1, cube_progress=0.0, success=False),
            dict(task='ten-cubes', seed=0, cube_progress=0.5, success=False),
            dict(task='ten-cubes', seed=1, cube_progress=1.0, success=True)]


def report(values):
    return aggregate(values, tasks=['one-cube', 'ten-cubes'], seeds=[0, 1],
                     partition='fixture', scope='unit check')


def test_progress_uses_equal_task_weights_and_seed_mean_error_bars():
    value = report(rows())
    # Seed means .75 and .50; this is not std over all four observations.
    assert value['single_progress_mean'] == .625
    assert value['single_progress_std'] == .125
    assert value['best_of_n_progress'] == 1.0
    # All cubes placed does not imply the robot returned to its required pose.
    assert value['full_task_pass_at_n_count'] == 1
    assert value['full_task_pass_at_n_fraction'] == .5
    assert value['inference_rollouts_per_task'] == 2


def test_missing_duplicate_and_extra_runs_cannot_change_denominator():
    with pytest.raises(ValueError, match='Incomplete'):
        report(rows()[:-1])
    with pytest.raises(ValueError, match='Duplicate'):
        report(rows() + [rows()[0]])
    with pytest.raises(ValueError, match='Unexpected'):
        report(rows() + [dict(task='extra', seed=0, cube_progress=1, success=True)])


@pytest.mark.parametrize('progress', [math.nan, math.inf, -0.1, 1.01, True])
def test_invalid_progress_is_rejected(progress):
    values = rows()
    values[0]['cube_progress'] = progress
    with pytest.raises(ValueError, match='Invalid cube progress'):
        report(values)


def test_single_seed_is_not_reported_with_a_measured_variance():
    value = aggregate([rows()[0]], tasks=['one-cube'], seeds=[0],
                      partition='smoke', scope='simulator diagnostic; no model')
    assert value['single_progress_std'] is None
    assert value['full_task_pass_at_n_count'] == 0


def test_inconsistent_success_is_rejected():
    values = rows()
    values[1]['success'] = True
    with pytest.raises(ValueError, match='requires all target cubes'):
        report(values)


def test_teacher_prompt_uses_only_public_history_and_one_recent_explanation():
    import json
    from reproductions.leact.builderbench.expert import payload
    history = [dict(observation=f'scene-{i}', action={'type': 'eef_target'},
                    decision_summary=f'public-summary-{i}', privileged_state='DO_NOT_INCLUDE')
               for i in range(18)]
    system = 'Environment rules. Always output actions as a single, valid JSON object with the specified key/value structure.'
    request = payload(system, 'current scene', history)
    text = json.dumps(request)
    assert 'DO_NOT_INCLUDE' not in text
    assert 'scene-0"' not in text
    assert text.count('public-summary-') == 1
    assert 'public-summary-17' in text
    assert len([m for m in request['messages'] if m['role'] == 'user']) == 16
    assert request['messages'][-1]['content'].startswith('Current Observation:\ncurrent scene')
    assert request['model'] == 'azure/openai/gpt-6-astra'
    assert 'Always output actions as a single' not in request['messages'][0]['content']
    assert '"decision_summary"' in request['messages'][0]['content']
    with pytest.raises(ValueError, match='Upstream output format changed'):
        payload('Changed upstream instructions', 'current scene', history)


@pytest.mark.parametrize('patch', [
    {'cube_id': True}, {'cube_id': 2}, {'grasp_yaw': 2}, {'grasp_yaw': False},
    {'pos': [0.5, math.nan, .02]}, {'pos': [0.5, 0]}, {'yaw': math.inf}, {'unexpected': 1},
])
def test_teacher_actions_reject_invalid_geometry_ids_and_extra_fields(patch):
    from reproductions.leact.builderbench.expert import validate_action
    action = dict(type='pick_and_place', cube_id=0, grasp_yaw=0, pos=[.5, 0, .02], yaw=0.0)
    with pytest.raises(ValueError):
        validate_action({**action, **patch}, 2)


def test_teacher_decoder_preserves_exact_action_and_public_summary():
    import json
    from reproductions.leact.builderbench.expert import decode, MODEL
    action = dict(type='eef_target', pos=[.3, 0, .25], yaw=0, gripper=0)
    value = dict(action=action, decision_summary='Return to the required arm pose.')
    response = dict(model=MODEL, choices=[dict(finish_reason='stop', message=dict(
        role='assistant', content=json.dumps(value), reasoning_content='private provider field'))])
    assert decode(response, 1) == value
    with pytest.raises(ValueError, match='model differs'):
        decode({**response, 'model': 'another-model'}, 1)
    response['choices'][0]['finish_reason'] = 'length'
    with pytest.raises(ValueError, match='truncated'):
        decode(response, 1)


def test_simulator_outcome_uses_target_masks_and_required_arm_pose():
    from types import SimpleNamespace
    from reproductions.leact.builderbench.collect import outcome
    env = SimpleNamespace(unwrapped=SimpleNamespace(_num_cubes=3, _goal_masks=[True, False, True]))
    info = {'privileged/block_0_success': [True], 'privileged/block_1_success': [False],
            'privileged/block_2_success': [True], 'privileged/effector_success': [False]}
    assert outcome(env, info)['cube_progress'] == 1
    assert outcome(env, info)['success'] is False
    info['privileged/effector_success'] = [True]
    assert outcome(env, info)['success'] is True
