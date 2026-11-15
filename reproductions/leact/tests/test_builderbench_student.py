import json
from pathlib import Path

import pytest

from reproductions.leact.builderbench.student import action_reward, split_output
from reproductions.leact.builderbench.student_data import context, forward_prompt, backward_prompt, pairs


ACTION = dict(type='pick_and_place', cube_id=0, grasp_yaw=0, pos=[.5,0,.02], yaw=0)


def state():
    return dict(context='Public scene without the target action', expert_action=ACTION, cubes=2)


def test_action_reward_matches_centimeter_threshold_and_meter_offset():
    assert action_reward(ACTION, ACTION) == 1
    assert action_reward({**ACTION, 'pos':[.505,0,.02]}, ACTION) == 1
    assert action_reward({**ACTION, 'pos':[.6,0,.02]}, ACTION) == pytest.approx(.9)
    assert action_reward({**ACTION, 'cube_id':1}, ACTION) == .1
    assert action_reward({**ACTION, 'type':'pick_and_hold'}, ACTION) == 0
    assert action_reward(None, ACTION) == 0
    eef = dict(type='eef_target', pos=[.3,0,.2], yaw=0, gripper=0)
    assert action_reward(eef, eef) == 1


def test_forward_input_does_not_leak_current_expert_action_or_summary():
    row = state(); row['reasoning'] = 'EXPERT_EXPLANATION_SENTINEL'
    assert json.dumps(ACTION, sort_keys=True) not in forward_prompt(row)
    assert row['reasoning'] not in forward_prompt(row)
    assert json.dumps(ACTION, sort_keys=True) in backward_prompt(row)
    history = [dict(observation=f'obs-{i}', action={'prior':i}, decision_summary='PRIVATE_OR_EXPERT_SENTINEL',
                    privileged='SIMULATOR_INTERNAL_SENTINEL') for i in range(19)]
    value = context('rules', 'current', history)
    assert 'obs-0\n' not in value and 'obs-18' in value
    assert 'PRIVATE_OR_EXPERT' not in value and 'SIMULATOR_INTERNAL' not in value


def test_epf_and_exit_keep_their_intended_targets():
    row = state(); other = {**ACTION, 'cube_id':1}
    joint = pairs(row, 'Place the visible cube.', method='leact')
    assert len(joint) == 2
    assert joint[0]['conversations'][-1]['value'].endswith(json.dumps(ACTION, sort_keys=True))
    assert joint[1]['conversations'][-1]['value'].endswith('DONE')
    exit_data = pairs(row, 'Place the visible cube.', method='exit', action=other)
    assert len(exit_data) == 1
    assert exit_data[0]['conversations'][-1]['value'].endswith(json.dumps(other, sort_keys=True))
    bc = pairs(row, 'MUST_NOT_BE_TRAINED', method='bc')
    assert 'MUST_NOT_BE_TRAINED' not in json.dumps(bc)


@pytest.mark.parametrize('text', [json.dumps(ACTION), '<think>unfinished',
    '<think>valid</think>```json\n' + json.dumps(ACTION) + '\n```',
    '<think>one</think><think>two</think>' + json.dumps(ACTION)])
def test_reasoning_parser_rejects_malformed_and_unterminated_answers(text):
    with pytest.raises(ValueError):
        split_output(text, cubes=2)


def test_reasoning_and_action_only_parsers_preserve_exact_actions():
    text = '<think>Use the visible target.</think>\n' + json.dumps(ACTION)
    assert split_output(text, cubes=2) == ('Use the visible target.', ACTION)
    assert split_output('<think>\n</think>\n' + json.dumps(ACTION), thinking=False, cubes=2) == ('', ACTION)
    with pytest.raises(ValueError):
        split_output(text, thinking=False, cubes=2)


def test_delta_selection_rejects_invalid_baseline_and_nonpositive_candidates(tmp_path):
    from reproductions.leact.data import fingerprint, file_sha256, write_json, write_jsonl
    from reproductions.leact.builderbench.iterate import select_and_export
    states = [dict(state(), state_id='s1', task='a'), dict(state(), state_id='s2', task='a')]
    corpus = dict(states=2, states_sha256=fingerprint(states))
    write_json(tmp_path/'corpus/manifest.json', corpus); write_jsonl(tmp_path/'corpus/states.jsonl', states)
    rows=[]
    for index, s in enumerate(states):
        rows.append(dict(state_id=s['state_id'], method='leact', corpus_sha256=corpus['states_sha256'],
            baseline={'valid':index==0}, candidates=[dict(candidate_id=0, valid=True, reasoning='reason',
                action=ACTION, delta=0.0 if index==0 else 1.0, reward=1.0)]))
    shard=tmp_path/'generation/shard-00.jsonl'; write_jsonl(shard,rows)
    write_json(shard.with_suffix('.complete.json'),{'sha256':file_sha256(shard)})
    assert select_and_export(tmp_path/'corpus',tmp_path/'generation',tmp_path/'sft',method='leact',tokenizer=None) is None
    assert (tmp_path/'sft/no_admissions.json').exists()


def test_positive_delta_keeps_expert_target_and_context_gate(tmp_path):
    from reproductions.leact.data import fingerprint, file_sha256, write_json, write_jsonl
    from reproductions.leact.builderbench.iterate import select_and_export
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs): return [1,2,3]
    states=[dict(state(),state_id='s1',task='a')]
    corpus=dict(states=1,states_sha256=fingerprint(states))
    write_json(tmp_path/'corpus/manifest.json',corpus);write_jsonl(tmp_path/'corpus/states.jsonl',states)
    rows=[dict(state_id='s1',method='leact',corpus_sha256=corpus['states_sha256'],baseline={'valid':True},
        candidates=[dict(candidate_id=i,valid=True,reasoning='reason',action={**ACTION,'cube_id':1},delta=d,reward=1)
                    for i,d in enumerate([.2,.1,-.1])])]
    shard=tmp_path/'generation/shard-00.jsonl';write_jsonl(shard,rows)
    write_json(shard.with_suffix('.complete.json'),{'sha256':file_sha256(shard)})
    result=select_and_export(tmp_path/'corpus',tmp_path/'generation',tmp_path/'sft',method='leact',tokenizer=Tokenizer())
    assert result['sft_examples']==4 and result['selected_traces']==2
    data=json.loads((tmp_path/'sft/data.json').read_text())
    assert all(p['conversations'][-1]['value'].endswith(json.dumps(ACTION,sort_keys=True)) for p in data[::2])


def test_fixed_observation_speed_requires_identical_prompts_and_keeps_failures():
    from reproductions.leact.builderbench.fixed_observations import paired
    reference=dict(stage='base',action_match_reward=.5,invalid=1,rows=[
        dict(state_id='a',prompt_sha256='x',latency_seconds=4),
        dict(state_id='b',prompt_sha256='y',latency_seconds=6)])
    student=dict(stage='joint-e3',action_match_reward=.7,invalid=0,rows=[
        dict(state_id='a',prompt_sha256='x',latency_seconds=1),
        dict(state_id='b',prompt_sha256='y',latency_seconds=4)])
    value=paired(reference,student)
    assert value['speedup']==2 and value['reference_invalid']==1 and value['states']==2
    student['rows'][1]['prompt_sha256']='different'
    with pytest.raises(ValueError,match='identical'):
        paired(reference,student)
