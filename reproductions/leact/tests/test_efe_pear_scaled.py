from copy import deepcopy

import pytest

from reproductions.leact.cluster.efe_pear_bridge import NativeInfrastructureError, native_reward, observation_payload
from reproductions.leact.cluster.launch_efe_pear_scaled import candidates, update_budget
from reproductions.leact.cluster.efe_action_loop import native_contract, parse_action


def test_standard22_bimanual_timing_and_joint_bounds_are_preserved():
    lower=[-2.61799,0.,0.,-1.5708,-1.5708,-2.0944,0.]*2
    upper=[3.05433,3.66519,3.66519,1.5708,1.5708,2.0944,1.0421052631578949]*2
    spec={'protocol':'eai-standard/2.2','control_hz':29.41176470588235,'max_chunk_s':10.,
          'cameras':{'top':{'rgb':True}},'action_types':{'joint_position':{'dim':14,
              'layout':[f'joint{i}' for i in range(14)],'limits':{'low':lower,'high':upper}},
              'not_yet':['ee_pose','joint_delta']}}
    contract=native_contract(spec)
    assert contract['control_dt']==pytest.approx(.034)
    assert contract['action_types']['joint_position']['lower']==lower
    assert contract['action_types']['joint_position']['upper']==upper
    action={'action_type':'joint_position','action':[0.]*14,'repeat':1,'summary':'fixture'}
    import json
    assert parse_action(json.dumps(action),contract=contract,allow_stop=True)==action
    with pytest.raises(ValueError,match='disagree'):native_contract({**spec,'control_dt':.02})
    bad=deepcopy(spec);bad['action_types']['joint_position']['lower']=[-1.]*14
    with pytest.raises(ValueError,match='Conflicting'):native_contract(bad)


def test_dexjoco_named_joint_limits_preserve_both_arms_and_finger_order():
    import json
    from pathlib import Path
    spec=json.loads((Path(__file__).parent/'fixtures/dexjoco-native-action-spec.json').read_text())['native_spec']
    # The unbounded Cartesian pose interface is deliberately not assigned
    # invented bounds. Both students can use the same explicit joint interface.
    with pytest.raises(ValueError,match='component bounds'):native_contract(spec)
    contract=native_contract(spec,allowed_action_types=['joint_position'])
    joint=contract['action_types']['joint_position']
    assert joint['dim']==46 and set(contract['action_types'])=={'joint_position'}
    assert joint['layout']==spec['action_types']['joint_position']['layout']
    assert joint['layout'][7]=='ffj0_right' and joint['layout'][30]=='rfj0_left'
    for i,name in enumerate(joint['layout']):
        assert [joint['lower'][i],joint['upper'][i]]==joint['limits'][name]
    action={'action_type':'joint_position','action':[(lo+hi)/2 for lo,hi in zip(joint['lower'],joint['upper'])],
            'repeat':1,'summary':'fixture'}
    assert parse_action(json.dumps(action),contract=contract,allow_stop=True)==action
    for selection in [[],['ee_delta'],['joint_position','joint_position']]:
        with pytest.raises(ValueError,match='missing or invalid'):native_contract(spec,allowed_action_types=selection)


@pytest.mark.parametrize('kind',['runtime','bridge'])
def test_user_pause_prevents_preparation_launch_and_retry(tmp_path,monkeypatch,kind):
    import json
    from reproductions.leact.cluster import launch_efe_pear_scaled as launcher
    monkeypatch.setattr(launcher,'DEST',tmp_path)
    (tmp_path/'control.json').write_text(json.dumps({'paused':True,'reason':'revise curriculum'}))
    def unexpected(*args,**kwargs):
        pytest.fail('A paused campaign must not contact the scheduler or submit jobs')
    monkeypatch.setattr(launcher,'command',unexpected)
    monkeypatch.setattr(launcher,'submit',unexpected)
    done,state=launcher.tick(kind)
    assert done and state['phase']=='paused_by_user'
    for operation in [launcher.prepare_plan,launcher.prepare_runtime,launcher.prepare_bridge,
                      lambda:launcher.launch(kind)]:
        with pytest.raises(RuntimeError,match='paused by the user'):
            operation()


def trial(success=0):
    return {'verifier_result':{'rewards':{'success':success,'resource_enforced':1,'finalized':1}}}


def test_native_failures_are_not_robot_failure_rewards():
    assert native_reward(trial(0),0)==0
    assert native_reward(trial(1),0)==1
    for flag in ['infra_error','grader_error','instance_mismatch','resource_monitor_missing','resource_switch_error']:
        value=trial();value['verifier_result']['rewards'][flag]=1
        with pytest.raises(NativeInfrastructureError):native_reward(value,0)
    for value,returncode in [({},0),(trial(),1),({'exception_info':{'exception_type':'TransportError'},**trial()},0)]:
        with pytest.raises(NativeInfrastructureError):native_reward(value,returncode)


def test_native_deadline_requires_finalized_clean_failure():
    value=trial();value['exception_info']={'exception_type':'AgentTimeoutError'}
    assert native_reward(value,0)==0
    for key,content in [('success',1),('finalized',0),('sim_error',1),('no_verdict',1)]:
        bad=deepcopy(value);bad['verifier_result']['rewards'][key]=content
        with pytest.raises(NativeInfrastructureError):native_reward(bad,0)


def test_candidate_expansion_preserves_holdouts_and_filters_locomotion():
    rows=[{'task':'arm','suite':'mujoco-playground','simulator_runtime':'mujoco'},
          {'task':'panda-pick','suite':'mujoco-playground','simulator_runtime':'mujoco'},
          {'task':'door','suite':'metaworldplus','simulator_runtime':'mujoco'}]
    split={'train_candidates':[r['task'] for r in rows],'held_out_suites':[],
           'held_out_runtimes':[],'protected_core_all_modes':[],'development':[]}
    assert [r['task'] for r in candidates(rows,split)]==['panda-pick','door']
    for field,value in [('held_out_suites',['metaworldplus']),('held_out_runtimes',['mujoco']),
                        ('development',['door']),('protected_core_all_modes',['door'])]:
        with pytest.raises(ValueError):candidates(rows,{**split,field:value})


def test_scaling_uses_task_count_not_group_size_or_reset_multiplicity():
    assert update_budget(40)==225
    assert update_budget(80)==450
    assert update_budget(41)==231
    with pytest.raises(ValueError):update_budget(0)


def test_only_public_single_observation_requests_are_accepted():
    body={'input':[{'role':'user','content':[{'type':'input_text','text':
        '{"mode":"task","action_contract":{"max_repeat":25},"objective":"pick"}'}]}]}
    assert observation_payload(body)[0]['objective']=='pick'
    bad=deepcopy(body);bad['input'].append({'role':'assistant','content':[]})
    with pytest.raises(ValueError):observation_payload(bad)
    bad=deepcopy(body);bad['input'][0]['content'].append({'type':'tool_call'})
    with pytest.raises(ValueError):observation_payload(bad)


def test_real_http_action_exchange_with_synthetic_harbor_fixture(tmp_path):
    """Exercise blocked request/reply, SSE framing and final grade collection.

    The child is explicitly a fake Harbor fixture, not a simulator qualification.
    """
    from pathlib import Path
    import shlex
    import sys
    from reproductions.leact.cluster.efe_pear_bridge import HarborEpisode
    script=tmp_path/'fake_harbor.py'
    script.write_text('''import json, os, pathlib, sys, urllib.request
args=sys.argv
output=pathlib.Path(args[args.index('-o')+1])/args[args.index('--job-name')+1]/'fixture'
body={'input':[{'role':'user','content':[{'type':'input_text','text':json.dumps({
 'mode':'task','objective':'fixture','action_contract':{'max_repeat':25}})}]}]}
request=urllib.request.Request(os.environ['OPENAI_BASE_URL']+'/responses',data=json.dumps(body).encode(),
 headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY'],'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=10) as response: data=response.read().decode()
event=json.loads(data.split('data: ',1)[1].split('\\n')[0])
action=json.loads(event['response']['output'][0]['content'][0]['text'])
assert action=={'stop':True,'summary':'fixture complete'}
output.mkdir(parents=True)
(output/'result.json').write_text(json.dumps({'task_name':'fixture-task','verifier_result':{
 'rewards':{'success':0,'resource_enforced':1,'finalized':1}}}))
''')
    executable=tmp_path/'fake-harbor'
    executable.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(script))+' "$@"\n')
    executable.chmod(0o755)
    source=Path(__file__).resolve().parents[1]/'cluster'
    episode=HarborEpisode({'task':'fixture-task','path':'fake','cutoff_seconds':1},
        {'episode_root':str(tmp_path/'episodes'),'reference':str(tmp_path),'agent_source':str(source),
         'harbor':str(executable),'action_config':{'mode':'task'},'advertise_address':'127.0.0.1'},'test')
    try:
        first=episode.start();assert not first['done']
        result=episode.step({'stop':True,'summary':'fixture complete'})
        assert result['done'] and result['reward']==0 and episode.calls==1
        assert Path(result['native_trial_path']).exists()
    finally:episode.close()
