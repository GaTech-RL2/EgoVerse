import copy
import json

import pytest

from reproductions.leact.cluster import efe_astra_wide_collect as teacher
from reproductions.leact.cluster.efe_astra_wide_collect import collection_plan, public_text


def row(name,reset):
    return {'partition':'train','admitted':True,'already_successful':False,
            'observation_complete':True,'scene_sha256':'scene-'+reset,
            'admission_receipt_sha256':'receipt','config_id':name+'/'+reset,
            'suite':'suite','task':name,'env_args':{'reset_id':reset}}


def test_teacher_balances_tasks_and_labels_repeated_frozen_resets():
    manifest={'partition':'train','qualification':'passed','configs':[
        row('one','0'),row('many','0'),row('many','1'),row('many','2'),row('many','3')]}
    plan=collection_plan(manifest)
    assert len(plan['cases'])==4 and plan['task_count']==2
    one=[c for c in plan['cases']if c['task']=='one']
    many=[c for c in plan['cases']if c['task']=='many']
    assert [c['repeated_reset']for c in one]==[False,True]
    assert [c['config_id']for c in many]==['many/0','many/2']
    assert len({c['case_id']for c in plan['cases']})==4
    assert plan==collection_plan(copy.deepcopy(manifest))


@pytest.mark.parametrize('field,value',[('partition','development'),('admitted',False),
    ('already_successful',True),('observation_complete',False),('scene_sha256','')])
def test_teacher_rejects_unqualified_or_protected_training_scenes(field,value):
    value_row=row('task','0');value_row[field]=value
    with pytest.raises(ValueError):
        collection_plan({'partition':'train','qualification':'passed','configs':[value_row]})


def test_teacher_saves_only_public_output_text():
    assert public_text({'output':[
        {'type':'reasoning','encrypted_content':'private'},
        {'type':'message','channel':'analysis','content':[{'type':'output_text','text':'private'}]},
        {'type':'message','channel':'final','content':[{'type':'output_text','text':'public action'}]}
    ]})=='public action'


def test_invalid_teacher_action_is_preserved_and_never_repaired(tmp_path,monkeypatch):
    import json
    verdict=tmp_path/'verdict.json';verdict.write_text(json.dumps({'success':False,'finalized':True}))
    contract={'protocol':'efe-constrained-action-v3','control_dt':.05,'max_repeat':25,
        'cameras':[],'action_types':{'joint_position':{'dim':2,'layout':['a','b'],
        'lower':[-1,-1],'upper':[1,1]}}}
    received=[]
    class Client:
        uid='episode'
        def __init__(self,*args,**kwargs):pass
        def open(self):return {'done':False,'rewards':[0],'action_contract':contract,
            'public_blocks':[{'type':'text','text':'Public robot state'}]}
        def step(self,action):
            received.append(action)
            return {'done':True,'rewards':[0],'native_trial_path':str(verdict),'executed_control_steps':2}
        def close(self):pass
    monkeypatch.setattr(teacher,'FileEpisodeClient',Client)
    replies=iter([{'action_type':'joint_position','action':[9,0],'repeat':2,'summary':'invalid'},
                  {'action_type':'joint_position','action':[.5,0],'repeat':2,'summary':'valid'}])
    calls=[]
    def response(body,**kwargs):
        calls.append(body)
        return {'public_output':json.dumps(next(replies)),'usage':{'output_tokens':10,'input_tokens':120},'returned_model':teacher.MODEL}
    monkeypatch.setattr(teacher,'api_response',response)
    case={**row('task','0'),'case_id':'case'}
    result=teacher.run_case({'teacher_root':str(tmp_path/'teacher'),'transport_root':'unused',
        'teacher_plan':{'curriculum_sha256':'qualified'},'teacher_endpoint':'unused',
        'teacher_credential_file':'unused'},case)
    assert result['status']=='graded_failure'
    assert result['valid_actions']==1 and result['invalid_actions']==1
    assert received==[{'action_type':'joint_position','action':[.5,0],'repeat':2,'summary':'valid'}]
    assert 'validation_error' in calls[1]['input'][0]['content'][0]['text']
    assert not result['sft_admitted']
    assert result['generated_tokens']==20 and result['input_tokens_sum']==240
    assert result['maximum_input_tokens_per_call']==120


def test_resume_preserves_interrupted_trace_and_cancels_only_its_episode(tmp_path):
    folder=tmp_path/'case/infra-00';folder.mkdir(parents=True)
    transport=tmp_path/'transport';episode=transport/'episodes/own';episode.mkdir(parents=True)
    other=transport/'episodes/other';other.mkdir()
    original={'status':'running','case':{'suite':'native_gr00t_yam'},'episode_id':'own','valid_actions':7}
    (folder/'result.json').write_text(json.dumps(original));(folder/'turns.jsonl').write_text('original\n')
    value=teacher.previous_attempt(folder,{'transport_root':str(transport)},now=123)
    assert value['status']=='infrastructure_failure' and value['valid_actions']==7
    assert value['interrupted_recovery_unix']==123
    assert (folder/'turns.jsonl').read_text()=='original\n'
    assert (episode/'cancel.json').exists() and not (other/'cancel.json').exists()
    assert teacher.previous_attempt(folder,{'transport_root':str(transport)},now=456)==value


@pytest.mark.parametrize('status',['infrastructure_failure','graded_failure','graded_success','ungraded_budget'])
def test_resume_never_overwrites_existing_terminal_attempt(tmp_path,status):
    path=tmp_path/'result.json';original={'status':status,'evidence':'preserve'}
    path.write_text(json.dumps(original));before=path.read_bytes()
    assert teacher.previous_attempt(tmp_path,{})==original
    assert path.read_bytes()==before


@pytest.mark.parametrize('action',[
    {'point':[float('nan'),.2]}, {'bbox':[.8,.1,.2,.4]},
    {'points':[[.1,.2]],'count':True}, {'action_type':'view','view':'side'},
    {'point':[1.1,.1]}, ['not an object'],
])
def test_reasoning_validation_returns_feedback_for_invalid_schema(action):
    with pytest.raises((ValueError,TypeError,KeyError)):
        teacher.reasoning_action(json.dumps(action),family='object_pointing',actions=[])


def test_reasoning_views_enforce_exact_production_budget():
    action={'action_type':'view','view':'overhead'}
    assert teacher.reasoning_action(json.dumps(action),family='active_perception_memory',actions=[action]*2)==action
    with pytest.raises(ValueError,match='budget'):
        teacher.reasoning_action(json.dumps(action),family='active_perception_memory',actions=[action]*3)


def test_balanced_reasoning_pass_includes_opposite_seed_parity():
    rows=[{**row('pointing',str(i)),'suite':'procedural_reasoning','env_args':{'seed':i}}
          for i in range(64)]
    plan=collection_plan({'partition':'train','qualification':'passed','configs':rows})
    assert [c['env_args']['seed']%2 for c in plan['cases']]==[0,1]
