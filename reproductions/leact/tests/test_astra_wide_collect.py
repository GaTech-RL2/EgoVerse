import copy

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
        return {'public_output':json.dumps(next(replies)),'usage':{'output_tokens':10},'returned_model':teacher.MODEL}
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
