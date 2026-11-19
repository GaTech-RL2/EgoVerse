import fcntl
import io
import json
import threading
import time

import pytest

from reproductions.leact.cluster import efe_astra_wide_collect as teacher
from reproductions.leact.cluster.switch_astra_wide_profile import profile_spec,PROFILE_ROOT
from reproductions.leact.cluster.launch_astra_wide_collection import OWNER,TEACHER_ROOT


def test_exact_chat_profile_preserves_multimodal_content_and_budget():
    profile=teacher.model_profile({'teacher_profile_id':'astra62-xhigh-v1'})
    result=teacher.request_body(profile,prompt='unchanged prompt',inputs=[{'role':'user','content':[
        {'type':'input_text','text':'public observation'},
        {'type':'input_image','image_url':'data:image/png;base64,IMAGE','detail':'auto'}]}],max_output_tokens=32768)
    assert result=={'model':'openai/openai/gpt-6.2-astra','messages':[
        {'role':'system','content':'unchanged prompt'},
        {'role':'user','content':[{'type':'text','text':'public observation'},
           {'type':'image_url','image_url':{'url':'data:image/png;base64,IMAGE','detail':'auto'}}]}],
        'stream':False,'store':False,'max_completion_tokens':32768,'reasoning_effort':'xhigh'}


def test_chat_usage_counts_reasoning_output_and_discards_private_payloads(tmp_path,monkeypatch):
    key=tmp_path/'key';key.write_text('test-secret');key.chmod(0o600)
    source={'id':'reply','model':'openai/openai/gpt-6.2-astra','choices':[{'finish_reason':'stop',
        'message':{'content':'public JSON','reasoning_content':'PRIVATE','analysis':'PRIVATE'}}],
        'usage':{'prompt_tokens':700,'completion_tokens':140,'total_tokens':840,
                 'completion_tokens_details':{'reasoning_tokens':120}}}
    requests=[]
    def urlopen(request,**kwargs):requests.append(request);return io.BytesIO(json.dumps(source).encode())
    monkeypatch.setattr(teacher.urllib.request,'urlopen',urlopen)
    reply=teacher.api_response({'model':source['model'],'messages':[],'reasoning_effort':'xhigh'},
                              endpoint='https://example.invalid/v1',credential_file=key)
    assert requests[0].full_url=='https://example.invalid/v1/chat/completions'
    assert reply['usage']['output_tokens']==140
    assert reply['usage']['output_tokens_details']['reasoning_tokens']==120
    assert reply['usage']['input_tokens']==700
    assert reply['requested_effort']=='xhigh' and reply['fallback_allowed']is False
    assert 'PRIVATE'not in json.dumps(reply)


def test_wrong_model_never_executes_fallback_action(tmp_path,monkeypatch):
    contract={'protocol':'efe-constrained-action-v3','control_dt':.05,'max_repeat':25,'cameras':[],
        'action_types':{'joint_position':{'dim':1,'layout':['a'],'lower':[-1],'upper':[1]}}}
    class Client:
        uid='test'
        def __init__(self,*a,**kw):pass
        def open(self):return {'done':False,'rewards':[0],'action_contract':contract,
                              'public_blocks':[{'type':'text','text':'public observation'}]}
        def step(self,*a):pytest.fail('A different model must not command the simulator')
        def close(self):pass
    monkeypatch.setattr(teacher,'FileEpisodeClient',Client)
    monkeypatch.setattr(teacher,'api_response',lambda *a,**kw:{'returned_model':teacher.MODEL,
        'public_output':'{"action_type":"joint_position","action":[0],"repeat":1,"summary":"public"}',
        'usage':{'input_tokens':10,'output_tokens':20}})
    result=teacher.run_case({'teacher_root':str(tmp_path),'teacher_profile_id':'astra62-xhigh-v1',
        'transport_root':'unused','teacher_plan':{'curriculum_sha256':'source'},'teacher_endpoint':'unused',
        'teacher_credential_file':'unused'},
        {'suite':'native_gr00t_yam','task':'task','case_id':'case','env_args':{}})
    assert result['status']=='infrastructure_failure' and result['valid_actions']==0
    assert result['generated_tokens']==20 and result['requested_effort']=='xhigh'
    turn=json.loads((tmp_path/'episodes/case/infra-00/turns.jsonl').read_text())
    assert turn['model_identity_mismatch']is True


def test_profile_root_changes_without_changing_native_recipe():
    old={'owner_model_workflow':OWNER,'teacher_root':TEACHER_ROOT,'teacher_plan':{'cases':[{'scene':'train'}]},
         'transport_root':'existing native transport','teacher_task_names':['task'],'max_calls':96,'wall_seconds':1500}
    new=profile_spec(old)
    assert new['teacher_root']==PROFILE_ROOT and new['teacher_profile_id']=='astra62-xhigh-v1'
    assert new['legacy_teacher_root']==TEACHER_ROOT
    for name in ['transport_root','teacher_task_names','max_calls','wall_seconds']:
        assert new[name]==old[name]
    assert new['teacher_plan']['cases']is old['teacher_plan']['cases']
    with pytest.raises(ValueError):profile_spec({**old,'owner_model_workflow':'hero'})


def test_recovery_keeps_selected_profile_and_never_restarts_legacy_collector(tmp_path,monkeypatch):
    import yaml
    from reproductions.leact.cluster import monitor_astra_wide_collection as watch
    old=tmp_path/'old';old.mkdir();future=tmp_path/'future';future.mkdir()
    (future/'submission.json').write_text('{}')
    spec={'owner_model_workflow':OWNER,'teacher_root':TEACHER_ROOT,'teacher_plan':{'cases':[]},
          'max_calls':96,'wall_seconds':1500,'transport_root':'preserve_native_transport'}
    (old/'spec.json').write_text(json.dumps(spec))
    (old/'workflow.yaml').write_text(yaml.safe_dump({'workflow':{'tasks':[{'image':'pinned-image','command':['python']}],
        'resources':{'default':{'gpu':2}}}}))
    root=tmp_path/'control';root.mkdir();(root/'active-profile.json').write_text('{"teacher_profile_id":"astra62-xhigh-v1"}')
    monkeypatch.setattr(watch,'ROOT',root);monkeypatch.setattr(watch,'frozen_files',lambda _: {})
    captured={}
    def build(candidate,*args,**kwargs):captured.update(candidate);return future
    monkeypatch.setattr(watch,'worker_build',build)
    assert watch.refresh_build({'workflow':'teacher','build':str(old),'label':'native-yam'})==future
    assert teacher.model_profile(captured)['model']=='openai/openai/gpt-6.2-astra'
    assert captured['teacher_native_only']is True and captured['teacher_root']==PROFILE_ROOT
    for key in ['max_calls','wall_seconds','transport_root']:assert captured[key]==spec[key]


def test_boundary_guard_keeps_future_legacy_cases_locked_after_completion(tmp_path):
    legacy=tmp_path/'legacy';new=tmp_path/'new';transport=tmp_path/'transport'
    cases=[{'task':'task','case_id':'one'},{'task':'task','case_id':'two'}]
    current=legacy/'episodes/one';current.mkdir(parents=True)
    old=(current/'writer.lock').open('a');fcntl.flock(old,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guards=teacher.LegacyCaseGuards({'teacher_root':str(new),'legacy_teacher_root':str(legacy),
        'transport_root':str(transport),'teacher_profile_id':'astra62-xhigh-v1'},cases)
    other=(legacy/'episodes/two/writer.lock').open('a')
    with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old.close();guards.wait_task('task')
    with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
    guards.close();fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB);other.close()


def test_unclaimed_legacy_open_can_retire_without_cancelling_native_owner(tmp_path):
    legacy=tmp_path/'legacy';new=tmp_path/'new';transport=tmp_path/'transport';uid='a'*32
    folder=legacy/'episodes/case';attempt=folder/'infra-00';attempt.mkdir(parents=True)
    lock=(folder/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (attempt/'result.json').write_text(json.dumps({'status':'opening','episode_id':uid}))
    episode=transport/'episodes'/uid;episode.mkdir(parents=True)
    pending=transport/'pending';pending.mkdir();(pending/(uid+'.json')).write_text('{"kind":"own"}')
    guards=teacher.LegacyCaseGuards({'teacher_root':str(new),'legacy_teacher_root':str(legacy),
        'transport_root':str(transport),'teacher_profile_id':'astra62-xhigh-v1'},[{'task':'task','case_id':'case'}])
    thread=threading.Thread(target=guards.wait_task,args=('task',));thread.start()
    deadline=time.monotonic()+2
    while not(episode/'infrastructure-error.json').exists()and time.monotonic()<deadline:time.sleep(.01)
    assert (episode/'profile-superseded-pending.json').exists()
    assert not(episode/'cancel.json').exists()
    # The old collector finalizes its infrastructure record before releasing
    # its writer lock; emulate that boundary rather than inventing a score.
    (attempt/'result.json').write_text('{"status":"infrastructure_failure"}')
    lock.close();thread.join(timeout=3);assert not thread.is_alive();guards.close()
