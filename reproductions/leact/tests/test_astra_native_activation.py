import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from reproductions.leact.cluster import activate_astra_native_service as activation
from reproductions.leact.cluster import monitor_astra_wide_collection as watch
from reproductions.leact.tests.test_astra_native_standby import staged_standby


def ready(tmp_path,monkeypatch):
    root_path=Path(activation.PROFILE_ROOT).with_name(activation.OWNER)
    fixed,build=staged_standby(tmp_path,spec_updates={
        'transport_root':str(root_path/'transport'),'allocation_control_base':str(root_path/'allocations'),
        'teacher_credential_file':'/dev/shm/rp_jhh_astra_api_key'})
    origin=json.loads((fixed/'teacher-heartbeat-overlay.json').read_text())['source_build']
    root=tmp_path/'controller';root.mkdir()
    (root/'launches.json').write_text(json.dumps({'original-dojo':{'build':origin}}))
    monkeypatch.setattr(activation,'ROOT',root)
    state={'workflow':'rp-jhh-astra-wide-robodojo-native-only-test','build':str(build),
           'allocation_build':str(build),'native_service_only':True,'native_standby_of':'original-dojo'}
    manifest_sha=hashlib.sha256((build/'source-manifest.json').read_bytes()).hexdigest()
    proof={'runtime_sources_verified':True,'payload_manifest_sha256':manifest_sha,
           'inference_credential_present':False,'existing_job':None,'gpu_processes':'','native_processes':[]}
    calls=[]
    monkeypatch.setattr(activation,'command',lambda *_:{'status':'RUNNING','tasks':[{'status':'RUNNING'}]})
    monkeypatch.setattr(activation,'remote',lambda *args,**kwargs:proof)
    def submit(state,build,**kwargs):
        calls.append((state,build,kwargs));return {'job_id':kwargs['job_id'],'allocation_preserved':True}
    monkeypatch.setattr(activation,'submit',submit)
    recorded=[];monkeypatch.setattr(activation,'record_launch',recorded.append)
    return state,build,proof,calls,recorded


def test_only_verified_idle_native_service_uses_existing_non_replacing_submit(tmp_path,monkeypatch):
    state,build,proof,calls,recorded=ready(tmp_path,monkeypatch)
    result=activation.activate(state)
    assert len(calls)==1 and calls[0][2]['replace_running']is False
    assert calls[0][2]['job_id'].startswith('rp-jhh-astra-native-rpc-')
    assert result['phase']=='native_service_activated'
    assert result['profile_deployment_forbidden']is True
    assert result['native_contract_runtime_check_pending']is True
    assert result['native_activation_proof']==proof and recorded==[result]


@pytest.mark.parametrize('scheduler',[{'status':'PENDING','tasks':[{'status':'SCHEDULING'}]},
    {'status':'RUNNING','tasks':[{'status':'PENDING'}]}])
def test_scheduler_barrier_prevents_any_node_exec(tmp_path,monkeypatch,scheduler):
    state,_,_,calls,_=ready(tmp_path,monkeypatch)
    monkeypatch.setattr(activation,'command',lambda *_:scheduler)
    monkeypatch.setattr(activation,'remote',lambda *args,**kwargs:pytest.fail('No exec during startup barrier'))
    with pytest.raises(RuntimeError,match='not fully running'):activation.activate(state)
    assert not calls


@pytest.mark.parametrize('change',[{'runtime_sources_verified':False},
    {'payload_manifest_sha256':'wrong'},{'inference_credential_present':True},
    {'existing_job':'someone-else'},{'gpu_processes':'123,512 MiB'},{'native_processes':[123]}])
def test_incomplete_source_credential_or_idle_proof_never_activates(tmp_path,monkeypatch,change):
    state,_,proof,calls,_=ready(tmp_path,monkeypatch);proof.update(change)
    with pytest.raises((ValueError,RuntimeError)):activation.activate(state)
    assert not calls


@pytest.mark.parametrize('change',[{'native_service_only':False},{'native_standby_of':None}])
def test_original_or_unrelated_worker_is_never_eligible(tmp_path,monkeypatch,change):
    state,_,_,calls,_=ready(tmp_path,monkeypatch);state.update(change)
    monkeypatch.setattr(activation,'command',lambda *_:pytest.fail('Unrelated allocation is not queried'))
    with pytest.raises(ValueError,match='separately authorized'):activation.activate(state)
    assert not calls


def test_idle_native_only_keeper_is_included_without_a_desired_job(tmp_path,monkeypatch):
    from reproductions.leact.cluster import switch_astra_wide_profile as profile
    controls=tmp_path/'controls';states={}
    for wf,service in [('new-native-only',True),('ordinary-unowned',False)]:
        node=controls/wf/'nodes/native.json';node.parent.mkdir(parents=True)
        node.write_text('{"phase":"waiting_for_job","job_id":null,"child_pid":null}')
        build=tmp_path/(wf+'-build');build.mkdir()
        (build/'spec.json').write_text(json.dumps({'allocation_control_base':str(controls)}))
        states[wf]={'build':str(build),'native_service_only':service}
    monkeypatch.setattr(watch,'TEACHER_ROOT',str(tmp_path/'legacy'))
    monkeypatch.setattr(profile,'PROFILE_ROOT',str(tmp_path/'profile'))
    def execute(host,script,task):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):exec(script,{})
        return json.loads(stream.getvalue().split('EFE_CAPTURE ',1)[1])
    monkeypatch.setattr(watch,'remote',execute)
    captured=watch.snapshot('fully-running-reader',states)
    assert set(captured['nodes'])=={'new-native-only'}
    assert captured['nodes']['new-native-only']['desired_id']is None


def test_existing_watcher_activates_only_the_idle_native_only_row(tmp_path,monkeypatch):
    _,build=staged_standby(tmp_path)
    state={'workflow':'rp-jhh-astra-wide-robodojo-native-only-test','build':str(build),
           'native_service_only':True,'native_standby_of':'original','job_id':None}
    root=tmp_path/'controller';root.mkdir();(root/'launches.json').write_text(json.dumps({state['workflow']:state}))
    monkeypatch.setattr(watch,'ROOT',root);monkeypatch.setattr(watch,'DEST',root)
    monkeypatch.setattr(watch,'record_launch',lambda *_:None)
    monkeypatch.setattr(watch.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,
        stdout=json.dumps({'status':'RUNNING','tasks':[{'status':'RUNNING'}]})))
    monkeypatch.setattr(watch,'snapshot',lambda *_:{'attempt_status':{},'sft_admitted':0,
        'nodes':{state['workflow']:{'node':{'phase':'waiting_for_job'},'desired_id':None}},
        'astra62_profile':{'attempt_status':{},'completed_cases':{},'workers':{}}})
    called=[]
    def activate(row):called.append(row);return {**row,'job_id':'native-fixed'}
    monkeypatch.setattr(activation,'activate',activate)
    result=watch.tick()
    assert len(called)==1 and result['recovery_actions']==[
        {'workflow':state['workflow'],'native_service_job':'native-fixed'}]


@pytest.mark.parametrize('failure',[None,'payload_corrupt','gpu_owned','credential_present'])
def test_real_node_preflight_checks_payload_credentials_and_gpu_before_submit(tmp_path,monkeypatch,failure):
    import shutil
    import subprocess
    import time
    state,build,_,calls,_=ready(tmp_path,monkeypatch)
    lane=tmp_path/'embedded-runtime';shutil.copytree(build,lane)
    # Runtime tar intentionally excludes scheduler-only YAML.
    (lane/'workflow.yaml').unlink()
    control=tmp_path/'allocation-control';(control/'nodes').mkdir(parents=True)
    (control/'nodes/native.json').write_text(json.dumps({'phase':'waiting_for_job','job_id':None,
        'child_pid':None,'updated_unix':time.time()}))
    if failure=='payload_corrupt':(lane/'efe_astra_heartbeat_io.py').write_text('# unexpected change\n')
    monkeypatch.delenv('NVIDIA_INFERENCE_API_KEY',raising=False)
    if failure=='credential_present':monkeypatch.setenv('NVIDIA_INFERENCE_API_KEY','private-test-value')
    def nvidia(args,**kwargs):
        if '--query-gpu=uuid'in args:return 'GPU-assigned-A\nGPU-assigned-B\n'
        return '123, 512 MiB'if failure=='gpu_owned'else''
    monkeypatch.setattr(subprocess,'check_output',nvidia)
    actual_spec=json.loads((build/'spec.json').read_text())
    original_control=actual_spec['allocation_control_base']+'/'+state['workflow']
    def execute(wf,script,task):
        script=script.replace("Path('/tmp/pear-native')",'Path('+repr(str(lane))+')')
        script=script.replace('Path('+repr(original_control)+')','Path('+repr(str(control))+')')
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):exec(script,{})
        return json.loads(stream.getvalue().split('EFE_CAPTURE ',1)[1])
    monkeypatch.setattr(activation,'remote',execute)
    if failure:
        with pytest.raises((ValueError,RuntimeError)):activation.activate(state)
        assert calls==[]
    else:
        result=activation.activate(state)
        assert len(calls)==1
        proof=result['native_activation_proof']
        assert proof['runtime_sources_verified']is True
        assert proof['inference_credential_present']is False
        assert proof['assigned_gpu_uuids']==['GPU-assigned-A','GPU-assigned-B']
        assert proof['payload_files_verified']>5
