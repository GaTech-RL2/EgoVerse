import json
from pathlib import Path
import shutil
import subprocess
import threading
from urllib.request import urlopen

import pytest

from reproductions.leact.tracking.reasoning import allocation,estimate
from reproductions.leact.tracking.export import snapshot
from reproductions.leact.tracking.reasoning import inventory
from reproductions.leact.tracking.server import ThreadingHTTPServer,handler,ingest


DEFAULT={'updates':32,'groups':128,'rollouts':8,'control_fraction':.75,'control_tokens':13000,
    'reasoning_tokens':4000,'retention':.65,'rollout_gpus':4,'trainer_gpus':8,
    'simulator_gpus':4,'tokens_per_node_second':2500,'parallel_environments':16,
    'episode_seconds':180,'learner_seconds_per_update':900,'availability':.85,'setup_hours':2}


def test_gpu_count_uses_running_tasks_and_per_task_resource_overrides():
    spec={'workflow':{'resources':{'default':{'gpu':4},'evaluation':{'gpu':1}},
        'groups':[{'tasks':[{'name':'train'},{'name':'eval','resource':'evaluation'},
            {'name':'done'},{'name':'future'}]}]}}
    query={'name':'experiment','pool':'l40s','status':'RUNNING','groups':[{'tasks':[
        {'name':'train','status':'RUNNING'},{'name':'eval','status':'RUNNING'},
        {'name':'done','status':'COMPLETED'},{'name':'future','status':'WAITING'}]}]}
    result=allocation(query,spec)
    assert result['allocated_gpus']==5
    assert [t['allocated_gpus']for t in result['tasks']]==[4,1]


def test_filtering_cost_and_async_occupancy_are_not_hidden_or_double_counted():
    r=estimate(**DEFAULT)
    assert r['retained_episodes']==32768 and r['attempted_episodes']==pytest.approx(32768/.65)
    assert r['gpus_per_run']==16
    assert r['wall_hours']==pytest.approx(2+max(r['phase_hours'].values())/.85)
    assert r['allocated_gpu_hours']==pytest.approx(16*r['wall_hours'])
    slower=estimate(**(DEFAULT|{'retention':.25}))
    assert slower['attempted_episodes']>r['attempted_episodes']and slower['wall_hours']>r['wall_hours']


@pytest.mark.parametrize('change',[{'retention':0},{'retention':1.1},{'availability':0},
    {'control_tokens':-1},{'rollout_gpus':6},{'trainer_gpus':4},{'updates':float('nan')}])
def test_invalid_planning_inputs_are_rejected(change):
    with pytest.raises(ValueError):estimate(**(DEFAULT|change))


def test_reasoning_export_is_small_and_separate_from_legacy_history(tmp_path):
    directory=tmp_path/'artifacts/reasoning-workspace';directory.mkdir(parents=True)
    payload={'title':'rpunamiya reasoning','inventory':{'entries':[{'id':'training-scene'}]}}
    (directory/'workspace.json').write_text(json.dumps(payload))
    value=snapshot(tmp_path,reasoning_only=True)
    assert value['status']['reasoning_workspace']==payload
    assert value['runs']==[]and value['training']==[]
    assert len(json.dumps(value).encode())<8_000_000
    assert 'reasoning_workspace'not in snapshot(tmp_path)['status']


@pytest.mark.parametrize('reserved',['official_core_evaluation','protected_core_all_modes',
    'development','simulator_holdout_reserve'])
def test_catalog_rejects_training_membership_in_reserved_splits(tmp_path,reserved):
    general=tmp_path/'generalization-v2';scaled=tmp_path/'pear-scaled-v1'
    general.mkdir();scaled.mkdir()
    (general/'splits.json').write_text(json.dumps({'train_candidates':['scene'],reserved:['scene']}))
    (scaled/'proposed-candidate-membership-v2.json').write_text(json.dumps({'rows':[{'task':'scene'}]}))
    with pytest.raises(ValueError,match='reserved splits'):inventory(tmp_path)


@pytest.mark.skipif(shutil.which('node')is None,reason='JavaScript validation requires Node')
def test_browser_calculator_matches_server_estimates():
    path=Path(__file__).parents[1]/'tracking/reasoning.js'
    cases=[DEFAULT,DEFAULT|{'updates':512,'rollout_gpus':96,'trainer_gpus':16,'simulator_gpus':16},
           DEFAULT|{'control_tokens':92000,'episode_seconds':1200,'learner_seconds_per_update':3600}]
    script='const f=require(process.argv[1]).pearEstimate;console.log(JSON.stringify(JSON.parse(process.argv[2]).map(f)))'
    result=subprocess.run(['node','-e',script,str(path),json.dumps(cases)],text=True,capture_output=True,check=True)
    for p,client in zip(cases,json.loads(result.stdout)):
        server=estimate(**p)
        for key in ['wall_hours','allocated_gpu_hours','attempted_episodes','generated_tokens']:
            assert client[key]==pytest.approx(server[key])


def test_new_home_page_and_inventory_endpoint_preserve_existing_tracker(tmp_path):
    db=tmp_path/'tracker.sqlite3';payload={'title':'rpunamiya reasoning','inventory':{'entries':[]}}
    ingest(db,{'schema_version':1,'runs':[],'training':[],'status':{'reasoning_workspace':payload}})
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(db,'reasoning'))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    root=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(root+'/')as response:assert b'<title>rpunamiya reasoning</title>'in response.read()
        with urlopen(root+'/experiments')as response:assert b'LeAct experiment tracker'in response.read()
        with urlopen(root+'/api/reasoning')as response:assert json.load(response)==payload
        with urlopen(root+'/reasoning.js')as response:
            assert b'pearEstimate'in response.read()
            assert "script-src 'self'"in response.headers['Content-Security-Policy']
    finally:
        server.shutdown();server.server_close();thread.join()
