import json
import base64
import hashlib
import gzip
from pathlib import Path
import shutil
import subprocess
import threading
from urllib.request import Request,urlopen
from urllib.error import HTTPError

import pytest

from reproductions.leact.tracking.reasoning import allocation,estimate,public_workspace
from reproductions.leact.tracking.export import snapshot
from reproductions.leact.tracking.reasoning import inventory
from reproductions.leact.tracking.reasoning_media import simulator_families
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


def test_family_lists_follow_task_novelty_and_keep_native_splits_pending():
    def row(ident,suite,role='core evaluation',source='EFE',semantic=None):
        return {'id':ident,'suite':suite,'role':role,'source':source,
            'semantic_task':semantic or ident,'description':ident}
    catalog={'entries':[row('train','alpha','PEAR training candidate'),row('new-task','alpha'),
        row('same-task-variant','alpha'),row('unseen-task','beta'),
        row('yam/a','native-gr00t-yam','native candidate; split pending','Native GR00T','pick'),
        row('yam/b','native-gr00t-yam','native candidate; split pending','Native GR00T','pick')]}
    review={'core_entries':[{'task':'new-task','suite':'alpha','cohort':'same_sim_new_task'},
        {'task':'same-task-variant','suite':'alpha','cohort':'same_sim_same_task_new_variant'}],
        'unseen_reserve_entries':[{'task':'unseen-task','suite':'beta'}]}
    result=simulator_families(catalog,review);families={r['id']:r for r in result['families']}
    assert result['counts']=={'train':1,'eval_seen':1,'eval_unseen':1,'pending':1}
    assert [r['id']for r in families['alpha']['eval_seen']]==['new-task']
    assert families['beta']['train']==[]and families['beta']['eval_seen']==[]
    assert families['native-gr00t-yam']['train']==[]and families['native-gr00t-yam']['eval_seen']==[]
    assert [r['id']for r in families['native-gr00t-yam']['pending']]==['pick']
    assert catalog['entries'][2]['role']=='core evaluation'
    review['unseen_reserve_entries'].append({'task':'train','suite':'alpha'})
    with pytest.raises(ValueError,match='simulator categories'):simulator_families(catalog,review)


def test_preview_route_serves_only_content_named_images_inside_media_root(tmp_path):
    media=tmp_path/'media';media.mkdir()
    png=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQAAAABJRU5ErkJggg==')
    name=hashlib.sha256(png).hexdigest()+'.png';(media/name).write_bytes(png)
    private=tmp_path/'private.txt';private.write_text('not a preview')
    linked='a'*64+'.jpg';(media/linked).symlink_to(private)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(tmp_path/'db.sqlite3',media_root=media))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    root=f'http://127.0.0.1:{server.server_port}/reasoning-media/'
    try:
        with urlopen(root+name)as response:
            assert response.read()==png
            assert response.headers['Content-Type']=='image/png'
            assert 'immutable'in response.headers['Cache-Control']
        for path in [linked,'../private.txt','%2e%2e/private.txt','not-content-named.png']:
            with pytest.raises(HTTPError)as error:urlopen(root+path)
            assert error.value.code==404
    finally:
        server.shutdown();server.server_close();thread.join()


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


def test_reasoning_instance_serves_only_its_own_pages_and_data(tmp_path):
    db=tmp_path/'tracker.sqlite3';payload={'title':'rpunamiya reasoning','inventory':{'entries':[]},'notes':'x'*20000}
    ingest(db,{'schema_version':1,'runs':[],'training':[],'status':{'reasoning_workspace':payload}})
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(db,'reasoning'))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    root=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(root+'/')as response:assert b'<title>rpunamiya reasoning</title>'in response.read()
        for route in ['/experiments','/api/snapshot','/api/traces','/app.js','/efe-rollouts.html']:
            with pytest.raises(HTTPError)as error:urlopen(root+route)
            assert error.value.code==404
        with urlopen(root+'/api/health')as response:assert json.load(response)['service']=='reasoning-tracker'
        with urlopen(root+'/api/reasoning')as response:assert json.load(response)==payload
        with urlopen(Request(root+'/api/reasoning',headers={'Accept-Encoding':'gzip'}))as response:
            assert response.headers['Content-Encoding']=='gzip'
            assert json.loads(gzip.decompress(response.read()))==payload
        with urlopen(Request(root+'/api/reasoning',headers={'Accept-Encoding':'gzip;q=0'}))as response:
            assert response.headers.get('Content-Encoding')is None
        with urlopen(root+'/reasoning.js')as response:
            assert b'pearEstimate'in response.read()
            assert "script-src 'self'"in response.headers['Content-Security-Policy']
    finally:
        server.shutdown();server.server_close();thread.join()


def test_public_workspace_excludes_private_crosslinks_and_unrelated_provenance():
    raw={'title':'rpunamiya reasoning','compute_usage':{'private_allocation':8},
         'run':{'id':'moe-warmstart496','revision':'fixed-496','model_path':'/private/leact/model',
                'url':'http://yam-coverage-rpunamiya.nvidia.com:8126/'},
         'notes':['PEAR run','private LeAct lineage'],'reward':.5}
    result=public_workspace(raw)
    assert result['run']=={'id':'moe-warmstart496','revision':'fixed-496'}
    assert result['reward']==.5 and result['notes']==['PEAR run']
    assert 'compute_usage'not in result
