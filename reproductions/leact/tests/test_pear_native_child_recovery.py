import json

import pytest

from reproductions.leact.cluster import efe_pear_native_child_recovery as recovery


@pytest.mark.parametrize('phase,matched,job,pin,expected',[
    ('idle_failed',True,'failed-child','old',True),
    ('running',True,'failed-child','old',False),
    ('idle_failed',False,'failed-child','old',False),
    ('idle_failed',True,'different-child','old',False),
    ('idle_failed',True,'failed-child','new',False),
])
def test_repair_requires_a_known_failed_child_and_an_actual_transport_upgrade(tmp_path,phase,matched,job,pin,expected):
    (tmp_path/'source-manifest.json').write_text(json.dumps({'efe_pear_file_rpc.py':pin}))
    item={'build':str(tmp_path),'job_id':'failed-child'}
    node={'phase':phase,'retired_request_failure':matched,'job_id':job}
    assert recovery.recoverable(item,node,rpc_sha256='new')is expected


def test_paused_campaign_never_reads_or_replaces_cluster_jobs(tmp_path,monkeypatch):
    monkeypatch.setattr(recovery,'DEST',tmp_path)
    (tmp_path/'control.json').write_text(json.dumps({'paused':True}))
    assert recovery.tick({'job_id':'model-job'})is None


def test_failed_simulator_reuses_its_allocation_without_replacing_a_healthy_peer(tmp_path,monkeypatch):
    from reproductions.leact.cluster import efe_pear_allocation_jobs as jobs,monitor_efe_preflight as monitor
    monkeypatch.setattr(recovery,'DEST',tmp_path);monkeypatch.setattr(recovery,'BASE',tmp_path)
    (tmp_path/'cluster').mkdir();(tmp_path/'cluster/efe_pear_file_rpc.py').write_text('new-rpc')
    build=tmp_path/'old';build.mkdir()
    (build/'spec.json').write_text(json.dumps({'allocation_control_base':'/retained','output':'/current/native'}))
    (build/'source-manifest.json').write_text(json.dumps({'efe_pear_file_rpc.py':'old-rpc'}))
    failed={'workflow':'sim-a','build':str(build),'allocation_build':str(build),
        'job_id':'old-child','model_job_id':'model-job','owner_model_workflow':'primary'}
    healthy={**failed,'workflow':'sim-b'}
    (tmp_path/'primary-native-queue.json').write_text(json.dumps({'failed':failed,'healthy':healthy}))
    monkeypatch.setattr(recovery,'command',lambda *a:{'status':'RUNNING','tasks':[{'name':'trainer','status':'RUNNING'}]})
    monkeypatch.setattr(monitor,'remote',lambda *a,**k:{'head_fresh':True,'nodes':{
        'failed':{'phase':'idle_failed','retired_request_failure':True,'job_id':'old-child'},
        'healthy':{'phase':'running','retired_request_failure':True,'job_id':'old-child'}}})
    repaired=tmp_path/'repaired';repaired.mkdir()
    monkeypatch.setattr(recovery,'repair_build',lambda item:repaired)
    calls=[]
    def submit(item,build,**kwargs):
        calls.append((dict(item),build,kwargs));return {'job_id':kwargs['job_id'],'workflow':item['workflow']}
    monkeypatch.setattr(jobs,'submit',submit)
    result=recovery.tick({'workflow':'primary','run':'/current','job_id':'model-job'})
    assert len(calls)==1 and calls[0][0]['workflow']=='sim-a'
    assert calls[0][0]['allocation_build']==str(build)and not calls[0][2].get('replace_running')
    queue=json.loads((tmp_path/'primary-native-queue.json').read_text())
    assert queue['failed']['superseded']and queue['healthy']==healthy
    assert queue['repaired']['workflow']=='sim-a'and queue['repaired']['allocation_build']==str(build)
    assert len(result['repairs'])==1
