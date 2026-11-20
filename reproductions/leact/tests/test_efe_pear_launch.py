import hashlib
import json
from types import SimpleNamespace

import pytest

from reproductions.leact.cluster import scale_efe
from reproductions.leact.cluster import launch_efe_pear_model_check as models


def test_multi_node_submission_accounts_capacity_and_preserves_pool(tmp_path,monkeypatch):
    root=tmp_path/'campaign';root.mkdir();builds=tmp_path/'builds';builds.mkdir()
    build=builds/'job';build.mkdir();payload=b'workflow: {}\n'
    (build/'workflow.yaml').write_bytes(payload)
    (build/'source-manifest.json').write_text(json.dumps({'workflow.yaml':hashlib.sha256(payload).hexdigest()}))
    (root/'compute-authorization.json').write_text(json.dumps({'max_nodes_per_job':16}))
    monkeypatch.setattr(scale_efe,'ROOT',root);monkeypatch.setattr(scale_efe,'BUILD_ROOT',builds)
    calls=[];reservations=[]
    def command(argv,**kwargs):
        calls.append(argv)
        return {'workflows':[]} if 'list' in argv else {'name':'fixture-16-nodes'}
    monkeypatch.setattr(scale_efe,'command',command)
    monkeypatch.setattr(scale_efe,'require_capacity',lambda active,additional_nodes:reservations.append(additional_nodes))
    monkeypatch.setattr(scale_efe.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='',stderr=''))
    receipt=scale_efe.submit(build,pool='groot-h100-02',priority='LOW',nodes=16)
    assert reservations==[16]
    assert json.loads((root/'node-reservations.json').read_text())=={'fixture-16-nodes':16}
    assert calls[-1][calls[-1].index('--pool')+1]=='groot-h100-02'
    assert calls[-1][calls[-1].index('--priority')+1]=='LOW'
    # Re-running acknowledged submissions must not submit or consume capacity twice.
    assert scale_efe.submit(build,pool='groot-h100-02',nodes=16)==receipt
    assert reservations==[16] and len(calls)==2
    with pytest.raises(ValueError,match='per-job'):
        scale_efe.submit(build,nodes=17)


def test_ambiguous_prior_submission_requires_reconciliation(tmp_path,monkeypatch):
    monkeypatch.setattr(scale_efe,'ROOT',tmp_path);monkeypatch.setattr(scale_efe,'BUILD_ROOT',tmp_path)
    build=tmp_path/'job';build.mkdir();(build/'submission-intent.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Reconcile'):
        scale_efe.submit(build)


def test_model_checks_wait_without_allocating_gpus_and_obey_user_pause(tmp_path,monkeypatch):
    monkeypatch.setattr(models,'DEST',tmp_path)
    def unexpected(*args,**kwargs):pytest.fail('No allocation is allowed before runtime readiness')
    monkeypatch.setattr(models,'submit',unexpected);monkeypatch.setattr(models,'command',unexpected)
    done,state=models.tick()
    assert not done and state['phase']=='waiting_for_shared_runtime' and state['primary_rl_updates']==0
    (tmp_path/'control.json').write_text(json.dumps({'paused':True}))
    done,state=models.tick()
    assert done and state['phase']=='paused_by_user'


def test_interrupted_catalog_shard_cannot_silently_omit_a_suite(tmp_path,monkeypatch):
    from reproductions.leact.cluster import launch_efe_pear_catalog as catalog
    from reproductions.leact.cluster import launch_efe_coverage as coverage
    monkeypatch.setattr(catalog,'DEST',tmp_path)
    monkeypatch.setattr(catalog,'SHARDS',[['robocasa','dexjoco','vlabench']])
    monkeypatch.setattr(catalog,'require_unpaused',lambda:None)
    monkeypatch.setattr(coverage,'infra_retryable',lambda _:False)
    build=tmp_path/'previous';build.mkdir()
    spec={'adapter_revision':'native22-cdi-v1','tasks':[{'suite':'dexjoco'}]}
    (build/'spec.json').write_text(json.dumps(spec))
    (tmp_path/'catalog-queues.json').write_text(json.dumps({'0':{'attempts':[{
        'workflow':'failed-dex','build':str(build),'pool':'groot-h100-01'}]}}))
    (tmp_path/'catalog-contracts.json').write_text(json.dumps({'robocasa':{'success':True}}))
    monkeypatch.setattr(catalog,'command',lambda _: {'status':'FAILED'})
    worker={'tasks':[{'suite':'dexjoco','success':False,'error':'fixture'}]}
    monkeypatch.setattr(catalog.subprocess,'run',lambda *a,**kw:SimpleNamespace(
        stdout='PEAR_CATALOG_PROBE '+json.dumps(worker)+'\n'))
    selected=[]
    def prepare(shard,index,only_suites):
        selected.extend(only_suites);return build
    monkeypatch.setattr(catalog,'prepare',prepare)
    monkeypatch.setattr(catalog,'submit',lambda *a,**kw:{'name':'vla-first-corrected-attempt'})
    done,state=catalog.tick()
    assert not done and selected==['vlabench']
    assert state['contracts']['robocasa']['success']
    assert state['contracts']['dexjoco']['error']=='fixture'
