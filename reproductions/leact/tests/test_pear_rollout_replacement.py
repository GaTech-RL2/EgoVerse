import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from reproductions.leact.cluster import launch_efe_pear_rollout_replacement as recovery


def parent_build(tmp_path):
    parent=tmp_path/'parent';parent.mkdir()
    files={'bootstrap.py':b'print("frozen bootstrap")\n','spec.json':b'{"rollout_nodes": 6}\n'}
    task={'name':'rollout-5','resource':'rollout','command':['python3'],
        'args':['-u','/tmp/rp-jhh-pear-primary/bootstrap.py'],
        'environment':{'EFE_ROLE':'rollout-5','EFE_WORKFLOW_ID':'{{workflow_id}}',
            'EFE_HEAD_HOST':'{{host:trainer}}'},
        'files':[{'localpath':'bootstrap.py','path':'/tmp/rp-jhh-pear-primary/bootstrap.py'}]}
    workflow={'workflow':{'name':'parent','resources':{'rollout':{'gpu':8}},
        'groups':[{'name':'model-cluster','tasks':[{'name':'trainer'},task]}]}}
    files['workflow.yaml']=yaml.safe_dump(workflow).encode()
    for name,raw in files.items():(parent/name).write_bytes(raw)
    (parent/'source-manifest.json').write_text(json.dumps({
        name:hashlib.sha256(raw).hexdigest()for name,raw in files.items()}))
    return parent,files


def test_replacement_keeps_frozen_worker_and_joins_original_attempt(tmp_path,monkeypatch):
    parent,files=parent_build(tmp_path)
    monkeypatch.setattr(recovery,'BUILD_ROOT',tmp_path)
    build=recovery.prepare(parent,'original-workflow','rollout-5','original-head')
    assert(build/'bootstrap.py').read_bytes()==files['bootstrap.py']
    assert(build/'spec.json').read_bytes()==files['spec.json']
    workflow=yaml.safe_load((build/'workflow.yaml').read_text())['workflow']
    assert'groups'not in workflow
    [task]=workflow['tasks']
    assert task['name']=='native'and task['resource']=='rollout'
    assert task['environment']=={'EFE_ROLE':'rollout-5',
        'EFE_WORKFLOW_ID':'original-workflow','EFE_HEAD_HOST':'original-head'}
    assert workflow['resources']['rollout']['gpu']==8
    assert recovery.prepare(parent,'original-workflow','rollout-5','original-head')==build


@pytest.mark.parametrize('role',['trainer','rollout-99'])
def test_replacement_cannot_create_a_second_trainer_or_an_extra_role(tmp_path,monkeypatch,role):
    parent,_=parent_build(tmp_path)
    monkeypatch.setattr(recovery,'BUILD_ROOT',tmp_path)
    with pytest.raises(ValueError,match='existing rollout role'):
        recovery.prepare(parent,'original-workflow',role,'original-head')


def test_replacement_rejects_changed_parent_inputs(tmp_path,monkeypatch):
    parent,_=parent_build(tmp_path)
    monkeypatch.setattr(recovery,'BUILD_ROOT',tmp_path)
    (parent/'bootstrap.py').write_text('print("changed")\n')
    with pytest.raises(ValueError,match='Frozen model startup source changed'):
        recovery.prepare(parent,'original-workflow','rollout-5','original-head')


def test_startup_disables_only_the_failed_optional_ppa_and_preserves_its_contents(tmp_path,monkeypatch):
    cluster=Path(recovery.__file__).parent
    monkeypatch.syspath_prepend(str(cluster))
    spec=importlib.util.spec_from_file_location('primary_package_setup',cluster/'efe_pear_primary_worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sources=tmp_path/'sources';sources.mkdir()
    ppa=b'deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main\n'
    (sources/'deadsnakes.list').write_bytes(ppa)
    ubuntu=b'deb http://archive.ubuntu.com/ubuntu jammy main\n'
    (sources/'ubuntu.sources').write_bytes(ubuntu)
    calls=[]
    def run(args,**kwargs):
        assert not(sources/'deadsnakes.list').exists()
        assert(sources/'ubuntu.sources').read_bytes()==ubuntu
        calls.append(args)
    monkeypatch.setattr(module.subprocess,'run',run)
    retries=tmp_path/'retry.conf'
    module.prepare_system_packages(sources,retries)
    assert(sources/'deadsnakes.list.disabled').read_bytes()==ppa
    assert retries.read_text()=='Acquire::Retries "5";\n'
    assert calls[0]==['apt-get','update','-qq']
