"""Recover one failed startup node without restarting the healthy model nodes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from .efe_pear_submission import submit_reconciled
from .launch_efe_pear_scaled import DEST,require_unpaused
from .launch_efe_training import freeze_sources
from .scale_efe import BUILD_ROOT,read,write


PREPARE='''from pathlib import Path
import os,sys
# This optional PPA is unrelated to the Ubuntu packages the model worker uses.
# Keep its definition locally for audit while avoiding its observed HTTP 503.
for p in Path('/etc/apt/sources.list.d').iterdir():
    if 'deadsnakes' in p.name and p.suffix in {'.list','.sources'}:
        p.rename(p.with_name(p.name+'.disabled'))
Path('/etc/apt/apt.conf.d/99rp_jhh_retries').write_text('Acquire::Retries "5";'+chr(10))
os.execv(sys.executable,[sys.executable,'-u','/tmp/rp-jhh-pear-primary/bootstrap.py'])
'''


def prepare(parent,workflow,role,host):
    parent=Path(parent);value=yaml.safe_load((parent/'workflow.yaml').read_text())
    tasks=value['workflow']['groups'][0]['tasks']
    if role=='trainer'or role not in {t['name']for t in tasks}:
        raise ValueError('Only an existing rollout role can be replaced')
    key=hashlib.sha256(json.dumps([workflow,role,host,PREPARE,read(parent/'source-manifest.json')],sort_keys=True).encode()).hexdigest()[:12]
    build=BUILD_ROOT/('rp-jhh-pear-worker-recover-'+role+'-'+key)
    if(build/'source-manifest.json').exists():return build
    build.mkdir()
    for name,digest in read(parent/'source-manifest.json').items():
        if hashlib.sha256((parent/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen model startup source changed')
        shutil.copyfile(parent/name,build/name)
    (build/'recovery.py').write_text(PREPARE)
    task=next(t for t in tasks if t['name']==role)
    task['name']='native';task['resource']='rollout';task['args']=['-u','/tmp/rp-jhh-pear-primary/recovery.py']
    task['environment'].update(EFE_ROLE=role,EFE_WORKFLOW_ID=workflow,EFE_HEAD_HOST=host)
    task['files'].append({'localpath':'recovery.py','path':'/tmp/rp-jhh-pear-primary/recovery.py'})
    value['workflow'].pop('groups')
    value['workflow'].update(name='leact-'+build.name,tasks=[task],
        labels={'pear_run_id':'moe-warmstart496-pearplusplus','pear_phase':'primary_model_recovery'})
    (build/'workflow.yaml').write_text(yaml.safe_dump(value,sort_keys=False));freeze_sources(build)
    return build


def launch(role,host):
    require_unpaused();state=read(DEST/'primary-launch.json')
    if state.get('scheduler_status')!='RUNNING':raise ValueError('The parent model cluster must still be running')
    build=prepare(state['build'],state['workflow'],role,host)
    receipt=submit_reconciled(build,priority='HIGH',pool=state['pool'],nodes=1)
    queue=read(DEST/'primary-native-queue.json',{})
    queue[build.name]={**receipt,'workflow':receipt['name'],'build':str(build),
        'pool':state['pool'],'priority':'HIGH','owner_model_workflow':state['workflow'],
        'kind':'model_rollout_replacement','replaces_role':role}
    write(DEST/'primary-native-queue.json',queue)
    state.setdefault('model_worker_replacements',{})[role]=receipt['name']
    write(DEST/'primary-launch.json',state)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--role',required=True);p.add_argument('--host',required=True)
    a=p.parse_args();print(json.dumps(launch(a.role,a.host)))
