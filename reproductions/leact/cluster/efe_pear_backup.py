"""Retry an unallocated standby request without changing a live cluster."""
import hashlib
from pathlib import Path
import shutil

import yaml

from .scale_efe import read,write,command
from .launch_efe_training import freeze_sources
from .efe_pear_submission import submit_reconciled


def maintain(path):
    path=Path(path);state=read(path,{})
    if not state.get('standby')or not state.get('workflow'):return state
    query=command(['osmo','workflow','query',state['workflow'],'--format-type','json'])
    state['scheduler_status']=query['status'];write(path,state)
    # A user cancellation must stay canceled; a running or initializing
    # allocation is preserved. Only an expired, never-started queue request
    # is replaced here.
    if query['status']!='FAILED_QUEUE_TIMEOUT'or query.get('start_time')or state.get('attempt',0)>=32:return state
    old=Path(state['build']);attempt=state.get('attempt',0)+1
    target=old.with_name(old.name.rsplit('-standby-r',1)[0]+'-standby-r'+str(attempt))
    if not(target/'source-manifest.json').exists():
        target.mkdir()
        for name,expected in read(old/'source-manifest.json').items():
            if hashlib.sha256((old/name).read_bytes()).hexdigest()!=expected:raise ValueError('Pinned backup source changed')
            shutil.copyfile(old/name,target/name)
        config=yaml.safe_load((target/'workflow.yaml').read_text())
        config['workflow']['name']='leact-'+target.name
        (target/'workflow.yaml').write_text(yaml.safe_dump(config,sort_keys=False));freeze_sources(target)
    spec=read(target/'spec.json');nodes=1+spec['rollout_nodes']
    receipt=submit_reconciled(target,priority='LOW',pool=state['pool'],nodes=nodes)
    history=state.get('history',[])+[{'workflow':state['workflow'],'status':query['status']}]
    state.update(**receipt,workflow=receipt['name'],build=str(target),attempt=attempt,
        priority='LOW',gpu=8*nodes,scheduler_status='PENDING',history=history,
        allocation_control=spec['allocation_control_base']+'/'+receipt['name'])
    write(path,state);return state
