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
    # Preserve live allocations and user cancellations. A terminal OSMO
    # controller failure can also end an allocated but idle standby cluster.
    # Those scheduler-ended nodes are gone; this never stops a live cluster.
    expired=query['status']=='FAILED_QUEUE_TIMEOUT'and not query.get('start_time')
    tasks=[t for g in query.get('groups',[])for t in g.get('tasks',[])]
    control_failed=query['status']=='FAILED'and any(t.get('exit_code')==2001 for t in tasks)
    limit=read(path.parent/'control.json',{}).get('backup_allocation_attempts',32)
    if type(limit)is not int or limit<0:raise ValueError('Backup allocation attempt limit must be a nonnegative integer')
    if not(expired or control_failed)or state.get('attempt',0)>=limit:return state
    if control_failed and read(Path(state['build'])/'spec.json',{}).get('standby')is not True:return state
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
