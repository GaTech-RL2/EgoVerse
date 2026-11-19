"""Repair a recognized simulator transport failure inside retained allocations."""
import hashlib
import time
from pathlib import Path

import yaml

from .scale_efe import BASE,read,write,command
from .launch_efe_pear_scaled import DEST


def repair_build(item):
    from .launch_efe_pear_primary import worker_build
    from .recover_efe_pear_graphics import frozen_files
    old=Path(item['build']);spec=read(old/'spec.json');files=frozen_files(old)
    files['efe_pear_file_rpc.py']=BASE/'cluster/efe_pear_file_rpc.py'
    workflow=yaml.safe_load((old/'workflow.yaml').read_text())['workflow'];task=workflow['tasks'][0]
    build=worker_build(spec,files,spec['allocation_job_entry'],task['image'],task['command'][0],
        **workflow['resources']['default'])
    if read(build/'spec.json')!=spec:raise ValueError('Native transport repair changed the simulator spec')
    return build


def recoverable(item,node,*,rpc_sha256):
    old=read(Path(item['build'])/'source-manifest.json',{}).get('efe_pear_file_rpc.py')
    return bool(node.get('phase')=='idle_failed'and node.get('retired_request_failure')
        and node.get('job_id')==item.get('job_id')and old and old!=rpc_sha256)


def tick(state,*,limit=4):
    if read(DEST/'control.json',{}).get('paused')or not state.get('job_id'):return
    path=DEST/'primary-native-queue.json';queue=read(path,{});items={};seen=set()
    for key,item in reversed(list(queue.items())):
        if(item.get('superseded')or item.get('owner_model_workflow')!=state['workflow']
            or item.get('model_job_id')!=state['job_id']or item['workflow']in seen):continue
        seen.add(item['workflow']);items[key]=item
    if not items:return
    from .monitor_efe_preflight import remote
    from .efe_pear_allocation_jobs import running_tasks
    running_tasks(command(['osmo','workflow','query',state['workflow'],'--format-type','json']))
    rows=[]
    for key,item in items.items():
        spec=read(Path(item['build'])/'spec.json')
        allocation=read(Path(item.get('allocation_build',item['build']))/'spec.json')
        rows.append({'key':key,'control':allocation['allocation_control_base']+'/'+item['workflow'],
            'workflow':item['workflow'],'output':spec.get('output')})
    script="""from pathlib import Path
import json,time
head=Path(RUN)/'head-heartbeat.json'
try:owner=json.loads(head.read_text())
except(FileNotFoundError,ValueError):owner={}
fresh=owner.get('workflow')==WORKFLOW and time.time()-owner.get('updated_unix',0)<120 and not(Path(RUN)/'finished.json').exists()
out={}
def tail(path):
 with path.open('rb')as stream:stream.seek(max(0,path.stat().st_size-6000));return stream.read().decode(errors='replace')
if fresh:
 for row in ROWS:
  p=Path(row['control'])/'nodes/native.json'
  try:node=json.loads(p.read_text())
  except(FileNotFoundError,ValueError):continue
  if node.get('phase')!='idle_failed':continue
  job=node['job_id'];logs=list((Path(row['control'])/'jobs'/job).glob('*.log'))
  if row.get('output'):logs+=list((Path(row['output'])/(row['workflow']+'-'+job)).glob('*/native.log'))
  node['retired_request_failure']=any('An abandoned request was already retired'in tail(p)for p in logs)
  out[row['key']]=node
print('EFE_CAPTURE '+json.dumps({'head_fresh':fresh,'nodes':out}),flush=True)
""".replace('RUN',repr(state['run'])).replace('ROWS',repr(rows)).replace('WORKFLOW',repr(state['workflow']))
    observed=remote(state['workflow'],script,task='trainer')
    progress={'observed_unix':time.time(),'model_job_id':state['job_id'],
        'head_fresh':observed['head_fresh'],'failed_children':len(observed['nodes']),'repairs':[]}
    if not observed['head_fresh']:
        write(DEST/'native-child-recovery.json',progress);return progress
    digest=hashlib.sha256((BASE/'cluster/efe_pear_file_rpc.py').read_bytes()).hexdigest()
    from .efe_pear_allocation_jobs import submit
    for key,node in observed['nodes'].items():
        item=items[key]
        if not recoverable(item,node,rpc_sha256=digest):continue
        build=repair_build(item)
        job='rpc-replay-'+hashlib.sha256((state['job_id']+item['workflow']+build.name).encode()).hexdigest()[:14]
        receipt=submit(item,build,job_id=job)
        replacement={**item,**receipt,'build':str(build),
            'allocation_build':item.get('allocation_build',item['build']),'superseded':False,
            'native_rpc_recovery_sha256':digest}
        for row in queue.values():
            if row['workflow']==item['workflow']:row.update(superseded=True,superseded_by=build.name)
        queue[build.name]=replacement;write(path,queue)
        progress['repairs'].append({'workflow':item['workflow'],'job_id':receipt['job_id'],'build':str(build)})
        write(DEST/'native-child-recovery.json',progress)
        if len(progress['repairs'])>=limit:break
    write(DEST/'native-child-recovery.json',progress);return progress
