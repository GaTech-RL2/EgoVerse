"""Launch another job in retained OSMO nodes, without scheduler cancellation."""
import hashlib
import json
from pathlib import Path
import subprocess
from uuid import uuid4

from .scale_efe import read,command
from .monitor_efe_preflight import remote


def running_tasks(query):
    tasks=[t for g in query.get('groups',[])for t in g['tasks']]
    tasks+=query.get('tasks',[])
    if query['status']!='RUNNING'or not tasks or any(t['status']!='RUNNING'for t in tasks):
        raise RuntimeError('Allocation is not fully running; do not exec during the OSMO startup barrier')
    return tasks


def status(state):
    spec=read(Path(state.get('allocation_build',state['build']))/'spec.json')
    if spec.get('allocation_lifecycle')!='persistent_cluster':return None
    script="""from pathlib import Path
import json
root=Path(CONTROL);run=Path(RUN)
nodes=[json.loads(p.read_text())for p in(root/'nodes').glob('*.json')]
print('EFE_CAPTURE '+json.dumps({'nodes':nodes,'completed_updates':len(list((run/'telemetry/steps').glob('*.json')))}),flush=True)
""".replace('CONTROL',repr(spec['allocation_control_base']+'/'+state['workflow'])).replace('RUN',repr(state['run']))
    return remote(state['workflow'],script,task='trainer')


def activate_embedded(state,*,previous_workflow=None):
    """Start the already-pinned initial job in an allocated standby cluster."""
    query=command(['osmo','workflow','query',state['workflow'],'--format-type','json']);running_tasks(query)
    script="""from pathlib import Path
import fcntl,hashlib,json,sys,time
lane=Path('/tmp/rp-jhh-pear-primary');sys.path.insert(0,str(lane))
from efe_pear_allocation_keeper import write,read
spec=read(lane/'spec.json');root=Path(spec['allocation_control_base'])/WORKFLOW
roles=['trainer']+['rollout-'+str(i)for i in range(spec['rollout_nodes'])]
nodes=[read(root/'nodes'/(role+'.json'),{})for role in roles]
if any(not node for node in nodes):raise RuntimeError('Allocation keepers are not all ready')
run=Path(spec['run']);run.mkdir(mode=0o700,parents=True,exist_ok=True)
with(run/'allocation-writer.lock').open('a')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 owner=read(run/'allocation-writer.json',{})
 if owner.get('workflow')not in {None,WORKFLOW,PREVIOUS}:raise RuntimeError('Another retained allocation owns this experiment')
 write(run/'allocation-writer.json',{'workflow':WORKFLOW,'updated_unix':time.time()})
request=read(root/'desired.json')
if request is None:
 names=[p.name for p in lane.iterdir()if p.is_file()and p.suffix in {'.py','.json','.yaml','.sh'}and p.name!='bootstrap.py']
 request={'id':'initial-'+hashlib.sha256((lane/'spec.json').read_bytes()).hexdigest()[:12],
  'bundle':str(lane),'entry':'efe_pear_primary_worker.py','roles':roles,
  'files':{n:hashlib.sha256((lane/n).read_bytes()).hexdigest()for n in names}}
 write(root/'desired.json',request)
print('EFE_CAPTURE '+json.dumps({'job_id':request['id'],'allocation_control':str(root),'workflow':WORKFLOW}),flush=True)
""".replace('WORKFLOW',repr(state['workflow'])).replace('PREVIOUS',repr(previous_workflow))
    return remote(state['workflow'],script,task='trainer')


def submit(state,build,*,replace_running=False,job_id=None,previous_workflow=None):
    """Verify and stage a complete frozen job on the existing shared volume."""
    build=Path(build);allocation_spec=read(Path(state.get('allocation_build',state['build']))/'spec.json')
    if allocation_spec.get('allocation_lifecycle')!='persistent_cluster':
        raise RuntimeError('This legacy allocation has no keeper; it must be preserved without automatic cancellation')
    candidate=read(build/'spec.json');model='rollout_nodes'in allocation_spec
    if model and candidate.get('rollout_nodes')!=allocation_spec['rollout_nodes']:
        raise ValueError('The job must fit the existing allocated node roles')
    query=command(['osmo','workflow','query',state['workflow'],'--format-type','json']);running_tasks(query)
    task='trainer'if model else'native';roles=['trainer',*['rollout-'+str(i)for i in range(candidate['rollout_nodes'])]]if model else['native']
    base=allocation_spec.get('allocation_control_base')
    if base is None:raise ValueError('Allocation has no durable control directory')
    control=base+'/'+state['workflow'];job=job_id or'job-'+uuid4().hex[:16];bundle=control+'/jobs/'+job+'/bundle'
    files=read(build/'source-manifest.json')
    for name,expected in files.items():
        if hashlib.sha256((build/name).read_bytes()).hexdigest()!=expected:raise ValueError('Local job source changed')
    script="""from pathlib import Path
import json
root=Path(CONTROL);nodes=[json.loads(p.read_text())for p in(root/'nodes').glob('*.json')]
if not nodes:raise RuntimeError('Allocation keeper is not ready')
desired=root/'desired.json'
if desired.exists()and json.loads(desired.read_text())['id']==JOB:
 print('EFE_CAPTURE '+json.dumps({'already_submitted':True,'job_id':JOB,'workflow':WORKFLOW,'allocation_preserved':True}),flush=True)
 raise SystemExit(0)
if not REPLACE and any(n['phase']=='running'for n in nodes):raise RuntimeError('Existing job is still running; allocation preserved')
Path(BUNDLE).mkdir(mode=0o700,parents=True,exist_ok=True)
print('EFE_CAPTURE '+json.dumps({'ready':True}),flush=True)
""".replace('CONTROL',repr(control)).replace('REPLACE',repr(replace_running)).replace('BUNDLE',repr(bundle)).replace('JOB',repr(job)).replace('WORKFLOW',repr(state['workflow']))
    ready=remote(state['workflow'],script,task=task)
    if ready.get('already_submitted'):return ready
    subprocess.run(['osmo','workflow','rsync','upload',state['workflow'],task,
        str(build)+'/:'+bundle+'/','--no-progress'],check=True,capture_output=True,text=True,timeout=180)
    request={'id':job,'bundle':bundle,'entry':'efe_pear_primary_worker.py'if model else candidate['allocation_job_entry'],
        'roles':roles,'files':files}
    manifest_sha=hashlib.sha256((build/'source-manifest.json').read_bytes()).hexdigest()
    metadata={k:v for k,v in request.items()if k!='files'}
    script="""from pathlib import Path
import fcntl,hashlib,json,os,time
root=Path(CONTROL);request=json.loads(REQUEST)
manifest=(Path(request['bundle'])/'source-manifest.json').read_bytes()
if hashlib.sha256(manifest).hexdigest()!=MANIFEST_SHA:raise ValueError('Staged source manifest changed')
request['files']=json.loads(manifest)
for name,expected in request['files'].items():
 if hashlib.sha256((Path(request['bundle'])/name).read_bytes()).hexdigest()!=expected:raise ValueError('Staged job checksum differs')
if MODEL:
 candidate=json.loads((Path(request['bundle'])/'spec.json').read_text());run=Path(candidate['run']);run.mkdir(mode=0o700,parents=True,exist_ok=True)
 with(run/'allocation-writer.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);p=run/'allocation-writer.json';owner=json.loads(p.read_text())if p.exists()else{}
  if owner.get('workflow')not in {None,WORKFLOW,PREVIOUS}:raise RuntimeError('Another retained allocation owns this experiment')
  p.write_text(json.dumps({'workflow':WORKFLOW,'updated_unix':time.time()}))
temporary=root/('desired.'+str(os.getpid())+'.tmp');temporary.write_text(json.dumps(request));temporary.replace(root/'desired.json')
print('EFE_CAPTURE '+json.dumps({'job_id':request['id'],'workflow':WORKFLOW,'allocation_preserved':True}),flush=True)
""".replace('CONTROL',repr(control)).replace('REQUEST',repr(json.dumps(metadata))).replace('MANIFEST_SHA',repr(manifest_sha)).replace('WORKFLOW',repr(state['workflow'])).replace('PREVIOUS',repr(previous_workflow)).replace('MODEL',repr(model))
    return remote(state['workflow'],script,task=task)
