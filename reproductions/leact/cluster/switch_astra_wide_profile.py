"""Stage an explicit teacher profile without restarting retained native servers."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .scale_efe import BASE,command,read,write
from .launch_astra_wide_collection import ROOT,TEACHER_ROOT,OWNER,freeze_worker
from .recover_efe_pear_graphics import frozen_files
from .efe_pear_allocation_jobs import running_tasks
from .monitor_efe_preflight import remote
from .efe_astra_wide_collect import ASTRA62_PROFILE,PROFILES

PROFILE_ROOT=TEACHER_ROOT+'-astra62-xhigh-v1'
PROFILE_FILES=['efe_astra_wide_collect.py','efe_astra_wide_profile_worker.py']


def profile_spec(spec):
    if spec.get('owner_model_workflow')!=OWNER or spec.get('teacher_root')not in {TEACHER_ROOT,PROFILE_ROOT}:
        raise ValueError('Refusing to retarget a non-teacher experiment')
    plan={**spec['teacher_plan'],'teacher_model':PROFILES[ASTRA62_PROFILE]['model'],
          'teacher_profile':dict(PROFILES[ASTRA62_PROFILE])}
    return {**spec,'teacher_root':PROFILE_ROOT,'legacy_teacher_root':TEACHER_ROOT,
        'teacher_profile_id':ASTRA62_PROFILE,'teacher_plan':plan,
        'finished':PROFILE_ROOT+'/finished.json','head_heartbeat':PROFILE_ROOT+'/head-heartbeat.json'}


def prepare(state):
    old=Path(state['build']);spec=profile_spec(read(old/'spec.json'));files=frozen_files(old)
    for name in PROFILE_FILES:files[name]=BASE/'cluster'/name
    key=hashlib.sha256(json.dumps([spec,{n:hashlib.sha256(p.read_bytes()).hexdigest()for n,p in files.items()}],sort_keys=True).encode()).hexdigest()[:12]
    build=old.parent/('rp-jhh-astra62-profile-'+key)
    if not(build/'source-manifest.json').exists():
        build.mkdir(exist_ok=True)
        for name,path in files.items():shutil.copyfile(path,build/name)
        write(build/'spec.json',spec)
        freeze_worker(build,[*files,'spec.json'])
    return build


def deploy(state):
    query=command(['osmo','workflow','query',state['workflow'],'--format-type','json']);running_tasks(query)
    allocation=read(Path(state.get('allocation_build',state['build']))/'spec.json')
    control=allocation['allocation_control_base']+'/'+state['workflow']
    if state.get('profile_sidecar'):
        script='''from pathlib import Path
import json
p=Path(__CONTROL__)/'profile-jobs'/__JOB__/'process.json';old=json.loads(p.read_text())if p.exists()else{}
live=False
if old.get('pid'):
 try:live=(str(Path(old['bundle'])/'efe_astra_wide_profile_worker.py').encode()in(Path('/proc')/str(old['pid'])/'cmdline').read_bytes())
 except FileNotFoundError:pass
print('EFE_CAPTURE '+json.dumps({'live':live,'process':old}),flush=True)
'''.replace('__CONTROL__',repr(control)).replace('__JOB__',repr(state['profile_job_id']))
        old=remote(state['workflow'],script,task='native')
        if old['live']:return {**state,'profile_sidecar':{**old['process'],'already_running':True}}
    build=prepare(state);job='rp-jhh-astra62-'+build.name.rsplit('-',1)[-1]
    bundle=control+'/profile-jobs/'+job+'/bundle';manifest=read(build/'source-manifest.json')
    for name,sha in manifest.items():
        if hashlib.sha256((build/name).read_bytes()).hexdigest()!=sha:raise ValueError('Local frozen profile source changed')
    script='''from pathlib import Path
import json
p=Path(__CONTROL__);node=json.loads((p/'nodes/native.json').read_text())
if node['phase']!='running':raise RuntimeError('Native worker child is not healthy; preserve allocation for recovery')
Path(__BUNDLE__).mkdir(mode=0o700,parents=True,exist_ok=True)
print('EFE_CAPTURE '+json.dumps({'ready':True,'native_child_pid':node['child_pid'],'native_job':node['job_id']}),flush=True)
'''.replace('__CONTROL__',repr(control)).replace('__BUNDLE__',repr(bundle))
    proof=remote(state['workflow'],script,task='native')
    result=subprocess.run(['osmo','workflow','rsync','upload',state['workflow'],'native',
        str(build)+'/:'+bundle+'/','--no-progress'],capture_output=True,text=True,timeout=180)
    if result.returncode:raise RuntimeError('Profile source staging failed: '+result.stderr[-500:])
    # Interpreter matches the existing task image, including its managed venv.
    import yaml
    python=yaml.safe_load((Path(state['build'])/'workflow.yaml').read_text())['workflow']['tasks'][0]['command'][0]
    script='''from pathlib import Path
import fcntl,hashlib,json,os,signal,subprocess,time
control=Path(__CONTROL__);bundle=Path(__BUNDLE__);job=__JOB__;entry=bundle/'efe_astra_wide_profile_worker.py'
manifest=json.loads((bundle/'source-manifest.json').read_text())
for name,expected in manifest.items():
 if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=expected:raise ValueError('Staged profile checksum differs')
spec=json.loads((bundle/'spec.json').read_text())
if spec.get('teacher_profile_id')!='astra62-xhigh-v1':raise ValueError('Wrong teacher profile')
record=control/'profile-jobs'/job/'process.json';lock=record.with_suffix('.lock')
with lock.open('a')as guard:
 fcntl.flock(guard,fcntl.LOCK_EX)
 old=json.loads(record.read_text())if record.exists()else{}
 if old.get('pid'):
  try:
   cmd=(Path('/proc')/str(old['pid'])/'cmdline').read_bytes()
   if str(entry).encode()in cmd:
    print('EFE_CAPTURE '+json.dumps({**old,'already_running':True}),flush=True);raise SystemExit(0)
  except FileNotFoundError:pass
 env=dict(os.environ);env['EFE_WORKFLOW_ID']=__WORKER__;env['PYTHONUNBUFFERED']='1'
 # Preserve protected credentials in their existing node-local file; do not
 # copy environment secrets into artifacts or pass them through command text.
 wrapper='import runpy,signal,sys;\\ndef stop(*args):raise SystemExit(143)\\nsignal.signal(signal.SIGTERM,stop);sys.path.insert(0,sys.argv[1]);runpy.run_path(sys.argv[2],run_name="__main__")'
 with(record.parent/'profile.log').open('a')as log:
  child=subprocess.Popen([__PYTHON__,'-u','-c',wrapper,str(bundle),str(entry)],env=env,
       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 value={'pid':child.pid,'job_id':job,'profile':'astra62-xhigh-v1','bundle':str(bundle),
  'started_unix':time.time(),'native_child_pid_preserved':__NATIVE_PID__}
 record.write_text(json.dumps(value));print('EFE_CAPTURE '+json.dumps(value),flush=True)
'''.replace('__CONTROL__',repr(control)).replace('__BUNDLE__',repr(bundle)).replace('__JOB__',repr(job)).replace(
        '__WORKER__',repr(state['workflow']+'-'+job)).replace('__PYTHON__',repr(python)).replace('__NATIVE_PID__',str(proof['native_child_pid']))
    receipt=remote(state['workflow'],script,task='native')
    value={**state,'teacher_profile_id':ASTRA62_PROFILE,'teacher_profile_root':PROFILE_ROOT,
        'profile_build':str(build),'profile_sidecar':receipt,'profile_job_id':job,'profile_phase':'staged_and_started'}
    # Use the same locked ledger helper as the retained-allocation launcher.
    from .launch_astra_wide_collection import record_launch
    record_launch(value);return value


def main():
    states=[v for v in read(ROOT/'launches.json').values()if not v.get('superseded_by')]
    states.sort(key=lambda v:(v['label']!='reasoning',v['label']!='native-yam',v['workflow']))
    for state in states:
        try:
            value=deploy(state)
            print(json.dumps({'workflow':value['workflow'],'profile_job':value['profile_job_id'],'pid':value['profile_sidecar']['pid']}),flush=True)
        except Exception as exc:
            print(json.dumps({'workflow':state['workflow'],'deployment_pending':str(exc)[:600]}),flush=True)


if __name__=='__main__':main()
