"""Activate only an idle, source-verified native-only Astra Dojo keeper."""
import hashlib
import json
from pathlib import Path

import yaml

from .efe_pear_allocation_jobs import running_tasks, submit
from .launch_astra_wide_collection import CREDENTIAL, OWNER, ROOT, record_launch
from .monitor_efe_preflight import remote
from .recover_efe_pear_graphics import frozen_files
from .scale_efe import command, read
from .stage_astra_rpc_heartbeat import patch_worker, recovery_sources
from .switch_astra_wide_profile import PROFILE_ROOT


def activate(state):
    if state.get('native_service_only')is not True or not state.get('native_standby_of'):
        raise ValueError('Only the separately authorized native-service-only replacement can activate')
    build=Path(state['build']);spec=read(build/'spec.json');files=frozen_files(build)
    if (spec.get('owner_model_workflow')!=OWNER or spec.get('teacher_root')!=PROFILE_ROOT
        or spec.get('teacher_profile_id')!='astra62-xhigh-v1' or spec.get('suite')!='robodojo'
        or spec.get('teacher_native_only')is not True or spec.get('teacher_native_service_only')is not True
        or spec.get('teacher_activation_required')is not True or spec.get('control_interface')
        or spec.get('transport_root')!=str(Path(PROFILE_ROOT).with_name(OWNER)/'transport')
        or spec.get('allocation_control_base')!=str(Path(PROFILE_ROOT).with_name(OWNER)/'allocations')
        or spec.get('teacher_credential_file')!='/dev/shm/rp_jhh_astra_api_key'
        or spec.get('max_calls',96)!=96 or spec.get('wall_seconds',1500)!=1500):
        raise ValueError('Native-only replacement changed its authorized teacher scope or recipe')
    original=read(ROOT/'launches.json')[state['native_standby_of']]
    baseline=recovery_sources(frozen_files(Path(original['build'])))
    for name,path in baseline.items():
        expected=patch_worker(path.read_text()).encode()if name=='efe_pear_remote_worker.py'else path.read_bytes()
        if files[name].read_bytes()!=expected:raise ValueError('Native replacement differs from its reviewed frozen baseline: '+name)
    workflow=yaml.safe_load((build/'workflow.yaml').read_text())['workflow'];task=workflow['tasks'][0]
    if CREDENTIAL in task.get('credentials',{}) or 'NVIDIA_INFERENCE_API_KEY'in task.get('environment',{}):
        raise ValueError('A native-only task cannot receive an inference collector credential')
    if task.get('environment',{}).get('PEAR_KEEPER_AUTOSTART')!='0':
        raise ValueError('An unverified native-only allocation must start with an idle keeper')
    query=command(['osmo','workflow','query',state['workflow'],'--format-type','json']);running_tasks(query)
    manifest_sha=hashlib.sha256((build/'source-manifest.json').read_bytes()).hexdigest()
    job='rp-jhh-astra-native-rpc-'+hashlib.sha256((state['workflow']+manifest_sha).encode()).hexdigest()[:20]
    control=spec['allocation_control_base']+'/'+state['workflow']
    script='''from pathlib import Path
import hashlib,json,os,subprocess,time
lane=Path('/tmp/pear-native');root=Path(__CONTROL__);manifest=(lane/'source-manifest.json').read_bytes()
if hashlib.sha256(manifest).hexdigest()!=__MANIFEST__:raise ValueError('Embedded native source manifest differs')
entries=json.loads(manifest);checked=[]
for name,expected in entries.items():
 if name=='workflow.yaml':continue  # Scheduler metadata is not part of the embedded runtime tar.
 if hashlib.sha256((lane/name).read_bytes()).hexdigest()!=expected:raise ValueError('Embedded native payload changed: '+name)
 checked.append(name)
spec=json.loads((lane/'spec.json').read_text())
if spec.get('teacher_native_service_only')is not True or spec.get('teacher_native_only')is not True:raise ValueError('Wrong embedded native role')
credential_present=bool(os.environ.get('NVIDIA_INFERENCE_API_KEY'))or Path(spec['teacher_credential_file']).exists()
if credential_present:raise ValueError('Unexpected inference collector credential on native-only node')
node=json.loads((root/'nodes/native.json').read_text());desired=root/'desired.json'
request=json.loads(desired.read_text())if desired.exists()else None
if request is not None and request.get('id')!=__JOB__:raise RuntimeError('Another native job owns the keeper')
if request is None and(node.get('phase')!='waiting_for_job'or node.get('job_id')is not None or node.get('child_pid')is not None):raise RuntimeError('Native keeper is not idle and unowned')
if time.time()-node.get('updated_unix',0)>120:raise RuntimeError('Native keeper heartbeat is stale')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()
if len(gpu)!=__GPUS__ or len(set(gpu))!=len(gpu):raise RuntimeError('Assigned native GPU visibility is ambiguous')
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True,timeout=10).strip()
if request is None and compute:raise RuntimeError('A GPU workload still owns the native node')
native=[]
if request is None:
 for path in Path('/proc').glob('[0-9]*/cmdline'):
  try:cmd=path.read_bytes().replace(b'\\x00',b' ').decode(errors='replace')
  except OSError:continue
  if any(name in cmd for name in['efe_astra_wide_worker.py','efe_pear_primary_rtx.py','efe_pear_native_entry.py','launch_experiment.py']):native.append(int(path.parent.name))
 if native:raise RuntimeError('A native CPU process still owns the node')
print('EFE_CAPTURE '+json.dumps({'observed_unix':time.time(),'node':node,'existing_job':None if request is None else request['id'],
 'payload_manifest_sha256':__MANIFEST__,'payload_files_verified':len(checked),'runtime_sources_verified':True,
 'inference_credential_present':credential_present,'assigned_gpu_uuids':gpu,'gpu_processes':compute,'native_processes':native,
 'original_collector_and_allocation_preserved':True}),flush=True)
'''.replace('__CONTROL__',repr(control)).replace('__MANIFEST__',repr(manifest_sha)).replace('__JOB__',repr(job)).replace('__GPUS__',repr(workflow['resources']['default']['gpu']))
    proof=remote(state['workflow'],script,task='native')
    if (proof.get('runtime_sources_verified')is not True or proof.get('payload_manifest_sha256')!=manifest_sha
        or proof.get('inference_credential_present')is not False
        or proof.get('existing_job')not in{None,job}):
        raise ValueError('Native-only node did not return a complete matching source/ownership proof')
    if proof.get('existing_job')is None and(proof.get('native_processes')or proof.get('gpu_processes')):
        raise RuntimeError('Native-only source proof still has a process owner')
    receipt=submit(state,build,replace_running=False,job_id=job)
    result={**state,'job_id':receipt['job_id'],'phase':'native_service_activated',
        'native_activation_proof':proof,'native_activation_receipt':receipt,
        'profile_deployment_forbidden':True,'native_contract_runtime_check_pending':True}
    record_launch(result)
    return result
