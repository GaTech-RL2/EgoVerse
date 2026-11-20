"""Deploy the finalized-native-reward recipe inside the retained model nodes."""
import hashlib
import time
from pathlib import Path

from .scale_efe import read,write,command
from .launch_efe_pear_scaled import DEST,REMOTE
from .efe_pear_native_finalization import VERSION


def tick(state):
    control=read(DEST/'control.json',{});progress=read(DEST/'native-reward-finalization-status.json',{})
    if(control.get('paused')or progress.get('phase')not in {'replacement_validating','waiting_for_configuration_validation','replacement_submitted'}
        or progress.get('current_workflow')!=state.get('workflow')):return None
    if progress['phase']=='replacement_submitted':
        from .efe_pear_allocation_jobs import running_tasks
        from .monitor_efe_preflight import remote
        running_tasks(command(['osmo','workflow','query',state['workflow'],'--format-type','json']))
        script="""from pathlib import Path
import json
p=Path(ROOT)/'telemetry/run.json'
print('EFE_CAPTURE '+json.dumps({'metadata':json.loads(p.read_text())if p.exists()else{}}),flush=True)
""".replace('ROOT',repr(state['run']))
        metadata=remote(state['workflow'],script,task='trainer')['metadata']
        if(metadata.get('workflow')==state['workflow']and metadata.get('native_reward_finalization')==VERSION
            and metadata.get('warmup_sampling')==control.get('warmup_sampling')):
            progress.update(phase='replacement_active',replacement_active=True,observed_unix=time.time())
            write(DEST/'native-reward-finalization-status.json',progress)
        return None
    spec=read(Path(state['build'])/'spec.json')
    if read(Path(state['build'])/'curriculum.json').get('native_reward_finalization')==VERSION:return None
    check=read(DEST/'primary-driver-check-launch.json',{})
    if not check.get('name'):return None
    query=command(['osmo','workflow','query',check['name'],'--format-type','json'])
    progress.update(phase='waiting_for_configuration_validation',driver_workflow=check['name'],driver_status=query['status'])
    write(DEST/'native-reward-finalization-status.json',progress)
    if query['status']!='COMPLETED':return None
    from .efe_pear_allocation_jobs import running_tasks,submit
    from .monitor_efe_preflight import remote
    running_tasks(command(['osmo','workflow','query',state['workflow'],'--format-type','json']))
    script="""from pathlib import Path
import json
check=Path(CHECK);root=Path(ROOT)
print('EFE_CAPTURE '+json.dumps({'check':json.loads(check.read_text()),'completed_updates':len(list((root/'telemetry/steps').glob('*.json')))}),flush=True)
""".replace('CHECK',repr(REMOTE+'/primary-driver-checks/'+check['name']+'/checks.json')).replace('ROOT',repr(state['run']))
    observed=remote(state['workflow'],script,task='trainer')
    if not observed['check'].get('passed'):raise ValueError('Replacement driver validation did not pass')
    write(DEST/'primary-driver-check-status.json',{**check,'workflow':check['name'],
        'scheduler_status':'COMPLETED','worker':observed['check']})
    if observed['completed_updates']:
        progress.update(phase='checkpoint_handoff_required',completed_updates=observed['completed_updates'])
        write(DEST/'native-reward-finalization-status.json',progress);return None
    from .launch_efe_pear_primary import prepare
    build=prepare(state['profile']);new=read(build/'spec.json')
    if new['model_revision']!=spec['model_revision']or new['rollout_nodes']!=spec['rollout_nodes']:
        raise ValueError('Reward repair changed the model initialization or allocation roles')
    job_id='native-repair-'+hashlib.sha256((state['workflow']+build.name).encode()).hexdigest()[:16]
    receipt=submit(state,build,replace_running=True,job_id=job_id)
    old={'run':state['run'],'build':state['build'],'job_id':state.get('job_id'),'completed_updates':0,
        'reason':'Legacy close could omit the official native grade; retain as pre-repair diagnostics.'}
    state.setdefault('job_history',[]).append(old)
    state.update(allocation_build=state.get('allocation_build',state['build']),build=str(build),run=new['run'],
        job_id=receipt['job_id'],native_workers_launched=False,in_allocation_retries=0,
        phase='native_reward_repair_submitted_in_existing_allocation')
    write(DEST/'primary-launch.json',state)
    progress.update(phase='replacement_submitted',replacement_active=False,job_id=receipt['job_id'],
        replacement_run=new['run'],submitted_unix=time.time(),allocation_preserved=True)
    write(DEST/'native-reward-finalization-status.json',progress)
    return state
