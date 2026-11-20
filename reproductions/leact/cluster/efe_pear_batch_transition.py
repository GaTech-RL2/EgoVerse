"""Apply the startup group batch after validation, preserving any optimizer work."""
import hashlib
from pathlib import Path
import time

from .scale_efe import read,write,command
from .launch_efe_pear_scaled import DEST,REMOTE


def tick(state):
    progress=read(DEST/'startup-batch-transition.json',{});control=read(DEST/'control.json',{})
    if(control.get('paused')or progress.get('phase')not in {'validating','waiting_for_configuration_validation'}
        or progress.get('current_workflow')!=state.get('workflow')):return None
    check=read(DEST/'primary-driver-check-launch.json',{})
    if not check.get('name'):return None
    query=command(['osmo','workflow','query',check['name'],'--format-type','json'])
    progress.update(phase='waiting_for_configuration_validation',driver_workflow=check['name'],driver_status=query['status'])
    write(DEST/'startup-batch-transition.json',progress)
    if query['status']!='COMPLETED':return None
    from .efe_pear_allocation_jobs import running_tasks,submit
    from .monitor_efe_preflight import remote
    allocation=command(['osmo','workflow','query',state['workflow'],'--format-type','json'])
    if allocation['status']!='RUNNING':return None
    running_tasks(allocation)
    script="""from pathlib import Path
import json
root=Path(ROOT);p=Path(CHECK)
print('EFE_CAPTURE '+json.dumps({'check':json.loads(p.read_text()),'updates':len(list((root/'telemetry/steps').glob('*.json')))}),flush=True)
""".replace('ROOT',repr(state['run'])).replace('CHECK',repr(REMOTE+'/primary-driver-checks/'+check['name']+'/checks.json'))
    observed=remote(state['workflow'],script,task='trainer')
    if not observed['check'].get('passed')or observed['check'].get('retained_groups_per_update')!=progress['target_groups']:
        raise ValueError('The startup group batch has not passed exact driver composition')
    write(DEST/'primary-driver-check-status.json',{**check,'workflow':check['name'],
        'scheduler_status':'COMPLETED','worker':observed['check']})
    if observed['updates']:
        progress.update(phase='checkpoint_handoff_required',completed_updates=observed['updates'])
        write(DEST/'startup-batch-transition.json',progress);return None
    from .launch_efe_pear_primary import prepare
    old=read(Path(state['build'])/'spec.json');build=prepare(state['profile']);new=read(build/'spec.json')
    manifest=read(build/'curriculum.json')
    if(new['model_revision']!=old['model_revision']or new['rollout_nodes']!=old['rollout_nodes']
        or manifest.get('retained_groups_per_update')!=progress['target_groups']):
        raise ValueError('The startup batch changed model initialization or allocated roles')
    job_id='startup32-'+hashlib.sha256((state['workflow']+build.name).encode()).hexdigest()[:16]
    receipt=submit(state,build,replace_running=True,job_id=job_id)
    state.setdefault('job_history',[]).append({'job_id':state['job_id'],'build':state['build'],'run':state['run'],
        'completed_updates':0,'reason':'Pre-update 128-group collection; startup variant lowers retained groups to 32.'})
    state.update(allocation_build=state.get('allocation_build',state['build']),build=str(build),run=new['run'],
        job_id=receipt['job_id'],native_workers_launched=False,in_allocation_retries=0,phase='startup32_job_submitted_in_existing_allocation')
    state.pop('allocation_job_status',None)
    write(DEST/'primary-launch.json',state);write(DEST/'primary-status.json',state)
    progress.update(phase='replacement_submitted',replacement_run=new['run'],job_id=receipt['job_id'],
        submitted_unix=time.time(),allocation_preserved=True)
    write(DEST/'startup-batch-transition.json',progress);return state
