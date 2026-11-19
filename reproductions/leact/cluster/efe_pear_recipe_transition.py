"""Advance the approved recipe after driver and known-case inference checks."""
import json
from pathlib import Path
import subprocess
import time

from .scale_efe import read,write,command
from .launch_efe_pear_scaled import DEST,REMOTE
from .efe_pear_signal_curriculum import POLICY


def replay_ready(cases,expected_tasks,model_revision):
    if len(cases)!=len(expected_tasks)or {c.get('task')for c in cases}!=set(expected_tasks):return False
    if any(c.get('status')!='completed'or c.get('model_revision')!=model_revision for c in cases):return False
    valid=sum(c.get('valid_actions',0)for c in cases);invalid=sum(c.get('invalid_actions',0)for c in cases)
    return bool(valid and valid/(valid+invalid)>=.95 and any(c.get('official_success')==1 for c in cases))


def capture(state,replay,driver):
    # Read through an existing single-task shared-volume worker. Never exec a
    # model group automatically while it might be at its OSMO startup barrier.
    reader=read(DEST/'pending-maintenance.json',{})
    if not reader.get('workflow'):return None
    status=command(['osmo','workflow','query',reader['workflow'],'--format-type','json'])
    if status['status']!='RUNNING':return None
    script="""from pathlib import Path
import json,time
root=Path(ROOT);replay=Path(REPLAY);check=Path(CHECK)
cases=[]
for p in replay.glob('*/result.json'):
 try:cases.append(json.loads(p.read_text()))
 except(ValueError,FileNotFoundError):pass
head=json.loads((root/'head-heartbeat.json').read_text())
print('EFE_CAPTURE '+json.dumps({'cases':cases,'head':head,
 'completed_updates':len(list((root/'telemetry/steps').glob('*.json'))),
 'driver_check':json.loads(check.read_text())if check.exists()else None}),flush=True)
""".replace('ROOT',repr(state['run'])).replace('REPLAY',repr(replay['output'])).replace('CHECK',repr(
        REMOTE+'/primary-driver-checks/'+driver['name']+'/checks.json'))
    from .monitor_efe_preflight import remote
    return remote(reader['workflow'],script,task='native')


def repeat_replay(state,replay,attempt):
    from .monitor_efe_preflight import remote
    target=replay['output'].split('-repeat-',1)[0]+'-repeat-'+str(attempt)
    script="""from pathlib import Path
import json,subprocess,time,shutil
source=Path(SOURCE);target=Path(TARGET);target.mkdir(mode=0o700,parents=True,exist_ok=True)
if(target/'launch.json').exists():
 print('EFE_CAPTURE '+(target/'launch.json').read_text(),flush=True)
 raise SystemExit(0)
for name in ['efe_pear_matched_smoke.py','efe_pear_file_rpc.py','efe_action_loop.py']:shutil.copyfile(source/name,target/name)
spec=json.loads((source/'spec.json').read_text());spec['output']=str(target)
(target/'spec.json').write_text(json.dumps(spec))
with(target/'worker.log').open('a')as log:
 p=subprocess.Popen([PYTHON,'-u',str(target/'efe_pear_matched_smoke.py')],
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':p.pid,'output':str(target),'started_unix':time.time()}
temporary=target/'launch.json.tmp';temporary.write_text(json.dumps(receipt));temporary.replace(target/'launch.json')
print('EFE_CAPTURE '+json.dumps(receipt),flush=True)
""".replace('SOURCE',repr(replay['output'])).replace('TARGET',repr(target)).replace('PYTHON',repr(REMOTE+'/runtime/envs/verl/bin/python'))
    result=remote(state['workflow'],script,task='trainer')
    value={**replay,**result,'attempt':attempt,'history':replay.get('history',[])+[
        {k:replay[k]for k in ['output','pid','started_unix']if k in replay}]}
    write(DEST/'matched-warmstart-smoke-h100-client.json',value)
    return value


def tick(state):
    transition=read(DEST/'recipe-transition.json',{})
    if(not transition.get('approved')or transition.get('from_workflow')!=state.get('workflow')
        or transition.get('next_sampling_policy')!=POLICY):return None
    control=read(DEST/'control.json',{})
    if control.get('paused')or state.get('run_id')not in control.get('enabled_training_arms',[]):return None
    if transition.get('phase')in {'requires_harness_review','requires_checkpoint_handoff'}:return None
    if transition.get('phase')=='reallocating_for_validated_recipe':
        query=command(['osmo','workflow','query',state['workflow'],'--format-type','json'])
        if query['status'].startswith('FAILED')or query['status']=='COMPLETED':
            from .launch_efe_pear_primary import launch,cancel_native_workers
            cancel_native_workers(state['workflow'])
            state.update(terminal_confirmed=True,source_repair_ready=True,next_profile=state['profile'],scheduler_status=query['status'])
            write(DEST/'primary-launch.json',state)
            result=launch(state['profile'],pool=state['pool'])
            transition.update(phase='replacement_submitted',replacement_workflow=result['workflow'])
            write(DEST/'recipe-transition.json',transition)
            return result
        transition['phase']='validating_replacement'
    replay=read(DEST/'matched-warmstart-smoke-h100-client.json',{});driver=read(DEST/'primary-driver-check-launch.json',{})
    if not replay.get('output')or not driver.get('name'):return None
    observed=capture(state,replay,driver)
    if observed is None:return None
    cases=observed['cases'];transition.update(phase='validating_replacement',observed_unix=time.time(),
        replay_cases=cases,completed_primary_updates=observed['completed_updates'])
    if observed['completed_updates']:
        transition['phase']='requires_checkpoint_handoff';write(DEST/'recipe-transition.json',transition);return None
    ready=replay_ready(cases,transition['known_training_tasks'],transition['model_revision'])
    terminal=len(cases)==len(transition['known_training_tasks'])and all(c.get('status')in
        {'completed','infrastructure_or_runtime_failure'}for c in cases)
    if terminal and not ready:
        attempt=replay.get('attempt',1)
        head=observed['head']
        if(attempt<3 and head.get('workflow')==state['workflow']and head.get('phase')=='running_full_mixture_pear'
            and time.time()-head.get('updated_unix',0)<120):
            repeat_replay(state,replay,attempt+1);transition['replay_attempt']=attempt+1
        else:transition['phase']='requires_harness_review'
    check=observed.get('driver_check')or{}
    cpu=command(['osmo','workflow','query',driver['name'],'--format-type','json'])
    cpu_ready=cpu['status']=='COMPLETED'and check.get('passed')and POLICY in check.get('sampling_policies',[])
    if cpu_ready:
        write(DEST/'primary-driver-check-status.json',{**driver,'workflow':driver['name'],
            'scheduler_status':'COMPLETED','worker':check})
    transition.update(driver_scheduler_status=cpu['status'],driver_passed=bool(cpu_ready),matched_control_passed=ready)
    write(DEST/'recipe-transition.json',transition)
    if not(ready and cpu_ready):return None
    from .launch_efe_pear_primary import prepare,launch,cancel_native_workers
    control['next_sampling_policy']=POLICY;write(DEST/'control.json',control)
    transition['validation_passed']=True;write(DEST/'recipe-transition.json',transition)
    build=prepare(state['profile'])
    next_spec=read(build/'spec.json')
    if next_spec['model_revision']!=transition['model_revision']or next_spec['rollout_nodes']!=4:
        raise ValueError('Recipe handoff changed the initialization or validated GPU allocation')
    transition.update(phase='reallocating_for_validated_recipe',prepared_build=str(build))
    write(DEST/'recipe-transition.json',transition)
    if not state['workflow'].startswith('leact-rp-jhh-pear-primary-'):
        raise ValueError('Recipe transition does not own this workflow')
    subprocess.run(['osmo','workflow','cancel',state['workflow'],'--message',
        'User-approved task curriculum and rolling family balancing. Exact driver checks and known-case native success passed; zero completed primary optimizer updates.'],
        check=True,capture_output=True,text=True,timeout=30)
    deadline=time.monotonic()+45
    while True:
        query=command(['osmo','workflow','query',state['workflow'],'--format-type','json'])
        if query['status'].startswith('FAILED')or query['status']=='COMPLETED':break
        if time.monotonic()>deadline:raise RuntimeError('Recipe handoff is waiting for confirmed allocation termination')
        time.sleep(2)
    cancel_native_workers(state['workflow'])
    state.update(terminal_confirmed=True,source_repair_ready=True,next_profile=state['profile'],scheduler_status=query['status'])
    write(DEST/'primary-launch.json',state)
    result=launch(state['profile'],pool=state['pool'])
    transition.update(phase='replacement_submitted',replacement_workflow=result['workflow'])
    write(DEST/'recipe-transition.json',transition)
    return result
