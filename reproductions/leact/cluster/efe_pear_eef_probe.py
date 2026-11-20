"""Discarded, model-free PEAR EEF motion checks in an isolated native sidecar.

No teacher calls, learning examples, optimizer updates, demonstration actions or
benchmark results are produced. Recorded camera frames and native feedback are
kept so direction and controller behavior can be reviewed.
"""
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from efe_action_loop import parse_action
from efe_pear_eef import VERSION
from efe_pear_file_rpc import FileEpisodeClient, atomic_json


def vector(result):
    state=result['state']['tcp_pose']['arms']
    return [value for side in ('left','right') for value in
        [*state[side]['position_m'],*state[side]['quaternion_wxyz'],
         state[side]['gripper_target_closed_fraction']]]


def save_observation(folder,result):
    folder.mkdir(parents=True,exist_ok=True)
    for camera,image in zip(result['action_contract']['cameras'],result.get('images')or[]):
        image.save(folder/(camera+'.png'))
    atomic_json(folder/'feedback.json',{k:result[k]for k in
        ['state','action_contract','executed_control_steps','requested_control_steps','eef_execution']if k in result})


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    if spec.get('control_interface')!=VERSION or spec.get('teacher_calls_permitted')is not False:
        raise ValueError('This probe is restricted to model-free PEAR EEF validation')
    output=Path(spec['probe_root']);output.mkdir(mode=0o700,parents=True,exist_ok=True)
    stopped=threading.Event();status={'phase':'starting_native_sidecar','checks':[],
        'control_interface':VERSION,'teacher_model_calls':0,'optimizer_updates':0,'training_data_produced':False}
    def heartbeat():
        while not stopped.is_set():
            atomic_json(Path(spec['head_heartbeat']),{'workflow':spec['owner_model_workflow'],'updated_unix':time.time()})
            atomic_json(output/'status.json',{**status,'updated_unix':time.time()})
            stopped.wait(10)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    server=None
    try:
        with(output/'server.log').open('a')as log:
            env=dict(os.environ);env['EFE_WORKFLOW_ID']=spec['probe_worker_id']
            server=subprocess.Popen([sys.executable,'-u',str(lane/'efe_pear_yam_server.py')],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status['server_pid']=server.pid;status['phase']='waiting_for_native_motion'
        for case_index,case in enumerate(spec['probe_cases']):
            for check in ['hold','x','x_negative','y','z','gripper','unreachable']:
                client=FileEpisodeClient(spec['transport_root'],'native_yam/'+case['task'],
                    case['env_args'],timeout=1800)
                record={'case':case['config_id'],'check':check,'episode_uid':client.uid}
                directory=output/'cases'/f'{case_index:02d}-{check}'
                try:
                    initial=client.open()
                    contract=initial['action_contract']
                    if contract.get('control_interface')!=VERSION or set(contract['action_types'])!={'ee_pose'}:
                        raise ValueError('Native sidecar did not expose the explicit PEAR EEF contract')
                    if initial['scene_sha256']!=case['scene_sha256']:
                        raise ValueError('Discarded motion probe reset differs from its admitted training scene')
                    if spec.get('probe_cartesian_approach'):
                        # Qualify Cartesian directions after a model-invocable
                        # downward approach as well as keeping the separately
                        # recorded reset-pose diagnostics. The near-straight
                        # reset is not an interior point of the 6D workspace.
                        start=vector(initial);setup=[]
                        for n in range(1,5):
                            pose=list(start);angle=n*math.pi/12
                            c,s=math.cos(angle/2),math.sin(angle/2)
                            w,x,y,z=start[3:7]
                            pose[:3]=[start[0]-.01*n,start[1],start[2]+.01*n]
                            pose[3:7]=[c*w-s*y,c*x+s*z,c*y+s*w,c*z-s*x]
                            command={'action_type':'ee_pose','action':pose,'repeat':25,
                                'summary':'Discarded controller probe: orient the open hand for a downward approach.'}
                            parse_action(json.dumps(command),contract=contract,allow_stop=False)
                            initial=client.step(command)
                            if initial['done']:raise ValueError('Native task terminated during Cartesian probe approach')
                            setup.append({'action':command,'feedback':initial.get('eef_execution')})
                        record['probe_precondition']={'type':'four_bounded_cartesian_approach_chunks',
                            'native_control_steps':sum(row['feedback']['executed_control_steps']for row in setup),
                            'actions':setup,'used_in_production_training':False}
                    save_observation(directory/'before',initial)
                    target=vector(initial);entry=contract['action_types']['ee_pose']
                    initial_target=copy.deepcopy(target)
                    if check in ['x','x_negative','y','z']:
                        axis='xyz'.index(check[0]);target[axis]+=-.02 if check=='x_negative'else .02
                    elif check=='gripper':target[7]=1.
                    elif check=='unreachable':target[:3]=[entry['upper'][i]-.01 for i in range(3)]
                    action={'action_type':'ee_pose','action':target,
                        'repeat':2 if check=='hold'else 25,'summary':'Discarded PEAR-only EEF '+check+' check.'}
                    parse_action(json.dumps(action),contract=contract,allow_stop=False)
                    after=client.step(action)
                    if after['done']:raise ValueError('Discarded motion check unexpectedly terminated the native task')
                    count=after['executed_control_steps']
                    if not 0<count<=action['repeat']:raise ValueError('Native motion exceeded its requested step budget')
                    save_observation(directory/'after',after)
                    reached=vector(after)
                    movement=[b-a for a,b in zip(initial_target[:3],reached[:3])]
                    position_error=math.sqrt(sum((a-b)**2 for a,b in zip(target[:3],reached[:3])))
                    record.update(control_steps=count,requested_control_steps=action['repeat'],
                        measured_delta_m=movement,position_error_m=position_error,
                        feedback=after.get('eef_execution'),passed=True)
                    if check=='hold':record['passed']=position_error<.015
                    elif check in ['x','x_negative','y','z']:
                        axis='xyz'.index(check[0]);sign=-1 if check=='x_negative'else 1
                        record['passed']=(.015<=sign*movement[axis]<=.035 and
                            math.sqrt(sum(x*x for i,x in enumerate(movement)if i!=axis))<=.015)
                    elif check=='gripper':
                        before=initial['state']['tcp_pose']['arms']['left']['gripper_measured_closed_fraction']
                        closed=after['state']['tcp_pose']['arms']['left']['gripper_measured_closed_fraction']
                        reopened=client.step({**action,'action':initial_target})
                        opened=reopened['state']['tcp_pose']['arms']['left']['gripper_measured_closed_fraction']
                        record.update(gripper_before=before,gripper_closed=closed,gripper_reopened=opened)
                        record['passed']=closed>before+.25 and opened<closed-.25
                    elif check=='unreachable':
                        record['passed']=position_error>.05 and not after['eef_execution']['tracking']['left']['target_reached']
                    final=client.step({'stop':True,'summary':'End discarded PEAR motion probe.'})
                    record['native_finalized']=final['done']and type(final['rewards'][0])in(int,float)
                    record['passed']=record['passed']and record['native_finalized']
                except Exception as exc:
                    record.update(passed=False,error=repr(exc));status['phase']='native_check_failed'
                finally:client.close()
                status['checks'].append(record)
                atomic_json(output/'status.json',{**status,'updated_unix':time.time()})
                print('PEAR_EEF_PROBE '+json.dumps(record),flush=True)
                if record.get('error'):raise RuntimeError(record['error'])
        status['phase']='finished';status['passed']=all(c['passed']for c in status['checks'])
        # A bounded IK API must execute a feasible displacement along each
        # axis and report unreachable goals truthfully. It cannot guarantee
        # that both signed translations are feasible at a fixed joint-limit
        # reset while preserving the exact wrist orientation.
        status['api_motion_passed']=all(
            all(any(c['case']==case['config_id']and c['passed']and c['check']in choices
                for c in status['checks'])for choices in
                [['hold'],['x','x_negative'],['y'],['z'],['gripper'],['unreachable']])
            for case in spec['probe_cases'])
        status['qualification_basis']='Measured reachable signed motion on each axis, gripper endpoints, '
        status['qualification_basis']+='truthful unreachable feedback, exact native step budgets and unchanged native finalization.'
    except Exception as exc:
        status.update(phase='failed',passed=False,error=repr(exc))
        raise
    finally:
        atomic_json(Path(spec['finished']),{'discarded_probe_finished':True})
        if server and server.poll()is None:
            # End only this independently staged probe child. The retained
            # allocation and original CPU/teacher workers remain untouched.
            try:server.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGTERM)
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        stopped.set();thread.join(timeout=2)
        atomic_json(output/'status.json',{**status,'updated_unix':time.time()})


if __name__=='__main__':main()
