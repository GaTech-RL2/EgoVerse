"""Discarded native Cartesian checks for ABC, DexJoCo and VLABench; no LLM."""
from concurrent.futures import ThreadPoolExecutor
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
from efe_pear_file_rpc import FileEpisodeClient,atomic_json
from efe_pear_eef_probe import save_observation


def vector(result,suite):
    observation=result['state']['public_observation']['proprio']
    state=dict(zip(observation['layout'],observation['now'],strict=True))
    entry=result['action_contract']['action_types']['ee_pose']
    if suite=='abc-sim':
        values=[]
        for side in ['left','right']:
            values.extend(state[side+'_tcp_'+k]for k in ['x','y','z','qw','qx','qy','qz'])
            values.append(max(0.,min(1.,1.-state['q_'+side+'_gripper'])))
        return values
    if suite=='vlabench':
        return [*[state['tcp_'+k]for k in ['x','y','z','qw','qx','qy','qz']],
            max(0.,min(1.,1.-(state['finger1_pos']+state['finger2_pos'])/.08))]
    if suite=='dexjoco':
        values=[]
        for index,name in enumerate(entry['layout']):
            key=name.replace('_target_','_')if index%23<7 else name
            values.append(state[key])
        return values
    raise ValueError('Unexpected Cartesian core simulator')


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    if spec.get('control_interface')!=VERSION or spec.get('teacher_calls_permitted')is not False:
        raise ValueError('Only discarded model-free PEAR EEF checks are authorized here')
    root=Path(spec['probe_root']);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    stop=threading.Event();lock=threading.Lock();server=None
    status={'phase':'starting_native_workers','checks':[],'teacher_model_calls':0,'optimizer_updates':0,
        'control_interface':VERSION,'training_data_produced':False}
    def heartbeat():
        while not stop.is_set():
            if server is not None and server.poll()is not None:
                for folder in (Path(spec['transport_root'])/'episodes').glob('*'):
                    if folder.is_dir():atomic_json(folder/'infrastructure-error.json',{
                        'error':'Discarded core native server exited with code '+str(server.returncode)})
            atomic_json(Path(spec['head_heartbeat']),{'workflow':spec['owner_model_workflow'],'updated_unix':time.time()})
            with lock:atomic_json(root/'status.json',{**status,'updated_unix':time.time()})
            stop.wait(10)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    def check_case(case):
        suite=case['suite'];client=FileEpisodeClient(spec['transport_root'],'efe/'+suite,case['env_args'],timeout=3600)
        try:
            current=client.open()
            if current['action_contract'].get('control_interface')!=VERSION:
                raise ValueError('Native episode has not exposed the EEF interface')
            if current['scene_sha256']!=case['scene_sha256']:raise ValueError('Native training reset scene mismatch')
            for check in ['hold','x','y','z','unreachable']:
                target=vector(current,suite);before=target[:3].copy();entry=current['action_contract']['action_types']['ee_pose']
                if suite=='dexjoco':
                    # Native reset command targets put both thumb-base joints
                    # at zero, below their advertised 0.263-rad lower bound.
                    # Request legal finger targets; do not widen that contract.
                    for i in range(len(target)):
                        if i%23>=7:target[i]=max(entry['lower'][i],min(entry['upper'][i],target[i]))
                if check in ['x','y','z']:target['xyz'.index(check)]+=.02
                elif check=='unreachable':target[:3]=[entry['upper'][i]-.01 for i in range(3)]
                action={'action_type':'ee_pose','action':target,'repeat':2 if check=='hold'else min(25,current['action_contract']['max_repeat']),
                    'summary':'Discarded PEAR-only '+suite+' '+check+' check.'}
                parse_action(json.dumps(action),contract=current['action_contract'],allow_stop=False)
                folder=root/'cases'/case['task']/check;save_observation(folder/'before',current)
                executions=[]
                # Native pose servos can approach a goal over several bounded
                # commands. Keep the original four-chunk movement diagnostic
                # allowance and the same measured motion tolerances. This is
                # not a change to the controller or the policy action budget.
                for attempt in range(4 if check in ['x','y','z']else 1):
                    current=client.step(action)
                    if current['done']:raise ValueError('Native task ended during the discarded control check')
                    executions.append(current['executed_control_steps'])
                    reached=vector(current,suite)
                    if math.sqrt(sum((a-b)**2 for a,b in zip(target[:3],reached[:3])))<=.005:break
                save_observation(folder/'after',current)
                reached=vector(current,suite);delta=[b-a for a,b in zip(before,reached[:3])]
                error=math.sqrt(sum((a-b)**2 for a,b in zip(target[:3],reached[:3])))
                steps=current['executed_control_steps'];passed=all(type(n)is int and 0<n<=action['repeat']for n in executions)
                if check=='hold':passed=passed and error<.015
                elif check in ['x','y','z']:
                    axis='xyz'.index(check)
                    passed=passed and .015<=delta[axis]<=.035 and math.sqrt(sum(v*v for i,v in enumerate(delta)if i!=axis))<=.015
                elif check=='unreachable':passed=passed and error>.05
                record={'suite':suite,'task':case['task'],'check':check,'passed':bool(passed),
                    'executed_control_steps':steps,'requested_control_steps':action['repeat'],
                    'control_calls':len(executions),'executed_steps_by_call':executions,
                    'executed_control_steps_total':sum(executions),
                    'measured_delta_m':delta,'position_error_m':error,'episode_uid':client.uid}
                with lock:
                    status['checks'].append(record);atomic_json(root/'status.json',{**status,'updated_unix':time.time()})
                print('PEAR_EEF_PROBE '+json.dumps(record),flush=True)
            final=client.step({'stop':True,'summary':'End discarded PEAR native EEF check.'})
            if not final['done']:raise ValueError('Native grade was not finalized')
            with lock:status.setdefault('finalized_tasks',[]).append(case['task'])
        except Exception as exc:
            with lock:
                status.setdefault('case_errors',{})[case['task']]=repr(exc)
                atomic_json(root/'status.json',{**status,'updated_unix':time.time()})
            print('PEAR_EEF_CASE_ERROR '+json.dumps({'task':case['task'],'error':repr(exc)}),flush=True)
            raise
        finally:client.close()
    try:
        if not spec.get('reuse_native_server'):
            with(root/'server.log').open('a')as log:
                server=subprocess.Popen([sys.executable,'-u',str(lane/'efe_pear_primary_native.py')],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status.update(server_pid=server.pid if server else None,phase='waiting_for_native_motion')
        with ThreadPoolExecutor(max_workers=3)as pool:
            for future in [pool.submit(check_case,case)for case in spec['probe_cases']]:future.result()
        status.update(phase='finished',passed=all(row['passed']for row in status['checks']))
    except Exception as exc:
        status.update(phase='failed',passed=False,error=repr(exc));raise
    finally:
        if not spec.get('reuse_native_server'):
            atomic_json(Path(spec['finished']),{'discarded_probe_finished':True})
        if server and server.poll()is None:
            try:server.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid,signal.SIGTERM)
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        stop.set();thread.join(timeout=2)
        atomic_json(root/'status.json',{**status,'updated_unix':time.time()})


if __name__=='__main__':main()
