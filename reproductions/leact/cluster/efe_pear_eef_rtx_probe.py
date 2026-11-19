"""Model-free, isolated PEAR EEF checks through the real RTX episode bridge."""
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


def vector(result):
    entry=result['action_contract']['action_types']['ee_pose']
    state=dict(zip(result['state']['proprio']['layout'],result['state']['proprio']['now'],strict=True))
    sides=['left','right']if entry['dim']==16 else['']
    values=[]
    mapping=entry['gripper_mapping']
    for side in sides:
        prefix=side+'_tcp_'if side else''
        values.extend(state[prefix+name]for name in ['x','y','z','qw','qx','qy','qz'])
        native=state[(side+'_'if side else'')+'gripper_cmd']
        values.append((native-mapping['native_open'])/(mapping['native_closed']-mapping['native_open']))
    return values


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    if spec.get('control_interface')!=VERSION or spec.get('teacher_calls_permitted')is not False:
        raise ValueError('This is a model-free PEAR-only native EEF check')
    root=Path(spec['probe_root']);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    stop=threading.Event();server=None
    status={'phase':'starting_native_server','checks':[],'teacher_model_calls':0,'optimizer_updates':0,
        'control_interface':VERSION,'training_data_produced':False}
    def heartbeat():
        while not stop.is_set():
            atomic_json(Path(spec['head_heartbeat']),{'workflow':spec['owner_model_workflow'],'updated_unix':time.time()})
            atomic_json(root/'status.json',{**status,'updated_unix':time.time()});stop.wait(10)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        with(root/'server.log').open('a')as log:
            server=subprocess.Popen([sys.executable,'-u',str(lane/'efe_pear_primary_rtx.py')],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status.update(server_pid=server.pid,phase='waiting_for_native_motion')
        for case in spec['probe_cases']:
            client=FileEpisodeClient(spec['transport_root'],'efe/'+spec['suite'],case['env_args'],timeout=1800)
            try:
                current=client.open()
                if current['action_contract'].get('control_interface')!=VERSION:
                    raise ValueError('RTX sidecar has not exposed the explicit EEF interface')
                if current['scene_sha256']!=case['scene_sha256']:raise ValueError('Native reset scene mismatch')
                for check in ['hold','x','y','z','gripper','unreachable']:
                    entry=current['action_contract']['action_types']['ee_pose'];target=vector(current)
                    before=target[:3].copy()
                    if check in ['x','y','z']:target['xyz'.index(check)]+=.02
                    elif check=='gripper':target[7]=1.
                    elif check=='unreachable':target[:3]=[entry['upper'][i]-.01 for i in range(3)]
                    action={'action_type':'ee_pose','action':target,'repeat':2 if check=='hold'else 25,
                        'summary':'Discarded PEAR-only native EEF '+check+' check.'}
                    parse_action(json.dumps(action),contract=current['action_contract'],allow_stop=False)
                    folder=root/'cases'/case['task']/check;save_observation(folder/'before',current)
                    current=client.step(action)
                    if current['done']:raise ValueError('Native task terminated during discarded motion check')
                    save_observation(folder/'after',current)
                    actual=vector(current);delta=[b-a for a,b in zip(before,actual[:3])]
                    error=math.sqrt(sum((a-b)**2 for a,b in zip(target[:3],actual[:3])))
                    steps=current['executed_control_steps']
                    passed=0<steps<=action['repeat']
                    if check=='hold':passed=passed and error<.015
                    elif check in ['x','y','z']:
                        axis='xyz'.index(check)
                        passed=passed and .015<=delta[axis]<=.035 and math.sqrt(sum(
                            x*x for i,x in enumerate(delta)if i!=axis))<=.015
                    elif check=='unreachable':passed=passed and error>.05
                    status['checks'].append({'task':case['task'],'check':check,'passed':bool(passed),
                        'executed_control_steps':steps,'requested_control_steps':action['repeat'],
                        'measured_delta_m':delta,'position_error_m':error,'episode_uid':client.uid})
                    atomic_json(root/'status.json',{**status,'updated_unix':time.time()})
                    print('PEAR_EEF_PROBE '+json.dumps(status['checks'][-1]),flush=True)
                final=client.step({'stop':True,'summary':'End discarded PEAR-only EEF check.'})
                if not final['done']:raise ValueError('Native finalization did not complete')
            finally:client.close()
        status.update(phase='finished',passed=all(row['passed']for row in status['checks']))
    except Exception as exc:
        status.update(phase='failed',passed=False,error=repr(exc));raise
    finally:
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
