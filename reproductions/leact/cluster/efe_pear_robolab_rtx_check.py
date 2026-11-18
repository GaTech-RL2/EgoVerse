"""Run the exact native RoboLab RGB service directly in an allocated RTX pod.

No nested containers, model inference or policy supervision. This gate checks
reset/action/camera/native-negative-verdict integration. Positive oracle
coverage is task-specific and is kept separate from this API probe.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

from efe_action_loop import native_contract,parse_action
from efe_pear_catalog_probe import probe_action
from efe_pear_observation_check import validate_frames


def main():
    from PIL import Image
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    root=Path(spec['output'])/os.environ['EFE_WORKFLOW_ID'];root.mkdir(parents=True,exist_ok=True)
    archive=Path(spec['payload']);raw=archive.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=spec['payload_sha256']:raise ValueError('Native file/instance payload changed')
    source=root/'native-inputs';source.mkdir()
    with tarfile.open(archive)as stream:stream.extractall(source,filter='data')
    for name,digest in spec['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned native service input changed')
    names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
    if len(names)!=1 or not any(k in names[0]for k in ['L40','RTX','L4']):raise ValueError('The native probe requires exactly one allocated RTX GPU')
    sys.path.insert(0,str(source/'standard'))
    from rcb_client import Robot
    results={}
    for row in spec['tasks']:
        target=root/row['task'];target.mkdir();srv=target/'srv';srv.mkdir();data=target/'data';data.mkdir()
        for name in ['server.py','live_video.py','rcb_robolab.py','finalize.sh']:
            original=source/'robolab'/('sim/'+name if name!='rcb_robolab.py'else name)
            shutil.copyfile(original,srv/name)
        for name in ['rcb1.py','rcb_limited.py']:shutil.copyfile(source/'standard'/name,srv/name)
        task=source/row['path'];instance=task/'environment/sim/instance.json'
        import yaml
        native=yaml.safe_load((task/'environment/docker-compose.yaml').read_text())['services']['sim']['environment']
        env={**os.environ,**{k:str(v)for k,v in native.items()},'PYTHONPATH':str(srv),
            'RCB_INSTANCE':str(instance),'RCB_DATA':str(data),'RCB_TOOL':str(srv/'rcb_robolab.py'),
            'RCB_SRV':str(srv),'RCB_SERVER':str(srv/'server.py'),'SIM_URL':'http://127.0.0.1:8000',
            'NVIDIA_DRIVER_CAPABILITIES':'all','OMNI_KIT_ALLOW_ROOT':'1'}
        value={'task':row['task'],'native_rtx_api_passed':False,'training_admitted':False,
            'model_inference_calls':0,'primary_rl_updates':0,'reference_oracle':row.get('oracle','none')}
        process=None
        try:
            with(target/'native.log').open('w')as log:
                process=subprocess.Popen([sys.executable,'-u',str(srv/'server.py')],env=env,
                    cwd=srv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            robot=Robot(base='http://127.0.0.1:8000',timeout=1000)
            raw_spec=robot.spec()
            (target/'native-spec.json').write_text(json.dumps(raw_spec,indent=2)+'\n')
            contract=native_contract(raw_spec,allowed_action_types=spec.get('allowed_action_types'))
            before=robot.get_obs(cameras=contract['cameras'],include_depth=False)
            first=validate_frames([Image.fromarray(before.last[c])for c in contract['cameras']],contract['cameras'])
            action=probe_action(contract);parse_action(json.dumps(action),contract=contract,allow_stop=False)
            after=robot.submit(action['action_type'],[action['action']],dt=contract['control_dt'],
                cameras=contract['cameras'],video='frames',include_depth=False,note='Discarded native RTX API qualification')
            second=validate_frames([Image.fromarray(after.last[c])for c in contract['cameras']],contract['cameras'])
            if after.done:raise ValueError('The native API probe terminated before its second nonterminal observation')
            for stage,observation in [('reset',before),('after_action',after)]:
                directory=target/stage;directory.mkdir()
                for camera in contract['cameras']:Image.fromarray(observation.last[camera]).save(directory/(camera+'.png'))
            # The unchanged native collect hook signals the native service,
            # reads its actual live grader and then finalizes the native video.
            with(target/'finalize.log').open('w')as log:
                finalized=subprocess.run(['sh',str(srv/'finalize.sh')],env=env,cwd=srv,
                    stdout=log,stderr=subprocess.STDOUT,timeout=3600)
            verdict=json.loads((data/'final.json').read_text())
            if (finalized.returncode or verdict.get('finalized')is not True or verdict.get('success')is not False
                or any(verdict.get(k)for k in ['infra_error','instance_mismatch','sim_error','no_verdict'])):
                raise ValueError('The native no-model control lacks a trustworthy finalized failure verdict')
            value.update(native_rtx_api_passed=True,camera_evidence={'reset':first,'after_action':second},
                native_action=action,action_contract=contract,native_final=verdict,
                positive_solver_control='Not exercised by this API check; see task-specific oracle metadata')
        except Exception as exc:value['error']=repr(exc)
        finally:
            if process and process.poll()is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        results[row['task']]=value
        (root/'checks.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        print('PEAR_NATIVE_RTX_CHECK '+json.dumps({'checked':len(results),'passed':sum(v['native_rtx_api_passed']for v in results.values()),
            'total':len(spec['tasks']),'primary_rl_updates':0,'training_admitted':False}),flush=True)
    if not all(v['native_rtx_api_passed']for v in results.values()):raise RuntimeError('Native RoboLab RTX API checks failed')


if __name__=='__main__':main()
