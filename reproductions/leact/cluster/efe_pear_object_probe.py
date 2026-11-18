"""Exercise a real pool-separated native reset, action and finalized verdict."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    root=Path(spec['output']);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    from efe_pear_file_rpc import atomic_json
    raw=os.environ.pop('RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION')
    credential=Path('/dev/shm/rp_jhh_object_probe_credentials.json')
    fd=os.open(credential,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w')as stream:stream.write(raw)
    os.environ['PEAR_OBJECTSTORE_CREDENTIAL_FILE']=str(credential)
    stopped=threading.Event();store=None;client=None;passed=False
    try:
        deps=Path('/tmp/rp_jhh_probe_packages')
        subprocess.run([sys.executable,'-m','pip','install','--quiet','--target',str(deps),'boto3==1.44.5'],check=True,timeout=300)
        sys.path.insert(0,str(deps))
        import numpy as np
        from efe_pear_object_archive import configured_client
        from efe_pear_object_transport import Store,ObjectEpisodeClient
        store=Store(spec['uri'],configured_client())
        def heartbeat():
            while not stopped.is_set():
                store.put_json('control/head-heartbeat.json',{'updated_unix':time.time(),'phase':'cross_pool_native_probe'})
                stopped.wait(10)
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        client=ObjectEpisodeClient(store,'native_yam/'+spec['task'],{'task':spec['task'],'reset_id':spec['reset_id']},timeout=1800)
        started=time.monotonic();opened=client.open();open_seconds=time.monotonic()-started
        if opened['done']or not opened.get('images'):raise ValueError('Native training reset lacks live images')
        cameras=[]
        for index,image in enumerate(opened['images']):
            array=np.asarray(image)
            if array.ndim!=3 or not np.isfinite(array).all()or float(array.std())<.1:
                raise ValueError('Native reset camera is blank or malformed')
            name=f'camera-{index}.png';image.save(root/name)
            cameras.append({'file':name,'size':list(image.size),'std':float(array.std()),
                'sha256':hashlib.sha256(array.tobytes()).hexdigest()})
        joints=opened['state']['proprio']['now']
        if not np.isfinite(joints).all():raise ValueError('Native joint observation is non-finite')
        action={'action_type':'joint_position','action':joints,'repeat':1}
        started=time.monotonic();stepped=client.step(action);action_seconds=time.monotonic()-started
        if stepped.get('executed_control_steps')!=1:raise ValueError('Command was not executed for one native step')
        terminal=stepped if stepped['done']else client.step({'stop':True})
        if not terminal['done']or not isinstance(terminal.get('native_termination_terms'),dict):
            raise ValueError('Native episode has no finalized termination verdict')
        passed=True
        receipt={'passed':True,'transport_roundtrip_passed':True,'task':spec['task'],'reset_id':spec['reset_id'],
            'transport_sha256':spec['transport_sha256'],'assets_root':spec['assets_root'],
            'episode_id':client.uid,'cameras':cameras,'executed_control_steps':1,
            'native_rewards':terminal['rewards'],'native_termination_terms':terminal['native_termination_terms'],
            'native_trial_path':terminal.get('native_trial_path'),'open_seconds':open_seconds,
            'action_seconds':action_seconds,'model_side_filesystem':spec['model_side_filesystem'],
            'simulator_side_filesystem':spec['simulator_side_filesystem'],'updated_unix':time.time()}
        atomic_json(root/'receipt.json',receipt)
        print('PEAR_CROSS_POOL_PROBE '+json.dumps(receipt),flush=True)
    finally:
        if client:client.close()
        stopped.set()
        if store:store.put_json('control/finished.json',{'completed':passed,'validation':True})
        credential.unlink(missing_ok=True)


if __name__=='__main__':main()
