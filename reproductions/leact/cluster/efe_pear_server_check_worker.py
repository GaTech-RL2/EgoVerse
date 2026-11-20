"""CPU-only check of the exact eight-server PEAR train/development layout."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    root=Path(spec['output'])/os.environ['EFE_WORKFLOW_ID'];root.mkdir(parents=True,exist_ok=True)
    membership=Path(spec['reasoning_membership'])
    if hashlib.sha256(membership.read_bytes()).hexdigest()!=spec['reasoning_membership_sha256']:
        raise ValueError('Qualified reasoning scene bank changed')
    with(root/'install.log').open('w')as log:
        subprocess.run(['apt-get','update','-qq'],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        subprocess.run(['apt-get','install','-y','-qq','libosmesa6','libgl1','libglib2.0-0'],
            check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    auth=root/'ray-auth';auth.write_text(os.urandom(32).hex());auth.chmod(0o600)
    source=Path(spec['pear_source']);runtime=Path(spec['remote_root'])/'runtime'
    env={**os.environ,'RAY_AUTH_MODE':'token','RAY_AUTH_TOKEN_PATH':str(auth),
        'PYTHONPATH':':'.join(map(str,[lane,source/'pear/utils/sitecustomize',source,source/'3rdparty/verl/src'])),
        'EFE_PEAR_REASONING_MEMBERSHIP':str(membership),'EFE_PEAR_QUALIFICATION_DIR':str(root),
        'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MUJOCO_GL':'osmesa',
        'PEAR_SIMULATOR_STARTUP_DIAGNOSTICS':'1','RAY_DEDUP_LOGS':'0'}
    try:
        subprocess.run([str(runtime/'envs/verl/bin/python'),'-u','-m','efe_pear_process_check','--ray'],
            env=env,check=True,timeout=900)
    finally:
        if spec.get('observation_check_files'):
            from efe_pear_observation_audit import run
            for name in spec['observation_check_files']:
                path=Path(name)
                if path.exists():
                    target=root/'control-receipts'/path.parent.name;target.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(path,target/path.name)
            run({'check_files':spec['observation_check_files'],'output':str(root/'control-observation-audit.json')})
        logs=Path('/tmp/rp-jhh-pear-server-check-ray/session_latest/logs')
        target=root/'ray-diagnostics';target.mkdir(exist_ok=True)
        for path in [*logs.glob('worker-*.err'),*logs.glob('raylet.err'),*logs.glob('debug_state.txt')]:
            shutil.copyfile(path,target/path.name)


if __name__=='__main__':main()
