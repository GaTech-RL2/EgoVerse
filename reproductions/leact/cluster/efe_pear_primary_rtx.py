"""Serve admitted RTX task episodes in their pinned native simulator image."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tarfile

from efe_pear_file_rpc import atomic_json
from efe_pear_remote_worker import serve
from efe_pear_rtx_env import NativeRtxEpisode


def main():
    lane=Path(__file__).resolve().parent;spec=json.loads((lane/'spec.json').read_text())
    root=Path(spec['output'])/os.environ['EFE_WORKFLOW_ID'];root.mkdir(parents=True,exist_ok=True)
    archive=Path(spec['payload']);digest=hashlib.sha256()
    with archive.open('rb')as stream:
        while block:=stream.read(8<<20):digest.update(block)
    if digest.hexdigest()!=spec['payload_sha256']:raise ValueError('Pinned native task payload changed')
    source=root/'source';source.mkdir()
    with tarfile.open(archive)as stream:stream.extractall(source,filter='data')
    for name,expected in spec['files'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Native service or task input changed')
    sys.path.insert(0,str(source/'standard'))
    credential=Path('/dev/shm/rp_jhh_rtx_archive_credentials.json')
    descriptor=os.open(credential,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w')as stream:stream.write(os.environ.pop('RP_JHH_PRIVATE_ARCHIVE_CONFIGURATION'))
    dependencies=root/'object-store-packages'
    subprocess.run([shutil.which('uv')or'/root/.local/bin/uv','pip','install','--python',sys.executable,
        '--target',str(dependencies),'boto3==1.44.5'],check=True,timeout=300)
    sys.path.insert(0,str(dependencies))
    os.environ['PEAR_OBJECTSTORE_CREDENTIAL_FILE']=str(credential)
    os.environ['PEAR_EPISODE_ARCHIVE']=spec['episode_archive']
    membership=json.loads((lane/'native-membership.json').read_text())
    config={'source':str(source),'native_tool_relative':spec['native_tool_relative'],
        'episode_root':str(root/'episodes'),'qualification_only':False,
        'training_tasks':{k:v for k,v in membership['training_tasks'].items()if v['row']['suite']==spec['suite']}}
    path=root/'native-config.json';atomic_json(path,config);os.environ['EFE_PEAR_RTX_CONFIG']=str(path)
    try:
        serve(spec['transport_root'],os.environ['EFE_WORKFLOW_ID'],{'efe/'+spec['suite']},
            NativeRtxEpisode,finished=Path(spec['finished']))
    finally:credential.unlink(missing_ok=True)


if __name__=='__main__':main()
