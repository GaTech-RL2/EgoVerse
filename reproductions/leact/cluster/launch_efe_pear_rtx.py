"""Prepare disjoint RTX qualification lanes for every eligible task in a suite."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import yaml

from .launch_efe_pear_scaled import DEST,REMOTE,require_unpaused
from .launch_efe_training import freeze_sources
from .scale_efe import BASE,BUILD_ROOT,read,write,submit
from .efe_pear_submission import submit_reconciled

SOURCES=DEST/'sources/efe-validation-pinned'
SUITES=('robolab','robodojo','yamlab')
FILES=['efe_pear_rtx_check.py','efe_pear_rtx_env.py','efe_pear_native_entry.py',
    'efe_pear_public_observation.py','efe_pear_file_rpc.py','efe_action_loop.py',
    'efe_pear_catalog_probe.py','efe_pear_bridge.py','efe_pear_observation_check.py','efe_pear_context.py',
    'efe_pear_episode_archive.py','efe_pear_object_archive.py']


def payload(suite):
    candidates=read(DEST/'proposed-candidate-membership-v2.json')['rows']
    audit=read(DEST/'source-scene-audit.json')
    rows=[r for r in candidates if r['suite']==suite and audit['tasks'][r['task']]['source_scene_check_passed']]
    native=DEST/'sources/efe-native-files';entries={}
    for directory in ['standard',suite]:
        for path in(native/directory).rglob('*'):
            if path.is_file():entries[str(path.relative_to(native))]=path
    for row in rows:
        directory=SOURCES/row['path']
        if not directory.is_dir():raise FileNotFoundError('Pinned native task is not staged: '+row['task'])
        for path in directory.rglob('*'):
            if path.is_file()and'__pycache__'not in path.parts:
                entries[str(path.relative_to(SOURCES))]=path
    digests={name:hashlib.sha256(path.read_bytes()).hexdigest()for name,path in entries.items()}
    key=hashlib.sha256(json.dumps(digests,sort_keys=True).encode()).hexdigest()[:16]
    archive=DEST/'sources'/f'rp_jhh_{suite}_qualified_inputs_{key}.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'w:gz')as stream:
            for name,path in sorted(entries.items()):stream.add(path,arcname=name,recursive=False)
    return rows,archive,digests


def prepare(suite,*,lanes=4):
    require_unpaused()
    if suite not in SUITES:raise ValueError('Unapproved RTX simulator')
    images=read(DEST/'native-rtx-images.json',{})
    if suite not in images:raise ValueError('Exact private native image is not published: '+suite)
    image=images[suite]
    if image.get('registry_manifest_readback_verified')is not True:
        raise ValueError('Native registry image has no verified receipt')
    rows,archive,digests=payload(suite);audit=read(DEST/'source-scene-audit.json')
    remote_archive=REMOTE+'/sources/'+archive.name
    upload=archive.with_suffix('.staged.json')
    if not upload.exists():
        workers=[read(DEST/(n+'-launch.json'),{})for n in ['native-rtx-publication','checkpoint-storage']
            if read(DEST/(n+'-status.json'),{}).get('scheduler_status')=='RUNNING']
        wf=next((v['name']for v in workers if v.get('name')),None)
        if not wf:raise ValueError('No owned staging workflow is available')
        result=subprocess.run(['osmo','workflow','rsync','upload',wf,'preflight',
            str(archive)+':'+REMOTE+'/sources/','--no-progress'],capture_output=True,text=True,timeout=180)
        if result.returncode:raise RuntimeError('Native RTX payload staging failed: '+result.stderr[-500:])
        write(upload,{'remote':remote_archive,'bytes':archive.stat().st_size})
    builds=[]
    for index in range(min(lanes,len(rows))):
        subset=rows[index::lanes]
        spec={'suite':suite,'tasks':subset,'image':image['image'],'primary_rl_updates':0,
            'source_revision':read(DEST/'proposed-candidate-membership-v2.json').get('source_revision'),
            'output':REMOTE+'/native-rtx-transport-checks/'+suite,
            'payload':remote_archive,'payload_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'files':digests,'positive_oracles':True,
            'native_tool_relative':'rcb_robolab.py'if suite=='robolab'else'privileged/environment/rcb_'+suite+'.py',
            'expected_scenes':{r['task']:audit['tasks'][r['task']]['scene']['scene_sha256']for r in subset},
            'adapter_digests':{n:hashlib.sha256((BASE/'cluster'/n).read_bytes()).hexdigest()for n in FILES}}
        key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
        build=BUILD_ROOT/('rp-jhh-pear-rtx-'+suite+'-'+key)
        if not(build/'source-manifest.json').exists():
            build.mkdir();write(build/'spec.json',spec)
            for name in FILES:shutil.copyfile(BASE/'cluster'/name,build/name)
            python='/opt/robolab/.venv/bin/python'if suite=='robolab'else'/opt/venv/bin/python'
            workflow={'workflow':{'name':'leact-'+build.name,
                'resources':{'default':{'gpu':1,'cpu':12,'memory':'96Gi','storage':'150Gi','platform':'ovx-l40'}},
                'timeout':{'queue_timeout':'20m','exec_timeout':'6h'},
                'labels':{'pear_run_id':'moe-warmstart496-pearplusplus','pear_phase':'rtx_transport_'+suite},
                'tasks':[{'name':'preflight','image':image['image'],'command':[python],
                    'args':['-u','/tmp/pear-rtx/efe_pear_rtx_check.py'],
                    'environment':{'EFE_WORKFLOW_ID':'{{workflow_id}}','PYTHONUNBUFFERED':'1',
                        'PYTHONDONTWRITEBYTECODE':'1','NVIDIA_DRIVER_CAPABILITIES':'all',
                        'OMP_NUM_THREADS':'2','PEAR_PRIMARY_TRAINING':'0'},
                    'files':[{'localpath':n,'path':'/tmp/pear-rtx/'+n}for n in FILES+['spec.json']]}]}}
            (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
        builds.append(build)
    return builds


def launch(suite,*,lanes=4):
    with(DEST/'rtx-transport-launch.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        queue=read(DEST/'rtx-transport-queue.json',{})
        for build in prepare(suite,lanes=lanes):
            if build.name in queue:continue
            receipt=submit_reconciled(build,priority='HIGH',pool='groot-l40-02',nodes=1)
            spec=read(build/'spec.json')
            queue[build.name]={**receipt,'workflow':receipt['name'],'build':str(build),'suite':suite,
                'tasks':[r['task']for r in spec['tasks']],'pool':'groot-l40-02','priority':'HIGH','gpu':1}
            write(DEST/'rtx-transport-queue.json',queue)
        return queue


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite',choices=SUITES);parser.add_argument('--lanes',type=int,default=4)
    args=parser.parse_args();print(json.dumps(launch(args.suite,lanes=args.lanes),indent=2))
