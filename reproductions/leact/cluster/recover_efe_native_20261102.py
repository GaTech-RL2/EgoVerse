"""Finish the seven still-ungraded legacy LeAct tasks after quota reclamation."""
import argparse
import shutil
import subprocess
import sys
import yaml

from . import recover_efe_native_20261101 as recovery
from .scale_efe import ROOT,BASE,BUILD_ROOT,read,write
from .launch_efe_training import freeze_sources
from .launch_efe_direct_warmstart import now

DEST=ROOT/'recovery-20261102/native'
BUILD=BUILD_ROOT/'efe-native-recovery-20261102-leact-rebuilt-v3'


def prepare():
    if not (BUILD/'source-manifest.json').exists():
        source=BUILD_ROOT/'efe-native-recovery-20261101-leact-rebuilt'
        BUILD.mkdir(exist_ok=True)
        for name in read(source/'source-manifest.json'):shutil.copyfile(source/name,BUILD/name)
        spec=read(BUILD/'spec.json')
        graded={t['task'] for t in read(ROOT/'training-evaluation-v1/leact-rebuilt/behavior.json')['trials']}
        spec['tasks']=[t for t in spec['tasks'] if t['task'] not in graded]
        if len(graded)!=14 or len(spec['tasks'])!=7:raise ValueError('Recovery membership changed; inspect saved grades')
        spec.update(local_results_directory='training-evaluation-v1/leact-rebuilt-recovery-20261102',
            recovery_of=read(source/'submission.json')['name'],
            recovery_reason='Recovered shared results from the reclaimed node. Keep all fourteen graded outcomes and evaluate only the remaining seven tasks.')
        write(BUILD/'spec.json',spec)
        if 'DURABLE_ARCHIVE = None' not in (BUILD/'worker.py').read_text():raise ValueError('Durable native receipt writer missing')
        w=yaml.safe_load((BUILD/'workflow.yaml').read_text());w['workflow']['name']='leact-efe-native-leact-rebuilt-recovery-v3'
        (BUILD/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(BUILD)
    recovery.DEST=DEST
    recovery.ARMS=[{'id':'leact-rebuilt','source':'efe-native-recovery-20261101-leact-rebuilt',
        'remaining':7,'suffix':'v3','build_name':BUILD.name,'directory_suffix':'-recovery-20261102'}]
    return recovery.prepare()


def start():
    prepare()
    with (DEST/'monitor.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u','-m','reproductions.leact.cluster.recover_efe_native_20261102','monitor'],
            cwd=BASE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(DEST/'monitor-launch.json',{'pid':child.pid,'updated_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','start','monitor']);args=parser.parse_args()
    if args.action=='start':start()
    else:
        prepare()
        if args.action=='monitor':recovery.monitor()
