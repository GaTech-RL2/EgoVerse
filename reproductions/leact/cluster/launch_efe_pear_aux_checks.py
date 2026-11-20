"""Monitor owned auxiliary checks and retry verified infrastructure failures."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

import yaml

from .launch_efe_pear_scaled import DEST
from .launch_efe_pear_warmstart import ARM
from .launch_efe_training import freeze_sources
from .scale_efe import command,read,write,submit

CHECKS={
    'primary-driver-check':('groot-l40-02','PEAR_DRIVER_CHECK '),
    'queue-checkpoint':('groot-l40-03','PEAR_QUEUE_CHECK '),
    'receipt-relay':('groot-l40-02','PEAR_RECEIPT_RELAY '),
    'native-robolab-rtx':('groot-l40-02','PEAR_NATIVE_RTX_CHECK '),
    'native-yam-transport':('groot-l40s-03','PEAR_NATIVE_YAM_CHECK '),
    'native-rtx-publication':('groot-l40-02','PEAR_NATIVE_IMAGE_PUBLISH '),
    'checkpoint-storage':('groot-l40-02','PEAR_ARCHIVE_CHECK '),
    'native-yam-07-transport':('groot-l40-03','PEAR_NATIVE_YAM_CHECK '),
}


def tick(name,pool,marker):
    from .efe_pear_submission import infrastructure_failure as infra_retryable
    from ..tracking.reasoning_live import command as logs
    launch_path=DEST/(name+'-launch.json');launch=read(launch_path,{})
    if not launch:return True
    workflow=launch.get('name',launch.get('workflow'))
    state=command(['osmo','workflow','query',workflow,'--format-type','json'])
    target=DEST/(name+'-status.json');prior=read(target,{})
    value={'workflow':workflow,'scheduler_status':state['status'],'primary_rl_updates':0,
        'updated_utc':datetime.now(timezone.utc).isoformat()}
    if prior.get('workflow')==workflow and prior.get('worker'):value['worker']=prior['worker']
    if state.get('start_time'):
        for line in logs(['osmo','workflow','logs',workflow,'-n','15'],timeout=5).splitlines():
            if marker in line:
                try:value['worker']=json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
                except ValueError:pass
    if state['status']=='COMPLETED':
        if name=='queue-checkpoint':
            from .monitor_efe_pear_full_qualification import download
            spec=read(Path(launch['build'])/'spec.json')
            path=download(spec['output']+'/'+workflow+'/checks.json',DEST/'queue-checkpoint-receipts')
            value['worker']=read(path)
            value['receipt_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        write(target,value);return True
    if state['status'].startswith('FAILED'):
        attempt=launch.get('attempt',0)
        if not infra_retryable(state)or attempt>=8:
            value['phase']='needs_source_review';write(target,value);return True
        original=Path(launch['build']);target_build=original.with_name(original.name.rsplit('-allocation-r',1)[0]+'-allocation-r'+str(attempt+1))
        if not(target_build/'source-manifest.json').exists():
            target_build.mkdir(exist_ok=True)
            for filename,digest in read(original/'source-manifest.json').items():
                if hashlib.sha256((original/filename).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen check source changed')
                shutil.copyfile(original/filename,target_build/filename)
            spec=yaml.safe_load((target_build/'workflow.yaml').read_text());spec['workflow']['name']='leact-'+target_build.name
            (target_build/'workflow.yaml').write_text(yaml.safe_dump(spec,sort_keys=False));freeze_sources(target_build)
        receipt=submit(target_build,priority='LOW',pool=launch.get('pool',pool),nodes=launch.get('nodes',1))
        launch.setdefault('history',[]).append({'workflow':workflow,'scheduler_status':state['status']})
        launch.update(**receipt,build=str(target_build),priority='LOW',attempt=attempt+1)
        write(launch_path,launch);value.update(workflow=receipt['name'],scheduler_status='PENDING',phase='infrastructure_retry_submitted')
        value.pop('worker',None)
    write(target,value);return False


def main():
    with(DEST/'aux-checks-watch.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finished={}
        while ARM in read(DEST/'control.json',{}).get('enabled_training_arms',[]):
            if read(DEST/'control.json',{}).get('paused'):return
            for name,(pool,marker)in CHECKS.items():
                launch=read(DEST/(name+'-launch.json'),{});identity=launch.get('name',launch.get('workflow'))
                if finished.get(name)==identity:continue
                try:
                    if tick(name,pool,marker):finished[name]=identity
                except Exception as exc:write(DEST/(name+'-observer-error.json'),{'error':str(exc)[-1000:]})
            time.sleep(30)


if __name__=='__main__':main()
