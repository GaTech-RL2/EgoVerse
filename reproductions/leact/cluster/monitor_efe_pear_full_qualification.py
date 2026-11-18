"""Reconcile owned transport checks, retain receipts and finish RTX publication.

Only infrastructure failures retry automatically. Native action/reward failures
remain visible and cannot silently remove a task from the training mixture.
"""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import yaml

from .launch_efe_pear_scaled import DEST,REMOTE
from .launch_efe_training import freeze_sources
from .scale_efe import command,read,write,submit
from .efe_pear_submission import submit_reconciled,scheduling_priority


def download(remote,target,*,native_yam=False):
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    candidates=[]
    if not native_yam:
        for name in ['receipt-relay','native-rtx-publication','checkpoint-storage']:
            status=read(DEST/(name+'-status.json'),{})
            if status.get('scheduler_status')=='RUNNING':
                candidates.append((status['workflow'],'preflight'))
    for item in read(DEST/'rtx-transport-queue.json',{}).values():
        if item.get('scheduler_status')=='RUNNING'and(item.get('filesystem')=='amlfs05')==native_yam:
            candidates.append((item['workflow'],'preflight'))
    # Read-only fallbacks through existing mounted filesystems. No settings or
    # processes in either independent experiment are changed.
    candidates.append(('pear-tgen-l40-20261118-v3-1','node')if native_yam else
        ('leact-efe-coverage-leact-from-step496-r1-v1-p7-r54-1','preflight'))
    for wf,task in dict.fromkeys(candidates):
        result=subprocess.run(['osmo','workflow','rsync','download',wf,task,
            remote+':'+str(target)+'/', '--no-progress'],capture_output=True,text=True,timeout=60)
        if not result.returncode:return target/Path(remote.rstrip('/')).name
        if 'No such file or directory'in result.stderr+result.stdout:break
    raise RuntimeError('Read-only receipt transfer failed')


def retry(item):
    old=Path(item['build']);attempt=item.get('attempt',0)+1
    target=old.with_name(old.name.rsplit('-allocation-r',1)[0]+'-allocation-r'+str(attempt))
    if not(target/'source-manifest.json').exists():
        target.mkdir()
        for name,digest in read(old/'source-manifest.json').items():
            if hashlib.sha256((old/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen worker source changed')
            shutil.copyfile(old/name,target/name)
        if item.get('resumed_checks'):
            source=Path(__file__).resolve().parent;spec=read(target/'spec.json')
            for name in ['efe_pear_rtx_check.py','efe_pear_validation_resume.py']:
                shutil.copyfile(source/name,target/name)
                spec['adapter_digests'][name]=hashlib.sha256((target/name).read_bytes()).hexdigest()
            spec['resumed_checks']=item['resumed_checks'];write(target/'spec.json',spec)
        value=yaml.safe_load((target/'workflow.yaml').read_text());value['workflow']['name']='leact-'+target.name
        if value['workflow'].get('labels',{}).get('pear_phase')=='primary_native_workers':
            # Measured cold Isaac image pulls can exceed the former 20-minute
            # queue window even after a GPU has been assigned.
            value['workflow']['timeout']['queue_timeout']='45m'
        if item.get('resumed_checks'):
            task=value['workflow']['tasks'][0];name='efe_pear_validation_resume.py'
            if not any(f['localpath']==name for f in task['files']):
                parent=str(Path(task['files'][0]['path']).parent)
                task['files'].append({'localpath':name,'path':parent+'/'+name})
        (target/'workflow.yaml').write_text(yaml.safe_dump(value,sort_keys=False));freeze_sources(target)
    # A quota reclaim can prevent even native startup. Alternate the two
    # compatible AMLFS07 pools rather than retrying the same quota repeatedly;
    # return to HIGH every third attempt and otherwise use LOW.
    priority='HIGH'if attempt%3==0 else'LOW'
    pool=item['pool']
    # Primary worker paths are frozen for the model's selected pool. Preserve
    # that selection; qualification-only probes can alternate compatible pools.
    if(pool in {'groot-l40-02','groot-l40-03'}and(not item.get('owner_model_workflow')or item.get('allow_amlfs07_pool_fallback'))
        and item.get('filesystem','amlfs07')=='amlfs07'):
        pool='groot-l40-03'if pool=='groot-l40-02'else'groot-l40-02'
    priority=scheduling_priority(pool,priority,read(DEST/'control.json',{}))
    result=submit_reconciled(target,priority=priority,pool=pool,nodes=1)
    item.setdefault('history',[]).append({'workflow':item['workflow'],'status':item.get('scheduler_status')})
    item.update(**result,workflow=result['name'],build=str(target),priority=priority,pool=pool,attempt=attempt)
    item.pop('receipt',None);item.pop('worker',None);item.pop('source_failure',None)
    item['scheduler_status']='PENDING'


def tick_queue(filename,*,native_yam=False):
    from .efe_pear_submission import infrastructure_failure as infra_retryable
    from ..tracking.reasoning_live import command as logs
    queue=read(DEST/filename,{})
    marker='PEAR_NATIVE_YAM_CHECK 'if native_yam else'PEAR_NATIVE_RTX_CHECK '
    for item in queue.values():
        if item.get('receipt')or item.get('source_failure'):continue
        state=command(['osmo','workflow','query',item['workflow'],'--format-type','json'])
        item['scheduler_status']=state['status']
        if state.get('start_time'):
            for line in logs(['osmo','workflow','logs',item['workflow'],'-n','10'],timeout=4).splitlines():
                if marker in line:
                    try:item['worker']=json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
                    except ValueError:pass
        if state['status'].startswith('FAILED') and infra_retryable(state) and item.get('attempt',0)<32:
            if not native_yam and state.get('start_time'):
                spec=read(Path(item['build'])/'spec.json')
                try:
                    path=download(spec['output']+'/'+item['workflow']+'/checks.json',
                        DEST/'production-transport-partial-receipts'/item['workflow'],
                        native_yam=item.get('filesystem')=='amlfs05')
                    from .efe_pear_validation_resume import passed_checks
                    sha=hashlib.sha256(path.read_bytes()).hexdigest()
                    retained=passed_checks(spec,read(path),sha)
                    item.setdefault('resumed_checks',{}).update(retained)
                except (RuntimeError,subprocess.TimeoutExpired,ValueError,FileNotFoundError):pass
            retry(item)
        elif state['status']=='COMPLETED' or state['status'].startswith('FAILED'):
            spec=read(Path(item['build'])/'spec.json')
            remote=spec['output']+'/'+item['workflow']+'/checks.json'
            try:path=download(remote,DEST/'production-transport-receipts'/item['workflow'],
                native_yam=native_yam or item.get('filesystem')=='amlfs05')
            except Exception:
                if state['status'].startswith('FAILED'):
                    item['source_failure']=True
                    item['failure_summary']=state.get('groups',[])
                    write(DEST/filename,queue)
                    continue
                raise
            checks=read(path)
            expected={r['env_name']if native_yam else r['task']for r in spec['tasks']}
            passed=expected==set(checks) and all(v.get('success')if native_yam else
                v.get('native_rtx_api_passed')and v.get('native_file_transport_passed') for v in checks.values())
            item['receipt']={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'passed':bool(passed),'checked':len(checks),'expected':len(expected)}
            if not passed:item['source_failure']=True
        write(DEST/filename,queue)
    return queue


def images():
    from .launch_efe_pear_rtx import launch
    published=read(DEST/'native-rtx-images.json',{})
    publication=read(DEST/'native-rtx-publication-status.json',{})
    progress=publication.get('worker',{})
    for suite in ['robodojo','yamlab']:
        if suite not in published:
            available=(publication.get('scheduler_status')=='COMPLETED'or
                progress.get('phase')=='private_image_publication_completed'or
                '/'+suite+'/'in progress.get('receipt','')or
                any('/'+suite+'/'in receipt for receipt in progress.get('receipts',[])))
            if not available:continue
            remote=REMOTE+'/native-image-cache/'+suite+'/private-registry-publication/registry-image.json'
            try:
                path=download(remote,DEST/'native-rtx-registry-receipts'/suite)
                value=read(path)
                if value.get('registry_manifest_readback_verified')and value.get('config_and_layers_unchanged'):
                    published[suite]=value;write(DEST/'native-rtx-images.json',published)
            except Exception:continue
        if suite in published:launch(suite,lanes=4 if suite=='robodojo'else 2)


def main():
    with(DEST/'full-qualification-watch.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not read(DEST/'control.json',{}).get('paused',True):
            try:
                images()
                rtx=tick_queue('rtx-transport-queue.json')
                yam=tick_queue('yam-production-transport-queue.json',native_yam=True)
                rows=[*rtx.values(),*yam.values()]
                value={'updated_utc':datetime.now(timezone.utc).isoformat(),'primary_rl_updates':0,
                    'checks':rows,'tasks_passed':sum(v.get('receipt',{}).get('checked',0)for v in rows if v.get('receipt',{}).get('passed')),
                    'source_failures':sum(bool(v.get('source_failure'))for v in rows),
                    'all_production_transports_passed':len(rtx)>=10 and len(yam)>=4 and all(v.get('receipt',{}).get('passed')for v in rows)}
                write(DEST/'full-qualification-status.json',value)
            except Exception as exc:
                write(DEST/'full-qualification-observer-error.json',{'error_type':type(exc).__name__,'error':str(exc)[-1200:]})
            time.sleep(90)


if __name__=='__main__':main()
