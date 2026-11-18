"""Verify owned migration checks, then release the existing primary-run gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import time

from .launch_efe_pear_scaled import DEST
from .scale_efe import BASE,read,write,command
from .efe_pear_submission import infrastructure_failure
from .monitor_efe_pear_full_qualification import retry
from ..tracking.reasoning_live import command as read_logs


def records(logs,marker):
    values=[]
    for line in logs.splitlines():
        if marker in line:
            try:values.append(json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0])
            except ValueError:pass
    return values


def verified_readiness(probe,assets,sources,rtx,transport_sha256):
    if not(assets.get('passed')and sources.get('passed')):return None
    if not(probe.get('passed')and probe.get('transport_roundtrip_passed')
        and probe.get('transport_sha256')==transport_sha256
        and probe.get('assets_root')==assets.get('assets_root')
        and probe.get('executed_control_steps')==1 and probe.get('cameras')
        and isinstance(probe.get('native_termination_terms'),dict)
        and probe.get('model_side_filesystem')=='AMLFS07'
        and probe.get('simulator_side_filesystem')=='LFS'):return None
    if set(rtx)!={'robodojo','robolab','yamlab'}or not all(row.get('passed')for row in rtx.values()):return None
    return {**probe,'assets':assets,'sources':sources,'rtx_migration_smokes':rtx,
        'passed':True,'verified_unix':time.time(),
        'scope':'Pinned input migration and one native reset/action/verdict smoke per graphics family; original full task admission is unchanged.'}


def tick():
    live=read(DEST/'l40s01-cross-pool-probe.json');rtx=read(DEST/'l40s01-rtx-smokes.json',{})
    jobs={**live['jobs'],**rtx}
    def query(item):
        role,row=item
        return role,command(['osmo','workflow','query',row['workflow'],'--format-type','json'])
    with ThreadPoolExecutor(max_workers=6)as executor:states=dict(executor.map(query,jobs.items()))
    status={'updated_unix':time.time(),'phase':'validating_pool_migration','primary_rl_updates':0,
        'jobs':{role:{'workflow':jobs[role]['workflow'],'status':state['status']}for role,state in states.items()}}
    for role,state in states.items():
        jobs[role]['scheduler_status']=state['status']
        if state['status'].startswith('FAILED'):
            if infrastructure_failure(state)and jobs[role].get('attempt',0)<8:
                retry(jobs[role])
                if role in live['jobs']:
                    live['jobs'][role]=jobs[role];write(DEST/'l40s01-cross-pool-probe.json',live)
                else:
                    rtx[role]=jobs[role];write(DEST/'l40s01-rtx-smokes.json',rtx)
            else:status.update(phase='migration_check_needs_source_review',failure_role=role)
    probe={};smokes={}
    if states['probe']['status']=='COMPLETED':
        rows=records(read_logs(['osmo','workflow','logs',jobs['probe']['workflow'],'-n','100'],timeout=8),'PEAR_CROSS_POOL_PROBE ')
        if rows:probe={**rows[-1],'validation_workflow':jobs['probe']['workflow']}
    for suite,row in rtx.items():
        if states[suite]['status']!='COMPLETED':continue
        rows=records(read_logs(['osmo','workflow','logs',row['workflow'],'-n','100'],timeout=8),'PEAR_NATIVE_RTX_CHECK ')
        if rows and rows[-1].get('checked')==1 and rows[-1].get('passed')==1 and rows[-1].get('total')==1:
            smokes[suite]={'passed':True,'task':row['task'],'workflow':row['workflow'],
                'spec_sha256':hashlib.sha256((Path(row['build'])/'spec.json').read_bytes()).hexdigest()}
    digest=hashlib.sha256((BASE/'cluster/efe_pear_object_transport.py').read_bytes()).hexdigest()
    ready=verified_readiness(probe,read(DEST/'l40s01-storage-receipts/asset-restore-receipt.json',{}),
        read(DEST/'l40s01-storage-receipts/source-restore-receipt.json',{}),smokes,digest)
    status.update(native_cross_pool_passed=bool(probe.get('passed')),rtx_smokes_passed=list(smokes))
    if ready:
        write(DEST/'l40s01-storage-ready.json',ready);status['phase']='pool_migration_verified'
    write(DEST/'l40s01-migration-status.json',status)
    return status


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    if not args.watch:print(json.dumps(tick()));return
    with(DEST/'l40s01-migration-watch.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not read(DEST/'control.json',{}).get('paused',True):
            try:
                value=tick();print(json.dumps(value),flush=True)
                if value['phase']in {'pool_migration_verified','migration_check_needs_source_review'}:return
            except Exception as exc:
                write(DEST/'l40s01-migration-observer-error.json',{'error_type':type(exc).__name__,
                    'error':str(exc)[-1000:],'updated_unix':time.time()})
            time.sleep(60)


if __name__=='__main__':main()
