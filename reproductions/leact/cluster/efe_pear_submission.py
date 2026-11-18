"""Reconcile uncertain scheduler acceptance before retrying a PEAR submission."""
from datetime import datetime,timezone
from pathlib import Path
import time

import yaml

from .scale_efe import command,read,write,submit,ROOT


def scheduling_priority(pool,default,control):
    key=('l40_priority'if pool.startswith('groot-l40')else
         'h100_priority'if pool.startswith('groot-h100')else None)
    priority=control.get(key,default)if key else default
    if priority not in {'HIGH','NORMAL','LOW'}:
        raise ValueError('Unsupported requested scheduling priority')
    return priority


def simulator_pool(default,control):
    pool=control.get('l40_pool',default)
    if pool not in {'groot-l40-02','groot-l40-03','groot-l40s-01'}:
        raise ValueError('Unqualified simulator pool requested')
    return pool


def infrastructure_failure(scheduler):
    from .launch_efe_coverage import infra_retryable
    details=str(scheduler).lower()
    device_rejection=(str(scheduler.get('status','')).startswith('FAILED')
        and'allocate failed'in details and'requested number of devices unavailable'in details
        and'nvidia.com/gpu'in details)
    return bool(device_rejection or infra_retryable(scheduler))


def submit_reconciled(build,*,priority,pool,nodes=1):
    build=Path(build)
    intent=read(build/'submission-intent.json',{})
    if intent and not(build/'submission.json').exists():
        name=yaml.safe_load((build/'workflow.yaml').read_text())['workflow']['name']
        rows=command(['osmo','workflow','list','--name',name,'--count','20','--format-type','json'])['workflows']
        matches=[v for v in rows if v['name']==name or v['name'].startswith(name+'-')]
        if len(matches)>1:raise RuntimeError('Multiple scheduler matches require review; no duplicate submitted')
        if matches:
            result={**matches[0],'reconciled_acceptance':True};write(build/'submission.json',result)
            reservations=read(ROOT/'node-reservations.json',{});reservations[result['name']]=nodes
            write(ROOT/'node-reservations.json',reservations)
            return result
        marker=build/'submission-absence-check.json';previous=read(marker,{})
        now=time.time()
        age=now-datetime.fromisoformat(intent['utc']).timestamp()
        if age<180 or not previous or now-previous['checked_unix']<30:
            write(marker,{'checked_unix':now,'scheduler_matches':0})
            raise RuntimeError('Waiting for a second scheduler absence check before retrying uncertain submission')
        # Two successful, spaced reads found no content-addressed workflow.
        # Retain the old intent as evidence before allowing a new submission.
        (build/'submission-intent.json').replace(build/f'submission-intent-unaccepted-{int(now)}.json')
    return submit(build,priority=priority,pool=pool,nodes=nodes)
