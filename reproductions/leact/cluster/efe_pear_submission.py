"""Reconcile uncertain scheduler acceptance before retrying a PEAR submission."""
from datetime import datetime,timezone
from pathlib import Path
import time

import yaml

from .scale_efe import command,read,write,submit,ROOT


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
