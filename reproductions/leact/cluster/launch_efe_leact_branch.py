"""Register an explicitly requested LeAct branch from the evaluated update 496."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from .scale_efe import read,write
from .launch_efe_coverage import DEST
from .launch_efe_coverage_train import build

ID='leact-from-step496-r1'


def parent_for_branch(result,pin,dataset,admission):
    ready=result.get('ready',{});replay=ready.get('loss_replay',{})
    if (result.get('success') is not True or result.get('step')!=496
        or ready.get('step')!=496 or ready.get('final') is not False
        or ready.get('additional_optimizer_updates')!=0 or replay.get('passed') is not True
        or replay.get('optimizer_updates')!=0 or not 0<=replay.get('absolute_error',float('inf'))<=.0005
        or pin.get('step')!=496 or dataset.get('sha256')!=pin.get('dataset_sha256')
        or admission.get('gate_passed') is not True
        or dataset.get('source_sha256')!=admission.get('decisions_sha256')):
        raise ValueError('LeAct branch requires the verified step-496 parent and its frozen training corpus')
    if not all(ready.get(k) for k in ['checkpoint','manifest','revision']):raise ValueError('Parent export is incomplete')
    return {'ready':ready,'dataset':dataset,'admission':admission}


def prepare():
    path=DEST/'branch-runs.json';registry=read(path,{'runs':[]})
    existing=next((r for r in registry['runs'] if r['id']==ID),None)
    if existing:return existing
    evidence=read(DEST/'evidence.json');result=read(DEST/'intermediate-496/result.json')
    pin_path=DEST/'intermediate-496/checkpoint-pin.json';pin=read(pin_path)
    if result['ready']['pin_sha256']!=hashlib.sha256(pin_path.read_bytes()).hexdigest():
        raise ValueError('Pinned parent receipt changed')
    parent=parent_for_branch(result,pin,evidence['training']['warmstart']['dataset'],evidence['admission'])
    arm={'id':ID,'label':'Warmstart 496 + LeAct R1','authorized':True,
        'parent_model':'warmstart-step496','baseline':'warmstart-step496','parent_step':496,
        'archive_root':str(Path(result['ready']['checkpoint']).parents[2]/ID),
        'local_results_directory':'fixed-action-v1/coverage-scale-v1/'+ID,
        'panel_sha256':hashlib.sha256((DEST/'evaluation-panel.json').read_bytes()).hexdigest(),
        'planned_tasks':36,'parent_ready':result['ready'],
        'scope':'User-requested LeAct branch initialized from evaluated warmstart update 496. '
            'One round on up to 2048 balanced states from the unchanged 256-episode training corpus: '
            '8 student explanations per state, 4096-token cap, top 2 positive complete-command likelihood gains, '
            '50% warmstart replay and one balanced pass-equivalent. Same 36-task native evaluation panel. '
            'The separate final 737-update warmstart comparison continues independently.',
        'created_utc':datetime.now(timezone.utc).isoformat()}
    folder=build('leact-r1',read(DEST/'plan.json'),parent,arm=arm)
    arm['build']=str(folder)
    directory=DEST/ID;directory.mkdir(exist_ok=True)
    write(directory/'plan.json',{**arm,'spec':read(folder/'spec.json'),
        'sequence':['Verify frozen step-496 model','Generate and score student explanations',
                    'Admit positive command-likelihood gains','Train with 50% replay',
                    'Verify saved loss and export','Matched 36-task native evaluation'],
        'interpretation':'Measures the combined effect of selected reasoning and additional optimization. '
            'A matched-budget continued-SFT control would be needed to isolate the selection rule.'})
    registry['runs'].append(arm);write(path,registry)
    return arm


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare']);parser.parse_args()
    print(json.dumps(prepare()))
