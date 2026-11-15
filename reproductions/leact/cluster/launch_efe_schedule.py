"""Run the matched three-pass EFE warmstart schedule comparison on H100s."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import yaml

from .efe_train_vl import training_plan
from .launch_efe_training import freeze_sources, launch_monitor
from .scale_efe import BASE, BUILD_ROOT, ROOT, read, write, submit

STUDY_ID='efe-builderbench-schedule-20261027-v1'
PLAN=ROOT/'training-pilot/schedule-comparison.json'
VARIANTS=[
    {'stage':'warm-bb-1e6-p3','lr':1e-6,'name':'Warmstart: 3 passes, LR 1e-6',
     'baseline':'qwen-warmstart','role':'Longer-budget control at the original learning rate'},
    {'stage':'warm-bb-1e5-p3','lr':1e-5,'name':'Warmstart: 3 passes, LR 1e-5',
     'baseline':'qwen-warm-bb-1e6-p3','role':'Intermediate learning-rate comparison at the same exposure'},
    {'stage':'warm-bb-1e4-p3','lr':1e-4,'name':'Warmstart: 3 passes, BuilderBench LR 1e-4',
     'baseline':'qwen-warm-bb-1e6-p3','role':'BuilderBench learning-rate and three-pass reference with EFE visual inputs'},
]


def comparison_spec(original, variant, rows):
    stage=variant['stage']
    spec={**original,'stage':stage,'balanced_passes':3.0,'lr':variant['lr'],
          'checkpoint_fractions':[1/3,2/3,1.0],
          'archive_root':'/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-'+stage+'-20261027-v1',
          'local_results_directory':'training-pilot/'+stage+'-v1',
          'scope':variant['role']+'. Frozen 256-state pilot corpus, base initialization, 32K visual context, batch 32 and balanced sampling retained. Three sampled passes are not three coverage-complete epochs.',
          'next':'Automatically evaluate the fixed final 48-update checkpoint on the same seven native development tasks and the independent capability screen. No selection on capability or core scores.'}
    for key in ('recovery_of','recovery_reason'):spec.pop(key,None)
    spec['schedule_comparison']={'study_id':STUDY_ID,
        'declared_changes':{key:{'previous':original.get(key,1.0),'current':spec[key]}
                            for key in ('lr','balanced_passes') if spec[key]!=original.get(key,1.0)}}
    steps,schedule,_,digest=training_plan(rows,spec)
    if steps!=48:raise ValueError('Expected the frozen 256-state, batch-32, three-pass pilot')
    spec.update(expected_optimizer_steps=steps,expected_schedule_sha256=digest)
    return spec,schedule


def build_all():
    source=BUILD_ROOT/'efe-training-pilot-20261027-v5'
    original=read(source/'spec.json')
    dataset=ROOT/'training-pilot/frozen-dataset-manifest.json'
    if hashlib.sha256(dataset.read_bytes()).hexdigest()!=original['prepared_dataset']['manifest_sha256']:
        raise ValueError('Comparison dataset differs from the original pilot')
    rows=read(dataset)['examples'];arms=[];builds=[];common_schedule=None
    for variant in VARIANTS:
        stage=variant['stage'];build=BUILD_ROOT/('efe-'+stage+'-20261027-v1');build.mkdir(exist_ok=True)
        spec,schedule=comparison_spec(original,variant,rows)
        if common_schedule is None:common_schedule=[r['id'] for r in schedule]
        elif common_schedule!=[r['id'] for r in schedule]:raise ValueError('LR variants have different state exposure')
        if (build/'source-manifest.json').exists():
            if read(build/'spec.json')!=spec:raise ValueError('Frozen comparison spec changed')
        else:
            write(build/'spec.json',spec)
            for source_name,target in [
                ('efe_training_resume.py','runner.py'),('efe_training_pilot.py','pilot_helpers.py'),
                ('efe_training_data.py','data.py'),('efe_train_vl.py','trainer.py'),
                ('efe_budget_gateway.py','gateway.py'),('text_capability_eval.py','cuda_helpers.py')]:
                shutil.copyfile(BASE/'cluster'/source_name,build/target)
            shutil.copyfile(source/'bootstrap.sh',build/'bootstrap.sh')
            workflow=yaml.safe_load((source/'workflow.yaml').read_text())
            workflow['workflow']['name']='leact-efe-'+stage
            (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False))
            freeze_sources(build)
        builds.append(build)
        arms.append({**variant,'id':'qwen-'+stage,'build':str(build),
            'local_results_directory':spec['local_results_directory'],
            'evaluation_directory':'training-pilot/'+stage+'-evaluation',
            'optimizer_steps':48,'effective_batch':32,'balanced_passes':3,
            'example_presentations':len(schedule),'distinct_decision_states':len({r['state_id'] for r in schedule}),
            'snapshot_steps':[16,32,48],'schedule_sha256':spec['expected_schedule_sha256']})
    plan={'study_id':STUDY_ID,'created_utc':datetime.now(timezone.utc).isoformat(),'arms':arms,
          'initial_checkpoint':original['base_model'],'dataset_sha256':original['prepared_dataset']['manifest_sha256'],
          'available_decisions':256,'teacher_episodes':21,'training_context':32768,
          'sampling':'Identical presampled hierarchical suite/task/episode/decision draws with replacement across all three LRs; exactly equal forward/backward mass.',
          'qualification':'Matches BuilderBench peak LR and three-pass budget in the reference arm. Uses the EFE VL model, 32K context, balanced sampling and per-example loss reduction; not an exact copy of the text-model experiment.',
          'evaluation':'Fixed final checkpoints on the same seven development tasks. Capability screens are reported separately and cannot select the recipe. Official 449-task core stays untouched.',
          'next':'Use development task success and executable-output validity to choose the broader warmstart/LeAct recipe; retain every schedule arm and its failures.'}
    if PLAN.exists():
        old=read(PLAN);plan['created_utc']=old['created_utc']
        if old!=plan:raise ValueError('Frozen schedule comparison plan changed')
    else:write(PLAN,plan)
    registry_path=ROOT/'capability-retention/checkpoints.json';registry=read(registry_path)
    for arm in arms:
        if any(x['id']==arm['id'] for x in registry['checkpoints']):continue
        registry['checkpoints'].append({'id':arm['id'],'name':arm['name'],'parent':'qwen-base',
            'status':'waiting_for_checkpoint','selection':'Fixed final 48-update schedule-comparison checkpoint, independent of capability scores'})
    write(registry_path,registry)
    return builds


def ensure_monitor(build):
    receipt=read(build/'monitor-launch.json',{})
    if receipt.get('pid'):
        try:os.kill(receipt['pid'],0)
        except ProcessLookupError:pass
        else:return
    launch_monitor(build)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    for build in build_all():
        if args.submit:
            receipt=submit(build);ensure_monitor(build)
            print(json.dumps({'build':str(build),'workflow':receipt['name']}),flush=True)
        else:print(build)
    from ..embodied_first_exam.report import publish
    publish()


if __name__=='__main__':main()
