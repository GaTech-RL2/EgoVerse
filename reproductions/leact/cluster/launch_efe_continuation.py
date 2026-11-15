"""Freeze the authorized step-32 to step-96 continuation; native evaluation first."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
import shutil

import yaml

from .efe_train_vl import training_plan
from .efe_training_resume import validate_reuse
from .efe_training_state import validate_continuation
from .launch_efe_training import freeze_sources
from .scale_efe import BASE,BUILD_ROOT,ROOT,read,write

STUDY='efe-warm-continue32-96-20261028-v1'
AUDIT=ROOT/'training-pilot/continuation-step32-96'
SOURCE=('/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-warm-bb-1e4-p3-20261027-v1-quota-retry4/'
        'leact-efe-warm-bb-1e4-p3-retry4-1')


def make_spec(original,rows,parent_schedule,hashes):
    lr=3.12696703292044e-5
    prefix=parent_schedule['exposures'][:32*32]
    spec={**original,'stage':'warm-continue32-96','balanced_passes':6.,'lr':lr,
          'checkpoint_fractions':[.5,2/3,5/6,1.], 'expected_optimizer_steps':96,
          'archive_root':'/mnt/amlfs-03/home/rpunamiya/leact/studies/'+STUDY,
          'local_results_directory':'training-pilot/warm-continue32-96-v1',
          'collective_timeout_seconds':900,'register_capability':False,
          'scope':'Authorized continuation from saved update 32 to total update 96: 64 additional updates. Same frozen corpus and sampled order. Initial bf16 model export with FP32 Adam restored; original per-rank CUDA RNG was not saved, so initial RNG is reset to the frozen seed. Subsequent retries restore full FP32 model, Adam and per-rank RNG.',
          'next':'Evaluate the final step-96 checkpoint on the same seven native EFE development tasks. Capability screen paused by user; official core untouched.'}
    for key in ('recovery_of','recovery_reason','expected_schedule_sha256'):spec.pop(key,None)
    spec['schedule_comparison']={'study_id':STUDY,'declared_changes':{
        key:{'previous':original.get(key,1.),'current':spec[key]}
        for key in ('lr','balanced_passes') if spec[key]!=original.get(key,1.)}}
    plan={'origin_step':32,'target_step':96,'initial_checkpoint':SOURCE+'/train/checkpoint-32',
          'initial_manifest':SOURCE+'/train/checkpoint-32/checkpoint-manifest.json',
          'initial_manifest_sha256':hashes['train/checkpoint-32/checkpoint-manifest.json'],
          'initial_optimizer_sha256':hashes['train/checkpoint-32/optimizer.pt'],
          'initial_learning_rate':lr,
          'learning_rate_policy':'Cosine decay from the saved step-32 learning rate over 64 additional updates; no LR jump or new warmup.',
          'source_prefix_sha256':hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest(),
          'restart_interval':2,'keep_restart_states':1,
          'state_directory':spec['archive_root']+'/restart-state',
          'validated_memory_preflight':SOURCE+'/smoke/complete.json',
          'validated_memory_preflight_sha256':hashes['smoke/complete.json']}
    spec['continuation']=plan
    steps,schedule,saves,digest=training_plan(rows,spec)
    assert steps==96 and saves==[48,64,80,96]
    spec['expected_schedule_sha256']=digest
    validate_reuse(spec,original);validate_continuation(spec,schedule)
    identity={'continuation':plan,'schedule':digest,'prepared_dataset':spec['prepared_dataset'],
              'seed':spec['seed'],'context':spec['context'],'effective_batch':spec['effective_batch'],
              'base_revision':spec['base_revision']}
    plan['lineage_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return spec,schedule


def prepare():
    source=BUILD_ROOT/'efe-training-pilot-20261027-v5'
    original=read(source/'spec.json');dataset=ROOT/'training-pilot/frozen-dataset-manifest.json'
    if hashlib.sha256(dataset.read_bytes()).hexdigest()!=original['prepared_dataset']['manifest_sha256']:
        raise ValueError('Frozen continuation dataset changed')
    hashes=read(AUDIT/'source-hashes.json')
    if not hashes:raise ValueError('Source optimizer and memory-preflight checksums are required before freezing')
    spec,schedule=make_spec(original,read(dataset)['examples'],read(AUDIT/'source/train/schedule.json'),hashes)
    manifest=AUDIT/'source/train/checkpoint-32/checkpoint-manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=spec['continuation']['initial_manifest_sha256']:
        raise ValueError('Local and remote checkpoint manifests differ')
    build=BUILD_ROOT/STUDY;build.mkdir(exist_ok=True)
    if (build/'source-manifest.json').exists():
        if read(build/'spec.json')!=spec:raise ValueError('Frozen continuation specification changed')
    else:
        write(build/'spec.json',spec)
        for src,target in [('efe_training_resume.py','runner.py'),('efe_training_pilot.py','pilot_helpers.py'),
                ('efe_training_data.py','data.py'),('efe_train_vl.py','trainer.py'),
                ('efe_budget_gateway.py','gateway.py'),('text_capability_eval.py','cuda_helpers.py'),
                ('efe_training_state.py','training_state.py')]:
            shutil.copyfile(BASE/'cluster'/src,build/target)
        shutil.copyfile(source/'bootstrap.sh',build/'bootstrap.sh')
        workflow=yaml.safe_load((source/'workflow.yaml').read_text())
        workflow['workflow']['name']='leact-efe-warm-continue32-96'
        workflow['workflow']['tasks'][0]['files'].append({'localpath':'training_state.py','path':'/tmp/efe_training_state.py'})
        (build/'workflow.yaml').write_text(yaml.safe_dump(workflow,sort_keys=False));freeze_sources(build)
    arm={'id':'qwen-warm-continue32-96','name':'Warmstart continuation · step 32 → 96',
         'stage':spec['stage'],'baseline':'qwen-warm-bb-1e4-p3-step32-native-v2-retry2',
         'build':str(build),'original_build':str(build),'attempt':1,'previous_attempts':[],
         'local_results_directory':spec['local_results_directory'],
         'original_local_results_directory':spec['local_results_directory'],
         'evaluation_directory':'training-pilot/warm-continue32-96-evaluation',
         'optimizer_steps':96,'origin_step':32,'additional_updates':64,'effective_batch':32,
         'lr':spec['lr'],'snapshot_steps':[48,64,80,96],
         'example_presentations':len(schedule),'distinct_decision_states':len({r['state_id'] for r in schedule}),
         'role':'Continue saved model and Adam state; native robot task performance is the primary outcome.'}
    plan={'study_id':STUDY,'created_utc':datetime.now(timezone.utc).isoformat(),'arms':[arm],
          'max_concurrent_training_nodes':1,'priority':'LOW','pool':'groot-h100-01',
          'execution_paused':True,'pause_reason':'Shared checkpoint volume full; awaiting explicit approval for the five-file optimizer cleanup proposal.',
          'capability_screen_enabled':False,'initialization':spec['scope'],
          'retry_policy':{'enabled':True,'backoff_seconds':300,'max_backoff_seconds':900}}
    path=ROOT/'training-pilot/continuation-recovery.json'
    if not path.exists():write(path,plan)
    dest=ROOT/spec['local_results_directory'];dest.mkdir(parents=True,exist_ok=True)
    if not (dest/'scheduler.json').exists():
        write(dest/'scheduler.json',{'status':'HELD','reason':plan['pause_reason']})
        write(dest/'progress.json',{'phase':'waiting_for_storage','detail':plan['pause_reason']})
    print(json.dumps({'build':str(build),'plan':str(path),'optimizer_updates':96,'additional_updates':64,
                      'capability_screen_enabled':False,'submitted':(build/'submission.json').exists()}))
    return build


if __name__=='__main__':prepare()
