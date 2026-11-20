"""One tracked Dojo infrastructure-only pass; existing native outcomes never retry."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import time

try:
    from .efe_astra_wide_collect import model_profile, run_case
    from .efe_pear_file_rpc import atomic_json
except ImportError:
    from efe_astra_wide_collect import model_profile, run_case
    from efe_pear_file_rpc import atomic_json


VERSION = 'astra62-dojo-infrastructure-retry-v1'
OWNER = 'rp_jhh_astra_wide_warmstart_20261119_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):return json.loads(Path(path).read_text())


def validate_spec(spec):
    profile=model_profile(spec)
    root=Path(spec['teacher_root'])
    if (profile['id']!='astra62-xhigh-v1' or profile['model']!='openai/openai/gpt-6.2-astra'
        or profile['effort']!='xhigh' or spec.get('suite')!='robodojo'
        or spec.get('owner_model_workflow')!=OWNER or not root.is_absolute()
        or root.name!=OWNER+'-astra62-xhigh-v1'
        or Path(spec['transport_root'])!=root.with_name(OWNER)/'transport'
        or spec.get('control_interface')
        or spec.get('max_calls',96)!=96 or spec.get('wall_seconds',1500)!=1500
        or spec.get('token_budget',160000)!=160000
        or spec.get('max_infrastructure_attempts',3)!=3
        or spec.get('teacher_plan',{}).get('partition')!='train'):
        raise ValueError('Retry pass requires the unchanged exact-profile training Dojo recipe')


def history(folder):
    """Any graded/finalized or ambiguous outcome is an exclusion, not a retry."""
    records=[];inflight=False
    for attempt in sorted(Path(folder).glob('infra-*')):
        if not attempt.is_dir():continue
        try:record=load(attempt/'result.json')
        except (OSError,ValueError):return records,'unreadable_existing_attempt'
        if not isinstance(record,dict):return records,'malformed_existing_attempt'
        if record.get('status')in{'graded_success','graded_failure'}or record.get('finalized_native_outcome')is True:
            return records,'prior_native_outcome'
        if (attempt/'native-verdict.json').exists():return records,'native_verdict_requires_review'
        if record.get('status')in{'opening','running'}:inflight=True
        records.append((attempt,record))
    return records,'inflight_attempt'if inflight else None


def prepare(spec):
    validate_spec(spec);selected=[];excluded=[];root=Path(spec['teacher_root'])
    names=set(spec['teacher_task_names'])
    cases=[case for case in spec['teacher_plan']['cases']if case['task']in names and case['suite']=='robodojo']
    for case in cases:
        if case.get('partition')!='train'or case.get('admitted')is not True:
            raise ValueError('A retry cannot admit an evaluation or unqualified case')
        folder=root/'episodes'/case['case_id'];records,reason=history(folder)
        indexed={p.name:(p,r)for p,r in records}
        if reason is None and any(re.fullmatch(r'infra-\d+',p.name)and int(p.name[6:])>=3 for p,r in records):
            reason='separate_retry_pass_already_started'
        original=[indexed.get(f'infra-{i:02d}')for i in range(3)]
        if reason is None and any(pair is None for pair in original):reason='original_pass_not_exhausted'
        if reason is None:
            for index,(path,record)in enumerate(original):
                if (record.get('status')!='infrastructure_failure' or record.get('case')!=case
                    or record.get('infrastructure_attempt')!=index
                    or record.get('requested_model')!='openai/openai/gpt-6.2-astra'
                    or record.get('requested_effort')!='xhigh'
                    or record.get('curriculum_sha256')!=spec['teacher_plan']['curriculum_sha256']):
                    reason='prior_attempt_identity_or_status_mismatch';break
        if reason:
            excluded.append({'case_id':case['case_id'],'reason':reason});continue
        preserved={str(path.relative_to(root)):sha(path)for attempt,_ in original
                   for name in ['result.json','turns.jsonl','system-prompt.txt','action-contract.json']
                   if(path:=attempt/name).is_file()}
        selected.append({'case':case,'new_infrastructure_attempts':[3,4,5],
            'preserved_files':preserved,'prior_errors':[r.get('error')for _,r in original]})
    identity={'version':VERSION,'teacher_root':str(root),'teacher_profile':model_profile(spec),
        'curriculum_sha256':spec['teacher_plan']['curriculum_sha256'],
        'cases':[r['case']['case_id']for r in selected],'new_attempts_per_case':3}
    identifier='rp_jhh_astra62_dojo_infra_retry_'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    return {'version':VERSION,'id':identifier,'identity':identity,'selected':selected,'excluded':excluded,
        'created_utc':datetime.now(timezone.utc).isoformat(),'training_recipe_changed':False,
        'model_calls_started':0,'requires_verified_native_service':True,'original_records_modified':False}


def verify_health(spec,health):
    validate_spec(spec)
    if (health.get('teacher_profile_id')!='astra62-xhigh-v1' or health.get('suite')!='robodojo'
        or health.get('source_payload_verified')is not True or health.get('native_reset_open_verified')is not True
        or health.get('same_native_action_contract')is not True
        or not re.fullmatch('[0-9a-f]{64}',str(health.get('action_contract_sha256','')))):
        raise RuntimeError('Native service contract/reset health is not verified; no model call allowed')
    worker=health.get('rpc_worker_id','')
    if not re.fullmatch(r'rp-jhh-astra-wide-[A-Za-z0-9_.-]+',worker):
        raise ValueError('Native service is outside the separately owned teacher allocation')
    heartbeat=load(Path(spec['transport_root'])/'workers'/(worker+'.json'))
    updated=heartbeat.get('updated_unix')
    if (heartbeat.get('worker_id')!=worker or type(updated)not in(int,float)or not math.isfinite(updated)
        or not -30<=time.time()-updated<=120
        or heartbeat.get('phase')not in{'ready','native_episode'}):
        raise RuntimeError('Native RPC heartbeat is not fresh; no model call allowed')


def run(spec,manifest,health,output):
    validate_spec(spec);output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=True)
    if(manifest.get('version')!=VERSION or manifest.get('identity',{}).get('teacher_root')!=spec['teacher_root']
       or manifest['identity'].get('curriculum_sha256')!=spec['teacher_plan']['curriculum_sha256']
       or manifest['identity'].get('teacher_profile')!=model_profile(spec)):
        raise ValueError('Retry manifest does not match the unchanged teacher profile/plan')
    identity=manifest['identity']
    expected='rp_jhh_astra62_dojo_infra_retry_'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    if (manifest.get('id')!=expected or identity.get('new_attempts_per_case')!=3
        or identity.get('cases')!=[row['case']['case_id']for row in manifest['selected']]):
        raise ValueError('Retry manifest identity changed')
    with(output/'run.lock').open('a')as cohort_lock:
        fcntl.flock(cohort_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        events=[]
        for row in manifest['selected']:
            case=row['case'];folder=Path(spec['teacher_root'])/'episodes'/case['case_id']
            if row['new_infrastructure_attempts']!=[3,4,5]:raise ValueError('Unexpected infrastructure retry allowance')
            if (case not in spec['teacher_plan']['cases']or case['task']not in spec['teacher_task_names']
                or case.get('suite')!='robodojo'or case.get('partition')!='train'or case.get('admitted')is not True):
                raise ValueError('Retry case left the exact admitted training taskset')
            required={str((folder/f'infra-{i:02d}/result.json').relative_to(Path(spec['teacher_root'])))for i in range(3)}
            if not required<=set(row['preserved_files']):raise ValueError('Missing original exhausted-pass evidence')
            for name,digest in row['preserved_files'].items():
                if sha(Path(spec['teacher_root'])/name)!=digest:raise ValueError('Original trace changed; review before retry')
            with(folder/'writer.lock').open('a')as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    events.append({'case_id':case['case_id'],'status':'skipped_active_case_writer'});continue
                for number in row['new_infrastructure_attempts']:
                    _,reason=history(folder)
                    if reason and reason not in{'inflight_attempt'}:
                        events.append({'case_id':case['case_id'],'status':'not_retried','reason':reason});break
                    # An interrupted *new* retry can resume only with the case
                    # writer lock. Never turn an original pass into a new ID.
                    active_original=any((p/'result.json').is_file()and load(p/'result.json').get('status')in{'opening','running'}
                        for p in folder.glob('infra-0[012]'))
                    if active_original:raise RuntimeError('Original collector has an inflight attempt')
                    verify_health(spec,health)
                    result=run_case(spec,case,infrastructure_attempt=number)
                    event={'case_id':case['case_id'],'infrastructure_attempt':number,
                        'episode_id':result.get('episode_id'),'status':result['status'],
                        'native_success':result.get('native_success')}
                    events.append(event);atomic_json(output/'status.json',{'id':manifest['id'],
                        'updated_unix':time.time(),'events':events,'original_records_modified':False})
                    if result['status']!='infrastructure_failure':break
                    time.sleep(10)  # Preserve the original collector's infrastructure backoff.
        final={'id':manifest['id'],'completed_utc':datetime.now(timezone.utc).isoformat(),
               'events':events,'original_records_modified':False}
        atomic_json(output/'complete.json',final);return final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--health',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--manifest-sha256')
    args=parser.parse_args();spec=load(args.spec)
    if args.health is None:
        if args.manifest.exists():raise ValueError('Preserve the existing immutable retry manifest')
        manifest=prepare(spec);atomic_json(args.manifest,manifest)
        print(json.dumps({'id':manifest['id'],'selected_cases':len(manifest['selected']),'model_calls':0}));return
    if args.output is None:parser.error('--output is required for the single bounded execution pass')
    if not args.manifest_sha256 or sha(args.manifest)!=args.manifest_sha256:
        raise ValueError('Persisted retry manifest checksum differs from the execution receipt')
    run(spec,load(args.manifest),load(args.health),args.output)


if __name__=='__main__':main()
