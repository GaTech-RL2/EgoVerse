"""Collect hundreds of strict-harness teacher episodes, then gate warmstart/LeAct."""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zlib

import yaml

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write, submit, command
from .compute_budget import reserved_nodes, node_limit
from .launch_efe_training import freeze_sources
from .monitor_efe_preflight import remote

DEST = ROOT / 'fixed-action-v1/coverage-scale-v1'
DURABLE = '/mnt/amlfs-01/home/rpunamiya/leact/studies/efe-coverage-20261105-v1'
TEMPLATE = BUILD_ROOT / 'efe-fixed-action-20261103-v3-astra-task'
LANE = '/tmp/efe-training-evaluation-v1/warm96'
PREFIX = 'leact-efe-coverage-v1-'
SUITES = ('metaworldplus', 'robocasa', 'embodied-safety-exam', 'dexhand')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_plan():
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / 'plan.json').exists():
        return read(DEST / 'plan.json')
    split = read(ROOT / 'generalization-v2/splits.json')
    inventory = read(ROOT / 'generalization-v2/inventory.json')
    known_collision = 'metaworldplus-pick-place-i00-standard'
    train = set(split['train_candidates'])
    protected = set(split['development'] + split['official_core_evaluation'] + split['simulator_holdout_reserve'])
    tasks = [dict(r, partition='train_candidates') for r in inventory if r['task'] in train and
             r['suite'] in SUITES and r['mode']=='standard' and r['task'] != known_collision]
    holdout = [dict(r, partition='protected') for r in inventory if r['task'] in protected and r['suite'] in SUITES]
    assert not {r['task'] for r in tasks} & protected
    assert not {r['simulator_runtime'] for r in tasks} & set(split['held_out_runtimes'])
    # Include known solvable controls early, followed by a frozen randomized order.
    controls = ['metaworldplus-handle-press-i00-standard', 'metaworldplus-push-i00-standard',
                'robocasa-close-drawer-i00-standard', 'robocasa-coffee-press-button-i00-standard']
    tasks.sort(key=lambda r:(SUITES.index(r['suite']), controls.index(r['task']) if r['task'] in controls else 999,
                            hashlib.sha256(('efe-coverage-v1/'+r['task']).encode()).hexdigest()))
    plan = {'created_utc':now(), 'version':'coverage-scale-v1', 'upstream_revision':read(TEMPLATE/'spec.json')['revision'],
        'target_successful_episodes':256, 'minimum_distinct_successful_tasks':24,
        'maximum_attempts_per_task':12, 'stop_task_after_zero_success_attempts':3,
        'maximum_teacher_nodes':3, 'maximum_total_nodes':node_limit(), 'maximum_nodes_per_job':2,
        'initial_batch_size':4, 'collection_priority':'LOW', 'tasks':tasks, 'protected_tasks':holdout,
        'candidate_tasks':len(tasks), 'suite_counts':dict(Counter(r['suite'] for r in tasks)),
        'maximum_policy_attempts':12*len(tasks), 'known_scene_collision_excluded':[known_collision],
        'split_sha256':sha(ROOT/'generalization-v2/splits.json'), 'durable_root':DURABLE,
        'coverage_policy':'Try every candidate before repeated attempts. Repeats are independent sampled trajectories on the allowed frozen scene, not new scene variants. Prefer underrepresented successful tasks. Stop a task after three graded failures with no success; preserve every outcome.',
        'admission':'Native success, complete nontruncated responses, exact executed actions and scene isolation against every protected candidate in these four suites. No success-only teacher performance estimate.',
        'warmstart':{'initialization':'Qwen/Qwen3.6-35B-A3B base, not the three-task adapter',
            'base_revision':'995ad96eacd98c81ed38be0c5b274b04031597b0', 'balanced_passes':3,
            'effective_batch':32, 'lr':1e-4, 'lora_rank':32, 'lora_alpha':64,
            'reasoning':'Astra public backward explanations of verified expert commands; paired forward and backward objectives.',
            'steps':'ceil(2 * admitted decision states * 3 / 32); exact schedule frozen after corpus admission',
            'durable_restart_interval_updates':4},
        'leact':{'rounds':1, 'candidate_count':8, 'positive_candidates_per_state':2,
            'candidate_tokens':4096, 'max_balanced_states':2048, 'balanced_passes':1,
            'warmstart_replay_fraction':0.5, 'lr':1e-4, 'effective_batch':32,
            'admission':'Positive expert physical-command token-log-likelihood improvement over forced-empty reasoning, current validated EFE variant. Not the paper robotics decoded-action-match filter.',
            'continue_rounds':'Use native closed-loop outcomes and failure analysis; no automatic blind extra rounds.'},
        'evaluation':'Same constrained harness and native budgets for base, expanded warmstart and expanded warmstart+LeAct. Report trained scenes, frozen same-simulator development tasks, and untouched simulator holdouts separately. Recompute cohorts from admitted corpus.',
        'hypothesis':'Coverage may contribute to poor closed-loop behavior. The prior 3-episode/43-state experiment does not establish it as the sole cause.',
        'training_gate':'256 verified successful episodes covering >=24 distinct tasks; otherwise report the actual shortfall without silently lowering the gate.'}
    write(DEST/'plan.json', plan)
    write(DEST/'queue.json', {'created_utc':now(), 'batches':[], 'state':'collecting'})
    return plan


def harden_gateway_shutdown(source):
    """Bound cleanup so a finished native episode can still be recorded."""
    if 'gateway_shutdown_forced = False' in source:
        return source
    old = '            gateway.terminate()\n            gateway.wait(timeout=20)\n            gateway = None\n'
    replacement = '''            gateway_shutdown_forced = False
            gateway.terminate()
            try:
                gateway.wait(timeout=20)
            except subprocess.TimeoutExpired:
                gateway_shutdown_forced = True
                gateway.kill()
                gateway.wait(timeout=10)
            gateway = None
'''
    field = "                      'resource_metrics': rewards, 'returncode': receipt['returncode'],"
    if source.count(old)!=1 or source.count(field)!=1:
        raise ValueError('Native gateway cleanup or result recording changed')
    return source.replace(old,replacement).replace(field,
        "                      'gateway_shutdown_forced': gateway_shutdown_forced,\n"+field)


def build_batch(batch, retry=0):
    ident = batch['id'] + (f'-r{retry}' if retry else '')
    build = BUILD_ROOT / ('efe-coverage-20261105-v1-'+ident)
    if (build/'source-manifest.json').exists():
        return build
    build.mkdir(exist_ok=True)
    for name in read(TEMPLATE/'source-manifest.json'):
        shutil.copyfile(TEMPLATE/name, build/name)
    spec = read(build/'spec.json')
    for key in ['training_parent', 'instruction_suffix_sha256', 'retry_of', 'resume_completed_trials']:
        spec.pop(key, None)
    spec.update(tasks=batch['tasks'], model_id='astra-coverage-v1', agent_slug='astra-coverage',
        agent_label='Astra expanded constrained teacher', export_role='coverage_train_astra_v1',
        protocol_version='efe-constrained-action-v4-task', action_durable_root=DURABLE+'/collection',
        coverage_root=DURABLE, coverage_plan_sha256=sha(DEST/'plan.json'),
        local_results_directory='fixed-action-v1/coverage-scale-v1/runs/'+ident,
        scope='Expanded train-only teacher collection. Repeated attempts are not held-out evaluations or new scenes.',
        training_admission='Training candidates only; native success, executed-action and protected-scene audits required.',
        purpose='Collect 256 usable successful episodes with at least 24 distinct tasks before scaled Qwen3.6 warmstart and LeAct.',
        retry_policy='Preserve every graded result. Retry only scheduler/preemption/control/init failures, up to 12 infrastructure attempts per batch. Harness/API failures require diagnosis.',
        coverage_batch={'id':batch['id'], 'infrastructure_retry':retry})
    spec['harness_extension']='Known non-vector ESE done auxiliary operation is omitted from motion actions; existing stop JSON and every native vector dimension/bound remain unchanged. Startup console and native spec retained.'
    spec['gateway_cleanup_policy']='Terminate for 20 seconds, then kill and wait for 10 seconds; continue native result recording after forced shutdown.'
    spec['fixed_action_config']['task_budget_note']='Same 96-call/1500-second controller limit; each task native deadline and allowed seed remain enforced.'
    write(build/'spec.json', spec)
    write(build/'action-config.json', spec['fixed_action_config'])
    shutil.copyfile(DEST/'plan.json', build/'coverage-plan.json')
    shutil.copyfile(BASE/'cluster/efe_coverage_teacher.py', build/'native-worker.py')
    for name in ['efe_action_agent.py', 'efe_action_loop.py', 'efe_action_artifacts.py']:
        shutil.copyfile(BASE/'cluster'/name, build/name)
    source = (BASE/'embodied_first_exam/scene_admission.py').read_text()
    source = source.replace('from .report import ROOT, collection, read, write',
                            "def read(path):\n    return json.loads(Path(path).read_text())")
    (build/'scene_identity.py').write_text(source)
    # Durable artifacts are written after every trial. No six-hour idle hold.
    pilot = (build/'native-pilot.py').read_text().replace('for _ in range(4320):', 'for _ in range(12):')
    pilot = pilot.replace("if k in {'exception_type','occurred_at'}", "if k in {'exception_type','exception_message','occurred_at'}")
    (build/'native-pilot.py').write_text(harden_gateway_shutdown(pilot))
    workflow = yaml.safe_load((build/'workflow.yaml').read_text())
    workflow['workflow']['name'] = PREFIX + ident
    workflow['workflow']['timeout'] = {'queue_timeout':'20m', 'exec_timeout':'12h'}
    files = workflow['workflow']['tasks'][0]['files']
    files += [{'localpath':name, 'path':LANE+'/'+name} for name in ['scene_identity.py', 'coverage-plan.json']]
    (build/'workflow.yaml').write_text(yaml.safe_dump(workflow, sort_keys=False))
    for path in build.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    freeze_sources(build)
    return build


def capture(host):
    old = read(DEST/'evidence.json', {'runs':{}})
    known = {name:hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()
             for name,row in old.get('runs',{}).items()}
    stages={stage:DURABLE+'/'+stage for stage in ['warmstart','leact-r1']}
    stages.update({arm['id']:arm['archive_root'] for arm in read(DEST/'branch-runs.json',{}).get('runs',[])})
    value = remote(host, """from pathlib import Path
import base64,hashlib,json,zlib
root=Path(ROOT_LITERAL)
known=KNOWN_LITERAL
def read(p):
 return json.loads(p.read_text()) if p.exists() else None
runs={}
for path in (root/'collection').glob('*/spec.json'):
 d=path.parent
 result=read(d/'results.json') or {}
 manifest=read(d/'retained-manifest.json') or {}
 runs[d.name]={'results':result,'complete':read(d/'complete.json'),'worker_complete':read(d/'worker-complete.json'),
  'progress':read(d/'progress.json'),
  'retained':[dict(task=e['task'],job=e['job'],candidate=e['candidate_pending_scene_and_native_event_audit'],
   sha256=e['retained_sha256'],file=e.get('durable_file'),exclusion_reasons=e['exclusion_reasons']) for e in manifest.get('episodes',[])]}
 # Check every durable result on each capture, but transfer only changed rows.
 # This keeps completed grades intact without repeatedly streaming megabytes.
 digest=hashlib.sha256(json.dumps(runs[d.name],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if known.get(d.name)==digest:del runs[d.name]
training={'restart-probe':read(root/'restart-probe/result.json') or {}}
evaluations={}
for d in (root/'evaluation').glob('*/*'):
 if not d.is_dir():continue
 evaluations[d.name]={'results':read(d/'results.json') or {},'complete':read(d/'complete.json') or {},
  'public_manifest':read(d/'public-episodes/manifest.json') or {},
  'progress':read(d/'progress.json') or {},'preflight':read(d/'student-preflight.json') or {},
  'worker_complete':read(d/'worker-complete.json') or {}}
for stage,stage_root in STAGES_LITERAL.items():
 d=Path(stage_root)
 data={'progress':read(d/'progress.json') or {},'complete':read(d/'complete.json') or {},
       'waiting_for_corpus':read(d/'waiting-for-corpus.json') or {},'error':read(d/'worker-error.json') or {},
       'ready':read(d/('warmstart-ready.json' if stage=='warmstart' else 'leact-ready.json')) or {},
       'selection':read(d/'selection.json') or {},'membership':read(d/'membership.json') or {}}
 worker=read(d/'worker-start.json') or {}
 data['worker_start']=worker
 states=[read(p) for p in (d/'restart-states').glob('checkpoint-*/training-state.json')]
 data['committed_restart_step']=max((s.get('step',0) for s in states if s),default=0)
 data['full_model_replay']=read(d/'full-model-replay.json') or {}
 if data['error'].get('workflow') and data['error']['workflow']!=worker.get('workflow'):
  data['prior_error']=data['error'];data['error']={}
 data['train_complete']=read(d/'train/complete.json') or {}
 path=d/'dataset-manifest.json'
 if path.exists():
  raw=path.read_bytes();dataset=json.loads(raw)
  data['dataset']={k:v for k,v in dataset.items() if k!='examples'}
  data['dataset']['sha256']=hashlib.sha256(raw).hexdigest()
 metrics=d/'train/metrics.jsonl'
 if metrics.exists():data['metrics']=[json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
 training[stage]=data
payload={'runs':runs,'scene_audit':read(root/'scene-audit/result.json'),
 'scene_audit_progress':read(root/'scene-audit/progress.json'),'scene_audit_incomplete':read(root/'scene-audit/incomplete.json'),
 'training':training,'evaluations':evaluations,'base_ready':read(root/'base-ready.json'),'admission':read(root/'corpus/admission.json'),
 'admission_shortfall':read(root/'corpus/admission-shortfall.json')}
raw=json.dumps(payload,separators=(',',':')).encode()
print('EFE_CAPTURE '+json.dumps({'capture_encoding':'zlib-base64','sha256':hashlib.sha256(raw).hexdigest(),
 'payload':base64.b64encode(zlib.compress(raw)).decode()}),flush=True)
""".replace('ROOT_LITERAL', repr(DURABLE)).replace('KNOWN_LITERAL',repr(known)).replace('STAGES_LITERAL',repr(stages)))
    if value.get('capture_encoding')=='zlib-base64':
        raw=zlib.decompress(base64.b64decode(value['payload']))
        if hashlib.sha256(raw).hexdigest()!=value['sha256']:raise ValueError('Capture transport checksum mismatch')
        value=json.loads(raw)
    old['runs'].update(value.pop('runs'))
    for key, item in value.items():
        if item is not None:
            old[key] = item
    old['captured_utc'] = now()
    write(DEST/'evidence.json', old)
    for stage,data in old.get('training',{}).items():
        if stage=='restart-probe':continue
        destination=DEST/stage;destination.mkdir(exist_ok=True)
        if data.get('metrics'):
            (destination/'train').mkdir(exist_ok=True)
            (destination/'train/metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data['metrics']))
        if data.get('train_complete'):write(destination/'train/complete.json',data['train_complete'])
        for key in ['progress','complete','dataset','ready','selection','membership']:
            if data.get(key):write(destination/(key+'.json'),data[key])
    return old


def outcomes(queue, evidence):
    result = {}
    for batch in queue['batches']:
        by_task = {r['task']:r for r in batch['tasks']}
        for attempt in batch.get('attempts', []):
            data = evidence.get('runs', {}).get(attempt.get('workflow'), {})
            retained = {r['job']:r for r in data.get('retained', [])}
            for trial in data.get('results', {}).get('trials', []):
                if trial['status'] != 'graded':
                    continue
                key = (trial['task'], by_task[trial['task']]['collection_attempt'])
                if key in result:
                    raise ValueError('A graded policy attempt was repeated by an infrastructure retry')
                result[key] = {**trial, 'workflow':attempt['workflow'], 'retained':retained.get(trial['job'], {})}
    return result


def next_batch(plan, queue, graded, excluded):
    allocated = Counter()
    successes = Counter()
    failures = Counter()
    for batch in queue['batches']:
        for row in batch['tasks']:
            allocated[row['task']] = max(allocated[row['task']], row['collection_attempt'])
    for (task, _), row in graded.items():
        (successes if row['score']==1 else failures)[task] += 1
    candidates = [r for r in plan['tasks'] if r['task'] not in excluded and
        allocated[r['task']] < plan['maximum_attempts_per_task'] and
        not (failures[r['task']] >= plan['stop_task_after_zero_success_attempts'] and successes[r['task']]==0)]
    if not candidates:
        return None
    # Every task's first attempt precedes repeats. Within the same round, prefer
    # tasks with fewer successful trajectories, keeping the frozen order as tie-break.
    suite_allocations = Counter()
    for row in plan['tasks']:
        suite_allocations[row['suite']] += allocated[row['task']]
    candidates.sort(key=lambda r:(allocated[r['task']], suite_allocations[r['suite']], successes[r['task']]))
    first = candidates[0]
    same = [r for r in candidates if r['suite']==first['suite'] and allocated[r['task']]==allocated[first['task']]]
    chosen = same[:plan['initial_batch_size']]
    return {'id':f"b{len(queue['batches'])+1:03d}", 'created_utc':now(),
            'tasks':[dict(r, collection_attempt=allocated[r['task']]+1) for r in chosen], 'attempts':[]}


def infra_retryable(scheduler):
    from .efe_coverage_retry import reviewed_init_interruption
    status = scheduler.get('status','')
    text = json.dumps(scheduler).lower()
    return status in {'FAILED_QUEUE_TIMEOUT','FAILED_START_ERROR','FAILED_START_TIMEOUT','FAILED_PREEMPTED'} or (
        status.startswith('FAILED') and any(x in text for x in
        ['groot-idle-job-shutdown', 'gpu quota enforcement', 'osmocontrol', 'osmo control', 'osmo-init', 'preempt'])) or reviewed_init_interruption(scheduler)


def tick():
    plan = prepare_plan()
    queue = read(DEST/'queue.json')
    previous_evaluation=read(DEST/'evaluation-status.json',{})
    if previous_evaluation.get('jobs') and all(j['status'] in {'COLLECTED','NO_ADMISSIONS_USE_WARMSTART'} for j in previous_evaluation['jobs']):
        status=read(DEST/'status.json',{})
        status.update(phase='completed',updated_utc=now(),summary='Expanded corpus, warmstart, LeAct and the frozen matched native evaluation panel are complete. Inspect cohort-specific results and limitations.')
        write(DEST/'status.json',status)
        from ..embodied_first_exam import report
        report.publish()
        return status
    evidence = read(DEST/'evidence.json', {'runs':{}})
    active = command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
    # Final receipts outlive worker allocations. When every allocation has
    # ended, use one bounded reader rather than waiting for a new policy run.
    if not active and queue['batches']:
        from .launch_efe_training import recovery_reader
        all_attempts=[a for b in queue['batches'] for a in b.get('attempts',[])]
        all_attempts += [a for s in read(DEST/'training-queue.json',{}).get('stages',{}).values() for a in s.get('attempts',[])]
        all_attempts += [a for b in read(DEST/'evaluation-queue.json',{}).get('batches',[]) for a in b.get('attempts',[])]
        ends=[s['end_time'] for a in all_attempts if (s:=read(Path(a['build'])/'scheduler.json',{})).get('end_time')]
        captured=datetime.fromisoformat(evidence['captured_utc']).timestamp() if evidence.get('captured_utc') else 0
        recovery_needed=bool(ends) and max(datetime.fromisoformat(x).replace(tzinfo=timezone.utc).timestamp() for x in ends)>captured
        candidates=[Path(a['build']) for b in queue['batches'] for a in b.get('attempts',[])
                    if not (Path(a['build'])/'completion-reader/submission.json').exists()]
        if candidates and recovery_needed:
            recovery_reader(candidates[-1])
            active = command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
    # PEAR qualification uses a different task name and has its own observer.
    hosts = [r['name'] for r in active if r['status']=='RUNNING'
             and not r['name'].startswith('leact-efe-pear-')]
    fresh_capture = False
    if hosts:
        try:
            evidence = capture(next((h for h in hosts if h.startswith(PREFIX)), hosts[0]))
            fresh_capture = True
        except Exception as error:
            write(DEST/'capture-error.json', {'utc':now(), 'error':str(error)[-1500:]})
    graded = outcomes(queue, evidence)
    audit = evidence.get('scene_audit', {})
    harness_scope = read(DEST/'harness-scope.json', {})
    deferred_reasons = {suite:audit.get('exclusion_reason','Scene identity audit incomplete')
                        for suite in audit.get('excluded_suites',[])}
    deferred_reasons.update(harness_scope.get('deferred_suites', {}))
    excluded = {r['task'] for r in audit.get('collisions', [])}
    excluded |= {r['task'] for r in plan['tasks'] if r['suite'] in deferred_reasons}
    failed_admission={(r['workflow'],r['task']) for r in evidence.get('admission_shortfall',{}).get('excluded',[])}
    eligible = [r for r in graded.values() if r['score']==1 and r['retained'].get('candidate') and r['task'] not in excluded
                and (r['workflow'],r['task']) not in failed_admission]
    task_count = len({r['task'] for r in eligible})
    collection_gate = len(eligible)>=plan['target_successful_episodes'] and task_count>=plan['minimum_distinct_successful_tasks'] and audit.get('complete')
    states = []
    for batch in queue['batches']:
        if not batch['attempts']:
            continue
        attempt = batch['attempts'][-1]
        reviewed=read(DEST/'reviewed-infrastructure-retries.json',{}).get(batch['id'],{})
        reviewed_once=reviewed.get('source_workflow')==attempt['workflow']
        if reviewed_once:
            batch.pop('quarantined_tasks',None)
            if reviewed.get('unrecoverable_tasks'):
                batch['unrecoverable_tasks']=reviewed['unrecoverable_tasks']
        state = (read(Path(attempt['build'])/'scheduler.json', {}) if batch.get('status')=='COLLECTED' else {})
        state = state or command(['osmo','workflow','query',attempt['workflow'],'--format-type','json'])
        attempt['status'] = state['status']
        write(Path(attempt['build'])/'scheduler.json', state)
        quarantined=set(batch.get('quarantined_tasks',[]))
        # A completed attempt with lost outcome evidence must not be rerolled
        # as an infrastructure retry. This exclusion applies only to this batch.
        unrecoverable=set(batch.get('unrecoverable_tasks',[]))
        tasks_left = [r for r in batch['tasks'] if (r['task'],r['collection_attempt']) not in graded and r['task'] not in excluded|quarantined|unrecoverable]
        batch['status'] = state['status'] if tasks_left else 'EXCLUDED' if all(r['task'] in excluded for r in batch['tasks']) else 'PARTIAL_EVIDENCE_UNAVAILABLE' if unrecoverable else 'PARTIAL_HARNESS_EXCLUDED' if quarantined else 'COLLECTED'
        if tasks_left and state['status'] not in {'RUNNING','PENDING','WAITING'}:
            # A fresh shared-storage capture is essential before retrying a
            # reclaimed job, otherwise a completed failure could be overwritten.
            ended = state.get('end_time')
            capture_after_end = fresh_capture and ended and datetime.fromisoformat(evidence['captured_utc']).timestamp() >= datetime.fromisoformat(ended).replace(tzinfo=timezone.utc).timestamp()
            if capture_after_end and infra_retryable(state) and len(batch['attempts'])<12:
                batch['retry_pending'] = tasks_left
            else:
                batch['status'] = 'NEEDS_REVIEW'
            native=evidence.get('runs',{}).get(attempt['workflow'],{})
            failures=[r for r in native.get('results',{}).get('trials',[]) if r['status']=='infrastructure_failure']
            transient=failures and all(any(x in (r.get('action_metrics',{}).get('error','')) for x in
                ['RemoteDisconnected','ConnectionResetError','ConnectionRefusedError','TimeoutError']) for r in failures)
            if capture_after_end and (transient or reviewed_once) and len(batch['attempts'])<12:
                batch['retry_pending']=tasks_left
            elif capture_after_end and failures and not infra_retryable(state):
                # An unsupported task must not consume repeated policy attempts.
                # Keep its failure, and advance untouched tasks in the batch.
                quarantined|={r['task'] for r in failures}
                batch['quarantined_tasks']=sorted(quarantined)
                remaining=[r for r in tasks_left if r['task'] not in quarantined]
                if remaining and len(batch['attempts'])<12:batch['retry_pending']=remaining
            if batch.get('retry_pending'):
                batch['status']='RETRY_QUEUED'
        states.append({'id':batch['id'], 'workflow':attempt['workflow'], 'status':batch['status'],
                       'tasks':len(batch['tasks']), 'graded':sum((r['task'],r['collection_attempt']) in graded for r in batch['tasks']),
                       'suite':batch['tasks'][0]['suite'],
                       'retry_count':len(batch['attempts'])-1})
    # Never create more work than the three-teacher / six-total allocation.
    own_active = sum(r['name'].startswith(PREFIX) for r in active)
    quarantined={t for b in queue['batches'] for t in b.get('quarantined_tasks',[])}
    allocation_policy=read(DEST/'allocation-policy.json', {})
    suite_limits=allocation_policy.get('maximum_concurrent_teacher_jobs_by_suite', {})
    active_names={r['name'] for r in active}
    active_suites=Counter(b['tasks'][0]['suite'] for b in queue['batches']
                         if b.get('attempts') and b['attempts'][-1]['workflow'] in active_names)
    def has_capacity(batch):
        suite=batch['tasks'][0]['suite']
        return active_suites[suite]<suite_limits.get(suite,plan['maximum_teacher_nodes'])
    # Permanent exclusions also apply to already queued infrastructure retries.
    # Running trials finish unchanged; allocation limits affect only new jobs.
    for batch in queue['batches']:
        if batch.get('retry_pending'):
            batch['retry_pending']=[r for r in batch['retry_pending'] if r['task'] not in excluded|quarantined|set(batch.get('unrecoverable_tasks',[]))]
    slots = max(0, min(plan['maximum_teacher_nodes']-own_active, node_limit()-reserved_nodes(active)))
    if not collection_gate:
        for _ in range(slots):
            retry = next((b for b in queue['batches'] if b.get('retry_pending') and has_capacity(b)), None)
            if retry:
                rows = retry.pop('retry_pending')
                build = build_batch({**retry, 'tasks':rows}, len(retry['attempts']))
                batch = retry
            else:
                batch = next((b for b in queue['batches'] if not b['attempts'] and has_capacity(b)
                              and all(r['task'] not in excluded|quarantined for r in b['tasks'])), None)
                at_limit={r['task'] for r in plan['tasks']
                          if active_suites[r['suite']]>=suite_limits.get(r['suite'],plan['maximum_teacher_nodes'])}
                batch = batch or next_batch(plan, queue, graded, excluded|quarantined|at_limit)
                if batch is None:
                    break
                if batch not in queue['batches']:
                    queue['batches'].append(batch)
                build = build_batch(batch)
            write(DEST/'queue.json', queue)
            receipt = submit(build, priority='LOW')
            attempt = {'workflow':receipt['name'], 'build':str(build), 'submitted_utc':now(), 'status':'PENDING'}
            batch['attempts'].append(attempt)
            batch['status'] = 'PENDING'
            active_suites[batch['tasks'][0]['suite']]+=1
            write(DEST/'queue.json', queue)
            states.append({'id':batch['id'], 'workflow':receipt['name'], 'status':'PENDING',
                           'tasks':len(batch['tasks']), 'suite':batch['tasks'][0]['suite'],
                           'graded':sum((r['task'],r['collection_attempt']) in graded for r in batch['tasks']),
                           'retry_count':len(batch['attempts'])-1})
    queue['state'] = 'admission_ready' if collection_gate else 'collecting'
    write(DEST/'queue.json', queue)
    from .launch_efe_coverage_train import tick as training_tick
    preparation_ready=(len(eligible)>=min(128,plan['target_successful_episodes'])
                       and task_count>=plan['minimum_distinct_successful_tasks'] and audit.get('complete'))
    training=training_tick(plan,evidence,collection_gate,active,preparation_ready=preparation_ready)
    from .launch_efe_coverage_eval import tick as eval_tick
    # Include allocations submitted by training_tick before assigning eval slots.
    active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
    eval_tick(plan,evidence,active)
    train_jobs={j['stage']:j for j in training['jobs']}
    if collection_gate:
        if train_jobs['leact-r1']['status']=='COMPLETED':queue['state']='native_evaluation'
        elif train_jobs['leact-r1']['status'] in {'RUNNING','PENDING','WAITING'}:queue['state']='leact'
        elif train_jobs['warmstart']['status'] in {'RUNNING','PENDING','WAITING'}:queue['state']='warmstart'
    write(DEST/'queue.json',queue)
    states=list({row['id']:row for row in states}.values())
    status = {'updated_utc':now(), 'phase':queue['state'], 'target_episodes':256, 'minimum_tasks':24,
        'candidate_tasks':plan['candidate_tasks'], 'suite_counts':plan['suite_counts'],
        'graded_episodes':len(graded), 'native_successes':sum(r['score'] for r in graded.values()),
        'candidate_successful_episodes':len(eligible), 'candidate_successful_tasks':task_count,
        'missing_outcome_episodes':[{'batch':b['id'],'task':r['task'],'collection_attempt':r['collection_attempt']}
                                    for b in queue['batches'] for r in b['tasks'] if r['task'] in b.get('unrecoverable_tasks',[])],
        'scene_audit_complete':bool(audit.get('complete')), 'scene_collisions_excluded':len(audit.get('collisions',[])),
        'eligible_tasks':plan['candidate_tasks']-len(excluded), 'deferred_suites':sorted(deferred_reasons),
        'deferred_suite_reasons':deferred_reasons,
        'allocation_policy':allocation_policy,
        'collection_by_suite':{suite:{'graded':sum(r['suite']==suite for r in graded.values()),
                                    'successes':sum(r['suite']==suite and r['score']==1 for r in graded.values())}
                               for suite in sorted({r['suite'] for r in graded.values()})},
        'harness_tasks_under_review':sorted(quarantined),
        'native_execution_admission_complete':bool(evidence.get('admission',{}).get('gate_passed')),
        'training_started':any(j['progress'].get('training_started') or evidence.get('training',{}).get(j['stage'],{}).get('metrics') for j in training['jobs']), 'jobs':states,
        'summary':f"Expanded Astra collection: {len(graded)} graded episodes, {len(eligible)}/256 candidate usable successes across {task_count}/24 required distinct tasks. Warmstart: {train_jobs['warmstart']['status']}; LeAct: {train_jobs['leact-r1']['status']}. Repeated trajectories are not new scenes."}
    write(DEST/'status.json', status)
    refresh_recovery_status(training,evidence)
    try:
        from ..embodied_first_exam import report
        report.publish()
    except Exception as error:
        write(DEST/'publish-error.json', {'utc':now(), 'error':str(error)[-1000:]})
    return status


def refresh_recovery_status(training,evidence):
    """Keep the recovery panel current after the manual recovery has finished."""
    path=DEST/'training-recovery-status.json'
    record=read(path,{})
    if not record.get('automatic_progress_updates'):return
    jobs={row['stage']:row for row in training.get('jobs',[])}
    warm=evidence.get('training',{}).get('warmstart',{})
    last=(warm.get('metrics') or [{}])[-1]
    job=jobs.get('warmstart',{})
    step=last.get('step',0);total=last.get('steps',record.get('planned_optimizer_updates',737))
    committed=warm.get('committed_restart_step',record.get('committed_restart_step',0))
    summary=f"Warmstart {job.get('status','UNKNOWN')}: {step}/{total} updates logged; latest committed restart {committed}."
    if last.get('forward_ce') is not None and last.get('backward_ce') is not None:
        summary+=f" Last forward/backward losses {last['forward_ce']:.4f}/{last['backward_ce']:.4f}."
    summary+=f" LeAct {jobs.get('leact-r1',{}).get('status','WAITING_FOR_WARMSTART')}."
    error=warm.get('error',{})
    if error.get('workflow')==job.get('workflow') and job.get('status')=='NEEDS_REVIEW':
        summary+=' Latest attempt stopped: '+str(error.get('error','Unknown worker error')).splitlines()[0][:240]+'.'
    evaluation=read(DEST/'evaluation-status.json',{})
    student=[r for r in evaluation.get('scores',[]) if r['model'] in {'warmstart','leact-r1'}]
    if student:
        summary+=' Native student results: '+'; '.join(
            f"{r['model']} {r['cohort']} {r['successes']}/{r['graded']}" for r in student)+'.'
    else:summary+=' No expanded student robot-success result is available yet.'
    record.update(utc=now(),workflow=job.get('workflow'),status=job.get('status'),
        last_recorded_update=step,committed_restart_step=committed,summary=summary,
        full_model_replay=warm.get('full_model_replay',{}),evidence_captured_utc=evidence.get('captured_utc'))
    replay=warm.get('full_model_replay',{})
    record.update(forward_ce=last.get('forward_ce'),backward_ce=last.get('backward_ce'),
        seconds_per_update=last.get('seconds'),full_model_replay_workflow=replay.get('workflow'),
        full_model_replay_passed=replay.get('passed'),
        full_model_replay_absolute_error=replay.get('absolute_error'))
    write(path,record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','tick','monitor'])
    args = parser.parse_args()
    plan = prepare_plan()
    if args.action=='prepare':
        print(json.dumps({k:plan[k] for k in ['candidate_tasks','suite_counts','maximum_policy_attempts','target_successful_episodes']}))
        return
    with (DEST/'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='monitor':
            from .launch_efe_coverage_train import PAYLOAD_VERSION
            launch=read(DEST/'monitor-launch.json',{})
            launch.update(pid=os.getpid(),utc=now(),controller_sha256=sha(Path(__file__)),
                          training_payload_version=PAYLOAD_VERSION,
                          supervisor=os.environ.get('EFE_MONITOR_SUPERVISOR','manual'))
            write(DEST/'monitor-launch.json',launch)
        while True:
            status={}
            try:
                status = tick()
                heartbeat={k:status.get(k) for k in ['updated_utc','phase','summary','training_started']}
                heartbeat.update(pid=os.getpid(),health='ok',free_bytes=shutil.disk_usage(DEST).free)
                write(DEST/'monitor-health.json',heartbeat)
                print(json.dumps(heartbeat), flush=True)
            except Exception as error:
                failure={'utc':now(),'pid':os.getpid(),'error':str(error)[-1800:]}
                # A full local disk must not kill the retry controller while it
                # is trying to record the original I/O error.
                try:write(DEST/'monitor-error.json',failure)
                except OSError:pass
                try:print(json.dumps(failure),file=sys.stderr,flush=True)
                except OSError:pass
                if args.action=='tick':
                    raise
            if args.action=='tick':
                return
            if status.get('phase')=='completed':return
            time.sleep(45)


if __name__=='__main__':
    main()
