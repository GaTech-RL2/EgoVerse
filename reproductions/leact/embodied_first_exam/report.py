"""Publish measured EFE progress beside the approved experiment comparison."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'artifacts/embodied-first-exam'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def schedule_plan(root):
    """Keep the frozen experiment plan and its interrupted attempts reviewable."""
    plan=read(root/'training-pilot/schedule-comparison.json',{})
    recovery=read(root/'training-pilot/schedule-recovery.json',{})
    if not recovery:return plan
    plan={**plan,'arms':[dict(a) for a in plan.get('arms',[])],'prior_attempts':[],'recovery':recovery,
          'recovery_status':read(root/'training-pilot/schedule-recovery-status.json',{})}
    retries={a['id']:a for a in recovery['arms']}
    for arm in plan['arms']:
        retry=retries.get(arm['id'])
        if not retry:continue
        plan['prior_attempts'].append({**arm,'id':arm['id']+'-interrupted-attempt-1',
                                      'name':arm['name']+' · interrupted attempt 1'})
        for prior in retry.get('previous_attempts',[]):
            plan['prior_attempts'].append({**arm,'id':arm['id']+'-interrupted-retry-'+str(prior['attempt']),
                'name':arm['name']+' · interrupted retry '+str(prior['attempt']),
                'build':prior['build'],'local_results_directory':prior['local_results_directory']})
        arm.update(build=retry['build'],local_results_directory=retry['local_results_directory'])
        arm['name']+=' · retry '+str(retry.get('attempt',1))
    return plan


def training_histories(root, comparison):
    """Include registered runs before their first update without inventing points."""
    metadata={
        'training-pilot/run-v5':{'id':'efe-warmstart-initial','name':'Initial warmstart · LR 1e-6 · 16 updates',
            'category':'efe-warmstart','phase':'Astra reasoning SFT','optimizer_steps':16},
        'training-pilot/leact-r1-run-v2':{'id':'efe-leact-r1','name':'LeAct round 1 · LR 1e-6 · 12 updates',
            'category':'efe-leact','phase':'LeAct with 50% warmstart replay','optimizer_steps':12},
        'fixed-action-v1/warmstart-lora-v1':{'id':'efe-qwen36-constrained-warm-lora',
            'name':'Qwen3.6 35B · attempt 1 · checkpoint serialization failed',
            'category':'efe-warmstart','phase':'43 actions · equal forward/backward SFT · language LoRA',
            'optimizer_steps':32},
        'fixed-action-v1/warmstart-lora-v2':{'id':'efe-qwen36-constrained-warm-lora-v2',
            'name':'Qwen3.6 35B · constrained warmstart · verified saver',
            'category':'efe-warmstart','phase':'Same 43 actions and 32-update recipe · checkpoint validation before training',
            'optimizer_steps':32}}
    for arm in comparison.get('arms',[])+comparison.get('prior_attempts',[]):
        metadata[arm['local_results_directory']]={**arm,'category':'efe-warmstart',
            'phase':'Step 32 to 96 continuation' if arm.get('origin_step') else 'Three-pass schedule comparison'}
    additional=read(root/'training-pilot/additional-runs.json',{}).get('runs',[])
    for arm in additional:
        metadata[arm['local_results_directory']]=arm
    directories={p.parent for p in (root/'training-pilot').glob('*/train') if (p/'metrics.jsonl').exists()}
    directories.update(root/arm['local_results_directory'] for arm in comparison.get('arms',[])+comparison.get('prior_attempts',[]))
    directories.update(root/arm['local_results_directory'] for arm in additional)
    if (root/'fixed-action-v1/warmstart-lora-v1/plan.json').exists():
        directories.add(root/'fixed-action-v1/warmstart-lora-v1')
    if (root/'fixed-action-v1/warmstart-lora-v2/plan.json').exists():
        directories.add(root/'fixed-action-v1/warmstart-lora-v2')
    result=[]
    for directory in sorted(directories):
        path=directory/'train/metrics.jsonl';history=[]
        for line in path.read_text().splitlines(keepends=True) if path.exists() else []:
            if not line.endswith('\n'):continue
            row=json.loads(line);count=row['forward_examples']+row['backward_examples']
            history.append({**row,'total_steps':row['steps'],'grad_norm':row['gradient_norm'],
                'learning_rate':row['lr'],'gpu_memory_gib':row['max_allocated_bytes_per_gpu']/1024**3,
                'loss':(row['forward_ce']*row['forward_examples']+row['backward_ce']*row['backward_examples'])/count})
        key=str(directory.relative_to(root));info=metadata.get(key,{})
        progress=read(directory/'progress.json',{});scheduler=read(directory/'scheduler.json',{})
        complete=bool(read(directory/'train/complete.json',{}).get('success'))
        phase=progress.get('phase');state=scheduler.get('status','Not started')
        detail=progress.get('detail')
        failure=next((g['failure_message'] for g in scheduler.get('groups',[]) if g.get('failure_message')),None)
        interruption=read(directory/'interruption.json',{})
        if complete:
            state='Training complete'
            if phase in {'export_recovery_queued','checkpoint_inspection','checkpoint_loss_replay','preparing_base','exporting_checkpoint'}:
                state='Training complete · export recovery'
            elif phase in {'export_recovery_failed','warmstart_failed'}:
                state='Training complete · export failed'
        elif state.startswith('FAILED'):
            state=('Canceled · GPU quota' if failure and 'GPU quota enforcement' in failure
                   else 'Preempted · higher priority' if interruption.get('kind')=='priority_preemption'
                   else 'Failed · GPU interconnect' if interruption.get('kind')=='gpu_interconnect_failure'
                   else 'Interrupted · OSMO initialization' if interruption.get('kind')=='osmo_initialization_failure'
                   else 'Interrupted · OSMO Control' if failure and 'OSMO Control failure' in failure
                   else 'Canceled' if state=='FAILED_CANCELED' else 'Failed')
            detail=(f'{state}. Last recorded update: {history[-1]["step"] if history else 0}/'
                    f'{info.get("optimizer_steps",history[-1]["steps"] if history else "unknown")}. '
                    'No completed training receipt; downstream evaluation has not been launched.')
        elif state=='COMPLETED':
            state='Awaiting training receipt'
        elif state=='HELD':state='Waiting for storage' if phase=='waiting_for_storage' else 'Held for review'
        elif phase in {'train','warmstart_training'}:state='Training'
        elif phase in {'optimizer_smoke','smoke'}:state='Optimizer smoke test'
        elif phase=='serialization_probe':state='Checkpoint saving preflight'
        elif phase=='checkpoint_loss_replay':state='Verifying saved checkpoint'
        elif phase=='leact_generation':state='Generating and scoring student explanations'
        elif phase=='leact_training':state='Training'
        elif phase=='no_admissions':state='No admitted candidates · no update'
        elif phase=='warmstart_failed':state='Failed; see run record'
        elif phase=='merging_checkpoint':state='Exporting checkpoint'
        elif phase=='checkpoint_saving' and read(directory/'restart-ready.json',{}).get('step',0)>=progress.get('step',1):
            state='Training';detail=f"Checkpoint {progress['step']}/{progress.get('steps','?')} committed; continuing the remaining updates."
        elif phase=='optimizer_memory_preflight':state='GPU preflight'
        elif phase=='needs_review':state='Failed; see run record'
        elif state in {'PENDING','WAITING'}:
            if any(g.get('status')=='INITIALIZING' for g in scheduler.get('groups',[])):
                state='Initializing H100 node';detail='An H100 node has been assigned; OSMO is initializing the task container.'
            else:
                if state=='PENDING':detail=f'Submitted to OSMO at {scheduler.get("priority","unspecified")} priority; waiting for a GPU allocation.'
                state='Queued'
        quality_hold=read(directory/'quality-hold.json',{})
        rejected_history=[]
        if quality_hold.get('active'):
            rejected=set(quality_hold.get('suspect_updates',[]))
            rejected_history=[row for row in history if row['step'] in rejected]
            history=[row for row in history if row['step'] not in rejected]
            state='Quality hold';complete=False
            detail=(quality_hold.get('reason') or quality_hold.get('summary') or
                'Training stopped for a numerical failure. Suspect updates are excluded from the valid training curve.')
        result.append({'id':info.get('id','efe-'+directory.name),'name':info.get('name','EFE '+directory.name),
            'category':info.get('category','efe-leact' if 'leact' in directory.name else 'efe-warmstart'),
            'phase':info.get('phase','EFE training'),'status':state,'detail':detail,
            'scheduler_status':scheduler.get('status'),'failure_reason':failure,
            'interruption':interruption,
            'priority':scheduler.get('priority'),
            'planned_steps':info.get('optimizer_steps',history[-1]['steps'] if history else None),
            'origin_step':info.get('origin_step'),
            'restart_checkpoint':read(directory/'restart-ready.json',{}),
            'workflow':scheduler.get('workflow',scheduler.get('name',progress.get('workflow'))),
            'source':'artifacts/embodied-first-exam/'+str(path.relative_to(root)),
            'history':history,'rejected_history':rejected_history,'quality_hold':quality_hold,'complete':complete})
    return result


def collection():
    workers=[]; attempts=[]
    totals={k:0 for k in ('requests','completed','http_200','http_errors','public_tool_calls','input_tokens','output_tokens','reasoning_tokens')}
    caps=set(); models=set()
    for path in [ROOT/'astra-pilot', *sorted((ROOT/'astra-workers').glob('*'))]:
        progress=read(path/'progress.json',{})
        results=read(path/'results.json',{})
        transport=read(path/'transport.json',{})
        scheduler=read(path/'scheduler.json',{})
        if not progress and not results:continue
        rows=results.get('trials',[])
        for row in rows:
            attempts.append({**row,'workflow':results.get('workflow',progress.get('workflow'))})
        graded=[r for r in rows if r['status']=='graded']
        workers.append({'workflow':progress.get('workflow',results.get('workflow')),
                        'scheduler_status':scheduler.get('status'), 'phase':progress.get('phase'),
                        'detail':progress.get('detail'), 'current_task':progress.get('current_task'),
                        'completed':len(rows), 'graded':len(graded),'successes':sum(r['score'] for r in graded),
                        'planned':progress.get('planned_trials',results.get('planned_trials')),
                        'model_requests':transport.get('completed',0), 'updated_utc':transport.get('updated_utc',progress.get('updated_utc'))})
        for k in totals:totals[k]+=transport.get(k,0)
        caps.update(transport.get('caps',[]));models.update(transport.get('returned_models',[]))
    totals.update(caps=sorted(caps),returned_models=sorted(models))
    trials={}
    for row in sorted(attempts,key=lambda r:r.get('completed_utc','')):
        previous=trials.get(row['task'])
        if previous:
            retry=row.get('retry_of') or {}
            if previous['status']!='infrastructure_failure' or any(retry.get(k)!=previous.get(k) for k in ('workflow','job')):
                raise ValueError('Unreviewed duplicate teacher task ownership: '+row['task'])
        trials[row['task']]=row
    return workers, {'trials':list(trials.values()),'attempts':attempts,
                     'planned_trials':read(ROOT/'parallel-collection-plan.json',{}).get('unique_planned_tasks',37)}, totals


def build():
    import importlib
    from . import capability_retention
    # Long-running collectors reload this report; refresh its dashboard helper too.
    capability_dashboard=importlib.reload(capability_retention).dashboard
    plan=read(ROOT/'study-plan.json'); status=read(ROOT/'status.json')
    recipe=read(ROOT/'training-recipe-v2.json'); split=read(ROOT/'generalization-v2/manifest.json')
    controls_progress=read(ROOT/'native-pilot/progress.json',{})
    progress=read(ROOT/'astra-pilot/progress.json',{}) or controls_progress
    diagnostics=read(ROOT/'native-pilot/results.json',{'trials':[]})
    admission=read(ROOT/'training-pilot/corpus-manifest.json',{})
    scored=read(ROOT/'evaluations/summary.json',{'arms':[]})
    scores={r['id']:r for r in scored['arms']}
    arms=[]
    for arm in plan['arms']:
        baseline=arm.get('parent')
        if arm['id']=='qwen-demo-backward': baseline='qwen-demo-actions'
        arms.append({'id':arm['id'],'name':arm['name'],'baseline':baseline,
                     'initial_checkpoint':arm.get('initial_checkpoint',arm.get('parent')),
                     'group':'demos' if arm['id'].startswith('qwen-demo-') else 'main',
                     'explanation':arm.get('training',arm.get('target','')),
                     'result':scores.get(arm['id'])})
    gates=[dict(x) for x in status['launch_gates']]
    nested=read(ROOT/'preflight/leact-efe-h100-runtime-preflight-2.json',{})
    gates.insert(5,{'id':'h100_nested_containers','status':'passed' if nested.get('nested_container_passed') else 'pending',
                    'detail':'H100 container smoke check. Actual Harbor task and verifier controls are separate gates.'})
    transport=read(ROOT/'preflight/astra-responses-image-tools-v1.json',{})
    if transport:
        gates.insert(4,{'id':'teacher_streaming_image_tools',
                       'status':'passed' if transport.get('tool_round_trip',{}).get('passed') else 'pending',
                       'detail':'Synthetic image recognition, streamed tool call and tool-result continuation via NVIDIA Responses API. This does not measure task success.'})
    routing=read(ROOT/'preflight/harbor-model-routing.json',{})
    workers, teacher_pilot, native_transport=collection()
    graded_teacher=[r for r in teacher_pilot.get('trials',[]) if r['status']=='graded']
    if routing:
        routed=bool(native_transport.get('http_200')) or bool(graded_teacher)
        gates.insert(5,{'id':'teacher_native_model_routing','status':'passed' if routed else 'pending',
                       'detail':'Native Astra model requests succeeded through the recorded model-routing fix; task success is graded separately.' if routed else 'The full Astra model ID works; the recorded Harbor routing fix is awaiting a successful native request.'})
    if native_transport:
        gates.insert(6,{'id':'native_generation_budget','status':'passed' if native_transport.get('http_200') and set(native_transport.get('caps',[]))<={16384,32768} else 'pending',
                        'detail':f'{native_transport.get("completed",0)} completed native requests; recorded caps {native_transport.get("caps",[])}. The output-cap adapter is a research-harness change.'})
    for gate in gates:
        if gate['id']=='compute_authorization':
            gate['detail']=read(ROOT/'execution-authorization.json',{}).get('compute',gate['detail'])
        if gate['id']=='native_harness_and_runtime' and controls_progress:
            gate['detail']=controls_progress.get('detail',controls_progress.get('phase','Simulator controls in progress'))
            gate['status']='passed' if diagnostics.get('all_controls_passed') else 'pending'
        if gate['id']=='resource_limit_enforcement' and graded_teacher:
            gate.update(status='passed',detail='Native teacher trial verified with resource_enforced=1 and a complete resource monitor.')
    cohorts=[
        {'name':'Same simulator, same task, new variant','count':split['planned_core_cohorts']['same_sim_same_task_new_variant'],
         'explanation':'Different frozen instance or audited perturbation of a trained semantic task, in the same mode. Scene audit and actual admitted exposure remain required.'},
        {'name':'Same simulator, new task','count':split['planned_core_cohorts']['same_sim_new_task'],
         'explanation':'The suite has training examples, while this semantic task is absent from training.'},
        {'name':'Held-out simulator runtime','count':split['planned_core_cohorts']['held_out_simulator'],
         'explanation':'35 SAPIEN and 6 Genesis core tasks. Both runtimes are excluded from train, development and native-demo augmentation. SAPIEN shares PhysX with Isaac Sim; only the 6 Genesis cases also hold out the physics engine.'}]
    now=datetime.now(timezone.utc).isoformat()
    training_note='EFE SFT has started.' if status.get('training_started') else 'EFE SFT has not started.'
    queue=read(ROOT/'launch-queue.json',{'builds':[]})
    launch_status=[{'build':Path(b).name,**read(Path(b)/'launcher-status.json',{})}
                  for b in queue['builds'] if not (Path(b)/'submission.json').exists()]
    service_error=any('503' in r.get('error','') for r in launch_status)
    live_workers=[w for w in workers if w.get('scheduler_status') in {'RUNNING','PENDING','WAITING'}]
    if live_workers:
        active=live_workers[0]
        progress={**progress,'workflow':active['workflow'],'phase':active['phase']}
    corpus_note=(f'{admission.get("teacher_episodes",0)} teacher episodes admitted' if admission else 'Warmstart corpus preparation is in progress')
    summary=(f'{len(graded_teacher)} native Astra tasks graded, {sum(r["score"] for r in graded_teacher)} successful; '
             f'{len(live_workers)} teacher workers active or queued. {corpus_note}; {training_note} '
             'Priority: base Qwen, Astra warmstart, then warmstart + LeAct.') if workers else progress.get('detail','Preparing EFE pilots.')
    if service_error:summary+=' OSMO API returns 503: new jobs await submission and live receipts may be stale; automatic retry is active.'
    training=status.get('training_pilot',{})
    training_directory=Path(training['directory']) if training.get('directory') else ROOT/'training-pilot/run-v2'
    trace_admission=read(training_directory/'trace-admission.json',{})
    if training:
        summary+=' Training pilot: '+training.get('progress',{}).get('detail',training.get('scheduler_status','preparing')).rstrip('.')+'.'
    scene=read(ROOT/'training-pilot/scene-audit-v1.json',{})
    if trace_admission and not admission:
        summary+=f' {trace_admission.get("decisions",0)} public decisions passed trace admission; the training corpus awaits complete explanation pairs.'
    elif scene and not admission:
        summary+=f' {scene.get("teacher_episodes_eligible_for_trace_review",0)} episodes passed the initial scene audit and await full trace admission.'
    schedule_comparison=schedule_plan(ROOT)
    continuation=read(ROOT/'training-pilot/continuation-recovery.json',{})
    continuation_prior=[]
    for arm in continuation.get('arms',[]):
        for prior in arm.get('previous_attempts',[]):
            continuation_prior.append({**arm,**prior,'id':arm['id']+'-attempt-'+str(prior['attempt']),
                'name':arm['name']+' · interrupted attempt '+str(prior['attempt'])})
    training_runs=training_histories(ROOT,{**schedule_comparison,
        'arms':schedule_comparison.get('arms',[])+continuation.get('arms',[]),
        'prior_attempts':schedule_comparison.get('prior_attempts',[])+continuation_prior})
    if schedule_comparison:
        summary+=' Warmstart schedule comparison: three matched 48-update runs at LR 1e-6, 1e-5 and the BuilderBench reference 1e-4; final checkpoints receive native development evaluations.'
        interrupted=[r for r in training_runs if r['id'] in {a['id'] for a in schedule_comparison.get('arms',[])}
                     and (r.get('scheduler_status') or '').startswith('FAILED') and not r['complete']]
        if interrupted:
            summary+=f' {len(interrupted)} comparison runs ended before completing training: '+', '.join(sorted({r['status'] for r in interrupted}))+'. No completed final 48-update comparison checkpoint is available for evaluation.'
        if schedule_comparison.get('recovery'):
            recovery=schedule_comparison['recovery']
            summary+=' Original comparison attempts were canceled by GPU quota enforcement.'
            if recovery.get('execution_paused'):
                summary+=' These from-base LR comparisons are paused while the authorized step-32 continuation and native robot evaluations take priority.'
            elif recovery.get('retry_policy',{}).get('enabled'):
                slots=recovery.get('max_concurrent_training_nodes',1)
                summary+=f' Capacity interruptions and verified OSMO or GPU interconnect failures are retried at LOW priority with 5–15 minute backoff and at most {slots} concurrent training jobs. The 48-update recipe and data remain unchanged.'
                if schedule_comparison.get('recovery_status',{}).get('next_retry_utc'):
                    summary+=' Next retry: '+schedule_comparison['recovery_status']['next_retry_utc']+'.'
            else:
                summary+=' Recovery allows one retry per arm with at most one training job at a time; each restarts the unchanged 48-update recipe from base.'
            if schedule_comparison.get('recovery_status',{}).get('status')=='needs_review':
                summary+=' The recovery queue stopped after a retry failure and requires review.'
            elif schedule_comparison.get('recovery_status',{}).get('review_required'):
                summary+=' A failed comparison arm is held for diagnosis; independent arms continue.'
    if continuation:
        current=('Current priority: native EFE simulator robot evaluations. Continue the saved step-32 warmstart to total step 96 (64 additional updates), restoring Adam state and keeping the frozen data order. Capability screens are paused. ')
        if continuation.get('execution_paused'):
            current+='Continuation is prepared but not submitted: '+continuation.get('pause_reason','execution paused')+' '
        else:
            row=next((r for r in training_runs if r['id']=='qwen-warm-continue32-96'),{})
            current+='Continuation status: '+str(row.get('status','awaiting receipt'))+'. '
        summary=current+summary
    budget_review=read(ROOT/'training-pilot/training-budget-review.json',{})
    if budget_review.get('interpretation'):
        summary+=' Training-budget review: '+budget_review['interpretation']
    student_dir=ROOT/read(ROOT/'qwen-active.json',{}).get('directory','qwen-base')
    student=read(student_dir/'progress.json',{})
    teacher_development=[r for r in teacher_pilot['trials'] if r.get('partition')=='development']
    teacher_development_valid=[r for r in teacher_development if r['status']=='graded']
    development=[{'id':'astra','name':'Astra teacher','baseline':None,'graded':len(teacher_development_valid),
         'successes':sum(r['score'] for r in teacher_development_valid),'planned':7,
         'simulator_calls':sum(r.get('resource_metrics',{}).get('calls',0) for r in teacher_development_valid),
         'detail':'Frontier reference on the same seven development tasks; partial coverage is not a final comparison.'}]
    development_entries=[
        ('qwen-base','Base Qwen, tool-aligned',student_dir,None),
        ('qwen-warmstart','Qwen + Astra warmstart',ROOT/read(ROOT/'training-pilot/warmstart-evaluation-active.json',{}).get('directory','training-pilot/warmstart-evaluation'),'qwen-base'),
        ('qwen-leact','Warmstart + LeAct',ROOT/read(ROOT/'training-pilot/leact-evaluation-active.json',{}).get('directory','training-pilot/leact-evaluation'),'qwen-warmstart')]
    comparison_arms={arm['id']:arm for arm in schedule_comparison.get('arms',[])}
    development_entries.extend((arm['id'],arm['name'],ROOT/arm['evaluation_directory'],arm['baseline'])
                               for arm in comparison_arms.values())
    continuation_arms={arm['id']:arm for arm in continuation.get('arms',[])}
    development_entries.extend((arm['id'],arm['name'],ROOT/arm['evaluation_directory'],arm['baseline'])
                               for arm in continuation_arms.values())
    intermediate={arm['id']:arm for arm in read(ROOT/'training-pilot/intermediate-evaluations.json',{}).get('runs',[])}
    development_entries.extend((arm['id'],arm['name'],ROOT/arm['evaluation_directory'],arm['baseline'])
                               for arm in intermediate.values())
    additional={arm['id']:arm for arm in read(ROOT/'training-pilot/additional-runs.json',{}).get('runs',[])}
    development_entries.extend((arm['id'],arm['name'],ROOT/arm['evaluation_directory'],arm['baseline'])
                               for arm in additional.values() if arm.get('evaluation_partition','development')=='development')
    for ident,label,directory,baseline in development_entries:
        results=read(directory/'results.json',{});rows=results.get('trials',[])
        valid=[r for r in rows if r.get('status')=='graded']
        update=read(directory/'progress.json',{})
        detail=update.get('detail','Waiting for the preceding checkpoint and validation')
        if ident in additional:
            arm=additional[ident]
            training_progress=read(ROOT/arm['local_results_directory']/'progress.json',{})
            pipeline=read(ROOT/arm['local_results_directory']/'pipeline-progress.json',{})
            pieces=[arm['description'],pipeline.get('detail',training_progress.get('detail','')),detail]
            detail=' '.join(dict.fromkeys(p for p in pieces if p))
        if ident in intermediate:
            arm=intermediate[ident]
            detail=(f"Intermediate checkpoint at update {arm['step']}/{arm['planned_final_step']}; "
                    'last saved predeclared snapshot after preemption. Separate from the fixed final-checkpoint comparison. '+detail)
        interpretation=read(directory/'interpretation.json',{})
        if interpretation.get('workflow')==update.get('workflow') and interpretation.get('detail'):
            detail+=' '+interpretation['detail']
        transport=read(directory/'student-serving.json',{}).get('serving_transport',{})
        if ident in intermediate:transport=intermediate[ident].get('serving_transport',transport)
        if transport.get('mode')=='nccl_no_p2p_v1':
            detail+=' GPU communication workaround: direct peer access and the custom all-reduce kernel are disabled. Any prior graded outcomes retain their source workflow. Runtime changed, so latency and time-limited scores require this caveat; do not use this attempt for a matched rollout-speed claim.'
        if ident=='qwen-base' and valid and not any(r.get('resource_metrics',{}).get('calls',0) for r in valid):
            detail+=' Completed attempts made no simulator calls; inspected traces wrote code and ended before robot execution.'
        if ident=='qwen-warmstart':
            budget=read(ROOT/'training-pilot/run-v5/budget-audit.json',{})
            if budget:
                detail=(f"Initial development pilot: {budget['optimizer_updates']} optimizer updates at effective batch {budget['effective_batch']}; "
                        f"{budget['example_presentations']} example presentations, {budget['distinct_decision_states_presented']} distinct decision states. "+detail)
                if len(valid)==7:
                    zero=sum(not row.get('resource_metrics',{}).get('calls',0) for row in valid)
                    detail+=f' {sum(row["score"] for row in valid)}/7 successes; {zero}/7 episodes made no simulator calls. These are initial-pilot results; the official core evaluation has not run.'
        if ident=='qwen-leact' and len(valid)==7:
            zero=sum(not row.get('resource_metrics',{}).get('calls',0) for row in valid)
            detail+=f' Initial LeAct pilot: {sum(row["score"] for row in valid)}/7 successes; {zero}/7 episodes made no simulator calls. Training loss reduction has not translated into executable task success.'
        if ident in comparison_arms:
            arm=comparison_arms[ident];run_dir=ROOT/arm['local_results_directory']
            training_progress=read(run_dir/'progress.json',{})
            scheduler=read(run_dir/'scheduler.json',{})
            training_row=next((r for r in training_runs if r['id']==ident),{})
            detail=(f"{arm['role']}. {arm['optimizer_steps']} updates at batch {arm['effective_batch']}, LR {arm['lr']:g}; "
                    f"{arm['example_presentations']} example presentations, {arm['distinct_decision_states']} distinct states. "
                    'Same frozen corpus and sampled order across LRs; three balanced passes, not complete epochs. '+
                    (training_row.get('detail') or training_progress.get('detail',scheduler.get('status','Awaiting scheduler receipt')))+'. '+detail)
        if ident in continuation_arms:
            arm=continuation_arms[ident]
            training_row=next((r for r in training_runs if r['id']==ident),{})
            native_complete=bool(read(directory/'complete.json',{})) and len(valid)==7
            continuation_status=('Training and native development evaluation complete; '+
                str(sum(r['score'] for r in valid))+'/7 successes' if native_complete else
                str(training_row.get('detail') or training_row.get('status','Awaiting submission')))
            detail=('Saved step 32 → total step 96: 64 additional updates with restored Adam state; '
                    'same frozen data order, batch 32 and 32K visual context. Initial LR 3.12697e-5 decays over the continuation. '
                    'Original checkpoint has bf16 weights and lacks per-rank CUDA RNG, so this is a documented continuation rather than an exact uninterrupted replay. '
                    'Capability screen paused. '+continuation_status+'. '+detail)
        development.append({'id':ident,'name':label,'baseline':baseline,'graded':len(valid),
             'successes':sum(r['score'] for r in valid),'planned':7,
             'infrastructure_failures':sum(r.get('status')!='graded' for r in rows),
             'simulator_calls':sum(r.get('resource_metrics',{}).get('calls',0) for r in valid),
             'workflow':results.get('workflow',update.get('workflow')),
             'detail':detail,
             'trials':[{'task':r['task'],'score':r.get('score'),'status':r.get('status')} for r in rows]})
    if intermediate:
        summary+=' Intermediate checkpoints are evaluated separately on the same seven development tasks; they are not final 48-update results.'
    if additional:
        notes=[]
        for arm in additional.values():
            directory=ROOT/arm['local_results_directory']
            update=read(directory/'progress.json',{})
            pipeline=read(directory/'pipeline-progress.json',{})
            scheduler=read(directory/'scheduler.json',{})
            evaluation=read(ROOT/arm['evaluation_directory']/'results.json',{}) if arm.get('evaluation_directory') else {}
            trials=evaluation.get('trials',[])
            graded=[r for r in trials if r.get('status')=='graded']
            if graded:
                cohort='training' if arm.get('evaluation_partition')=='training_evaluation' else 'development'
                measured=(f'{len(graded)}/{evaluation.get("planned_trials",7)} {cohort} tasks graded, '
                    f'{sum(r.get("score") or 0 for r in graded)} successes, '
                    f'{sum(r.get("resource_metrics",{}).get("control_steps",0) for r in graded)} robot-control steps. ')
            else:
                measured=''
            detail=pipeline.get('detail',update.get('detail',scheduler.get('status','queued')))
            if scheduler.get('status','').startswith('FAILED') and not read(directory/'train/complete.json',{}).get('success'):
                detail=f"Allocation ended; last recorded update {update.get('step',0)}. No completed training checkpoint from this attempt."
            notes.append(arm['name']+': '+measured+detail)
        summary='Current follow-up: '+' '.join(notes)+' '+summary
    student_audit=read(student_dir/'interpretation.json',{})
    if student_audit.get('workflow')==student.get('workflow') and student_audit.get('detail'):
        student={**student,'detail':student_audit['detail'],'audit':student_audit}
    leact_result=next((r for r in development if r['id']=='qwen-leact' and r['graded']==7),None)
    if leact_result:
        evaluation_note=(f'Initial LeAct development evaluation completed: {leact_result["successes"]}/7 successes, '
                         f'{leact_result["simulator_calls"]} simulator calls. The longer schedule comparisons are separate runs.')
        if training.get('progress',{}).get('phase')=='leact_complete':
            old_note=training['progress'].get('detail','').rstrip('.')+'.'
            summary=summary.replace('Training pilot: '+old_note,'Training pilot: '+evaluation_note)
            training={**training,'progress':{**training['progress'],'detail':evaluation_note}}
    direct_status=read(ROOT/'direct-control-v1/status.json',{})
    followups=read(ROOT/'native-followups-v1/status.json',{})
    recovery=read(ROOT/'recovery-20261101/native/status.json',{})
    newer_recovery=read(ROOT/'recovery-20261102/native/status.json',{})
    if newer_recovery:
        merged={r['id']:r for r in recovery.get('arms',[])}
        merged.update({r['id']:r for r in newer_recovery.get('arms',[])})
        recovery={**newer_recovery,'arms':list(merged.values())}
    recovered={r['id']:r for r in recovery.get('arms',[])}
    if recovered:
        followups={**followups,'arms':[{**a,**recovered.get(a['id'],{})} for a in followups.get('arms',[])]}
    curated=read(ROOT/'direct-control-v1/curated-warmstart-v1/status.json',{})
    next_leact=read(ROOT/'direct-control-v1/curated-leact-v1/queue-status.json',{})
    next_plan=read(ROOT/'direct-control-v1/curated-leact-v1/plan.json',{})
    next_progress=read(ROOT/'direct-control-v1/curated-leact-v1/progress.json',{})
    next_selection=read(ROOT/'direct-control-v1/curated-leact-v1/selection.json',{})
    next_recovery=read(ROOT/'direct-control-v1/curated-leact-v1/recovery.json',{})
    next_complete=read(ROOT/'direct-control-v1/curated-leact-v1/pipeline-complete.json',{})
    if next_complete.get('success'):
        trained=next_complete.get('training',{}).get('step')
        native=next_complete.get('native',{})
        next_leact={**next_leact,'status':'COMPLETED','experiment_complete':True}
        next_progress={**next_progress,'phase':'complete',
            'detail':f"Training finished {trained}/{trained} updates and the final checkpoint was verified. Native robot evaluation: {native.get('successes',0)}/{native.get('graded',0)} successes. Low training loss has not translated into reliable task execution."}
    if direct_status.get('workers'):
        direct_graded=sum(w.get('graded',0) for w in direct_status['workers'])
        direct_success=sum(w.get('successes',0) for w in direct_status['workers'])
        direct_eligible=sum(w.get('eligible_episodes',0) for w in direct_status['workers'])
        direct_attempts=sum(max(w.get('graded',0),w.get('progress',{}).get('completed_trials',0)) for w in direct_status['workers'])
        summary=(f'Direct-control Astra collection: {direct_success}/{direct_attempts} completed attempts succeeded '
                 f'({direct_graded} completed without a recorded infrastructure interruption); '
                 f'{direct_eligible} episodes provisionally pass collection checks. '
                 'Base Qwen and warmstart comparisons use the same direct-control instruction suffix; new curated SFT is separate. '+summary)
    warm=next((a for a in curated.get('arms',[]) if a['id']=='curated-warm64'),None)
    if warm:
        active_note=warm.get('progress',{}).get('detail',warm['status'])
        training_progress=warm.get('training_progress',{})
        durable=read(ROOT/'direct-control-v1/curated-warmstart-v1/curated-warm64/durable-training-ready.json',{})
        if durable.get('all_files_verified'):
            active_note=f"Warmstart finished {training_progress.get('steps','?')} updates; final checkpoint verified. "
            if warm.get('graded',0):
                active_note+=f"Native training-task evaluation: {warm.get('successes',0)}/{warm['graded']} successes, {warm.get('planned',6)} planned; {warm.get('control_steps',0)} robot-control steps across graded tasks. "
            else:
                active_note+='Native training-task evaluation has no graded results yet. '
            active_note+=warm.get('progress',{}).get('detail',warm['status'])
            diagnosis=read(ROOT/'direct-control-v1/students/curated-warm64/rollout-diagnosis.json',{})
            if diagnosis.get('summary'):
                active_note+=' '+diagnosis['summary']
        elif training_progress.get('phase') in {'train','checkpoint_saving','warmstart_training'}:
            active_note=training_progress.get('detail',active_note)
            restart=read(ROOT/'direct-control-v1/curated-warmstart-v1/curated-warm64/production/restart-ready.json',{})
            if training_progress.get('phase')=='checkpoint_saving' and restart.get('step',0)>=training_progress.get('step',1):
                active_note=f"Checkpoint {restart['step']}/{training_progress.get('steps','?')} committed; continuing the remaining updates"
            if training_progress.get('step'):
                active_note+=f". Forward loss {training_progress['forward_ce']:.3f}; backward loss {training_progress['backward_ce']:.3f}. These are training losses; curated-model task success is not yet measured."
        resume_note=' '.join(f"{r['id']}: {r['successes']}/{r['graded']} successes, {r['status'].lower()}" for r in recovery.get('arms',[]))
        next_note=('The first curated LeAct round is waiting for a verified final warmstart checkpoint; it will launch automatically within the node limit. '
            if next_plan.get('enabled') and next_leact.get('phase')=='waiting_for_verified_curated_warmstart' else '')
        if next_plan.get('enabled') and next_leact.get('workflow'):
            next_note='Curated LeAct: '+next_progress.get('detail',next_leact.get('status','Submitted'))+' '
            if next_recovery.get('summary') and (next_progress.get('phase')=='needs_review' or next_leact.get('status') in {'PENDING','WAITING','FAILED'}):
                next_note='Curated LeAct: '+next_recovery['summary']+' '
            if next_selection:
                next_note+=f"Admitted {next_selection['admitted_candidates']} explanations across {next_selection['admitted_states']}/{next_selection['states_attempted']} states. Admission measures action-likelihood improvement, not robot task success. "
        resume=read(ROOT/'recovery-20261102/progress-note.json',{})
        interruption_note=(resume['summary']+' ' if resume else
            'The previous 64K run ended in an OSMO Control failure before checkpointing; its last collected update was 7/83. ')
        summary=('Current curated study: '+active_note+' '+interruption_note+
            'The new attempt retains its corpus on shared storage and saves full model/Adam/RNG restart state periodically. '+
            next_note+'Legacy native evaluations: '+resume_note+'. '+summary)
        progress={**progress,'workflow':warm.get('workflow'),'phase':'Curated warmstart: '+warm.get('progress',{}).get('phase',warm['status'])}
    curated_students=curated.get('arms',[])
    if next_plan.get('enabled'):
        behavior=read(ROOT/'direct-control-v1/students/curated-leact/behavior.json',{})
        curated_students=curated_students+[{'id':'curated-leact','label':'Curated warmstart + LeAct R1',
            'status':next_leact.get('status','Waiting for curated warmstart'),
            'graded':behavior.get('graded',0),'successes':behavior.get('successes',0),'planned':6,
            'control_steps':sum(t['control_steps'] for t in behavior.get('trials',[])),
            'progress':read(ROOT/'direct-control-v1/curated-leact-v1/pipeline-progress.json',
                {'detail':'Automatically follows the verified curated warmstart. Same six-task native comparison.'})}]
    active_warm=read(ROOT/'fixed-action-v1/active-warmstart.json',
        {'directory':'fixed-action-v1/warmstart-lora-v1','history_id':'efe-qwen36-constrained-warm-lora'})
    warm_directory=ROOT/active_warm['directory']
    return {'updated_utc':now,'approval':read(ROOT/'execution-authorization.json',{}),
            'continuation':continuation,
            'summary':summary,
            'workflow':progress.get('workflow'),'phase':progress.get('phase','Preparing native simulator controls'),
            'training_started':status.get('training_started',False),'benchmark_rollouts_started':status.get('benchmark_rollouts_started',False),
            'counts':{'train_candidates':split['train_candidate_tasks'],'development':split['development_tasks'],
                      'core':split['official_core_tasks'],'admitted_teacher_episodes':admission.get('teacher_episodes',0),
                      'admitted_native_episodes':admission.get('native_episodes',0)},
            'cohorts':cohorts,'arms':arms,'gates':gates,'diagnostics':diagnostics,'recipe':recipe,
            'capability_retention':capability_dashboard(),
            'teacher_pilot':teacher_pilot,'native_transport':native_transport,'workers':workers,
            'launch_status':launch_status,'osmo_service_error':service_error,
            'student':student,'development':development,'training_pilot':training,
            'training_task_evaluation':read(ROOT/'training-evaluation-v1/status.json',{}),
            'trace_diagnosis':read(ROOT/'trace-diagnosis-v1/report.json',{}),
            'student_qualification':{'plan':read(ROOT/'structured-control-v2/plan.json',{}),
                                     'status':read(ROOT/'structured-control-v2/status.json',{}),
                                     'attempts':[row for i in (1,2) if (row:=read(ROOT/f'structured-control-v2/attempts/attempt-{i}.json',{}))],
                                     'published_audit':read(ROOT/'structured-control-v2/published-audit.json',{})},
            'coverage_expansion':{'plan':{k:v for k,v in read(ROOT/'fixed-action-v1/coverage-scale-v1/plan.json',{}).items() if k not in {'tasks','protected_tasks'}},
                'status':read(ROOT/'fixed-action-v1/coverage-scale-v1/status.json',{}),
                'admission':read(ROOT/'fixed-action-v1/coverage-scale-v1/corpus-admission-summary.json',{}),
                'storage_recovery':read(ROOT/'fixed-action-v1/coverage-scale-v1/storage-recovery-status.json',{}),
                'training_recovery':read(ROOT/'fixed-action-v1/coverage-scale-v1/training-recovery-status.json',{}),
                'training':read(ROOT/'fixed-action-v1/coverage-scale-v1/training-status.json',{}),
                'intermediate_evaluations':read(ROOT/'fixed-action-v1/coverage-scale-v1/intermediate-evaluations.json',{}),
                'branch_runs':read(ROOT/'fixed-action-v1/coverage-scale-v1/branch-runs.json',{}),
                'selection_workers':read(ROOT/'fixed-action-v1/coverage-scale-v1/selection-workers.json',{}),
                'execution_priorities':read(ROOT/'fixed-action-v1/coverage-scale-v1/execution-priorities.json',{}),
                'hf_scoring_recovery':read(ROOT/'fixed-action-v1/coverage-scale-v1/hf-recovery-status.json',{}),
                'performance':read(ROOT/'fixed-action-v1/coverage-scale-v1/performance-analysis.json',{}),
                'evaluation':read(ROOT/'fixed-action-v1/coverage-scale-v1/evaluation-status.json',{})},
            'pear_scaled':{'plan':read(ROOT/'pear-scaled-v1/plan.json',{}),
                           'status':read(ROOT/'pear-scaled-v1/status.json',{}),
                           'catalog_checks':read(ROOT/'pear-scaled-v1/catalog-status.json',{}),
                           'model_staging':read(ROOT/'pear-scaled-v1/model-files-ready.json',{}),
                           'model_checks':read(ROOT/'pear-scaled-v1/model-check-status.json',{}),
                           'compute_authorization':read(ROOT/'compute-authorization.json',{}),
                           'native_bridge':read(ROOT/'pear-scaled-v1/bridge-status.json',{}),
                           'checkpoint_names':read(ROOT/'pear-scaled-v1/checkpoint-names.json',{})},
            'pear_comparison':{'plan':read(ROOT/'pear-v1/plan.json',{}),
                               'status':read(ROOT/'pear-v1/status.json',{}),
                               'qualification':read(ROOT/'pear-v1/qualification-result.json',{}),
                               'native_qualification':{'plan':read(ROOT/'pear-v1/native-qualification-plan.json',{}),
                                   'result':read(ROOT/'pear-v1/native-qualification-result.json',{})}},
            'fixed_action':{'plan':read(ROOT/'fixed-action-v1/plan.json',{}),
                            'status':read(ROOT/'fixed-action-v1/status.json',{}),
                            'completed_comparison':read(ROOT/'fixed-action-v1/runs/v4/completed-comparison.json',
                                read(ROOT/'fixed-action-v1/runs/v3/completed-comparison.json',{})),
                            'warmstart':{'plan':read(warm_directory/'plan.json',{}),
                                'status':read(warm_directory/'status.json',{}),
                                'results':read(warm_directory/'results-summary.json',{}),
                                'leact':{'plan':read(ROOT/'fixed-action-v1/leact-lora-v1/plan.json',{}),
                                    'status':read(ROOT/'fixed-action-v1/leact-lora-v1/status.json',{}),
                                    'results':read(ROOT/'fixed-action-v1/leact-lora-v1/results-summary.json',{}),
                                    'followup':read(ROOT/'fixed-action-v1/leact-followup-v1/status.json',{}),
                                    'history':next((r for r in training_runs if r['id']=='efe-qwen36-constrained-leact-lora-v1'),{})},
                                'admission':read(warm_directory/'admission.json',{}),
                                'history':next((r for r in training_runs if r['id']==active_warm['history_id']),{})}},
            'direct_control':{'plan':read(ROOT/'direct-control-v1/plan.json',{}),
                              'status':direct_status,
                              'students':[a for a in followups.get('arms',[]) if a.get('direct')]
                                  +curated_students,
                              'next_leact':{'plan':next_plan,'status':next_leact,'progress':next_progress,'selection':next_selection,'recovery':next_recovery},
                              'training_preparation':read(ROOT/'direct-control-v1/training-preparation.json',{})},
            'native_followups':followups,
            'training_runs':training_runs,'trace_admission':trace_admission,'scene_admission':{
                'eligible_episodes':scene.get('teacher_episodes_eligible_for_trace_review',0),
                'excluded_episodes':[{'task':r['task'],'reason':r.get('reason','Scene overlaps protected task'),
                                     'collisions':r.get('collisions',[])} for r in scene.get('excluded_episodes',[])],
                'scope':scene.get('scope')},'priority':plan.get('execution_priority',[]),
            'schedule_comparison':schedule_comparison,
            'training_budget_review':budget_review,
            'comparison_note':'Each student is compared with its linked baseline. Qwen uses local Codex context compaction; Astra currently uses provider compaction. Both new runs cap output at 32768 tokens and retain the newest 40 images, but their long-context histories are not identical. Demo reasoning versus demo actions isolates reasoning supervision on the same admitted native decisions. Extra reasoning tokens are reported as extra compute.',
            'coverage_note':'Cohort counts describe planned training coverage. Recompute from actual admitted data before reporting generalization; missing measurements stay blank.',
            'resources':read(ROOT/'execution-authorization.json',{}).get('compute','Compute authorization pending')}


def publish():
    with (BASE/'cluster/.build/builderbench-publish.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value=build();write(ROOT/'dashboard.json',value)
        path=BASE/'artifacts/experiment-plan.json'; plan=read(path)
        plan['embodied_first_exam']['dashboard']=value
        plan['embodied_first_exam']['status']=value['summary']
        plan['embodied_first_exam']['approved']=bool(value['approval'])
        for item in plan['sequence']:
            if item['id']=='embodied_first_exam':item['status']=value['phase']
        plan['updated_utc']=value['updated_utc'];write(path,plan)
        from ..builderbench.queue import publish as publish_dashboard
        publish_dashboard()
    return value


if __name__=='__main__':
    value=publish()
    print(json.dumps({'phase':value['phase'],'arms':len(value['arms']),'updated_utc':value['updated_utc']}))
