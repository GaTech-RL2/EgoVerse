"""Queue a method-controlled teacher pilot, then a balanced training collection."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write, submit, command
from .compute_budget import node_limit, reserved_nodes
from .launch_efe_training import freeze_sources
from .launch_efe_leact_warm96 import collect, download
from .monitor_efe_preflight import remote

DEST=ROOT/'direct-control-v1'
REMOTE='/tmp/efe-direct-control-v1'
PREFIX='efe-direct-control-20261030-v1'
DURABLE='/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-direct-control-20261030-v1'

# All membership is checked against the immutable train partition below.
TASKS=[
 ('metaworldplus','handle-press-side','press',True),
 ('robocasa','turn-on-sink-faucet','turn',True),
 ('metaworldplus','push-wall','push',True),
 ('robocasa','open-double-door','open',True),
 ('metaworldplus','pick-out-of-hole','grasp and lift',True),
 ('metaworldplus','assembly','align and insert',True),
 ('metaworldplus','button-press','press',False),
 ('metaworldplus','button-press-topdown-wall','press around obstacle',False),
 ('metaworldplus','coffee-button','press',False),
 ('metaworldplus','dial-turn','turn',False),
 ('metaworldplus','door-lock','slide latch',False),
 ('metaworldplus','door-unlock','slide latch',False),
 ('metaworldplus','faucet-close','turn',False),
 ('metaworldplus','window-open','slide',False),
 ('metaworldplus','window-close','slide',False),
 ('metaworldplus','peg-insert-side','align and insert',False),
 ('metaworldplus','pick-place','grasp and place',False),
 ('metaworldplus','sweep-into','push into opening',False),
 ('robocasa','open-single-door','open',False),
 ('robocasa','close-single-door','close',False),
 ('robocasa','close-drawer','slide',False),
 ('robocasa','turn-sink-spout','turn',False),
 ('robocasa','coffee-press-button','press',False),
 ('robocasa','pnp-cab-to-counter','grasp and place',False),
]


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    DEST.mkdir(exist_ok=True)
    if (DEST/'plan.json').exists():return read(DEST/'plan.json')
    splits=read(ROOT/'generalization-v2/splits.json')
    inventory={r['task']:r for r in read(ROOT/'generalization-v2/inventory.json')}
    teachers={r['task']:r for r in read(ROOT/'trace-diagnosis-v1/report.json')['teachers']}
    rows=[]
    for suite,name,family,pilot in TASKS:
        task=f'{suite}-{name}-i00-standard';row=inventory[task]
        assert task in splits['train_candidates'] and task not in splits['development']
        assert task not in splits['official_core_evaluation'] and task not in splits['simulator_holdout_reserve']
        assert row['seeds']['allowed']==[0] and not row['resources'].get('gpu_required')
        prior=teachers.get(task)
        if pilot:assert prior and prior['score']==1
        rows.append({**row,'partition':'train_candidates','control_family':family,'pilot':pilot,
            'prior_teacher_episode':prior['source_episode'] if prior else None})
    assert len(rows)==24 and len({r['task'] for r in rows})==24
    template=read(BUILD_ROOT/'efe-astra-pilot-20261026-v2/spec.json')
    assert template['revision']=='b88c450c46de8a96bba83bf9606b66396cccfff5'
    batches=[('pilot',1,[r for r in rows if r['pilot']])]
    for attempt in range(1,4):
        remaining=[r for r in rows if attempt>1 or not r['pilot']]
        batches.extend((f'a{attempt}-{i//6+1:02d}',attempt,remaining[i:i+6]) for i in range(0,len(remaining),6))
    builds=[]
    for name,attempt,tasks in batches:
        build=BUILD_ROOT/f'{PREFIX}-{name}';build.mkdir(exist_ok=True)
        spec={**template,'tasks':tasks,'max_output_tokens':32768,'max_images':40,
            'archive_root':REMOTE,'node_local_results':True,'local_episode_outputs':False,
            'local_results_directory':f'direct-control-v1/{name}',
            'purpose':'Training-only direct-control teacher collection. Preserve every attempt and admit action targets using their executed feedback.',
            'protocol_version':'direct-control-v1','protocol_label':'New research condition: native tool set plus recorded direct-control instruction suffix. Separate from legacy unrestricted-controller EFE scores.',
            'instruction_suffix_file':'/tmp/efe-direct-control-prompt.txt',
            'instruction_suffix_sha256':sha(BASE/'cluster/efe_direct_control_prompt.txt'),
            'attempt_index':attempt,'environment_seed':0,
            'agent_label':'Astra direct control','agent_slug':'astra-direct',
            'retry_policy':'Keep every graded success/failure. Capacity retries use LOW priority; no automatic policy retries beyond the three predeclared attempts.'}
        write(build/'spec.json',spec)
        files=[{'localpath':'spec.json','path':'/tmp/efe-native-spec.json'}]
        for source,target in [('efe_native_pilot.py','/tmp/efe_native_pilot.py'),
            ('efe_astra_pilot.py','/tmp/efe-astra-pilot.py'),
            ('efe_budget_gateway.py','/tmp/efe_budget_gateway.py'),
            ('export_efe_viewer.py','/tmp/export_efe_viewer.py'),
            ('efe_direct_control_worker.py','/tmp/efe_direct_control_worker.py'),
            ('efe_direct_control_prompt.txt','/tmp/efe-direct-control-prompt.txt')]:
            content=(BASE/'cluster'/source).read_text()
            if source=='efe_astra_pilot.py':
                # Keep node-local artifacts until the collector verifies them.
                content=content.replace('for _ in range(60):\n            if (runtime.ROOT / \'collected\')',
                                        'for _ in range(4320):\n            if (runtime.ROOT / \'collected\')')
            (build/source).write_text(content)
            if source.endswith('.py'):compile(content,str(build/source),'exec')
            files.append({'localpath':source,'path':target})
        bootstrap="""set -euo pipefail
mkdir -p /tmp/efe-native
python3 -m venv --system-site-packages /tmp/efe-native/emimic
source /tmp/efe-native/emimic/bin/activate
python -m pip install --disable-pip-version-check aiohttp==3.13.3
cp /tmp/efe_budget_gateway.py /tmp/efe-budget-gateway.py
python -u /tmp/efe_direct_control_worker.py
"""
        (build/'bootstrap.sh').write_text(bootstrap);files.append({'localpath':'bootstrap.sh','path':'/tmp/efe-native-bootstrap.sh'})
        w=yaml.safe_load((BUILD_ROOT/'efe-astra-pilot-20261026-v2/workflow.yaml').read_text())
        w['workflow']['name']='leact-efe-direct-'+name+'-v1'
        w['workflow']['resources']['default'].update(storage='800Gi',memory='128Gi')
        w['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'18h'}
        w['workflow']['tasks'][0]['files']=files
        (build/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False))
        assert len({sha(build/f['localpath']) for f in files})==len(files)
        freeze_sources(build);builds.append({'name':name,'attempt':attempt,'build':str(build),'tasks':[r['task'] for r in tasks]})
    plan={'created_utc':now(),'protocol':'direct-control-v1','tasks':rows,'builds':builds,
        'pilot_tasks':6,'training_tasks':24,'suites':2,'attempts_per_task':3,'total_attempts':72,
        'target_successful_episodes':48,'target_successful_episodes_per_task':2,
        'automatic_expansion_gate':{'pilot_graded':6,'minimum_method_compliant_successes':4,'infrastructure_failures':0},
        'max_teacher_nodes':2,'study_node_cap':node_limit(),
        'splits_sha256':sha(ROOT/'generalization-v2/splits.json'),
        'inventory_sha256':sha(ROOT/'generalization-v2/inventory.json'),
        'prompt_sha256':sha(BASE/'cluster/efe_direct_control_prompt.txt'),
        'summary':[
            'Collect six Astra pilot tasks using one repeated method: inspect the observation, submit a short explicit native action block, inspect the result, and adjust. Preserve native task rules and grading.',
            'If at least four of six tasks succeed with the declared method and all six have valid grader outcomes, automatically expand to 24 training tasks (16 MetaWorld+, 8 RoboCasa), with three predeclared teacher attempts per task: 72 attempts in total, not 72 promised successes.',
            'Aim for at least two admitted successful episodes per task (48 in total). Balance by task and episode; retain complete action/recovery sequences. Failed tool calls, empty process polls and helper setup are excluded as positive action targets; observed errors remain context for valid recoveries.',
            'These task packages permit only seed 0. Repeated teacher attempts add policy trajectories, not new randomized environments. The 24 frozen task instances span two suites sharing MuJoCo; additional variant and simulator coverage remains separate work.',
            'No development, official-core, or held-out-simulator task enters this collection. Run base Qwen and the existing warmstart under the same new method before attributing gains to new SFT. A later warmstart and LeAct round must pass prompt, token, image and target-likelihood parity checks.'
        ]}
    write(DEST/'plan.json',plan)
    return plan


def collect_build(build, workflow, dest, *, host=None, persistent=False):
    root=(DURABLE if persistent else REMOTE)+'/'+workflow
    reader=host or workflow
    collect(reader,root,dest,['progress.json','results.json','complete.json','setup-error.json',
        'native-config.json','spec.json','public-episodes/manifest.json'])
    preserved=read(dest/'durable-retained.json',{})
    for row in read(dest/'public-episodes/manifest.json',{'episodes':[]})['episodes']:
        if row.get('id'):
            folder=dest/'public-episodes'/row['id']
            pairs=[('trace.json','trace_sha256'),('replay.mp4','video_sha256'),('action-audit.json','action_audit_sha256')]
            def valid():return all((folder/n).is_file() and sha(folder/n)==row[k] for n,k in pairs)
            if not valid():download(reader,root+'/public-episodes/'+row['id'],dest/'public-episodes',timeout=120)
            if not valid():raise ValueError('Public episode checksum mismatch')
        name=row['retained_file']
        if Path(name).name != name:raise ValueError('Unexpected retained artifact path')
        if preserved.get(name,{}).get('sha256')!=row['retained_sha256']:
            # Shared project storage is writable again (checked on this new
            # allocation). Preserve exact bundles there so 72 attempts do not
            # fill the laptop; training allocations can consume these paths.
            script="""from pathlib import Path
import hashlib,json,shutil
src=Path(SOURCE);dst=Path(DESTINATION);expected=EXPECTED
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
dst.parent.mkdir(parents=True,exist_ok=True)
if not dst.exists():
 temporary=dst.with_suffix(dst.suffix+'.copy.tmp');shutil.copyfile(src,temporary)
 if digest(temporary)!=expected:raise ValueError('Durable artifact copy checksum mismatch')
 temporary.replace(dst)
if digest(dst)!=expected:raise ValueError('Existing durable artifact differs; refusing to overwrite')
print('EFE_CAPTURE '+json.dumps({'path':str(dst),'sha256':expected,'bytes':dst.stat().st_size}),flush=True)
""".replace('SOURCE',repr(root+'/retained/'+name)).replace('DESTINATION',repr(DURABLE+'/'+workflow+'/retained/'+name)).replace('EXPECTED',repr(row['retained_sha256']))
            preserved[name]=remote(reader,script)
            write(dest/'durable-retained.json',preserved)
    complete=read(dest/'complete.json')
    if complete:
        write(dest/'collection-complete.json',{'updated_utc':now(),'workflow':workflow,
            'complete':complete,'manifest_sha256':sha(dest/'public-episodes/manifest.json')})
        if not persistent:
            remote(workflow,"from pathlib import Path\nPath('/tmp/efe-native/collected').touch()\nprint('EFE_CAPTURE {\"ack\":true}',flush=True)")


def reassign_reclaimed_batch(plan, item, build, workflow):
    """Use verified durable grades to move only unfinished task/attempt pairs."""
    folder=DEST/item['name'];results=read(folder/'results.json',{'trials':[]})
    trials=results['trials'];graded={r['task'] for r in trials if r['status']=='graded'}
    if len(graded)!=len(trials):raise ValueError('Non-graded recorded teacher outcomes require inspection')
    if trials and results.get('workflow')!=workflow:raise ValueError('Recovered outcomes belong to another allocation')
    if not graded<=set(item['tasks']):raise ValueError('Recovery changed teacher membership')
    durable=read(folder/'durable-retained.json',{})
    media=read(folder/'public-episodes/manifest.json',{'episodes':[]})['episodes']
    if {r['task'] for r in media}!=graded:raise ValueError('Native grades and retained teacher episodes differ')
    for row in media:
        if durable.get(row['retained_file'],{}).get('sha256')!=row['retained_sha256']:
            raise ValueError('Teacher recovery requires verified retained raw bundles')
    remaining=[t for t in item['tasks'] if t not in graded]
    if not remaining:
        write(folder/'collection-complete.json',{'workflow':workflow,'retained_graded':len(graded),
            'completion_scope':'Every planned native grade and raw episode recovered from shared storage.','updated_utc':now()})
        return plan
    number=1
    existing={r['name'] for r in plan['builds']}
    prefix=item.get('root_batch',item['name'])
    while prefix+'-recovery'+str(number) in existing:number+=1
    name=prefix+'-recovery'+str(number)
    target=BUILD_ROOT/('efe-direct-auto-recovery-'+name)
    if not (target/'source-manifest.json').exists():
        target.mkdir(exist_ok=True)
        for filename in read(build/'source-manifest.json'):shutil.copyfile(build/filename,target/filename)
        spec=read(target/'spec.json');spec['tasks']=[t for t in spec['tasks'] if t['task'] in remaining]
        if len(spec['tasks'])!=len(remaining):raise ValueError('Recovery task subset differs')
        spec.update(local_results_directory='direct-control-v1/'+name,recovery_of=workflow,
            recovery_reason='Automatic recovery after cluster quota reclamation. Durable native grades retained; only unfinished tasks reassigned.',durable_root=DURABLE)
        write(target/'spec.json',spec)
        w=yaml.safe_load((target/'workflow.yaml').read_text());w['workflow']['name']='leact-efe-direct-'+name
        w['workflow']['timeout']['queue_timeout']='12h'
        (target/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(target)
    # An empty recovered batch remains an explicit infrastructure attempt, with
    # zero task/attempt pairs counted toward the 72 graded-trial study budget.
    entries=[]
    for row in plan['builds']:
        if row['name']!=item['name']:entries.append(row);continue
        entries.extend([{**row,'tasks':[t for t in row['tasks'] if t in graded],
            'original_tasks':row.get('original_tasks',row['tasks']),'remaining_reassigned_to':name},
            {'name':name,'attempt':row['attempt'],'build':str(target),'tasks':remaining,
             'root_batch':prefix,'recovery_of':workflow,'priority':'LOW'}])
    pairs=[(row['attempt'],task) for row in entries for task in row['tasks']]
    if len(pairs)!=len(set(pairs)) or len(pairs)!=72:raise ValueError('Teacher recovery duplicated or dropped a planned task/attempt')
    record={'workflow':workflow,'retained_graded':len(graded),'reassigned_ungraded':remaining,
        'completion_scope':'Partial native outcomes retained; remaining tasks reassigned after quota reclamation.','updated_utc':now()}
    write(folder/'collection-complete.json',record)
    plan={**plan,'builds':entries,'recovery_updated_utc':now()};write(DEST/'plan.json',plan)
    write(folder/'automatic-recovery.json',{**record,'next_build':str(target),'next_batch':name})
    return plan


def controller():
    plan=prepare()
    lock=(DEST/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from ..embodied_first_exam.report import publish
    while True:
        try:
            plan=read(DEST/'plan.json')
            active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
            summaries=[];pilot_pass=False;completed=0;plan_changed=False
            teacher_active=sum(any(r['name'].startswith('leact-efe-direct-'+item['name']+'-')
                                  for item in plan['builds']) for r in active)
            for item in plan['builds']:
                dest=DEST/item['name'];dest.mkdir(exist_ok=True)
                active_build=read(dest/'active-build.json',{})
                build=Path(active_build.get('build',item['build']))
                receipt=read(build/'submission.json')
                if not receipt:
                    if item['name']!='pilot' and not pilot_pass:continue
                    queued=read(DEST/'curated-warmstart-v1/plan.json',{}).get('arms',[])
                    active_names={r['name'] for r in active}
                    pending_curated=False
                    for arm in queued:
                        control=DEST/'curated-warmstart-v1'/arm['id']
                        queued_build=Path(read(control/'active-build.json',{}).get('build',arm['build']))
                        queued_receipt=read(queued_build/'submission.json',{})
                        finished=read(control/'pipeline-complete.json',{}) or read(ROOT/arm['directory']/'worker-complete.json',{})
                        pending_curated|=not finished and queued_receipt.get('name') not in active_names
                    if pending_curated and reserved_nodes(active)>=node_limit()-1:continue
                    if teacher_active>=plan['max_teacher_nodes'] or reserved_nodes(active)>=node_limit():continue
                    receipt=submit(build,priority=item.get('priority','HIGH'));active.append({'name':receipt['name'],'status':'PENDING'});teacher_active+=1
                state=command(['osmo','workflow','query',receipt['name'],'--format-type','json'])
                write(dest/'scheduler.json',state)
                if state['status']=='RUNNING' and not (dest/'collection-complete.json').exists():
                    collect_build(build,receipt['name'],dest)
                complete=read(dest/'complete.json',{})
                manifest=read(dest/'public-episodes/manifest.json',{'episodes':[]})
                outcomes=read(dest/'results.json',{'trials':[]})['trials']
                graded=[r for r in outcomes if r['status']=='graded']
                eligible=sum(r.get('candidate_episode',False) for r in manifest['episodes'])
                done=(dest/'collection-complete.json').exists()
                if not done and state.get('cancelled_by')=='groot-idle-job-shutdown' and state['status'].startswith('FAILED'):
                    host=next((r['name'] for r in active if r['status']=='RUNNING'),None)
                    if host:
                        collect_build(build,receipt['name'],dest,host=host,persistent=True)
                        plan=reassign_reclaimed_batch(plan,item,build,receipt['name'])
                        plan_changed=True
                        break
                    summaries.append({'name':item['name'],'workflow':receipt['name'],'scheduler':state['status'],
                        'graded':len(graded),'successes':sum(r['score'] for r in graded),'eligible_episodes':eligible,'collected':False,
                        'progress':{'detail':'Waiting for a running allocation to recover durable receipts before reassigning unfinished tasks.'}})
                    continue
                if done:completed+=1
                if item['name']=='pilot':
                    pilot_pass=done and len(graded)==6 and eligible>=4 and not complete.get('error')
                    if done and not pilot_pass:
                        write(DEST/'status.json',{'phase':'pilot_needs_review','detail':f'Pilot complete: {len(graded)}/6 valid outcomes, {eligible}/6 method-compliant successes. Expansion requires at least 4/6; inspect failures before changing the protocol.','updated_utc':now()})
                        publish();return
                if state['status'] in {'FAILED_QUEUE_TIMEOUT','FAILED_START_ERROR','FAILED_START_TIMEOUT','FAILED_PREEMPTED'} and not (dest/'progress.json').exists():
                    number=int(active_build.get('retry_number',0))+1
                    retry=Path(item['build']+'-low'+str(number));retry.mkdir(exist_ok=True)
                    if not (retry/'source-manifest.json').exists():
                        for name in read(Path(item['build'])/'source-manifest.json'):
                            target=retry/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(item['build'])/name,target)
                        w=yaml.safe_load((retry/'workflow.yaml').read_text());w['workflow']['name']+='-low'+str(number);w['workflow']['timeout']['queue_timeout']='12h'
                        (retry/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(retry)
                    receipt=submit(retry,priority='LOW');write(dest/'active-build.json',{'build':str(retry),'retry_number':number})
                elif (state['status'].startswith('FAILED') or state['status']=='COMPLETED') and not done:
                    raise RuntimeError(receipt['name']+' ended before verified collection; preserve all outcomes and inspect infrastructure')
                summaries.append({'name':item['name'],'workflow':receipt['name'],'scheduler':state['status'],
                    'graded':len(graded),'successes':sum(r['score'] for r in graded),'eligible_episodes':eligible,'collected':done,
                    'progress':read(dest/'progress.json',{})})
            if plan_changed:continue
            detail=f'Direct-control collection: {sum(r["graded"] for r in summaries)} graded attempts, {sum(r["eligible_episodes"] for r in summaries)} provisional eligible successful episodes. '
            if pilot_pass:
                detail+='Pilot gate passed; expanding automatically.'
            else:
                pilot_status=next((r for r in summaries if r['name']=='pilot'),None)
                detail+=('Six-task pilot running: '+pilot_status['progress'].get('detail','initializing the next task')+'.'
                    if pilot_status and pilot_status['scheduler']=='RUNNING'
                    else 'Six-task pilot submitted or waiting for an available authorized node.')
            write(DEST/'status.json',{'phase':'complete' if completed==len(plan['builds']) else 'collecting',
                'detail':detail,'workers':summaries,'pilot_gate_passed':pilot_pass,'updated_utc':now()})
            from ..embodied_first_exam.direct_control_viewer import build as build_viewer
            from ..embodied_first_exam.direct_control_corpus import refresh as refresh_corpus
            refresh_corpus()
            build_viewer()
            publish()
            from .launch_efe_direct_profile import service as profile_corpus
            profile_ready=profile_corpus(active)
            if completed==len(plan['builds']) and profile_ready:return
        except Exception as error:
            write(DEST/'controller-error.json',{'error':str(error)[-1800:],'updated_utc':now()})
        time.sleep(30)


def start():
    DEST.mkdir(exist_ok=True)
    with (DEST/'controller.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','reproductions.leact.cluster.launch_efe_direct_control','monitor'],
            cwd=BASE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(DEST/'controller-launch.json',{'pid':p.pid,'updated_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','start','monitor']);args=parser.parse_args()
    if args.action=='monitor':controller()
    else:
        plan=prepare();print(json.dumps({'tasks':len(plan['tasks']),'builds':len(plan['builds']),'attempts':plan['total_attempts']}))
        if args.action=='start':start()
