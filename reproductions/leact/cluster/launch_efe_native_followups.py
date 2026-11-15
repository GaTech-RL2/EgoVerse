"""Resume rebuilt LeAct evaluation and run matched direct-prompt student controls."""
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

from .scale_efe import BASE, ROOT, BUILD_ROOT, read, write, command, submit
from .compute_budget import reserved_nodes,node_limit
from .launch_efe_training import freeze_sources
from .launch_efe_leact_warm96 import collect,download
from .launch_efe_training_eval import publish_status,publish_viewer
from .monitor_efe_preflight import remote

DEST=ROOT/'native-followups-v1'
LANE='/tmp/efe-training-evaluation-v1/warm96'
OLD='leact-efe-warm96-rebuild-v4-1'
OLD_ROOT='/tmp/efe-leact-warm96-rebuild-v2/'+OLD
REBUILT_ROOT='/mnt/amlfs-03/home/rpunamiya/leact/studies/efe-leact-rebuilt-20261030-v1/'+OLD
ARMS=[
 {'id':'leact-rebuilt','label':'Warmstart 96 + LeAct rebuilt · original prompt','directory':'training-evaluation-v1/leact-rebuilt','direct':False},
 {'id':'base-direct','label':'Base Qwen · direct-control prompt','directory':'direct-control-v1/students/base-direct','direct':True},
 {'id':'warm96-direct','label':'Warmstart 96 · direct-control prompt','directory':'direct-control-v1/students/warm96-direct','direct':True},
]


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    DEST.mkdir(exist_ok=True)
    original=BUILD_ROOT/'efe-training-evaluation-20261029-v1/warm96'
    template=read(original/'spec.json')
    pilot_root=ROOT/'direct-control-v1/pilot/public-episodes'
    pilot=read(pilot_root/'manifest.json')
    assert len(pilot['episodes'])==6 and all(r['candidate_episode'] for r in pilot['episodes'])
    for arm in ARMS:
        build=BUILD_ROOT/('efe-native-followups-20261030-v1-'+arm['id']);arm['build']=str(build)
        local=ROOT/arm['directory'];local.mkdir(parents=True,exist_ok=True)
        if (build/'source-manifest.json').exists():continue
        build.mkdir(exist_ok=True)
        spec={**template,'model_id':arm['id'],'agent_label':arm['label'],'agent_slug':'qwen-'+arm['id'],
            'export_role':'training_'+arm['id'].replace('-','_'),'reuse_system_packages':False,
            'local_results_directory':arm['directory'],'archive_root':LANE+'/results',
            'purpose':'Native rollout followup; no training or checkpoint selection on these outcomes.'}
        if arm['direct']:
            tasks=[]
            for row in pilot['episodes']:
                path=pilot_root/row['id']/'trace.json';assert sha(path)==row['trace_sha256']
                trace=read(path);task=next(t for t in template['tasks'] if t['task']==row['task'])
                initial=next(e['state_sha'] for e in trace['simulator_events'] if e.get('ev')=='start')
                tasks.append({**task,'teacher_initial_state_sha256':initial,
                    'source_teacher_episode':'leact-efe-direct-pilot-v1-1/'+row['job'],
                    'teacher_trace_sha256':row['trace_sha256']})
            spec.update(tasks=tasks,instruction_suffix_file=LANE+'/method.txt',
                instruction_suffix_sha256=sha(BASE/'cluster/efe_direct_control_prompt.txt'),
                protocol_version='direct-control-v1-qwen-legacy-vision',
                protocol_label='Direct-control instruction suffix applied to the frozen native student deployment; weights, tools, image processing, sampling and native grading retained. Six training-task diagnostic; separate from held-out scores.',
                scope='Matched six-task direct-control prompt diagnostic. One attempt per checkpoint/task, seed 0. No new training.',
                comparison_to_development='Only task membership and the recorded direct-control method suffix change. Legacy image preprocessing and local context compaction are retained; parity must be verified before using this deployment contract for new SFT.')
            if arm['id']=='base-direct':
                base=read(BUILD_ROOT/'efe-qwen-base-20261026-v4/spec.json')
                spec.update(student_checkpoint=base['student_checkpoint'],
                    student_manifest=str(Path(base['student_checkpoint']).parent/'checkpoint-manifest.json'),
                    student_revision=base['student_revision'])
        else:
            ready=read(ROOT/'training-pilot/leact-warm96-rebuild-v2/portable-ready.json')
            spec.update(student_checkpoint=REBUILT_ROOT+'/model',student_manifest=REBUILT_ROOT+'/model/checkpoint-manifest.json',
                student_revision=ready['revision'],
                scope='21-task evaluation of the rebuilt checkpoint. Exact retained data and schedule; serialized weight hashes differ from the original, so retain a distinct model label.',
                recovery_of=OLD,recovery_reason='Training finished and weights were collected. Evaluate on a fresh allocation with a complete immutable mounted spec; never replace a mounted file.')
        write(build/'spec.json',spec)
        files=[{'localpath':'spec.json','path':LANE+'/spec.json'}]
        for name in ['efe_native_pilot.py','student.py','pilot.py','efe_budget_gateway.py','export_efe_viewer.py','worker.py']:
            text=(original/name).read_text()
            if name=='pilot.py':
                # Current pilot preserves original native settings and adds
                # optional suffix forwarding plus valid timeout classification.
                text=(BASE/'cluster/efe_astra_pilot.py').read_text().replace('port = 18861','port = 19861')
                text=text.replace('/tmp/efe-budget-gateway.py',LANE+'/efe_budget_gateway.py')
                text=text.replace("for _ in range(60):\n            if (runtime.ROOT / 'collected')", "for _ in range(4320):\n            if (runtime.ROOT / 'collected')")
            if name=='efe_budget_gateway.py':text=(BASE/'cluster/efe_budget_gateway.py').read_text()
            if name=='worker.py':
                text=text.replace("if name == 'system_packages':", "if name == 'system_packages' and spec.get('reuse_system_packages', True):")
                text=text.replace("'scope':'Closed-loop student rollouts on the 21 exact training tasks. One attempt per checkpoint/task. No new training. These are not held-out scores.'", "'scope':spec['scope']")
                # Save an exact forwarded student interface for the next data
                # extraction. Inputs contain no future target observations.
                before="        if name == 'results.json':\n            publish_results(value)"
                after="""        if name == 'results.json':
            publish_results(value)
            destination = archive/'student-prompt-requests'
            destination.mkdir(exist_ok=True)
            if not (archive/'student-prompt-manifest.json').exists():
                for trial in value.get('trials', []):
                    requests = archive/'requests'/trial['job']
                    for record_path in sorted(requests.glob('*.json')):
                        record = json.loads(record_path.read_text())
                        request_path = record_path.with_suffix('.request.json.gz')
                        if record.get('http_status') != 200 or record.get('status') != 'completed' or not request_path.exists():
                            continue
                        saved = {}
                        for source in [record_path, request_path]:
                            shutil.copyfile(source,destination/source.name)
                            saved[source.name] = {'sha256':sha(source),'bytes':source.stat().st_size}
                        write(archive/'student-prompt-manifest.json',{'files':saved,'source_job':trial['job'],
                            'student_revision':spec['student_revision'],
                            'instruction_suffix_sha256':record.get('instruction_suffix_sha256')})
                        break
                    if (archive/'student-prompt-manifest.json').exists():break"""
                if text.count(before)!=1:raise ValueError('Worker hook changed')
                text=text.replace(before,after)
            (build/name).write_text(text);compile(text,str(build/name),'exec')
            files.append({'localpath':name,'path':LANE+'/'+name})
        if arm['direct']:
            shutil.copyfile(BASE/'cluster/efe_direct_control_prompt.txt',build/'method.txt');files.append({'localpath':'method.txt','path':LANE+'/method.txt'})
        shutil.copyfile(BASE/'cluster/efe_eval_fresh_bootstrap.py',build/'bootstrap.py');files.append({'localpath':'bootstrap.py','path':'/tmp/efe_eval_bootstrap.py'})
        w=yaml.safe_load((BUILD_ROOT/'efe-training-warm96-resume-20261030-v4/workflow.yaml').read_text())
        w['workflow']['name']='leact-efe-followup-'+arm['id']+'-v1';w['workflow']['timeout']={'queue_timeout':'20m','exec_timeout':'24h'}
        w['workflow']['tasks'][0].update(command=['python3'],args=['-u','/tmp/efe_eval_bootstrap.py'],files=files)
        (build/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False))
        assert len({sha(build/f['localpath']) for f in files})==len(files)
        freeze_sources(build)
    write(DEST/'plan.json',{'created_utc':now(),'arms':ARMS,'node_cap':node_limit(),
        'scope':'Restore interrupted LeAct evaluation and isolate the direct-control prompt using base and existing warmstart checkpoints before new curated SFT.'})
    return ARMS


def collect_arm(arm,workflow):
    dest=ROOT/arm['directory'];root=arm.get('remote_archive',LANE+'/results/'+workflow)
    reader=arm.get('host_workflow',workflow)
    collect(reader,root,dest,['worker-start.json','worker-complete.json','progress.json','results.json','complete.json',
        'behavior.json','student-serving.json','student-preflight.json','student-serving-failure.json','setup-error.json',
        'public-episodes/manifest.json','student-prompt-manifest.json','retained-manifest.json'])
    for row in read(dest/'public-episodes/manifest.json',{'episodes':[]})['episodes']:
        folder=dest/'public-episodes'/row['id']
        def valid():return all((folder/n).is_file() and sha(folder/n)==row[k]
            for n,k in [('trace.json','trace_sha256'),('replay.mp4','video_sha256')])
        if not valid():download(reader,root+'/public-episodes/'+row['id'],dest/'public-episodes',timeout=120)
        if not valid():raise ValueError('Student episode checksum mismatch')
    prompts=read(dest/'student-prompt-manifest.json',{})
    if prompts:
        def valid_prompt():return all((dest/'student-prompt-requests'/name).is_file() and sha(dest/'student-prompt-requests'/name)==record['sha256'] for name,record in prompts['files'].items())
        if not valid_prompt():download(reader,root+'/student-prompt-requests',dest,timeout=120)
        if not valid_prompt():raise ValueError('Student deployment request checksum mismatch')
    retained=read(dest/'retained-manifest.json',{})
    if retained and arm.get('retained_storage')=='durable':
        expected={row['durable_file']:row['retained_sha256'] for row in retained['episodes']}
        digest=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
        if read(dest/'retained-verification.json',{}).get('manifest_sha256')!=digest:
            check=remote(reader,"from pathlib import Path\nimport hashlib,json\nexpected="+repr(expected)+"\nverified={}\nfor name,wanted in expected.items():\n h=hashlib.sha256()\n with Path(name).open('rb') as f:\n  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)\n verified[name]=h.hexdigest()==wanted\nprint('EFE_CAPTURE '+json.dumps({'verified':verified}),flush=True)")
            if check['verified']!={p:True for p in expected}:raise ValueError('Durable action bundle checksum mismatch')
            write(dest/'retained-verification.json',{'manifest_sha256':digest,'verified_utc':now(),
                'files':expected,'all_files_verified':True,'storage':'Shared filesystem; exact bundles retained without duplicating image payloads on the local dashboard machine.'})
    elif retained:
        def valid_retained():
            return all((dest/'retained'/row['retained_file']).is_file() and
                sha(dest/'retained'/row['retained_file'])==row['retained_sha256']
                for row in retained['episodes'])
        if not valid_retained():download(reader,root+'/retained',dest,timeout=300)
        if not valid_retained():raise ValueError('Complete action/observation bundle checksum mismatch')
    if arm.get('require_retained') and read(dest/'complete.json'):
        expected={t['job'] for t in read(dest/'results.json',{}).get('trials',[])}
        if expected!={r['job'] for r in retained.get('episodes',[])}:
            raise ValueError('Refusing release before complete action evidence is collected.')
    if read(dest/'complete.json') and not arm.get('remote_archive'):
        remote(workflow,"from pathlib import Path\nPath("+repr(LANE+'/runtime/collected')+").touch()\nprint('EFE_CAPTURE {\"ack\":true}',flush=True)")
    return read(dest/'worker-complete.json',{})


def verified_native_completion(arm):
    """Recognize complete native grades when the final wrapper receipt was missed."""
    dest=ROOT/arm['directory'];complete=read(dest/'complete.json',{})
    results=read(dest/'results.json',{});behavior=read(dest/'behavior.json',{})
    trials=results.get('trials',[]);planned=complete.get('planned_trials')
    if not planned or complete.get('error') or complete.get('completed_trials')!=planned:
        return {}
    if len(trials)!=planned or len({t['task'] for t in trials})!=planned:
        return {}
    if any(t.get('status')!='graded' or t.get('score') not in (0,1) for t in trials):return {}
    scores={t['task']:t['score'] for t in trials}
    if scores!={t['task']:t['score'] for t in behavior.get('trials',[])}:return {}
    manifest=read(dest/'public-episodes/manifest.json',{})
    if {r['task'] for r in manifest.get('episodes',[])}!=set(scores):return {}
    for row in manifest['episodes']:
        folder=dest/'public-episodes'/row['id']
        for name,key in [('trace.json','trace_sha256'),('replay.mp4','video_sha256')]:
            if not (folder/name).is_file() or sha(folder/name)!=row[key]:return {}
    record={'success':True,'graded':planned,'successes':sum(scores.values()),
        'verification':'Collected native completion, exact task grades, behavior records, and checksummed public traces/videos agree. Final worker-wrapper receipt was not collected.',
        'evidence_sha256':{n:sha(dest/n) for n in ['complete.json','results.json','behavior.json','public-episodes/manifest.json']},
        'verified_utc':now()}
    write(dest/'verified-completion.json',record)
    return record


def preserve_and_release():
    dest=ROOT/'training-pilot/leact-warm96-rebuild-v2'
    if (dest/'old-allocation-released.json').exists():return
    state=command(['osmo','workflow','query',OLD,'--format-type','json'])
    if state['status']!='RUNNING':return
    collect(OLD,OLD_ROOT,dest,['durable-copy-progress.json','durable-copy-error.json','durable-model-ready.json','durable-checkpoint-ready.json'])
    if read(dest/'durable-checkpoint-ready.json',{}).get('all_files_verified'):
        receipt=command(['osmo','workflow','cancel',OLD,'--message','Training complete; model and Adam state preserved with checksums. Native evaluation continues in a fresh workflow after the mounted-spec handoff failure.','--format-type','json'])
        write(dest/'old-allocation-released.json',{'updated_utc':now(),'receipt':receipt})


def monitor():
    arms=prepare();lock=(DEST/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from ..embodied_first_exam.report import publish
    while True:
        try:
            preserve_and_release()
            active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
            rows=[];done=0
            for arm in arms:
                dest=ROOT/arm['directory'];control=DEST/arm['id'];control.mkdir(exist_ok=True)
                state_build=read(control/'active-build.json',{})
                build=Path(state_build.get('build',arm['build']))
                receipt=read(build/'submission.json')
                if not receipt:
                    if arm['id']=='leact-rebuilt' and not read(ROOT/'training-pilot/leact-warm96-rebuild-v2/durable-model-ready.json',{}).get('all_files_verified'):
                        rows.append({**arm,'status':'waiting_for_verified_shared_weights'});continue
                    if reserved_nodes(active)>=node_limit():rows.append({**arm,'status':'waiting_for_compute_slot'});continue
                    receipt=submit(build,priority='HIGH');active.append({'name':receipt['name'],'status':'PENDING'})
                state=command(['osmo','workflow','query',receipt['name'],'--format-type','json']);write(control/'scheduler.json',state)
                complete=read(dest/'worker-complete.json',{})
                if state['status']=='RUNNING' and not complete:
                    complete=collect_arm(arm,receipt['name'])
                if complete:
                    done+=1
                    if not complete.get('success'):write(control/'experiment-error.json',complete)
                elif state['status'] in {'FAILED_QUEUE_TIMEOUT','FAILED_PREEMPTED','FAILED_START_ERROR','FAILED_START_TIMEOUT'} and not (dest/'worker-start.json').exists():
                    retry_number=int(state_build.get('retry_number',0))+1
                    retry=Path(arm['build']+'-low'+str(retry_number));retry.mkdir(exist_ok=True)
                    if not (retry/'source-manifest.json').exists():
                        for name in read(Path(arm['build'])/'source-manifest.json'):shutil.copyfile(Path(arm['build'])/name,retry/name)
                        w=yaml.safe_load((retry/'workflow.yaml').read_text());w['workflow']['name']+='-low'+str(retry_number);w['workflow']['timeout']['queue_timeout']='12h'
                        (retry/'workflow.yaml').write_text(yaml.safe_dump(w,sort_keys=False));freeze_sources(retry)
                    submit(retry,priority='LOW');write(control/'active-build.json',{'build':str(retry),'retry_number':retry_number})
                elif state['status'].startswith('FAILED') or state['status']=='COMPLETED':
                    write(control/'experiment-error.json',{'scheduler':state['status'],'detail':'Ended without verified experiment completion; preserve outcomes before resuming.'})
                behavior=read(dest/'behavior.json',{})
                rows.append({**arm,'workflow':receipt['name'],'status':state['status'],
                    'graded':behavior.get('graded',0),'successes':behavior.get('successes',0),
                    'planned':len(read(Path(arm['build'])/'spec.json')['tasks']),
                    'control_steps':sum(t['control_steps'] for t in behavior.get('trials',[])),
                    'progress':read(dest/'progress.json',{}),'complete':complete,
                    'serving_caveat':'Legacy image preprocessing retained to isolate the prompt change; new curated SFT requires a separate parity check.' if arm['direct'] else 'Rebuilt weights differ from original; separate model label.'})
            write(DEST/'status.json',{'updated_utc':now(),'arms':rows,'complete':done==len(arms)})
            publish_status();publish_viewer()
            from ..embodied_first_exam.direct_control_viewer import build as build_viewer
            build_viewer();publish()
            if done==len(arms):return
        except Exception as error:write(DEST/'monitor-error.json',{'error':str(error)[-1800:],'updated_utc':now()})
        time.sleep(30)


def start():
    DEST.mkdir(exist_ok=True)
    with (DEST/'monitor.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','reproductions.leact.cluster.launch_efe_native_followups','monitor'],
            cwd=BASE.parents[1],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(DEST/'monitor-launch.json',{'pid':p.pid,'updated_utc':now()})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','start','monitor']);args=parser.parse_args()
    if args.action=='monitor':monitor()
    else:
        print(json.dumps(prepare()))
        if args.action=='start':start()
