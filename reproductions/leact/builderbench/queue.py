"""Submit the approved H100 roles, feed their dependency, and publish live results."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import zlib

from ..data import file_sha256, fingerprint, write_json
from .student_data import freeze

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
STUDY = '/mnt/amlfs-03/home/rpunamiya/leact/studies/builderbench-gpt6-student-20261024-v1'
REMOTE = '/mnt/amlfs-07/shared/egoverse/rpunamiya/leact/runs/'
ENV = dict(os.environ, PYTHONPATH=str(PROJECT), PYTHONDONTWRITEBYTECODE='1')


def stopped_by_user():
    path=BASE/'artifacts/builderbench-stop.json'
    return path.exists() and json.loads(path.read_text()).get('stop_requested',False)


def utc():
    return datetime.now(timezone.utc).isoformat()


def command(argv, *, cwd=None, timeout=90):
    result = subprocess.run(argv, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=timeout, check=True)
    return json.loads(result.stdout[result.stdout.index('{'):])


def remote(workflow, script):
    import pexpect
    packed = base64.b64encode(zlib.compress(script.encode(), 9)).decode()
    entry = 'python3 -c ' + shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
    child = pexpect.spawn('osmo', ['workflow','exec',workflow,'train','--entry',entry],
                          encoding='utf-8', timeout=90, maxread=65536, echo=False)
    try:
        child.expect('BUILDER_DEPENDENCY_READY', timeout=120)
        child.expect(pexpect.EOF, timeout=30)
        child.close()
        if child.exitstatus not in (0,None):
            raise RuntimeError('Dependency publication failed')
    finally:
        if child.isalive(): child.close(force=True)


def iteration_dependency(build, workflow):
    receipt = build/'iteration-uploaded.json'
    if receipt.exists(): return
    status_path = BASE/'artifacts/builderbench/iteration-expert-status.json'
    if not status_path.exists(): return
    status = json.loads(status_path.read_text())
    if status['status'] != 'completed': return
    output = BASE/'artifacts/builderbench/corpus-gpt6-iteration-v1'
    spec = freeze(BASE/'artifacts/builderbench/teacher-gpt6-iteration-v1',
        BASE/'cluster/.build/builderbench-20261023/reference', output,
        parent=BASE/'artifacts/builderbench/corpus-gpt6-warmstart-v1')
    archive = build/'iteration-input.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in ['manifest.json','states.jsonl']:
            data=(output/name).read_bytes(); member=tarfile.TarInfo('iteration-corpus/'+name)
            member.size=len(data);tar.addfile(member,io.BytesIO(data))
    digest=file_sha256(archive)
    subprocess.run([sys.executable,'-m','reproductions.leact.cluster.upload_payload',workflow,str(archive),
                    '--sha256',digest,'--task','train'],env=ENV,check=True,timeout=300)
    script = f'''from pathlib import Path
import hashlib,json,tarfile
archive=Path('/osmo/run/workspace/payload.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={digest!r}
root=Path({STUDY!r})
with tarfile.open(archive) as tar:
    assert set(tar.getnames())=={{'iteration-corpus/manifest.json','iteration-corpus/states.jsonl'}}
    for name in tar.getnames():
        data=tar.extractfile(name).read();p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists(): assert p.read_bytes()==data
        else: p.write_bytes(data)
ready=root/'iteration-corpus/ready.json'
ready.write_text(json.dumps({{'states_sha256':{spec['states_sha256']!r}}}))
print('BUILDER_DEPENDENCY_READY',flush=True)
'''
    remote(workflow,script)
    write_json(receipt,dict(updated_utc=utc(),states=spec['states'],states_sha256=spec['states_sha256'],archive_sha256=digest))


def publish():
    from .speed import publish as speed_publish
    from ..tracking.export import snapshot
    from ..tracking.server import ingest
    builds = sorted((BASE/'cluster/.build').glob('builderbench-gpt6-*-20261024-v1'))
    roles, evaluations, speed_records, previews, checkpoints = {}, {}, {}, [], {}
    fixed_report = None
    for build in builds:
        role = ('evaluate' if 'student-evaluate' in build.name else 'train' if 'student-train' in build.name
                else 'fixed' if 'student-fixed' in build.name else None)
        if role is None: continue
        state_path=build/'status-current.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        root=build/'collected/latest'
        if (root/'fixed-report.json').exists():
            fixed_report=json.loads((root/'fixed-report.json').read_text())
        progress=root/f'{role}-progress.json'
        roles[role]=dict(workflow=state.get('name'),scheduler_status=state.get('status','Preparing'),
                        progress=json.loads(progress.read_text()) if progress.exists() else {})
        for path in root.glob('ready/*.json'):
            value=json.loads(path.read_text());checkpoints[value['id']]=value
        for path in root.glob('evaluations/*/metrics.json'):
            value=json.loads(path.read_text());old=evaluations.get(path.parent.name)
            if old is None or len(value['evaluated_seeds'])>=len(old['evaluated_seeds']):
                evaluations[path.parent.name]=value
        for path in root.glob('evaluations/*/speed.json'):
            manifest_path=path.with_name('manifest.json')
            if not manifest_path.exists():continue
            value=json.loads(path.read_text());manifest=json.loads(manifest_path.read_text())
            if value['run_sha256']!=fingerprint(manifest):raise ValueError('Imported timing report manifest mismatch')
            old=speed_records.get(path.parent.name)
            if old is None or value['completed_rollouts']>=old['completed_rollouts']:
                speed_records[path.parent.name]=value
        for path in root.glob('evaluations/*/preview.json'):
            previews.extend(dict(stage=path.parent.name,**r) for r in json.loads(path.read_text()))
    record=dict(updated_utc=utc(),roles=roles,evaluations=list(evaluations.values()),examples=previews[-12:],
        checkpoints=list(checkpoints.values()),fixed_observation_speed=fixed_report,
        root=STUDY,autonomous_progression=True,
        initial_corpus=dict(tasks=23,held_out_tasks=28,eligible_action_steps=153,bc_examples=153,joint_examples=306),
        next='The training role automatically consumes the second expert corpus and runs LeAct plus matched ExIt; the evaluation role drains all checkpoint/seed jobs.')
    if stopped_by_user():
        stop=json.loads((BASE/'artifacts/builderbench-stop.json').read_text())
        record.update(autonomous_progression=False,stop=stop,
                      next='Stopped by user. Preserve completed results; focus exclusively on EFE.')
    write_json(BASE/'artifacts/builderbench/student-status.json',record)
    write_json(BASE/'artifacts/builderbench/imported-speed-reports.json',dict(reports=list(speed_records.values()),
        evidence='Checksum-collected reports computed from full immutable remote calls; each report checked against its collected manifest'))
    speed_publish(BASE)
    plan_path=BASE/'artifacts/experiment-plan.json';plan=json.loads(plan_path.read_text())
    summary='; '.join(f'{role}: {v["scheduler_status"]}, {v["progress"].get("phase","awaiting startup")}' for role,v in roles.items())
    if stopped_by_user():
        summary='Stopped by user; automatic launches disabled. Remote cancellation: '+stop.get('remote_cancellation_status','pending')
        plan['builderbench'].update(auto_submission_enabled=False,expert_collection_enabled=False,
                                    execution_status='stopped_by_user',stop=stop)
    plan['builderbench']['implementation_status']=summary
    plan['updated_utc']=utc()
    for row in plan['sequence']:
        if row['id']=='builderbench':row['status']=summary
    write_json(plan_path,plan)
    value=snapshot(BASE)
    ingest(BASE/'cluster/.build/tracking/experiments.sqlite3',value)
    from ..tracking.local_service import mirror_snapshot
    mirror_snapshot(value)


def main():
    if stopped_by_user():
        print('BuilderBench stopped by user; no workflow will be launched or resumed.',flush=True)
        return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--role',choices=['train','evaluate','fixed'],required=True)
    args=parser.parse_args();build=args.build.resolve()
    lock=(build/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    payload=build/'payload.tar.gz';digest=file_sha256(payload)
    assert json.loads((build/'cpu-validation.json').read_text())['returncode']==0
    flags=['--pool','groot-h100-01','--set-string','payload_sha256='+digest]
    if args.role=='fixed' and not (build/'submission.json').exists():
        training_build=BASE/'cluster/.build/builderbench-gpt6-student-train-20261024-v1'
        while True:
            if stopped_by_user():return
            path=training_build/'status-current.json'
            state=json.loads(path.read_text()) if path.exists() else {}
            complete=training_build/'collected/latest/train-complete.json'
            if state.get('status')=='COMPLETED' and complete.exists():break
            write_json(build/'dependency-status.json',dict(updated_utc=utc(),status='Waiting for training node to finish and release its allocation'))
            time.sleep(30)
    if (build/'submission.json').exists():
        submission=json.loads((build/'submission.json').read_text())
    else:
        assert not (build/'submission-intent.json').exists(), 'Reconcile the previous submission before another submit'
        budget_lock=(BASE/'cluster/.build/builderbench-submission.lock').open('a')
        fcntl.flock(budget_lock,fcntl.LOCK_EX)
        if stopped_by_user():return
        active=command(['osmo','workflow','list','--name','leact','--status','RUNNING','PENDING','WAITING','--format-type','json'])['workflows']
        from ..cluster.compute_budget import require_capacity
        require_capacity(active)
        result=subprocess.run(['osmo','workflow','validate','workflow.yaml',*flags],cwd=build,env=ENV,capture_output=True,text=True,check=True,timeout=90)
        (build/'validation.log').write_text(result.stdout+result.stderr)
        write_json(build/'submission-intent.json',dict(updated_utc=utc(),sha256=digest))
        submission=command(['osmo','workflow','submit','workflow.yaml',*flags,'--priority','HIGH','--format-type','json'],cwd=build)
        write_json(build/'submission.json',submission)
        write_json(build/'launch-record.json',{**submission,'priority':'HIGH','pool':'groot-h100-01','gpu_count':8,
            'payload_sha256':digest,'phase':'BuilderBench '+args.role})
        fcntl.flock(budget_lock,fcntl.LOCK_UN)
    workflow=submission['name'];collector=None
    print(json.dumps(dict(workflow=workflow,role=args.role)),flush=True)
    for _ in range(10000):
        if stopped_by_user():return
        try:
            state=command(['osmo','workflow','query',workflow,'--format-type','json'])
            write_json(build/'status-current.json',state)
            if state['status']=='RUNNING':
                if not (build/'upload-complete.json').exists():
                    with (build/'upload.log').open('a') as log:
                        subprocess.run([sys.executable,'-m','reproductions.leact.cluster.upload_payload',workflow,str(payload),
                            '--sha256',digest,'--task','train'],env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
                    write_json(build/'upload-complete.json',dict(workflow=workflow,sha256=digest,updated_utc=utc()))
                if collector is None or collector.poll() is not None:
                    with (build/'collector.log').open('a') as log:
                        collector=subprocess.Popen([sys.executable,'-u','-m','reproductions.leact.cluster.collect_results',
                            workflow,'--task','train','--source-root',REMOTE+workflow+'/source','--experiment-root',STUDY,
                            '--output',str(build/'collected'),'--builderbench-role',args.role,'--watch-seconds','600',
                            '--interval','20','--max-rollout-bytes','1000000'],env=ENV,stdout=log,stderr=subprocess.STDOUT)
                # The bootstrap must have extracted the original payload before its inbox is reused.
                if args.role=='train' and (build/'collected/latest/train-protocol.json').exists():
                    iteration_dependency(build,workflow)
            publish_lock=(BASE/'cluster/.build/builderbench-publish.lock').open('a')
            with publish_lock:
                fcntl.flock(publish_lock,fcntl.LOCK_EX);publish()
            write_json(build/'monitor-status.json',dict(updated_utc=utc(),workflow=workflow,status=state['status'],pid=os.getpid()))
            if state['status'].startswith('FAILED') or state['status']=='COMPLETED':
                if collector and collector.poll() is None:collector.wait(timeout=45)
                publish()
                quota = (state['status']=='FAILED_CANCELED' and state.get('cancelled_by')=='groot-idle-job-shutdown'
                         and 'GPU quota enforcement' in str(state.get('groups'))
                         and (args.role in ('evaluate', 'fixed') or not (build/'upload-complete.json').exists()))
                retry_path=build/'retry.json'
                retry=json.loads(retry_path.read_text()).get('attempt',0) if retry_path.exists() else 0
                if quota and retry<4:
                    delay=[300,600,1200,1800][retry]
                    archive=build/f'prior-{workflow}'
                    archive.mkdir(exist_ok=False)
                    for name in ['submission.json','submission-intent.json','launch-record.json','status-current.json','upload-complete.json']:
                        if (build/name).exists(): (archive/name).write_bytes((build/name).read_bytes())
                    write_json(retry_path,dict(attempt=retry+1,delay_seconds=delay,reason='Ordinary quota retry; evaluation resumes immutable saved episodes'))
                    time.sleep(delay)
                    if stopped_by_user():return
                    for name in ['submission.json','submission-intent.json','upload-complete.json']:(build/name).unlink(missing_ok=True)
                    os.execv(sys.executable,[sys.executable,'-u','-m','reproductions.leact.builderbench.queue',
                        '--build',str(build),'--role',args.role])
                return
        except Exception as error:
            print(json.dumps(dict(updated_utc=utc(),monitor_error=repr(error))),flush=True)
            write_json(build/'monitor-error.json',dict(updated_utc=utc(),error=repr(error)))
        time.sleep(30)


if __name__=='__main__':
    main()
