"""Retain complete public action/observation evidence before releasing a pilot node."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def mirror_evidence(archive, spec, workflow, name, value):
    """Keep grades and replay artifacts recoverable after a collector outage."""
    if not spec.get('action_durable_root'):
        return
    durable = Path(spec['action_durable_root'])/workflow
    if name in {'results.json','complete.json','progress.json','setup-error.json','native-config.json','spec.json',
                'student-preflight.json','student-serving.json','student-serving-failure.json'}:
        write(durable/name,value)
    if name not in {'results.json','complete.json'}:
        return
    for source in (archive/'public-episodes').rglob('*'):
        if not source.is_file():
            continue
        target = durable/source.relative_to(archive)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file() and sha(target)==sha(source):
            continue
        temporary=target.with_suffix(target.suffix+'.copy.tmp')
        shutil.copyfile(source,temporary)
        if sha(temporary)!=sha(source):raise ValueError('Durable replay checksum mismatch.')
        temporary.replace(target)
    for filename in ['behavior.json','retained-manifest.json']:
        source=archive/filename
        if source.exists():write(durable/filename,json.loads(source.read_text()))
    if name=='complete.json':
        expected=['results.json','complete.json','retained-manifest.json','public-episodes/manifest.json']
        available={filename:sha(durable/filename) if (durable/filename).exists() else None for filename in expected}
        write(durable/'durable-complete.json',{'workflow':workflow,'completed_utc':datetime.now(timezone.utc).isoformat(),
            'results_sha256':available['results.json'], 'complete_sha256':available['complete.json'],
            'retained_manifest_sha256':available['retained-manifest.json'],
            'public_manifest_sha256':available['public-episodes/manifest.json'],
            'complete_public_evidence_verified':not value.get('error') and all(available.values()),
            'error':value.get('error'),'missing_evidence':[name for name,digest in available.items() if digest is None]})


def retain_trials(archive, spec, results, workflow):
    manifest_path = archive/'retained-manifest.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = {row['job']:row for row in old.get('episodes', [])}
    root = archive/'retained'; root.mkdir(exist_ok=True)
    for trial in results.get('trials', []):
        job = trial['job']
        if job in entries:
            continue
        requests = archive/'requests'/job
        records = [json.loads(p.read_text()) for p in sorted(requests.glob('*.json'))]
        complete = [r for r in records if r.get('http_status') == 200 and r.get('status') == 'completed']
        trajectories = list((archive/'jobs'/job).glob('*/agent/trajectory.json'))
        contracts = list((archive/'jobs'/job).glob('*/agent/action-contract.json'))
        trajectory = json.loads(trajectories[0].read_text()) if len(trajectories) == 1 else {}
        steps = [s for s in trajectory.get('steps', []) if s.get('source') == 'agent']
        targets = []
        by_sha = {r['request_sha256']:r for r in complete}
        for step in steps:
            record = by_sha.get(step.get('request_sha256'))
            executed = any(json.loads(r['content']).get('execution_completed') is True
                for r in step.get('observation', {}).get('results', []) if r.get('content'))
            targets.append({'step_id':step['step_id'], 'request_sha256':step.get('request_sha256'),
                'request_record_id':record.get('id') if record else None,
                'complete_request_and_execution':bool(record and executed),
                'exclusion_reason':None if record and executed else 'No completed matching response and executed command.'})
        reasons = []
        if spec['fixed_action_mode'] != 'task': reasons.append('movement_diagnostic_excluded')
        if trial.get('status') != 'graded' or trial.get('score') != 1: reasons.append('no_native_task_success')
        if trial.get('model_errors'): reasons.append('model_transport_errors')
        if len(contracts) != 1 or len(trajectories) != 1: reasons.append('missing_contract_or_trajectory')
        if len(steps) != len(complete): reasons.append('incomplete_response_trace_alignment')
        if any(not (requests/(r['id']+'.request.json.gz')).is_file() for r in complete):
            reasons.append('missing_forwarded_model_inputs')
        if not any(t['complete_request_and_execution'] for t in targets): reasons.append('no_executed_target')
        bundle = root/(job+'.tar.gz')
        temporary = bundle.with_suffix('.tmp')
        sources = [p for p in requests.glob('*') if p.is_file() and
                   (p.name.endswith('.request.json.gz') or p.name.endswith('.source-request.json.gz') or p.suffix == '.json')]
        for p in (archive/'jobs'/job).rglob('*'):
            relative = str(p.relative_to(archive))
            if p.is_file() and (p.name == 'result.json' or
                '/agent/' in relative and p.name in {'trajectory.json','action-contract.json','action-metrics.json','native-spec.json','console.txt','artifact-download-errors.json'} or
                '/verifier/' in relative and p.suffix in {'.json','.jsonl'}):
                sources.append(p)
        with tarfile.open(temporary, 'w:gz') as tar:
            for source in sorted(sources):
                tar.add(source, arcname=str(source.relative_to(archive)))
        temporary.replace(bundle)
        row = {'task':trial['task'], 'job':job, 'status':trial['status'], 'score':trial['score'],
            'retained_file':bundle.name, 'retained_sha256':sha(bundle), 'retained_bytes':bundle.stat().st_size,
            'request_count':len(records), 'completed_requests':len(complete), 'targets':targets,
            'candidate_pending_scene_and_native_event_audit':not reasons, 'exclusion_reasons':reasons,
            'files':{str(p.relative_to(archive)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sources}}
        if spec.get('action_durable_root'):
            durable = Path(spec['action_durable_root'])/workflow/'retained'/bundle.name
            durable.parent.mkdir(parents=True, exist_ok=True)
            staging = durable.with_suffix('.copy.tmp')
            shutil.copyfile(bundle, staging)
            if sha(staging) != row['retained_sha256']:
                raise ValueError('Durable action bundle checksum mismatch.')
            staging.replace(durable)
            row['durable_file'] = str(durable)
        entries[job] = row
    value = {'workflow':workflow, 'updated_utc':datetime.now(timezone.utc).isoformat(),
        'protocol':spec['protocol_version'], 'episodes':list(entries.values()),
        'admission_note':'Candidates still require scene isolation, exact native submit matching, interface parity and complete nontruncated target checks. Invalid commands remain context and are excluded as targets. No automatic SFT admission.'}
    write(manifest_path, value)
    if spec.get('action_durable_root'):
        write(Path(spec['action_durable_root'])/workflow/'retained-manifest.json', value)
