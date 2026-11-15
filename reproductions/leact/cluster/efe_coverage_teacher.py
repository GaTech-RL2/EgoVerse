"""Durable, split-audited collection with the shared constrained action bridge."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def audit_scenes(reference, spec):
    """One writer; resumable metadata-only audit, never a solution/oracle read."""
    from scene_identity import scene_identity
    root = Path(spec['coverage_root']) / 'scene-audit'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'writer.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return
    try:
        plan = json.loads((Path(__file__).parent / 'coverage-plan.json').read_text())
        if (root / 'result.json').exists():
            result = json.loads((root / 'result.json').read_text())
            if result['plan_sha256'] != spec['coverage_plan_sha256']:
                raise ValueError('Scene audit plan changed')
            return
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=reference, text=True).strip()
        if revision != spec['revision']:
            raise ValueError('Scene audit revision changed')
        receipts, errors = [], []
        # DexHand's published task families do not share the scene.py layout
        # assumed by the pilot canonicalizer. Defer the entire suite, including
        # its protected scenes, until that identity check is independently fixed.
        excluded_suites = ['dexhand']
        rows = [r for r in plan['tasks'] + plan['protected_tasks'] if r['suite'] not in excluded_suites]
        for index, row in enumerate(rows):
            cached = root / 'receipts' / (row['task'] + '.json')
            if cached.exists():
                receipt = json.loads(cached.read_text())
            else:
                names = ['task.toml', 'DIGEST']
                if row['suite'] in {'metaworldplus', 'robocasa'}:
                    names += ['environment/sim/instance.npz']
                elif row['suite'] == 'embodied-safety-exam':
                    names += ['environment/sim/instance.json']
                elif row['suite'] == 'dexhand':
                    names += ['environment/sim/instance.json', 'environment/sim/scene.py']
                else:
                    raise ValueError('Unaudited suite')
                try:
                    for name in names:
                        relative = row['path'] + '/' + name
                        dest = root / 'metadata' / relative
                        if not dest.exists():
                            raw = subprocess.check_output(['git', 'show', revision + ':' + relative],
                                                          cwd=reference, timeout=180)
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            temporary = dest.with_suffix(dest.suffix + '.tmp')
                            temporary.write_bytes(raw)
                            temporary.replace(dest)
                    receipt = {**scene_identity(root / 'metadata', row), 'partition':row['partition']}
                    write(cached, receipt)
                except Exception as error:
                    errors.append({'task':row['task'], 'partition':row['partition'], 'error':str(error)})
                    continue
            if receipt['task_toml_sha256'] != row['task_toml_sha256'] or receipt['upstream_digest'] != row['upstream_digest']:
                raise ValueError('Cached scene metadata changed')
            receipts.append(receipt)
            write(root / 'progress.json', {'updated_utc':now(), 'completed':index+1,
                                          'total':len(rows), 'errors':errors})
        protected = {}
        for row in receipts:
            if row['partition'] == 'protected':
                protected.setdefault((row['suite'], row['scene_sha256']), []).append(row['task'])
        collisions = [{'task':r['task'], 'protected_tasks':protected[(r['suite'], r['scene_sha256'])]}
                      for r in receipts if r['partition']=='train_candidates' and
                      (r['suite'], r['scene_sha256']) in protected]
        result = {'upstream_revision':revision, 'plan_sha256':spec['coverage_plan_sha256'],
                  'rows':receipts, 'collisions':collisions, 'errors':errors,
                  'excluded_suites':excluded_suites,
                  'exclusion_reason':'DexHand physical-scene metadata layout is not covered by the validated canonicalizer.',
                  'complete':not errors, 'updated_utc':now(),
                  'scope':'Immutable scene metadata only; no oracle policy or demonstration inspected.'}
        # A failed protected audit cannot silently certify any training scene.
        write(root / ('result.json' if not errors else 'incomplete.json'), result)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    import efe_native_pilot as runtime
    import student
    from efe_action_artifacts import retain_trials, mirror_evidence
    from export_efe_viewer import export_episode
    spec = json.loads((Path(__file__).parent / 'spec.json').read_text())
    archive = Path(spec['archive_root']) / runtime.WORKFLOW
    archive.mkdir(parents=True, exist_ok=True)
    durable = Path(spec['action_durable_root']) / runtime.WORKFLOW
    public = archive / 'public-episodes'
    public.mkdir(exist_ok=True)
    tasks = {r['task']:r for r in spec['tasks']}
    exports = {}
    original_write, original_setup = runtime.write, runtime.setup

    def publish(results):
        rows = []
        for trial in results.get('trials', []):
            if trial['status'] != 'graded':
                continue
            key = trial['job']
            if key not in exports:
                row = export_episode(dict(trial, role=spec['export_role']), public)
                path = public / row['id'] / 'trace.json'
                trace = json.loads(path.read_text())
                trace.update(training_candidate=not spec.get('coverage_evaluation',False),
                    evaluation_only=spec.get('coverage_evaluation',False),
                    collection_attempt=tasks[trial['task']]['collection_attempt'],
                    model_revision=spec['student_revision'])
                expected=tasks[trial['task']].get('teacher_initial_state_sha256')
                if expected:
                    actual=next((e.get('state_sha') for e in trace['simulator_events'] if e.get('ev')=='start'),None)
                    trace.update(teacher_initial_state_sha256=expected,initial_state_matches_teacher=actual==expected)
                path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
                row.update(trace_sha256=sha(path), collection_attempt=tasks[trial['task']]['collection_attempt'])
                if expected:row['initial_state_matches_teacher']=trace['initial_state_matches_teacher']
                exports[key] = row
            rows.append({'task':trial['task'], 'suite':trial['suite'], 'score':trial['score'],
                'collection_attempt':tasks[trial['task']]['collection_attempt'],
                'model_requests':trial['model_requests'], 'action_metrics':trial.get('action_metrics', {}),
                'public_episode':exports[key]['id'], 'resource_metrics':trial['resource_metrics']})
        write(public / 'manifest.json', {'created_utc':now(), 'episodes':list(exports.values()),
            'partition':'evaluation' if spec.get('coverage_evaluation') else 'train_candidates', 'planned':len(tasks), 'new_rollouts':len(exports),
            'student_protocol':spec['protocol_label'], 'student_revision':spec['student_revision']})
        write(archive / 'behavior.json', {'updated_utc':now(), 'model':spec['model_id'],
            'planned':len(tasks), 'graded':len(rows), 'successes':sum(r['score'] for r in rows),
            'trials':rows, 'scope':spec['scope']})

    def recorded_write(name, value):
        original_write(name, value)
        if name == 'results.json':
            retain_trials(archive, spec, value, runtime.WORKFLOW)
            publish(value)
        mirror_evidence(archive, spec, runtime.WORKFLOW, name, value)

    def setup(value):
        result = original_setup(value)
        if not spec.get('coverage_evaluation'):audit_scenes(result[0], spec)
        return result

    runtime.write, runtime.setup = recorded_write, setup
    receipt = {'started_utc':now(), 'success':False, 'workflow':runtime.WORKFLOW,
               'pid':os.getpid(), 'planned':len(tasks)}
    write(archive / 'worker-start.json', receipt)
    write(durable / 'worker-start.json', receipt)
    try:
        student.main()
        results = json.loads((archive / 'results.json').read_text())
        graded = [r for r in results.get('trials', []) if r['status']=='graded']
        receipt.update(success=len(graded)==len(tasks), graded=len(graded),
                       successes=sum(r['score'] for r in graded), error=results.get('setup_error'))
    except Exception as error:
        receipt.update(error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        receipt['completed_utc'] = now()
        write(archive / 'worker-complete.json', receipt)
        write(durable / 'worker-complete.json', receipt)


if __name__ == '__main__':
    main()
