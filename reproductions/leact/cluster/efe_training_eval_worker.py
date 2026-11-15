"""Run a frozen native student on the exact admitted training task instances."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(lane, spec):
    """Clone the already-tested environment; keep the original environment intact."""
    original = Path('/tmp/efe-native/emimic')
    environment = lane / 'emimic'
    if not environment.exists():
        shutil.copytree(original, environment, symlinks=True)
        for script in (environment / 'bin').iterdir():
            if script.is_symlink() or not script.is_file():
                continue
            raw = script.read_bytes()
            if raw.startswith(b'#!'):
                script.write_bytes(raw.replace(str(original).encode(), str(environment).encode()))
    paths = list(environment.glob('lib/python*/site-packages/harbor/agents/installed/codex.py'))
    if len(paths) != 1:
        raise ValueError('Cannot identify the copied Harbor agent')
    path = paths[0]
    routing = spec['model_routing']
    if sha(path) == routing['patched_file_sha256']:
        raw = path.read_bytes().replace(routing['new_line'].encode(), routing['old_line'].encode())
        if hashlib.sha256(raw).hexdigest() != routing['upstream_file_sha256']:
            raise ValueError('Copied Harbor routing restoration did not match pinned source')
        path.write_bytes(raw)
    if sha(path) != routing['upstream_file_sha256']:
        raise ValueError('Unexpected copied Harbor source')
    env = {**os.environ, 'VIRTUAL_ENV': str(environment), 'PATH': str(environment / 'bin') + ':' + os.environ['PATH'],
           'EFE_SERVING_PYTHON': sys.executable, 'PYTHONUNBUFFERED': '1'}
    os.execve(str(environment / 'bin/python'), [str(environment / 'bin/python'), str(Path(__file__)), '--run'], env)


def main():
    lane = Path(__file__).resolve().parent
    spec = json.loads((lane / 'spec.json').read_text())
    if '--run' not in sys.argv:
        prepare(lane, spec)
    import efe_native_pilot as runtime
    from export_efe_viewer import export_episode
    import student
    archive = Path(spec['archive_root']) / runtime.WORKFLOW
    archive.mkdir(parents=True, exist_ok=True)
    public = archive / 'public-episodes'
    public.mkdir(exist_ok=True)
    original_write, original_run = runtime.write, runtime.run
    tasks = {row['task']: row for row in spec['tasks']}
    exports = {}

    def publish_results(results):
        rows = []
        for trial in results.get('trials', []):
            if trial['status'] != 'graded':
                continue
            task = tasks[trial['task']]
            if trial['task'] not in exports:
                row = export_episode(dict(trial, role=spec['export_role']), public)
                path = public / row['id'] / 'trace.json'
                trace = json.loads(path.read_text())
                initial = next((e.get('state_sha') for e in trace['simulator_events'] if e.get('ev') == 'start'), None)
                trace.update(teacher_initial_state_sha256=task['teacher_initial_state_sha256'],
                    initial_state_matches_teacher=bool(initial and initial == task['teacher_initial_state_sha256']),
                    source_teacher_episode=task['source_teacher_episode'],
                    evaluation_only=True, student_revision=spec['student_revision'])
                path.write_text(json.dumps(trace, ensure_ascii=False) + '\n')
                row.update(trace_sha256=sha(path), initial_state_matches_teacher=trace['initial_state_matches_teacher'],
                           source_teacher_episode=task['source_teacher_episode'])
                exports[trial['task']] = row
            row = exports[trial['task']]
            trace = json.loads((public / row['id'] / 'trace.json').read_text())
            metrics = trial['resource_metrics']
            rows.append({'task':trial['task'], 'suite':trial['suite'], 'score':trial['score'],
                'control_steps':metrics.get('control_steps', metrics.get('control_steps_total', metrics.get('n_actions',0))),
                'simulator_calls':metrics.get('calls',0), 'model_requests':trial['model_requests'],
                'tool_calls':sum(len(s['tool_calls']) for s in trace['steps']),
                'most_repeated_call':trace['repeated_calls'][0] if trace['repeated_calls'] else None,
                'initial_state_matches_teacher':row['initial_state_matches_teacher'],
                'elapsed_s':metrics.get('elapsed_s'), 'public_episode':row['id']})
        write(public / 'manifest.json', {'created_utc':now(), 'episodes':list(exports.values()),
            'student_protocol':spec['protocol_label'], 'partition':'training_evaluation',
            'student_revision':spec['student_revision'], 'planned':len(tasks), 'new_rollouts':len(exports)})
        write(archive / 'behavior.json', {'updated_utc':now(), 'model':spec['model_id'],
            'planned':len(tasks), 'graded':len(rows), 'successes':sum(r['score'] for r in rows),
            'success_rate':sum(r['score'] for r in rows)/len(rows) if rows else None,
            'matched_initial_states':sum(r['initial_state_matches_teacher'] for r in rows), 'trials':rows,
            'scope':'Closed-loop student rollouts on the 21 exact training tasks. One attempt per checkpoint/task. No new training. These are not held-out scores.'})

    def recorded_write(name, value):
        original_write(name, value)
        if name == 'results.json':
            publish_results(value)

    def cached_run(name, argv, **kwargs):
        if name == 'system_packages':
            if not all(shutil.which(c) for c in ['git', 'curl', 'newuidmap', 'slirp4netns']):
                raise RuntimeError('Previously validated native system packages missing')
            return {'returncode':0, 'detail':'Reused installed packages from the completed native evaluation on this allocation.'}
        return original_run(name, argv, **kwargs)

    runtime.write, runtime.run = recorded_write, cached_run
    receipt = {'started_utc':now(), 'model':spec['model_id'], 'success':False}
    write(archive / 'worker-start.json', {**receipt, 'pid':os.getpid(), 'python':sys.executable})
    try:
        student.main()
        results = json.loads((archive / 'results.json').read_text())
        valid = [r for r in results.get('trials', []) if r['status'] == 'graded']
        receipt.update(success=len(valid) == len(tasks), graded=len(valid),
            successes=sum(r['score'] for r in valid), error=results.get('setup_error'))
    except Exception as exc:
        receipt['error'] = str(exc)
        raise
    finally:
        receipt['completed_utc'] = now()
        write(archive / 'worker-complete.json', receipt)


if __name__ == '__main__':
    main()
