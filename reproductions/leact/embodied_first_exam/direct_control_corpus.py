"""Provenance and scene admission for the growing direct-control collection."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from .report import ROOT, read, write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
    root = ROOT / 'direct-control-v1'
    plan = read(root / 'plan.json', {})
    scene_path = root / 'scene-audit/result.json'
    scenes = read(scene_path, {})
    scene_rows = {r['task']: r for r in scenes.get('rows', []) if r['partition'] == 'train_candidates'}
    collisions = {r['task']: r['protected_task'] for r in scenes.get('collisions', [])}
    accepted, excluded = [], []
    counts = Counter()
    for batch in plan.get('builds', []):
        folder = root / batch['name']
        results = read(folder / 'results.json', {'trials': []})
        trials = {r['job']: r for r in results['trials']}
        durable = read(folder / 'durable-retained.json', {})
        for row in read(folder / 'public-episodes/manifest.json', {'episodes': []})['episodes']:
            trial = trials.get(row['job'])
            if not trial:
                continue
            reason = []
            counts['attempts'] += 1
            counts['successes'] += trial.get('score') == 1 and trial.get('status') == 'graded'
            if not row.get('candidate_episode'):
                reason.append('Episode failed native success, transport-health, or method admission')
            if row['task'] not in scene_rows:
                reason.append('Scene audit pending')
            if row['task'] in collisions:
                reason.append('Physical scene overlaps protected task: ' + collisions[row['task']])
            receipt = durable.get(row['retained_file'])
            if not receipt or receipt['sha256'] != row['retained_sha256'] or receipt['bytes'] != row['retained_bytes']:
                reason.append('Durable raw bundle verification pending')
            audit_path = folder / 'public-episodes' / row.get('id', '') / 'action-audit.json'
            if not audit_path.is_file() or sha(audit_path) != row.get('action_audit_sha256'):
                reason.append('Executed-call audit verification pending')
            ident = results['workflow'] + '/' + row['job']
            if reason:
                excluded.append({'episode': ident, 'task': row['task'], 'reasons': reason})
                continue
            if trial['partition'] != 'train_candidates' or row['task'] not in batch['tasks']:
                raise ValueError('Unexpected task membership: ' + ident)
            accepted.append({'episode': ident, 'batch': batch['name'], 'trial': dict(trial, workflow=results['workflow']),
                'scene': scene_rows[row['task']], 'bundle': receipt,
                'action_audit': read(audit_path), 'action_audit_sha256': row['action_audit_sha256'],
                'action_audit_source': str(audit_path), 'eligible_tool_targets': row['eligible_targets']})
    prompt = root / 'students/base-direct/student-prompt-manifest.json'
    prompt_manifest = read(prompt, {})
    prompt_dir = prompt.parent / 'student-prompt-requests'
    prompt_valid = bool(prompt_manifest) and all(
        (prompt_dir / name).is_file() and sha(prompt_dir / name) == record['sha256']
        for name, record in prompt_manifest.get('files', {}).items())
    by_task = Counter(r['trial']['task'] for r in accepted)
    summary = {
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'phase': 'preparing_corpus', 'student_training_launched': False,
        'observed_attempts': counts['attempts'], 'native_successes': counts['successes'],
        'scene_admitted_episodes': len(accepted), 'scene_admitted_tasks': len(by_task),
        'eligible_tool_targets_before_request_grouping': sum(r['eligible_tool_targets'] for r in accepted),
        'training_candidate_scenes': len(scene_rows) - len(collisions),
        'scene_exclusions': scenes.get('collisions', []),
        'protected_scenes_verified': scenes.get('protected_scenes_reproduced', 0),
        'episodes_per_task': dict(sorted(by_task.items())),
        'tasks_with_two_successful_episodes': sum(n >= 2 for n in by_task.values()),
        'student_deployment_prompt_verified': prompt_valid,
        'pending_checks': ['Complete-context coverage', 'Storage and optimizer-memory preflight',
                           'Training/deployment image, token and target-likelihood parity'],
        'detail': f'{len(accepted)} successful episodes across {len(by_task)} tasks have verified raw bundles, executed-call audits and isolated scenes. '
                  f'{len(collisions)} of {len(scene_rows)} candidate scenes excluded for protected-scene overlap. '
                  'The growing corpus is being checked for complete context coverage and deployment parity.',
    }
    profile = read(root / 'preparation-profile/latest.json', {})
    if profile:
        summary['context_profile'] = profile
    curated = read(root / 'curated-warmstart-v1/status.json', {})
    warm = next((a for a in curated.get('arms', []) if a['id']=='curated-warm64'), None)
    if warm:
        summary['curated_warmstart'] = warm
        summary['student_training_launched'] = bool(warm.get('training_progress', {}).get('training_started'))
        summary['detail'] += ' First frozen 16-episode, 441-decision warmstart: ' + warm.get('progress', {}).get('detail', warm['status']) + '.'
    return summary, accepted, excluded, prompt_manifest


def refresh():
    summary, _, excluded, _ = inventory()
    write(ROOT / 'direct-control-v1/training-preparation.json', summary)
    write(ROOT / 'direct-control-v1/training-exclusions.json', {'episodes': excluded})
    return summary


if __name__ == '__main__':
    import json
    print(json.dumps(refresh(), indent=2))
