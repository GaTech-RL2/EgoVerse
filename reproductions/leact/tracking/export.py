"""Build a tracker snapshot from actual run artifacts, without model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parents[1]


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def snapshot(base: Path = BASE):
    artifacts, build = base / 'artifacts', base / 'cluster/.build'
    status = read(artifacts / 'run-status.json', {})
    astra_status = read(artifacts / 'astra-expert/status.json')
    if astra_status:
        status['astra_expert'] = astra_status
    projection_status = read(artifacts / 'projection-status.json')
    if projection_status:
        status['projection_sampling'] = projection_status
    projection_ablation = read(artifacts / 'projection-ablation.json')
    if projection_ablation:
        status['projection_ablation'] = projection_ablation
    self_followup = read(artifacts / 'self-followup.json')
    if self_followup:
        status['self_followup'] = self_followup
    self_leact = read(artifacts / 'self-leact-status.json')
    if self_leact:
        status['self_leact'] = self_leact
    experiment_plan = read(artifacts / 'experiment-plan.json')
    if experiment_plan:
        status['experiment_plan'] = experiment_plan
    builderbench_preflight = read(artifacts / 'builderbench/preflight/preflight.json')
    if builderbench_preflight:
        status['builderbench_preflight'] = builderbench_preflight
    builderbench_probe = read(artifacts / 'builderbench/expert-probe/gpt6-v2/metrics.json')
    if builderbench_probe:
        status['builderbench_probe'] = builderbench_probe
    builderbench_speed = read(artifacts / 'builderbench/speed-report.json')
    if builderbench_speed:
        # Full per-rollout timing rows remain in the source report. The UI uses
        # its aggregate statistics and paired comparisons, avoiding multi-MB
        # copies in every SQLite status snapshot.
        for report in builderbench_speed.get('reports', []):
            report.pop('rows', None)
        status['builderbench_speed'] = builderbench_speed
    builderbench_expert = read(artifacts / 'builderbench/expert-status.json')
    builderbench_student = read(artifacts / 'builderbench/student-status.json')
    if builderbench_student:
        status['builderbench_student'] = builderbench_student
    builderbench_iteration = read(artifacts / 'builderbench/iteration-expert-status.json')
    if builderbench_iteration:
        status['builderbench_iteration_expert'] = builderbench_iteration
    if builderbench_expert:
        status['builderbench_expert'] = builderbench_expert
        root = Path(builderbench_expert.get('output', '')).resolve()
        allowed = (artifacts / 'builderbench').resolve()
        if root.is_relative_to(allowed):
            recent = sorted(root.glob('rollouts/*/seed-*/steps/*.json'), key=lambda p: p.stat().st_mtime)[-6:]
            status['builderbench_examples'] = [dict(task=p.parents[2].name, seed=p.parents[1].name,
                                                    step=p.stem, **read(p, {})) for p in recent]
    overnight = read(artifacts / 'overnight-status.json')
    if overnight:
        status['overnight'] = overnight
        status['status'] = overnight.get('status', status.get('status'))
        status['active_workflows'] = sorted(set(status.get('active_workflows', [])) |
                                           set(overnight.get('active_workflows', [])))
    runs = {}
    paths = list(build.glob('*launch-record.json')) + list(build.glob('*/launch-record.json'))
    for path in sorted(paths):
        record = read(path, {})
        name = record.get('name') or record.get('workflow')
        if not name:
            continue
        latest = {}
        for suffix in ['status-initial.json', 'status-handoff.json', 'status-current.json',
                       'terminal-status.json', 'status-verified-after-cancellation.json',
                       'status-final.json', 'tracking-status.json']:
            query = read(path.parent / suffix, {})
            if query.get('name') == name:
                latest = query
        merged = {**record, **latest}
        runs[name] = {k: merged.get(k) for k in ['name', 'status', 'priority', 'pool', 'submit_time',
                                               'start_time', 'end_time', 'phase', 'payload_sha256',
                                               'failure_stage', 'cancelled_by', 'gpu_count']}
        runs[name]['name'] = name
        runs[name]['url'] = 'https://us-west-2-aws.osmo.nvidia.com/workflows/' + name
        runs[name]['evidence'] = str(path.relative_to(base))
    status['active_workflows'] = sorted(name for name, run in runs.items()
                                        if run.get('status') in {'RUNNING', 'PENDING', 'WAITING'})
    training = []
    specifications = [('original-bc', 'Original BC', artifacts/'coldstart-training-evidence.json',
                       artifacts/'coldstart-generation-check.json'),
                      ('masked-bc', 'BC with corrected loss mask', artifacts/'masked-bc-training-evidence.json',
                       artifacts/'masked-bc-generation-check.json')]
    for key, label, evidence, gate_path in specifications:
        record = read(evidence)
        if not record:
            continue
        history = record.get('first_steps', []) + record.get('last_steps', [])
        history_complete = False
        if key == 'masked-bc':
            state = read(build/'full-5/collected/latest/coldstart/checkpoints/trainer_state.json')
            if state:
                history = [r for r in state['log_history'] if 'loss' in r]
                history_complete = True
        training.append({'id': key, 'label': label, 'workflow': record.get('workflow'),
                         'steps': record.get('global_step'), 'epochs': record.get('epoch'),
                         'history': history, 'history_complete': history_complete,
                         'final_metrics': record.get('final_metrics'), 'gate': read(gate_path),
                         'evidence': str(evidence.relative_to(base))})
    # Future collectors provide live trainer_log.jsonl and complete trainer_state.json.
    live = []
    for root in build.glob('*/collected/latest'):
        if root.parents[1].name.startswith('builderbench-gpt6-student-evaluate-'):
            continue  # Both collectors see the same shared training directory.
        for path in sorted(root.glob('**/trainer_log.jsonl')):
            rows = []
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    if 'loss' in row:
                        rows.append({**row, 'step': row.get('current_steps', row.get('step')),
                                     'learning_rate': row.get('learning_rate', row.get('lr'))})
                except json.JSONDecodeError:
                    continue
            if rows:
                live.append({'run': root.parents[1].name, 'phase': str(path.parent.relative_to(root)),
                             'history': rows, 'source': str(path.relative_to(base)),
                             'dataset': read(path.parent.parent/'manifest.json', {}),
                             'verification': read(path.parent.parent/'training_verification.json',
                                                  read(path.parent.parent.parent/'training_verification.json')),
                             'completed': (path.parent.parent/'training_complete.json').exists(),
                             'modified': path.stat().st_mtime})
    live.sort(key=lambda record: record['modified'])
    lineage = []
    for root in sorted(build.glob('*/collected/latest')):
        if root.parents[1].name.startswith('builderbench-gpt6-student-evaluate-'):
            continue
        for path in sorted(root.glob('**/sft/train.yaml')):
            config = yaml.safe_load(path.read_text()) or {}
            stage = str(path.parent.parent.relative_to(root))
            training_record = next((t for t in live if t['run'] == root.parents[1].name
                             and t['phase'] == stage + '/sft/checkpoints'), {})
            history = training_record.get('history', [])
            lineage.append(dict(run=root.parents[1].name, stage=stage,
                initial_checkpoint=config.get('model_name_or_path'),
                epochs=config.get('num_train_epochs'), complete=training_record.get('completed', False),
                total_steps=history[-1].get('total_steps') if history else None,
                source=str(path.relative_to(base))))
    if lineage:
        status['training_lineage'] = lineage
    comparisons = []
    prompt_diagnostics = []
    for path in sorted(build.glob('*/collected/latest/result.json')):
        value = read(path, {})
        if 'arms' in value and 'protocol' in value:
            prompt_diagnostics.append({'run': path.parents[2].name, **value})
    for path in sorted(build.glob('*/collected/latest/**/comparison.json')):
        value = read(path, {})
        if value.get('rows'):
            comparisons.append({'label': str(path.parent.relative_to(build)), **value})
    evaluations = [{'label': 'Original BC · final epoch only',
                    'summary': read(artifacts/'original-bc-evaluation/summary.json'),
                    'comparison': read(artifacts/'original-bc-evaluation/comparison.json'),
                    'selection': 'final epoch only; best-across-epochs not measured'}]
    for path in sorted(build.glob('*/collected/latest/**/summary.json')):
        value = read(path, {})
        if 'single' in value and 'oracle_bon' in value:
            evaluations.append({'label': str(path.parent.relative_to(build)), 'summary': value,
                                'selection': 'per-checkpoint; use aggregate comparison for epoch selection'})
    recipe_path = build/'full-5/collected/latest/coldstart/train.yaml'
    recipe = yaml.safe_load(recipe_path.read_text()) if recipe_path.exists() else {}
    evidence_paths = [artifacts/p for p in ['appendix-audit.json', 'run-status.json',
                       'overnight-status.json',
                       'masked-bc-training-evidence.json', 'masked-bc-generation-check.json',
                       'base-gate-sft-admission.json', 'base-baseline-validity-audit.json',
                       'prompt-diagnostic-result.json', 'diagnostic-result.json']]
    return {'schema_version': 1, 'title': 'LeAct · Leduc 6r2s',
            'paper': 'https://arxiv.org/html/2607.21856v1', 'status': status,
            'audit': read(artifacts/'appendix-audit.json', {}), 'recipe': recipe,
            'runs': sorted(runs.values(), key=lambda r: r.get('submit_time') or ''),
            'training': training, 'live_training': live, 'evaluations': evaluations,
            'comparisons': comparisons,
            'prompt_diagnostics': prompt_diagnostics,
            'evidence': [{'path': str(p.relative_to(base)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in evidence_paths if p.exists()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = (json.dumps(snapshot(), indent=2, allow_nan=False)+'\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_bytes(data)
    temporary.replace(args.output)
    print(json.dumps({'output': str(args.output), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}))


if __name__ == '__main__':
    main()
