"""Build an offline HTML comparison from captured public teacher and student traces."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
STAGES = ['base', 'bc-e3', 'joint-e3', 'leact-r1-e3']
LABELS = {'base':'Base Qwen3-8B', 'bc-e3':'Action-only BC', 'joint-e3':'Frontier warmstart SFT',
          'leact-r1-e3':'Warmstart + LeAct R1', 'teacher':'GPT-6 Astra · frontier teacher'}
CASES = [
    ('cube-1-task-1','A simple move, much shorter output','The teacher and reasoning students solve this one-cube task. Compare the base student’s long output with the concise trained students. Action-only BC fails this particular rollout.'),
    ('cube-4-task-2','LeAct succeeds on an offset stack','On this seed, LeAct completes all four targets. Warmstart reaches two of four; the base student reaches none. This is an illustrative success, not an aggregate score.'),
    ('cube-3-task-1','A regression after LeAct','Warmstart solves this three-cube task, while LeAct runs out of simulation time with two cubes at their targets. The teacher also succeeds.'),
    ('cube-6-task-2','A held-out warmstart success','This task has no successful training demonstration. Warmstart succeeds on this seed; the teacher’s collected rollout and the LeAct student both fail. Held-out tasks were chosen by teacher failure, so this is a harder subset.'),
    ('cube-2-task-2','An invalid first decision','LeAct invents an untargeted “cube 2” in a scene containing only cubes 0 and 1. The simulator schema rejects the first action, so no action is executed. The explanation sounds plausible but refers to a nonexistent object.'),
    ('cube-20-task-1','The observation history fills the context','All four students stop after five executed actions because the next prompt exceeds 8,192 tokens. Shorter generated explanations do not fix a long observation-and-action history.'),
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode(directory, *, teacher=False):
    result = read(directory / 'result.json')
    calls = sorted((directory / 'calls').glob('*.json'))
    steps = []
    for path in calls:
        call = read(path)
        step_path = directory / 'steps' / path.name
        step = read(step_path) if step_path.exists() else {}
        if step:
            assert step['request_sha256'] == call['request_sha256']
        if teacher:
            # Only the explicitly requested, public decision summary is exported.
            # Never copy the provider response or any private reasoning fields.
            answer = call.get('result') or {}
            reasoning = answer.get('decision_summary', '')
            action = answer.get('action')
            raw = json.dumps(answer, ensure_ascii=False, indent=2)
            usage = step.get('usage', {})
        else:
            reasoning = call.get('reasoning', '')
            action = call.get('action')
            raw = call.get('generated', {}).get('text', '')
            usage = call.get('response', {}).get('usage', {})
        prompt = call.get('request', {}).get('prompt', '') if not teacher else ''
        observation = step.get('observation') or (prompt.rsplit('Current observation:\n', 1)[-1].split('\n\nChoose the next action', 1)[0] if 'Current observation:\n' in prompt else None)
        steps.append(dict(index=call['action_index'], observation=observation,
            next_observation=step.get('next_observation'), reasoning=reasoning, action=action,
            raw=raw, valid=call.get('accepted'), error=call.get('error'),
            latency=call.get('latency_seconds'), tokens=call.get('generated', {}).get('generated_tokens', usage.get('completion_tokens')),
            prompt_tokens=call.get('generated', {}).get('prompt_tokens', usage.get('prompt_tokens')),
            outcome=step.get('outcome'), executed=bool(step), request_sha256=call['request_sha256'],
            source=str(path), source_sha256=sha(path),
            step_source_sha256=sha(step_path) if step else None))
    return dict(result=result, steps=steps, source=str(directory), result_sha256=sha(directory/'result.json'))


def build_data(collected):
    corpus = BASE / 'artifacts/builderbench/corpus-gpt6-warmstart-v1'
    partition = read(corpus/'manifest.json')['partition']
    metrics = []
    for stage in STAGES:
        d=collected/'metrics-snapshot'/stage
        m=read(d/'metrics.json'); speed=read(d/'speed.json')
        assert m['evaluated_seeds']==list(range(6))
        assert speed['completed_rollouts']==306
        metrics.append(dict(stage=stage,label=LABELS[stage],seeds=6,rollouts=306,
            progress=m['partitions']['all']['single_progress_mean'],
            in_domain=m['partitions']['in_domain']['single_progress_mean'],
            held_out=m['partitions']['held_out']['single_progress_mean'],
            pass_n=m['partitions']['all']['full_task_pass_at_n_count'],
            successes=speed['full_successes'],success_rate=speed['success_rate'],
            seconds=speed['decision_seconds']['mean'],tokens=speed['completion_tokens']['mean'],
            invalid=m['invalid_outputs'],overflow=m['context_overflows'],
            metric_sha256=sha(d/'metrics.json'),manifest_sha256=sha(d/'manifest.json')))
    cases=[]
    for task,title,note in CASES:
        teacher=episode(BASE/'artifacts/builderbench/teacher-gpt6-v1/rollouts'/task/'seed-0',teacher=True)
        students={st:episode(collected/'evaluations'/st/'rollouts'/task/'seed-0') for st in STAGES}
        for st,ep in [('teacher',teacher),*students.items()]:
            video=BASE/'artifacts/builderbench/trace-videos'/task/(st+'.mp4')
            receipt=video.with_suffix('.json')
            if video.exists() and receipt.exists():
                record=read(receipt)
                assert record['replay_verified'] and record['video_sha256']==sha(video)
                assert record['result_sha256']==ep['result_sha256']
                ep['video']={**record,'base64':base64.b64encode(video.read_bytes()).decode()}
        initial=teacher['steps'][0]['observation']
        for value in students.values():
            assert value['steps'][0]['observation']==initial, f'Initial observation differs: {task}'
        cases.append(dict(id=task,title=title,note=note,partition='Demonstrated task' if task in partition['in_domain'] else 'Held-out task',
            seed=0,teacher=teacher,students=students))
    iteration = BASE/'artifacts/builderbench/corpus-gpt6-iteration-v1'
    states={s['state_id']:s for s in map(json.loads,(iteration/'states.jsonl').read_text().splitlines())}
    selection=list(map(json.loads,(collected/'training/leact-r1/sft/selection.jsonl').read_text().splitlines()))
    selected_count=sum(len(r['selected_ids']) for r in selection)
    assert selected_count==94
    # Deliberately include the first one-cube state: a positive action delta can
    # coexist with an inaccurate explanation mentioning nonexistent upper cubes.
    first=next(r for r in selection if states[r['state_id']]['task']=='cube-1-task-1' and states[r['state_id']]['action_index']==0)
    chosen=[first]
    seen={states[first['state_id']]['task']}
    for admitted in [True,False]:
        for row in sorted(selection,key=lambda r:(states[r['state_id']]['cubes'],states[r['state_id']]['task'],states[r['state_id']]['action_index'])):
            state=states[row['state_id']]
            if bool(row['selected_ids'])==admitted and state['task'] not in seen and row['candidates']:
                chosen.append(row);seen.add(state['task'])
                if len(chosen)>=(4 if admitted else 7):break
    training=[]
    for row in chosen:
        state=states[row['state_id']]
        training.append(dict(state=state,baseline=row.get('baseline'),candidates=row['candidates'],
            selected_ids=row['selected_ids'],source='training/leact-r1/sft/selection.jsonl',
            selection_sha256=sha(collected/'training/leact-r1/sft/selection.jsonl')))
    roles={}
    for role in ['train','evaluate','fixed']:
        p=BASE/f'cluster/.build/builderbench-gpt6-student-{role}-20261024-v1/status-current.json'
        v=read(p) if p.exists() else {}
        roles[role]=dict(workflow=v.get('name'),status=v.get('status','Waiting for training completion'))
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),labels=LABELS,metrics=metrics,cases=cases,
        training=training,training_selected=selected_count,training_examples=188,
        roles=roles,teacher=dict(model='azure/openai/gpt-6-astra',tasks=51,successes=23,progress=0.6230397071109145,seeds=1),
        evidence=read(collected/'collection.json'),
        corpus_sha256=read(corpus/'manifest.json')['states_sha256'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--collected',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=build_data(args.collected)
    template=Path(__file__).with_name('trace_report.html').read_text()
    encoded=json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
    document=template.replace('<!-- EMBEDDED_DATA -->','<script id="trace-data" type="application/json">'+encoded+'</script>')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(document)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(dict(created_utc=data['created_utc'],
        html_sha256=sha(args.output),cases=len(data['cases']),training_states=len(data['training']),
        videos=sum('video' in episode for case in data['cases'] for episode in [case['teacher'],*case['students'].values()]),
        source_bundle_sha256=data['evidence']['bundle_sha256'],
        student_stages=STAGES,teacher_model=data['teacher']['model']),indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),bytes=args.output.stat().st_size,cases=len(data['cases']))))


if __name__=='__main__':main()
