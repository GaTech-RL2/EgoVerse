"""Audit retained action targets against executed feedback, without changing data."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .report import ROOT, read, write

ERROR = re.compile(r'Process exited with code [1-9]\d*|Unknown process id|Traceback \(most recent call last\)')
ERROR_KINDS = {
    'missing_apply_patch': r'apply_patch: (?:command )?not found',
    'permission_denied': r'Permission denied',
    'unknown_process': r'Unknown process id',
    'syntax_error': r'SyntaxError',
    'missing_file': r'No such file or directory|can.t open file',
    'missing_module': r'ModuleNotFoundError|No module named',
    'invalid_argument': r'invalid choice|unrecognized arguments',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feedback(trace):
    return {r['source_call_id']: str(r.get('content', ''))
            for step in trace['steps'] for r in step.get('observation', {}).get('results', [])
            if r.get('source_call_id')}


def audit():
    source = ROOT/'rollout-viewer-v3/collected/training-viewer-v3-export'
    manifest = read(source/'manifest.json')
    decisions, teachers, students = [], [], []
    for row in manifest['episodes']:
        path = source/row['id']/'trace.json'
        if sha(path) != row['trace_sha256']:
            raise ValueError('Teacher trace checksum changed: '+row['id'])
        trace = read(path); responses = feedback(trace)
        teachers.append({'task':trace['task'], 'source_episode':trace['source_episode'],
            'trace_sha256':sha(path), 'score':trace['score'],
            'control_steps':trace['resource_metrics'].get('control_steps', 0),
            'simulator_calls':trace['resource_metrics'].get('calls', 0),
            'elapsed_s':trace['resource_metrics'].get('elapsed_s')})
        for state in trace['training_decisions']:
            calls = state['action']
            if any(c['id'] not in responses for c in calls):
                raise ValueError('Unmatched retained target: '+state['id'])
            errors = [{'call_id':c['id'], 'name':c['function']['name'],
                       'feedback_excerpt':responses[c['id']][-1200:]}
                      for c in calls if ERROR.search(responses[c['id']])]
            draws = sum(e['draws'] for e in state['leact_training_exposures'] if e['source']=='leact')
            decisions.append({'id':state['id'], 'task':state['task'], 'episode':state['episode'],
                'trace_step_ids':state['trace_step_ids'],
                'error_returning_target':bool(errors), 'errors':errors,
                'empty_poll_only':bool(calls) and all(c['function']['name']=='write_stdin'
                    and not c['function']['arguments'].get('chars') for c in calls),
                'image_only':bool(calls) and all(c['function']['name']=='view_image' for c in calls),
                'uses_apply_patch':any('apply_patch' in str(c['function']['arguments'].get('cmd','')) for c in calls),
                'leact_sampled':state['leact_sampled'],
                'leact_selected':bool(state['leact_selected_candidates']),
                'leact_new_example_draws':draws,
                'viewer_url':'/efe-rollouts.html#training/'+state['episode'].replace('/','%2F')+'/'+state['id']})
    for model in ['warm96','leact','leact-rebuilt']:
        folder = ROOT/'training-evaluation-v1'/model
        for row in read(folder/'public-episodes/manifest.json', {'episodes':[]})['episodes']:
            path = folder/'public-episodes'/row['id']/'trace.json'
            if sha(path) != row['trace_sha256']:
                raise ValueError('Student trace checksum changed')
            trace = read(path); responses = feedback(trace)
            kinds = {name:sum(bool(re.search(pattern,text)) for text in responses.values())
                     for name,pattern in ERROR_KINDS.items()}
            students.append({'model':model, 'task':trace['task'], 'score':trace['score'],
                'trace_sha256':sha(path), 'model_requests':trace['model_requests'],
                'tool_calls':sum(len(s['tool_calls']) for s in trace['steps']),
                'control_steps':trace['resource_metrics'].get('control_steps',0),
                'simulator_calls':trace['resource_metrics'].get('calls',0),
                'error_observation_counts':kinds, 'most_repeated_calls':trace['repeated_calls'][:3]})
    totals = {'retained_decisions':len(decisions), 'teacher_episodes':len(teachers), 'unmatched_targets':0}
    for key in ['error_returning_target','empty_poll_only','image_only','uses_apply_patch']:
        selected = [d for d in decisions if d[key]]
        totals[key] = {'decisions':len(selected), 'leact_sampled':sum(d['leact_sampled'] for d in selected),
            'leact_selected':sum(d['leact_selected'] for d in selected),
            'leact_states_drawn':sum(d['leact_new_example_draws']>0 for d in selected),
            'leact_example_draws':sum(d['leact_new_example_draws'] for d in selected)}
    value = {'created_utc':datetime.now(timezone.utc).isoformat(), 'totals':totals,
        'teacher_manifest_sha256':sha(source/'manifest.json'), 'decisions':decisions,
        'teachers':teachers, 'student_episodes':students,
        'summary':[
            'The saved student episodes fail before robot control: repeated invalid shell commands or polls consume the task budget.',
            f"The {len(decisions)} retained decisions come from successful teacher episodes, but {totals['error_returning_target']['decisions']} target calls that returned errors and {totals['empty_poll_only']['decisions']} consist only of empty process polls.",
            f"LeAct admitted explanations for {totals['error_returning_target']['leact_selected']} error-returning targets and actually drew {totals['error_returning_target']['leact_states_drawn']} of those states during training. Positive likelihood gain is not an action-quality check.",
            'This round generated student explanations at frozen teacher states; it did not train on student closed-loop mistakes and recoveries.',
            'The next study keeps tasks diverse while standardizing observe–act–check behavior, and filters individual action targets using their executed outcomes.'
        ],
        'limitations':[
            'This is a descriptive audit, not proof that any single factor caused failure. Tool formatting, image preprocessing and train/serve context differences remain possible contributors.',
            'A nonzero shell exit can be a harmless inspection failure. Empty polls can be useful orchestration, but they are not robot actions; the counts do not claim every such target was harmful.',
            'Target-type counts overlap and should not be added together.',
            'The selected successful teacher corpus is not an unbiased teacher success-rate estimate.',
            'Only collected student recordings are included here. An adjudicated graded timeout without a collected recording is counted in the score table, not fabricated here.'
        ]}
    (ROOT/'trace-diagnosis-v1').mkdir(exist_ok=True)
    write(ROOT/'trace-diagnosis-v1/report.json', value)
    return value


if __name__ == '__main__':
    print(json.dumps(audit()['totals'], indent=2))
