"""Freeze successful public demonstrations and export matched BuilderBench SFT."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from ..data import file_sha256, fingerprint, write_json, write_jsonl
from ..experiment import MODEL, MODEL_REVISION
from ..training import conversation, fits_context
from .expert import HISTORY_OBSERVATIONS, validate_action
from .preflight import UPSTREAM_REVISION

PROMPT_VERSION = 'builderbench-student-public-history-v1'
CUTOFF = 8192


def system_prompt(reference):
    path = Path(reference) / 'builderbench/creative_cube_language_env.py'
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ENV_SYSTEM_PROMPT' for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == 'strip':
                return ast.literal_eval(value.func.value).strip()
    raise ValueError('Could not recover the pinned public environment prompt')


def context(system, observation, history):
    parts = [system, '\nPUBLIC OBSERVATION AND ACTION HISTORY']
    for step in history[-(HISTORY_OBSERVATIONS - 1):]:
        parts.extend(['Observation:\n' + step['observation'],
                      'Executed action:\n' + json.dumps(step['action'], sort_keys=True)])
    parts.append('Current observation:\n' + observation)
    return '\n\n'.join(parts)


def forward_prompt(state, *, thinking=True):
    body = state['context']
    if thinking:
        body = body.replace('Always output actions as a single, valid JSON object with the specified key/value structure.',
                            'After the reasoning block, output one valid JSON action with the specified key/value structure.')
        return body + ('\n\nChoose the next action using only these public observations and prior actions. '
                       'Explain the relevant geometry and action briefly (at most 200 words) in <think>...</think>, '
                       'then output exactly one bare action JSON object. Do not output a sequence of actions.')
    return body + '\n\nChoose the next action. Output only one bare action JSON object. /no_think'


def backward_prompt(state):
    return (state['context'] + '\n\nA successful expert chose this next action:\n' +
            json.dumps(state['expert_action'], sort_keys=True) +
            '\n\nExplain briefly why this action fits the visible state and task (at most 200 words). '
            'Output the explanation in <think>...</think> followed by DONE. Do not invent unseen observations.')


def pairs(state, reasoning, *, method='joint', action=None):
    target = json.dumps(state['expert_action'] if action is None else action, sort_keys=True)
    if method == 'bc':
        return [conversation(forward_prompt(state, thinking=False), '<think>\n\n</think>\n\n' + target)]
    answer = '<think>\n' + reasoning.strip() + '\n</think>\n\n' + target
    result = [conversation(forward_prompt(state), answer)]
    if method in {'joint', 'leact'}:
        result.append(conversation(backward_prompt(state), '<think>\n' + reasoning.strip() + '\n</think>\n\nDONE'))
    return result


def freeze(teacher, reference, output, *, parent=None):
    teacher, reference, output = map(Path, [teacher, reference, output])
    spec = json.loads((teacher / 'manifest.json').read_text())
    teacher_sha = fingerprint(spec)
    if spec['upstream_revision'] != UPSTREAM_REVISION:
        raise ValueError('Unexpected simulator version')
    system = system_prompt(reference)
    if fingerprint(system) != spec['system_prompt_sha256']:
        raise ValueError('Public system prompt differs from expert collection')
    tasks = {row['id']: row for row in spec['task_metadata']}
    for task in tasks.values():
        if file_sha256(reference / 'builderbench/tasks' / f'cube-{task["cubes"]}.npz') != task['task_file_sha256']:
            raise ValueError('Simulator task hash differs')
    expected = {(t, s) for t in spec['task_ids'] for s in spec['seeds']}
    records, states, evidence = {}, [], {'manifest.json': file_sha256(teacher / 'manifest.json')}
    for path in sorted(teacher.glob('rollouts/*/seed-*/result.json')):
        result = json.loads(path.read_text())
        key = result['task'], result['seed']
        if key not in expected or key in records or result['run_sha256'] != teacher_sha:
            raise ValueError('Foreign or duplicate expert rollout')
        if path.parent.name != f'seed-{key[1]}' or path.parent.parent.name != key[0]:
            raise ValueError('Expert rollout path mismatch')
        records[key] = result
        evidence[str(path.relative_to(teacher))] = file_sha256(path)
        if not result['success']:
            continue
        steps = sorted((path.parent / 'steps').glob('*.json'))
        if result['invalid_outputs'] or len(steps) != result['actions'] or result['cube_progress'] != 1:
            raise ValueError('Successful trajectory has incomplete action evidence')
        history = []
        for index, step_path in enumerate(steps):
            step = json.loads(step_path.read_text())
            call_path = path.parent / 'calls' / f'{index:03d}.json'
            call = json.loads(call_path.read_text())
            if (step_path.stem != f'{index:03d}' or not call['accepted'] or call['run_sha256'] != teacher_sha
                    or (call['task'], call['seed'], call['action_index']) != (*key, index)
                    or fingerprint(call['request']) != step['request_sha256']
                    or call['request_sha256'] != step['request_sha256']):
                raise ValueError('Expert action request identity mismatch')
            if history and history[-1]['next_observation'] != step['observation']:
                raise ValueError('Broken public observation history')
            if call['result'] != {'action':step['action'], 'decision_summary':step['decision_summary']}:
                raise ValueError('Stored step differs from the accepted expert action and public explanation')
            validate_action(step['action'], tasks[key[0]]['cubes'])
            summary = step['decision_summary'].strip()
            if not summary or any(t in summary for t in ['<think', '</think', '<|im_']):
                raise ValueError('Invalid public decision summary')
            states.append(dict(state_id=fingerprint([teacher_sha, *key, index]), task=key[0], seed=key[1],
                action_index=index, cubes=tasks[key[0]]['cubes'], context=context(system, step['observation'], history),
                expert_action=step['action'], reasoning=summary, source_step=str(step_path.relative_to(teacher)),
                source_request_sha256=step['request_sha256']))
            evidence[str(step_path.relative_to(teacher))] = file_sha256(step_path)
            evidence[str(call_path.relative_to(teacher))] = file_sha256(call_path)
            history.append(step)
    if records.keys() != expected:
        raise ValueError('Wait for the complete declared teacher collection')
    successful = sorted({t for (t, _), r in records.items() if r['success']})
    if parent is None:
        partition = dict(in_domain=successful, held_out=sorted(set(tasks) - set(successful)),
                         rule='Fixed by successful initial GPT-6 collection, before student training')
    else:
        parent_spec = json.loads((Path(parent) / 'manifest.json').read_text())
        partition = parent_spec['partition']
        if set(tasks) != set(partition['in_domain']):
            raise ValueError('Iterative collection must remain on the fixed in-domain task set')
        if set(spec['seeds']) & set(parent_spec['teacher_seeds']):
            raise ValueError('Iterative trajectories must be separate from warmstart trajectories')
    manifest = dict(schema_version=1, teacher_model=spec['model'], teacher_sha256=teacher_sha,
        teacher_seeds=spec['seeds'], upstream_revision=UPSTREAM_REVISION, task_metadata=spec['task_metadata'],
        task_ids=spec['task_ids'], partition=partition, successful_tasks=successful, states=len(states),
        prompt_version=PROMPT_VERSION, history_observations=HISTORY_OBSERVATIONS,
        system_prompt=system, source_files=evidence, states_sha256=fingerprint(states),
        role='iteration' if parent else 'warmstart',
        differences=['GPT-6 public decision summaries, not the paper three-model corpus',
                     'Student history contains public observations and executed actions, without earlier expert summaries',
                     'New GPT-6-specific task split; all 51 tasks remain in the evaluation denominator'])
    if not states:
        raise ValueError('No successful expert states to train on')
    if output.exists():
        if json.loads((output / 'manifest.json').read_text()) != manifest:
            raise ValueError('Immutable corpus differs')
        if fingerprint([json.loads(line) for line in (output / 'states.jsonl').read_text().splitlines()]) != manifest['states_sha256']:
            raise ValueError('Frozen states changed')
        return manifest
    write_jsonl(output / 'states.jsonl', states)
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_corpus(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    states = [json.loads(line) for line in (path / 'states.jsonl').read_text().splitlines()]
    if len(states) != manifest['states'] or fingerprint(states) != manifest['states_sha256']:
        raise ValueError('Frozen corpus hash mismatch')
    return states, manifest


def export_dataset(output, data, *, method, corpus, selection, tokenizer):
    output = Path(output)
    if not data or not all(fits_context(p, tokenizer, CUTOFF) for p in data):
        raise ValueError('Empty or over-context training data')
    lengths = [len(tokenizer.apply_chat_template([{'role': 'user' if c['from'] == 'human' else 'assistant',
        'content': c['value']} for c in p['conversations']], tokenize=True, enable_thinking=method != 'bc')) for p in data]
    if (output / 'manifest.json').exists():
        existing = json.loads((output / 'manifest.json').read_text())
        if json.loads((output / 'data.json').read_text()) != data or existing['corpus_sha256'] != corpus['states_sha256']:
            raise ValueError('Previously exported dataset differs')
        return existing
    write_json(output / 'data.json', data)
    write_json(output / 'dataset_info.json', {'leact_local': {'file_name': 'data.json', 'formatting': 'sharegpt',
        'columns': {'messages': 'conversations'}, 'tags': {'role_tag': 'from', 'content_tag': 'value',
                                                        'user_tag': 'human', 'assistant_tag': 'gpt'}}})
    write_jsonl(output / 'selection.jsonl', selection)
    manifest = dict(method=method, corpus_sha256=corpus['states_sha256'], sft_examples=len(data),
        selected_traces=sum(r.get('selected_traces', int(r.get('admitted', False))) for r in selection),
        cutoff_len=CUTOFF, context_length_checked=True, data_sha256=file_sha256(output / 'data.json'),
        maximum_example_tokens=max(lengths), total_tokens_per_epoch_including_masked_prompts=sum(lengths),
        prompt_version=PROMPT_VERSION, selection_sha256=file_sha256(output / 'selection.jsonl'))
    write_json(output / 'manifest.json', manifest)
    return manifest


def export_warmstart(corpus_path, output, tokenizer):
    states, corpus = load_corpus(corpus_path)
    datasets, audit = {'bc': [], 'joint': []}, []
    for state in states:
        candidates = {method: pairs(state, state['reasoning'], method=method) for method in datasets}
        admitted = all(fits_context(p, tokenizer, CUTOFF) for values in candidates.values() for p in values)
        audit.append(dict(state_id=state['state_id'], task=state['task'], admitted=admitted,
                          reason=None if admitted else 'At least one matched example exceeds 8192 tokens'))
        if admitted:
            for method in datasets:
                datasets[method].extend(candidates[method])
    if {r['task'] for r in audit if r['admitted']} != set(corpus['partition']['in_domain']):
        raise ValueError('A declared in-domain task has no context-eligible demonstration')
    return {method: export_dataset(Path(output) / method / 'sft', values, method=method,
                                   corpus=corpus, selection=audit, tokenizer=tokenizer)
            for method, values in datasets.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['teacher', 'reference', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--datasets', type=Path)
    args = parser.parse_args()
    result = freeze(args.teacher, args.reference, args.output, parent=args.parent)
    if args.datasets:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
        result = export_warmstart(args.output, args.datasets, tokenizer)
    print(json.dumps(result if args.datasets else {k: result[k] for k in ['states', 'successful_tasks', 'role']}, indent=2))


if __name__ == '__main__':
    main()
