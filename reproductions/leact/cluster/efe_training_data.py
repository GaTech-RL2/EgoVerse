"""Frozen public teacher decisions and complete-target VL training examples."""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import random

try:
    from .efe_budget_gateway import aligned_instructions, flatten_namespaces, public_value
except ImportError:  # Standalone OSMO-mounted module.
    from efe_budget_gateway import aligned_instructions, flatten_namespaces, public_value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    p = path.with_suffix(path.suffix + '.tmp')
    p.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'); p.replace(path)


def plain_content(value):
    """Responses text/images to the Qwen processor's public chat representation."""
    if isinstance(value, str): return value
    if not isinstance(value, list): raise ValueError('Unsupported public content')
    out = []
    for part in value:
        typ = part.get('type')
        if typ in {'input_text', 'output_text', 'text'}:
            out.append({'type': 'text', 'text': part['text']})
        elif typ in {'input_image', 'image_url'}:
            url = part['image_url']; url = url['url'] if isinstance(url, dict) else url
            if not url.startswith('data:image/'):
                raise ValueError('Image is not a self-contained archived observation')
            out.append({'type': 'image', 'image': url})
        else:
            raise ValueError('Unsupported content type: ' + str(typ))
    return out


def chat_inputs(body):
    messages = [{'role': 'system', 'content': body['instructions']}]
    for item in body['input']:
        kind = item.get('type', 'message')
        if kind == 'message':
            messages.append({'role': item['role'], 'content': plain_content(item['content'])})
        elif kind == 'function_call':
            messages.append({'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'id': item['call_id'],
                             'function': {'name': item['name'], 'arguments': json.loads(item['arguments'])}}]})
        elif kind == 'function_call_output':
            messages.append({'role': 'tool', 'tool_call_id': item['call_id'], 'content': plain_content(item['output'])})
        else:
            raise ValueError('Unsupported public history item: ' + str(kind))
    return messages


def forward_target(action, rationale):
    return {'role': 'assistant', 'content': '<think>\n' + rationale.strip() + '\n</think>\n\n',
            'tool_calls': copy.deepcopy(action)}


def backward_messages(messages, action):
    return copy.deepcopy(messages) + [{'role': 'user', 'content':
        'The expert chose the following complete tool action for the current public state. '
        'Give a brief decision explanation connecting the observation to this action. '
        'Explain its purpose and relevant constraints; do not reproduce executable code. '
        'Do not use future observations. Finish with DONE.\nExpert action:\n' + json.dumps(action, ensure_ascii=False)}]


def extract_decisions(spec, root):
    audit = json.loads(Path(spec['scene_audit']).read_text())
    if sha(spec['scene_audit']) != spec['scene_audit_sha256']: raise ValueError('Scene audit changed')
    baseline = Path(spec['student_prompt_requests'])
    baseline_records = sorted(baseline.glob('*.json'))
    seed_record_path = next(p for p in baseline_records if p.with_suffix('.request.json.gz').exists())
    seed_record = json.loads(seed_record_path.read_text())
    seed = json.loads(gzip.decompress(seed_record_path.with_suffix('.request.json.gz').read_bytes()))
    seed, _ = flatten_namespaces(seed)
    policy = spec.get('action_admission', 'legacy-episode-success')
    if policy not in {'legacy-episode-success','executed-direct-control-v1'}:
        raise ValueError('Unknown action admission policy')
    if policy == 'executed-direct-control-v1':
        suffix = Path(spec['instruction_suffix_file']).read_text()
        suffix_sha = hashlib.sha256(suffix.encode()).hexdigest()
        if suffix_sha != spec['instruction_suffix_sha256'] or seed_record.get('instruction_suffix_sha256') != suffix_sha:
            raise ValueError('Student deployment method prompt differs from training')
        if not seed['instructions'].endswith(suffix):
            raise ValueError('Recorded student request is missing the method protocol')
        # This is the already-forwarded deployment request. Reapplying the
        # legacy alignment would remove the direct protocol's apply_patch line
        # and append conflicting controller instructions.
        alignment = {'mode':'exact-recorded-deployment-instructions','method_sha256':suffix_sha}
    else:
        seed, alignment = aligned_instructions(seed)
    schemas = {t['name']: t for t in seed['tools'] if t['type'] == 'function'}
    tools = [{'type': 'function', 'function': t} for t in seed['tools'] if t['type'] == 'function']
    write(root / 'student-interface.json', {'instructions': seed['instructions'], 'tools': tools,
          'alignment': alignment, 'source_request': str(baseline), 'source_condition': policy})
    decisions, excluded, evidence = [], [], []
    for episode in audit['eligible_episodes']:
        trial = episode['trial']; directory = Path(trial['archive'])
        candidates = list(directory.glob('*/result.json'))
        matches = [json.loads(p.read_text()) for p in candidates]
        matches = [r for r in matches if r.get('task_name', '').removeprefix('embodied-first-exam/') == trial['task']]
        if len(matches) != 1: raise ValueError('Missing/ambiguous native trial result: ' + trial['task'])
        actual = matches[0]
        if hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() != trial['trial_result_sha256']:
            raise ValueError('Native grade receipt changed: ' + trial['task'])
        rewards = actual['verifier_result']['rewards']
        if rewards['success'] != 1 or rewards['resource_enforced'] != 1 or actual.get('exception_info'):
            raise ValueError('Non-successful trial entered training')
        if any(rewards.get(k) for k in ('instance_mismatch','infra_error','grader_error','resource_monitor_missing','resource_switch_error')):
            raise ValueError('Native verifier admission failed')
        action_eligibility = None
        if policy == 'executed-direct-control-v1':
            action_audit_path = Path(episode['action_audit'])
            if sha(action_audit_path) != episode['action_audit_sha256']:
                raise ValueError('Executed-action audit changed')
            action_audit = json.loads(action_audit_path.read_text())
            if not action_audit['direct_method_pass']:
                raise ValueError('Episode violates the declared direct-control method')
            action_eligibility = {r['call_id']:r for r in action_audit['targets']}
        requests = directory.parent.parent / 'requests' / trial['job']
        records = [json.loads(p.read_text()) for p in requests.glob('*.json')]
        if len(records) != trial['model_requests']: raise ValueError('Incomplete archived request collection')
        for record in sorted(records, key=lambda r: r['started_utc']):
            ident = record['id']; path = requests / (ident + '.request.json.gz')
            try:
                if record.get('http_status') != 200 or record.get('status') != 'completed' or record.get('error'):
                    raise ValueError('Incomplete/error response')
                if policy == 'executed-direct-control-v1' and record.get('instruction_suffix_sha256') != suffix_sha:
                    raise ValueError('Teacher method prompt differs from student deployment')
                raw = json.loads(gzip.decompress(path.read_bytes()))
                if public_value(raw) != raw: raise ValueError('Unexpected private reasoning fields')
                body, mapping = flatten_namespaces(raw)
                reverse = {(v['namespace'], v['name']): k for k, v in mapping.items()}
                action = []
                for item in record['public_output']:
                    if item['type'] == 'message': continue
                    if item['type'] != 'function_call': raise ValueError('Unsupported expert action type')
                    name = reverse.get((item.get('namespace'), item['name']), item['name'])
                    if name not in schemas: raise ValueError('Expert tool unavailable to deployed student: ' + name)
                    action.append({'type': 'function', 'id': item['call_id'], 'function': {'name': name, 'arguments': json.loads(item['arguments'])}})
                if not action: raise ValueError('No executable expert action')
                if action_eligibility is not None:
                    for call in action:
                        outcome = action_eligibility.get(call['id'])
                        if not outcome or not outcome['eligible_action_target']:
                            raise ValueError('Target lacks an eligible executed outcome')
                for item in body['input']:
                    if item.get('type') == 'function_call' and item['name'] not in schemas:
                        raise ValueError('Historical expert tool unavailable to deployed student')
                # Only the public deployment instruction/tool contract is replaced;
                # all recorded past messages and observations remain unchanged.
                body['instructions'] = seed['instructions']
                messages = chat_inputs(body)
                rationale = '\n'.join(c['text'] for item in record['public_output'] if item['type'] == 'message'
                                      for c in item.get('content', []) if c['type'] == 'output_text').strip()
                state = {'id': hashlib.sha256((trial['workflow'] + '/' + trial['job'] + '/' + ident).encode()).hexdigest(),
                         'task': trial['task'], 'suite': episode['suite'], 'episode': trial['workflow'] + '/' + trial['job'],
                         'messages': messages, 'tools': tools, 'action': action, 'public_commentary': rationale,
                         'source_request': str(path), 'source_request_sha256': sha(path),
                         'source_output': str(requests / (ident + '.json')), 'source_output_sha256': sha(requests / (ident + '.json')),
                         'scene_sha256': episode['scene_sha256']}
                decisions.append(state)
            except (ValueError, KeyError, TypeError) as exc:
                excluded.append({'task': trial['task'], 'request': ident, 'reason': str(exc)})
        evidence.append({'task': trial['task'], 'workflow': trial['workflow'], 'scene_sha256': episode['scene_sha256'],
                         'trial_result_sha256': trial['trial_result_sha256']})
    if not decisions: raise ValueError('No public executable teacher decisions could be admitted')
    out = root / 'decisions.jsonl'
    with out.open('w') as f:
        for row in decisions: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    write(root / 'trace-admission.json', {'teacher_episodes': len(evidence), 'decisions': len(decisions),
          'decision_file_sha256': sha(out), 'scene_audit_sha256': spec['scene_audit_sha256'],
          'episodes': evidence, 'excluded_decisions': excluded, 'exclusion_counts': dict(Counter(r['reason'] for r in excluded)),
          'private_reasoning_exported': False, 'future_observations_exported': False,
          'action_admission_policy': policy,
          'rationale_status': 'Public commentary retained for audit; explicit backward explanations generated separately.'})
    return decisions


def balanced_order(rows, seed=20261027):
    """Round robin suites/tasks, randomize decisions within each task; no replacement."""
    rng = random.Random(seed); groups = defaultdict(lambda: defaultdict(list))
    for r in rows: groups[r['suite']][r['task']].append(r)
    for tasks in groups.values():
        for group in tasks.values(): rng.shuffle(group)
    suites = sorted(groups); cursors = defaultdict(int); result = []
    while any(group for tasks in groups.values() for group in tasks.values()):
        for suite in suites:
            tasks = sorted(t for t, group in groups[suite].items() if group)
            if tasks:
                task = tasks[cursors[suite] % len(tasks)]; cursors[suite] += 1
                result.append(groups[suite][task].pop())
    return result


def encode_example(processor, messages, tools, target, cutoff, *, action_only=False):
    """Locate the target by identical rendering prefixes; never truncate targets."""
    from PIL import Image
    prefix = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    complete = processor.apply_chat_template(messages + [target], tools=tools, tokenize=False, add_generation_prompt=False)
    if not complete.startswith(prefix): raise ValueError('Training/serving chat prefix mismatch')
    images = []
    for message in messages:
        if isinstance(message.get('content'), list):
            for part in message['content']:
                if part['type'] == 'image':
                    images.append(Image.open(io.BytesIO(base64.b64decode(part['image'].split(',', 1)[1]))).convert('RGB'))
    kwargs = {'images': images} if images else {}
    # The pixel limit is shared with the native serving condition.
    full = processor(text=[complete], **kwargs, return_tensors='pt', padding=False, truncation=False)
    prompt = processor(text=[prefix], **kwargs, return_tensors='pt', padding=False, truncation=False)
    start = prompt['input_ids'].shape[1]
    if not full['input_ids'][0, :start].equal(prompt['input_ids'][0]):
        # Empty reasoning merges the prompt's final LF with the blank target LF.
        # Both are masked context in action-only scoring. Training stays strict.
        empty=(target.get('content')=='<think>\n\n</think>\n\n' or target.get('reasoning_content',None)=='')
        merged=(action_only and empty and full['input_ids'][0,:start-1].equal(prompt['input_ids'][0,:start-1])
                and processor.tokenizer.decode(prompt['input_ids'][0,start-1:start])=='\n'
                and processor.tokenizer.decode(full['input_ids'][0,start-1:start])=='\n\n')
        if not merged:raise ValueError('Token prefix mismatch')
    length = full['input_ids'].shape[1]
    if length > cutoff: return None, {'reason': 'complete_example_exceeds_context', 'tokens': length}
    if start >= length: raise ValueError('Empty target')
    target_start=start
    if action_only:
        end_think=processor.tokenizer.convert_tokens_to_ids('</think>')
        closing=[i for i,v in enumerate(full['input_ids'][0].tolist()) if v==end_think and i>=start-1]
        if len(closing)!=1:raise ValueError('Ambiguous reasoning/action boundary')
        target_start=closing[0]+1
    labels = full['input_ids'].clone(); labels[:, :target_start] = -100
    full['labels'] = labels
    return dict(full), {'tokens': length, 'prompt_tokens': start, 'target_start_token':target_start,
                  'target_tokens': length-target_start,'action_only':action_only,
                  'images': len(images), 'full_target_retained': True}
