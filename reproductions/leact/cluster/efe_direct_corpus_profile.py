"""CPU-only complete-history sizing of scene-admitted direct-control traces."""
import argparse
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import tarfile
import traceback

from efe_training_data import extract_decisions, forward_target, backward_messages, encode_example, sha, write


def profile(spec_path):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    torch.set_num_threads(2)
    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    root = spec_path.parent
    admitted = []
    for episode in spec['episodes']:
        bundle = Path(episode['bundle']['path'])
        if bundle.stat().st_size != episode['bundle']['bytes'] or sha(bundle) != episode['bundle']['sha256']:
            raise ValueError('Raw episode bundle changed')
        destination = root / 'raw' / episode['trial']['workflow']
        with tarfile.open(bundle) as archive:
            for member in archive.getmembers():
                if member.issym() or member.islnk() or not member.isfile():
                    raise ValueError('Nonregular retained bundle member')
                relative = Path(member.name)
                if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in {'requests', 'jobs'}:
                    raise ValueError('Unexpected retained bundle path')
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream:
                    target.write_bytes(stream.read())
        audit = root / 'action-audits' / (episode['episode'].replace('/', '--') + '.json')
        # Preserve the original checksum-verified bytes, not reserialized JSON.
        audit.parent.mkdir(exist_ok=True)
        audit.write_bytes(base64.b64decode(episode['action_audit_base64']))
        if sha(audit) != episode['action_audit_sha256']:
            raise ValueError('Executed-call audit payload changed')
        trial = dict(episode['trial'], archive=str(destination / 'jobs' / episode['trial']['job']))
        admitted.append({**episode['scene'], 'trial': trial,
            'original_trial': episode['trial'], 'bundle': episode['bundle'],
            'action_audit': str(audit), 'action_audit_sha256': episode['action_audit_sha256']})
    scene = root / 'scene-admission.json'
    write(scene, {'eligible_episodes': admitted, 'scene_audit_sha256': spec['scene_audit_sha256'],
        'scope': 'Raw paths rebased after exact-bundle verification; original trial receipts retained.'})
    extraction = {'scene_audit': str(scene), 'scene_audit_sha256': sha(scene),
        'student_prompt_requests': str(root / 'student-prompt'),
        'action_admission': 'executed-direct-control-v1',
        'instruction_suffix_file': str(root / 'method.txt'),
        'instruction_suffix_sha256': spec['instruction_suffix_sha256']}
    write(root / 'extraction-spec.json', extraction)
    states = extract_decisions(extraction, root)
    processor = AutoProcessor.from_pretrained(spec['base_model'], local_files_only=True)
    processor.image_processor.max_pixels = 1048576
    processor.image_processor.size['longest_edge'] = 1048576
    write(root / 'processor.json', processor.image_processor.to_dict())
    image_cache = {}
    rows, validation = [], []
    short = 'Explain the intended effect from the current observation.'
    reserve = 512

    def grids(messages):
        result = []
        for message in messages:
            content = message.get('content')
            for item in content if isinstance(content, list) else []:
                if item['type'] != 'image':
                    continue
                data = base64.b64decode(item['image'].split(',', 1)[1])
                digest = hashlib.sha256(data).hexdigest()
                if digest not in image_cache:
                    image = Image.open(io.BytesIO(data)).convert('RGB')
                    encoded = processor.image_processor(images=[image], return_tensors='pt')
                    pixels = encoded['pixel_values']
                    image_cache[digest] = {'grid': encoded['image_grid_thw'][0].tolist(),
                        'bytes': pixels.numel() * pixels.element_size(), 'dimensions': list(image.size),
                        'pixel_sha256': hashlib.sha256(pixels.contiguous().numpy().tobytes()).hexdigest()}
                result.append(image_cache[digest])
        return result

    def tokens(messages, tools, target, images):
        text = processor.apply_chat_template(messages + [target], tools=tools, tokenize=False, add_generation_prompt=False)
        if text.count(processor.image_token) != len(images):
            raise ValueError('Rendered image placeholder count differs')
        for image in images:
            size = 1
            for n in image['grid']:
                size *= n
            text = text.replace(processor.image_token, '<|placeholder|>' * (size // processor.image_processor.merge_size**2), 1)
        text = text.replace('<|placeholder|>', processor.image_token)
        return processor.tokenizer([text], padding=False, truncation=False, return_tensors='pt')['input_ids']

    seen_suites = set()
    for index, state in enumerate(states):
        images = grids(state['messages'])
        forward = forward_target(state['action'], short)
        backward = {'role': 'assistant', 'content': '<think>\n' + short + '\n</think>\n\nDONE'}
        backward_input = backward_messages(state['messages'], state['action'])
        forward_ids = tokens(state['messages'], state['tools'], forward, images)
        backward_ids = tokens(backward_input, state['tools'], backward, images)
        row = {'state_id': state['id'], 'suite': state['suite'], 'task': state['task'], 'episode': state['episode'],
            'forward_tokens': forward_ids.shape[1], 'backward_tokens': backward_ids.shape[1],
            'images': len(images), 'pixel_bytes_per_direction': sum(r['bytes'] for r in images)}
        if index < 3 or state['suite'] not in seen_suites:
            for direction, messages, target, expected in [
                ('forward', state['messages'], forward, forward_ids),
                ('backward', backward_input, backward, backward_ids)]:
                encoded, receipt = encode_example(processor, messages, state['tools'], target, 1000000)
                if encoded is None or not torch.equal(expected, encoded['input_ids']):
                    raise ValueError('Cached profile differs from full HF processor')
                validation.append({'state_id': state['id'], 'direction': direction, 'tokens': receipt['tokens'],
                                   'exact_input_ids_match': True})
            seen_suites.add(state['suite'])
        rows.append(row)
        if index % 20 == 0:
            write(root / 'progress.json', {'completed_states': index + 1, 'total_states': len(states),
                'unique_images': len(image_cache), 'phase': 'complete_history_profile'})
    values = sorted(max(r['forward_tokens'], r['backward_tokens']) + reserve for r in rows)
    coverage = {}
    for context in [32768, 65536, 131072]:
        eligible = [r for r in rows if max(r['forward_tokens'], r['backward_tokens']) + reserve <= context and r['images'] <= 64]
        full = Counter(r['episode'] for r in rows)
        partial = Counter(r['episode'] for r in eligible)
        coverage[str(context)] = {'complete_decisions': len(eligible), 'total_decisions': len(rows),
            'complete_episodes': sum(partial[e] == n for e, n in full.items()), 'total_episodes': len(full),
            'tasks': len({r['task'] for r in eligible}),
            'excluded_state_ids': [r['state_id'] for r in rows if r not in eligible]}
    result = {'snapshot': spec['snapshot'], 'teacher_episodes': len(admitted), 'decisions': len(rows),
        'tasks': len({r['task'] for r in rows}), 'suites': sorted({r['suite'] for r in rows}),
        'context_tokens_with_512_reserve': {'min': values[0], 'median': values[len(values)//2],
            'p95': values[min(len(values)-1, int(len(values)*.95))], 'max': values[-1]},
        'max_images': max(r['images'] for r in rows), 'unique_images': len(image_cache),
        'pixel_bytes_duplicated_forward_backward': 2 * sum(r['pixel_bytes_per_direction'] for r in rows),
        'pixel_bytes_unique_images': sum(r['bytes'] for r in image_cache.values()),
        'coverage': coverage, 'exact_hf_profile_validation': validation,
        'training_deployment_parity_checked': False, 'student_training_launched': False,
        'scope': 'Exact complete-history token sizing using the HF training image cap, with a 512-token rationale reserve. '
                 'Token IDs checked against full HF encoding on the first three states and each suite. '
                 'This is neither a training run nor a training/deployment likelihood parity check.'}
    write(root / 'states-profile.json', rows)
    write(root / 'images-profile.json', image_cache)
    write(root / 'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    try:
        profile(args.spec)
    except Exception as error:
        write(Path(args.spec).parent / 'error.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
