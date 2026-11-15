"""Freeze one complete native PEAR group, preserving original sampled token IDs.

This is a discarded base-policy optimizer qualification, not a primary RL arm.
Run preprocessing with the sampler's pinned Transformers 4.57.6 processor.
"""
import argparse
import base64
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path


def schedule_rows(rows, world=8):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate captured requests')
    if any(type(r['generated_tokens']) is not int or r['generated_tokens'] < 1 for r in rows):
        raise ValueError('All complete requests need generated tokens')
    ordered = [dict(r, weight=1) for r in sorted(rows, key=lambda r:(r['tokens'], r['id']), reverse=True)]
    # Dummy exposures keep FSDP collective order identical. They have no loss
    # mass and never enter the generated-token denominator.
    while len(ordered) % world:
        ordered.append(dict(ordered[-1], weight=0))
    return ordered, sum(r['generated_tokens'] for r in rows)


def captured_path(source, capture):
    path = Path(capture['file'])
    if not path.is_absolute():
        path = Path(source)/path
    if not path.resolve().is_relative_to(Path(source).resolve()):
        raise ValueError('Capture escaped its pinned native group')
    return path


def prepare(spec, root):
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from efe_training_data import chat_inputs, sha, write
    from efe_visual_tensor_store import save
    from efe_pear_loss import validate_sampling
    from efe_pear_groups import admit_group
    from efe_pear_sampler import digest_body
    if importlib.metadata.version('transformers') != '4.57.6':
        raise ValueError('Original sampler processor version required')
    source = Path(spec['input']['group_root'])
    for name, key in [('pear-episodes.json','episodes_sha256'), ('pear-native-result.json','group_result_sha256')]:
        if sha(source/name) != spec['input'][key]:
            raise ValueError('Frozen native evidence changed: '+name)
    episodes = json.loads((source/'pear-episodes.json').read_text())['episodes']
    membership_path = Path(spec['membership'])
    if sha(membership_path) != spec['membership_sha256']:
        raise ValueError('Frozen training membership changed')
    group = admit_group(episodes, json.loads(membership_path.read_text()), spec['base_revision'])
    if not group['admitted'] or group != spec['input']['group']:
        raise ValueError('A complete, informative and unchanged native group is required')
    processor = AutoProcessor.from_pretrained(spec['local_base_model'], local_files_only=True)
    processor.image_processor.size.update(spec['image_processor_size'])
    rows, seen_requests = [], set()
    for episode, advantage in zip(episodes, group['advantages']):
        for capture in episode['captures']:
            path = captured_path(source, capture)
            if sha(path) != capture['sha256']:
                raise ValueError('Original sampler capture changed')
            record = json.loads(gzip.decompress(path.read_bytes()))
            count = validate_sampling(record, len(processor.tokenizer), 32)
            if record['id'] in seen_requests or record['id'] != capture['id'] or count != capture['tokens']:
                raise ValueError('Duplicate or misjoined native sampler request')
            seen_requests.add(record['id'])
            if record['sampler_revision'] != spec['base_revision'] or not record['sampler_capture_complete']:
                raise ValueError('Incomplete or stale sampler policy')
            if digest_body(record['request']) != record['request_sha256']:
                raise ValueError('Original public observation changed')
            ident = hashlib.sha256((episode['episode_id']+'/'+record['id']).encode()).hexdigest()
            tensor_path, receipt_path = root/'tensors'/(ident+'.pt'), root/'receipts'/(ident+'.json')
            if receipt_path.exists():
                row = json.loads(receipt_path.read_text())
                if row['capture_sha256'] != capture['sha256'] or sha(tensor_path) != row['tensor_sha256']:
                    raise ValueError('Prepared tensor no longer matches the original capture')
                if row['advantage'] != advantage:
                    raise ValueError('Cached reward advantage differs')
            else:
                messages = chat_inputs(record['request'])
                images = [Image.open(io.BytesIO(base64.b64decode(part['image'].split(',',1)[1]))).convert('RGB')
                    for m in messages if isinstance(m.get('content'),list) for part in m['content'] if part['type']=='image']
                prompt = processor.apply_chat_template(messages, tools=[], tokenize=False, add_generation_prompt=True)
                batch = dict(processor(text=[prompt], images=images, return_tensors='pt'))
                if batch['input_ids'][0].tolist() != record['prompt_token_ids']:
                    raise ValueError('Sampler and trainer multimodal prompt IDs differ')
                generated = torch.tensor([record['token_ids']], dtype=batch['input_ids'].dtype)
                batch['labels'] = torch.cat([torch.full_like(batch['input_ids'],-100), generated], -1)
                batch['input_ids'] = torch.cat([batch['input_ids'], generated], -1)
                batch['attention_mask'] = torch.ones_like(batch['input_ids'])
                if 'mm_token_type_ids' in batch:
                    batch['mm_token_type_ids'] = torch.cat([batch['mm_token_type_ids'], torch.zeros_like(generated)],-1)
                if batch['input_ids'].shape[1] > spec['context']:
                    raise ValueError('Full sequence exceeds context; truncation is forbidden')
                batch.update(pear_logq=torch.tensor(record['sampled_logprobs'],dtype=torch.float32),
                    pear_topk_ids=torch.tensor(record['topk_token_ids'],dtype=torch.long),
                    pear_topk_logq=torch.tensor(record['topk_logprobs'],dtype=torch.float32),
                    pear_advantage=torch.tensor(advantage,dtype=torch.float32))
                save(batch,tensor_path)
                row = {'id':ident, 'episode_id':episode['episode_id'], 'request_id':record['id'],
                    'capture_sha256':capture['sha256'], 'tensor_file':str(tensor_path.relative_to(root)),
                    'tensor_sha256':sha(tensor_path), 'generated_tokens':count,
                    'tokens':batch['input_ids'].shape[1], 'advantage':advantage,
                    'finish_reason':record['finish_reason']}
                write(receipt_path,row)
            rows.append(row)
            if len(rows)%16==0:
                print('PEAR_UPDATE_DATA '+json.dumps({'requests_prepared':len(rows)}),flush=True)
    schedule, total = schedule_rows(rows)
    if len(rows) != spec['input']['requests'] or total != spec['input']['generated_tokens']:
        raise ValueError('Native calls or generated targets were lost during preprocessing')
    result = {'examples':rows,'group':group,'generated_tokens':total,
        'requests':len(rows),'scheduled_microbatches':len(schedule)//8,
        'sampler_revision':spec['base_revision'], 'source':spec['input'],
        'processor_version':importlib.metadata.version('transformers'),
        'truncated_targets':0,'tokenization':'Original sampled IDs appended to exact verified multimodal prompt'}
    write(root/'dataset-manifest.json',result)
    print('PEAR_UPDATE_DATA '+json.dumps({k:result[k] for k in ['generated_tokens','requests','scheduled_microbatches','truncated_targets']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();prepare(json.loads(a.spec.read_text()),a.root)
