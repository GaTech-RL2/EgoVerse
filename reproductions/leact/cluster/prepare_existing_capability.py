"""Register completed BuilderBench weights for evaluation-only capability checks.

This does not resume BuilderBench training or simulator rollouts. Its original
stop marker remains authoritative for those jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .monitor_efe_preflight import remote
from .launch_efe_capability import SHARED, build_run, transfer
from ..embodied_first_exam.capability_retention import DEST, prepare
from ..embodied_first_exam.report import BASE, read, write

FAMILY = 'qwen3-8b-existing'
STAGES = [
    ('base', 'text-qwen-base', 'Untouched Qwen3-8B (text)', None),
    ('joint-e3', 'bb-warmstart-e3', 'BuilderBench: frontier reasoning SFT / warmstart', 'text-qwen-base'),
    ('leact-r1-e3', 'bb-leact-r1-e3', 'BuilderBench: warmstart + LeAct, round 1', 'bb-warmstart-e3'),
    ('bc-e3', 'bb-bc-e3', 'BuilderBench: action-only SFT control', 'text-qwen-base'),
]


def prepare_existing(via_workflow):
    common = prepare()
    family_dir = DEST / 'families' / FAMILY
    (family_dir / 'manifests').mkdir(parents=True, exist_ok=True)
    source_path = BASE / 'artifacts/builderbench/student-status.json'
    source = read(source_path)
    checkpoints = {r['id']: r for r in source['checkpoints']}
    records = []
    for original_id, stage_id, name, parent in STAGES:
        original = checkpoints[original_id]
        if not original['final_epoch']:
            raise ValueError('Only prespecified final epochs are included')
        records.append({'id': stage_id, 'name': name, 'parent': parent,
                        'original_id': original_id, 'path': original['checkpoint'],
                        'expected': original['checkpoint_sha256'],
                        'manifest': SHARED + '/checkpoint-manifests/' + FAMILY + '/' + stage_id + '.json'})
    script = 'records=' + repr(records) + '\n' + '''
from pathlib import Path
import hashlib,json,time
out=[]
for record in records:
 root=Path(record['path'])
 config=json.loads((root/'config.json').read_text())
 if config['model_type']!='qwen3':raise ValueError('Unexpected model architecture')
 index=json.loads((root/'model.safetensors.index.json').read_text())
 if set(index['weight_map'].values())!=set(record['expected']):raise ValueError('Published shard set differs from checkpoint index')
 files={}
 for p in sorted(root.iterdir()):
  if p.name.endswith('.safetensors') or p.name in {'config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt'}:
   if not p.is_file():raise ValueError('Checkpoint file unavailable: '+p.name)
   digest=record['expected'][p.name] if p.name.endswith('.safetensors') else hashlib.sha256(p.read_bytes()).hexdigest()
   files[p.name]={'bytes':p.stat().st_size,'sha256':digest,'resolved_path':str(p.resolve())}
 manifest={'model':'Qwen/Qwen3-8B','base_revision':'b968826d9c46dd6066d109eabc6255188de91218','checkpoint':record['id'],'path':str(root),'files':files,
           'weight_hash_source':'Published completed-training checkpoint receipt; every byte is rehashed by the evaluator before generation.',
           'selection':'Untouched base or fixed final epoch 3; not selected using capability benchmark scores.'}
 dest=Path(record['manifest']);dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if json.loads(dest.read_text())!=manifest:raise ValueError('Existing checkpoint manifest changed')
 else:
  temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\\n');temp.replace(dest)
 out.append({'id':record['id'],'manifest':manifest,'manifest_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
s='EFE_CAPTURE '+json.dumps({'records':out})+'\\n'
for i in range(0,len(s),1000):print(s[i:i+1000],end='',flush=True);time.sleep(.02)
'''
    verified = remote(via_workflow, script)
    for row in verified['records']:
        write(family_dir / 'manifests' / (row['id'] + '.json'), row['manifest'])
    suite_path = family_dir / 'suite.json'
    if not suite_path.exists():
        plan = {**common, 'id': 'qwen3-8b-existing-capability-v1',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'model': 'Qwen/Qwen3-8B', 'base_revision': 'b968826d9c46dd6066d109eabc6255188de91218',
                'stages': [r['id'] for r in records],
                'stage_note': 'Existing text-Qwen weights: untouched base, frontier reasoning SFT/warmstart, LeAct round 1, and action-only SFT. All trained checkpoints are fixed epoch 3. EFE uses a different VL model and is compared separately.',
                'selection': {**common['selection'], 'checkpoint_rule': 'Fixed final training epochs, before any capability evaluation.'}}
        write(suite_path, plan)
    registry = read(DEST / 'checkpoints.json')
    family = {'name': 'Existing Qwen3-8B text checkpoints', 'model': 'Qwen/Qwen3-8B',
              'base_checkpoint': 'text-qwen-base', 'text_only': True,
              'suite': str(suite_path.relative_to(DEST)),
              'bundle': str((family_dir / 'frozen-inputs.tar.gz').relative_to(DEST)),
              'runner': 'cluster/text_capability_eval.py'}
    previous_family = registry.setdefault('families', {}).get(FAMILY)
    if previous_family and previous_family != family:
        raise ValueError('Existing family registration changed')
    registry['families'][FAMILY] = family
    existing = {r['id']: r for r in registry['checkpoints']}
    for record in records:
        row = {k: record[k] for k in ['id', 'name', 'parent', 'path', 'manifest']}
        row.update(status='ready', family=FAMILY, base_id='text-qwen-base', build_version='v1',
                   selection='Untouched base' if record['original_id']=='base' else 'Fixed final epoch 3',
                   source_study='BuilderBench; weights reused only for general capability evaluation')
        if row['id'] in existing:
            if any(existing[row['id']].get(k) != row[k] for k in ['path','manifest','parent','family']):
                raise ValueError('Existing checkpoint registration changed')
        else:
            registry['checkpoints'].append(row)
    write(DEST / 'checkpoints.json', registry)
    builds = [build_run(r['id']) for r in records]
    protocols = {read(b / 'spec.json')['protocol_sha256'] for b in builds}
    if len(protocols) != 1:
        raise ValueError('Matched checkpoints use different evaluation protocols')
    transfer(builds[0], via_workflow)
    receipt = read(builds[0] / 'input-transfer.json')
    for build in builds[1:]:
        if read(build / 'spec.json')['bundle_sha256'] != receipt['sha256']:
            raise ValueError('Input bundle differs between checkpoints')
        write(build / 'input-transfer.json', receipt)
    write(family_dir / 'registration.json', {
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'source_receipt_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'checkpoints': [{k:v for k,v in r.items() if k!='expected'} for r in records],
        'protocol_sha256': protocols.pop(), 'builds': [str(b) for b in builds],
        'purpose': 'General capability regression only. BuilderBench simulator/training stop remains in force.',
        'allocation': 'One capability H100 node at a time; reserve two node slots for main EFE work.'})
    return builds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--via-workflow', required=True)
    args = parser.parse_args()
    print(json.dumps({'builds': [str(b) for b in prepare_existing(args.via_workflow)]}))


if __name__ == '__main__':
    main()
