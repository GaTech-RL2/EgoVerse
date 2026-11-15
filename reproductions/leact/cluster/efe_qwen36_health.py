"""Real-model loss checks that gate initialization, resume and checkpoint saves."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random


def measure(wrapped, root, rows):
    import torch
    import torch.distributed as dist
    from trainer import load_tensor,target_loss
    from efe_training_data import sha
    from efe_training_state import object_group
    saved=(torch.get_rng_state(),torch.cuda.get_rng_state(),random.getstate(),wrapped.training)
    try:
        row=rows[dist.get_rank()];path=Path(root)/row['tensor_file']
        if sha(path)!=row['tensor_sha256']:raise ValueError('Full-model replay input changed')
        batch={k:v.to(torch.device('cuda',int(os.environ['LOCAL_RANK']))) for k,v in load_tensor(path).items()}
        wrapped.eval()
        with torch.no_grad():value=target_loss(wrapped,batch).item()
        values=[None]*dist.get_world_size()
        dist.all_gather_object(values,value,group=object_group())
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite full-model replay loss')
        return sum(values)/len(values),values
    finally:
        torch.set_rng_state(saved[0]);torch.cuda.set_rng_state(saved[1]);random.setstate(saved[2])
        wrapped.train(saved[3])


def validate(wrapped,spec,root,schedule,start_step):
    import torch.distributed as dist
    from efe_training_data import write
    root=Path(root)
    if start_step:
        expected=spec['active_restore']['receipt'].get('post_update_full_target_ce')
        if expected is None:raise ValueError('Resume checkpoint lacks a real-model replay reference')
        selected=schedule
        kind='resume'
    elif spec['stage']=='warmstart':
        first=json.loads((root/'smoke/metrics.jsonl').read_text().splitlines()[0])
        expected=first['forward_ce'] if first['forward_examples'] else first['backward_ce']
        lookup={r['id']:r for r in json.loads((root/'dataset-manifest.json').read_text())['examples']}
        selected=[lookup[k] for k in json.loads((root/'smoke/schedule.json').read_text())['exposures'][:dist.get_world_size()]]
        kind='untouched_base'
    else:
        return  # LeAct additionally checks its fixed selection/training probes.
    actual,values=measure(wrapped,root,selected);error=abs(actual-expected)
    result={'kind':kind,'step':start_step,'expected_ce':expected,'actual_ce':actual,
        'per_rank_ce':values,'absolute_error':error,'tolerance':.0005,'passed':error<=.0005,
        'optimizer_updates':0,'workflow':os.environ.get('EFE_WORKFLOW_ID'),
        'utc':datetime.now(timezone.utc).isoformat()}
    if dist.get_rank()==0:write(root/'full-model-replay.json',result)
    if not result['passed']:raise ValueError('Full-model replay failed before optimizer updates: '+json.dumps(result))


def suspicious_loss(current,previous,factor=5.0):
    return bool(previous) and all(current[key]>max(5.0,factor*previous[key])
                                  for key in ['forward_ce','backward_ce'])


def training_reference(spec):
    from efe_training_data import sha
    reference=spec.get('reference_training_metrics')
    if not reference:return {}
    path=Path(reference['path'])
    if sha(path)!=reference['sha256']:raise ValueError('Trusted training replay reference changed')
    value=json.loads(path.read_text())
    if value['lineage_sha256']!=spec['durable_restart']['lineage_sha256']:
        raise ValueError('Trusted replay belongs to a different training recipe')
    return {row['step']:row for row in value['metrics']}


def check_training_reference(current,expected,step):
    if expected is None:return
    for key in ['forward_ce','backward_ce']:
        if abs(current[key]-expected[key])>max(.02,.02*abs(expected[key])):
            raise ValueError('Trusted training replay differs before optimizer update: '+json.dumps(
                {'step':step,'current':current,'expected':{k:expected[k] for k in ['forward_ce','backward_ce']},
                 'absolute_tolerance_floor':.02,'relative_tolerance':.02}))
