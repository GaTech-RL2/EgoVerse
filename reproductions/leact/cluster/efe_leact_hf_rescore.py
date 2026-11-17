"""Rescore cached explanations with the exact FSDP training model, without updates."""
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time


def corrected_record(original,scores,source_sha256):
    """Keep all sampled texts, then apply the original positive/top-two rule."""
    expected={-1}|{c['candidate']for c in original['candidates']if c.get('rationale')}
    if set(scores)!=expected:raise ValueError('Incomplete cached-candidate rescoring')
    reference=scores[-1]
    if any(not math.isfinite(s['mean_target_logp'])for s in scores.values()):
        raise ValueError('Nonfinite HF command score')
    contract=(reference['target_tokens'],reference['target_ids_sha256'])
    if any((s['target_tokens'],s['target_ids_sha256'])!=contract for s in scores.values()):
        raise ValueError('Rescored conditions do not share executable command tokens')
    value=deepcopy(original)
    value['original_vllm_record_sha256']=source_sha256
    value['original_vllm_base']=value['base'];value['original_vllm_selected']=value['selected']
    value['base']=reference
    for candidate in value['candidates']:
        if not candidate.get('rationale'):continue
        candidate['original_vllm_score']=candidate['score']
        candidate['original_vllm_delta']=candidate['delta']
        candidate['original_vllm_eligible']=candidate['eligible']
        candidate.pop('full_json_delta',None)
        candidate['score']=scores[candidate['candidate']]
        delta=candidate['score']['mean_target_logp']-reference['mean_target_logp']
        candidate.update(delta=delta,eligible=delta>0,reason=None if delta>0 else 'nonpositive_command_delta')
    selected=sorted([c for c in value['candidates']if c['eligible']],key=lambda c:(-c['delta'],c['candidate']))[:2]
    value['selected']=[c['candidate']for c in selected]
    value['score_backend']='exact_training_fsdp_fp32_master_bf16_forward'
    return value


def processor_for_frozen_inputs(spec,root):
    """Select the processor implementation only by exact frozen tensor equality."""
    import torch
    from transformers import AutoProcessor,AutoConfig
    from trainer import load_tensor,modality_inputs
    from efe_training_data import balanced_order,encode_example
    from leact import physical_positions
    states=balanced_order([json.loads(s)for s in gzip.decompress(
        (Path(spec['parent_corpus'])/'decisions.jsonl.gz').read_bytes()).splitlines()],spec['seed'])
    errors=[]
    config=AutoConfig.from_pretrained(spec.get('local_base_model',spec['base_model']),local_files_only=True)
    for kwargs in [{'use_fast':False},{}]:
        processor=AutoProcessor.from_pretrained(spec.get('local_base_model',spec['base_model']),local_files_only=True,**kwargs)
        processor.image_processor.size['shortest_edge']=65536
        processor.image_processor.size['longest_edge']=1048576
        matched=True
        for probe in json.loads((Path(root)/'scoring-probes.json').read_text()):
            state=states[int(probe['condition'].split('-')[-1])]
            target={'role':'assistant','reasoning_content':'','content':json.dumps(state['action'],ensure_ascii=False)}
            batch,meta=encode_example(processor,state['messages'],[],target,spec['context'],action_only=True)
            positions=physical_positions(processor.tokenizer,batch['input_ids'][0].tolist(),meta['target_start_token'])
            labels=torch.full_like(batch['input_ids'],-100);labels[0,positions]=batch['input_ids'][0,positions];batch['labels']=labels
            saved=modality_inputs(load_tensor(Path(root)/probe['tensor_file']),config)
            different=sorted(set(batch)^set(saved))+[k for k in set(batch)&set(saved)if not torch.equal(batch[k],saved[k])]
            if different:matched=False;errors.append({'options':kwargs,'condition':probe['condition'],'different':different});break
        if matched:return processor,{'options':kwargs,'image_processor_class':type(processor.image_processor).__name__,
            'original_probe_tensors_exactly_equal':True,'hf5_modality_labels':'validated against token IDs with the existing training adapter'}
    raise ValueError('No processor reproduces the frozen original tensors: '+json.dumps(errors))


def run(wrapped,spec,root):
    import torch
    import torch.distributed as dist
    from transformers import AutoProcessor
    from trainer import target_loss,load_tensor,modality_inputs
    from leact import physical_positions
    from efe_training_data import balanced_order,encode_example,sha,write
    from efe_training_state import object_group
    rank=dist.get_rank();world=dist.get_world_size();device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    request=spec['hf_rescore_execution'];out=Path(request['output_root']);out.mkdir(parents=True,exist_ok=True)
    shard=request['shard'];shards=request['shards'];worker=out/f'worker-{shard}'
    worker.mkdir(exist_ok=True);(out/'candidates').mkdir(exist_ok=True)
    source=Path(spec['parent_corpus'])/'decisions.jsonl.gz'
    if sha(source)!=spec['decisions_sha256']:raise ValueError('Original teacher decisions changed')
    states=balanced_order([json.loads(s)for s in gzip.decompress(source.read_bytes()).splitlines()],spec['seed'])
    states=states[:spec['max_balanced_states']]
    if [s['id']for s in states]!=json.loads((root/'membership.json').read_text())['states']:
        raise ValueError('Rescore state membership differs')
    selected=[s for i,s in enumerate(states)if i%shards==shard]
    processor,processor_receipt=processor_for_frozen_inputs(spec,root)
    if rank==0:write(worker/'processor-check.json',processor_receipt)
    originals={};tasks=[]
    for state in selected:
        path=root/'candidates'/(state['id']+'.json');original=json.loads(path.read_text())
        originals[state['id']]=(original,sha(path))
        done=out/'candidates'/path.name
        if done.exists():
            value=json.loads(done.read_text())
            if value.get('original_vllm_record_sha256')!=sha(path):raise ValueError('Previously rescored source changed')
            continue
        tasks.append((state,-1,None))
        tasks.extend((state,c['candidate'],c['rationale'])for c in original['candidates']if c.get('rationale'))
    if not tasks:
        if rank==0:write(worker/'complete.json',{'success':True,'states':len(selected),'optimizer_updates':0})
        return
    # Every rank independently checks one of the original immutable probe tensors
    # against the current processor before any rescoring, including pixel values.
    probes=json.loads((root/'scoring-probes.json').read_text())
    probe=probes[rank%len(probes)];state=states[int(probe['condition'].split('-')[-1])]
    saved=modality_inputs(load_tensor(root/probe['tensor_file']),wrapped.config)
    def encode(state,rationale):
        target={'role':'assistant','reasoning_content':rationale or '','content':json.dumps(state['action'],ensure_ascii=False)}
        batch,meta=encode_example(processor,state['messages'],[],target,spec['context'],action_only=True)
        if batch is None:raise ValueError('Frozen cached explanation no longer fits complete context')
        ids=batch['input_ids'][0].tolist();positions=physical_positions(processor.tokenizer,ids,meta['target_start_token'])
        labels=torch.full_like(batch['input_ids'],-100);labels[0,positions]=batch['input_ids'][0,positions]
        batch['labels']=labels
        return batch,positions
    current,_=encode(state,None)
    if set(current)!=set(saved)or any(not torch.equal(current[k],saved[k])for k in saved):
        raise ValueError('Rescore processor changed the frozen original probe tensors')
    wrapped.eval()
    with torch.no_grad():initial=-target_loss(wrapped,{k:v.to(device)for k,v in current.items()}).item()
    expected=request['hf_reference_logp'][probe['condition']]
    error=abs(initial-expected)
    errors=[None]*world;dist.all_gather_object(errors,error,group=object_group())
    if not all(math.isfinite(e)and e<=.0005 for e in errors):
        raise ValueError('HF rescore model does not reproduce the independently measured training score')
    if rank==0:write(worker/'reference-check.json',{'passed':True,'absolute_errors':errors,
        'tolerance':.0005,'frozen_input_tensors_equal':True,'optimizer_updates':0})
    scores={};checked_states=set();started=time.monotonic();published=0
    for offset in range(0,len(tasks),world):
        index=offset+rank;state,candidate,rationale=tasks[min(index,len(tasks)-1)]
        original,_=originals[state['id']]
        batch,positions=encode(state,rationale)
        if state['id']not in checked_states and original['selected']:
            chosen=next(c for c in original['candidates']if c['candidate']==original['selected'][0])
            reference=load_tensor(root/'tensors'/(state['id']+'-leact-'+str(chosen['candidate'])+'-forward.pt'))
            check,_=encode(state,chosen['rationale'])
            keys=set(reference)-{'labels'}
            if not keys<=set(check)or any(not torch.equal(check[k],reference[k])for k in keys):
                raise ValueError('Cached rationale or observation differs from its original training tensor')
            checked_states.add(state['id'])
        ids=batch['input_ids'][0,positions].tolist();target_sha=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        prior=original['base']if candidate==-1 else next(c['score']for c in original['candidates']if c['candidate']==candidate)
        if (target_sha!=prior['target_ids_sha256']or len(positions)!=prior['target_tokens']
                or batch['input_ids'].shape[1]!=prior['complete_prompt_tokens']):
            raise ValueError('Scored token membership or complete context changed')
        with torch.no_grad():value=-target_loss(wrapped,{k:v.to(device)for k,v in batch.items()}).item()
        if not math.isfinite(value):raise ValueError('Nonfinite complete-command HF score')
        row={'state_id':state['id'],'candidate':candidate,'score':{'mean_target_logp':value,
            'target_tokens':len(positions),'target_ids_sha256':target_sha,'complete_prompt_tokens':batch['input_ids'].shape[1],
            'backend':'training_hf_fsdp','full_json_score_measured':False}}
        gathered=[None]*world;dist.all_gather_object(gathered,row,group=object_group())
        if rank==0:
            for j,item in enumerate(gathered):
                if offset+j>=len(tasks):continue
                bucket=scores.setdefault(item['state_id'],{});bucket[item['candidate']]=item['score']
                orig,digest=originals[item['state_id']]
                expected=1+sum(bool(c.get('rationale'))for c in orig['candidates'])
                if len(bucket)==expected:
                    record=corrected_record(orig,bucket,digest)
                    write(out/'candidates'/(item['state_id']+'.json'),record);published+=1
                    del scores[item['state_id']]
            progress={'completed_states_this_attempt':published,'states_assigned':len(selected),
                'scored_conditions':min(offset+world,len(tasks)),'conditions_total':len(tasks),
                'seconds':time.monotonic()-started,'optimizer_updates':0,'workflow':os.environ.get('EFE_WORKFLOW_ID'),
                'updated_utc':datetime.now(timezone.utc).isoformat()}
            write(worker/'progress.json',progress)
            if offset%(world*16)==0:print('EFE_HF_RESCORE '+json.dumps(progress),flush=True)
    if rank==0:
        if scores:raise ValueError('Unpublished incomplete HF score groups')
        completed=sum((out/'candidates'/(s['id']+'.json')).exists()for s in selected)
        if completed!=len(selected):raise ValueError('Incomplete assigned HF rescoring')
        write(worker/'complete.json',{'success':True,'states':completed,'optimizer_updates':0,
            'workflow':os.environ.get('EFE_WORKFLOW_ID'),'completed_utc':datetime.now(timezone.utc).isoformat()})
