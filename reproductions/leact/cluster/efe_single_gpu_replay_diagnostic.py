"""Read-only full-model replay, independently of FSDP initialization/collectives."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def run(root, output):
    import torch
    from transformers import AutoModelForImageTextToText
    from trainer import load_tensor, target_loss
    from efe_qwen36_adapter_io import attach_exact, load_into_exact
    from efe_training_data import sha
    from safetensors.torch import load_file
    spec=json.loads((root/'spec.json').read_text())
    reference=json.loads((root/'smoke/complete.json').read_text())['post_update_full_target_ce']
    model=AutoModelForImageTextToText.from_pretrained(spec.get('local_base_model',spec['base_model']),
        dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',local_files_only=True)
    model.config.use_cache=False
    model=attach_exact(model,root/'smoke/checkpoint-2');model.eval()
    rows={r['id']:r for r in json.loads((root/'dataset-manifest.json').read_text())['examples']}
    selected=json.loads((root/'smoke/schedule.json').read_text())['exposures'][:8]
    def score():
        cached={}
        for ident in selected:
            if ident in cached:continue
            row=rows[ident];path=root/row['tensor_file']
            if sha(path)!=row['tensor_sha256']:raise ValueError('Frozen replay input changed')
            batch={k:v.to('cuda:0') for k,v in load_tensor(path).items()}
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                value=target_loss(model,batch).item()
            cached[ident]=value
        return sum(cached[k] for k in selected)/len(selected)
    actual=score()
    result={'utc':datetime.now(timezone.utc).isoformat(),'kind':'single-gpu-bf16-diagnostic',
        'optimizer_updates':0,'expected_smoke_ce':reference,'actual_smoke_ce':actual,
        'absolute_error':abs(actual-reference),'torch':torch.__version__,'cuda':torch.version.cuda,
        'note':'Independent diagnostic. This BF16 single-GPU path is not the production FP32/FSDP replay gate.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    suspect=root/'attempts/leact-efe-coverage-warmstart-v1-p6-1/loss-spike-checkpoints/checkpoint-22'
    load_into_exact(model,suspect);model.eval();result['step22_smoke_input_ce']=score()
    norms={}
    for label,path in [('discarded_smoke',root/'smoke/checkpoint-2'),('suspect_step22',suspect)]:
        state=load_file(str(path/'adapter_model.safetensors'),device='cpu')
        norms[label]={}
        for part in ['lora_A','lora_B']:
            tensors=[v.float() for k,v in state.items() if part in k]
            norms[label][part]={'rms':(sum(v.square().sum().item() for v in tensors)/sum(v.numel() for v in tensors))**.5,
                              'max_abs':max(v.abs().max().item() for v in tensors)}
    result.update(adapter_norms=norms,completed_utc=datetime.now(timezone.utc).isoformat())
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-lane',type=Path,required=True)
    args=parser.parse_args();sys.path.insert(0,str(args.training_lane))
    try:run(args.root,args.output)
    except Exception as error:
        args.output.with_suffix('.error.json').write_text(json.dumps({'error':repr(error),'utc':datetime.now(timezone.utc).isoformat()})+'\n')
        raise
