"""Merge a verified LoRA checkpoint into a complete native-serving model."""
import argparse
import json
from pathlib import Path
import shutil

from efe_training_data import sha, write


def main():
    import torch
    from efe_qwen36_adapter_io import attach_exact
    from transformers import AutoModelForImageTextToText, AutoProcessor
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);args=p.parse_args()
    spec=json.loads(args.spec.read_text());root=Path(spec['archive_root'])
    replay=json.loads((root/'train-loss-replay.json').read_text())
    if not replay['passed']:raise ValueError('Reloaded final checkpoint must reproduce the recorded training loss')
    ready=json.loads((root/spec.get('ready_name','adapter-ready.json')).read_text())
    manifest=json.loads(Path(ready['manifest']).read_text())
    for name,row in manifest['files'].items():
        if sha(Path(ready['checkpoint'])/name)!=row['sha256']:raise ValueError('Adapter checksum mismatch')
    model=AutoModelForImageTextToText.from_pretrained(spec.get('local_base_model',spec['base_model']),dtype=torch.bfloat16,
        device_map='cpu',attn_implementation='sdpa',local_files_only=True)
    adapted=attach_exact(model,ready['checkpoint'],root/'export-adapter-verification.json')
    loaded={name:p.detach().clone() for name,p in adapted.named_parameters() if 'lora_B' in name}
    if not loaded or not any(p.abs().max().item()>0 for p in loaded.values()):raise ValueError('Reloaded adapter has no learned update')
    witness=None
    for name,module in adapted.named_modules():
        if not hasattr(module,'get_delta_weight'):continue
        delta=module.get_delta_weight('default').detach()
        index=delta.abs().flatten().argmax().item()
        if delta.flatten()[index].abs().item()>0:
            layer=module.get_base_layer();before=layer.weight.detach().flatten()[index].float().item()
            if (layer.weight.detach().float()+delta.float()).to(layer.weight.dtype).flatten()[index].float().item()!=before:
                witness=(name,layer,index,before);break
    if witness is None:raise ValueError('Adapter merge would not change a witnessed native weight')
    merged=adapted.merge_and_unload(safe_merge=True)
    if witness[1].weight.detach().flatten()[witness[2]].float().item()==witness[3]:raise ValueError('Native merged weight did not change')
    destination=root/'model';destination.mkdir(exist_ok=False)
    merged.config.use_cache=True
    merged.save_pretrained(destination,safe_serialization=True,max_shard_size='4GB')
    # Preserve the exact base serving metadata across the training-library upgrade.
    # LoRA changes weights only; the native vLLM evaluation keeps its original tokenizer/processor.
    for asset in Path(spec['base_model']).iterdir():
        if asset.is_file() and asset.suffix in {'.json','.txt','.jinja','.model'} and asset.name!='model.safetensors.index.json':
            shutil.copyfile(asset,destination/asset.name)
    files={f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in destination.iterdir() if f.is_file()}
    index=json.loads((destination/'model.safetensors.index.json').read_text())
    if not set(index['weight_map'].values()).issubset(files):raise ValueError('Merged model lacks weight shards')
    base_index=json.loads((Path(spec['base_model'])/'model.safetensors.index.json').read_text())['weight_map']
    expected={key for key in base_index if not key.startswith('mtp.')}
    if set(index['weight_map'])!=expected:
        write(root/'export-key-mismatch.json',{'missing':sorted(expected-set(index['weight_map'])),
            'unexpected':sorted(set(index['weight_map'])-expected)})
        raise ValueError('Exported model keys differ from the original serving checkpoint')
    steps=ready['step']
    revision=spec.get('checkpoint_revision_prefix','efe-qwen36-constrained-warm-lora-v1')+'-step'+str(steps)
    write(destination/'checkpoint-manifest.json',{'revision':revision,'files':files,'step':steps,
        'source_base':spec['model'],'source_revision':spec['base_revision'],'adapter_checkpoint':ready['checkpoint'],
        'full_parameters':False,'merged_adapter':True,'adapter_reload_verified':True,'merged_weight_update_verified':witness[0],
        'loss_replay':replay,'dataset_sha256':sha(root/'dataset-manifest.json')})
    write(root/spec.get('export_ready_name','warmstart-ready.json'),{'checkpoint':str(destination),'manifest':str(destination/'checkpoint-manifest.json'),
        'revision':revision,'step':steps,'final':True,'selection':f'Fixed {steps}-update pilot budget; no selection on evaluation results.'})


if __name__=='__main__':main()
