"""Stable rp_jhh_ checkpoint names without changing existing training lineage."""
from datetime import datetime, timezone
import json
from pathlib import Path

PREFIX='rp_jhh_'
COVERAGE='/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-coverage-20261105-v1'
PEAR='/mnt/amlfs-07/shared/rpunamiya/leact/studies/efe-pear-scaled-20261116-v1'
REGISTRY='/mnt/amlfs-07/shared/rpunamiya/leact/checkpoints'


def discover(coverage=Path(COVERAGE),pear=Path(PEAR)):
    found={}
    def add(label,target,kind,step=None):
        target=Path(target)
        if target.is_dir():found[PREFIX+label]={'target':str(target.resolve()),'kind':kind,'step':step}
    stages=[('qwen36_moe_warmstart',coverage/'warmstart'),
            ('qwen36_moe_leact_from496_r1',coverage/'leact-from-step496-r1'),
            ('qwen36_moe_leact_from737_r1',coverage/'leact-r1')]
    for label,root in stages:
        for directory in [root/'train',root/'restart-states']:
            for checkpoint in directory.glob('checkpoint-*'):
                path=checkpoint/'checkpoint-manifest.json'
                if not path.is_file():continue
                manifest=json.loads(path.read_text());step=int(manifest['step'])
                add(f'{label}_step{step:06d}_adapter',checkpoint,'adapter_and_training_state',step)
        for name in ['warmstart-ready.json','leact-ready.json']:
            path=root/name
            if not path.is_file():continue
            ready=json.loads(path.read_text());step=int(ready['step'])
            add(f'{label}_step{step:06d}',ready['checkpoint'],'merged_model',step)
    for folder in (coverage/'intermediate').glob('warmstart-step*'):
        if not (folder/'checkpoint-pin.json').is_file():continue
        step=int(folder.name.removeprefix('warmstart-step'))
        add(f'qwen36_moe_warmstart_step{step:06d}',folder/'model','merged_model',step)
        if (folder/'adapter').is_dir():
            add(f'qwen36_moe_warmstart_step{step:06d}_adapter',folder/'adapter','adapter',step)
    for arm,label in [('moe-warmstart496-pearplusplus','qwen36_moe_warmstart496_pearplusplus'),
                      ('qwen38-27b-base-pearplusplus','qwen38_27b_base_pearplusplus')]:
        root=pear/'runs'/arm/'ckpt'
        latest=root/'latest_checkpointed_iteration.txt'
        if not latest.is_file():continue
        committed=int(latest.read_text().strip())
        for folder in root.glob('global_step_*'):
            step=int(folder.name.removeprefix('global_step_'))
            if step<=committed:add(f'{label}_step{step:06d}',folder,'full_model_and_training_state',step)
    return found


def sync(registry=Path(REGISTRY),coverage=Path(COVERAGE),pear=Path(PEAR)):
    registry.mkdir(parents=True,exist_ok=True)
    aliases=discover(coverage,pear)
    index=registry/'rp_jhh_checkpoint_index.json'
    previous=json.loads(index.read_text()).get('checkpoints',{}) if index.exists() else {}
    for name,item in aliases.items():
        path=registry/name;target=Path(item['target'])
        if path.is_symlink():
            if path.readlink()!=target:raise ValueError('Checkpoint alias collision: '+name)
        elif path.exists():raise ValueError('Refusing to replace an existing checkpoint path: '+str(path))
        else:path.symlink_to(target,target_is_directory=True)
        item['path']=str(path)
    for name,item in previous.items():
        path=registry/name
        if name not in aliases and name.startswith(PREFIX) and path.is_symlink():
            if path.readlink()==Path(item['target']) and not path.exists():path.unlink()
    result={'prefix':PREFIX,'updated_utc':datetime.now(timezone.utc).isoformat(),
            'checkpoint_count':len(aliases),'checkpoints':aliases,
            'policy':'Stable names for committed checkpoints. Canonical paths, immutable manifests and resume state stay unchanged.'}
    temporary=index.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(index)
    return result


if __name__=='__main__':
    print(json.dumps(sync()))
