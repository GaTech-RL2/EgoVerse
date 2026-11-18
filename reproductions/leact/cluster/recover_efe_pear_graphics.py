"""Add graphics capacity after a real cross-pool relay check; retain live lanes."""
import hashlib
from pathlib import Path

import yaml

from .launch_efe_pear_scaled import DEST
from .scale_efe import BASE,read,write
from .efe_pear_submission import submit_reconciled

LFS_ROOT='/mnt/lfs/shared/rpunamiya/rp_jhh_pear_scaled_20261116_v1'


def relay_ready(ready,model):
    digest=hashlib.sha256((BASE/'cluster/efe_pear_file_relay.py').read_bytes()).hexdigest()
    return bool(ready.get('passed')and ready.get('model_file_relay_passed')
        and ready.get('transport_roundtrip_passed')and ready.get('relay_sha256')==digest
        and ready.get('transport_sha256')==model['adapter_digests']['efe_pear_object_transport.py']
        and ready.get('assets',{}).get('passed')and ready.get('sources',{}).get('passed')
        and ready.get('executed_control_steps')==1 and ready.get('cameras')
        and isinstance(ready.get('native_termination_terms'),dict)
        and set(ready.get('rtx_migration_smokes',{}))=={'robodojo','robolab','yamlab'}
        and all(v.get('passed')for v in ready['rtx_migration_smokes'].values()))


def frozen_files(build):
    files={}
    for name,digest in read(build/'source-manifest.json').items():
        path=build/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen native worker input changed')
        if name not in {'bootstrap.py','spec.json','workflow.yaml'}:files[name]=path
    return files


def prepare(state,ready,queue):
    from .launch_efe_pear_primary import worker_build
    model=read(Path(state['build'])/'spec.json')
    if not relay_ready(ready,model):raise ValueError('A matching real native file-relay check is required')
    wire=hashlib.sha256((state['workflow']+ready['relay_sha256']+ready['assets_root']).encode()).hexdigest()[:16]
    root=LFS_ROOT+'/runs/'+wire;uri=model['checkpoint_archive']+'/file-relay/'+wire
    originals={key:v for key,v in queue.items()if v.get('owner_model_workflow')==state['workflow']
        and str(v.get('pool','')).startswith('groot-l40')and not v.get('graphics_recovery_wire')}
    if len(originals)!=16:raise ValueError('The complete original 16 graphics lanes must be retained')
    builds=[];gpu=0;kinds=set();bridge_files=None;base_image=None
    for key,item in originals.items():
        old=Path(item['build']);spec=read(old/'spec.json');files=frozen_files(old)
        workflow=yaml.safe_load((old/'workflow.yaml').read_text())['workflow']
        task=workflow['tasks'][0];resources=dict(workflow['resources']['default']);resources['platform']='ovx-l40s'
        gpu+=resources['gpu'];bridge_files=files
        if(spec['transport_root']!=model['run']+'/transport'or spec['finished']!=model['run']+'/finished.json'
            or not spec['output'].startswith(model['run']+'/')):
            raise ValueError('Original graphics paths do not belong to the current primary run')
        spec.update(output=spec['output'].replace(model['run'],root),transport_root=root+'/transport',
            finished=root+'/finished.json',owner_model_workflow=state['workflow'])
        if 'head_heartbeat'in spec:spec['head_heartbeat']=root+'/head-heartbeat.json'
        for path_key,sha_key in [('payload','payload_sha256'),('source_archive_path','source_archive_sha256'),
            ('generated_reset_archive','generated_reset_archive_sha256')]:
            if path_key not in spec:continue
            name=Path(spec[path_key]).name;entry=ready['sources']['files'][name]
            if spec.get(sha_key)and spec[sha_key]!=entry['sha256']:
                raise ValueError('Pool-local source payload differs from the admitted worker')
            spec[path_key]=LFS_ROOT+'/sources/'+name
        if spec.get('kind')=='rtx':
            kinds.add('efe/'+spec['suite']);entry='efe_pear_primary_rtx.py'
            # The probe image is also a qualified CPU-only object-store reader.
        elif 'generated_reset_tasks'in spec:
            kinds.update('native_yam/'+name for name in spec['generated_reset_tasks'])
            if spec['assets_release']['release_manifest_sha256']!=ready['assets']['release_manifest_sha256']:
                raise ValueError('Native YAM input release changed during pool recovery')
            spec['assets_release']['assets_root']=ready['assets_root']
            spec['asset_aliases']=['/mnt/amlfs-07/shared/datasets/gear_sim/assets'];entry='efe_pear_yam_server.py'
        else:raise ValueError('Unexpected graphics worker in the recovery ledger')
        build=worker_build(spec,files,entry,task['image'],task['command'][0],**resources)
        builds.append({'build':str(build),'pool':'groot-l40s-01','gpu':resources['gpu']})
    if gpu!=24 or len(kinds)!=19:raise ValueError('Graphics recovery would change the full admitted mixture')
    probe=read(DEST/'l40s01-cross-pool-probe.json')
    old_probe=yaml.safe_load((Path(probe['jobs']['probe']['build'])/'workflow.yaml').read_text())['workflow']
    base_image=old_probe['tasks'][0]['image']
    bridge_files={**bridge_files,'efe_pear_file_relay.py':BASE/'cluster/efe_pear_file_relay.py'}
    for mode,filesystem,pool,platform in [('model',model['run'],'groot-l40-02','ovx-l40'),
        ('native',root,'groot-l40s-01','ovx-l40s')]:
        spec={'kind':'file_relay','mode':mode,'root':filesystem+'/transport','uri':uri,
            'owner_model_workflow':state['workflow'],'kinds':sorted(kinds)}
        build=worker_build(spec,bridge_files,'efe_pear_file_relay.py',base_image,'python3',
            gpu=0,cpu=8,memory='24Gi',storage='10Gi',platform=platform)
        builds.insert(0,{'build':str(build),'pool':pool,'gpu':0})
    return {'wire':wire,'workflow':state['workflow'],'graphics_gpus':gpu,'model_allocation_preserved':True,
        'relay_sha256':ready['relay_sha256'],'uri':uri,'root':root,'builds':builds,'originals':list(originals)}


def tick(state):
    control=read(DEST/'control.json',{})
    if control.get('graphics_recovery_target_pool')!='groot-l40s-01':return None
    if control.get('paused')or state.get('scheduler_status')!='RUNNING':return None
    if state.get('run_id')not in control.get('enabled_training_arms',[]):return None
    path=DEST/'graphics-recovery-state.json';progress=read(path,{})
    if progress.get('workflow')==state['workflow']and progress.get('phase')=='graphics_replacement_submitted':return progress
    ready=read(DEST/'l40s01-storage-ready.json',{});model=read(Path(state['build'])/'spec.json')
    if not relay_ready(ready,model):
        result={'workflow':state['workflow'],'phase':'awaiting_native_file_relay_validation',
            'model_allocation_preserved':True}
        write(path,result);return result
    queue=read(DEST/'primary-native-queue.json',{})
    plan=prepare(state,ready,queue);write(DEST/'graphics-recovery-plan.json',plan)
    for row in plan['builds']:
        build=Path(row['build'])
        if build.name in queue:continue
        receipt=submit_reconciled(build,priority='HIGH',pool=row['pool'],nodes=1)
        queue[build.name]={**receipt,'workflow':receipt['name'],'build':str(build),'pool':row['pool'],
            'priority':'HIGH','gpu':row['gpu'],'owner_model_workflow':state['workflow'],
            'graphics_recovery_wire':plan['wire']}
        write(DEST/'primary-native-queue.json',queue)
    for key in plan['originals']:
        item=queue[key];workflow=item['workflow']
        if not workflow.startswith('leact-rp-jhh-pear-worker-'):
            raise ValueError('Refusing to replace a workflow outside the owned graphics ledger')
        # Existing lanes may already be executing a student's episode. Keep
        # them serving and eligible for the authorized compatible-pool retry.
        item.update(allow_amlfs07_pool_fallback=True,parallel_graphics_wire=plan['wire'])
        write(DEST/'primary-native-queue.json',queue)
    result={**plan,'phase':'graphics_replacement_submitted','priority':'HIGH','existing_graphics_retained':True};write(path,result)
    return result
