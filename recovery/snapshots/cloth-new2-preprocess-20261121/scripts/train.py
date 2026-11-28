"""Step-budget runner with atomic full-state checkpoints and fail-closed release gates."""
import argparse,contextlib,fcntl,hashlib,json,os,random,signal,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from .policy import CampaignPolicy
from .gaze_policy import GazePolicy
from .data import EpisodeDataset,sha
from .contracts import release_path,check_normalizer

def atomic_save(obj,path):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    tmp.replace(path)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])

def code_hash(root):
    h=hashlib.sha256()
    for base in ("campaign","source/egomimic","assets"):
        for p in sorted((root/base).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(root)).encode());h.update(sha(p).encode())
    h.update(sha(root/"configs/hd2-reference.yaml").encode())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text());root=Path(os.environ['CAMPAIGN_ROOT'])
    release=release_path(root,cfg);ready=release/'READY.json'
    assert ready.exists(),'Data release not ready; no training'
    gate=json.loads(ready.read_text());assert args.smoke or not gate.get('synthetic_only',False)
    assert gate['passed'] and gate['normalizer_frozen'] and gate['retarget_verified']
    manifest=release/(cfg['split_role']+'.json');normpath=release/'normalizer.json'
    assert sha(manifest)==gate['manifests'][cfg['split_role']] and sha(normpath)==gate['normalizer_sha256']
    assert gate['recording_hz']==60 and gate['horizon_rows']==64 and gate['dino_offset_rows']==10
    norm=json.loads(normpath.read_text());check_normalizer(cfg,norm,json.loads(manifest.read_text()),sha(manifest))
    if cfg['stage']=='pretrain' and cfg['arm']=='C':assert gate['clean_view_audited']
    if cfg['stage']=='pretrain' and cfg['arm']=='D':assert gate['fk_reference_verified']
    if cfg['arm']=='E':
        assert cfg['stage']=='scratch','Only Cart E scratch is implemented/registered in this pilot'
        assert gate['clean_view_audited'] and gate['e_global_audited'] and gate['gaze_projection_verified']
        assert cfg['gaze_weight']==.1
        if not args.smoke:assert gate['gaze_loss_scale_verified'],'Review real-data gaze/action loss and gradient scales before formal E'
    run=root/'runs'/(cfg['id']+('-smoke' if args.smoke else ''));run.mkdir(parents=True,exist_ok=True)
    lock=(run/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    arm=cfg['arm'] if cfg['stage']=='pretrain' or cfg['arm']=='E' else 'B'
    if not args.smoke:
        smoke=json.loads((root/'runs'/(cfg['id']+'-smoke')/'proof.json').read_text())
        assert smoke['passed'] and smoke['config_sha256']==sha(args.config) and smoke['normalizer_sha256']==sha(normpath)
        assert smoke['manifest_sha256']==sha(manifest) and smoke['code_sha256']==code_hash(root)
        assert smoke['step']==200 and smoke['resumed_from']==100 and smoke['changed_weights']
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    if cfg.get('skynet_gpu_allocation')=='generic_gres_matching_nodes':
        assert cfg['skynet_gpu_type'].upper() in torch.cuda.get_device_name().upper()
        assert torch.cuda.is_bf16_supported(),'Native BF16 GPU required for matched target scratch'
    print(json.dumps(dict(run=cfg['id'],smoke=args.smoke,device=torch.cuda.get_device_name(),normalizer_fit_partition=norm['fit_partition'],release=str(release))),flush=True)
    robot=json.loads((root/'assets/robot.json').read_text());robot['urdf']=str(root/'assets/robot.urdf')
    if arm=='E':model=GazePolicy(root/'configs/hd2-reference.yaml',norm,robot,cfg['gaze_weight']).cuda()
    else:model=CampaignPolicy(root/'configs/hd2-reference.yaml',arm,norm,robot).cuda()
    model.photometric_config=cfg['photometric']
    for encoder in model.policy.encoders.values():encoder.colour_brightness_jitter=0.;encoder.colour_channel_jitter=0.
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['scheduler_steps'],eta_min=1e-5)
    train=EpisodeDataset(manifest,'train',arm,norm);valid=EpisodeDataset(manifest,'valid',arm,norm)
    if cfg['stage']!='pretrain':assert len({x['uuid'] for x in train.meta['episodes'] if x['split']=='train'})==cfg['train_recordings']
    sampler=torch.Generator().manual_seed(cfg['seed']);order=None;cursor=0;step=0;last=run/'last.pt'
    identity=dict(config_sha256=sha(args.config),normalizer_sha256=sha(normpath),manifest_sha256=sha(manifest),code_sha256=code_hash(root))
    resumed_from=None
    initial={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False);assert ck['identity']==identity
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        restore_rng(ck['rng']);sampler.set_state(ck['sampler']);step=ck['step'];order=ck['order'];cursor=ck['cursor'];resumed_from=step
        initial=ck['initial'] if args.smoke else initial
    elif cfg['stage']=='finetune':
        source=root/'runs'/cfg['parent']/f"step_{cfg['parent_step']:06d}.pt"
        ck=torch.load(source,map_location='cpu',weights_only=False)
        assert ck['step']==cfg['parent_step'] and ck['identity']['normalizer_sha256']==identity['normalizer_sha256']
        assert ck.get('validation') is not None and np.isfinite(ck['validation'])
        assert ck['identity']==cfg['source_identity']
        assert sha(source)==cfg['parent_checkpoint_sha256'] if cfg.get('parent_checkpoint_sha256') else ck['step']==180000
        if cfg['parent_step']==180000:
            sourceproof=json.loads((source.parent/'proof.json').read_text())
            assert sourceproof['passed'] and sourceproof['step']==180000 and sourceproof['full_validation']
        def tensor_digest(sd):
            h=hashlib.sha256()
            for k,v in sorted(sd.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        fresh_dino={k:v.clone() for k,v in model.state_dict().items() if k.startswith('policy.heads.rby1_dino_wm_future.')}
        fresh_dino_sha=tensor_digest(fresh_dino)
        transferred=model.transfer_policy(ck['model'])
        own=model.state_dict()
        assert all(torch.equal(own[k].cpu(),ck['model'][k].cpu()) for k in transferred)
        assert all(torch.equal(own[k],v) for k,v in fresh_dino.items())
        assert not optimizer.state and scheduler.last_epoch==0 and step==0
        assert all(p.requires_grad for p in model.parameters())
        (run/'transfer-audit.json').write_text(json.dumps(dict(passed=True,parent=str(source),parent_step=ck['step'],parent_sha256=sha(source),transferred_keys=transferred,fresh_dino_sha256=fresh_dino_sha,optimizer_fresh=True,scheduler_step=0,target_step=0,all_policy_parameters_trainable=True),indent=2))
        del ck

        initial={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
    stop=[False];signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    started=time.monotonic();target=200 if args.smoke else cfg['steps'];last_val=None
    def checkpoint(path):
        atomic_save(dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),sampler=sampler.get_state(),order=order,cursor=cursor,identity=identity,validation=last_val,initial=initial if args.smoke else None),path)
    def validation(full=False):
        saved=rng_state();model.eval();values=[];parts={};rows=0
        with torch.no_grad():
            for start in range(0,len(valid) if full else min(len(valid),32*10),32):
                random.seed(20261115+start);np.random.seed(20261115+start);torch.manual_seed(20261115+start);torch.cuda.manual_seed_all(20261115+start)
                b=default_collate([valid[i] for i in range(start,min(start+32,len(valid)))])
                b={k:v.cuda() for k,v in b.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):v,metrics=model(b)
                count=len(b['action']);rows+=count
                assert torch.isfinite(v);values.append(v.item()*count)
                for k,x in metrics.items():parts[k]=parts.get(k,0.)+x.item()*count
        with (run/'full_validation_components.jsonl').open('a') as f:
            f.write(json.dumps(dict(step=step,full=full,rows=rows,**{k:v/rows for k,v in parts.items()}))+'\n')
        if arm=='E':
            with (run/'gaze_validation.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step,action_condition='predicted',gaze_ce_condition='teacher_forced',full=full,**{k:v/rows for k,v in parts.items()}))+'\n')
        restore_rng(saved);model.train();return sum(values)/rows
    while step<target:
        model.train()
        if order is None or step%100==0 or cursor+32>len(order):
            order=torch.randperm(len(train),generator=sampler);cursor=0
        assert len(order)>=32
        ids=order[cursor:cursor+32].tolist();cursor+=32
        batch={k:v.cuda() for k,v in default_collate([train[i] for i in ids]).items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,metrics=model(batch)
        assert torch.isfinite(loss);loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step();scheduler.step();step+=1
        if step%100==0 or args.smoke:
            with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=loss.item(),**{k:v.item() for k,v in metrics.items()}))+'\n')
        if step%1000==0 or (args.smoke and step in (100,200)):checkpoint(last)
        if step%5000==0 or step==target:
            val=validation(full=True);last_val=val
            with (run/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=val))+'\n')
        if step%10000==0:checkpoint(run/f'step_{step:06d}.pt')
        if (stop[0] or time.monotonic()-started>13500) and step<target:
            checkpoint(last);return 75
        if args.smoke and step==100:
            checkpoint(last);return 75 # resume must be a separate process
    val=last_val if last_val is not None else validation(full=True);last_val=val
    changed=any(not torch.equal(initial[k],v.detach().cpu()) for k,v in model.named_parameters())
    if args.smoke:assert changed and resumed_from==100,'Smoke requires separate-process resume 100→200 and changed weights'
    checkpoint(last);checkpoint(run/f'step_{step:06d}.pt')
    (run/'proof.json').write_text(json.dumps(dict(passed=True,step=step,final_val=val,resumed_from=resumed_from,changed_weights=changed,full_validation=True,**identity),indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
