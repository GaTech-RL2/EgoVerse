"""Freeze five independently gated runs; never submit training from this script."""
import hashlib,json,random,shutil
from pathlib import Path

R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
C=Path('/coc/flash7/cxin34/warp-data/collections/cxin34_aria_20261125_0556_114')
for d in ['reports','scripts','logs','drawer/episodes','cart/reports']:
    (R/d).mkdir(parents=True,exist_ok=True)
def freeze(p,obj):
    if p.exists(): assert json.loads(p.read_text())==obj, str(p)
    else:
        with p.open('x')as f:json.dump(obj,f,indent=2)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb')as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
raw=json.load(open(C/'download-manifest.json'));assert len(raw)==114
assert len({x['uuid']for x in raw})==114
uids=sorted(x['uuid']for x in raw);random.Random(42).shuffle(uids);val=set(uids[:11])
records=[]
for x in raw:
    p=C/'raw'/x['file'];assert p.stat().st_size==x['bytes']
    rec=dict(x,vrs=str(p),task='drawer',split='valid'if x['uuid']in val else'train',head_height_m=1.57,tracking_source='ondevice',mps_requests_authorized=False,look_at=False)
    e=R/'drawer/episodes'/x['uuid'];(e/'gaze').mkdir(parents=True,exist_ok=True)
    freeze(e/'input.json',rec);records.append(rec)
freeze(R/'drawer/input-manifest.json',dict(records=records,split_seed=42,split_unit='original_recording_uuid',validation_scope='same_collection_day',expected_train_before_qc=103,expected_valid_before_qc=11,raw_manifest_sha256=sha(C/'download-manifest.json')))
old=Path('/coc/flash7/scratch/cxin34/egoa-cart107-bs-strong-ft-20261124/releases/cart/target.json')
cart=json.load(open(old));assert len(cart['episodes'])==107
freeze(R/'cart/input-manifest.json',dict(source_manifest=str(old),source_manifest_sha256=sha(old),records=[{k:e[k]for k in ['uuid','take','split','rows','episode_index','demo']}for e in cart['episodes']],tracking_source='existing_fixed_nonLA',observation_root='/coc/flash7/czhang883/warp-data/collections/chuye_cart_1102/processed/fix1124',retarget_root='/coc/flash7/czhang883/warp-data/collections/chuye_cart_1102/processed/fix1124_bs',look_at=False))
runs=[]
for task in ['cart','drawer']:
    for stage,steps in [('finetune',60000),('scratch',180000)]:
        runs.append(dict(id=f'{task}-nola-fix-bs-{stage}-MSCstrong-s42',task=task,stage=stage,steps=steps,priority='first'if stage=='finetune'else'after_finetune',dino_head='reset'if stage=='finetune'else'random_init'))
runs.append(dict(id='cart-nola-fix-bs-finetune-dinoLoRA-MSCstrong-s42',task='cart',stage='finetune',steps=60000,priority='first',dino_head='inherit_pretrained_head_base_frozen_LoRA_only',other_policy_parameters='full_finetune'))
freeze(R/'reports/plan.json',dict(runs=runs,seed=42,msc='strong',retarget_fix=True,base_smoothing=True,look_at=False,drawer_head_height_m=1.57,drawer_mps=False,checkpoint_interval=10000,finetune_parent_sha256='94b695a5786fa97a554076ed1c86272f575a3cf3979d873fcb5c3f812c87a430',release_policy='independent_per_dataset',launch_policy='each_run_after_own_data_QC_and_smoke_pass',global_all_five_barrier=False,duplicates_policy='exclusive_lock_intent_receipt_reconcile_slurm_before_retry',other_jobs='unchanged',formal_authorized=True))
print(json.dumps(dict(root=str(R),runs=len(runs),drawer_records=len(records),cart_records=107,independent_launch=True)))
