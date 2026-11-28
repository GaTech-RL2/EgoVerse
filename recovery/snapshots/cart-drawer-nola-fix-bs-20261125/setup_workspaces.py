import json,shutil
from pathlib import Path
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125')
FT=Path('/coc/flash7/scratch/cxin34/egoa-cart107-bs-strong-ft-20261124')
SC=Path('/coc/flash7/scratch/cxin34/bs-fixed-target-scratch180k-20261125/cart')
for task in ['cart','drawer']:
 for stage in ['finetune','scratch']+(['lora']if task=='cart'else[]):
  dst=R/'training'/task/stage;src=SC if stage=='scratch'else FT
  marker=dst/'reports/template-copied.json'
  if marker.exists():continue
  assert not dst.exists(),str(dst)
  for d in ['reports','runs','logs','configs','scripts','cache','tensors','releases/'+task]:
   (dst/d).mkdir(parents=True,exist_ok=True)
  for d in ['campaign','source','assets']:
   shutil.copytree(src/d,dst/d,symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
  for name in ['env.sh','train.sbatch','chain.sbatch','smoke_gate.py','photometric_audit.py']:
   shutil.copy2(src/'scripts'/name,dst/'scripts'/name)
  if stage=='scratch':shutil.copy2(src/'scripts/scratch_norm.py',dst/'scripts/scratch_norm.py')
  shutil.copy2(src/'configs/hd2-reference.yaml',dst/'configs/hd2-reference.yaml')
  key='cart107-la-fix1124-bs-scratch180k-MSCstrong-s42'if stage=='scratch'else'egoa180-cart107-la-bs-ft60k-MSCstrong-s42'
  c=json.load(open(src/'configs'/(key+'.json')))
  rid=f'{task}-nola-fix-bs-'+('finetune-dinoLoRA'if stage=='lora'else stage)+'-MSCstrong-s42'
  c.update(id=rid,task=task,release_id=task,train_recordings=97 if task=='cart'else 103,valid_recordings=10 if task=='cart'else 11,look_at=False,retarget_fix=True,base_smoothing=True,data_repair='calibrated CPF + measured wrists; Base Smoothing S; non-look-at',deployment_preset='TGZ10HD2'if task=='cart'else'TBD_DRAWER_PRESET')
  if task=='drawer':c.update(head_height_m=1.57,tracking_source='ondevice',mps_requests=False)
  if stage=='lora':c['dino_head_lora']=dict(rank=8,alpha=16,dropout=0.,inherit=True,base_frozen=True,other_policy_parameters='full_finetune')
  (dst/'configs'/(rid+'.json')).write_text(json.dumps(c,indent=2))
  if stage!='scratch':
   (dst/'runs'/c['parent']).symlink_to((FT/'runs'/c['parent']).resolve(),target_is_directory=True)
  # Allow the new task name only; preserve all other contract checks.
  p=dst/'campaign/contracts.py';s=p.read_text();s=s.replace("['cart','basket','cloth']","['cart','basket','cloth','drawer']").replace("['cart','cloth','basket']","['cart','cloth','basket','drawer']");p.write_text(s)
  marker.write_text(json.dumps(dict(template=str(src),run=rid,stage=stage,ready=False),indent=2))
  print('PREPARED',rid,flush=True)
