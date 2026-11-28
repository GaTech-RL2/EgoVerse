import hashlib,json,shutil,subprocess
from pathlib import Path
OLD=Path('/coc/flash7/cxin34/cart-B180-s20261110-FT10k-robotpc-20261118')
W=Path('/coc/flash7/cxin34/cart-msc-ft-20261118')
R=Path('/coc/flash7/cxin34/cart-B180-MSCaug-FT10k-robotpc-20261118')
run=W/'runs/cart-B180-msc-aug-ft10k-s42'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert not R.exists(),'Reconcile existing bundle; do not overwrite'
proof=json.loads((run/'proof.json').read_text());assert proof['passed'] and proof['step']==10000 and proof['full_validation']
assert 'COMPLETED' in subprocess.check_output(['/opt/slurm/Ubuntu-24.04/current/bin/sacct','-X','-n','-j','4794212','-o','State,ExitCode'],text=True)
R.mkdir()
for d in ['assets','source','gaze']:shutil.copytree(OLD/d,R/d,symlinks=False,ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(W/'campaign',R/'campaign',ignore=shutil.ignore_patterns('__pycache__'))
for d in ['configs','reports','scripts','testdata']:(R/d).mkdir()
shutil.copy2(W/'configs/cart-B180-msc-aug-ft10k-s42.json',R/'configs/training.json')
shutil.copy2(W/'configs/hd2-reference.yaml',R/'configs/hd2-reference.yaml')
shutil.copy2(W/'releases/cart/normalizer.json',R/'normalizer.json')
assert sha(R/'configs/training.json')==proof['config_sha256'] and sha(R/'normalizer.json')==proof['normalizer_sha256']
source=run/'step_010000.pt';source_sha=sha(source)
oldplan=json.loads((OLD/'reports/plan.json').read_text())
for p in [OLD/'serve_robotpc.py',OLD/'check_server.py',*(OLD/'scripts').glob('*')]:
 if not p.is_file():continue
 text=p.read_text().replace(str(OLD),str(R)).replace(oldplan['source_checkpoint_sha256'],source_sha).replace('8170','8172').replace('cart_B180_s20261110_FT10k','cart_B180_MSCaug_FT10k')
 (R/p.relative_to(OLD)).write_text(text)
plan=dict(source_checkpoint=str(source),source_checkpoint_sha256=source_sha,target_manifest=str(W/'releases/cart/target.json'),step=10000,
 pretraining_arm='B',pretraining_steps=180000,target_train_recordings=30,target_valid_recordings=10,seed=42,
 target_architecture='B action+futureDINO',training_recipe='MSC augmentation only; no consistency',
 normalizer='source_train frozen matchedcontrol',virtual_camera='ON',port=8172,status='EXPORT_PARITY_PENDING',hardware_not_tested=True)
(R/'reports/plan.json').write_text(json.dumps(plan,indent=2))
for p in (R/'scripts').glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
args=['/opt/slurm/Ubuntu-24.04/current/bin/sbatch','--parsable','--partition=overcap','--account=overcap','--qos=debug','--nice=30000',
 '--cpus-per-task=4','--mem=32G','--time=00:30:00','--job-name=cart-msc-robotpc-export-check',
 '--output='+str(R/'reports/export-check-%j.log'),str(R/'scripts/test.sbatch')]
(R/'reports/test-submit.intent').write_text(json.dumps(args))
jid=subprocess.check_output(args,text=True).strip();assert jid.isdigit()
(R/'reports/test-submit.json').write_text(json.dumps(dict(job_id=jid,command=args),indent=2))
print(json.dumps(dict(root=str(R),job_id=jid,source_sha256=source_sha),indent=2))
