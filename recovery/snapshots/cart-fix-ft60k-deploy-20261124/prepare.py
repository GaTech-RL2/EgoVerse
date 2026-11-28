from pathlib import Path
import json,hashlib,shutil,subprocess,ast
B=Path('/coc/flash7/cxin34');OLD=B/'egoa180-cart-fix-FT10k-robotpc-20261124';R=B/'egoa180-cart-fix-FT60k-robotpc-20261124';C=B/'cart-fix-ft60k-deploy-control-20261124'
C.mkdir(exist_ok=True);(C/'logs').mkdir(exist_ok=True)
with(C/'prepare.intent').open('x')as f:f.write('User confirmed original non-look-at Cart fix exact60k')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
oldplan=json.load(open(OLD/'reports/plan.json'));ck=Path(oldplan['source_checkpoint']).with_name('step_060000.pt');assert ck.is_file();expected=sha(ck)
contract=json.load(open(OLD/'reports/deployment-contract.json'))
for n,h in contract['required_sha256'].items():assert sha(OLD/n)==h,n
R.mkdir(exist_ok=False)
for d in ['campaign','source','configs','assets','gaze','client-support','scripts','testdata']:
 shutil.copytree(OLD/d,R/d,ignore=shutil.ignore_patterns('__pycache__','*.pyc','reference.npz'))
(R/'reports').mkdir()
for n in ['serve_robotpc.py','check_server.py','normalizer.json']:shutil.copy2(OLD/n,R/n)
for n in ['geometry-audit.json','native-assets-audit.json']:shutil.copy2(OLD/'reports'/n,R/'reports'/n)
plan=dict(oldplan,source_checkpoint=str(ck),source_checkpoint_sha256=expected,step=60000,port=8282,package=str(R),preset='TGZ10HD2',look_at=False,camera_file='virtual_cam_cart_1102.npz',geometry_reused_from=str(OLD))
(R/'reports/plan.json').write_text(json.dumps(plan,indent=2))
reuse={}
for d in ['assets','gaze']:
 for p in (R/d).rglob('*'):
  if p.is_file():n=str(p.relative_to(R));assert sha(p)==sha(OLD/n);reuse[n]=sha(p)
(R/'reports/asset-reuse-audit.json').write_text(json.dumps(dict(passed=True,source_package=str(OLD),reason='Same non-look-at fixed target dataset/split/geometry at10k and60k',sha256=reuse,geometry_audit_sha256=sha(R/'reports/geometry-audit.json'),native_assets_audit_sha256=sha(R/'reports/native-assets-audit.json')),indent=2))
for p in [R/'serve_robotpc.py',R/'check_server.py',*(R/'scripts').iterdir()]:
 if p.is_file()and p.suffix in ['.py','.sh','.sbatch']:
  s=p.read_text().replace(str(OLD),str(R)).replace(OLD.name,R.name).replace(oldplan['source_checkpoint_sha256'],expected).replace('8270','8282')
  if p.name in ['serve_robotpc.py','export-check.py']:s=s.replace("ck['step']==10000","ck['step']==60000")
  if p.name=='test.sbatch':s='\n'.join(l for l in s.splitlines()if 'scripts/build_geometry.py'not in l)+'\n'
  p.write_text(s)
for p in R.rglob('*.py'):ast.parse(p.read_text())
for p in (R/'scripts').glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
(C/'packages.json').write_text(json.dumps([plan],indent=2))
with(C/'cart-test.intent').open('x')as f:f.write(json.dumps(plan))
cmd=['/opt/slurm/Ubuntu-24.04/current/bin/sbatch','--parsable','--account=rl2-lab','--partition=rl2-lab','--qos=debug','--cpus-per-task=4','--mem=48G','--time=01:00:00','--exclude=hal','--job-name=deploy-cart-fix-60k',f'--output={C}/logs/%j-cart.out',str(R/'scripts/test.sbatch')]
p=subprocess.run(cmd,capture_output=True,text=True);d=dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
(C/'cart-test-submit.json').write_text(json.dumps(d,indent=2));assert p.returncode==0,d;print(p.stdout.strip(),flush=True)
