from pathlib import Path
import json,hashlib,subprocess
C=Path('/coc/flash7/cxin34/egoa-shared-fix1124-ft-20261124/deploy-control')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
results=[]
for plan in json.load(open(C/'packages.json')):
 task=plan['task'];r=Path(plan['package']);receipt=json.load(open(C/f'{task}-recovery-submit.json'));job=receipt['stdout'].strip().split(';')[0]
 acct=subprocess.run(['/opt/slurm/Ubuntu-24.04/current/bin/sacct','-X','-j',job,'--format=JobID,State,ExitCode','-P','-n'],capture_output=True,text=True,check=True,timeout=120).stdout
 assert job+'|COMPLETED|0:0'in acct,acct
 test=json.load(open(r/'reports/skynet-server-test.json'));geom=json.load(open(r/'reports/geometry-audit.json'));native=json.load(open(r/'reports/native-assets-audit.json'))
 assert test['passed']and geom['passed']and native['passed']
 assert test['original_checkpoint_sha256']==plan['source_checkpoint_sha256']==sha(Path(plan['source_checkpoint']))
 assert test['portable_checkpoint_sha256']==sha(r/'checkpoint.pt')
 assert sha(r/'normalizer.json')==plan['normalizer_sha256']and sha(r/'configs/training.json')==plan['training_config_sha256']
 for n,h in plan['gaze_sha256'].items():assert sha(r/'gaze'/n)==h
 required={str(p.relative_to(r)):sha(p)for p in sorted(r.rglob('*'))if p.is_file()and not str(p.relative_to(r)).startswith('reports/')and p.name not in ['SHA256SUMS','README.md']}
 contract=dict(task=task,arm='B',step=10000,seed=42,port=plan['port'],preset=plan['preset'],run=plan['run'],look_at=plan['look_at'],source_checkpoint_sha256=plan['source_checkpoint_sha256'],required_sha256=required,training_split=dict(train=plan['train_recordings'],valid=plan['valid_recordings']),gaze_retrained=False,gaze_source=plan['gaze_source'],geometry_refit=True,hardware_tested=False)
 (r/'reports/deployment-contract.json').write_text(json.dumps(contract,indent=2))
 p=subprocess.run(['/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python',str(r/'scripts/verify_assets.py')],capture_output=True,text=True);assert p.returncode==0,(p.stdout,p.stderr)
 (r/'reports/offline-contract-check.json').write_text(p.stdout)
 error=max(max(c['plain_max'],c['rtc_max'])for c in test['cases']);assert error==0
 text=f'''# {task} shared fix1124 FT10k deployment
Run: {plan['run']}. Exact10k; original EgoVerse180k initialization, strong MSC, seed42.
Split: {plan['train_recordings']}train/{plan['valid_recordings']}validation. Original code/config and source-frozen normalization included.
Preset {plan['preset']}; port{plan['port']}; 64x49 actions,16 flow steps,RTC8/10/13.
Cart uses shared fix1124 look-at actions/head frame. No runtime head override. Native HD2 preset plus the package-specific look-at VV_MODEL implements virtual camera ON.
Basket uses shared merged HD2Z5 clouds. Six task gaze models reused unchanged; not retrained for this release (Basket retains original23train gaze provenance).
Camera {plan['camera_file']} and native circular reset fitted only on retained target training recordings; validation excluded. Shared HDF5 q matches training tensors; source global cloud rows match published release tensors. Actual tg transforms and HDF5 hashes retained.
Heldout camera median {geom['new_camera']['translation_cm_median']:.3f}cm/{geom['new_camera']['rotation_deg_median']:.3f}deg.
CPU{job} COMPLETED0: exact model tensor preservation,3realheldout plain/RTC WebSocket maxerror{error},unseeded finite. Native gaze/reset/camera/preview passed. No robotpc CUDA/client/hardware claim.
Source checkpoint {plan['source_checkpoint']}
Source SHA256 {plan['source_checkpoint_sha256']}
Download all files into a new directory; do not mix assets with earlier fixed/pre-fix packages.

On robotpc:
sha256sum -c SHA256SUMS
bash scripts/client.sh inspect-client
Terminal1: bash scripts/serve.sh
Terminal2: bash scripts/check.sh && bash scripts/client.sh dry-run
Operator-only physical launch after checks: bash scripts/client.sh run
Only one native client should control the robot. Existing native HD2/HD2Z5 client, CUDA environment and drivers required. No hardware command was executed during packaging.
'''
 (r/'README.md').write_text(text)
 acceptance=dict(passed=True,package=str(r),task=task,step=10000,run=plan['run'],port=plan['port'],job_id=job,source_checkpoint_sha256=plan['source_checkpoint_sha256'],portable_checkpoint_sha256=test['portable_checkpoint_sha256'],ordinary_rtc_max_error=error,camera_validation=geom['new_camera'],train_only_geometry=True,look_at=plan['look_at'],hardware_tested=False,robotpc_client_checked=False)
 (r/'reports/package-acceptance.json').write_text(json.dumps(acceptance,indent=2))
 files=[p for p in sorted(r.rglob('*'))if p.is_file()and p.name!='SHA256SUMS'];assert not any(p.is_symlink()for p in r.rglob('*'))
 (r/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(r))+'\n'for p in files))
 p=subprocess.run(['sha256sum','-c','SHA256SUMS','--quiet'],cwd=r,capture_output=True,text=True);assert p.returncode==0,(p.stdout,p.stderr)
 for p in (r/'scripts').glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
 acceptance.update(files=len(files),bytes=sum(p.stat().st_size for p in files),checksum_manifest_sha256=sha(r/'SHA256SUMS'))
 results.append(acceptance);print(json.dumps(acceptance),flush=True)
(C/'complete.json').write_text(json.dumps(results,indent=2))
