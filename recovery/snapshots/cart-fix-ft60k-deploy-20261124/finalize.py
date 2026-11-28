from pathlib import Path
import json,hashlib,subprocess
C=Path('/coc/flash7/cxin34/cart-fix-ft60k-deploy-control-20261124')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
results=[]
for plan in json.load(open(C/'packages.json')):
 task=plan['task'];r=Path(plan['package']);receipt=json.load(open(C/f'{task}-test-submit.json'));job=receipt['stdout'].strip().split(';')[0]
 acct=subprocess.run(['/opt/slurm/Ubuntu-24.04/current/bin/sacct','-X','-j',job,'--format=JobID,State,ExitCode','-P','-n'],capture_output=True,text=True,check=True,timeout=120).stdout
 assert job+'|COMPLETED|0:0'in acct,acct
 test=json.load(open(r/'reports/skynet-server-test.json'));geom=json.load(open(r/'reports/geometry-audit.json'));native=json.load(open(r/'reports/native-assets-audit.json'))
 assert test['passed']and geom['passed']and native['passed']
 assert test['original_checkpoint_sha256']==plan['source_checkpoint_sha256']==sha(Path(plan['source_checkpoint']))
 assert test['portable_checkpoint_sha256']==sha(r/'checkpoint.pt')
 assert sha(r/'normalizer.json')==plan['normalizer_sha256']and sha(r/'configs/training.json')==plan['training_config_sha256']
 for n,h in plan['gaze_sha256'].items():assert sha(r/'gaze'/n)==h
 required={str(p.relative_to(r)):sha(p)for p in sorted(r.rglob('*'))if p.is_file()and not str(p.relative_to(r)).startswith('reports/')and p.name not in ['SHA256SUMS','README.md']}
 contract=dict(task=task,arm='B',step=60000,seed=42,port=plan['port'],preset=plan['preset'],run=plan['run'],look_at=plan['look_at'],source_checkpoint_sha256=plan['source_checkpoint_sha256'],required_sha256=required,training_split=dict(train=plan['train_recordings'],valid=plan['valid_recordings']),gaze_retrained=False,gaze_source=plan['gaze_source'],geometry_refit=True,hardware_tested=False)
 (r/'reports/deployment-contract.json').write_text(json.dumps(contract,indent=2))
 p=subprocess.run(['/coc/flash7/cxin34/mobilemani-pretraining/env-train/bin/python',str(r/'scripts/verify_assets.py')],capture_output=True,text=True);assert p.returncode==0,(p.stdout,p.stderr)
 (r/'reports/offline-contract-check.json').write_text(p.stdout)
 error=max(max(c['plain_max'],c['rtc_max'])for c in test['cases']);assert error==0
 text=f'''# Original non-look-at Cart fix FT60k deployment
Run: {plan['run']}; exact step60,000; EgoVerse180k initialization, strong MSC, seed42.
Dataset: own EgoPoser corrected Cart30train/10valid. This is NOT sharedfix1124 look-at.
Preset TGZ10HD2, virtual camera ON; port8282; 64x49 actions;16 flow steps; standard RTC8/10/13.
Includes matching original code/config, frozen source normalizer, six Cart gaze models and task reset/camera.
The camera/reset/gaze assets and native asset checks are reused bit-for-bit from the accepted same-run10k package; asset-reuse-audit.json binds hashes. No camera refit or gaze retraining was necessary.
CPU job{job} COMPLETED0 verifies exact60k model tensors,3realheldout plain/RTC WebSocket maxerror{error},finite unseeded requests. Fresh CUDA/installed-client/hardware verification is still required on robotpc.
Original checkpoint: {plan['source_checkpoint']}
Original SHA256: {plan['source_checkpoint_sha256']}
Portable SHA256: {test['portable_checkpoint_sha256']}

Download into its own directory; do not mix look-at or older checkpoint assets.
sha256sum -c SHA256SUMS
bash scripts/client.sh inspect-client
Terminal1: bash scripts/serve.sh
Terminal2: bash scripts/check.sh && bash scripts/client.sh dry-run
Operator-controlled physical launch after checks: bash scripts/client.sh run
Standard native launcher, not the separate background-observation launcher. No camera or robot was operated by packaging. Only one client should control the robot.
'''
 (r/'README.md').write_text(text)
 acceptance=dict(passed=True,package=str(r),task=task,step=60000,run=plan['run'],port=plan['port'],job_id=job,source_checkpoint_sha256=plan['source_checkpoint_sha256'],portable_checkpoint_sha256=test['portable_checkpoint_sha256'],ordinary_rtc_max_error=error,camera_validation=geom['new_camera'],train_only_geometry=True,look_at=plan['look_at'],hardware_tested=False,robotpc_client_checked=False)
 (r/'reports/package-acceptance.json').write_text(json.dumps(acceptance,indent=2))
 files=[p for p in sorted(r.rglob('*'))if p.is_file()and p.name!='SHA256SUMS'];assert not any(p.is_symlink()for p in r.rglob('*'))
 (r/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(r))+'\n'for p in files))
 p=subprocess.run(['sha256sum','-c','SHA256SUMS','--quiet'],cwd=r,capture_output=True,text=True);assert p.returncode==0,(p.stdout,p.stderr)
 for p in (r/'scripts').glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
 acceptance.update(files=len(files),bytes=sum(p.stat().st_size for p in files),checksum_manifest_sha256=sha(r/'SHA256SUMS'))
 results.append(acceptance);print(json.dumps(acceptance),flush=True)
(C/'complete.json').write_text(json.dumps(results,indent=2))
